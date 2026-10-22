#!/usr/bin/env python3
"""B9A11V: FIXED-q FEM -> B5 inverse observability and noise sweep.

Dependencies from the EXISTING Mac project (kept unchanged):
  65AM_B9A11T_M_exact_FEM_vs_B5_REVISED.py
  65AM_B9A11U_C_mac_train_validate_FDJAC_FIXED.py
  original centered_grad B5 checkpoints, candidate B9A11U checkpoints,
  B9A11T_exports, B9_coupled_anchor, B9A11R locked sensors.

No FEM solves, retraining, true-parameter initialization, checkpoint writes,
physical strain observations, or full seven-region re-equilibration.

Stages: selftest | check | run.
All values are synthetic conditional inverse results using old B9A11T cases.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROPS = ("E1", "E2", "G12")
REGIONS = ("inner_left", "inner_right")
OUT = "65AM_B9A11V_noise_observability"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def finite_jac(fun, z, h=0.01):
    """Fixed-step central/one-sided difference in normalized [-1,1] z.

    Never use SciPy's default microscopic 2-point FD with a float32 network.
    """
    z = np.asarray(z, float)
    y0 = np.asarray(fun(z), float).ravel()
    J = np.empty((len(y0), len(z)), float)
    for j in range(len(z)):
        hi = z.copy(); lo = z.copy()
        hi[j] = min(1.0, z[j] + h)
        lo[j] = max(-1.0, z[j] - h)
        span = hi[j] - lo[j]
        if span < h * 0.2:
            raise ValueError("Finite difference collapsed")
        J[:, j] = (np.asarray(fun(hi), float).ravel() -
                   np.asarray(fun(lo), float).ravel()) / span
    if not np.isfinite(J).all():
        raise ValueError("Invalid Jacobian")
    return J


def svd_metrics(J, prior_sd=0.75):
    """J derivative with respect to FIT-normalized z, whitened by assumed sigma."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    norms = np.linalg.norm(J, axis=0)
    corr = (J.T @ J) / np.maximum(np.outer(norms, norms), 1e-30)
    fish = J.T @ J
    # Conditional local Gaussian/Laplace approximation. Not calibrated Bayesian UQ.
    laplace = np.linalg.inv(fish + np.eye(3) / prior_sd**2)
    no_prior_cov = np.linalg.pinv(fish, rcond=1e-12)
    return dict(
        singular_values=s,
        weakest_normalized_direction=Vt[-1, :],
        condition=s[0] / max(s[-1], 1e-14),
        column_similarity=corr,
        posterior_std_z=np.sqrt(np.maximum(np.diag(laplace), 0)),
        no_prior_std_z=np.sqrt(np.maximum(np.diag(no_prior_cov), 0)),
    )


def solve_one(predict, obs, sigma_likelihood, method, prior_sd, fd_step):
    from scipy.optimize import least_squares
    obs = np.asarray(obs, float).ravel()
    prior_sd = float(prior_sd)
    if method not in ("unregularized", "gaussian_MAP"):
        raise ValueError(method)
    if sigma_likelihood <= 0 or prior_sd <= 0:
        raise ValueError("Invalid likelihood/prior scale")

    def resid(z):
        d = (np.asarray(predict(z), float).ravel() - obs) / sigma_likelihood
        if method == "gaussian_MAP":
            d = np.r_[d, np.asarray(z, float) / prior_sd]
        return d

    answer = least_squares(
        resid, np.zeros(3, float),
        jac=lambda z: finite_jac(resid, z, fd_step),
        bounds=(-np.ones(3), np.ones(3)),
        method="trf", max_nfev=115,
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    if not np.isfinite(answer.x).all():
        raise RuntimeError("Nonfinite inverse solution")
    residual_U = (np.asarray(predict(answer.x), float).ravel() - obs)
    return answer, float(np.sqrt(np.mean(residual_U**2)))


def selftest():
    from scipy.optimize import least_squares  # dependency test
    rng = np.random.default_rng(707)
    A = rng.normal(size=(60, 3)) * 0.0007
    truth = np.array([0.35, -0.20, 0.50])
    # Simulates a float32 forward model to catch previous vanishing-FD error.
    def f(z):
        return (A @ np.asarray(z, np.float32)).astype(np.float32)
    J = finite_jac(f, np.zeros(3))
    assert np.allclose(J, A, atol=3e-8)
    m = svd_metrics(J / 0.0002)
    assert m['condition'] < 100
    assert m['column_similarity'].shape == (3, 3)
    assert np.linalg.norm(m['weakest_normalized_direction']) > 0.999
    r, rmse = solve_one(f, f(truth), 0.00002, 'unregularized', 0.75, 0.01)
    assert np.linalg.norm(r.x - truth) < 0.0001, (r.x, truth)
    assert rmse < 1e-8
    y = f(truth) + rng.normal(0, 0.0002, size=60)
    a, _ = solve_one(f, y, 0.0002, 'gaussian_MAP', 0.75, 0.01)
    assert np.isfinite(a.x).all()
    print('B9A11V SELFTEST PASS: float32 FD, inverse, regularized solve, SVD')


class Operator:
    """Fast matched B5 forward evaluations at locked sensors and fixed FEM q."""
    def __init__(self, d, model, branch, lift, sensor_ids):
        import torch
        self.model = model.eval()
        self.sensor_ids = np.asarray(sensor_ids, int)
        self.basis = np.asarray(d['basis'][:, self.sensor_ids, :], np.float32)
        self.lift = np.asarray(lift[0, self.sensor_ids, :], np.float32)
        self.branch = np.asarray(branch[0], float).copy()
        self.mean = np.asarray(d['branch_mean'], float)
        self.std = np.asarray(d['branch_std'], float)
        fit = np.asarray(d['ds']['branch_inputs'][d['fit'], :3], float)
        self.mid = (fit.min(axis=0) + fit.max(axis=0)) / 2
        self.half = (fit.max(axis=0) - fit.min(axis=0)) / 2
        if np.any(self.half <= 0):
            raise RuntimeError('FIT parameter bounds invalid')

    def __call__(self, z):
        import torch
        z = np.asarray(z, float)
        if len(z) != 3 or not np.isfinite(z).all():
            raise ValueError('Three finite material parameters expected')
        raw = self.branch.copy()
        raw[:3] = self.mid + self.half * z
        x = np.asarray((raw-self.mean)/self.std, np.float32).reshape(1,-1)
        with torch.no_grad():
            c = self.model(torch.from_numpy(x)).detach().cpu().numpy()[0]
        field = self.lift + np.einsum('r,rnc->nc', c, self.basis)
        return field.reshape(-1).astype(float)


def load_context(args):
    import torch
    root = Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    torch.set_num_threads(min(2, torch.get_num_threads()))
    t = importlib.import_module('65AM_B9A11T_M_exact_FEM_vs_B5_REVISED')
    u = importlib.import_module('65AM_B9A11U_C_mac_train_validate_FDJAC_FIXED')
    root, b6, b9, anchor, data, md, info = t.context(
        SimpleNamespace(project=str(root), design_tag=args.design_tag))
    candidate_dir = root / 'results' / '65AM_B9A11U_sensitivity_finetune' / args.candidate_tag
    if not (candidate_dir / 'B9A11U_training_summary.json').is_file():
        raise FileNotFoundError('Missing B9A11U training summary: '+str(candidate_dir))
    packs, _, _ = u.context(root)
    refs = {}
    hashlist = {}
    for region in REGIONS:
        pk = info[region]
        d = pk['d']
        ds = d['ds']
        ds['_b9_region'] = region
        ictx = b6.b.CO.interface_context(root, ds, torch.device('cpu'))
        branch, lift = b9.build_branch_lift(root, ds, anchor, ictx, torch.device('cpu'))
        trained = copy.deepcopy(pk['model'])
        ckpath = candidate_dir / region / 'B9A11U_best_candidate.pt'
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        if ck['source_checkpoint_sha256'] != packs[region]['source_ck_sha']:
            raise RuntimeError('Checkpoint ancestry differs: '+region)
        trained.load_state_dict(ck['state_dict'], strict=True)
        trained.eval()
        frozen = pk['model'].eval()
        idx = np.asarray(pk['layouts'][args.layout], int)
        ref = np.asarray(data['U_'+region][:, pk['indices'], :], float)[:, idx, :]
        models = {
            'original_B5': Operator(d, frozen, branch, lift, idx),
            'trained_B5': Operator(d, trained, branch, lift, idx)
        }
        # Validate fast operator against the original B6 forward implementation.
        for name, model in (('original_B5', frozen),('trained_B5', trained)):
            op = models[name]
            for state in (0, 2, 4, 6):
                mat = np.asarray(data['material_MPa'][state], float)
                z = (mat - op.mid) / op.half
                b = branch.copy()
                b[0,:3] = mat
                U = b6.predict_b5(d, model, b, lift, torch.device('cpu'))[0][idx,:].reshape(-1)
                discrepancy = np.max(abs(op(z)-U))
                if discrepancy > 2e-6:
                    raise RuntimeError(f'Fast B5 predictor mismatch {region} {name}: {discrepancy:g} mm')
        refs[region] = dict(reference=ref, models=models)
        hashlist[region] = dict(original_checkpoint=packs[region]['source_ck_sha'],
                                trained_checkpoint=digest(ckpath))
        print('CHECK PASS', region, 'layout', args.layout,
              'FEM cases', ref.shape[0], 'sensors', len(idx), flush=True)
    return root, t, data, refs, hashlist


def check(args):
    load_context(args)
    print('B9A11V CHECK PASS: fixed FEM q, paired models, locked sensors; no output written')


def run(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, t, data, refs, hashes = load_context(args)
    out = root / 'results' / OUT / args.tag
    if out.exists():
        raise RuntimeError('Results protected; choose a new --tag: '+str(out))
    if args.states.strip().lower() == 'all':
        states = list(range(7))
    else:
        states = [int(v) for v in args.states.split(',')]
        if not states or min(states)<0 or max(states)>6 or len(states)!=len(set(states)):
            raise ValueError('Invalid --states: use all or unique comma-separated indices 0..6')
    levels = [float(v) for v in args.noise_levels.split(',')]
    if len(levels)!=len(set(levels)) or not levels or any(v<0 for v in levels):
        raise ValueError('Noise levels must be distinct nonnegative mm values')

    rows=[]; diag=[]; correlations=[]
    for ri, region in enumerate(REGIONS):
        item=refs[region]
        # FEM fixed-q derivatives on exactly the same 20 displacement sensors.
        # Convert each +/-3% physical-property response into a derivative in the
        # FIT-normalized z variables used by the B5 inverse optimizer.
        template = item['models']['original_B5']
        anchor_material = np.asarray(data['material_MPa'][0], float)
        Jf = np.empty((item['reference'].shape[1]*3,3),float)
        for j in range(3):
            plus = item['reference'][2+2*j].reshape(-1)
            minus = item['reference'][1+2*j].reshape(-1)
            Jf[:,j] = (plus-minus)/(2*0.03*anchor_material[j])*template.half[j]/args.reference_sigma
        matrices={'Exact_FEM':Jf}
        for model_name, op in item['models'].items():
            z_anchor=(anchor_material-op.mid)/op.half
            matrices[model_name]=finite_jac(op,z_anchor,args.fd_step)/args.reference_sigma
        for model_name, Jwhite in matrices.items():
            # Local linearized conditional observability. This is NOT an exact
            # posterior, does not account for model discrepancy or uncertain q.
            m = svd_metrics(Jwhite,args.prior_sd_z)
            diag.append(dict(region=region,model=model_name,layout=args.layout,
                  min_singular_value=float(m['singular_values'][-1]),
                  max_singular_value=float(m['singular_values'][0]),
                  condition_number=float(m['condition']),
                  prior_sd_z=args.prior_sd_z,
                  reference_sigma_mm=args.reference_sigma,
                  anchor_J_column_E1=float(np.linalg.norm(Jwhite[:,0])),
                  anchor_J_column_E2=float(np.linalg.norm(Jwhite[:,1])),
                  anchor_J_column_G12=float(np.linalg.norm(Jwhite[:,2])),
                  **{f'weakest_direction_{p}':float(m['weakest_normalized_direction'][j]) for j,p in enumerate(PROPS)},
                  **{f'linearized_sd_{p}_pct_FIT_halfspan':float(m['posterior_std_z'][j]*op.half[j]/op.mid[j]*100)
                     for j,p in enumerate(PROPS)}))
            for i,a in enumerate(PROPS):
                for j,b in enumerate(PROPS):
                    correlations.append(dict(region=region,model=model_name,
                                             parameter_a=a,parameter_b=b,
                                             sensitivity_cosine=float(m['column_similarity'][i,j])))

        for state in states:
            ptrue=np.asarray(data['material_MPa'][state],float)
            yclean=item['reference'][state].reshape(-1)
            # Every model and inversion method sees identical FEM observations.
            noises=[]
            for rep in range(args.reps):
                rng=np.random.default_rng(args.seed+ri*100000+state*1000+rep)
                noises.append(rng.normal(size=len(yclean)))
            for level in levels:
                for rep in range(1 if level==0 else args.reps):
                    y=yclean if level==0 else yclean+level*noises[rep]
                    sigma_like=max(level,args.sigma_floor)
                    for model_name, op in item['models'].items():
                        for method in ('unregularized','gaussian_MAP'):
                            res, misfit=solve_one(op,y,sigma_like,method,args.prior_sd_z,args.fd_step)
                            est=op.mid+op.half*res.x
                            row=dict(region=region,case=t.CASES[state],case_index=state,
                              model=model_name,method=method,
                              sensor_layout=args.layout,noise_sigma_mm=level,
                              likelihood_sigma_mm=sigma_like,rep=rep,converged=bool(res.success),
                              boundary_hit=bool(np.any(abs(res.x)>0.99)),
                              residual_RMSE_mm=misfit,nfev=int(res.nfev),
                              FIT_midpoint_prior=True,
                              **{f'{p}_true_MPa':float(ptrue[j]) for j,p in enumerate(PROPS)},
                              **{f'{p}_estimated_MPa':float(est[j]) for j,p in enumerate(PROPS)},
                              **{f'{p}_abs_rel_pct':float(100*abs(est[j]-ptrue[j])/ptrue[j]) for j,p in enumerate(PROPS)})
                            rows.append(row)
            print('EVALUATED',region,t.CASES[state],flush=True)

    df=pd.DataFrame(rows)
    di=pd.DataFrame(diag)
    cor=pd.DataFrame(correlations)
    out.mkdir(parents=True,exist_ok=False)
    df.to_csv(out/'B9A11V_noise_sweep_cases.csv',index=False)
    di.to_csv(out/'B9A11V_observability.csv',index=False)
    cor.to_csv(out/'B9A11V_parameter_correlation.csv',index=False)

    parts=[]
    for (noise,model,method),g in df.groupby(['noise_sigma_mm','model','method'],sort=True):
        q=dict(noise_sigma_mm=float(noise),model=model,method=method,
               n_fits=int(len(g)),boundary_rate=float(g.boundary_hit.mean()),
               convergence_rate=float(g.converged.mean()),
               median_forward_residual_mm=float(g.residual_RMSE_mm.median()))
        for p in PROPS:
            q[p+'_median_abs_rel_pct']=float(g[p+'_abs_rel_pct'].median())
            q[p+'_90pct_abs_rel_pct']=float(g[p+'_abs_rel_pct'].quantile(0.90))
        parts.append(q)
    sm=pd.DataFrame(parts)
    sm.to_csv(out/'B9A11V_noise_summary.csv',index=False)

    for metric,basename,ylabel in (
        ('_median_abs_rel_pct','B9A11V_noise_recovery.png','Median absolute relative error (%)'),
        ('_90pct_abs_rel_pct','B9A11V_noise_tail_error.png','90th-percentile absolute relative error (%)')
    ):
        fig, axes = plt.subplots(1,3,figsize=(13.5,4.0),sharex=True)
        for p,ax in zip(PROPS,axes):
            for (model,method),group in sm.groupby(['model','method']):
                series=group.sort_values('noise_sigma_mm')
                ax.plot(series.noise_sigma_mm*1e3,series[p+metric],'.-',
                        label=f'{model} | {method}')
            ax.set_title(p);ax.set_xlabel('Noise SD (micrometers)')
            ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle('B9A11V: same known-q FEM truth and paired noise; internal B9A11T')
        fig.tight_layout()
        fig.savefig(out/basename,dpi=175)
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5))
    for (model,method),g in sm.groupby(['model','method']):
        g=g.sort_values('noise_sigma_mm')
        ax.plot(g.noise_sigma_mm*1e3,100*g.boundary_rate,'o-',label=f'{model} | {method}')
    ax.set_xlabel('Noise SD (micrometers)')
    ax.set_ylabel('Fits reaching material bounds (%)')
    ax.set_title('B9A11V: material-boundary hits under increasing observation noise')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'B9A11V_boundary_rate.png',dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,r in zip(axes,REGIONS):
        for model in ('Exact_FEM','original_B5','trained_B5'):
            item=di[(di.region==r)&(di.model==model)].iloc[0]
            vals=[item['anchor_J_column_'+p] for p in PROPS]
            ax.plot(PROPS,vals,'o-',label=model)
        ax.set_ylabel('Noise-whitened Jacobian column norm')
        ax.set_title(r);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('B9A11V: strength of material responses under locked measurements')
    fig.tight_layout();fig.savefig(out/'B9A11V_column_observability.png',dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,r in zip(axes,REGIONS):
        for model in ('Exact_FEM','original_B5','trained_B5'):
            row=di[(di.region==r)&(di.model==model)].iloc[0]
            vals=[row['weakest_direction_'+p] for p in PROPS]
            ax.plot(PROPS,abs(np.asarray(vals)),'o-',label=model)
        ax.set_ylabel('Absolute component of weakest right singular vector')
        ax.set_title(r);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Weakest material combination (orientation/sign omitted)')
    fig.tight_layout();fig.savefig(out/'B9A11V_weak_parameter_direction.png',dpi=175);plt.close(fig)

    # Exact FEM vs both B5s: strength of all 3 independent material combinations.
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3))
    for ax,region in zip(axes,REGIONS):
        for model in ('Exact_FEM','original_B5','trained_B5'):
            op=refs[region]['models']['original_B5']
            reference=np.asarray(data['material_MPa'][0],float)
            if model=='Exact_FEM':
                finite=refs[region]['reference']
                J=np.stack([(finite[2+2*j]-finite[1+2*j]).ravel()*op.half[j]/(2*0.03*reference[j]*args.reference_sigma)
                            for j in range(3)],axis=1)
            else:
                operator=refs[region]['models'][model]
                J=finite_jac(operator,(reference-operator.mid)/operator.half,args.fd_step)/args.reference_sigma
            s=np.linalg.svd(J,compute_uv=False)
            ax.semilogy([1,2,3],s,'o-',label=model)
        ax.set_title(region);ax.set_xlabel('Singular-value index');ax.set_xticks([1,2,3]);ax.grid(alpha=.2)
        ax.set_ylabel('Whitened sensitivity singular value');ax.legend(fontsize=8)
    fig.suptitle('B9A11V: exact FEM vs learned material observability | fixed q')
    fig.tight_layout();fig.savefig(out/'B9A11V_FEM_vs_B5_singular_spectra.png',dpi=175);plt.close(fig)

    summary=dict(stage='B9A11V',scope='known-FEM-interface conditional inverse only',
       validated_B9A11T_fixed_q_data=True,no_new_FEM_or_retraining=True,
       not_blinded_external_validation=True,not_operational_digital_twin=True,
       noise_levels_mm=levels,noise_repetitions=args.reps,
       material_states=[t.CASES[i] for i in states],models=['original_B5','trained_B5'],
       methods=['unregularized','gaussian_MAP'],
       prior=dict(type='independent truncated Gaussian on FIT-normalized z within [-1,1]',
                  center='FIT parameter midpoint, NOT FEM truth',sd_z=args.prior_sd_z),
       posterior_status='No posterior sampling; local Gaussian/Laplace noise-only estimate, ignores model discrepancy',
       sigma_likelihood_floor_mm=args.sigma_floor,
       reference_sigma_observability_mm=args.reference_sigma,
       warning='No strain-augmented inversion: B9A11T has U-only, true surface strain signals not exported',
       checkpoints=hashes,
       sample_count=int(len(df)),
       noise_summary=parts)
    (out/'B9A11V_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11V COMPLETED',out)
    print(sm[['noise_sigma_mm','model','method']+[p+'_median_abs_rel_pct' for p in PROPS]+['boundary_rate']].to_string(index=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',required=True,choices=('selftest','check','run'))
    ap.add_argument('--project',default='.')
    ap.add_argument('--tag',default='pilot01')
    ap.add_argument('--candidate-tag',default='pilot01')
    ap.add_argument('--design-tag',default='pilot01')
    ap.add_argument('--layout',choices=('baseline','optimized'),default='optimized')
    ap.add_argument('--states',default='0,1,3,5',help='0..6 comma separated, or all')
    ap.add_argument('--reps',type=int,default=2)
    ap.add_argument('--noise-levels',default='0,0.00001,0.000025,0.00005,0.0001,0.0002')
    ap.add_argument('--fd-step',type=float,default=0.01)
    ap.add_argument('--sigma-floor',type=float,default=0.00001,
                    help='Likelihood scale when actual simulated noise is zero')
    ap.add_argument('--reference-sigma',type=float,default=0.0002)
    ap.add_argument('--prior-sd-z',type=float,default=0.75,
                    help='Assumed Gaussian prior SD in normalized FIT-parameter coordinates')
    ap.add_argument('--seed',type=int,default=651115)
    a=ap.parse_args()
    if a.stage=='selftest':return selftest()
    if not (1<=a.reps<=50 and 0.001<=a.fd_step<=0.05 and
            0<a.sigma_floor<=1e-3 and 0<a.reference_sigma<=1e-2 and
            0.1<=a.prior_sd_z<=5.0):
        raise ValueError('Unreasonable experiment parameters')
    if a.stage=='check':return check(a)
    run(a)


if __name__=='__main__':main()