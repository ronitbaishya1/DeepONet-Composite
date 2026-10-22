#!/usr/bin/env python3
"""B9A11S: fit-only B5 discrepancy calibration and INTERNAL tune inverse audit.

No new Abaqus, no checkpoint modification, NO live FE--NO coupling.
The correction is an auxiliary FEM-trained residual model, NOT pure B5.
All interface PCA q come from FEM; no real-sensor or independent validation.

Depends on existing local files: 65AM_B9A11Q_A_inverse_diagnostics.py,
65AM_B9A11P_A_readiness.py, 65AM_B5_common.py, and saved 64J dataset.
Requires R locked design in results/65AM_B9A11R_sensor_inverse/<design-tag>/.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys
from pathlib import Path
import numpy as np

PROPS=('E1','E2','G12')
REGIONS=('inner_left','inner_right')
BASE='65AM_B9A11S_forward_inverse_consistency'
ALPHAS=(0.001,0.01,0.1,1.,10.,100.,1000.)


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def mesh_hash(xyz):
    return hashlib.sha256(np.ascontiguousarray(xyz,dtype=np.float64).tobytes()).hexdigest()


def ridge_fit_cv(X,Y,alphas=ALPHAS):
    """Select ridge lambda and correction gain with analytical FIT-only LOO.

    X includes a constant column; the intercept is not penalized.
    Y is FIT residual, FEM - frozen B5, [nfit, nfree*3].
    """
    X=np.asarray(X,dtype=np.float64); Y=np.asarray(Y,dtype=np.float64)
    if X.ndim!=2 or Y.ndim!=2 or X.shape[0]!=Y.shape[0] or X.shape[0]<X.shape[1]+2:
        raise ValueError('Invalid FIT design or insufficient training cases')
    if not np.allclose(X[:,0],1.) or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Invalid fitted inputs/targets')
    G=X.T@X;P=np.diag([0.]+[1.]*(X.shape[1]-1)); XtY=X.T@Y
    options=[]
    # No correction is always permitted and gives FIT-only baseline.
    base=float(np.mean(Y**2))
    options.append((base, 0., 1.e3, np.zeros((X.shape[1],Y.shape[1])), base))
    for lam in alphas:
        M=np.linalg.solve(G+float(lam)*P, np.eye(G.shape[0]))
        B=M@XtY
        Hdiag=np.einsum('ij,jk,ik->i',X,M,X)
        if np.any(Hdiag>0.999999) or not np.isfinite(Hdiag).all():
            continue
        E=Y-X@B
        Eloo=E/(1-Hdiag)[:,None]
        for gain in (0.25,0.5,0.75,1.0):
            cv=float(np.mean((Y-gain*(Y-Eloo))**2))
            options.append((cv,gain,float(lam),B,base))
    cv,gain,lam,B,base=min(options,key=lambda t:t[0])
    return B*gain, dict(cv_mse=cv,baseline_fit_mse=base,gain=float(gain),
                        ridge_lambda=float(lam),fit_corrected_rmse=float(np.sqrt(np.mean((Y-X@(B*gain))**2))),
                        fit_baseline_rmse=float(np.sqrt(base)))


def basis_features(branch,config):
    branch=np.asarray(branch,dtype=float)
    return np.r_[1.,(branch-np.asarray(config['d']['branch_mean'],float))/np.asarray(config['d']['branch_std'],float)]


def active_sensor_layouts(root,configs,design_tag,sensors):
    p=root/'results'/'65AM_B9A11R_sensor_inverse'/design_tag/'B9A11R_locked_design.json'
    if not p.is_file():raise FileNotFoundError('Missing locked R design: '+str(p))
    j=json.loads(p.read_text())
    if j.get('sensors')!=sensors:raise ValueError('R design sensor count mismatch')
    result={}
    for reg,c in configs.items():
        record=j['regions'][reg]
        xyz=np.asarray(c['ds']['coordinates'],float)
        if mesh_hash(xyz)!=record['mesh_sha256']:
            raise RuntimeError('Locked R sensor mesh hash mismatch: '+reg)
        free=np.setdiff1d(np.arange(len(xyz)),np.asarray(c['d']['fixed'],int))
        layouts={}
        for name in ('baseline','optimized'):
            ids=np.asarray(record[name],dtype=int)
            if len(ids)!=sensors or len(np.unique(ids))!=sensors or not np.isin(ids,free).all():
                raise RuntimeError('Invalid R sensor indices: '+reg+' '+name)
            layouts[name]=ids
        result[reg]=layouts
    return result,p


def load_project(args):
    import torch
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    torch.set_num_threads(min(2,torch.get_num_threads()))
    P=importlib.import_module('65AM_B9A11P_A_readiness')
    P.audit(root,args.variant,args.loss,None)
    Q=importlib.import_module('65AM_B9A11Q_A_inverse_diagnostics')
    configs={r:Q.load_region(root,r,args.variant,args.loss,args.sensors) for r in REGIONS}
    layouts,lock=active_sensor_layouts(root,configs,args.design_tag,args.sensors)
    return root,Q,configs,layouts,lock


def with_nodes(c,ids):
    import torch
    t=dict(c);t['ids']=np.asarray(ids,dtype=int)
    t['B']=torch.as_tensor(np.asarray(c['d']['basis'][:,ids,:],np.float32).reshape(c['d']['rank'],-1))
    return t


def fit_region(c):
    import torch
    d=c['d'];ds=c['ds'];ids=np.asarray(d['fit'],dtype=int)
    n=len(ds['coordinates']);free=np.setdiff1d(np.arange(n),np.asarray(d['fixed'],int))
    # Actual frozen B5 model + original FEM only on original 44 FIT cases.
    U,_=c['C'].predict(c['model'],d,ids,torch.device('cpu'))
    target=(np.asarray(ds['U'][ids],float)-U)[:,free,:].reshape(len(ids),-1)
    X=np.stack([basis_features(c['phys'][k],c) for k in ids])
    beta,meta=ridge_fit_cv(X,target)
    return dict(beta=beta,free=free,metadata=meta,fit_ids=ids,X=X,Y=target,
                U_B5_fit=U)


def save_fit(root,args,configs,layouts,lock):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'results'/BASE/args.tag
    if dest.exists():raise FileExistsError('Output exists; use a different --tag: '+str(dest))
    dest.mkdir(parents=True)
    report={'stage':'B9A11S-fit','variant':args.variant,'loss':args.loss,
            'design_tag':args.design_tag,'locked_design_sha256':sha256(lock),
            'source':'44 original 64J FIT cases only per region',
            'fit_only_cv':'analytic ridge leave-one-out; gain and lambda selected FIT-only',
            'not_full_coupling':True,'not_new_B5_training':True,'regions':{}}
    rows=[];spatial=[]
    for reg,c in configs.items():
        obj=fit_region(c)
        beta=obj['beta']; meta=obj['metadata']
        np.savez_compressed(dest/('S_fit_'+reg+'.npz'),beta=beta,free=obj['free'],
                            fit_ids=obj['fit_ids'],mesh_sha256=mesh_hash(c['ds']['coordinates']),
                            checkpoint_sha256=sha256(c['C'].ck_path(root,args.variant,args.loss,reg)))
        rep=dict(**meta,fit_cases=len(obj['fit_ids']),
                 correction_is_FEM_supervised_not_original_B5=True)
        report['regions'][reg]=rep
        free=obj['free'];xyz=np.asarray(c['ds']['coordinates'],float)
        E=obj['Y'].reshape(len(obj['fit_ids']),len(free),3)
        prediction=(obj['X']@beta).reshape(E.shape)
        for layout,inds in layouts[reg].items():
            pos={int(k):i for i,k in enumerate(free)}
            idx=np.array([pos[int(k)] for k in inds],int)
            bef=float(np.sqrt(np.mean(E[:,idx,:]**2)))
            aft=float(np.sqrt(np.mean((E-prediction)[:,idx,:]**2)))
            rows.append(dict(region=reg,layout=layout,fit_B5_rmse_mm=bef,
                             fit_corrected_rmse_mm=aft,fit_only=True))
        x=xyz[free,0]
        bins=np.linspace(float(x.min()),float(x.max()),17)
        for i in range(16):
            sel=(x>=bins[i])&(x<=bins[i+1] if i==15 else x<bins[i+1])
            if not sel.any():continue
            spatial.append(dict(region=reg,x_mm=.5*(bins[i]+bins[i+1]),
                original_RMS_mm=float(np.sqrt(np.mean(E[:,sel,:]**2))),
                corrected_RMS_mm=float(np.sqrt(np.mean((E-prediction)[:,sel,:]**2))),
                fit_only=True))
        print('FIT',reg,'LOOCV gain',round(meta['gain'],3),'ridge',meta['ridge_lambda'],
              'FIT whole-field rmse',meta['fit_baseline_rmse'],'->',meta['fit_corrected_rmse'],flush=True)
    pd.DataFrame(rows).to_csv(dest/'B9A11S_fit_sensor_rmse.csv',index=False)
    spatialdf=pd.DataFrame(spatial);spatialdf.to_csv(dest/'B9A11S_fit_spatial_residuals.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.2))
    for reg,g in spatialdf.groupby('region'):
        g=g.sort_values('x_mm')
        ax.plot(g.x_mm,g.original_RMS_mm,'-o',ms=3,label=reg+' B5')
        ax.plot(g.x_mm,g.corrected_RMS_mm,'--',label=reg+' + fit-only calibration')
    ax.set_xlabel('Subdomain x (mm)');ax.set_ylabel('FIT displacement RMSE (mm)')
    ax.set_title('B9A11S training residual localization (NOT held-out)')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(dest/'B9A11S_fit_residual_profile.png',dpi=180);plt.close(fig)
    (dest/'B9A11S_fit_summary.json').write_text(json.dumps(report,indent=2))
    print('FIT COMPLETE:',dest)


class CorrectedOp:
    """Frozen B5 prediction + fit-only linear discrepancy on free nodes."""
    def __init__(self,base,config,coef,free):
        self.base=base;self.x=config;self.true=base.true;self.base_FEM=base.base_FEM
        lookup={int(k):i for i,k in enumerate(free)}
        if any(int(i) not in lookup for i in config['ids']):raise RuntimeError('Sensor not in correction free nodes')
        si=np.array([lookup[int(i)] for i in config['ids']],dtype=int)
        cols=(3*si[:,None]+np.arange(3)[None,:]).reshape(-1)
        self.beta=np.asarray(coef,dtype=np.float64)[:,cols]
        self.mean=np.asarray(config['d']['branch_mean'],float)
        self.std=np.asarray(config['d']['branch_std'],float)
    def normalized_true(self):return self.base.normalized_true()
    def pred_and_jac(self,z,jac=True):
        z=np.asarray(z,float)
        phys=self.x['mid']+self.x['half']*z
        branch=np.r_[phys,self.base.q.detach().numpy().astype(float)]
        features=np.r_[1.,(branch-self.mean)/self.std]
        corr=features@self.beta
        if jac:
            val,J=self.base.pred_and_jac(z,True)
            Jcorr=self.beta[1:4,:].T*(self.x['half']/self.std[:3])[None,:]
            return val+corr,J+Jcorr
        return self.base.pred_and_jac(z,False)+corr


def evaluate(root,Q,args,configs,layouts,lock):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'results'/BASE/args.tag
    summaryfile=dest/'B9A11S_fit_summary.json'
    if not summaryfile.is_file():raise FileNotFoundError('Run --stage fit first: '+str(summaryfile))
    summary=json.loads(summaryfile.read_text())
    if summary['locked_design_sha256']!=sha256(lock) or summary['variant']!=args.variant or summary['loss']!=args.loss:
        raise RuntimeError('Locked fit provenance mismatch')
    for name in ['B9A11S_eval_cases.csv','B9A11S_eval_summary.json']:
        if (dest/name).exists():raise FileExistsError('Evaluation already present, preserve results: '+name)
    rows=[];forward_rows=[];jac_rows=[]
    for ri,(reg,c) in enumerate(configs.items()):
        pack=dest/('S_fit_'+reg+'.npz')
        if not pack.is_file():raise FileNotFoundError(str(pack))
        with np.load(pack,allow_pickle=False) as a:
            beta=a['beta'].copy();free=a['free'].copy();fit_ids=a['fit_ids'].copy()
            assert str(a['mesh_sha256'].item())==mesh_hash(c['ds']['coordinates'])
            if str(a['checkpoint_sha256'].item())!=sha256(c['C'].ck_path(root,args.variant,args.loss,reg)):
                raise RuntimeError('B5 checkpoint changed since fitting')
        if not np.array_equal(fit_ids,np.asarray(c['d']['fit'],int)):
            raise RuntimeError('FIT indices changed since training')
        valid=[int(k) for k in c['d']['tune'] if
               np.all(c['phys'][k,:3]>=c['lo']) and np.all(c['phys'][k,:3]<=c['hi'])]
        if not valid:raise RuntimeError('No in-bound TUNE cases for '+reg)
        for cid in valid[:args.cases]:
            # Paired, repeatable FULL mesh noise for fairness across layouts.
            noises=[]
            for rep in range(args.noise_reps):
                rng=np.random.default_rng(args.seed+ri*1000000+cid*1000+rep)
                noises.append(rng.normal(0.,args.sigma_u,size=(len(c['ds']['coordinates']),3)))
            for layout,ids in layouts[reg].items():
                cc=with_nodes(c,ids)
                raw=Q.ConditionalOperator(cc,cid)
                corrected=CorrectedOp(raw,cc,beta,free)
                yF=raw.base_FEM
                for modelname,op in [('frozen_B5',raw),('B5_plus_FIT_correction',corrected)]:
                    ztrue=op.normalized_true()
                    ypred,J=op.pred_and_jac(ztrue,True)
                    rmse=float(np.sqrt(np.mean((ypred-yF)**2)))
                    norm=float(np.linalg.norm(J/args.sigma_u,axis=0).mean())
                    sing=np.linalg.svd(J/args.sigma_u,compute_uv=False)
                    forward_rows.append(dict(region=reg,case_id=cid,layout=layout,
                        model=modelname,forward_RMSE_mm=rmse,
                        forward_RMSE_over_sigma=rmse/args.sigma_u,
                        jacobian_condition=float(sing[0]/max(sing[-1],1e-14)),
                        jacobian_smallest_singular=float(sing[-1]),
                        jacobian_mean_column_norm=norm))
                    for k,prop in enumerate(PROPS):
                        score=np.sqrt(np.mean((J[:,k]*(0.05*op.true[k]/cc['half'][k])/args.sigma_u)**2))
                        jac_rows.append(dict(region=reg,case_id=cid,layout=layout,model=modelname,
                                             property=prop,importance=float(score)))
                    jobs=[('FEM_zero_noise',-1,yF)]
                    for rep,noise in enumerate(noises):
                        jobs.append(('FEM_noisy',rep,yF+noise[ids,:].reshape(-1)))
                    for mode,rep,yobs in jobs:
                        fit,whitened=Q.inversion(op,yobs,args.sigma_u,args.max_nfev)
                        estimate=cc['mid']+cc['half']*fit.x
                        row=dict(region=reg,case_id=cid,layout=layout,model=modelname,
                            mode=mode,rep=rep,fit_success=bool(fit.success),
                            fit_whitened_RMS=whitened,
                            any_bound_hit=bool(np.any(np.abs(fit.x)>.99)),
                            forward_RMSE_mm=rmse)
                        for k,prop in enumerate(PROPS):
                            row[prop+'_truth_MPa']=float(op.true[k]);row[prop+'_estimated_MPa']=float(estimate[k])
                            row[prop+'_absolute_error_pct']=float(100*abs(estimate[k]-op.true[k])/op.true[k])
                        rows.append(row)
            print('EVAL',reg,'case',cid,flush=True)
    df=pd.DataFrame(rows);fr=pd.DataFrame(forward_rows);jj=pd.DataFrame(jac_rows)
    df.to_csv(dest/'B9A11S_eval_cases.csv',index=False)
    fr.to_csv(dest/'B9A11S_eval_forward_errors.csv',index=False)
    jj.to_csv(dest/'B9A11S_eval_sensitivities.csv',index=False)
    med=df.groupby(['mode','layout','model'])[[p+'_absolute_error_pct' for p in PROPS]].median()
    agg=fr.groupby(['layout','model'])['forward_RMSE_mm'].median().to_dict()
    fig,axs=plt.subplots(1,3,figsize=(14,4.7))
    for i,p in enumerate(PROPS):
        ax=axs[i];x=np.arange(4);vals=[]
        for layout in ('baseline','optimized'):
            for model in ('frozen_B5','B5_plus_FIT_correction'):
                vals.append(float(med.loc[('FEM_noisy',layout,model),p+'_absolute_error_pct']))
        ax.bar(x,vals)
        ax.set_xticks(x);ax.set_xticklabels(['old B5','old B5+corr','opt B5','opt B5+corr'],rotation=25,ha='right')
        ax.set_title(p);ax.set_ylabel('Median noisy FEM inverse error (%)')
    fig.suptitle('B9A11S | FIT-only correction evaluated on INTERNAL TUNE; known FEM q')
    fig.tight_layout();fig.savefig(dest/'B9A11S_noisy_inverse_before_after.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    xx=np.arange(4)
    labels=['old B5','old B5+corr','opt B5','opt B5+corr']
    values=[agg[(layout,model)] for layout in ('baseline','optimized') for model in ('frozen_B5','B5_plus_FIT_correction')]
    ax.bar(xx,values);ax.set_xticks(xx);ax.set_xticklabels(labels,rotation=15)
    ax.set_ylabel('Median held-out U forward RMSE (mm)')
    ax.set_title('B9A11S | FEM vs predicted U on INTERNAL TUNE')
    fig.tight_layout();fig.savefig(dest/'B9A11S_forward_rmse_before_after.png',dpi=190);plt.close(fig)
    means=jj.groupby(['layout','model','property'])['importance'].mean()
    fig,ax=plt.subplots(figsize=(9,4.4));xx=np.arange(3)
    for k,(layout,model) in enumerate((('baseline','frozen_B5'),('baseline','B5_plus_FIT_correction'),
                                       ('optimized','frozen_B5'),('optimized','B5_plus_FIT_correction'))):
        ax.bar(xx+(k-1.5)*.19,[means.loc[(layout,model,p)] for p in PROPS],width=.18,label=layout+' '+model)
    ax.set_xticks(xx);ax.set_xticklabels(PROPS)
    ax.set_ylabel('RMS local response shift / U noise (5% property change)')
    ax.set_title('B9A11S | conditional model sensitivities (NOT exact FEM derivatives)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(dest/'B9A11S_sensitivity_before_after.png',dpi=190);plt.close(fig)
    report=dict(stage='B9A11S-evaluate',sensor_sites=args.sensors,
                noise_sigma_U_mm=args.sigma_u,noisy_repetitions=args.noise_reps,
                tune_cases_evaluated=len(fr)//4,rows=len(df),
                median_inverse_error_pct={str(k):{p:float(v[p+'_absolute_error_pct']) for p in PROPS}
                                          for k,v in med.iterrows()},
                median_forward_RMSE_mm={str(k):float(v) for k,v in agg.items()},
                corrected_model_is_fit_only_FEM_residual_regression=True,
                original_B5_unchanged=True,known_FEM_q=True,
                evaluation_is_INTERNAL_original_64J_TRAIN_tune=True,
                not_independent_validation=True,not_a_full_coupled_FE_NO=True,
                no_exact_FEM_sensitivity_jacobians=True,
                next='If fit-only correction improves independent internal tune forward RMSE and inverse errors, test NEW independent Abaqus subdomains with prescribed matched q; later fix coupled solver.')
    (dest/'B9A11S_eval_summary.json').write_text(json.dumps(report,indent=2))
    print('EVALUATION COMPLETE:',dest)
    print('MEDIAN INVERSE ERRORS:',med.to_string())
    print('WARNING: Fit-only FEM-supervised calibration and known FEM q; not a deployable twin.')


def selftest():
    rng=np.random.default_rng(5)
    X=np.c_[np.ones(36),rng.normal(size=(36,12))]
    truth=rng.normal(size=(13,19));Y=X@truth+rng.normal(scale=.005,size=(36,19))
    B,info=ridge_fit_cv(X,Y)
    assert info['cv_mse']<info['baseline_fit_mse'] and info['gain']>0
    assert np.linalg.norm(B-truth)/np.linalg.norm(truth)<0.12
    # The derivative contribution of the linear correction should be exact.
    c={'d':{'branch_mean':np.zeros(12),'branch_std':np.ones(12)}}
    f=basis_features(np.arange(12,dtype=float),c)
    assert f.shape==(13,) and f[1]==0 and f[-1]==11
    print('B9A11S SELFTEST PASS: FIT-only ridge LOOCV, correction, shapes')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('selftest','check','fit','evaluate'))
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='pilot01')
    p.add_argument('--design-tag',default='pilot01')
    p.add_argument('--variant',default='centered',choices=('centered','standardized'))
    p.add_argument('--loss',default='grad',choices=('U','grad'))
    p.add_argument('--sensors',type=int,default=20)
    p.add_argument('--cases',type=int,default=6)
    p.add_argument('--noise-reps',type=int,default=10)
    p.add_argument('--sigma-u',type=float,default=.0002)
    p.add_argument('--max-nfev',type=int,default=150)
    p.add_argument('--seed',type=int,default=651023)
    a=p.parse_args()
    if a.stage=='selftest':selftest();return
    if not (3<=a.sensors<=50 and 1<=a.cases<=12 and 1<=a.noise_reps<=100 and a.sigma_u>0):
        raise ValueError('Invalid experiment options')
    root,Q,configs,layouts,lock=load_project(a)
    if a.stage=='check':
        for reg,c in configs.items():
            ids=np.asarray(c['d']['fit'],int)
            valid=[int(k) for k in c['d']['tune'] if np.all(c['phys'][k,:3]>=c['lo']) and np.all(c['phys'][k,:3]<=c['hi'])]
            op=Q.ConditionalOperator(with_nodes(c,layouts[reg]['optimized']),valid[0])
            y,J=op.pred_and_jac(op.normalized_true(),True)
            print('CHECK',reg,'FIT',len(ids),'valid TUNE',len(valid),
                  'sensors',len(layouts[reg]['optimized']),'Jacobian',J.shape)
        print('B9A11S CHECK PASS: data/checkpoints/locked design; no fit or outputs written')
    elif a.stage=='fit':save_fit(root,a,configs,layouts,lock)
    else:evaluate(root,Q,a,configs,layouts,lock)

if __name__=='__main__':main()