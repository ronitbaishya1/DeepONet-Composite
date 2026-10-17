#!/usr/bin/env python3
"""65AM-B9A10-K / Mac: fit ONE scalar hourglass-proxy scale from the TWO
B9A10-J controlled-anchor measurements; freeze; evaluate all 24 B9A9 cases.

No new Abaqus jobs, B5 training, Broyden or modification of existing outputs.
This tests a candidate proxy against actual Abaqus RF, not exact Abaqus forces.
Results must not be reused for fitting/tuning after seeing the held-out labels.
"""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIONS=('L','R')
NAMES=[f'B9A9_{r}{k:02d}' for r in REGIONS for k in range(1,13)]
OUT=Path('results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def import_code(path,alias):
    spec=importlib.util.spec_from_file_location(alias,str(path))
    if not path.is_file() or spec is None:raise FileNotFoundError(str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))


def inputs(root):
    paths={
      'j':root/'B9A10J_exports'/'B9A10J_stabilization_fields.npz',
      'jm':root/'B9A10J_exports'/'B9A10J_windows_summary.json',
      'h':root/'B9A10H_exports'/'B9A10H_independent_fields.npz',
      'hm':root/'B9A10H_exports'/'B9A10H_windows_summary.json',
      'plan':root/'B9A9_transfer_to_windows'/'B9A9_case_plan.npz',
      'anchor':root/'B9_coupled_anchor'/'B9_65V_anchor.npz',
      'hcode':root/'65AM_B9A10H_mac_weakform_validation.py',
      'op':root/'65AM_B9A10K_hourglass_operator.py',
    }
    for name,path in paths.items():
        if not path.is_file():raise FileNotFoundError('Missing '+name+': '+str(path))
    jinfo=json.loads(paths['jm'].read_text());hinfo=json.loads(paths['hm'].read_text())
    if digest(paths['j'])!=jinfo['export_sha256']:raise RuntimeError('J exported NPZ SHA mismatch')
    if digest(paths['h'])!=hinfo['output_sha256']:raise RuntimeError('H exported NPZ SHA mismatch')
    if len(jinfo.get('cases',[]))!=8 or len(hinfo.get('cases',[]))!=24:
        raise RuntimeError('Expected 8 controlled J and 24 independent H cases')
    if {c['job'] for c in jinfo['cases']}!={f'B9A10J_{r}_{s}' for r in REGIONS for s in ('S05','S10','S20','ENH')}:
        raise RuntimeError('Wrong J jobs, or incomplete export')
    if {c['job'] for c in hinfo['cases']}!=set(NAMES):raise RuntimeError('Incomplete 24-case H export')
    with np.load(paths['plan'],allow_pickle=False) as z:
        for r in ('inner_left','inner_right'):
            if z['x_'+r].shape!=(12,12):raise RuntimeError('B9A9 plan wrong case features')
    with np.load(paths['anchor'],allow_pickle=False) as z:
        if z['material'].shape!=(3,) or float(z['material'][2])<=0:
            raise RuntimeError('Saved anchor material missing G12')
    return paths,jinfo,hinfo


def as_j(z,tag,J):
    return J.load(z,tag)


def csvsave(p,rows):
    if not rows:raise RuntimeError('Empty results '+str(p))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metric(d,f):
    """Percentage of corresponding Abaqus boundary-reaction norm."""
    X=d['X'];b=np.isclose(X[:,0],X[:,0].min(),atol=2e-5)|np.isclose(X[:,0],X[:,0].max(),atol=2e-5)
    if int(b.sum())!=224:raise RuntimeError('Wrong interface mask')
    scale=norm(d['RF'][b]);
    if scale<1e-9:raise RuntimeError('Zero Abaqus interface reaction')
    gap=f-d['RF']
    return 100*norm(gap)/scale,100*norm(gap[b])/scale,100*norm(gap[~b])/scale


def template(d,g12,K):
    idx,P,length=K.geometry(d['node_labels'],d['X'],d['connectivity'])
    return K.numpy_template(d['U'],idx,P,length,g12)


def run(root,check_only,device):
    import torch
    paths,jmeta,hmeta=inputs(root)
    K=import_code(paths['op'],'b9a10k_hourglass_operator')
    H=import_code(paths['hcode'],'b9a10h_existing_audit')
    if check_only:
        with np.load(paths['j'],allow_pickle=False) as z:
            for r in REGIONS:
                J=import_code(root/'65AM_B9A10J_mac_sensitivity.py','b9a10j_existing') if (root/'65AM_B9A10J_mac_sensitivity.py').exists() else None
                # Check exact fields even without the original J Mac script.
                for s in ('S05','S10','S20'):
                    name=f'B9A10J_{r}_{s}'
                    for part in ('node_labels','X','U','RF','element_labels','connectivity'):
                        if name+'_'+part not in z:raise RuntimeError('Missing '+name+'_'+part)
        print('B9A10K CHECK PASS | J calibration fields, H unseen cases, B9A9 plan, saved anchor')
        return
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Use --device cpu; PyTorch may need upgrading')
    out=root/OUT
    if not out.is_dir():raise RuntimeError('Create in Finder first: '+str(out))
    output=[out/name for name in ('B9A10K_anchor_fit.csv','B9A10K_independent_cases.csv',
      'B9A10K_pca_modes.csv','B9A10K_summary.json',
      'B9A10K_generalization.png','B9A10K_pca_comparison.png',
      'B9A10K_calibrated_coefficient.json')]
    if any(p.exists() for p in output):raise RuntimeError('Results protected. Move prior results manually in Finder before rerunning.')
    importJ=import_code(root/'65AM_B9A10J_mac_sensitivity.py','b9a10j_previous') if (root/'65AM_B9A10J_mac_sensitivity.py').is_file() else None
    # We use J for the exact stress integral when present, otherwise H for
    # stress assembly. Both use the same validated C3D8R Piola convention.
    if importJ is None:raise FileNotFoundError('Place earlier 65AM_B9A10J_mac_sensitivity.py at project root')
    J=importJ
    ab={};calibration=[];denom=0.;numer=0.
    with np.load(paths['anchor'],allow_pickle=False) as anchor,\
         np.load(paths['j'],allow_pickle=False) as zj:
        G12_anchor=float(anchor['material'][2]);
        for r in REGIONS:
            cases={s:as_j(zj,f'B9A10J_{r}_{s}',J) for s in ('S05','S10','S20')}
            d=cases['S10']
            for s in ('S05','S20'):
                for k in ('node_labels','X','U','connectivity','element_labels'):
                    if not np.allclose(cases[s][k],d[k],rtol=0,atol=1e-10):
                        raise RuntimeError('J case kinematics differ '+r+s+k)
            measured=(cases['S20']['RF']-cases['S05']['RF'])/1.5
            fstress=J.assemble(d)
            before=metric(d,fstress)[0]
            if abs(before-{'L':1.1770788954990998,'R':1.1730814646256726}[r])>0.01:
                raise RuntimeError('B9A10J stress-only baseline regression failure '+r)
            t=template(d,G12_anchor,K)
            n=float(np.sum(t*measured));dn=float(np.sum(t*t))
            if dn<1e-15:raise RuntimeError('Zero analytic hourglass force proxy')
            numer+=n;denom+=dn
            ab[r]=(d,t,measured,fstress)
            calibration.append(dict(region='inner_left' if r=='L' else 'inner_right',
               original_stress_only_gap_pct=before,unscaled_template_norm_N=norm(t),
               measured_stiffness_slope_norm_N=norm(measured),
               template_measured_direction_cosine=n/(np.sqrt(dn)*norm(measured)),
               independent_numer=n,independent_denom=dn))
        beta=numer/denom
        if not np.isfinite(beta) or beta<=0:
            raise RuntimeError('Nonpositive pooled beta: proxy physically inconsistent. STOP: '+str(beta))
        # K coefficient frozen from just two anchor runs. No validation data used to fit it.
        for record,r in zip(calibration,REGIONS):
            d,t,measured,fstress=ab[r]
            residual=beta*t-measured
            record.update(beta_frozen=beta,
                predicted_slope_relative_error_pct=100*norm(residual)/max(norm(measured),1e-12),
                predicted_total_gap_pct=metric(d,fstress+beta*t)[0],
                stress_only_gap_pct=metric(d,fstress)[0])
        print('B9A10K FIT | pooled scalar beta = %.8g | trained on only L/R saved 65V states'%beta)
        print('  anchor slope errors L/R:',*[('%.3f%%'%rec['predicted_slope_relative_error_pct']) for rec in calibration])
        # Verify Torch assembled proxy equals exactly the same frozen NumPy force.
        for r in REGIONS:
            d,t,_,_=ab[r]
            model=K.HourglassProxy(d['node_labels'],d['X'],d['connectivity'],beta,device=device)
            dtype=torch.float32 if device=='mps' else torch.float64
            u=torch.tensor(d['U'],dtype=dtype,device=device,requires_grad=True)
            pred=model(u,torch.tensor(G12_anchor,dtype=dtype,device=device))
            target=beta*t
            tol=3e-4 if device=='mps' else 1e-9
            if norm(pred.detach().cpu().numpy()-target)/max(norm(target),1e-12)>tol:
                raise RuntimeError('Torch/NumPy force discrepancy on anchor '+r)
            e=model.energy(u,G12_anchor)
            grad=torch.autograd.grad(e,u)[0]
            if norm((grad-pred).detach().cpu().numpy())/max(norm(target),1e-12)>tol:
                raise RuntimeError('Energy derivative inconsistent with force '+r)
            print('B9A10K AUTOGRAD PASS',r,'device',device)
        rows=[];modes=[];groups=['material']*4+['interface']*4+['combined']*4
        with np.load(paths['h'],allow_pickle=False) as hz,np.load(paths['plan'],allow_pickle=False) as plans:
            for r in REGIONS:
                mat=np.asarray(plans['x_inner_left' if r=='L' else 'x_inner_right'])
                for i in range(12):
                    tag=f'B9A9_{r}{i+1:02d}'
                    d=H.make_case(hz,tag)
                    if not np.array_equal(d['node_labels'],ab[r][0]['node_labels']) or\
                       not np.array_equal(d['connectivity'],ab[r][0]['connectivity']) or\
                       np.max(abs(d['X']-ab[r][0]['X']))>1e-8:
                        raise RuntimeError('Different mesh, must rebuild model: '+tag)
                    g12=float(mat[i,2]);
                    if g12<=0:raise RuntimeError('Invalid G12 for '+tag)
                    fs,_=H.assemble(d) # exact S from ODB, NOT differentiable from predicted U
                    fh=template(d,g12,K)*beta
                    old=metric(d,fs);new=metric(d,fs+fh)
                    true_hg=d['RF']-fs
                    # This true_hg contains the small remaining non-stabilization terms.
                    relative=100*norm(fh-true_hg)/max(norm(true_hg),1e-12)
                    cosine=np.sum(fh*true_hg)/max(norm(fh)*norm(true_hg),1e-12)
                    rows.append(dict(job=tag,region='inner_left' if r=='L' else 'inner_right',
                        condition=groups[i],G12_MPa=g12,
                        stress_only_total_gap_pct=old[0],proxy_total_gap_pct=new[0],
                        stress_only_interface_gap_pct=old[1],proxy_interface_gap_pct=new[1],
                        stress_only_free_gap_pct=old[2],proxy_free_gap_pct=new[2],
                        proxy_vs_missing_force_relative_pct=relative,
                        proxy_vs_missing_force_cosine=cosine,
                        improvement_pct=(old[0]-new[0])/max(old[0],1e-12)*100))
                    for iface in H.FACES[r]:
                        pr,scale=H.pca_force(d,d['RF'],anchor,iface)
                        ps,_=H.pca_force(d,fs,anchor,iface)
                        pp,_=H.pca_force(d,fs+fh,anchor,iface)
                        for j in range(len(scale)):
                            modes.append(dict(job=tag,interface=iface,mode=j+1,
                                old_abs_scaled=abs((ps[j]-pr[j])/scale[j]),
                                new_abs_scaled=abs((pp[j]-pr[j])/scale[j])))
        # Field responses: use no unseen labels to update beta.
        groups_r={}
        for r in REGIONS:
            rs=[a for a in rows if a['region']==('inner_left' if r=='L' else 'inner_right')]
            groups_r[r]={
              'cases':len(rs),
              'stress_only_pooled_total_pct':float(np.sqrt(np.mean(np.square([a['stress_only_total_gap_pct'] for a in rs])))),
              'proxy_pooled_total_pct':float(np.sqrt(np.mean(np.square([a['proxy_total_gap_pct'] for a in rs])))),
              'mean_missing_force_relative_error_pct':float(np.mean([a['proxy_vs_missing_force_relative_pct'] for a in rs])),
              'max_proxy_total_gap_pct':max(a['proxy_total_gap_pct'] for a in rs),
              'improved_case_count':sum(a['proxy_total_gap_pct']<a['stress_only_total_gap_pct'] for a in rs)}
        all_improved=all(d['improved_case_count']==12 for d in groups_r.values())
        # Screening guidelines, not proof of exact Abaqus law nor approval to train B5.
        pass_numerical=all(d['proxy_pooled_total_pct']<=.1 and d['max_proxy_total_gap_pct']<=.15 for d in groups_r.values())
        state=dict(step='65AM-B9A10-K',description='Positive semidefinite affine-free proxy, NOT exact Abaqus hourglass law',
           calibration_dataset='two B9A10J full-field replay stiffness slopes ONLY',
           heldout_dataset='24 B9A9 independent subdomain cases; never used to tune beta',
           beta_global=beta,model='E=beta*G12/2 sum_e V^(1/3)||P_affine_null u||^2',
           anchor=calibration,independent=groups_r,
           numerical_screening_pass=bool(pass_numerical and all_improved),
           numerical_screening_thresholds={'pooled_total_max_pct':.1,'worst_total_max_pct':.15,'improve_all_cases':True},
           physics_loss_ready=False,full_coupled_65V_approved=False,
           limitations=['Uses Abaqus ODB local stresses for stress-only term (not differentiable).',
                        'Hourglass proxy uses scalar G12 and is not Abaqus constitutive stabilization law.',
                        'Training beta on two states does not prove other materials/geometries.',
                        'No B5 training or Broyden solver changes.'],
           source_files_sha256={k:digest(paths[k]) for k in ('j','h','plan','anchor')},
           torch_device_tested=device)
        csvsave(output[0],calibration);csvsave(output[1],rows);csvsave(output[2],modes)
        output[3].write_text(json.dumps(state,indent=2))
        # Also store coefficient separately to make model setup explicit, never silently inject into B5.
        output[6].write_text(json.dumps({'beta':float(beta),'shear_input_name':'G12_MPa',
            'geometry':'element_center_affine_null_orthogonal_projector',
            'trained_on':['B9A10J_L_S05/S20','B9A10J_R_S05/S20'],
            'physical_exactness':'unverified candidate'},indent=2))
        fig,axs=plt.subplots(1,2,figsize=(13,4))
        for ax,r in zip(axs,REGIONS):
            sr=[e for e in rows if e['region']==('inner_left' if r=='L' else 'inner_right')]
            xs=np.arange(12);ax.plot(xs,[e['stress_only_total_gap_pct'] for e in sr],marker='o',label='Stress-only reconstruction')
            ax.plot(xs,[e['proxy_total_gap_pct'] for e in sr],marker='s',label='Stress + frozen HG proxy')
            ax.set_title('inner_left' if r=='L' else 'inner_right');ax.set_xticks(xs,np.arange(1,13));ax.set_xlabel('Independent B9A9 case')
            ax.set_ylabel('All-node force difference / Abaqus boundary RF norm (%)');ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(output[4],dpi=170);plt.close(fig)
        names=('left_inner','center_left','center_right','right_inner')
        mode_old=[];mode_new=[];labels=[]
        for face in names:
            for md in sorted({x['mode'] for x in modes if x['interface']==face}):
                sub=[x for x in modes if x['interface']==face and x['mode']==md]
                labels.append(face+' '+str(md));mode_old.append(np.sqrt(np.mean([x['old_abs_scaled']**2 for x in sub])));mode_new.append(np.sqrt(np.mean([x['new_abs_scaled']**2 for x in sub])))
        fig,ax=plt.subplots(figsize=(max(11,len(labels)*.58),4.5))
        x=np.arange(len(labels));ax.bar(x-.2,mode_old,.38,label='Stress only');ax.bar(x+.2,mode_new,.38,label='With candidate proxy')
        ax.set_xticks(x,labels,rotation=65,ha='right',fontsize=8);ax.set_ylabel('PCA force error / saved 65V scale');ax.legend();fig.tight_layout();fig.savefig(output[5],dpi=170);plt.close(fig)
        for r,v in groups_r.items():print('B9A10K HELD-OUT',r,'old %.5f%% -> proxy %.5f%% | improved %d/12 | worst %.4f%%'%(v['stress_only_pooled_total_pct'],v['proxy_pooled_total_pct'],v['improved_case_count'],v['max_proxy_total_gap_pct']))
        print('B9A10K DONE | numerical screening:',state['numerical_screening_pass'],'| physics loss ready: NO')
        print('Saved to:',out)


def selftest(root):
    K=import_code(root/'65AM_B9A10K_hourglass_operator.py','b9a10k_hourglass_operator')
    K.selftest()
    # Fitting exactly-known scalar on a synthetic held-back response.
    rng=np.random.default_rng(42)
    templates=[rng.normal(size=(10,3)) for _ in range(2)]
    true=1.375
    responses=[true*t for t in templates]
    beta=sum(np.sum(t*r) for t,r in zip(templates,responses))/sum(np.sum(t*t) for t in templates)
    if abs(beta-true)>1e-12:raise RuntimeError('Calibration recovery test failed')
    print('B9A10K SELFTEST PASS | calibrate frozen positive scale | no validation leakage')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('selftest','check','validate'),default='validate')
    p.add_argument('--device',choices=('cpu','mps'),default='cpu')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if a.stage=='selftest':selftest(root)
    else:run(root,a.stage=='check',a.device)

if __name__=='__main__':main()
