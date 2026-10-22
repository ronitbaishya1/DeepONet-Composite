#!/usr/bin/env python3
"""B9A11W0: guarded fixed-q weak-form audit for existing glass-epoxy B5 models.

Reuse the already exported B9A9/H Abaqus reference displacement, reaction, stress
and force audit. Evaluate an explicitly APPROXIMATE C3D8R/orthotropic/hourglass
candidate on FEM U, original B5 U and sensitivity-trained B5 U.  This is NOT
an Abaqus-equivalent force law, not a fully coupled digital twin, and not
physical surface-strain validation. B9A10L's gate remains 'not physics-loss
ready'. The optional refinement is an exploratory, non-production trial only.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib, json, sys
from pathlib import Path
import numpy as np

REGIONS=(('inner_left','L'),('inner_right','R'))
BASE='65AM_B9A11W0_weak_form_gate'
BETA_PATH='results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_calibrated_coefficient.json'
H_PATH='B9A10H_exports/B9A10H_independent_fields.npz'
H_META='B9A10H_exports/B9A10H_windows_summary.json'
PLAN='B9A9_transfer_to_windows/B9A9_case_plan.npz'
TRAIN='results/65AM_B9A11U_sensitivity_finetune'
FACES={'L':('left_inner','center_left'),'R':('center_right','right_inner')}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def json_out(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def csv_out(path,rows):
    if not rows:raise RuntimeError('No rows for '+str(path))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)


def metrics(force,ref,bc):
    force=np.asarray(force,dtype=np.float64);ref=np.asarray(ref,dtype=np.float64)
    if force.shape!=ref.shape or force.ndim!=2 or force.shape[1]!=3:
        raise ValueError('Force/reference shape mismatch')
    boundary=np.asarray(bc,bool)
    if len(boundary)!=len(force) or not boundary.any() or boundary.all():
        raise ValueError('Invalid fixed boundary')
    scale=float(np.linalg.norm(ref[boundary]))
    if scale<1e-12:raise ValueError('Zero Abaqus boundary reaction norm')
    diff=force-ref
    return dict(force_gap_pct=100*float(np.linalg.norm(diff))/scale,
        force_free_gap_pct=100*float(np.linalg.norm(diff[~boundary]))/scale,
        force_boundary_gap_pct=100*float(np.linalg.norm(diff[boundary]))/scale,
        free_force_pct=100*float(np.linalg.norm(force[~boundary]))/scale,
        reference_free_force_pct=100*float(np.linalg.norm(ref[~boundary]))/scale,
        boundary_reference_force_N=scale)


def test_math():
    f=np.zeros((8,3));r=np.zeros((8,3));bc=np.zeros(8,bool);bc[[0,7]]=True
    r[[0,7],0]=[10.,-10.]
    assert metrics(f,r,bc)['force_gap_pct']==100
    assert metrics(r,r,bc)['force_gap_pct']==0
    f[3,1]=.1
    assert metrics(f,r,bc)['force_free_gap_pct']>0
    assert abs(metrics(f,r,bc)['reference_free_force_pct'])<1e-12
    print('B9A11W0 SELFTEST PASS | fixed/free force norm separation')


def require(root):
    paths=[H_PATH,H_META,PLAN,BETA_PATH,
        'B9_coupled_anchor/B9_65V_anchor.npz',
        'B9_coupled_anchor/B9_65V_anchor_manifest.json',
        '65AM_B9A10L_torch_force.py','65AM_B9A10K_hourglass_operator.py',
        '65AM_B9A10H_mac_weakform_validation.py','65AM_B5_common.py',
        '65AM_B6_mac.py','65AM_B9_mac_saved_state.py']
    notfound=[str(root/p) for p in paths if not (root/p).is_file()]
    if notfound:raise FileNotFoundError('Missing project files:\n'+'\n'.join(notfound))
    manifest=json.loads((root/H_META).read_text())
    if sha(root/H_PATH)!=manifest['output_sha256']:
        raise RuntimeError('B9A10H export SHA mismatch')
    beta=float(json.loads((root/BETA_PATH).read_text())['beta'])
    if not 0<beta<1:raise RuntimeError('Invalid fitted hourglass beta')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return beta


def context(root,args):
    import torch
    from scipy.spatial import cKDTree
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region as hard
    torch.set_num_threads(min(3,torch.get_num_threads()))
    dev=torch.device('cpu')
    beta=require(root)
    B5=importlib.import_module('65AM_B5_common')
    B6=importlib.import_module('65AM_B6_mac')
    H=importlib.import_module('65AM_B9A10H_mac_weakform_validation')
    L=importlib.import_module('65AM_B9A10L_torch_force')
    B9=importlib.import_module('65AM_B9_mac_saved_state')
    anchor,_=B9.load_anchor(root)
    with np.load(root/PLAN,allow_pickle=False) as z:plan={k:np.asarray(z[k]).copy() for k in z.files}
    data={}
    for region,letter in REGIONS:
        d,mu,sd=B5.prepare(root,region)
        original,ck=B5.load_model(root,'centered','grad',region,d,mu,sd,dev)
        ckpath=B5.ck_path(root,'centered','grad',region)
        candidate=root/TRAIN/args.candidate_tag/region/'B9A11U_best_candidate.pt'
        if not candidate.is_file():raise FileNotFoundError(str(candidate))
        pack=torch.load(candidate,map_location='cpu',weights_only=False) # trusted locally produced file only
        if pack.get('region')!=region or pack.get('source_checkpoint_sha256')!=sha(ckpath):
            raise RuntimeError('Trained B5 checkpoint provenance mismatch: '+region)
        improved,ck2=B5.load_model(root,'centered','grad',region,d,mu,sd,dev)
        improved.load_state_dict(pack['state_dict'],strict=True)
        original.eval();improved.eval()
        branches=np.asarray(plan['x_'+region],dtype=np.float32)
        if branches.shape!=(12,12):raise RuntimeError('Unexpected B9A9 plan for '+region)
        if np.max(np.abs(branches[:,:3]))<=0:raise RuntimeError('Invalid material plan')
        ctx=B6.b.CO.interface_context(root,d['ds'],dev)
        n=len(d['ds']['coordinates'])
        with torch.no_grad():
            lifting=hard(torch.zeros((12,n,3),dtype=torch.float32),
                torch.as_tensor(branches),ctx).cpu().numpy()
        # Call existing verified B5 forward implementation; no custom remapping of PCA.
        u0=B6.predict_b5(d,original,branches,lifting,dev)
        u1=B6.predict_b5(d,improved,branches,lifting,dev)
        data[region]=dict(d=d,branch=branches,u0=u0,u1=u1,lift=lifting,
            coords=np.asarray(d['ds']['coordinates'],np.float64),
            checkpoint_shas={'original':sha(ckpath),'trained':sha(candidate)},
            cKDTree=cKDTree)
    return dict(beta=beta,torch=torch,dev=dev,H=H,L=L,anchor=anchor,data=data)


def assess(root,args,write):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c=context(root,args)
    H=c['H']; L=c['L']; torch=c['torch']; result=[]; modes=[]
    with np.load(root/H_PATH,allow_pickle=False) as z:
        for region,letter in REGIONS:
            x=c['data'][region]
            for i in range(args.cases):
                job=f'B9A9_{letter}{i+1:02d}'
                a=H.make_case(z,job)
                dist,ids=x['cKDTree'](x['coords']).query(a['X'],k=1)
                if len(np.unique(ids))!=len(ids) or dist.max()>2e-5:
                    raise RuntimeError('Invalid B5/Abaqus node matching '+job)
                uref=np.asarray(a['U'],np.float64)
                # The known FEM interface must agree with the predicted hard lift;
                # check this BEFORE drawing physical conclusions from forces.
                bc=H.boundary(a['X'])
                lift=x['lift'][i][ids,:]
                bc_error=float(np.max(np.abs(lift[bc]-uref[bc])))
                if bc_error>args.bc_tolerance_mm:
                    raise RuntimeError('%s prescribed-q mismatch %.7g mm (limit %.7g). '
                        'Do NOT treat this as a matched condition.'%(
                        job,bc_error,args.bc_tolerance_mm))
                fm=L.ConstitutiveC3D8R(a['node_labels'],a['X'],a['connectivity'],
                    c['beta'],root/'65AM_B9A10K_hourglass_operator.py',
                    device='cpu',dtype=torch.float64)
                params=torch.tensor(x['branch'][i,:3],dtype=torch.float64)
                for name,u in [('FEM',uref),('original_B5',x['u0'][i][ids]),
                               ('trained_B5',x['u1'][i][ids])]:
                    u=np.asarray(u,np.float64)
                    with torch.no_grad():
                        pred=fm(torch.tensor(u,dtype=torch.float64),params).cpu().numpy()
                    m=metrics(pred,a['RF'],bc)
                    m.update(dict(job=job,region=region,model=name,
                        displacement_rmse_mm=float(np.sqrt(np.mean((u-uref)**2))),
                        displacement_rel_pct=100*float(np.linalg.norm(u-uref))/max(float(np.linalg.norm(uref)),1e-12),
                        max_known_q_mismatch_mm=bc_error))
                    result.append(m)
                    for iface in FACES[letter]:
                        ps,scale=H.pca_force(a,pred,c['anchor'],iface)
                        rs,_=H.pca_force(a,a['RF'],c['anchor'],iface)
                        for k,v in enumerate((ps-rs)/scale):
                            modes.append(dict(job=job,region=region,model=name,interface=iface,
                                pca_mode=k+1,normalized_error=float(v)))
                print('CHECK',job,'FEM force gap %.4f%%'%result[-3]['force_gap_pct'],
                      'B5 force gap %.4f%%'%result[-2]['force_gap_pct'],
                      'trained %.4f%%'%result[-1]['force_gap_pct'],flush=True)
    fem=[r for r in result if r['model']=='FEM']
    # Preliminary screening ONLY, not Abaqus physical equivalence approval.
    screen=max(r['force_gap_pct'] for r in fem)<=args.max_reference_force_pct and \
           max(r['force_free_gap_pct'] for r in fem)<=args.max_reference_free_pct
    report={'stage':'B9A11W0','scope':'Fixed prescribed q, 24 previously inspected B9A9 subdomain cases',
        'paper_inspiration':'Weak-form equilibrium and test-time residual correction',
        'force_law':'B9A10L approximate finite-deformation orthotropic + B9A10K fitted hourglass',
        'fitted_hourglass_beta':c['beta'],
        'FEM_reference_force_control_pass':bool(screen),
        'force_law_approved_for_training':False,
        'force_law_approved_for_coupled_inverse':False,
        'physics_residual_does_not_prove_exact_Abaqus_equilibrium':True,
        'interface_from_reference_FEM':True,'new_abq_jobs':0,
        'max_reference_force_pct':max(r['force_gap_pct'] for r in fem),
        'max_reference_free_pct':max(r['force_free_gap_pct'] for r in fem),
        'original_B5_checkpoints_frozen':True,'regions':{}}
    for region,_ in REGIONS:
        report['regions'][region]={}
        for model in ('FEM','original_B5','trained_B5'):
            ss=[r for r in result if r['region']==region and r['model']==model]
            mm=[v['normalized_error'] for v in modes if v['region']==region and v['model']==model]
            report['regions'][region][model]={
                'mean_force_gap_pct':float(np.mean([v['force_gap_pct'] for v in ss])),
                'max_force_gap_pct':float(max(v['force_gap_pct'] for v in ss)),
                'rms_free_force_gap_pct':float(np.sqrt(np.mean([v['force_free_gap_pct']**2 for v in ss]))),
                'mean_U_RMSE_mm':float(np.mean([v['displacement_rmse_mm'] for v in ss])),
                'PCA_force_error_RMS_65V_scale':float(np.sqrt(np.mean(np.square(mm))))}
    if not write:
        print('W0 CHECK PASS | existing FEM/H, original/trained B5, candidate force law')
        print('W0 control screening:',screen,'| still NOT physics-loss approved')
        return report
    out=root/'results'/BASE/args.tag
    if out.exists():raise FileExistsError('Results protected; use a new --tag: '+str(out))
    out.mkdir(parents=True)
    csv_out(out/'B9A11W0_case_force_errors.csv',result)
    csv_out(out/'B9A11W0_interface_PCA_errors.csv',modes)
    json_out(out/'B9A11W0_summary.json',report)
    labels=['FEM','original_B5','trained_B5']
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,(region,_) in zip(axs,REGIONS):
        rr=report['regions'][region]
        y=[rr[k]['rms_free_force_gap_pct'] for k in labels]
        ax.bar(labels,y)
        ax.set_title(region);ax.set_ylabel('Free-node force gap / FEM reaction norm (%)')
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('B9A11W0 candidate force law — SAME prescribed FEM q; not production physics')
    fig.tight_layout();fig.savefig(out/'B9A11W0_free_force_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,(region,_) in zip(axs,REGIONS):
        rr=report['regions'][region]
        y=[rr[k]['PCA_force_error_RMS_65V_scale'] for k in labels]
        ax.bar(labels,y)
        ax.set_title(region);ax.set_ylabel('Generalized interface force error (65V normalized units)')
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('B9A11W0 interface force: stress/constitutive approximation + B5 error')
    fig.tight_layout();fig.savefig(out/'B9A11W0_interface_PCA_comparison.png',dpi=170);plt.close(fig)
    print('W0 OUTPUT:',out)
    print('W0 FEM reference control:',screen,'| physics-loss readiness: NO')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['selftest','check','run'])
    p.add_argument('--project',default='.')
    p.add_argument('--candidate-tag',default='pilot01')
    p.add_argument('--tag',default='audit01')
    p.add_argument('--cases',type=int,default=12)
    p.add_argument('--bc-tolerance-mm',type=float,default=2e-4)
    p.add_argument('--max-reference-force-pct',type=float,default=.5)
    p.add_argument('--max-reference-free-pct',type=float,default=.5)
    a=p.parse_args()
    if a.stage=='selftest':test_math();return
    if not 1<=a.cases<=12 or a.bc_tolerance_mm<=0 or a.max_reference_force_pct<=0 or a.max_reference_free_pct<=0:
        p.error('Invalid experiment settings')
    assess(Path(a.project).resolve(),a,write=(a.stage=='run'))

if __name__=='__main__':main()