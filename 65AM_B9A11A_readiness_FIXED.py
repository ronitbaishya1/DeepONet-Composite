#!/usr/bin/env python3
"""B9A11-A: NO TRAINING. Evaluate existing B5 predictions using B9A10-L+K.

Cohort: 12 INTERNAL development/tune cases per region, NOT independent B9A9.
Inputs: original 64J training data; verified B5 checkpoints; K frozen beta;
original L/K/H implementations, H geometry and 65V interface PCA anchor.
Abaqus RF for these 12 64J cases is NOT assumed available. Therefore the
'physics' residual is compared to the same operator applied to reference FEM U,
not to actual Abaqus reactions. This is a readiness audit, NOT force validation.
"""
import argparse,csv,hashlib,importlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib.util import spec_from_file_location,module_from_spec

LEFT='inner_left'; RIGHT='inner_right'
REGIONS=(LEFT,RIGHT)
OUT=Path('results/65AM_B9_coupled_feasibility/B9A11A_B5_readiness')
BETA=Path('results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_calibrated_coefficient.json')
VARIANTS=(('centered','U'),('centered','grad'),('standardized','U'),('standardized','grad'))
DEFAULT={LEFT:('standardized','grad'),RIGHT:('centered','grad')}
RESULTS=('B9A11A_cases.csv','B9A11A_gradients.csv','B9A11A_modes.csv',
         'B9A11A_summary.json','B9A11A_force_errors.png','B9A11A_gradient_health.png')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def local_import(path,name):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(str(path))
    spec=spec_from_file_location(name,str(path)); m=module_from_spec(spec);spec.loader.exec_module(m)
    return m


def csvwrite(path,rows):
    if not rows:raise RuntimeError('No data for '+str(path))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def value(x):
    return float(x.detach().cpu().item()) if isinstance(x,torch.Tensor) else float(x)


def as_np(x):return np.asarray(x,dtype=np.float64)


def geometry_match(X,Y):
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3:return False,math.inf
    x=np.asarray(X,dtype=np.float64);y=np.asarray(Y,dtype=np.float64)
    idx=np.lexsort((np.round(x[:,2],6),np.round(x[:,1],6),np.round(x[:,0],6)))
    idy=np.lexsort((np.round(y[:,2],6),np.round(y[:,1],6),np.round(y[:,0],6)))
    mx=float(np.abs(x[idx]-y[idy]).max())
    return mx<2e-5,mx


def choose(a,region):
    if a.all_variants:return VARIANTS
    v=getattr(a,'left' if region==LEFT else 'right')
    s=tuple(v.split('_',1))
    if len(s)!=2 or s not in VARIANTS:
        raise ValueError('Unknown model '+v)
    return (s,)


def check_source(root,a,load=False):
    # Original unmodified source code and checkpoint are required.
    required=('65AM_B5_common.py','65AM_B_common.py','65AM_B9A10L_torch_force.py',
              '65AM_B9A10K_hourglass_operator.py','65AM_B9A10H_mac_weakform_validation.py')
    for n in required:
        if not (root/n).is_file():raise FileNotFoundError('Missing original '+str(root/n))
    if not (root/BETA).is_file():raise FileNotFoundError('Missing frozen K beta: '+str(root/BETA))
    kc=json.loads((root/BETA).read_text())
    beta=float(kc['beta'])
    if not 0<beta<.05:raise RuntimeError('Unexpected calibration coefficient')
    # Require the exact B9A10H geometry used in K validation.
    hp=root/'B9A10H_exports/B9A10H_independent_fields.npz'
    hm=root/'B9A10H_exports/B9A10H_windows_summary.json'
    if not hp.is_file() or not hm.is_file():raise FileNotFoundError('Missing B9A10H authenticated exports')
    mh=json.loads(hm.read_text())
    if sha(hp)!=mh['output_sha256']:
        raise RuntimeError('H source checksum mismatch')
    ksum=root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_summary.json'
    if not ksum.is_file():raise FileNotFoundError('Missing K summary')
    ks=json.loads(ksum.read_text())
    if not ks.get('numerical_screening_pass',False) or abs(float(ks['beta_global'])-beta)>1e-12:
        raise RuntimeError('K fit/screening check failed')
    if ks['source_files_sha256']['h']!=sha(hp):raise RuntimeError('K and H belong to different dataset')
    # original module imports by unusual numeric prefix; import through importlib.
    sys.path.insert(0,str(root)) if str(root) not in sys.path else None
    c=importlib.import_module('65AM_B5_common')
    L=local_import(root/'65AM_B9A10L_torch_force.py','b9a11a_L_orig')
    H=local_import(root/'65AM_B9A10H_mac_weakform_validation.py','b9a11a_H_orig')
    with np.load(hp,allow_pickle=False) as hz:
        original={r:H.make_case(hz,'B9A9_'+('L' if r==LEFT else 'R')+'01') for r in REGIONS}
    anchorpath=root/'B9_coupled_anchor/B9_65V_anchor.npz'
    if not anchorpath.is_file():raise FileNotFoundError('PCA anchor missing')
    checked={}
    for region in REGIONS:
        d,mu,sd=c.prepare(root,region)
        ds=d['ds'];x=as_np(ds['coordinates']); conn=np.asarray(ds['element_node_indices'],dtype=np.int64)
        ref=original[region]
        if x.shape!=(1008,3) or conn.shape!=(728,8):
            raise RuntimeError('Original B5 mesh shape inconsistent with K mesh: '+region)
        same,diff=geometry_match(x,ref['X'])
        if not same:raise RuntimeError('B5 vs K/H geometry mismatch '+region+': '+str(diff))
        if np.any(conn<0) or np.any(conn>=len(x)):
            raise RuntimeError('Bad B5 connectivity '+region)
        phys=as_np(ds['branch_inputs'])
        if phys.ndim!=2 or phys.shape[1]!=12:
            raise RuntimeError('B5 original branch_inputs must be [n_cases,12] with [E1,E2,G12] in columns 0:3; got '+str(phys.shape))
        if not np.all(np.isfinite(phys[:,0:3])) or np.min(phys[:,0:3])<=1e3:
            raise RuntimeError('B5 branch_inputs[:,:3] must contain positive E1,E2,G12 in MPa; inspect source mapping')
        mask=np.zeros(len(x),dtype=bool);mask[d['fixed']]=True
        plane=np.isclose(x[:,0],x[:,0].min(),atol=2e-5)|np.isclose(x[:,0],x[:,0].max(),atol=2e-5)
        if not np.array_equal(mask,plane):
            raise RuntimeError('B5 prescribed DOFs do not match the 224 force-boundary nodes in '+region)
        labels=np.arange(1,len(x)+1,dtype=np.int64)
        force_model=L.ConstitutiveC3D8R(labels,x,conn+1,beta,root/'65AM_B9A10K_hourglass_operator.py')
        # Critically check local element ordering/derivatives against original 64J.
        AA=d['ds']; dN=as_np(c.b.AA.element_dN_numpy(AA))
        mismatch=float(np.max(np.abs(dN-force_model.grad.detach().cpu().numpy())))
        if mismatch>1.e-5:
            raise RuntimeError('C3D8R local node ordering / shape derivatives disagree with 64J: %.6g'%mismatch)
        models={}
        for variant,loss in choose(a,region):
            net,ck=c.load_model(root,variant,loss,region,d,mu,sd,torch.device('cpu'))
            models[variant+'_'+loss]=(net,ck)
        checked[region]=dict(d=d,mu=mu,sd=sd,force=force_model,boundary=torch.as_tensor(mask,dtype=torch.bool),
                             models=models,geometry_diff_mm=diff,grad_diff=mismatch)
        print('B9A11A CHECK PASS',region,'tune cases',len(d['tune']),'models',','.join(models),
              'K mesh diff mm',round(diff,8),'dN diff',round(mismatch,10),flush=True)
    return c,H,checked,beta,anchorpath,dict(H=sha(hp),K=sha(root/BETA))


def audit(root,a):
    torch.set_num_threads(2)
    ls=local_import(Path(__file__).resolve().with_name('65AM_B9A11A_losses.py'),'b9a11a_loss')
    c,H,checked,beta,anchorpath,sourcehash=check_source(root,a,load=True)
    out=root/OUT
    # Never overwrite any previous audit (including partial runs).
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output protected: move/rename existing folder first: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    records=[];grad_rows=[];mode_rows=[]
    with np.load(anchorpath,allow_pickle=False) as anchor:
        for region,state in checked.items():
            d=state['d']; ds=d['ds'];ids=d['tune'];b=state['boundary'];operator=state['force']
            ref_U=as_np(ds['U'][ids]);dref={}
            for k,case_id in enumerate(ids):
                mat=torch.as_tensor(as_np(ds['branch_inputs'][int(case_id),:3]),dtype=torch.float64)
                U=torch.as_tensor(ref_U[k],dtype=torch.float64)
                with torch.no_grad(): fr=operator(U,mat)
                dref[int(case_id)]=(fr,mat,U)
            # Single source of FEM truth; never used as model input.
            for modelname,(net,ck) in state['models'].items():
                variant,loss=modelname.split('_',1)
                up,coeff=c.predict(net,d,ids,torch.device('cpu'))
                if up.shape!=ref_U.shape:raise RuntimeError('B5 prediction shape mismatch')
                nfree=int((~b).sum());nf=int(b.sum())
                basis=torch.as_tensor(as_np(d['basis']),dtype=torch.float64)
                liftmap={int(v):i for i,v in enumerate(np.r_[d['fit'],d['tune']])}
                with torch.enable_grad():
                    for k,cid in enumerate(ids):
                        fr,mat,truth=dref[int(cid)]
                        lift=torch.as_tensor(as_np(d['lift'][liftmap[int(cid)]]),dtype=torch.float64)
                        coeff_t=torch.as_tensor(coeff[k],dtype=torch.float64).detach().requires_grad_(True)
                        ut=lift+torch.einsum('r,rnc->nc',coeff_t,basis)
                        # Rebuild from B5 POD coefficients, no FEM oracle coefficients.
                        maxgap=float(torch.abs(ut.detach()-torch.as_tensor(up[k],dtype=torch.float64)).max())
                        if maxgap>3e-6:raise RuntimeError('B5 coefficient-to-U reconstruction mismatch %.6g'%maxgap)
                        if float(torch.abs(ut.detach()[b]-truth[b]).max())>1e-5:
                            raise RuntimeError('B5 hard interface displacement condition not satisfied')
                        fp=operator(ut,mat)
                        metrics=ls.force_metrics(fp,fr,b)
                        goals=ls.normalized_objectives(ut,truth,fp,fr,b)
                        grads=[]
                        for v in goals.values():
                            g=torch.autograd.grad(v,coeff_t,retain_graph=True)[0]
                            grads.append(g)
                        stat=ls.gradient_characteristics(*grads)
                        if not stat['all_gradients_finite']:raise RuntimeError('Nonfinite B5 coefficient gradients')
                        scale=ls.norm(fr[b]).detach().item()
                        if not np.isfinite(scale) or scale<=1e-9:raise RuntimeError('Zero oracle traction norm')
                        record=dict(region=region,model=modelname,case_id=int(cid),cohort='original64J_TRAIN_internal_tune',
                           displacement_rel_pct=100*value(ls.norm(ut-truth)/torch.clamp(ls.norm(truth),min=1e-12)),
                           displacement_rmse_mm=value(torch.sqrt((ut-truth).square().mean())),
                           **{key:value(val) for key,val in metrics.items()},
                           loss_data=value(goals['data']),loss_equilibrium=value(goals['eq']),
                           loss_interface_oracle_diagnostic=value(goals['interface_oracle_diagnostic']),
                           n_free_nodes=nfree,n_interface_nodes=nf,E1_MPa=float(mat[0]),E2_MPa=float(mat[1]),G12_MPa=float(mat[2]))
                        records.append(record)
                        grad_rows.append(dict(region=region,model=modelname,case_id=int(cid),**stat))
                        # Optional original PCA interface comparison: no direct Abaqus RF label.
                        vals={'X':as_np(ds['coordinates']), 'node_labels':np.arange(1,len(ds['coordinates'])+1)}
                        for face in H.FACES['L' if region==LEFT else 'R']:
                            modes1,scl=H.pca_force(vals,fp.detach().numpy(),anchor,face)
                            modes0,_=H.pca_force(vals,fr.detach().numpy(),anchor,face)
                            for mi,(v,v0,s) in enumerate(zip(modes1,modes0,scl),1):
                                mode_rows.append(dict(region=region,model=modelname,case_id=int(cid),interface=face,mode=mi,
                                        predicted_internal_force_N=float(v),oracle_internal_force_N=float(v0),
                                        abs_difference_on_65V_scale=float(abs(v-v0)/s)))
                        print('B9A11A',region,modelname,'case',int(cid),'U',round(record['displacement_rel_pct'],4),
                              'free-floor',round(record['ref_free_pct'],4),'pred-free',round(record['pred_free_pct'],4),
                              'interface-diff',round(record['pred_minus_ref_interface_pct'],4),flush=True)
    csvwrite(out/RESULTS[0],records);csvwrite(out/RESULTS[1],grad_rows);csvwrite(out/RESULTS[2],mode_rows)
    # Plots are diagnostics, not accuracy guarantees. No pre-chosen favorable colors.
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for ai,region in enumerate(REGIONS):
        subset=[r for r in records if r['region']==region]
        for model in sorted(set(r['model'] for r in subset)):
            rr=[r for r in subset if r['model']==model]
            xx=np.arange(1,len(rr)+1)
            ax[ai].plot(xx,[r['pred_free_pct'] for r in rr],'-o',markersize=3,label=model+' predicted')
            ax[ai].plot(xx,[r['ref_free_pct'] for r in rr],':',linewidth=1,label=model+' FEM floor')
        ax[ai].set_title(region);ax[ai].set_xlabel('Internal tune case');ax[ai].set_ylabel('Free-node residual / FEM interface force (%)')
        ax[ai].legend(fontsize=7)
    fig.suptitle('B9A11-A | Reference operator floor vs B5 equilibrium residual')
    fig.tight_layout();fig.savefig(out/RESULTS[4],dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for ai,region in enumerate(REGIONS):
        subset=[r for r in grad_rows if r['region']==region]
        for model in sorted(set(r['model'] for r in subset)):
            rr=[r for r in subset if r['model']==model]
            ax[ai].plot(np.arange(1,len(rr)+1),[r['grad_cos_data_eq'] for r in rr],'-o',markersize=3,label=model)
        ax[ai].axhline(0,linestyle='--',color='gray');ax[ai].set_ylim(-1.05,1.05)
        ax[ai].set_title(region);ax[ai].set_xlabel('Internal tune case');ax[ai].set_ylabel('cosine: supervised vs equilibrium gradient')
        ax[ai].legend(fontsize=8)
    fig.suptitle('B9A11-A | Gradient directions in POD-coefficient space')
    fig.tight_layout();fig.savefig(out/RESULTS[5],dpi=150);plt.close(fig)
    summary={}
    for region in REGIONS:
        summary[region]={}
        for model in sorted(set(x['model'] for x in records if x['region']==region)):
            rr=[r for r in records if r['region']==region and r['model']==model]
            gg=[r for r in grad_rows if r['region']==region and r['model']==model]
            summary[region][model]={
               'cases':len(rr),'mean_displacement_rel_pct':float(np.mean([r['displacement_rel_pct'] for r in rr])),
               'mean_oracle_free_floor_pct':float(np.mean([r['ref_free_pct'] for r in rr])),
               'mean_pred_free_pct':float(np.mean([r['pred_free_pct'] for r in rr])),
               'mean_pred_minus_oracle_free_pct':float(np.mean([r['pred_minus_ref_free_pct'] for r in rr])),
               'mean_pred_minus_oracle_interface_pct':float(np.mean([r['pred_minus_ref_interface_pct'] for r in rr])),
               'mean_data_eq_gradient_cosine':float(np.mean([r['grad_cos_data_eq'] for r in gg])),
               'fraction_conflicting_gradients':float(np.mean([r['grad_cos_data_eq']<0 for r in gg])),
               'median_grad_eq_over_data':float(np.median([r['grad_eq_over_data'] for r in gg])),
               'finite_coefficient_gradients':bool(all(r['all_gradients_finite'] for r in gg))}
    payload={'step':'65AM-B9A11-A','diagnostic_cohort':'original 64J TRAIN 12 internal tuning cases per region',
      'NOT_independent_generalization':True, 'actual_Abaqus_RF_for_tune_cases_evaluated':False,
      'model':'frozen B9A10-L Cauchy stress plus frozen B9A10-K hourglass proxy',
      'beta':beta,'b5_unchanged':True,'broyden_65V_unchanged':True,
      'no_training':True,'no_ckpt_writes':True,'physics_loss_approved':False,
      'formula_reference':'f_ref is the same approximate operator on Abaqus reference displacement (not Abaqus RF)',
      'formula_equilibrium':'||f_B5[free]||^2 / ||f_ref[interface]||^2',
      'formula_interface':'oracle-labelled diagnostic ||f_B5[interface]-f_ref[interface]||^2 / ||f_ref[interface]||^2',
      'units':'mm, MPa, N; force errors relative to f_ref interface N',
      'failure_modes':['Different 64J and K mesh geometry','Different local C3D8R node ordering',
                       'Unstable gradients in coefficient space','Nonzero operator floor even at exact FE displacement',
                       'Interface oracle diagnostic cannot be used as unsupervised coupling loss'],
      'source_sha256':sourcehash,'results':summary}
    (out/RESULTS[3]).write_text(json.dumps(payload,indent=2))
    print('B9A11A COMPLETE | NOT TRAINING | results',out)
    print('Send: '+', '.join(RESULTS))


def selftest(root):
    ls=local_import(Path(__file__).resolve().with_name('65AM_B9A11A_losses.py'),'b9a11a_losses_selftest')
    ls.selftest()
    L=local_import(root/'65AM_B9A10L_torch_force.py','b9a11a_lself')
    H=root/'65AM_B9A10K_hourglass_operator.py'
    if not H.is_file():raise FileNotFoundError('Missing K code beside L code')
    L.selftest()
    # Short end-to-end loss-through-force test, single element.
    x=np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0],[0,0,3],[2,0,3],[2,1,3],[0,1,3]],float)
    m=L.ConstitutiveC3D8R(np.arange(1,9),x,np.arange(1,9).reshape(1,8),.004381715594969032,H)
    b=torch.tensor([True,True,False,False,True,True,False,False])
    c=torch.tensor([.001,-.002],dtype=torch.float64,requires_grad=True)
    basis=torch.from_numpy(np.random.default_rng(27).normal(size=(2,8,3))*.01).to(torch.float64)
    basis[:,b,:]=0
    truth=torch.as_tensor(x*.004,dtype=torch.float64)
    pred=truth+torch.einsum('r,rnc->nc',c,basis)
    mat=torch.tensor([46500.,12500.,4750.],dtype=torch.float64)
    ref=m(truth,mat).detach();fp=m(pred,mat)
    loss=ls.normalized_objectives(pred,truth,fp,ref,b)
    g=torch.autograd.grad(loss['eq']+loss['data'],c)[0]
    if not torch.isfinite(g).all() or torch.linalg.norm(g)<=1e-14:raise RuntimeError('No gradient to coefficient')
    print('B9A11A SELFTEST PASS | single element full differentiable force -> POD coefficients')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('selftest','check','audit'),required=True)
    p.add_argument('--left',default='standardized_grad',choices=[v+'_'+l for v,l in VARIANTS])
    p.add_argument('--right',default='centered_grad',choices=[v+'_'+l for v,l in VARIANTS])
    p.add_argument('--all-variants',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if a.stage=='selftest':selftest(root)
    elif a.stage=='check':check_source(root,a)
    elif a.stage=='audit':audit(root,a)

if __name__=='__main__':main()
