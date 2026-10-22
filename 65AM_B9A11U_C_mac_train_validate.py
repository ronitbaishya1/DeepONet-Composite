#!/usr/bin/env python3
"""B9A11U-C: sensitivity-aware *COPY* of frozen B5, fixed interface q.

Run stages: selftest, check, train, compare. Uses NEW FIT-only Abaqus U pairs for
training, one untouched FIT-source pair for development, and the old B9A11T
anchor ONLY in stage compare. No full 7-region coupling, no physics residual added.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, copy, sys
from pathlib import Path
import numpy as np
REGIONS=('inner_left','inner_right')
PROPS=('E1','E2','G12'); SHIFTS=('E1_M','E1_P','E2_M','E2_P','G12_M','G12_P')
EPS=.03
RESULTS='65AM_B9A11U_sensitivity_finetune'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for q in iter(lambda:f.read(1<<20),b''):h.update(q)
    return h.hexdigest()

def center_deltas(U):
    """U shape (centers,6,nodes,3), returns (centers,3,nodes,3) +/- half shifts."""
    v=np.asarray(U)
    if v.ndim!=4 or v.shape[1]!=6:raise ValueError('Expected [centers,6,nodes,xyz]')
    return (v[:,1::2]-v[:,::2])/2.

def test_math():
    rng=np.random.default_rng(21);a=rng.normal(size=(4,6,11,3))
    j=center_deltas(a);assert j.shape==(4,3,11,3)
    assert np.allclose(j[:,0],(a[:,1]-a[:,0])/2)
    print('B9A11U-C SELFTEST PASS')

def read_inputs(root):
    plan=root/'B9A11U_plan'; exp=root/'B9A11U_exports'
    pa=plan/'B9A11U_plan.npz';pm=plan/'B9A11U_plan_manifest.json'
    ef=exp/'B9A11U_training_pairs.npz';em=exp/'B9A11U_windows_summary.json'
    for p in (pa,pm,ef,em):
        if not p.is_file():raise FileNotFoundError('Missing plan/FEM export '+str(p))
    mp=json.loads(pm.read_text());me=json.loads(em.read_text())
    if sha(pa)!=mp['plan_sha256'] or sha(ef)!=me['npz_sha256']:
        raise RuntimeError('STOP: plan or Windows FEM export checksum mismatch')
    if me['plan_sha256']!=mp['plan_sha256'] or me['job_count']!=48 or not me.get('all_jobs_pass'):
        raise RuntimeError('Windows export has not verified 48 planned jobs')
    with np.load(pa,allow_pickle=False) as z:p={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(ef,allow_pickle=False) as z:e={k:np.asarray(z[k]).copy() for k in z.files}
    if tuple(e['shift_ids'].astype(str))!=SHIFTS or abs(float(e['fraction'])-EPS)>1e-10:
        raise RuntimeError('Bad FEM perturbations')
    for r in REGIONS:
        if e['U_'+r].shape!=(4,6,1008,3) or e['branch_'+r].shape!=(4,12):
            raise RuntimeError('Unexpected FEM U/branch shape '+r)
        if not np.allclose(e['branch_'+r],p['branch_'+r],rtol=0,atol=1e-8):
            raise RuntimeError('Planned branch inputs do not match solved FEM cases '+r)
        if not np.array_equal(e['fit_ids_'+r],p['fit_ids_'+r]):raise RuntimeError('Wrong FIT ids '+r)
        if tuple(e['roles_'+r].astype(str))!=('train','train','train','dev'):
            raise RuntimeError('Expected 3 training + 1 development centers '+r)
    return mp,me,p,e

def context(root):
    import torch
    from scipy.spatial import cKDTree
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    mp,me,plan,e=read_inputs(root)
    c=importlib.import_module('65AM_B5_common')
    oldtrain=importlib.import_module('65AM_B_train')
    b6=importlib.import_module('65AM_B6_mac')
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region as hard
    if mp['anchor_sha256']!=me['source_anchor_sha256']:
        raise RuntimeError('PCA anchor checksum disagrees')
    out={}
    dev=torch.device('cpu')
    for r in REGIONS:
        d,mu,sd=c.prepare(root,r)
        if sha(c.ck_path(root,'centered','grad',r))!=mp['original_B5_checkpoint_hashes'][r]:
            raise RuntimeError('Original B5 checkpoint changed since planning '+r)
        model,ck=c.load_model(root,'centered','grad',r,d,mu,sd,dev)
        ids=e['fit_ids_'+r].astype(int)
        if not np.isin(ids,d['fit']).all() or np.isin(ids,d['tune']).any():
            raise RuntimeError('FIT/TUNE contamination '+r)
        if not np.allclose(d['ds']['branch_inputs'][ids],e['branch_'+r],atol=2e-5,rtol=0):
            raise RuntimeError('Mac train branch changed '+r)
        coords=np.asarray(d['ds']['coordinates'],float)
        dist,order=cKDTree(np.asarray(e['xyz_'+r],float)).query(coords,k=1)
        if len(np.unique(order))!=len(coords) or float(dist.max())>2e-5:
            raise RuntimeError('FEM/B5 coordinates mismatch '+r)
        Uref=np.asarray(e['U_'+r][:,:,order,:],dtype=np.float32)
        base=np.asarray(e['branch_'+r],np.float32)
        matbase=base[:,:3]
        branches=np.repeat(base[:,None,:],6,axis=1)
        for j in range(3):
            branches[:,2*j,j]=matbase[:,j]*(1-EPS)
            branches[:,2*j+1,j]=matbase[:,j]*(1+EPS)
        if not np.all(np.isfinite(branches)):raise RuntimeError('Nonfinite branches')
        for j in range(3):
            lo=np.min(d['ds']['branch_inputs'][d['fit'],j]);hi=np.max(d['ds']['branch_inputs'][d['fit'],j])
            if branches[:,:,j].min()<lo-1e-3 or branches[:,:,j].max()>hi+1e-3:
                raise RuntimeError('New FEM materials outside B5 FIT range '+r+' '+PROPS[j])
        ctx=b6.b.CO.interface_context(root,d['ds'],dev)
        with torch.no_grad():
            x=torch.as_tensor(branches.reshape(-1,12),dtype=torch.float32)
            lift=hard(torch.zeros((len(x),len(coords),3),dtype=torch.float32),x,ctx)
        lift=lift.reshape(4,6,len(coords),3)
        # Exact fixed q: every perturbation about a FIT center shares identical interface lift.
        if float(torch.max(torch.abs(lift-lift[:,0:1])))>2e-6:
            raise RuntimeError('Material changes changed interface lifting '+r)
        Uft=torch.as_tensor(Uref,dtype=torch.float32)
        out[r]=dict(d=d,ck=ck,model=model,basis=torch.as_tensor(d['basis'],dtype=torch.float32),
                    branches=torch.as_tensor((branches-d['branch_mean'])/d['branch_std'],dtype=torch.float32),
                    lift=lift, Uref=Uft, fit_ids=ids, coords=coords,
                    fitpack=oldtrain.array_pack(d,d['fit'],dev),
                    tunepack=oldtrain.array_pack(d,d['tune'],dev),
                    rawplan=base, source_ck_sha=sha(c.ck_path(root,'centered','grad',r)))
        print('INPUT CHECK',r,'FIT ids',ids.tolist(),'max coordinate mismatch mm',float(dist.max()),flush=True)
    return out,c,oldtrain

def model_U(model,z,basis,lift):
    """Frozen lifting plus trainable residual POD coefficients."""
    import torch
    z=z.reshape(-1,z.shape[-1]);coeff=model(z)
    U=lift.reshape(-1,lift.shape[-2],3)+torch.einsum('br,rnc->bnc',coeff,basis)
    return U.reshape(*lift.shape)

def gradient_loss(model,pack,centers,scales):
    import torch
    z=pack['branches'][centers];lift=pack['lift'][centers]
    pred=model_U(model,z,pack['basis'],lift)
    half=(pred[:,1::2]-pred[:,::2])/2.
    ref=(pack['Uref'][centers,1::2]-pack['Uref'][centers,::2])/2.
    # Separate each property normalization to avoid suppressing E2.
    err=(half-ref)/scales.reshape(1,3,1,1)
    return torch.mean(err*err)

def scales_of(pack):
    import torch
    ref=(pack['Uref'][:3,1::2]-pack['Uref'][:3,::2])/2.
    s=torch.sqrt(torch.mean(ref*ref,dim=(0,2,3)))
    return torch.clamp(s,min=2.e-6).detach()

def orig_objective(model,pack,oldtrain,subset):
    import torch
    d=pack['d'];ds=d['ds'];b=oldtrain.b
    con=torch.as_tensor(np.array(ds['element_node_indices'],copy=True),dtype=torch.long)
    dN=b.np_tens(b.AA.element_dN_numpy(ds),torch.device('cpu'))
    sU=b.np_tens(d['sU'],torch.device('cpu')).reshape(1,1,3)
    sG=b.np_tens(d['sG'],torch.device('cpu')).reshape(1,1,3,3)
    return oldtrain.objective(model,pack[subset],pack['basis'],con,dN,sU,sG,'B2',(0.2,0.1,0.1))[0]

def train(root,args):
    import torch
    import pandas as pd
    packs,c,oldtrain=context(root)
    out=root/'results'/RESULTS/args.tag
    if out.exists() and any(out.iterdir()):raise RuntimeError('Previous results protected. Use fresh --tag '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    rows=[];summ={}
    for r,pack in packs.items():
        original=pack['model'];model=copy.deepcopy(original)
        scaler=scales_of(pack)
        original.eval();model.eval()
        with torch.no_grad():
            base_fit=float(orig_objective(original,pack,oldtrain,'fitpack'))
            base_tune=float(orig_objective(original,pack,oldtrain,'tunepack'))
            base_trainJ=float(gradient_loss(original,pack,slice(0,3),scaler))
            base_devJ=float(gradient_loss(original,pack,slice(3,4),scaler))
        if min(base_fit,base_tune,base_trainJ,base_devJ)<0:raise RuntimeError('Invalid losses')
        lam=np.clip(args.lambda_ratio * max(base_fit,1e-8)/max(base_trainJ,1e-8),.0001,500.)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
        best=base_devJ;best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
        hist=[];stale=0
        for ep in range(1,args.epochs+1):
            model.train();opt.zero_grad(set_to_none=True)
            L0=orig_objective(model,pack,oldtrain,'fitpack')
            LJ=gradient_loss(model,pack,slice(0,3),scaler)
            L=L0+float(lam)*LJ
            if not torch.isfinite(L):raise RuntimeError('Nonfinite training loss')
            L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0);opt.step()
            model.eval()
            with torch.no_grad():
                devJ=float(gradient_loss(model,pack,slice(3,4),scaler))
                tune=float(orig_objective(model,pack,oldtrain,'tunepack'))
            accepted=devJ<best-1e-6 and tune<=base_tune*(1+args.max_tune_worsening)
            if accepted:
                best=devJ;best_epoch=ep;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            hist.append(dict(epoch=ep,fit_B2_objective=float(L0),FIT_gradient_loss=float(LJ),
                             DEV_gradient_loss=devJ,internal_TUNE_B2_objective=tune,
                             lambda_J=float(lam),candidate_accepted=bool(accepted)))
            if ep==1 or ep%20==0 or ep==args.epochs:
                print('U3',r,'epoch',ep,'fit',round(float(L0),5),'devJ',round(devJ,4),
                      'tune',round(tune,5),'best epoch',best_epoch,flush=True)
            if stale>=args.patience:break
        model.load_state_dict(best_state)
        regout=out/r;regout.mkdir(parents=True,exist_ok=True)
        ckfile=regout/'B9A11U_best_candidate.pt'
        torch.save(dict(stage='B9A11U',region=r,source_checkpoint_sha256=pack['source_ck_sha'],
             original_checkpoint_unchanged=True,FIT_cases_used=pack['fit_ids'][:3].tolist(),
             DEV_case=pack['fit_ids'][3:].tolist(),selection_uses_DEV_FIT_source_only=True,
             hyperparameters=dict(epochs=args.epochs,lr=args.lr,lambda_J=float(lam),
             max_tune_worsening=args.max_tune_worsening),
             selected_epoch=best_epoch,development_J_baseline=base_devJ,
             development_J_candidate=best,state_dict=best_state),ckfile)
        pd.DataFrame(hist).to_csv(regout/'B9A11U_training_history.csv',index=False)
        summ[r]=dict(B5_frozen_checkpoint_sha256=pack['source_ck_sha'],
            new_candidate_sha256=sha(ckfile),baseline_FIT_B2=base_fit,baseline_TUNE_B2=base_tune,
            baseline_TRAIN_gradient_loss=base_trainJ,baseline_DEV_gradient_loss=base_devJ,
            best_DEV_gradient_loss=best,selected_epoch=best_epoch,
            selected_new_candidate=best_epoch>0,scales_MM=scaler.numpy().tolist(),lambda_J=float(lam))
    summary=dict(stage='B9A11U-train',only_new_FIT_gradient_data=True,
                 B9A11T_not_used_for_checkpoint_selection=True,
                 known_FEM_q=True,not_FULL_SEVEN_REGION_COUPLED=True,regions=summ)
    (out/'B9A11U_training_summary.json').write_text(json.dumps(summary,indent=2))
    print('U3 TRAIN COMPLETE:',out)

def compare(root,args):
    import torch,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Uses verified original B9A11T exact FEM and old B5 prediction code.
    t=importlib.import_module('65AM_B9A11T_M_exact_FEM_vs_B5_REVISED')
    from types import SimpleNamespace
    ctxargs=SimpleNamespace(project=str(root),design_tag=args.design_tag)
    root,b6,b9,anchor,ref,md,info=t.context(ctxargs)
    packs,c,oldtrain=context(root)
    folder=root/'results'/RESULTS/args.tag
    train_summary=folder/'B9A11U_training_summary.json'
    if not train_summary.is_file():raise FileNotFoundError('Run U3 training first '+str(train_summary))
    out=folder/'validation_B9A11T'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Comparison protected; use a new training tag')
    rows=[];cases=[]
    for region,pk in info.items():
        model0=pk['model'];model1=copy.deepcopy(model0)
        ck=torch.load(folder/region/'B9A11U_best_candidate.pt',map_location='cpu',weights_only=False)
        if ck['source_checkpoint_sha256']!=packs[region]['source_ck_sha']:
            raise RuntimeError('Candidate checkpoint source hash mismatch '+region)
        model1.load_state_dict(ck['state_dict'],strict=True);model1.eval()
        FEM=ref['U_'+region][:,pk['indices'],:]
        for label,model in (('frozen_B5',model0),('sensitivity_B5',model1)):
            testpack=dict(pk);testpack['model']=model
            Up,_=t.predict_all(root,b6,b9,anchor,ref,region,testpack)
            for layout,ids in pk['layouts'].items():
                Ja=t.normalized_jacobian(FEM[:,ids,:],args.sigma_u)
                Jb=t.normalized_jacobian(Up[:,ids,:],args.sigma_u)
                svf=np.linalg.svd(Ja,compute_uv=False);svb=np.linalg.svd(Jb,compute_uv=False)
                total=float(100*np.linalg.norm(Jb-Ja)/max(np.linalg.norm(Ja),1e-14))
                for j,p in enumerate(PROPS):
                    er,cs=t.compare_vectors(Ja[:,j],Jb[:,j])
                    rows.append(dict(region=region,layout=layout,property=p,model=label,
                        jacobian_rel_error_pct=er,gradient_cosine=cs,
                        fem_importance=t.norm_shift(Ja[:,j],1),
                        model_importance=t.norm_shift(Jb[:,j],1),
                        whole_jacobian_error_pct=total,
                        fem_smallest_singular_value=svf[-1],model_smallest_singular_value=svb[-1],
                        fem_condition=svf[0]/svf[-1],model_condition=svb[0]/max(svb[-1],1e-12)))
                for state in range(7):
                    cases.append(dict(region=region,layout=layout,model=label,case=t.CASES[state],
                          displacement_RMSE_mm=float(np.sqrt(np.mean((FEM[state,ids]-Up[state,ids])**2)))))
    df=pd.DataFrame(rows);cases=pd.DataFrame(cases)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'B9A11U_B9A11T_exact_gradient_comparison.csv',index=False)
    cases.to_csv(out/'B9A11U_B9A11T_forward_comparison.csv',index=False)
    for metric,filename,ylabel in [('jacobian_rel_error_pct','B9A11U_exact_gradient_before_after.png','Matched FEM gradient error (%)'),
            ('gradient_cosine','B9A11U_gradient_cosine_before_after.png','Gradient cosine similarity')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,r in zip(axes,REGIONS):
            p=df[(df.region==r)&(df.layout=='optimized')].pivot(index='property',columns='model',values=metric).reindex(PROPS)
            x=np.arange(3);ax.bar(x-.2,p['frozen_B5'],.4,label='Original B5');ax.bar(x+.2,p['sensitivity_B5'],.4,label='Sensitivity-trained B5')
            ax.set_xticks(x);ax.set_xticklabels(PROPS);ax.set_title(r);ax.set_ylabel(ylabel);ax.legend()
        fig.suptitle('Reserved B9A11T exact FEM | optimized sensors | unchanged interface q')
        fig.tight_layout();fig.savefig(out/filename,dpi=175);plt.close(fig)
    summary=dict(stage='B9A11U-compare',preserved_B9A11T_for_holdout=True,
        full_beam_online_re_equilibration=False,source_data_source='old fixed-q B9A11T',
        model_training_uses_only_new_FIT_sensitivity_perturbations=True,
        grouped=df.groupby(['layout','model'])[['jacobian_rel_error_pct','whole_jacobian_error_pct']].mean().reset_index().to_dict('records'))
    (out/'B9A11U_validation_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11U EXACT FEM VALIDATION COMPLETE:',out)
    print(df[df.layout=='optimized'][['region','model','property','jacobian_rel_error_pct','gradient_cosine']].to_string(index=False))

def inverse(root,args):
    """Diagnostic fixed-65V-q inverse, comparing original and U-trained B5.

    Uses seven B9A11T material states and *identical* noise per model.
    Does NOT use true parameters as an optimizer starting point.
    """
    import torch, pandas as pd
    from scipy.optimize import least_squares
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    t=importlib.import_module('65AM_B9A11T_M_exact_FEM_vs_B5_REVISED')
    root,b6,b9,anchor,data,md,info=t.context(SimpleNamespace(project=str(root),design_tag=args.design_tag))
    packs,c,oldtrain=context(root)
    folder=root/'results'/RESULTS/args.tag
    if not (folder/'B9A11U_training_summary.json').is_file():raise FileNotFoundError('Complete U3 train first')
    out=folder/'inverse_B9A11T'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Results protected: '+str(out))
    rows=[]
    for ri,(region,pk) in enumerate(info.items()):
        d=pk['d'];ds=d['ds'];ds['_b9_region']=region
        context=b6.b.CO.interface_context(root,ds,torch.device('cpu'))
        branch,lift=b9.build_branch_lift(root,ds,anchor,context,torch.device('cpu'))
        models={'original_B5':pk['model']}
        upgraded=copy.deepcopy(pk['model'])
        ck=torch.load(folder/region/'B9A11U_best_candidate.pt',map_location='cpu',weights_only=False)
        if ck['source_checkpoint_sha256']!=packs[region]['source_ck_sha']:
            raise RuntimeError('Candidate provenance mismatch')
        upgraded.load_state_dict(ck['state_dict']);upgraded.eval();models['trained_B5']=upgraded
        fitm=np.asarray(ds['branch_inputs'][d['fit'],:3],float)
        lo=fitm.min(axis=0);hi=fitm.max(axis=0)
        mid=(lo+hi)/2.;half=(hi-lo)/2.
        ids=np.asarray(pk['layouts']['optimized'],int)
        truth=np.asarray(data['material_MPa'],float)
        Utruth=np.asarray(data['U_'+region][:,pk['indices'],:])
        def mk_predict(model,z):
            phys=mid+half*np.asarray(z,float)
            b=branch.copy();b[0,:3]=phys
            return b6.predict_b5(d,model,b,lift,torch.device('cpu'))[0][ids,:].reshape(-1)
        for k,case in enumerate(t.CASES):
            yclean=Utruth[k,ids,:].reshape(-1)
            for rep in range(-1,args.noise_reps):
                rng=np.random.default_rng(args.seed+100000*ri+100*k+max(rep,0))
                noise=np.zeros_like(yclean) if rep<0 else rng.normal(0,args.sigma_u,size=yclean.size)
                y=yclean+noise
                for label,model in models.items():
                    def residual(z):
                        dz=(mk_predict(model,z)-y)/args.sigma_u
                        # Weak, identical Tikhonov regularizer for both forward models.
                        return np.r_[dz,0.03*np.asarray(z)]
                    result=least_squares(residual,np.zeros(3),bounds=(-np.ones(3),np.ones(3)),
                                         max_nfev=90,ftol=1.e-8,xtol=1.e-8,gtol=1.e-8)
                    est=mid+half*result.x
                    row=dict(region=region,case=case,model=label,
                             mode='zero_noise' if rep<0 else 'noisy',rep=rep,
                             optimizer_success=bool(result.success),fit_residual_RMS=float(np.sqrt(np.mean(result.fun[:len(y)]**2))),
                             boundary_hit=bool(np.any(abs(result.x)>.99)))
                    for j,prop in enumerate(PROPS):
                        row[prop+'_true_MPa']=float(truth[k,j]);row[prop+'_estimated_MPa']=float(est[j])
                        row[prop+'_abs_rel_pct']=float(100*abs(est[j]-truth[k,j])/truth[k,j])
                    rows.append(row)
            print('U4 INVERSE',region,case,flush=True)
    df=pd.DataFrame(rows);out.mkdir(parents=True)
    df.to_csv(out/'B9A11U_inverse_recovery.csv',index=False)
    med=df[df['mode']=='noisy'].groupby('model')[[p+'_abs_rel_pct' for p in PROPS]].median()
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(3)
    for dx,label in ((-.2,'original_B5'),(.2,'trained_B5')):
        ax.bar(x+dx,med.loc[label].to_numpy(),width=.4,label=label)
    ax.set_xticks(x);ax.set_xticklabels(PROPS)
    ax.set_ylabel('Median noisy material recovery error (%)')
    ax.set_title('B9A11U | reserved B9A11T known-q inverse comparison')
    ax.legend();fig.tight_layout();fig.savefig(out/'B9A11U_inverse_error_before_after.png',dpi=180);plt.close(fig)
    summ=dict(stage='B9A11U-conditional_inverse',fixed_FEM_interface_q=True,
              B9A11T_was_previously_inspected_not_blinded_external_validation=True,
              compared_models_with_identical_observations=True,
              median_noisy_errors_pct=med.to_dict('index'),
              boundary_hit_counts=df.groupby(['model','mode'])['boundary_hit'].sum().astype(int).to_dict())
    # Tuple keys aren't JSON; convert them explicitly.
    summ['boundary_hit_counts']={str(k):int(v) for k,v in summ['boundary_hit_counts'].items()}
    (out/'B9A11U_inverse_summary.json').write_text(json.dumps(summ,indent=2))
    print('B9A11U INVERSE DONE:',out,'\n',med.to_string())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.')
    ap.add_argument('--stage',required=True,choices=('selftest','check','train','compare','inverse'))
    ap.add_argument('--tag',default='pilot01');ap.add_argument('--design-tag',default='pilot01')
    ap.add_argument('--epochs',type=int,default=160);ap.add_argument('--patience',type=int,default=55)
    ap.add_argument('--lr',type=float,default=5e-5);ap.add_argument('--lambda-ratio',type=float,default=.2)
    ap.add_argument('--max-tune-worsening',type=float,default=.10)
    ap.add_argument('--sigma-u',type=float,default=.0002);ap.add_argument('--noise-reps',type=int,default=2);ap.add_argument('--seed',type=int,default=651102)
    a=ap.parse_args()
    if a.stage=='selftest':return test_math()
    if not (a.sigma_u>0 and 1<=a.epochs<=500 and a.lr>0 and a.lambda_ratio>=0 and a.patience>=1):
        raise ValueError('Invalid optimization settings')
    root=Path(a.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    if a.stage=='check':context(root);print('B9A11U-C CHECK PASS, no training');return
    if a.stage=='train':train(root,a);return
    if a.stage=='inverse':inverse(root,a);return
    compare(root,a)
if __name__=='__main__':main()
