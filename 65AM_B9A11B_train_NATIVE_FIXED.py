#!/usr/bin/env python3
"""B9A11-B: PHYSICS-INFORMED RESIDUAL POD ADAPTER ABOVE FROZEN B5.

Safe training trial; never overwrites original B5 / Broyden / 65V.
Uses 64J original fit cases for adapter training and subset-of-fit model selection.
The original tune 12-case cohort is reported ONLY at the end, and is NOT independent.

Training compares equally sized adapters:
 control    = supervised coefficient/displacement correction only
 physics    = same correction + a floor-aware equilibrium penalty, linearized
              about the original B5 predictions via PyTorch JVPs.
The final report validates both with the EXACT nonlinear L+K force operator.
This is NOT full B5 weight fine-tuning. It is a parameter-efficient adapter.
"""
import os, sys, csv, json, math, time, hashlib, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_OUT=Path('results/65AM_B9_coupled_feasibility/B9A11B_adapter_finetuning')
REGIONS=('inner_left','inner_right')

def imp(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(sp)
    if sp is None or sp.loader is None:raise ImportError(str(path))
    sp.loader.exec_module(m);return m

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def jserial(x):
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x)
    if isinstance(x,(np.ndarray,)):return x.tolist()
    if isinstance(x,(torch.Tensor,)):return x.detach().cpu().tolist()
    raise TypeError('Not JSON serializable '+str(type(x)))

def csvout(p,rows):
    if not rows:raise RuntimeError('No rows '+str(p))
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def vfloat(x):return float(x.detach().cpu().item() if torch.is_tensor(x) else x)

def vec(x):return torch.linalg.vector_norm(x.reshape(-1))

def get_dep(root):
    a=root/'65AM_B9A11A_readiness_FIXED.py'
    o=root/'65AM_B9A11B_objectives.py'
    if not a.is_file() or not o.is_file():raise FileNotFoundError('Require B9A11A_readiness_FIXED.py and B9A11B_objectives.py in project root')
    A=imp(a,'b9a11a_existing_for_b')
    O=imp(o,'b9a11b_obj')
    return A,O

def verify(root, args):
    A,O=get_dep(root)
    c,H,state,beta,anchor,source=A.check_source(root,args)
    # Require expected frozen baseline and no orphaned/mismatched case arrays.
    a_sum=root/'results/65AM_B9_coupled_feasibility/B9A11A_B5_readiness/B9A11A_summary.json'
    if not a_sum.is_file():raise FileNotFoundError('Run B9A11A audit before B9A11B: '+str(a_sum))
    aj=json.loads(a_sum.read_text())
    if aj.get('no_training') is not True or not math.isclose(beta,float(aj['beta']),rel_tol=1e-12):
        raise RuntimeError('B9A11A provenance / beta mismatch')
    for region in REGIONS:
        d=state[region]['d']
        fit=np.asarray(d['fit'],dtype=int);tune=np.asarray(d['tune'],dtype=int)
        if len(fit)<8 or len(tune)!=12 or len(set(fit)&set(tune)):
            raise RuntimeError(f'Invalid fit / tune split for {region}: fit={len(fit)}, tune={len(tune)}')
        if len(state[region]['models'])!=1:
            raise RuntimeError('B9A11B requires ONE baseline model per region')
        # REAL B5 forward-pass smoke test, not merely a checkpoint/shape check.
        model,ckpt=next(iter(state[region]['models'].values()))
        up,coeff=predict_frozen_b5_native(c,model,d,fit[:1])
        assert up.shape == (1,len(d['ds']['coordinates']),3) and coeff.shape[0] == 1
        print('B9A11B FORWARD PASS',region,'FP32 B5 -> FP64 predictions:',up.shape,coeff.shape,flush=True)
        print('B9A11B CHECK PASS',region,'fit',len(fit),'internal tune',len(tune),'baseline',list(state[region]['models'])[0],flush=True)
    return A,O,c,H,state,beta,source

def feature_statistics(samples,train_idx):
    # Uses only the chosen adapter-train cases, never validation or internal tune.
    ft=torch.stack([s['raw_features'] for s in samples])
    tr=ft[train_idx]
    mean=tr.mean(0)
    std=tr.std(0,unbiased=False).clamp_min(1.e-7)
    # Robust coefficient scale: a zero-variance mode cannot cause division by zero.
    coeff=torch.stack([s['coeff0'] for s in samples])[train_idx]
    coeff_scale=coeff.std(0,unbiased=False).clamp_min(1.e-4)
    return mean,std,coeff_scale

def predict_frozen_b5_native(c, net, d, ids):
    """Use the ORIGINAL 65AM_B5_common.predict, not a reimplementation.

    B5's trained coefficient net and tensors are FP32. It is not safe to load
    the model while torch's global default is FP64: nn.Linear would be Double
    while B5's explicit input tensors / coefficient buffers remain Float.
    The frozen predictions are converted to FP64 only AFTER native inference.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        raise ValueError('B5 prediction requires at least one case')
    incorrect = [(name, str(param.dtype)) for name,param in net.named_parameters()
                 if param.is_floating_point() and param.dtype != torch.float32]
    incorrect += [(name, str(buf.dtype)) for name,buf in net.named_buffers()
                  if buf.is_floating_point() and buf.dtype != torch.float32]
    if incorrect:
        raise RuntimeError('B5 model loaded with mixed/non-FP32 dtypes: '+str(incorrect[:8])+

                           '. Ensure torch.set_default_dtype(torch.float32) BEFORE A.check_source loads B5.')
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('Default dtype must be float32 at B5 prediction boundary')
    with torch.no_grad():
        # Identical to the B5 inference already used successfully by B9A11-A.
        up,coeff = c.predict(net,d,ids,torch.device('cpu'))
    up=np.asarray(up,dtype=np.float64)
    coeff=np.asarray(coeff,dtype=np.float64)
    if (up.ndim != 3 or up.shape[0] != len(ids) or up.shape[-1] != 3 or
            coeff.ndim != 2 or coeff.shape[0] != len(ids) or
            not np.isfinite(up).all() or not np.isfinite(coeff).all()):
        raise RuntimeError('Unexpected frozen B5 native prediction shapes/values')
    return up,coeff

def assemble_samples(c,state,region,ids):
    d=state['d']; ds=d['ds'];mask=state['boundary']; force=state['force']
    modelname,(net,ck)=next(iter(state['models'].items()))
    up, coeff=predict_frozen_b5_native(c,net,d,ids)
    up=np.asarray(up,dtype=np.float64)
    coeff=np.asarray(coeff,dtype=np.float64)
    basis=torch.as_tensor(np.asarray(d['basis'],dtype=np.float64),dtype=torch.float64)
    if coeff.ndim!=2 or basis.shape[0]!=coeff.shape[1] or up.shape[0]!=len(ids):
        raise RuntimeError('B5 POD prediction / basis mismatch')
    x=np.asarray(ds['branch_inputs'],dtype=np.float64)
    # lift indexed by concatenated original fit+tune arrays per B9A11A.
    lookup={int(k):i for i,k in enumerate(np.r_[d['fit'],d['tune']])}
    ans=[]
    for k,cid in enumerate(ids):
        cid=int(cid)
        if cid not in lookup:raise RuntimeError('Missing B5 lift for '+str(cid))
        mat=torch.as_tensor(x[cid,:3],dtype=torch.float64)
        lift=torch.as_tensor(np.asarray(d['lift'][lookup[cid]],dtype=np.float64),dtype=torch.float64)
        cref=torch.as_tensor(np.asarray(ds['U'][cid],dtype=np.float64),dtype=torch.float64)
        cp=torch.as_tensor(coeff[k],dtype=torch.float64)
        u0=lift+torch.einsum('r,rnc->nc',cp,basis)
        maxgap=vfloat(torch.abs(u0-torch.as_tensor(up[k],dtype=torch.float64)).max())
        if maxgap>3.e-6:raise RuntimeError('Original B5 reconstruction mismatch; STOP %.4g'%maxgap)
        if vfloat(torch.abs(u0[mask]-cref[mask]).max())>1.e-5:raise RuntimeError('Dirichlet interface mismatch')
        with torch.no_grad():
            f0=force(u0,mat);fr=force(cref,mat)
        den=vec(fr[mask]).clamp_min(1.e-9)
        floor=vec(fr[~mask])/den
        raw=torch.cat([torch.as_tensor(x[cid],dtype=torch.float64),cp])
        ans.append(dict(id=cid,mat=mat,coeff0=cp,lift=lift,ref=cref,u0=u0,f0=f0,
                        fr=fr,den=den,floor=floor,raw_features=raw,
                        baseline_force_err=float(100*vec(f0[~mask])/den),
                        baseline_disp_pct=float(100*vec((u0-cref)[~mask])/vec(cref[~mask]).clamp_min(1e-12))))
    return ans,basis,mask,modelname

def make_linearized_tangents(samples,basis,mask,force):
    rank=basis.shape[0]
    for t,s in enumerate(samples):
        mat=s['mat'];lift=s['lift'];cp=s['coeff0']
        def fn(p):
            u=lift+torch.einsum('r,rnc->nc',p,basis)
            return force(u,mat)[~mask].reshape(-1)
        # JVP gives a local, differentiable POD tangent without an explicit
        # full 3024x3024 stiffness matrix.
        cols=[]
        for j in range(rank):
            direction=torch.zeros_like(cp);direction[j]=1.
            _,dj=torch.autograd.functional.jvp(fn,cp,direction,create_graph=False,strict=True)
            cols.append(dj.detach())
        s['jac']=torch.stack(cols,dim=-1)
        if not torch.isfinite(s['jac']).all():raise RuntimeError('Nonfinite physical POD tangent')
        if t%10==0:print('B9A11B POD FORCE TANGENT',t+1,'/',len(samples),flush=True)

def get_u(s,adapter,features,coeff_std,basis,mask,trust_pct):
    delta=adapter(features,coeff_std)
    du=torch.einsum('r,rnc->nc',delta,basis)
    den=vec(s['u0'][~mask]).detach().clamp_min(1e-12)
    relative=vec(du[~mask])/den
    scale=torch.clamp((trust_pct/100.)/relative.clamp_min(1e-15),max=1.)
    delta=delta*scale
    u=s['u0']+torch.einsum('r,rnc->nc',delta,basis)
    return u,delta

def objectives(s,adapter,ft,cs,basis,mask,Obj,lambda_eq,trust_pct,margin):
    u,delta=get_u(s,adapter,ft,cs,basis,mask,trust_pct)
    data=Obj.normalized_data_loss(u,s['ref'],~mask)
    # Use the local JVP linearization during TRAINING only.
    f_linear=s['f0'][~mask].reshape(-1)+s['jac']@delta
    eq=Obj.floor_aware_equilibrium(f_linear,s['den'],s['floor'],margin)
    return data+lambda_eq*eq, data,eq

def calibrate_lambda(train_samples,basis,mask,Obj,margin,target_fraction):
    ratios=[]
    for s in train_samples:
        p=s['coeff0'].clone().requires_grad_(True)
        u=s['lift']+torch.einsum('r,rnc->nc',p,basis)
        loss_d=Obj.normalized_data_loss(u,s['ref'],~mask)
        fp=s['f0'][~mask].reshape(-1)+s['jac']@(p-s['coeff0'])
        loss_e=Obj.floor_aware_equilibrium(fp,s['den'],s['floor'],margin)
        gd=torch.autograd.grad(loss_d,p,retain_graph=True)[0]
        ge=torch.autograd.grad(loss_e,p)[0]
        a,b,_=Obj.gradient_scale(gd,ge,1.)
        if a>1e-20 and b>1e-20:ratios.append(b/a)
    if not ratios:raise RuntimeError('Cannot calibrate physics loss: no usable gradients')
    ratio=float(np.median(ratios))
    lam=float(np.clip(target_fraction/ratio,1e-8,1e-2))
    return lam,ratio

def train_arm(samples,train_idx,val_idx,feat_mean,feat_std,cs,basis,mask,Obj,lam,epochs,lr,trust_pct,margin,seed):
    torch.manual_seed(seed)
    mdl=Obj.PODResidualAdapter(len(feat_mean),basis.shape[0],hidden=24).double()
    optim=torch.optim.Adam(mdl.parameters(),lr=lr)
    features=[(s['raw_features']-feat_mean)/feat_std for s in samples]
    history=[]; best=float('inf');best_sd=None;best_ep=0
    # Avoid selecting on the 12 original internal tune cases.
    for epoch in range(1,epochs+1):
        mdl.train();perm=np.random.default_rng(seed+epoch).permutation(train_idx)
        for i in perm:
            optim.zero_grad(set_to_none=True)
            total,dat,phy=objectives(samples[i],mdl,features[i],cs,basis,mask,Obj,lam,trust_pct,margin)
            if not torch.isfinite(total):raise RuntimeError('Training generated nonfinite loss')
            total.backward();torch.nn.utils.clip_grad_norm_(mdl.parameters(),max_norm=1.)
            optim.step()
        mdl.eval()
        with torch.no_grad():
            vals=[objectives(samples[i],mdl,features[i],cs,basis,mask,Obj,lam,trust_pct,margin) for i in val_idx]
            means=[float(torch.stack([z[k] for z in vals]).mean()) for k in range(3)]
        history.append(dict(epoch=epoch,val_objective=means[0],val_data=means[1],val_equilibrium=means[2],lambda_eq=lam))
        if means[0]<best-1e-15:
            best=means[0];best_ep=epoch
            best_sd={k:v.detach().clone() for k,v in mdl.state_dict().items()}
    if best_sd is None:raise RuntimeError('No valid checkpoint')
    mdl.load_state_dict(best_sd)
    return mdl,history,best_ep

def evaluate_exact(samples,adapter,mean,std,cs,basis,mask,force,Obj,trust_pct,margin,region,arm,cohort):
    rows=[]
    for s in samples:
        ft=(s['raw_features']-mean)/std
        with torch.no_grad():
            u,delta=get_u(s,adapter,ft,cs,basis,mask,trust_pct)
            f=force(u,s['mat'])
            flinear=s['f0'][~mask].reshape(-1)+s['jac']@delta if 'jac' in s else None
            fdiff=vec(f[~mask]-s['fr'][~mask])
            exact=vfloat(100*vec(f[~mask])/s['den'])
            floor=vfloat(100*s['floor'])
            disp=vfloat(100*vec((u-s['ref'])[~mask])/vec(s['ref'][~mask]).clamp_min(1e-12))
            interf=vfloat(100*vec((f-s['fr'])[mask])/s['den'])
            linear_err=(vfloat(vec(f[~mask].reshape(-1)-flinear)/vec(f[~mask]).clamp_min(1e-10)) if flinear is not None else math.nan)
            rows.append(dict(region=region,cohort=cohort,arm=arm,case_id=s['id'],
                             displacement_pct=disp,free_residual_pct=exact,
                             free_reference_floor_pct=floor,additional_free_force_pct=vfloat(100*fdiff/s['den']),
                             reference_interface_force_difference_pct=interf,
                             pod_coeff_delta_norm=vfloat(vec(delta)),linearization_rel_error=linear_err))
    return rows

def blank_adapter(mean,basis,Obj):return Obj.PODResidualAdapter(len(mean),basis.shape[0],hidden=24).double()

def inspect_result(root,args):
    out=root/ROOT_OUT/args.run_tag
    sm=out/'B9A11B_summary.json'
    if not sm.is_file():raise FileNotFoundError('Run train first: '+str(sm))
    print(sm.read_text())

def run(root,args):
    t0=time.time();torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    # Original B5 is created in FP32. All FE/POD tensors are explicitly FP64.
    A,Obj,c,H,checked,beta,source=verify(root,args)
    out=root/ROOT_OUT/args.run_tag
    if out.exists():raise RuntimeError('Output already exists (possibly from failed trial). Keep it unchanged and choose a NEW --run-tag, e.g. trial02: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    report=[];hist=[];mode_records=[];summ={'step':'B9A11B','run_tag':args.run_tag,'type':'frozen B5 + trainable POD residual adapter (NOT backbone training)',
                        'independent_validation':False,'validation_scope':'original 64J internal tune 12 cases; NOT unseen',
                        'reference_FEM_force_used_for_training_normalization_and_floor_only_not_as_traction_target':True,'physical_operator':'B9A10L Cauchy + frozen K hourglass',
                        'beta_frozen':beta,'source_sha256':source,'baseline_checkpoint_modified':False,'results':{}}
    for region in REGIONS:
        st=checked[region];d=st['d'];force=st['force'];ids=np.asarray(d['fit'],dtype=int)
        rng=np.random.default_rng(args.seed + (1 if region=='inner_left' else 2))
        perm=rng.permutation(len(ids));nval=max(2,int(np.ceil(len(ids)*.2)))
        vidx=perm[:nval];tidx=perm[nval:]
        all_fit,basis,mask,model=assemble_samples(c,st,region,ids)
        mean,std,cs=feature_statistics(all_fit,tidx)
        # Physics Jacobians only for original fit cases, no 12-case tuning leakage.
        make_linearized_tangents(all_fit,basis,mask,force)
        lam,ratio=calibrate_lambda([all_fit[i] for i in tidx],basis,mask,Obj,args.floor_margin,args.target_grad_fraction)
        bestmodels={};epochs_sel={}
        for arm,weight in (('supervised',0.),('physics',lam)):
            model0,hh,bestep=train_arm(all_fit,tidx,vidx,mean,std,cs,basis,mask,Obj,weight,
                                args.epochs,args.lr,args.max_correction_pct,args.floor_margin,args.seed)
            bestmodels[arm]=model0;epochs_sel[arm]=bestep
            for row in hh:hist.append(dict(region=region,arm=arm,**row))
            # Adapter-only checkpoint, not a B5 checkpoint. Save to new output folder.
            ck={'kind':'B9A11B POD adapter','region':region,'baseline_model':model,
                'state_dict':{k:v.detach().cpu() for k,v in model0.state_dict().items()},
                'feature_mean':mean,'feature_std':std,'coefficient_std':cs,
                'beta':beta,'lambda_eq':weight,'trust_pct':args.max_correction_pct,
                'best_epoch':bestep,'rank':int(basis.shape[0]),'fit_train_ids':ids[tidx].tolist(),
                'fit_val_ids':ids[vidx].tolist(),'source_sha256':source}
            torch.save(ck,out/(region+'_'+arm+'_adapter.pt'))
        # New predictions on pre-existing internal tune cases only AFTER arm selection.
        tuning,basis_t,mask_t,model_t=assemble_samples(c,st,region,d['tune'])
        if model_t!=model or not torch.allclose(basis,basis_t) or not torch.equal(mask,mask_t):
            raise RuntimeError('Basis/architecture changed between training and internal tune')
        for cohort,cases in (('fit_val', [all_fit[i] for i in vidx]),('internal_tune',tuning)):
            for arm,m in [('B5_baseline',blank_adapter(mean,basis,Obj)),*bestmodels.items()]:
                report+=evaluate_exact(cases,m,mean,std,cs,basis,mask,force,Obj,args.max_correction_pct,
                                       args.floor_margin,region,arm,cohort)
        # Independent from the training objectives: original 65V PCA force-mode diagnostic.
        with np.load(root/'B9_coupled_anchor/B9_65V_anchor.npz',allow_pickle=False) as anc:
            values={'X':np.asarray(d['ds']['coordinates'],dtype=float),
                    'node_labels':np.arange(1,len(d['ds']['coordinates'])+1,dtype=np.int64)}
            for s in tuning:
                for arm,m in [('B5_baseline',blank_adapter(mean,basis,Obj)),*bestmodels.items()]:
                    feat=(s['raw_features']-mean)/std
                    with torch.no_grad():
                        u,_=get_u(s,m,feat,cs,basis,mask,args.max_correction_pct)
                        ff=force(u,s['mat']).numpy()
                    for face in H.FACES['L' if region=='inner_left' else 'R']:
                        pred,scales=H.pca_force(values,ff,anc,face)
                        ref,_=H.pca_force(values,s['fr'].detach().numpy(),anc,face)
                        for index,(v,refv,scale) in enumerate(zip(pred,ref,scales),start=1):
                            mode_records.append(dict(region=region,cohort='internal_tune',arm=arm,
                                case_id=s['id'],interface=face,mode=index,
                                normalized_pca_force_difference=abs(float(v)-float(refv))/float(scale)))
        rr=[r for r in report if r['region']==region and r['cohort']=='internal_tune']
        vals={}
        for arm in ('B5_baseline','supervised','physics'):
            sel=[r for r in rr if r['arm']==arm]
            vals[arm]={k:float(np.mean([x[k] for x in sel])) for k in
                   ('displacement_pct','free_residual_pct','additional_free_force_pct',
                    'reference_interface_force_difference_pct')}
        base,sup,phys=vals['B5_baseline'],vals['supervised'],vals['physics']
        # Physics must improve relative to same-capacity supervised adapter;
        # displacement cannot degrade >5% vs B5 or >5% vs supervised baseline.
        linerrs=[r['linearization_rel_error'] for r in report if r['region']==region and r['cohort']=='fit_val' and r['arm']=='physics']
        worst_linearization=float(np.nanmax(linerrs))
        gate=(worst_linearization<=.05 and phys['free_residual_pct']<=.90*sup['free_residual_pct'] and
              phys['displacement_pct']<=1.05*min(base['displacement_pct'],sup['displacement_pct']) and
              phys['reference_interface_force_difference_pct']<=1.20*base['reference_interface_force_difference_pct'])
        summ['results'][region]=dict(baseline_model=model,fit_count=int(len(ids)),
                train_count=int(len(tidx)),val_count=int(len(vidx)),tune_count=int(len(tuning)),
                lambda_equilibrium=lam,median_physics_to_data_grad_ratio=ratio,
                early_stopping_epoch=epochs_sel,means_internal_tune=vals,
                max_fit_val_linearization_relative_error=worst_linearization,
                internal_tune_gate_pass=bool(gate),
                caution='These original internal tuning cases are NOT independent validation')
        print('B9A11B REGION',region,'lambda',lam,'grad_ratio',ratio,'B5',base,'supervised',sup,'physics',phys,
              'screen_gate',gate,flush=True)
    csvout(out/'B9A11B_cases.csv',report)
    csvout(out/'B9A11B_training.csv',hist)
    csvout(out/'B9A11B_modes.csv',mode_records)
    summ['physics_adapter_internal_screening_pass']=bool(all(x['internal_tune_gate_pass'] for x in summ['results'].values()))
    summ['approved_for_coupled_training']=False
    summ['approved_for_independent_generalization']=False
    summ['elapsed_seconds']=time.time()-t0
    (out/'B9A11B_summary.json').write_text(json.dumps(summ,indent=2,default=jserial))
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for k,region in enumerate(REGIONS):
        rr=[r for r in report if r['region']==region and r['cohort']=='internal_tune']
        labels=('B5_baseline','supervised','physics')
        ax[k].bar(np.arange(3),[np.mean([r['free_residual_pct'] for r in rr if r['arm']==arm]) for arm in labels])
        ax[k].set_xticks(range(3),labels,rotation=15)
        ax[k].set_title(region);ax[k].set_ylabel('Mean free equilibrium force (% of reference interface norm)')
    fig.suptitle('B9A11B | Internal tune ONLY, not independent')
    fig.tight_layout();fig.savefig(out/'B9A11B_equilibrium.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for k,region in enumerate(REGIONS):
        rr=[r for r in report if r['region']==region and r['cohort']=='internal_tune']
        labels=('B5_baseline','supervised','physics')
        ax[k].bar(np.arange(3),[np.mean([r['displacement_pct'] for r in rr if r['arm']==arm]) for arm in labels])
        ax[k].set_xticks(range(3),labels,rotation=15);ax[k].set_title(region)
        ax[k].set_ylabel('Mean displacement relative error (%)')
    fig.suptitle('B9A11B | Does equilibrium loss preserve displacement?')
    fig.tight_layout();fig.savefig(out/'B9A11B_displacement.png',dpi=140);plt.close(fig)
    print('B9A11B COMPLETE | output:',out,'| NO B5 CHECKPOINTS MODIFIED')

def selftest(root):
    # Work under the exact same dtype regime as training, not a different one.
    torch.set_default_dtype(torch.float32)
    A,Obj=get_dep(root)
    Obj.selftest()  # older utility sets FP64 default; reset immediately.
    torch.set_default_dtype(torch.float32)
    # Model mirrors the FLOAT32 B5 forward conventions: FP32 Linear + buffers.
    class TinyB5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core=torch.nn.Linear(12,2)
            self.register_buffer('coeff_mean',torch.zeros(2,dtype=torch.float32))
            self.register_buffer('coeff_scale',torch.ones(2,dtype=torch.float32))
        def forward(self,x):
            return self.coeff_mean[None,:]+self.coeff_scale[None,:]*self.core(x)
    net=TinyB5().eval()
    rng=np.random.default_rng(10)
    ds={'branch_inputs':rng.normal(size=(3,12)).astype(np.float32)}
    d={'fit':np.array([0,1]),'tune':np.array([2]),'ds':ds,
       'branch_mean':np.zeros(12,dtype=np.float32),
       'branch_std':np.ones(12,dtype=np.float32),
       'basis':rng.normal(size=(2,8,3)).astype(np.float32)*.001,
       'lift':np.zeros((3,8,3),dtype=np.float32)}
    class NativeB5:
        @staticmethod
        def predict(m,data,ids,dev):
            ids=np.asarray(ids,dtype=int)
            x=(data['ds']['branch_inputs'][ids]-data['branch_mean'])/data['branch_std']
            cp=m(torch.as_tensor(x,dtype=torch.float32))
            base=torch.as_tensor(data['basis'],dtype=torch.float32)
            lift=torch.as_tensor(data['lift'][ids],dtype=torch.float32)
            u=lift+torch.einsum('br,rnc->bnc',cp,base)
            return u.detach().numpy(),cp.detach().numpy()
    ub,cb=predict_frozen_b5_native(NativeB5,net,d,np.array([0,2]))
    assert ub.dtype==np.float64 and cb.dtype==np.float64
    assert ub.shape==(2,8,3) and cb.shape==(2,2) and np.isfinite(ub).all()
    # Verify that the smoke test catches the precise reported failure.
    bad=TinyB5().double()
    try:
        predict_frozen_b5_native(NativeB5,bad,d,np.array([0]))
    except RuntimeError as exc:
        assert 'non-FP32' in str(exc)
    else:
        raise AssertionError('Mixed precision regression test failed to catch error')
    torch.manual_seed(1)
    basis=torch.randn(3,8,3,dtype=torch.float64)*.001
    cp=torch.randn(3,dtype=torch.float64)
    model=Obj.PODResidualAdapter(15,3).double()
    feature=torch.randn(15,dtype=torch.float64)
    scale=torch.tensor([.1,.2,.3],dtype=torch.float64)
    assert torch.allclose(model(feature,scale),torch.zeros(3,dtype=torch.float64))
    def spring(p):return (torch.einsum('r,rnc->nc',p,basis)*1000).flatten()
    cols=[]
    for i in range(3):
        _,j=torch.autograd.functional.jvp(spring,cp,torch.eye(3,dtype=torch.float64)[i],strict=True)
        cols.append(j)
    J=torch.stack(cols,dim=-1)
    guess=torch.randn(3,dtype=torch.float64)*1e-3
    assert torch.allclose(spring(cp+guess),spring(cp)+J@guess,rtol=1e-10,atol=1e-10)
    with torch.no_grad():model.net[-1].bias.copy_(torch.tensor([.05,-.03,.02],dtype=torch.float64))
    v=model(feature,scale)
    loss=(spring(cp+v)-spring(cp)).square().sum()
    g=torch.autograd.grad(loss,model.parameters(),allow_unused=True)
    assert any(x is not None and torch.isfinite(x).all() for x in g)
    print('B9A11B SELFTEST PASS | native FP32 B5 -> FP64 force | JVP | autograd')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('selftest','check','train','report'),required=True)
    p.add_argument('--run-tag',default='trial01')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=27)
    p.add_argument('--floor-margin',type=float,default=1.25)
    p.add_argument('--target-grad-fraction',type=float,default=.20)
    p.add_argument('--max-correction-pct',type=float,default=.10)
    p.add_argument('--left',default='standardized_grad')
    p.add_argument('--right',default='centered_grad')
    p.add_argument('--all-variants',action='store_false',dest='all_variants',default=False)
    args=p.parse_args()
    if not args.run_tag.replace('_','').replace('-','').isalnum():raise ValueError('run-tag must be letters/numbers/_/-')
    if args.epochs<1 or args.epochs>500:raise ValueError('epochs must be 1..500')
    if not (0<args.max_correction_pct<=.5 and 1<=args.floor_margin<=3):raise ValueError('Invalid correction/floor settings')
    root=Path(args.project).expanduser().resolve()
    # Must be set BEFORE importing/loading original 65AM_B5_common.
    torch.set_default_dtype(torch.float32)
    if args.stage=='selftest':selftest(root)
    elif args.stage=='check':verify(root,args)
    elif args.stage=='train':run(root,args)
    else:inspect_result(root,args)

if __name__=='__main__':main()
