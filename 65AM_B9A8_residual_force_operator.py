#!/usr/bin/env python3
"""65AM-B9A8: Residual-corrected ORIGINAL 65V force head (Mac only).

Needs existing 65AM_B9A7_force_head.py unchanged, original Mac 64Q force-head
exports (or copied 65V Windows no_models_7region), force PCA mappings,
64J datasets, B9A7 checkpoints, saved B9 anchor, B9A3 true Abaqus reference.

Stages:
 python3 65AM_B9A8_residual_force_operator.py --stage selftest
 python3 65AM_B9A8_residual_force_operator.py --stage check --project .
 python3 65AM_B9A8_residual_force_operator.py --stage train --project .
 python3 65AM_B9A8_residual_force_operator.py --stage anchor --project .

Only fits train split; selects by validation; evaluates TEST only after selecting.
The B9A3 true-Abaqus saved-65V anchor is never used for fitting or selection.
B5 displacement and original 65V remain frozen. NO live-coupling authorization.
No deletion/rename/folder creation; make B9A8 output folder in Finder first.
"""
from __future__ import annotations
import argparse, importlib.util, hashlib, json, math, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('results/65AM_B9_coupled_feasibility/B9A8_residual_correction')
REGIONS = {'inner_left': ('left_inner','center_left'),
           'inner_right': ('center_right','right_inner')}
RIDGE_ALPHA = (1e-4, .001, .01, .1, 1., 10., 100., 1000.)
SHRINK = (0., .1, .25, .5, .75, 1.)
NN_BLEND = (0., .25, .5, .75, 1.)
SEEDS = (17, 37)


def load_previous(root):
    p = root/'65AM_B9A7_force_head.py'
    if not p.is_file():
        raise FileNotFoundError('Keep original B9A7 script in project root: '+str(p))
    spec = importlib.util.spec_from_file_location('B9A7_frozen_utils', str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def read_npz(p):
    if not Path(p).is_file():raise FileNotFoundError(str(p))
    with np.load(p,allow_pickle=False) as a:
        return {k: np.asarray(a[k]).copy() for k in a.files}


def ensure_finite(a, name):
    a=np.asarray(a,dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError('Nonfinite value in '+name)
    return a


def choose_dir(root, explicit, options, required_names, label):
    if explicit:
        d=Path(explicit).expanduser()
        if not d.is_absolute():d=root/d
        d=d.resolve()
        if not all((d/n).is_file() for n in required_names):
            raise FileNotFoundError('Missing '+label+' files in '+str(d)+': '+', '.join(required_names))
        return d
    for d in options:
        if all((d/n).is_file() for n in required_names):return d
    raise FileNotFoundError('Missing '+label+'. Checked:\n'+ '\n'.join(str(d) for d in options)+
      '\nCopy original, frozen 65V exports into one of these directories using Finder (DO NOT regenerate/refit force PCA).')


def gelu_exact(x):
    x=np.asarray(x,float)
    u=x/math.sqrt(2.)
    erf=np.fromiter((math.erf(float(t)) for t in u.ravel()),float,count=u.size).reshape(u.shape)
    return .5*x*(1.+erf)


def export_old_pred(model, x):
    x=ensure_finite(x,'branch')
    if x.ndim==1:x=x[None,:]
    mean=ensure_finite(model['branch_mean'],'branch_mean').ravel()
    std=ensure_finite(model['branch_std'],'branch_std').ravel()
    if x.shape[1]!=len(mean) or len(std)!=len(mean) or np.any(std<=0):
        raise RuntimeError('Original 65V exported force head feature count/scale mismatch')
    assert x.ndim==2
    z=(x-mean)/std
    fc=[]
    nm=int(np.asarray(model['number_members']).ravel()[0])
    if nm<1 or nm>100:raise RuntimeError('Invalid ensemble member count')
    for i in range(nm):
        a=z.copy()
        nl=int(np.asarray(model[f'member_{i}_number_layers']).ravel()[0])
        for k in range(nl):
            W=ensure_finite(model[f'member_{i}_W{k}'],'old layer')
            b=ensure_finite(model[f'member_{i}_b{k}'],'old bias')
            a=a@W.T+b
            if k<nl-1:a=gelu_exact(a)
        fc.append(a)
    p=np.mean(fc,axis=0)
    force_mean=ensure_finite(model['force_coeff_mean'],'force_coeff_mean').ravel()
    force_std=ensure_finite(model['force_coeff_std'],'force_coeff_std').ravel()
    if p.shape[1]!=len(force_mean):raise RuntimeError('Original force PCA latent size mismatch')
    return p*force_std+force_mean


def original_g(region, x, model, pca_map):
    m=export_old_pred(model,x)
    offsets=[]
    cursor=0
    for name in REGIONS[region]:
        z=pca_map[name]
        n=int(np.asarray(z['number_modes']).ravel()[0])
        gm=ensure_finite(z['g_mean'],'g_mean').ravel()
        mat=ensure_finite(z['g_matrix'],'g_matrix')
        if mat.shape[0]!=len(gm) or mat.shape[1]!=n:
            raise RuntimeError('PCA g_matrix shape mismatch '+name)
        if cursor+n>m.shape[1]:raise RuntimeError('Latent force length mismatch '+name)
        offsets.append(gm + m[:,cursor:cursor+n]@mat.T)
        cursor+=n
    if cursor!=m.shape[1]:raise RuntimeError('Extra unused force PCA components '+region)
    ans=np.concatenate(offsets,axis=1)
    return ensure_finite(ans,'original '+region+' predicted force')


def load_everything(root, old_dir, pca_dir, verbose=True):
    prev=load_previous(root)
    anchor,reference,anchor_hash=prev.load_anchor(root)
    datasets={r:prev.load_region(root,anchor,r) for r in REGIONS}
    model_dir=choose_dir(root,old_dir,[root/'exports/hybrid_force_models_7region',root/'no_models_7region'],
        [r+'_direct_ensemble_force.npz' for r in REGIONS], 'original 65V force-head exports')
    pca_folder=choose_dir(root,pca_dir,[root/'data/hybrid_force_pca_7region',root/'force_pca_7region'],
        ['force_pca_'+i+'.npz' for pair in REGIONS.values() for i in pair], 'original force PCA mappings')
    out={}
    old=np.asarray(anchor['old_NO_g'],float)
    all_scale=np.asarray(anchor['force_scale'],float)
    for r,names in REGIONS.items():
        modelf=model_dir/(r+'_direct_ensemble_force.npz')
        model=read_npz(modelf)
        pcafiles={n:pca_folder/('force_pca_'+n+'.npz') for n in names}
        pm={n:read_npz(p) for n,p in pcafiles.items()}
        # Check that the provided model EXACTLY reproduces what 65V used at the anchor.
        feature=np.concatenate([anchor['material'],anchor['c_'+names[0]],anchor['c_'+names[1]]])
        pred=original_g(r,feature,model,pm)[0]
        mode_slices=[prev.SL[n] for n in names]
        old_at=np.concatenate([old[s] for s in mode_slices])
        scale=np.concatenate([all_scale[s] for s in mode_slices])
        normalized_gap=float(np.sqrt(np.mean(((pred-old_at)/scale)**2)))
        if normalized_gap>2e-4:
            raise RuntimeError('STOP: loaded original force head does not reproduce 65V anchor for '+r+
              ' (scaled gap %.6g). Do not train a correction to the wrong baseline.'%normalized_gap)
        if 'g_std' in model:
            gstd=np.maximum(np.abs(np.asarray(model['g_std'],float).ravel()),1.)
            if gstd.shape!=scale.shape or not np.allclose(gstd,scale,rtol=2e-4,atol=2e-4):
                raise RuntimeError('Original 65V force normalization mismatch in '+r)
        ds=datasets[r]
        old_samples=original_g(r,ds['x'],model,pm)
        assert old_samples.shape==ds['y'].shape
        b9a7_path=root/prev.OUT_REL/('B9A7_'+r+'_force_head.npz')
        st=read_npz(b9a7_path)
        for field,val in (('data_x_sha256',ds['x_sha256']),('data_y_sha256',ds['y_sha256']),('split_sha256',ds['split_sha256'])):
            if str(np.asarray(st[field]).item())!=val:
                raise RuntimeError('B9A7 checkpoint no longer matches training dataset: '+r+'/'+field)
        b9a7_pred=prev.infer(st,ds['x'])
        out[r]=dict(ds=ds,original=old_samples,model=model,pca=pm,b9a7=b9a7_pred,
            b9a7_state=st,feature=feature,force_scale=scale,
            paths={'model':str(modelf),'B9A7':str(b9a7_path),**{'pca_'+n:str(p) for n,p in pcafiles.items()}},
            source_hashes={'old_model':sha(modelf),'B9A7':sha(b9a7_path),**{'pca_'+n:sha(p) for n,p in pcafiles.items()}},
            anchor_replay_gap_scaled_RMS=normalized_gap)
        if verbose:
            print('CHECK PASS',r,'train/val/test',*[len(ds['split'][sp]) for sp in ('train','validation','test')],
                  'original baseline anchor gap %.8f'%normalized_gap)
    return prev,anchor,reference,anchor_hash,out,model_dir,pca_folder


def fit_ridge(X,Y,alpha):
    D=np.column_stack([X,np.ones(len(X))])
    pen=np.diag(np.r_[np.ones(X.shape[1]),0.])*alpha
    return np.linalg.solve(D.T@D+pen+np.eye(D.shape[1])*1e-10,D.T@Y)


def ridge_pred(X,W):return np.column_stack([X,np.ones(len(X))])@W

def scaled_rms(pred,truth,scale):
    return float(np.sqrt(np.mean(((pred-truth)/scale)**2)))


def smallnn(X,W):
    Z=np.tanh(X@W['W1'].T+W['b1'])
    return np.tanh(Z@W['W2'].T+W['b2'])@W['W3'].T+W['b3']


def train_network(X,Y,baseline,train,val,seed,epochs,patience,device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    xt=torch.tensor(X[train],dtype=torch.float32,device=device)
    xv=torch.tensor(X[val],dtype=torch.float32,device=device)
    yt=torch.tensor(Y[train]-baseline[train],dtype=torch.float32,device=device)
    vtarget=torch.tensor(Y[val],dtype=torch.float32,device=device)
    vbase=torch.tensor(baseline[val],dtype=torch.float32,device=device)
    model=nn.Sequential(nn.Linear(X.shape[1],24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,Y.shape[1])).to(device)
    with torch.no_grad():
        nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=5e-3)
    best=float('inf');best_ep=0;state=None
    for ep in range(1,epochs+1):
        model.train();opt.zero_grad()
        val0=model(xt)
        loss=((val0-yt)**2).mean()+1e-4*(val0**2).mean()
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            err=float(torch.sqrt(torch.mean((vbase+model(xv)-vtarget)**2)).cpu())
        if err<best-1e-7:
            best=err;best_ep=ep
            state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        if ep-best_ep>=patience:break
    if state is None:raise RuntimeError('NN did not produce checkpoint')
    w={}
    for i,n in zip((0,2,4),('1','2','3')):
        w['W'+n]=state[f'{i}.weight'];w['b'+n]=state[f'{i}.bias']
    return w,best_ep,best


def fit_one(region,part,epochs,patience,device,use_nn):
    ds=part['ds'];y=ds['y'];scale=part['force_scale']
    old=part['original'];X=ds['xx'];target=(y-old)/scale
    train,val=ds['split']['train'],ds['split']['validation']
    baseline_val=scaled_rms(old[val],y[val],scale)
    winner={'type':'unchanged','W':np.zeros((X.shape[1]+1,y.shape[1])), 'blend':0.,'weights':{},'nn_blend':0.,'val':baseline_val}
    options=[]
    for alpha in RIDGE_ALPHA:
        W=fit_ridge(X[train],target[train],alpha)
        pred=ridge_pred(X,W)
        for blend in SHRINK:
            score=float(np.sqrt(np.mean((target[val]-blend*pred[val])**2)))
            options.append((score,alpha,blend,W))
    options.sort(key=lambda v:v[0]);score,alpha,blend,W=options[0]
    if score < .99*winner['val']:
        winner.update(type='ridge',W=W,blend=blend,val=score)
    print(region,'old validation RMS %.6f best ridge %.6f'%(baseline_val,score),'alpha',alpha,'blend',blend)
    trails=[]
    if use_nn:
        # NN is trained to improve the selected ridge correction (no test labels).
        fixed=ridge_pred(X,winner['W'])*winner['blend']
        for seed in SEEDS:
            w,ep,nnscore=train_network(X,target,fixed,train,val,seed,epochs,patience,device)
            net=smallnn(X,w)
            candidates=[(float(np.sqrt(np.mean((target[val]-fixed[val]-b*net[val])**2))),b) for b in NN_BLEND]
            newscore,nb=min(candidates,key=lambda x:x[0])
            trails.append(dict(seed=seed,best_epoch=ep,raw_val=nnscore,blended_val=newscore,blend=nb))
            if newscore<.98*winner['val'] and nb>0:
                winner.update(type='ridge_plus_nn',weights=w,nn_blend=nb,val=newscore)
    winner['alpha']=float(alpha)
    return winner,trails


def apply_correction(part,model,X=None):
    ds=part['ds']; x=ds['xx'] if X is None else (np.asarray(X,float)-ds['x_mean'])/ds['x_scale']
    if x.ndim==1:x=x[None,:]
    c=model['blend']*ridge_pred(x,model['W'])
    if model['type']=='ridge_plus_nn':c+=model['nn_blend']*smallnn(x,model['weights'])
    return c


def predict_final(part,model,X=None,old=None):
    if X is None:X=part['ds']['x']
    if old is None:
        # For train/test original is saved; for anchor call original_g explicitly.
        if X is part['ds']['x']:old=part['original']
        else:old=original_g(part['region'],X,part['model'],part['pca'])
    corr=apply_correction(part,model,X)
    return old+corr*part['force_scale']


def save_chk(p,model,part,anchor_hash):
    a=dict(type=np.asarray(model['type']),ridge_W=model['W'],ridge_blend=np.asarray(model['blend']),
       nn_blend=np.asarray(model['nn_blend']),validation_scaled_RMS=np.asarray(model['val']),
       x_mean=part['ds']['x_mean'],x_scale=part['ds']['x_scale'],force_scale=part['force_scale'],
       data_x_sha256=np.asarray(part['ds']['x_sha256']),data_y_sha256=np.asarray(part['ds']['y_sha256']),
       split_sha256=np.asarray(part['ds']['split_sha256']),anchor_sha256=np.asarray(anchor_hash),
       old_model_sha256=np.asarray(part['source_hashes']['old_model']),
       B9A7_model_sha256=np.asarray(part['source_hashes']['B9A7']))
    for k,v in part['source_hashes'].items():
        if k.startswith('pca_'):a[k+'_sha256']=np.asarray(v)
    if model['type']=='ridge_plus_nn':a.update(model['weights'])
    np.savez_compressed(p,**a)


def load_chk(path,part,anchor_hash):
    z=read_npz(path)
    pairs={'data_x_sha256':part['ds']['x_sha256'],'data_y_sha256':part['ds']['y_sha256'],
           'split_sha256':part['ds']['split_sha256'],'anchor_sha256':anchor_hash,
           'old_model_sha256':part['source_hashes']['old_model'],
           'B9A7_model_sha256':part['source_hashes']['B9A7']}
    for k,v in part['source_hashes'].items():
        if k.startswith('pca_'):pairs[k+'_sha256']=v
    for k,v in pairs.items():
        if str(np.asarray(z[k]).item())!=v:raise RuntimeError('B9A8 source changed after training '+k)
    model=dict(type=str(z['type'].item()), W=z['ridge_W'],blend=float(z['ridge_blend']),
        nn_blend=float(z['nn_blend']),val=float(z['validation_scaled_RMS']),
        weights={k:z[k] for k in ('W1','b1','W2','b2','W3','b3') if k in z})
    return model


def out_folder(root):
    d=root/OUT
    if not d.is_dir():raise RuntimeError('Create empty output directory using Finder first: '+str(d))
    return d


def require_free(paths):
    for p in paths:
        if p.exists():raise RuntimeError('Refuse to overwrite existing result: '+str(p))
        if not p.parent.is_dir():raise RuntimeError('Parent result folder missing: '+str(p.parent))


def write_csv(p,rows):
    if not rows:raise ValueError('No report rows')
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save_json(p,report):p.write_text(json.dumps(report,indent=2))


def train_stage(root,options):
    prev,anchor,ref,anchash,allparts,model_dir,pca_dir=load_everything(root,options.old_force_dir,options.force_pca_dir)
    out=out_folder(root)
    paths=[out/('B9A8_'+r+'_correction.npz') for r in REGIONS]
    paths += [out/x for x in ('B9A8_split_metrics.csv','B9A8_test_modes.csv','B9A8_training_summary.json')]
    require_free(paths)
    import torch
    device=options.device
    if device=='auto':device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS not supported; use --device cpu')
    details=[];reports=[];testm=[]
    for region,part in allparts.items():
        part['region']=region
        winner,trials=fit_one(region,part,options.epochs,options.patience,device,not options.ridge_only)
        prediction=predict_final(part,winner)
        scale=part['force_scale'];ds=part['ds'];y=ds['y']
        metrics={}
        for tag,pred in (('Original65V',part['original']),('B9A7_head',part['b9a7']),('B9A8_corrected',prediction)):
            for split in ('train','validation','test'):
                ids=ds['split'][split]
                metrics[(tag,split)]=scaled_rms(pred[ids],y[ids],scale)
                reports.append(dict(region=region,model=tag,split=split,samples=len(ids),
                    scaled_force_RMS=metrics[(tag,split)],
                    force_rel_L2_pct=100*np.linalg.norm((pred[ids]-y[ids]).ravel())/max(np.linalg.norm(y[ids].ravel()),1e-12)))
            ids=ds['split']['test'];pos=0
            for interface in REGIONS[region]:
                n=prev.SL[interface].stop-prev.SL[interface].start
                for j in range(n):
                    m=pos+j
                    testm.append(dict(region=region,model=tag,interface=interface,mode=j+1,
                        scaled_test_RMSE=float(np.sqrt(np.mean(((pred[ids,m]-y[ids,m])/scale[m])**2)))))
                pos+=n
        gain=metrics[('Original65V','test')]-metrics[('B9A8_corrected','test')]
        print('TEST',region,'Original65V %.6f | B9A7 %.6f | B9A8 %.6f'%(metrics[('Original65V','test')],metrics[('B9A7_head','test')],metrics[('B9A8_corrected','test')]))
        details.append(dict(region=region,method=winner['type'],alpha=winner['alpha'],
            ridge_blend=winner['blend'],nn_blend=winner['nn_blend'],
            selected_validation_scaled_RMS=winner['val'],
            original_validation_scaled_RMS=metrics[('Original65V','validation')],
            test_old_RMS=metrics[('Original65V','test')],
            test_B9A7_RMS=metrics[('B9A7_head','test')],
            test_B9A8_RMS=metrics[('B9A8_corrected','test')],
            test_improvement_vs_old_RMS=gain,nn_trials=trials,
            original_anchor_replay_gap_scaled_RMS=part['anchor_replay_gap_scaled_RMS'],
            source_hashes=part['source_hashes']))
        save_chk(out/('B9A8_'+region+'_correction.npz'),winner,part,anchash)
    write_csv(out/'B9A8_split_metrics.csv',reports)
    write_csv(out/'B9A8_test_modes.csv',testm)
    save_json(out/'B9A8_training_summary.json',dict(step='65AM-B9A8',phase='train_validate_test',
        original_force_model_folder=str(model_dir),force_pca_folder=str(pca_dir),
        saved_65V_anchor_sha256=anchash,B5_frozen=True,original_65V_frozen=True,
        training_uses_only_64J_train=True,hyperparameter_selection_only_validation=True,
        B9A3_true_anchor_NEVER_used_for_selection=True,
        note='Held-out test figures are a diagnostic comparison. Neither checkpoint is approved for live coupling.',
        models=details,live_Broyden_approved=False))
    print('B9A8 TRAIN+TEST COMPLETE. All new outputs:',out)


def anchor_stage(root,options):
    prev,anchor,ref,anchash,parts,model_dir,pca_dir=load_everything(root,options.old_force_dir,options.force_pca_dir)
    out=out_folder(root)
    paths=[out/x for x in ('B9A8_anchor_modes.csv','B9A8_anchor_summary.json','B9A8_anchor_modes.png')]
    require_free(paths)
    old=np.asarray(anchor['old_NO_g'],float).copy()
    fe=np.asarray(anchor['FE_g_inferred'],float)
    sc=np.asarray(anchor['force_scale'],float)
    true=np.asarray(ref['reference_NO_g'],float)
    mask=np.asarray(ref['reference_mask'],bool)
    new=old.copy();b9a7=old.copy();regrep={}
    for region,names in REGIONS.items():
        part=parts[region];part['region']=region
        ck=load_chk(out/('B9A8_'+region+'_correction.npz'),part,anchash)
        feature=part['feature']
        b_old=original_g(region,feature,part['model'],part['pca'])
        b_new=predict_final(part,ck,X=feature[None,:],old=b_old)[0]
        b_7=prev.infer(part['b9a7_state'],feature)[0]
        j=0
        for name in names:
            n=prev.SL[name].stop-prev.SL[name].start
            new[prev.SL[name]]=b_new[j:j+n]
            b9a7[prev.SL[name]]=b_7[j:j+n]
            j+=n
        norm=(feature-part['ds']['x_mean'])/part['ds']['x_scale']
        nearest=np.min(np.linalg.norm(part['ds']['xx'][part['ds']['split']['train']]-norm[None,:],axis=1))
        regrep[region]=dict(nearest_train_normalized_distance=float(nearest),
            is_exact_train_duplicate=bool(nearest<1e-4),source_hashes=part['source_hashes'])
    oracle=old.copy();oracle[mask]=true[mask]
    def residual(a):return scaled_rms(a+fe,np.zeros(26),sc)
    metrics=dict(saved_65V_old_NO_RMS=residual(old),true_Abaqus_NO_at_saved_q_RMS=residual(oracle),
        B9A7_force_head_RMS_ESTIMATE=residual(b9a7),B9A8_residual_corrected_RMS_ESTIMATE=residual(new),
        old_vs_true_inner_scaled_RMS=scaled_rms(old[mask],true[mask],sc[mask]),
        B9A7_vs_true_inner_scaled_RMS=scaled_rms(b9a7[mask],true[mask],sc[mask]),
        B9A8_vs_true_inner_scaled_RMS=scaled_rms(new[mask],true[mask],sc[mask]))
    rows=[]
    for name in prev.INTERFACES:
        s=prev.SL[name]
        for j,idx in enumerate(range(s.start,s.stop)):
            rows.append(dict(interface=name,mode=j+1,inner_mode=bool(mask[idx]),
                force_scale_N=float(sc[idx]),FE_reference_generalized_N=float(fe[idx]),
                old_NO_generalized_N=float(old[idx]),B9A7_generalized_N=float(b9a7[idx]),
                B9A8_generalized_N=float(new[idx]),
                true_Abaqus_NO_N=float(true[idx]) if mask[idx] else '',
                normalized_old_residual=float((fe[idx]+old[idx])/sc[idx]),
                normalized_B9A7_residual=float((fe[idx]+b9a7[idx])/sc[idx]),
                normalized_B9A8_residual=float((fe[idx]+new[idx])/sc[idx]),
                normalized_B9A8_vs_true=float((new[idx]-true[idx])/sc[idx]) if mask[idx] else '',
                normalized_old_vs_true=float((old[idx]-true[idx])/sc[idx]) if mask[idx] else ''))
    write_csv(paths[0],rows)
    save_json(paths[1],dict(step='65AM-B9A8',status='diagnostic_only',anchor_sha256=anchash,
         metrics=metrics,regions=regrep,anchor_labels_never_used_in_training=True,
         no_new_Abaqus_FEM_solve=True,not_re_equilibrated=True,
         note='Smaller saved-state residual may be cancellation, NOT proof of physical accuracy. Coupling not authorized.',
         live_Broyden_approved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['inner_mode']]
    x=np.arange(len(selected));w=.26
    fig,ax=plt.subplots(figsize=(15,5))
    for k,(name,col) in enumerate((('Old', 'normalized_old_vs_true'),('B9A7','b9a7'),('B9A8','normalized_B9A8_vs_true'))):
        if name=='B9A7':y=[abs((r['B9A7_generalized_N']-r['true_Abaqus_NO_N'])/r['force_scale_N']) for r in selected]
        else:y=[abs(r[col]) for r in selected]
        ax.bar(x+(k-1)*w,y,width=w,label=name)
    ax.set_xticks(x);ax.set_xticklabels([r['interface']+'/m'+str(r['mode']) for r in selected],rotation=55,ha='right')
    ax.set_ylabel('Absolute generalized-force error / original 65V scale')
    ax.set_title('B9A8: saved-state Abaqus-reference force-mode errors (diagnostic only)')
    ax.legend();fig.tight_layout();fig.savefig(paths[2],dpi=190);plt.close(fig)
    print('B9A8 ANCHOR:',json.dumps(metrics,indent=2))
    print('IMPORTANT: saved-state estimate only; no approval for live Broyden')


def selftest():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(60,12));y=x@rng.normal(size=(12,9)) + 1.
    W=fit_ridge(x,y,1e-6)
    assert scaled_rms(ridge_pred(x,W),y,np.ones(9))<1e-5
    weights=dict(W1=np.zeros((24,12)),b1=np.zeros(24),W2=np.zeros((24,24)),b2=np.zeros(24),W3=np.zeros((9,24)),b3=np.zeros(9))
    assert smallnn(x,weights).shape==(60,9)
    assert np.allclose(smallnn(x,weights),0.)
    fake={'branch_mean':np.zeros(12),'branch_std':np.ones(12),'force_coeff_mean':np.zeros(2),'force_coeff_std':np.ones(2),'number_members':np.array([1]),
          'member_0_number_layers':np.array([1]),'member_0_W0':np.zeros((2,12)),'member_0_b0':np.array([.4,-.3])}
    assert np.allclose(export_old_pred(fake,x),np.array([.4,-.3]))
    print('B9A8 SELFTEST PASS: old export evaluation, ridge, NN residual mapping')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('selftest','check','train','anchor'))
    p.add_argument('--project',default='.')
    p.add_argument('--old-force-dir',default=None,help='Directory holding original 65V exported inner force models')
    p.add_argument('--force-pca-dir',default=None,help='Directory holding frozen original force PCA maps')
    p.add_argument('--ridge-only',action='store_true',help='Skip optional small NN force residual')
    p.add_argument('--epochs',type=int,default=350)
    p.add_argument('--patience',type=int,default=75)
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    a=p.parse_args()
    if a.stage=='selftest':return selftest()
    root=Path(a.project).expanduser().resolve()
    if a.stage=='check':
        _,anchor,ref,hash_,data,mdir,pdir=load_everything(root,a.old_force_dir,a.force_pca_dir)
        print('CHECK COMPLETE: anchor verified',hash_[:16],'\nOld force model directory',mdir,'\nFrozen force PCA directory',pdir)
        return
    if a.epochs<40 or a.patience<15:raise ValueError('Use at least 40 epochs, patience 15')
    if a.stage=='train':return train_stage(root,a)
    if a.stage=='anchor':return anchor_stage(root,a)

if __name__=='__main__':main()
