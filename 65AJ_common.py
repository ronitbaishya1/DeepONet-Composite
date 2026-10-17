#!/usr/bin/env python3
"""65AJ shared utilities. Original 65AI code and checkpoints remain unmodified.

Do not load untrusted checkpoints: torch.load uses pickle in the original
pipeline. Only read checkpoints you generated in this existing project.
"""
import importlib, json
from pathlib import Path
import numpy as np
import torch
ai=importlib.import_module('65AI_common')
REGIONS=ai.REGIONS
ROOT='65AJ_architecture_verification'
CONTROL_HIDDEN=166

def new_folder(root,sub):
    p=Path(root)/'results'/ROOT/sub
    if p.exists() and any(p.iterdir()):
        raise RuntimeError('Results already exist; no overwrite. Remove ONLY: '+str(p))
    p.mkdir(parents=True,exist_ok=True)
    return p

def control_path(root,region,member):
    return Path(root)/'results'/ROOT/'B_matched_mlp'/region/f'member_{member:02d}'/'local_mechanics.pt'

def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))

def inspect_control(st,root,region,member,device):
    lgno,_=ai.load_candidate(root,'local_lgno',st,device)
    params=parameter_count(lgno)
    matched=ai.DisplacementToLocalMechanics(lgno.input_dim,'pointwise',CONTROL_HIDDEN,lgno.cap).to(device)
    nmatch=parameter_count(matched)
    if abs(nmatch-params)/params > .01:
        raise RuntimeError(f'Parameter gap > 1% ({nmatch} vs {params})')
    return params,nmatch

def load_control(root,st,device):
    p=control_path(root,st['region'],st['member'])
    if not p.is_file():raise FileNotFoundError(str(p))
    ck=torch.load(p,map_location='cpu',weights_only=False)  # trusted local only
    needed=dict(architecture=ai.ARCH,arm='pointwise',region=st['region'],
                member=st['member'],prior_sha=st['prior_sha'],source_sha=st['source_sha'],hidden=CONTROL_HIDDEN)
    for k,v in needed.items():
        if ck.get(k)!=v:raise RuntimeError('65AJ control provenance mismatch: '+k)
    model=ai.DisplacementToLocalMechanics(ck['input_dim'],'pointwise',ck['hidden'],ck['cap']).to(device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    return model,ck

def models_for_comparison(root,st,device):
    out={}
    for arm in ('pointwise','local_lgno'):
        out['65AI_'+arm]=ai.load_candidate(root,arm,st,device)[0]
    out['65AJ_matched_mlp']=load_control(root,st,device)[0]
    return out

def attach_external(root,st,ext,device):
    """Augment in memory with fresh FE cases, retaining ORIGINAL training scalers.

    The reference FE U is used ONLY for constructing the interface PCA inputs,
    as an offline oracle-BC evaluation, and as a score label. It is NEVER passed
    as an interior model feature. For actual online prediction, interface PCA
    is supplied by the hybrid FE/NO coupling solver instead.
    """
    ds=st['ds'];n0=len(ds['branch_inputs'])
    mat=np.asarray(ext['materials'],np.float32)
    ufem=np.asarray(ext['U'],np.float32)
    LE=np.asarray(ext['LE'],np.float32)
    S=np.asarray(ext['S'],np.float32)
    if mat.ndim!=2 or mat.shape[1]!=3:raise ValueError('materials must be [K,3]')
    k=len(mat);N=len(ds['coordinates']);E=len(ds['ip_coordinates'])
    if k==0 or ufem.shape!=(k,N,3) or LE.shape!=(k,E,6) or S.shape!=(k,E,6):
        raise ValueError(f'External field dimensions invalid. Expected U {(k,N,3)}, LE/S {(k,E,6)}')
    if min(np.min(mat),1.0)<=0 or any(not np.isfinite(a).all() for a in (mat,ufem,LE,S)):
        raise ValueError('Nonfinite or invalid FEM arrays')
    for key,ref in [('coordinates',ds['coordinates']),('ip_coordinates',ds['ip_coordinates'])]:
        if key not in ext:raise ValueError('External NPZ must include '+key+' to prove same geometry/order')
        given=np.asarray(ext[key],dtype=np.float64)
        if given.shape!=ref.shape or np.max(np.linalg.norm(given-ref,axis=-1))>1e-5:
            raise RuntimeError('External node/IP geometry or ordering mismatches training template: '+key)
    # New material cases must not be identical to ANY original sample.
    old=np.asarray(ds['branch_inputs'][:,:3],np.float64)
    width=np.maximum(np.ptp(old,axis=0),1.0)
    nearest=np.linalg.norm((mat[:,None,:]-old[None,:,:])/width,axis=-1).min(axis=1)
    if np.any(nearest<1e-4):raise RuntimeError('External materials duplicate prior cases; make NEW cases')
    ctx=ai.co.interface_context(root,ds,device)
    # Original interface PCA basis; least-squares projection of new physical
    # FEM interface displacement for identical geometry and local node order.
    br=np.zeros((k,int(ctx['branch_dim'])),np.float32);br[:,:3]=mat
    resid={}
    for side_name in ('left','right'):
        side=ctx[side_name]
        if side is None:continue
        ix=ctx[side_name+'_local_indices']
        ix=ix.detach().cpu().numpy() if isinstance(ix,torch.Tensor) else np.asarray(ix)
        Y=ufem[:,ix,:].reshape(k,-1).astype(np.float64)
        mean=np.asarray(side['mean'],np.float64).reshape(-1)
        basis=np.asarray(side['basis'],np.float64)
        coeff=np.linalg.lstsq(basis,(Y-mean).T,rcond=None)[0].T
        sl=ctx[side_name+'_slice']
        if coeff.shape!=(k,sl.stop-sl.start):raise RuntimeError('PCA dimensions mismatch')
        br[:,sl]=coeff.astype(np.float32)
        rec=mean[None,:]+coeff@basis.T
        resid[side_name+'_rms_mm']=float(np.sqrt(np.mean((Y-rec)**2)))
        resid[side_name+'_max_abs_mm']=float(np.max(np.abs(Y-rec)))
    aug=dict(ds)
    for key,add in [('branch_inputs',br),('U',ufem),('LE',LE),('S',S)]:
        aug[key]=np.concatenate([ds[key],add],axis=0)
    # Keep ORIGINAL train/validation/test indices untouched, so scalers and
    # normalization are frozen and none of the new data can modify training.
    aug['splits']=ds['splits']
    aug['_root']=root
    ids=np.arange(n0,n0+k,dtype=np.int64)
    normalized,_,_=ai.co.scaled_inputs(st['ck'],aug)
    inputs=ai.ag.ac.aa.tensor_inputs(st['ck'],aug,device)
    src=ai.co.make_model(st['ck'],device).eval()
    U_pred=ai.ag.ac.aa.predict_numpy(src,inputs,ids,batch_size=2,correct=True)
    del src
    st['ds']=aug
    st['mats']=torch.as_tensor(np.array(normalized,copy=True),dtype=torch.float32,device=device)
    nold=len(st['pred'])
    st['pred']=np.concatenate((st['pred'],U_pred.astype(np.float32)),axis=0)
    st['indices']=np.concatenate((st['indices'],ids))
    st['index_map'].update({int(i):nold+j for j,i in enumerate(ids)})
    return ids,br,resid,nearest.tolist()
