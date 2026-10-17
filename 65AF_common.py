#!/usr/bin/env python3
"""65AF: local LGNO-inspired strain refinement, trained on original 64J TRAIN.

Reuses frozen 65AA displacement/LE/S/force heads and frozen 65AC LGNO.
Model learns a local correction to the six MATERIAL logarithmic strain fields,
using C3D8R nodal displacement neighborhoods, existing gradients, material
properties and the existing learned strain head. This does NOT force the new
LE to be the derivative of the frozen U field. Never deploy automatically.

Research inspiration: https://github.com/baiming-zhang/LGNO
Adaptation is original code, not copied from that repo.
"""
from pathlib import Path
import importlib, sys, hashlib
import numpy as np
import torch
from torch import nn

ac=importlib.import_module('65AC_common')
ROOT='65AF_local_strain'
REGIONS=('inner_left','inner_right')
ARMS=('pointwise','lgno')
LABELS=('11','22','33','12','13','23')
ARCH='65AF_local_material_Hencky_strain_v1'
N_FEATURES=99


def relative(x,y):
    return ac.aa.co.relative_l2(np.asarray(x),np.asarray(y))


def safe_output(root,sub):
    d=root/'results'/ROOT/sub
    if d.exists() and any(d.iterdir()):
        raise RuntimeError('Existing results protected. If rerunning, manually delete ONLY this folder:\n'+str(d))
    d.mkdir(parents=True,exist_ok=True)
    return d


def cases(root,region,member,device,need_predictions=True):
    st=ac.prepare(root,region,member,device,need_predictions)
    file=ac.checkpoint_path(root,'lgno',region,member)
    if not file.is_file(): raise FileNotFoundError('Need completed Step65AC LGNO checkpoint: '+str(file))
    lg,info=ac.load_adapter(file,'lgno',region,member,st['source_sha'],device)
    st['lgno_model']=lg
    st['lgno_file']=file
    st['lgno_sha']=ac.aa.co.sha256(file)
    return st


def head_predictions(st,ids,device,batch=4):
    """Existing frozen source V4 LE/S heads, with original normalization."""
    ck=st['ck'];ds=st['ds']
    bnorm,_,ipnorm=ac.aa.co.scaled_inputs(ck,ds)
    net=ac.aa.co.make_model(ck,device).eval()
    ip=torch.tensor(np.array(ipnorm,copy=True),dtype=torch.float32,device=device)
    le=[];s=[]
    with torch.no_grad():
        for k in range(0,len(ids),batch):
            j=ids[k:k+batch]
            b=torch.tensor(np.array(bnorm[j],copy=True),dtype=torch.float32,device=device)
            le.append(net.strain_operator(b,ip).detach().cpu().numpy())
            s.append(net.stress_operator(b,ip).detach().cpu().numpy())
    del net
    L=np.concatenate(le)*np.asarray(ck['LE_std'],dtype=np.float32)+np.asarray(ck['LE_mean'],dtype=np.float32)
    S=np.concatenate(s)*np.asarray(ck['S_std'],dtype=np.float32)+np.asarray(ck['S_mean'],dtype=np.float32)
    if L.shape!=ds['LE'][ids].shape or S.shape!=L.shape:raise RuntimeError('Mechanics heads output wrong shape')
    return L.astype(np.float32),S.astype(np.float32)


def stiffness(ds,ids):
    # First three branch columns are E1, E2, G12 in MPa.
    return ac.aa.co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][ids,:3]).astype(np.float32)


def C_LE(LE,C):
    return np.einsum('bci,bei->bec',C,LE,optimize=True)


def strain_scales(ds):
    tr=ds['splits']['train']
    # Weighted losses normalize each component, with a floor for small terms.
    std=np.std(ds['LE'][tr],axis=(0,1)).astype(np.float32)
    floor=max(float(std.max())*.03,1e-5)
    return np.maximum(std,floor)


def stress_scales(ds):
    tr=ds['splits']['train']
    std=np.std(ds['S'][tr],axis=(0,1)).astype(np.float32)
    floor=max(float(std.max())*.03,.10)
    return np.maximum(std,floor)


class LocalStrainRefiner(nn.Module):
    """LGNO arm: learn local nodal stencil weights -> six LE corrections.

    Baseline = frozen V4 strain head. Correction is a local convolution-like
    contraction of learned weights [6,8,3] with local centered displacement
    increments. The MLP control predicts six corrections from identical inputs.
    """
    def __init__(self,arm,hidden=64,cap=1.0):
        super().__init__()
        if arm not in ARMS:raise ValueError('Arm not recognized')
        self.arm=arm;self.hidden=int(hidden);self.cap=float(cap)
        self.net=nn.Sequential(nn.Linear(N_FEATURES,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,144 if arm=='lgno' else 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,features,local_du,baseline_LE,le_std):
        v=self.cap*torch.tanh(self.net(features))
        if self.arm=='lgno':
            # centered differences annihilate constant displacement;
            # the *frozen learned LE baseline* does not necessarily do so.
            corr=torch.einsum('beqic,beic->beq',v.reshape(*v.shape[:2],6,8,3),local_du)
            corr=corr/np.sqrt(24.)
        else:
            corr=v
        return baseline_LE + corr*le_std.reshape(1,1,6)


def build_features(st,ids,device,heads=None,lgno_grad=None):
    """Build LOCAL features using only predicted quantities at inference.

    Does not read ds['LE'] or ds['S'] except externally for scoring/training.
    Returns features, normalized displacement differences, head LE/S and GL.
    """
    ds=st['ds'];ck=st['ck']
    ids=np.asarray(ids,dtype=int)
    if heads is None:HL,HS=head_predictions(st,ids,device)
    else:HL,HS=heads
    if lgno_grad is None:
        GL=ac.split_predict(st,st['lgno_model'],ids,device,batch=4)
    else:GL=lgno_grad
    lgn=ac.logstrain_from_gradient(GL).astype(np.float32)
    conn=st['geometry']['conn']
    if 'pred' not in st:raise RuntimeError('Call cases(...,need_predictions=True)')
    pos=np.asarray([st['index_map'][int(j)] for j in ids],dtype=int)
    U=np.asarray(st['pred'][pos],np.float32)
    local=U[:,conn,:]
    ustd=np.asarray(ck['U_std'],dtype=np.float32)
    du=(local-local.mean(axis=2,keepdims=True))/ustd[None,None,None,:]
    le_std=strain_scales(ds)
    gw=np.asarray(GL,np.float32)/st['gs'][None,None,:,:]
    N,E=U.shape[0],len(conn)
    g=st['geometry']
    def broadcast(a):return np.broadcast_to(a,(N,)+a.shape[1:])
    # 24 du + 9 grad + 6 LGNO logstrain + 6 head strain + 24 offsets
    # + 24 shape derivatives + 3 element position + 3 material values = 99.
    feats=[du.reshape(N,E,24),gw.reshape(N,E,9),
        lgn/le_std[None,None,:],HL/le_std[None,None,:],
        broadcast(g['offset'][None].reshape(1,E,24)),
        broadcast((g['dN']*g['step'][:,None,:])[None].reshape(1,E,24)),
        broadcast(g['position'][None].reshape(1,E,3)),
        st['mats'][ids,:3].detach().cpu().numpy()[:,None,:].repeat(E,axis=1)]
    F=np.concatenate(feats,axis=-1).astype(np.float32)
    if F.shape!=(N,E,N_FEATURES):raise RuntimeError('Invalid feature dimensions: '+str(F.shape))
    if not np.isfinite(F).all():raise RuntimeError('Nonfinite local stencil features')
    return F,du.astype(np.float32),HL,HS,GL.astype(np.float32)


def predict(model,F,du,HL,le_std,device,batch=4):
    result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(F),batch):
            sl=slice(start,start+batch)
            def ten(a):return torch.as_tensor(np.array(a[sl],copy=True),dtype=torch.float32,device=device)
            Y=model(ten(F),ten(du),ten(HL),
                    torch.tensor(np.array(le_std,copy=True),dtype=torch.float32,device=device))
            result.append(Y.detach().cpu().numpy())
    return np.concatenate(result)


def derived_metrics(L,S,C,ds,idx):
    refL=ds['LE'][idx];refS=ds['S'][idx]
    out={}
    for c,name in enumerate(LABELS):
        out['LE'+name+'_pct']=relative(L[...,c],refL[...,c])
        out['S'+name+'_pct']=relative(S[...,c],refS[...,c])
    phys=C_LE(L,C)
    out['constitutive_S22_pct']=relative(phys[...,1],refS[...,1])
    out['constitutive_disagreement_pct']=relative(S,phys)
    out['S22_RMSE_MPa']=float(np.sqrt(np.mean((S[...,1]-refS[...,1])**2)))
    return out


def checkpoint_path(root,arm,reg,member):
    return root/'results'/ROOT/('B_'+arm)/reg/('member_%02d'%member)/'strain_local.pt'


def load_model(path,arm,reg,member,source_sha,lgno_sha,device):
    ck=torch.load(path,map_location='cpu',weights_only=False) # trusted project file only
    expected={'arch':ARCH,'arm':arm,'region':reg,'member':member,
              'source_sha':source_sha,'lgno_sha':lgno_sha}
    for k,v in expected.items():
        if ck.get(k)!=v:raise RuntimeError('65AF checkpoint mismatch at '+k+': '+str(path))
    net=LocalStrainRefiner(arm,int(ck['hidden']),float(ck['cap'])).to(device)
    net.load_state_dict(ck['model_state_dict'],strict=True)
    net.eval()
    return net,ck
