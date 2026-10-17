#!/usr/bin/env python3
"""65AH: all-output end-to-end DeepONet + optional adaptive local C3D8R stencil.

Physical inputs: [E1,E2,G12] + actual interface PCA coefficients. Geometry
enters via nodal and element-IP coordinates and the original C3D8R B matrices.
Outputs: 3 nodal displacements + 6 IP material logarithmic strains + 6 IP
stresses. Existing generalized-force head is kept EXACTLY frozen.

This is an independently written LGNO-style *local neighbor-weight* adaptation,
inspired by https://github.com/baiming-zhang/LGNO, NOT a copy of its periodic
Navier--Stokes stencil implementation.
"""
from __future__ import annotations
import hashlib, importlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
aa=importlib.import_module('65AA_common')
co=aa.co
SEGMENTS=('inner_left','inner_right')
ARMS=('shared','local')
ROOT_NAME='65AH_unified_operator'
ARCH='65AH_joint_V4_LGNO_C3D8R_v1'
COMP=('11','22','33','12','13','23')


def sha(path):return co.sha256(path)


def source_path(root,seg,member):
    return root/'results'/'65AA_gradient_learning'/'B_gradient'/seg/('member_%02d'%member)/'best_v4_7region.pt'


def output_dir(root,part):
    out=root/'results'/ROOT_NAME/part
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing results protected. To rerun, remove ONLY this generated folder manually:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out


def as_tensor(a,dev,dtype=torch.float32):
    return torch.tensor(np.array(a,copy=True),device=dev,dtype=dtype)


def load_source(root,seg,member,device):
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    ds=aa.load_data(root,seg)
    ds['_root']=root
    src=source_path(root,seg,member)
    ck=co.load_checkpoint(src)
    if ck.get('65AA_gradient_experiment',{}).get('arm')!='gradient':
        raise RuntimeError('Require trained 65AA gradient checkpoint: '+str(src))
    base_file=aa.source_checkpoint(root,seg,member)
    if ck['65AA_gradient_experiment'].get('initialization_sha256')!=sha(base_file):
        raise RuntimeError('65AA checkpoint provenance does not match original 65Z')
    model=co.make_model(ck,device)
    bnorm,nnorm,ipnorm=co.scaled_inputs(ck,ds)
    conn=np.asarray(ds['element_node_indices'],dtype=np.int64)
    dn=aa.element_dN_numpy(ds)
    ctx=co.interface_context(root,ds,device)
    if ctx['branch_dim']!=ck['branch_dim'] or bnorm.shape[1]!=ck['branch_dim']:
        raise RuntimeError('Physical branch must include [E1,E2,G12] + interface coefficients')
    if ck['number_force_coefficients']<=0:raise RuntimeError('Expected nonempty force head')
    train=ds['splits']['train']
    gg=aa.gradient_numpy(ds['U'][train],ds)
    sc=np.std(gg,axis=(0,1)).astype(np.float32)
    sc=np.maximum(sc,max(float(sc.max())*.05,1e-5)).astype(np.float32)
    # Hard compatibility will use exact reconstructed interface targets.
    t=dict(branch=as_tensor(bnorm,device),physical_branch=as_tensor(ds['branch_inputs'],device),
           node=as_tensor(nnorm,device),ip=as_tensor(ipnorm,device),
           conn=as_tensor(conn,device,torch.long),dn=as_tensor(dn,device),
           grad_scale=as_tensor(sc,device).view(1,1,3,3),
           U=as_tensor(ds['U'],device),LE=as_tensor(ds['LE'],device),S=as_tensor(ds['S'],device),
           umean=as_tensor(ck['U_mean'],device).view(1,1,3),
           ustd=as_tensor(ck['U_std'],device).view(1,1,3),
           lemean=as_tensor(ck['LE_mean'],device).view(1,1,6),
           lestd=as_tensor(ck['LE_std'],device).view(1,1,6),
           smean=as_tensor(ck['S_mean'],device).view(1,1,6),
           sstd=as_tensor(ck['S_std'],device).view(1,1,6),
           interface=ctx)
    if not (torch.isfinite(t['branch']).all() and torch.isfinite(t['U']).all() and
            (t['ustd']>0).all() and (t['lestd']>0).all() and (t['sstd']>0).all()):
        raise RuntimeError('Nonfinite training arrays or output normalization')
    return dict(ds=ds,ck=ck,backbone=model,t=t,source=src,source_sha=sha(src),seg=seg,member=member)


def split_in_train(ds,seg,m,tune=12):
    ids=np.asarray(ds['splits']['train'],dtype=np.int64)
    if len(ids)<=tune+8:raise RuntimeError('Too few original train cases')
    perm=np.random.default_rng(650880+100*SEGMENTS.index(seg)+m).permutation(ids)
    fit=np.sort(perm[:-tune]);hold=np.sort(perm[-tune:])
    if set(fit)|set(hold)!=set(ids) or set(fit)&set(hold):raise RuntimeError('Train split overlap')
    assert not (set(fit)|set(hold)) & (set(ds['splits']['test'])|set(ds['splits']['validation']))
    return fit,hold


class EndToEndUnified(nn.Module):
    """Train ALL three pretrained V4 heads AND shared coupling end-to-end.

    The original force head is strictly frozen. At initial zero residual heads,
    original 65AA U/LE/S are reproduced, including exact hard interfaces.
    """
    def __init__(self,ck,device,arm='local',hidden=64,cap=.5):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.hidden=int(hidden);self.cap=float(cap)
        self.backbone=co.make_model(ck,device)
        for p in self.backbone.force_coefficient_head.parameters():p.requires_grad_(False)
        branch_dim=int(ck['branch_dim'])
        self.shared_branch=nn.Sequential(nn.Linear(branch_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.shared_trunk=nn.Sequential(nn.Linear(3,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.node_head=nn.Linear(hidden,3)
        # Adaptive learned stencil: local displacement deltas[24], FEM gradients[9],
        # and common latent[hidden] -> 24 neighbor/axis weights. Applied to du only.
        self.stencil=nn.Sequential(nn.Linear(hidden+24+9,hidden),nn.GELU(),
                                   nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,24))
        # Both arms have the SAME trainable representation, but shared arm sets
        # the learned stencil modification to exactly zero.
        self.ip_mixer=nn.Sequential(nn.Linear(hidden+9,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.strain_head=nn.Linear(hidden,6)
        self.stress_head=nn.Linear(hidden,6)
        for head in (self.node_head,self.stencil[-1],self.strain_head,self.stress_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,ids,t):
        from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
        branch=t['branch'][ids];physical=t['physical_branch'][ids]
        raw=self.backbone.displacement_operator(branch,t['node'])*t['ustd']+t['umean']
        hl=self.shared_branch(branch)
        hn=torch.tanh(hl[:,None,:]+self.shared_trunk(t['node'])[None,:,:])
        raw=raw+self.cap*torch.tanh(self.node_head(hn))*t['ustd']
        U=apply_hard_compatibility_7region(raw,physical,t['interface'])
        LE=self.backbone.strain_operator(branch,t['ip'])*t['lestd']+t['lemean']
        S=self.backbone.stress_operator(branch,t['ip'])*t['sstd']+t['smean']
        hip=torch.tanh(hl[:,None,:]+self.shared_trunk(t['ip'])[None,:,:])
        ue=U[:,t['conn'],:]
        du=ue-ue.mean(dim=2,keepdim=True)
        g=torch.einsum('beic,eij->becj',ue,t['dn'])
        g_norm=(g/t['grad_scale']).clamp(-25,25)
        if self.arm=='local':
            feat=torch.cat((hip,(du/t['ustd'].view(1,1,1,3)).clamp(-20,20).flatten(2),
                            g_norm.flatten(2)),dim=-1)
            weights=.5*torch.tanh(self.stencil(feat)).reshape(U.shape[0],len(t['dn']),8,3)
            dg=torch.einsum('beia,beic->beca',weights*t['dn'][None,:,:,:],du)
            g_norm=((g+dg)/t['grad_scale']).clamp(-25,25)
        mix=self.ip_mixer(torch.cat((hip,g_norm.flatten(2)),dim=-1))
        LE=LE+self.cap*torch.tanh(self.strain_head(mix))*t['lestd']
        S=S+self.cap*torch.tanh(self.stress_head(mix))*t['sstd']
        return U,LE,S


def get_model(st,device,arm='local',hidden=64,cap=.5):
    net=EndToEndUnified(st['ck'],device,arm,hidden,cap).to(device)
    net.backbone.load_state_dict(st['ck']['model_state_dict'],strict=True)
    return net


def ck_path(root,arm,seg,m):
    return root/'results'/ROOT_NAME/('B_'+arm)/seg/('member_%02d'%m)/'unified.pt'


def load_candidate(root,st,device,arm):
    p=ck_path(root,arm,st['seg'],st['member'])
    if not p.is_file():raise FileNotFoundError('Missing model: '+str(p))
    checkpoint=torch.load(p,map_location='cpu',weights_only=False) # trusted local .pt only
    for k,v in {'arch':ARCH,'arm':arm,'seg':st['seg'],'member':st['member'],
                'source_sha':st['source_sha']}.items():
        if checkpoint.get(k)!=v:raise RuntimeError('Checkpoint provenance mismatch: '+k)
    net=get_model(st,device,arm,checkpoint['hidden'],checkpoint['cap'])
    net.load_state_dict(checkpoint['state_dict'],strict=True)
    return net.eval(),checkpoint


def predict_numpy(net,t,indices,batch=4):
    net.eval();outs=[[],[],[]]
    with torch.no_grad():
        for sl in range(0,len(indices),batch):
            ids=np.array(indices[sl:sl+batch],dtype=np.int64)
            values=net(ids,t)
            for target,p in zip(outs,values):target.append(p.detach().cpu().numpy())
    return tuple(np.concatenate(parts,axis=0) for parts in outs)


def baseline_numpy(st,ids,device,batch=4):
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    t=st['t'];model=st['backbone'].eval();outputs=[[],[],[]]
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            ii=np.asarray(ids[start:start+batch],np.int64)
            b=t['branch'][ii]
            U=model.displacement_operator(b,t['node'])*t['ustd']+t['umean']
            U=apply_hard_compatibility_7region(U,t['physical_branch'][ii],t['interface'])
            L=model.strain_operator(b,t['ip'])*t['lestd']+t['lemean']
            S=model.stress_operator(b,t['ip'])*t['sstd']+t['smean']
            for dst,tensor in zip(outputs,(U,L,S)):dst.append(tensor.detach().cpu().numpy())
    return tuple(np.concatenate(a) for a in outputs)


def metrics(pred,ref):
    U,LE,S=pred;u,l,s=ref
    ans={'U_vector_pct':co.relative_l2(U,u),'U2_pct':co.relative_l2(U[...,1],u[...,1]),
         'LE_vector_pct':co.relative_l2(LE,l),'S_vector_pct':co.relative_l2(S,s)}
    for i,c in enumerate(COMP):
        ans['LE'+c+'_pct']=co.relative_l2(LE[...,i],l[...,i])
        ans['S'+c+'_pct']=co.relative_l2(S[...,i],s[...,i])
    return ans


def reference(st,ids):
    ds=st['ds'];return (ds['U'][ids],ds['LE'][ids],ds['S'][ids])


def objective(v):
    # Existing internal tune set is not independent, used only to select weights.
    return float(v['S22_pct']+.45*v['LE22_pct']+2*v['U2_pct']+.08*v['S11_pct']+.06*v['LE11_pct'])


def boundary_error(U,st,case_ids,device):
    from src.hard_interface_compatibility_7region import reconstruct_interface_target
    t=st['t'];ctx=t['interface'];bp=t['physical_branch'][case_ids]
    bound=0.
    with torch.no_grad():
        for side in ('left','right'):
            if ctx[side] is None:continue
            pred=as_tensor(U,device)[:,ctx[side+'_local_indices'],:]
            ref=reconstruct_interface_target(bp,ctx[side],ctx[side+'_slice'])
            bound=max(bound,float((pred-ref).abs().max().detach().cpu()))
    return bound


def source_force_equal(net,st):
    base=st['ck']['model_state_dict'];now=net.backbone.state_dict()
    keys=[k for k in base if k.startswith('force_coefficient_head.')]
    return bool(keys and all(torch.equal(base[k].detach().cpu(),now[k].detach().cpu()) for k in keys))
