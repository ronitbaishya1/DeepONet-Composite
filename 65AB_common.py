#!/usr/bin/env python3
"""Step 65AB: a zero-initialized, boundary-vanishing displacement residual.

All diagnostics use the original 64J train/validation split. The original
hard-corrected 65Z displacement is the *unchanged* baseline; a small new
operator only learns interior corrections, zero at the region interfaces.

u_new(X; p) = hard_corrected_u_65Z(X; p) + 4t(1-t)*U_std*R_theta(p, X).

This is an offline architecture candidate, NOT a drop-in replacement for 64V.
"""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import sys
import numpy as np
import torch
from torch import nn

spec=spec_from_file_location('step65aa_existing',str(Path(__file__).with_name('65AA_common.py')))
aa=module_from_spec(spec);sys.modules[spec.name]=aa;spec.loader.exec_module(aa)
ROOT_NAME='65AB_interface_lift_residual'
SEGMENTS=aa.SEGMENTS
ARCHITECTURE='65AB_zero_initialized_interior_bubble_v1'


def envelope_numpy(ds):
    x=np.asarray(ds['coordinates'][:,0],dtype=np.float64)
    a,b=float(x.min()),float(x.max())
    if b-a < 1e-6: raise RuntimeError('Zero-length NO region')
    t=np.clip((x-a)/(b-a),0,1)
    p=4*t*(1-t)
    ends=np.isclose(x,a,atol=1e-6,rtol=0)|np.isclose(x,b,atol=1e-6,rtol=0)
    p[ends]=0.0
    if not np.any(~ends):raise RuntimeError('No interior nodes for bubble correction')
    return p.astype(np.float32), np.flatnonzero(ends)


def boundary_local_indices(ctx,ds):
    """Ensure only *actual* known interfaces are checked (one-sided outer regions)."""
    out=[]
    for side in ('left','right'):
        if ctx['context'].get(side) is not None:
            key=side+'_local_indices'
            if ctx['context'].get(key) is None:
                raise RuntimeError('Hard-interface context lacks '+key)
            indices = ctx['context'][key]
            if isinstance(indices, torch.Tensor):
                indices = indices.detach().cpu().numpy()
            out.extend(np.asarray(indices, dtype=np.int64).tolist())
    if not out:
        raise RuntimeError('Expected at least one prescribed interface')
    ids=np.unique(out)
    env,_=envelope_numpy(ds)
    if np.any(np.abs(env[ids])>1e-7):
        raise RuntimeError('Prescribed interface not on zero envelope; check coordinates')
    return ids


class BubbleOperator(nn.Module):
    """Uses the *original repository* VectorDeepONet for the correction."""
    def __init__(self,branch_dim,hidden_dim=64,latent_dim=48,depth=3):
        super().__init__()
        from src.hybrid_bulk_operator_v4_7region import VectorDeepONet
        self.net=VectorDeepONet(branch_dim=branch_dim,output_dim=3,
                                hidden_dim=hidden_dim,latent_dim=latent_dim,depth=depth)
        # Exact zero output at initialization: no sudden displacement change.
        last=self.net.branch_network.network[-1]
        if not isinstance(last,nn.Linear):
            raise RuntimeError('Original branch network layout has changed')
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        nn.init.zeros_(self.net.bias)

    def forward(self,branch_n,node_coords_n):
        return self.net(branch_n,node_coords_n)


def make_operator(dim,device,hidden_dim=64,latent_dim=48,depth=3,seed=None):
    if seed is not None:torch.manual_seed(int(seed))
    return BubbleOperator(dim,hidden_dim,latent_dim,depth).to(device)


def corrected_from_residual(base_u,operator,ctx,case_ids,phi):
    """Differentiable correction. base_u is a detached precomputed 65Z value."""
    branch=ctx['branch'][case_ids]
    r=operator(branch,ctx['nodes'])
    return base_u + phi.reshape(1,-1,1)*r*ctx['ustd']


def score_dict(u,ref,ds):
    grad=aa.gradient_numpy(u,ds)
    gt=aa.gradient_numpy(ref,ds)
    d=aa.errors(u,ref,ds,LE_ref=None)
    return d


def candidate_dir(root,arm,seg,member):
    return root/'results'/ROOT_NAME/('B_'+arm)/seg/('member_%02d'%member)


def load_candidate(root,arm,seg,member,ck,device):
    fp=candidate_dir(root,arm,seg,member)/'bubble_residual.pt'
    if not fp.is_file():raise FileNotFoundError(str(fp))
    payload=torch.load(fp,map_location='cpu',weights_only=False)  # trusted local checkpoint
    if payload.get('architecture')!=ARCHITECTURE or payload.get('segment')!=seg or payload.get('member')!=member:
        raise RuntimeError('Wrong 65AB candidate metadata: '+str(fp))
    source=aa.source_checkpoint(root,seg,member)
    if payload['source_sha256']!=aa.co.sha256(source):
        raise RuntimeError('Original 65Z model changed since 65AB training: '+str(source))
    net=make_operator(ck['branch_dim'],device,payload['hidden_dim'],
                      payload['latent_dim'],payload['depth'])
    net.load_state_dict(payload['residual_model_state_dict'],strict=True)
    net.eval()
    return net,payload


def predictions_from_candidate(root,seg,member,ds,ck,ctx,net,indices,batch_size=4):
    """Identical 65Z baseline + zero-at-interface network correction."""
    from numpy import asarray
    device=next(net.parameters()).device
    env,_=envelope_numpy(ds)
    phi=torch.as_tensor(env,dtype=torch.float32,device=device)
    base=[];new=[]
    # The source model must be reloaded for exact independent verification.
    source_model=aa.co.make_model(ck,device).eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            ids=asarray(indices[start:start+batch_size],dtype=np.int64)
            u0=aa.predict_torch(source_model,ctx,ids,correct=True)
            u1=corrected_from_residual(u0,net,ctx,ids,phi)
            base.append(u0.detach().cpu().numpy())
            new.append(u1.detach().cpu().numpy())
    return np.concatenate(base),np.concatenate(new)
