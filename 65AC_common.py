#!/usr/bin/env python3
"""Step 65AC shared utilities: LGNO-inspired C3D8R *gradient estimator*.

Design reference: https://github.com/baiming-zhang/LGNO,
'2D Navier-Stokes/_lgno_impl.py' class VectorStencilLGNO.
This is an ORIGINAL independent FEM-element adaptation, not a verbatim copy.
The source predicts a PDE update on periodic grids; here an MLP predicts
non-periodic C3D8R element stencil weights for local displacement gradients.

IMPORTANT: The corrected gradient is a learned estimate and is NOT necessarily
an integrable gradient of the unchanged displacement field. Do not replace
Abaqus strains or deploy in the FE-NO coupling without additional verification.
"""
import json, sys, hashlib
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
AAFILE=HERE/'65AA_common.py'
if not AAFILE.is_file():
    raise FileNotFoundError('Missing existing 65AA_common.py in project root: '+str(AAFILE))
spec=spec_from_file_location('65ac_existing_aa',AAFILE)
aa=module_from_spec(spec);sys.modules[spec.name]=aa;spec.loader.exec_module(aa)

SEGMENTS=('inner_left','inner_right')
ROOT_NAME='65AC_local_stencil'
ARCH='65AC_c3d8r_lgno_element_v1'
ARMS=('pointwise','lgno')
ORDER=('11','22','33','12','13','23')


def source_path(root,seg,member):
    return root/'results'/'65AA_gradient_learning'/'B_gradient'/seg/('member_%02d'%member)/'best_v4_7region.pt'


def make_geometry(ds):
    xyz=np.asarray(ds['coordinates'],dtype=np.float64)
    conn=np.asarray(ds['element_node_indices'],dtype=np.int64)
    pos=xyz[conn]                      # [E,8,3]
    center=pos.mean(axis=1,keepdims=True)
    half_range=np.maximum(np.ptp(pos,axis=1,keepdims=True),1e-6)
    offsets=(pos-center)/half_range
    dN=aa.element_dN_numpy(ds).astype(np.float64)
    # Check derivative consistency before fitting any learned stencil.
    if np.max(np.abs(dN.sum(axis=1)))>1e-5:
        raise RuntimeError('C3D8R derivative does not annihilate constants')
    identity=np.einsum('eia,eib->eab',pos,dN,optimize=True)
    err=float(np.max(np.abs(identity-np.eye(3)[None,:,:])))
    if err>5e-3:raise RuntimeError('C3D8R gradient fails affine consistency: %.6g'%err)
    xyz_range=np.maximum(xyz.max(axis=0)-xyz.min(axis=0),1e-6)
    loc=((center[:,0,:]-xyz.min(axis=0))/xyz_range*2-1)
    return dict(conn=conn,dN=dN.astype(np.float32),
                offset=offsets.astype(np.float32),position=loc.astype(np.float32),
                step=half_range[:,0,:].astype(np.float32),affine_error=err)


def gradient_from_u(U,ds):
    return aa.gradient_numpy(U,ds).astype(np.float32)


def logstrain_from_gradient(grad):
    """NumPy material Hencky strain at element centers, engineering shear.

    C=F.T F; LE=.5 log(C), with 12/13/23 exported as engineering shear.
    This is a *diagnostic*: learned gradient may not integrate to a displacement.
    """
    g=np.asarray(grad,dtype=np.float64)
    if g.shape[-2:]!=(3,3):raise ValueError('Gradient last dimensions must be 3x3')
    F=np.eye(3)+g
    C=np.swapaxes(F,-2,-1)@F
    vals,Q=np.linalg.eigh(C)
    if not np.isfinite(vals).all() or np.min(vals)<=1e-10:
        raise RuntimeError('Non-positive / invalid right Cauchy-Green tensor')
    lC=(Q*np.log(vals)[...,None,:])@np.swapaxes(Q,-2,-1)
    E=.5*lC
    return np.stack((E[...,0,0],E[...,1,1],E[...,2,2],
                     2*E[...,0,1],2*E[...,0,2],2*E[...,1,2]),axis=-1)


def rel2(x,y):return aa.co.relative_l2(x,y)


def make_features_dimensions():
    # 24 local displacement deltas + 9 gradients + 24 offsets
    # + 24 scaled shape gradients + 3 local positions + 3 material inputs.
    return 87


class LocalStencilGradientOperator(nn.Module):
    """Non-periodic FEM counterpart of LGNO's 'local MLP -> kernel -> contraction'.

    Input: hard-corrected source displacement U at all region nodes.
    Output: estimated element-center gradient [batch,elements,3 components,3 axes].
    The original C3D8R gradient is preserved at initialization. LGNO arm adds
    learned state-dependent derivative weights applied to LOCAL DIFFERENCES;
    gradients of constant displacement remain exactly zero.

    The 'pointwise' control uses the same input features but directly outputs
    a gradient correction; its learned correction need not be zero for constants.
    """
    def __init__(self,arm,hidden=64,coeff_limit=.8):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown arm '+arm)
        self.arm=arm;self.hidden=int(hidden);self.coeff_limit=float(coeff_limit)
        n_out=24 if arm=='lgno' else 9
        self.mlp=nn.Sequential(nn.Linear(make_features_dimensions(),hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,n_out))
        # EXACT original C3D8R estimate at init (both arms).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,U,materials,context):
        # U [B,N,3] in mm, materials [B,3] normalized E1/E2/G12
        con=context['conn'];dn=context['dN']
        local=U[:,con,:]                       # [B,E,8,3]
        du=local-local.mean(dim=2,keepdim=True)
        base=torch.einsum('beic,eij->becj',local,dn)
        B,E=U.shape[0],dn.shape[0]
        ustd=context['ustd'].view(1,1,1,3)
        gstd=context['gstd'].view(1,1,3,3)
        dw=(du/ustd).clamp(-20,20).reshape(B,E,24)
        gw=(base/gstd).clamp(-20,20).reshape(B,E,9)
        off=context['offset'].reshape(1,E,24).expand(B,-1,-1)
        scaled_dn=(dn*context['step'][:,None,:]).reshape(1,E,24).expand(B,-1,-1)
        pos=context['position'].reshape(1,E,3).expand(B,-1,-1)
        mats=materials[:,:3].reshape(B,1,3).expand(-1,E,-1).clamp(-10,10)
        feat=torch.cat((dw,gw,off,scaled_dn,pos,mats),dim=-1)
        if feat.shape[-1]!=make_features_dimensions():raise RuntimeError('Feature dimension mismatch')
        weights=self.mlp(feat)
        if self.arm=='lgno':
            # LGNO-style nonlinear stencil coefficients times neighboring values.
            multipliers=self.coeff_limit*torch.tanh(weights.reshape(B,E,8,3))
            delta_w=multipliers*dn.reshape(1,E,8,3)
            correction=torch.einsum('beia,beic->beca',delta_w,du)
        else:
            correction=self.coeff_limit*torch.tanh(weights.reshape(B,E,3,3))*gstd
        return base+correction


def tensor_context(ds,ck,geometry,dev,train_ids):
    gtrain=gradient_from_u(ds['U'][train_ids],ds)
    gs=np.std(gtrain,axis=(0,1))
    scale=max(float(gs.max()),1e-5)
    gs=np.maximum(gs,0.05*scale).astype(np.float32)
    ustd=np.asarray(ck['U_std'],dtype=np.float32)
    if not np.all(np.isfinite(ustd)) or np.min(ustd)<=0:raise RuntimeError('Bad checkpoint U_std')
    def t(arr,dtype=torch.float32):
        return torch.tensor(np.array(arr,copy=True),dtype=dtype,device=dev)
    return dict(conn=t(geometry['conn'],torch.long),dN=t(geometry['dN']),
                offset=t(geometry['offset']),position=t(geometry['position']),
                step=t(geometry['step']),ustd=t(ustd),gstd=t(gs)),gs


def prepare(root,segment,member,device,need_predictions=True):
    # Reuse the exact original data normalization and hard correction.
    ds,base_ck,_,source_ctx,base_path=aa.init_case(root,segment,member,device)
    new_path=source_path(root,segment,member)
    if not new_path.is_file():raise FileNotFoundError('Run 65AA-B gradient first: '+str(new_path))
    src_ck=aa.co.load_checkpoint(new_path)
    info=src_ck.get('65AA_gradient_experiment',{})
    if info.get('arm')!='gradient' or info.get('initialization_sha256')!=aa.co.sha256(base_path):
        raise RuntimeError('65AA gradient model is not based on current 65Z source')
    if not aa.frozen_heads_equal(base_ck,src_ck):
        raise RuntimeError('65AA candidate changed LE/S/force heads: '+str(new_path))
    for key in ('branch_mean','branch_std','coordinate_min','coordinate_max','U_mean','U_std'):
        if not np.allclose(base_ck[key],src_ck[key],atol=1e-8,rtol=1e-8):
            raise RuntimeError('65AA candidate changed input/output normalization '+key)
    geometry=make_geometry(ds)
    ctxt,gs=tensor_context(ds,src_ck,geometry,device,ds['splits']['train'])
    bnorm,_,_=aa.co.scaled_inputs(src_ck,ds)
    materials=torch.tensor(np.array(bnorm,copy=True),dtype=torch.float32,device=device)
    store=dict(ds=ds,ck=src_ck,ctx=ctxt,gs=gs,mats=materials,geometry=geometry,
               source_sha=aa.co.sha256(new_path),source_path=new_path)
    if need_predictions:
        idx=np.concatenate((ds['splits']['train'],ds['splits']['validation']))
        src_model=aa.co.make_model(src_ck,device).eval()
        pred=aa.predict_numpy(src_model,source_ctx,idx,batch_size=4,correct=True)
        store['indices']=idx
        store['pred']=pred.astype(np.float32)
        store['index_map']={int(c):j for j,c in enumerate(idx)}
        del src_model
        if device.type=='mps':torch.mps.empty_cache()
    return store


def checkpoint_path(root,arm,seg,member):
    return root/'results'/ROOT_NAME/('B_'+arm)/seg/('member_%02d'%member)/'local_stencil.pt'


def load_adapter(path,arm,seg,member,source_sha,device):
    if not path.is_file():raise FileNotFoundError(str(path))
    ck=torch.load(path,map_location='cpu',weights_only=False)  # TRUSTED local checkpoint only
    for key,value in [('architecture',ARCH),('arm',arm),('segment',seg),('member',member),('source_sha256',source_sha)]:
        if ck.get(key)!=value:raise RuntimeError('Adapter metadata mismatch %s in %s'%(key,path))
    mod=LocalStencilGradientOperator(arm,int(ck['hidden']),float(ck['coeff_limit'])).to(device)
    mod.load_state_dict(ck['model_state_dict'],strict=True)
    mod.eval()
    return mod,ck


def split_predict(store,model,indices,device,batch=4):
    res=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch):
            ids=indices[start:start+batch]
            pos=np.array([store['index_map'][int(j)] for j in ids],dtype=np.int64)
            u=torch.tensor(np.array(store['pred'][pos],copy=True),device=device,dtype=torch.float32)
            mats=store['mats'][ids]
            if model is None:
                arr=torch.einsum('beic,eij->becj',u[:,store['ctx']['conn'],:],store['ctx']['dN'])
            else:
                arr=model(u,mats,store['ctx'])
            res.append(arr.detach().cpu().numpy())
    return np.concatenate(res)


def score(grad,truth_grad,LE_ref=None):
    out={'gradient_rel_l2_pct':rel2(grad,truth_grad),
         'gradient_rmse':float(np.sqrt(np.mean((np.asarray(grad)-truth_grad)**2)))}
    for comp in range(3):
        for axis in range(3):
            key='dU%d_d%s'%(comp+1,'XYZ'[axis])
            out[key+'_rel_l2_pct']=rel2(grad[...,comp,axis],truth_grad[...,comp,axis])
    if LE_ref is not None:
        L=logstrain_from_gradient(grad)
        for idx,name in ((0,'LE11'),(1,'LE22'),(3,'LE12')):
            out['logU_'+name+'_rel_l2_pct']=rel2(L[...,idx],LE_ref[...,idx])
    return out


def safety(candidate,baseline):
    # Diagnostic safety gate (NO displacement ever changes in 65AC).
    return (candidate['gradient_rel_l2_pct']<=baseline['gradient_rel_l2_pct']*1.01 and
            candidate['logU_LE11_rel_l2_pct']<=max(baseline['logU_LE11_rel_l2_pct']*1.10,
                                                  baseline['logU_LE11_rel_l2_pct']+.2) and
            candidate['logU_LE22_rel_l2_pct']<=max(baseline['logU_LE22_rel_l2_pct']*1.02,
                                                  baseline['logU_LE22_rel_l2_pct']+.2))
