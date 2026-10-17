#!/usr/bin/env python3
"""Shared, read-only geometry/checkpoint helpers for the Step 65Z Mac experiment.

All finite-strain kinematics use original-reference coordinates and the original
64J C3D8R center-point derivative matrix. Engineering shear is returned in the
same [11,22,33,12,13,23] convention supported by 65Y-A's Abaqus audit.

The kinematic target is a DIAGNOSTIC from a *frozen, hard-corrected predicted U*.
It is not silently presumed accurate merely because reference-FEM U passes.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SEGMENTS=('inner_left','inner_right')
COMPONENTS=('11','22','33','12','13','23')
WEIGHTS=np.array([1.,2.5,1.,2.,1.,1.],dtype=np.float32)
FREEZE=('displacement_operator.','force_coefficient_head.')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as fp:
        for chunk in iter(lambda:fp.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def device_from_option(option):
    if option=='auto':return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if option=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but not available; use --device cpu')
    return torch.device(option)


def load_checkpoint(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    # Trusted user-generated PyTorch checkpoint ONLY. Do not load untrusted .pt.
    ck=torch.load(path,map_location='cpu',weights_only=False)
    needed=('model_state_dict','branch_dim','number_force_coefficients','branch_mean',
            'branch_std','coordinate_min','coordinate_max','U_mean','U_std',
            'LE_mean','LE_std','S_mean','S_std')
    absent=[k for k in needed if k not in ck]
    if absent:raise RuntimeError('%s missing checkpoint keys: %s'%(path,absent))
    return ck


def make_model(ck,device):
    from src.hybrid_bulk_operator_v4_7region import HybridBulkOperatorV4SevenRegion
    net=HybridBulkOperatorV4SevenRegion(branch_dim=int(ck['branch_dim']),
        number_force_coefficients=int(ck['number_force_coefficients']),
        hidden_dim=int(ck.get('hidden_dim',128)),
        latent_dim=int(ck.get('latent_dim',128)),depth=int(ck.get('depth',4)))
    net.load_state_dict(ck['model_state_dict'],strict=True)
    return net.to(device)


def segment_data(root,seg):
    path=root/'data'/'hybrid_bulk_7region'/seg
    need=['branch_inputs.npy','coordinates.npy','ip_coordinates.npy',
          'element_node_indices.npy','B_center.npy','LE.npy','S.npy',
          'split_assignment.csv','metadata.json']
    missing=[str(path/f) for f in need if not (path/f).is_file()]
    if missing:raise FileNotFoundError('Missing segment files:\n'+'\n'.join(missing))
    x={key:np.load(path/(key+'.npy'),allow_pickle=False) for key in
       ('branch_inputs','coordinates','ip_coordinates','element_node_indices','B_center','LE','S')}
    split=pd.read_csv(path/'split_assignment.csv')
    if not {'Index','Split'}.issubset(split.columns):raise RuntimeError('Invalid split table')
    x['splits']={s:split.loc[split.Split==s,'Index'].to_numpy(dtype=int)
                  for s in ('train','validation','test')}
    n=len(x['branch_inputs']);all_ix=np.concatenate(list(x['splits'].values()))
    if np.unique(all_ix).size != n or sorted(all_ix.tolist())!=list(range(n)):
        raise RuntimeError('Training, validation and test indices overlap/omit cases')
    xyz=x['coordinates'];ip=x['ip_coordinates'];conn=x['element_node_indices'];B=x['B_center']
    if xyz.ndim!=2 or xyz.shape[1]!=3 or ip.ndim!=2 or ip.shape[1]!=3:
        raise RuntimeError('Expected Nx3 coordinates')
    if conn.shape!=(len(ip),8) or B.shape!=(len(ip),6,24):
        raise RuntimeError('Unexpected C3D8R element shape')
    if np.min(conn)<0 or np.max(conn)>=len(xyz):raise RuntimeError('Connectivity indexes outside nodes')
    if x['LE'].shape!=(n,len(ip),6) or x['S'].shape != x['LE'].shape:
        raise RuntimeError('Target shapes inconsistent')
    with (path/'metadata.json').open() as f:x['metadata']=json.load(f)
    if any(len(v)<1 for v in x['splits'].values()):raise RuntimeError('Empty data split')
    return x


def scaled_inputs(ck,ds):
    br=ds['branch_inputs'].astype(np.float32)
    bm=np.asarray(ck['branch_mean'],dtype=np.float32)
    bs=np.asarray(ck['branch_std'],dtype=np.float32)
    cm=np.asarray(ck['coordinate_min'],dtype=np.float32)
    cx=np.asarray(ck['coordinate_max'],dtype=np.float32)
    if bs.shape!=(br.shape[1],) or bm.shape!=bs.shape or np.any(bs<=0):
        raise RuntimeError('Branch normalization incompatible')
    if not np.allclose(br[ds['splits']['train']].mean(0),bm,rtol=5e-4,atol=2e-3):
        raise RuntimeError('Branch mean differs from original checkpoint training split')
    den=np.maximum(cx-cm,1e-8)
    node=2*(ds['coordinates'].astype(np.float32)-cm)/den-1
    ip=2*(ds['ip_coordinates'].astype(np.float32)-cm)/den-1
    if max(np.max(abs(node)),np.max(abs(ip)))>1.002:
        raise RuntimeError('Coordinate normalization outside original segment range')
    return (br-bm)/bs,node,ip


def relative_l2(pred,truth):
    p=np.asarray(pred,dtype=np.float64);r=np.asarray(truth,dtype=np.float64)
    return float(100*np.linalg.norm((p-r).ravel())/max(np.linalg.norm(r.ravel()),1e-14))


def log_material_strain_eng(displacement,connectivity,B):
    """From nodal U: F=I+grad_X U; E=.5*log(F.T@F); shear doubled.

    Input U: [nodes,3] and per-element C3D8R B_center in original coordinates.
    Returns [elements,6], in engineering shear convention of the 65Y audit.
    Implements SPD log using np.linalg.eigh (CPU double precision, no autograd).
    """
    U=np.asarray(displacement,dtype=np.float64)
    co=np.asarray(connectivity,dtype=np.int64)
    B=np.asarray(B,dtype=np.float64)
    ue=U[co]
    dn=np.stack((B[:,0,0::3],B[:,1,1::3],B[:,2,2::3]),axis=2)
    gradient=np.einsum('eai,eaj->eij',ue,dn)
    F=np.eye(3)[None,:,:]+gradient
    c=np.einsum('eji,ejk->eik',F,F)
    values,vectors=np.linalg.eigh(c)
    if not np.isfinite(values).all() or np.min(values)<=1e-12:
        raise RuntimeError('Non-positive C=F.T F; check displacement and element ordering')
    logc=np.einsum('eik,ek,ejk->eij',vectors,np.log(values),vectors)
    e=.5*logc
    return np.stack([e[:,0,0],e[:,1,1],e[:,2,2],2*e[:,0,1],2*e[:,0,2],2*e[:,1,2]],axis=-1).astype(np.float32)


def orthotropic_stiffness():
    E1,E2,E3=46500.,12500.,12000.
    v12,v13,v23=.28,.28,.40
    G12,G13,G23=4750.,4500.,3500.
    a=np.zeros((6,6),dtype=np.float64)
    a[np.diag_indices(6)]=[1/E1,1/E2,1/E3,1/G12,1/G13,1/G23]
    a[0,1]=a[1,0]=-v12/E1
    a[0,2]=a[2,0]=-v13/E1
    a[1,2]=a[2,1]=-v23/E2
    C=np.linalg.inv(a)
    if np.linalg.eigvalsh(C).min()<=0:raise RuntimeError('Material stiffness not SPD')
    return C.astype(np.float32)



def orthotropic_stiffness_batch_numpy(materials):
    """Batch C(E1,E2,G12) using ORIGINAL 64J branch columns, not fixed middle E.

    64J puts [E1_MPa,E2_MPa,G12_MPa] in first three branch inputs.
    E3, nu12, nu13, nu23, G13, G23 fixed by original 64N training code.
    """
    mat=np.asarray(materials,dtype=np.float64)
    if mat.ndim!=2 or mat.shape[1]!=3 or not np.isfinite(mat).all() or np.any(mat<=0):
        raise RuntimeError('Expected physically positive [E1_MPa,E2_MPa,G12_MPa] cases')
    n=len(mat)
    E1,E2,G12=mat[:,0],mat[:,1],mat[:,2]
    Q=np.zeros((n,6,6),dtype=np.float64)
    Q[:,0,0]=1/E1;Q[:,1,1]=1/E2;Q[:,2,2]=1/12000.
    Q[:,0,1]=Q[:,1,0]=-.28/E1
    Q[:,0,2]=Q[:,2,0]=-.28/E1
    Q[:,1,2]=Q[:,2,1]=-.40/E2
    Q[:,3,3]=1/G12;Q[:,4,4]=1/4500.;Q[:,5,5]=1/3500.
    if np.any(np.linalg.eigvalsh(Q) <= 0):
        raise RuntimeError('Training material compliance not SPD; inspect branch inputs')
    return np.linalg.inv(Q).astype(np.float32)

def interface_context(root,ds,device):
    from src.hard_interface_compatibility_7region import build_hard_boundary_context_7region
    meta=ds['metadata']
    return build_hard_boundary_context_7region(coordinates=ds['coordinates'],
        interface_dir=str(root/'data'/'candidate_partition_interfaces_7region'),
        left_interface_name=meta['left_interface'],
        right_interface_name=meta['right_interface'],device=device)


def corrected_U(model,ck,ds,root,branch_normalized,node_normalized,device,batch=4):
    """The SAME hard-interface correction used by original 64N training."""
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    ctx=interface_context(root,ds,device)
    uma=np.asarray(ck['U_mean'],np.float32)
    ust=np.asarray(ck['U_std'],np.float32)
    bma=np.asarray(ck['branch_mean'],np.float32)
    bst=np.asarray(ck['branch_std'],np.float32)
    coords_t=torch.as_tensor(node_normalized,dtype=torch.float32,device=device)
    up=torch.as_tensor(uma,dtype=torch.float32,device=device).reshape(1,1,3)
    us=torch.as_tensor(ust,dtype=torch.float32,device=device).reshape(1,1,3)
    bm=torch.as_tensor(bma,dtype=torch.float32,device=device).reshape(1,-1)
    bs=torch.as_tensor(bst,dtype=torch.float32,device=device).reshape(1,-1)
    model.eval();outputs=[]
    with torch.no_grad():
        for start in range(0,len(branch_normalized),batch):
            b=torch.as_tensor(branch_normalized[start:start+batch],dtype=torch.float32,device=device)
            u=model.displacement_operator(b,coords_t)*us+up
            hard=apply_hard_compatibility_7region(raw_displacement=u,
                branch_physical=b*bs+bm,context=ctx)
            outputs.append(hard.detach().cpu().numpy())
    return np.concatenate(outputs,axis=0)


def reference_audit_gate(root):
    folder=root/'results'/'65Y_middle_mechanics'/'A_reference_physics_audit'
    f=folder/'65Y_A_kinematics_and_stress_comparison.csv'
    if not f.is_file():raise FileNotFoundError('Required completed reference audit: '+str(f))
    d=pd.read_csv(f)
    rows=d[(d['region'].isin(['NO_L','NO_R'])) &
           (d['method']=='logU_material_engineering') &
           (d['quantity'].isin(['LE11','LE22','LE12']))]
    if len(rows)!=6 or rows['relative_l2_pct'].isna().any():
        raise RuntimeError('Incomplete logarithmic-strain reference audit')
    worst=float(rows['relative_l2_pct'].max())
    if worst>=0.5:
        raise RuntimeError('Reference logU-mechanics audit failed: maximum relative L2 %.4f%%'%worst)
    return worst
