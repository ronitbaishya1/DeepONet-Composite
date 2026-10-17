#!/usr/bin/env python3
"""65AM-B shared implementation: TRAIN-subset-only FE/POD residual operator.

Scope: 64J inner_left / inner_right. Does not modify 65V, 65AI, checkpoints,
or any original files. This is supervised FE kinematics, NOT a PDE-equilibrium
or fully constitutive physics-informed solver.
"""
from __future__ import annotations
import hashlib
import sys
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

AA = importlib.import_module('65AA_common')
CO = AA.co
REGIONS = ('inner_left', 'inner_right')
ROOT_NAME = '65AM_B_FE_residual_operator'
SEED = 65012026
RANKS = (8, 16, 24, 32, 40)


def sha(path):
    return CO.sha256(Path(path))


def output_dir(root, step):
    d = root/'results'/ROOT_NAME/step
    if d.exists() and any(d.iterdir()):
        raise RuntimeError('OUTPUT PROTECTED (already exists). Do not overwrite. Inspect it in Finder:\n' + str(d))
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_train(ds):
    ids = np.asarray(ds['splits']['train'], dtype=np.int64)
    if len(ids) != 56:
        raise RuntimeError('Expected 56 original 64J TRAIN cases; found %d. Stop rather than changing splits.' % len(ids))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(ids)
    fit, tune = np.sort(perm[:44]), np.sort(perm[44:])
    if len(np.intersect1d(fit, tune)) or len(fit)!=44 or len(tune)!=12:
        raise RuntimeError('Invalid 44/12 internal split')
    forbidden = np.r_[ds['splits']['validation'], ds['splits']['test']]
    if len(np.intersect1d(np.r_[fit,tune], forbidden)):
        raise RuntimeError('Data leakage: original validation or test case included')
    return fit, tune


def load_region(root, region, *, baseline_required=True):
    if region not in REGIONS:
        raise ValueError(region)
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    ds = AA.load_data(root,region)
    ds['_root'] = root
    fit, tune = split_train(ds)
    if len(ds['coordinates']) < 50 or not np.isfinite(ds['U']).all():
        raise RuntimeError('Unexpected geometry or invalid U')
    conn = np.asarray(ds['element_node_indices'],np.int64)
    dN = AA.element_dN_numpy(ds).astype(np.float64)
    xyz = np.asarray(ds['coordinates'],np.float64)
    affine = np.einsum('eai,eaj->eij', xyz[conn], dN)
    affine_error=float(np.max(np.abs(affine-np.eye(3)[None,:,:])))
    if affine_error>5e-3:raise RuntimeError('C3D8R affine-gradient audit failed: %.6g' % affine_error)
    src = root/'results'/'65AA_gradient_learning'/'B_gradient'/region/'member_01'/'best_v4_7region.pt'
    if baseline_required and not src.is_file():
        raise FileNotFoundError('65AA frozen baseline checkpoint missing: '+str(src))
    return ds,fit,tune,src,affine_error


def boundary_indices(ctx):
    idx=[]
    for name in ('left_local_indices','right_local_indices'):
        if ctx[name] is not None:
            idx.append(ctx[name].detach().cpu().numpy().astype(np.int64))
    if not idx:raise RuntimeError('No fixed interfaces')
    return np.unique(np.concatenate(idx))


def build_lift(root,ds,ids,*,device='cpu'):
    # The same hard compatibility routine used in the trained hybrid operator.
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    dev=torch.device(device)
    ctx=CO.interface_context(root,ds,dev)
    fixed=boundary_indices(ctx)
    n=len(ds['coordinates'])
    buf=[]
    with torch.no_grad():
        for k in range(0,len(ids),8):
            cases=np.asarray(ids[k:k+8],np.int64)
            b=torch.as_tensor(np.array(ds['branch_inputs'][cases],copy=True),device=dev,dtype=torch.float32)
            zeros=torch.zeros((len(cases),n,3),device=dev,dtype=torch.float32)
            lifted=apply_hard_compatibility_7region(zeros,b,ctx)
            buf.append(lifted.detach().cpu().numpy())
    out=np.concatenate(buf).astype(np.float64)
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite interface lift')
    return out,fixed


def scales_from_fit(Ufit, Gfit, residual_fit,free):
    # Component scales computed ONLY on fitting cases. Avoid extreme small denominators.
    sU=np.maximum(np.sqrt(np.mean(Ufit**2,axis=(0,1))),1e-4)
    sG=np.maximum(np.sqrt(np.mean(Gfit**2,axis=(0,1))),1e-4)
    sR=np.maximum(np.std(residual_fit[:,free,:],axis=(0,1)),1e-4)
    return sU,sG,sR


def build_basis(root,region):
    ds,fit,tune,source,aff=load_region(root,region)
    all_ids=np.r_[fit,tune]
    lift,fixed=build_lift(root,ds,all_ids)
    total=len(ds['coordinates'])
    free=np.setdiff1d(np.arange(total,dtype=np.int64),fixed)
    nfit=len(fit)
    Uf=np.asarray(ds['U'][fit],np.float64)
    Ut=np.asarray(ds['U'][tune],np.float64)
    residual_fit=Uf-lift[:nfit]
    sU,sG,sR=scales_from_fit(Uf,AA.gradient_numpy(Uf,ds),residual_fit,free)
    # Coefficients are for normalized residual field, not absolute displacement.
    X=(residual_fit[:,free,:]/sR).reshape(nfit,-1)
    _,singular,vt=np.linalg.svd(X,full_matrices=False)
    nz=int(np.sum(singular>max(float(singular[0])*1e-10,1e-10)))
    if nz<8:raise RuntimeError('Residual basis numerical rank < 8, inspect fit snapshots')
    # Matrix [rank, nodes, components], physical displacement units per coefficient.
    V=np.zeros((len(singular),total,3),np.float64)
    V[:,free,:]=vt.reshape(len(singular),len(free),3)*sR[None,None,:]
    if np.max(abs(V[:,fixed,:]))>0:raise RuntimeError('Basis does not vanish at interfaces')
    # Make displacement oracle projection in SAME whitened free-node norm used to form POD.
    report=[]
    for rank in RANKS:
        if rank>nz:continue
        for split,Ui,L in [('fit',Uf,lift[:nfit]),('tune',Ut,lift[nfit:])]:
            raw=((Ui-L)[:,free,:]/sR).reshape(len(Ui),-1)
            coeff=raw@vt[:rank].T
            pred=L+np.einsum('br,rnc->bnc',coeff,V[:rank],optimize=True)
            stats=metrics(pred,Ui,ds,ds['LE'][fit] if split=='fit' else ds['LE'][tune])
            report.append(dict(region=region,rank=rank,split=split,**stats))
    stats_df=pd.DataFrame(report)
    # Predefined capacity gate: minimum rank satisfying fit oracle representation thresholds.
    eligible=stats_df[(stats_df.split=='fit')&
           (stats_df.grad_rel_pct<=5.0)&(stats_df.U_vec_rel_pct<=1.0)&
           (stats_df.LE22_rel_pct<=10.0)]
    choice=int(eligible.sort_values('rank').iloc[0]['rank']) if len(eligible) else None
    branch=np.asarray(ds['branch_inputs'][fit],np.float64)
    bmean=branch.mean(0)
    bstd=np.maximum(branch.std(0),1e-6)
    return dict(ds=ds,fit=fit,tune=tune,fixed=fixed,free=free,lift=lift,
                V=V,sU=sU,sG=sG,sR=sR,branch_mean=bmean,branch_std=bstd,
                selected_rank=choice,metrics=stats_df,singular=singular,
                source=source,source_sha=sha(source),affine_error=aff,
                interface_fit_rmse_mm=float(np.sqrt(np.mean((Uf[:,fixed,:]-lift[:len(fit),fixed,:])**2))))


def saved_basis_path(root,region):
    return root/'results'/ROOT_NAME/'B0_basis'/region/'basis.npz'


def load_saved(root,region):
    ds,fit,tune,source,aff=load_region(root,region)
    p=saved_basis_path(root,region)
    if not p.is_file():raise FileNotFoundError('Run B0 first: '+str(p))
    with np.load(p,allow_pickle=False) as a:
        vals={k:np.array(a[k],copy=True) for k in a.files}
    if not np.array_equal(vals['fit_ids'],fit) or not np.array_equal(vals['tune_ids'],tune):
        raise RuntimeError('Saved 65AM B0 split is incompatible')
    if str(vals['source_sha'].item())!=sha(source):raise RuntimeError('Frozen baseline SHA mismatch')
    if abs(float(vals['affine_error'])-aff)>1e-5:raise RuntimeError('C3D8R geometry mismatch')
    rank=int(vals['selected_rank'])
    if rank<=0:raise RuntimeError('B0 capacity gate did NOT pass. Do not train; inspect B0 audit')
    if rank>len(vals['basis']):raise RuntimeError('Invalid selected rank')
    lift,fixed=build_lift(root,ds,np.r_[fit,tune])
    if not np.array_equal(vals['fixed'],fixed):raise RuntimeError('Interface degrees of freedom changed')
    if np.max(abs(vals['basis'][:,fixed,:]))>1e-12:raise RuntimeError('Nonzero basis at interface')
    if float(np.max(abs(vals['fit_lift']-lift[:len(fit)])))>1e-5:
        raise RuntimeError('Lift changed since B0; stop')
    return dict(ds=ds,fit=fit,tune=tune,source=source,source_sha=sha(source),
       rank=rank,basis=vals['basis'][:rank].astype(np.float32),
       lift=lift.astype(np.float32),fixed=fixed,
       sU=vals['sU'].astype(np.float32),sG=vals['sG'].astype(np.float32),
       branch_mean=vals['branch_mean'].astype(np.float32),
       branch_std=vals['branch_std'].astype(np.float32))


class ResidualCoeffNet(nn.Module):
    def __init__(self,branch_dim,rank,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(branch_dim,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),
                               nn.Linear(hidden,rank))
        # Initially zero interior correction; baseline is the exact interface lifting.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)


def metrics(U_pred,U_ref,ds,LE_ref):
    Gp=AA.gradient_numpy(U_pred,ds)
    Gr=AA.gradient_numpy(U_ref,ds)
    rel=CO.relative_l2
    estimated=np.stack([CO.log_material_strain_eng(u,ds['element_node_indices'],ds['B_center']) for u in U_pred])
    # C3D8R mean per-case relative error, never obscure an erroneous near-zero component.
    return dict(U_vec_rel_pct=rel(U_pred,U_ref),U2_rel_pct=rel(U_pred[...,1],U_ref[...,1]),
                grad_rel_pct=rel(Gp,Gr),dU2dY_rel_pct=rel(Gp[...,1,1],Gr[...,1,1]),
                dU1dY_rel_pct=rel(Gp[...,0,1],Gr[...,0,1]),
                LE22_rel_pct=rel(estimated[...,1],LE_ref[...,1]),
                LE12_rel_pct=rel(estimated[...,3],LE_ref[...,3]),
                U_rmse_mm=float(np.sqrt(np.mean((U_pred-U_ref)**2))),
                G_rmse=float(np.sqrt(np.mean((Gp-Gr)**2))))


def baseline_predictions(root,region,ds,ids,device):
    src=root/'results'/'65AA_gradient_learning'/'B_gradient'/region/'member_01'/'best_v4_7region.pt'
    ck=CO.load_checkpoint(src)
    model=CO.make_model(ck,device)
    inputs=AA.tensor_inputs(ck,ds,device)
    U=AA.predict_numpy(model,inputs,ids,correct=True).astype(np.float64)
    del model,inputs
    return U


def candidate_path(root,arm,region):
    return root/'results'/ROOT_NAME/(arm+'_train')/region/'best_model.pt'


def check_reference(ds,ids):
    U=np.asarray(ds['U'][ids],np.float64)
    le=np.stack([CO.log_material_strain_eng(u,ds['element_node_indices'],ds['B_center']) for u in U])
    gt=ds['LE'][ids]
    out={k:CO.relative_l2(le[...,j],gt[...,j]) for k,j in [('LE11',0),('LE22',1),('LE12',3)]}
    # The original 65Y audit previously established <=0.5% for reference.
    if max(out.values())>0.5:
        raise RuntimeError('Abaqus reference logarithmic strain convention mismatch: '+str(out))
    return out


def np_tens(x,dev):return torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=dev)


def predict_candidate(net,loaded,ids,dev,batch=16):
    d=loaded; concat=np.r_[d['fit'],d['tune']]; ix={int(v):i for i,v in enumerate(concat)}
    model_x=(d['ds']['branch_inputs'][ids]-d['branch_mean'])/d['branch_std']
    lift=d['lift'][np.array([ix[int(i)] for i in ids])]
    basis=np_tens(d['basis'],dev)
    result=[]
    net.eval()
    with torch.no_grad():
        for s in range(0,len(ids),batch):
            x=np_tens(model_x[s:s+batch],dev)
            c=net(x)
            U=np_tens(lift[s:s+batch],dev)+torch.einsum('br,rnc->bnc',c,basis)
            result.append(U.detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def load_candidate(root,arm,region,loaded,dev):
    p=candidate_path(root,arm,region)
    if not p.is_file():raise FileNotFoundError('Missing candidate: '+str(p))
    ck=torch.load(p,map_location='cpu',weights_only=False)  # ONLY trusted local training outputs
    for key,value in [('arm',arm),('region',region),('basis_rank',loaded['rank']),
                      ('source_sha',loaded['source_sha'])]:
        if ck.get(key)!=value:raise RuntimeError('Candidate provenance mismatch '+key)
    net=ResidualCoeffNet(int(loaded['ds']['branch_inputs'].shape[1]),loaded['rank'],ck['hidden']).to(dev)
    net.load_state_dict(ck['state_dict'],strict=True)
    net.eval()
    return net,ck
