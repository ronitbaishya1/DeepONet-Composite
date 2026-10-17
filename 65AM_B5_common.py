#!/usr/bin/env python3
"""Step 65AM-B5: coefficient-centered/standardized FE-residual operator.

Only original 64J TRAIN fit (44) is used to estimate coefficient statistics.
Every basis, split, interface, source model and normalization is reused from B0.
No Abaqus, 65V, 65AI, 64J validation/test, or AJ external cases are touched.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
import importlib

b = importlib.import_module('65AM_B_common')
VARIANTS = ('centered','standardized')
LOSSES = ('U','grad')
ROOT = '65AM_B5_coeff_normalized'


def paths(root,variant,loss,region=None):
    if variant not in VARIANTS or loss not in LOSSES:
        raise ValueError('Unknown variant or loss')
    base=Path(root)/'results'/ROOT/(variant+'_'+loss)
    return base if region is None else base/region


def report_path(root):
    return Path(root)/'results'/ROOT/'B5_comparison'


def guarded_directory(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()):
        raise RuntimeError('Existing B5 output protected. Inspect in Finder; do not overwrite:\n'+str(p))
    p.mkdir(parents=True,exist_ok=True)
    return p


def coefficient_targets(data,ids):
    """FEM-oracle POD coefficients for SUPERVISED TRAINING or metrics, NEVER inference."""
    d=data; ds=d['ds']; ids=np.asarray(ids,dtype=np.int64)
    fit_tune=np.r_[d['fit'],d['tune']]
    if not np.isin(ids,fit_tune).all():
        raise RuntimeError('Data leakage: coefficient oracle outside original 44/12 TRAIN split')
    with np.load(b.saved_basis_path(ds['_root'],ds.get('_region','inner_left')),allow_pickle=False) as z:
        sR=np.asarray(z['sR'],dtype=np.float64)
    free=np.setdiff1d(np.arange(d['basis'].shape[1]),d['fixed'])
    modes=(np.asarray(d['basis'][:,free,:],dtype=np.float64)/sR[None,None,:]).reshape(d['rank'],-1)
    gram=modes@modes.T
    if np.max(np.abs(gram-np.eye(d['rank'])))>0.002:
        raise RuntimeError('POD basis Gram check failed; cannot compute oracle coefficients')
    lookup={int(v):i for i,v in enumerate(fit_tune)}
    lift=np.asarray(d['lift'][[lookup[int(v)] for v in ids]],dtype=np.float64)
    residual=((np.asarray(ds['U'][ids],dtype=np.float64)-lift)[:,free,:]/sR).reshape(len(ids),-1)
    coeff=np.linalg.solve(gram,(residual@modes.T).T).T
    if not np.isfinite(coeff).all():raise RuntimeError('Nonfinite POD coefficients')
    return coeff.astype(np.float32)


def stats_from_fit(root,region,data):
    # Build on original saved basis only; never re-fit basis using tune cases.
    data['ds']['_root']=Path(root)
    data['ds']['_region']=region
    c=coefficient_targets(data,data['fit']).astype(np.float64)
    mean=c.mean(axis=0).astype(np.float32)
    raw_std=c.std(axis=0).astype(np.float32)
    if np.any(~np.isfinite(raw_std)) or float(raw_std.max())<=1.e-9:
        raise RuntimeError('Invalid coefficient distribution: inspect B4 before training '+region)
    # Avoid numerical instability for nearly inactive POD modes. Does not affect
    # actual B4 data, where all reported FIT SDs are substantially larger.
    floor=max(1.e-4,1.e-4*float(raw_std.max()))
    std=np.maximum(raw_std,floor).astype(np.float32)
    return mean,std


class ScaledCoeffNet(nn.Module):
    """Same trainable network as B1/B2. Fixed output map c=mean+scale*z."""
    def __init__(self,branch_dim,rank,hidden,mean,scale):
        super().__init__()
        self.core=b.ResidualCoeffNet(int(branch_dim),int(rank),int(hidden))
        m=torch.as_tensor(np.array(mean,copy=True),dtype=torch.float32)
        s=torch.as_tensor(np.array(scale,copy=True),dtype=torch.float32)
        if m.shape!=(rank,) or s.shape!=(rank,) or not torch.all(s>0):
            raise RuntimeError('Invalid coefficient transforms')
        self.register_buffer('coeff_mean',m)
        self.register_buffer('coeff_scale',s)
    def forward(self,x):
        return self.coeff_mean[None,:]+self.coeff_scale[None,:]*self.core(x)


def prepare(root,region):
    d=b.load_saved(Path(root),region)
    mu,sd=stats_from_fit(root,region,d)
    # mean is a fixed FE coefficient baseline; modes vanish on prescribed interfaces.
    fixed=d['fixed'];basis=d['basis']
    if np.max(np.abs(basis[:,fixed,:]))>1e-9:
        raise RuntimeError('Interface basis is not zero')
    if np.max(np.abs(d['lift'][:,fixed,:]-d['ds']['U'][np.r_[d['fit'],d['tune']]][:,fixed,:]))>1e-4:
        raise RuntimeError('FEM target vs prescribed PCA interface mismatch')
    return d,mu,sd


def build_model(data,mu,sd,variant,hidden,device):
    if variant not in VARIANTS:raise ValueError(variant)
    scale=np.ones_like(sd) if variant=='centered' else sd
    return ScaledCoeffNet(int(data['ds']['branch_inputs'].shape[1]),data['rank'],hidden,mu,scale).to(device)


def ck_path(root,variant,loss,region):
    return paths(root,variant,loss,region)/'best_model.pt'


def load_model(root,variant,loss,region,data,mu,sd,dev):
    ckfile=ck_path(root,variant,loss,region)
    if not ckfile.is_file():raise FileNotFoundError(str(ckfile))
    ck=torch.load(ckfile,map_location='cpu',weights_only=False) # trusted locally generated checkpoint only
    expected={'variant':variant,'loss':loss,'region':region,'rank':int(data['rank']),
              'baseline_sha':data['source_sha'],'basis_sha':b.sha(b.saved_basis_path(Path(root),region))}
    for k,v in expected.items():
        if ck.get(k)!=v:raise RuntimeError('B5 checkpoint provenance mismatch: '+k)
    if not np.array_equal(ck['fit_ids'],data['fit']) or not np.array_equal(ck['tune_ids'],data['tune']):
        raise RuntimeError('B5 training split mismatch')
    if np.max(abs(np.asarray(ck['coeff_mean'])-mu))>2e-5 or np.max(abs(np.asarray(ck['coeff_std'])-sd))>2e-5:
        raise RuntimeError('B5 coefficient statistics changed')
    model=build_model(data,mu,sd,variant,int(ck['hidden']),dev)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    return model,ck


def predict(model,data,ids,dev):
    # Never infer from FEM U; only normalized branch inputs and prescribed lifting.
    allids=np.r_[data['fit'],data['tune']]
    lookup={int(v):i for i,v in enumerate(allids)}
    ids=np.asarray(ids,dtype=np.int64)
    if not np.isin(ids,allids).all(): raise RuntimeError('B5 inference limited to original train 44/12 during development')
    x=(data['ds']['branch_inputs'][ids]-data['branch_mean'])/data['branch_std']
    lifted=data['lift'][[lookup[int(i)] for i in ids]]
    base=b.np_tens(data['basis'],dev)
    outputs=[];coeffs=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(ids),16):
            c=model(b.np_tens(x[i:i+16],dev))
            out=b.np_tens(lifted[i:i+16],dev)+torch.einsum('br,rnc->bnc',c,base)
            coeffs.append(c.cpu().numpy())
            outputs.append(out.cpu().numpy())
    return np.concatenate(outputs).astype(np.float64),np.concatenate(coeffs).astype(np.float64)
