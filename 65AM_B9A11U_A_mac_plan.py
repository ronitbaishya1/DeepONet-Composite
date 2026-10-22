#!/usr/bin/env python3
"""B9A11U-A: plan FIT-only fixed-q FEM sensitivity pairs; preserve B9A11T holdout.

Mac: python3 65AM_B9A11U_A_mac_plan.py --stage selftest|check|plan --project .
Writes only B9A11U_plan/*. Training centers are from ORIGINAL B5 FIT (never TUNE/TEST).
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys
from pathlib import Path
import numpy as np
REGIONS=('inner_left','inner_right')
SHIFTS=('E1_M','E1_P','E2_M','E2_P','G12_M','G12_P')
EPS=0.03
N_TRAIN=3; N_DEV=1

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for q in iter(lambda:f.read(1<<20),b''):h.update(q)
    return h.hexdigest()

def stabledesign(bounds, inputs, fit_ids, anchor_mat, n=4):
    mat=np.asarray(inputs)[fit_ids,:3].astype(float)
    low,hi=np.asarray(bounds[0]),np.asarray(bounds[1])
    keep=np.all((mat*(1-EPS)>low+1.e-3)&(mat*(1+EPS)<hi-1.e-3),axis=1)
    # Do not select a center essentially at the old T reserved anchor.
    scale=np.maximum(hi-low,1.)
    keep&=(np.linalg.norm((mat-anchor_mat[None,:])/scale,axis=1)>.08)
    inds=np.where(keep)[0]
    if len(inds)<n:raise RuntimeError('Not enough in-bound FIT material centers. Need '+str(n)+' found '+str(len(inds)))
    z=(mat[inds]-low)/scale
    chosen=[int(np.argmin(np.sum((z-0.5)**2,axis=1)))]
    while len(chosen)<n:
        sep=np.min(np.linalg.norm(z[:,None,:]-z[np.array(chosen)][None,:,:],axis=2),axis=1)
        sep[chosen]=-1.
        chosen.append(int(np.argmax(sep)))
    return fit_ids[inds[np.array(chosen)]]

def context(root):
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import torch
    c=importlib.import_module('65AM_B5_common')
    b9=importlib.import_module('65AM_B9_mac_saved_state')
    b6=importlib.import_module('65AM_B6_mac')
    anchor,md=b9.load_anchor(root)
    stuff={}
    for r in REGIONS:
        d,mu,sd=c.prepare(root,r)
        # Require same Mac/Windows interface PCA coordinate/basis definition.
        ctx=b6.b.CO.interface_context(root,d['ds'],torch.device('cpu'))
        for side,name in b9.SIDES[r].items():b9.check_pca(anchor,ctx,side,name)
        fit=np.asarray(d['fit'],int)
        src=np.asarray(d['ds']['branch_inputs'],float)
        if src.shape[1]!=12 or len(fit)!=44:raise RuntimeError('B5 training FIT source changed '+r)
        mat=src[fit,:3]; low=mat.min(axis=0);hi=mat.max(axis=0)
        ids=stabledesign((low,hi),src,fit,np.asarray(anchor['material'],float))
        if len(np.unique(ids))!=N_TRAIN+N_DEV:raise RuntimeError('Duplicate cases')
        stuff[r]=dict(d=d,ids=ids,branches=src[ids].copy(),cksha=sha(c.ck_path(root,'centered','grad',r)),bounds=(low,hi))
        print('PLAN AUDIT',r,'FIT ids',ids.tolist(),'material MPa',src[ids,:3].round(3).tolist(),flush=True)
    return anchor,md,stuff

def selftest():
    lo=np.array([10.,20.,30.]);hi=np.array([20.,40.,60.]);rng=np.random.default_rng(10)
    a=np.tile(lo,(60,1))+rng.uniform(.2,.8,(60,3))*(hi-lo)
    ids=stabledesign((lo,hi),np.c_[a,np.zeros((60,9))],np.arange(60),np.array([30.,50.,80.]))
    assert len(np.unique(ids))==4
    print('B9A11U-A SELFTEST PASS')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',required=True,choices=('selftest','check','plan'))
    ap.add_argument('--project',default='.')
    args=ap.parse_args()
    if args.stage=='selftest':return selftest()
    root=Path(args.project).resolve()
    anchor,md,stuff=context(root)
    if args.stage=='check':print('U1 CHECK PASS; no files written');return
    folder=root/'B9A11U_plan'
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('Plan output exists; refusing overwrite: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    payload={'anchor_sha256':np.asarray(md['anchor_sha256']),'fraction':np.asarray(EPS),
             'shift_ids':np.asarray(SHIFTS)}
    for r,p in stuff.items():
        payload['branch_'+r]=p['branches'].astype(np.float64)
        payload['fit_ids_'+r]=p['ids'].astype(np.int64)
        payload['roles_'+r]=np.asarray(['train']*N_TRAIN+['dev'])
    nf=folder/'B9A11U_plan.npz';np.savez_compressed(nf,**payload)
    manifest=dict(stage='B9A11U-FIT-only-plan',plan_sha256=sha(nf),anchor_sha256=md['anchor_sha256'],
                  fraction=EPS,train_centers_per_region=N_TRAIN,dev_centers_per_region=N_DEV,
                  FEM_jobs=2*(N_TRAIN+N_DEV)*len(SHIFTS),
                  original_B5_checkpoint_hashes={r:p['cksha'] for r,p in stuff.items()},
                  FIT_only=True,original_12_TUNE_unused_for_gradient_pairs=True,
                  B9A11T_reserved_as_unseen_FIXED_Q_65V_anchor=True,not_full_coupled=True)
    (folder/'B9A11U_plan_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('U1 PLAN SAVED',nf,'\nCopy BOTH B9A11U_plan files to Windows ODBS\\B9A11U_plan')

if __name__=='__main__':main()
