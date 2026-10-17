#!/usr/bin/env python3
"""Step 65AL shared read-only audit of frozen U derivatives.

Reads Step 65AK-A cached FEM/reference predictions and original 65AA FEM shape
functions. NEVER optimizes, updates a checkpoint, or modifies Abaqus data.
"""
from pathlib import Path
import importlib
import json
import hashlib
import numpy as np

ak=importlib.import_module('65AK_common')
aa=importlib.import_module('65AA_common')
ac=importlib.import_module('65AC_common')
REGIONS=('inner_left','inner_right')
BASE='65AL_paper_informed_gradient_audit'
TOL=2e-5

def relpct(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape
    return float(100*np.linalg.norm((a-b).reshape(-1))/max(np.linalg.norm(b.reshape(-1)),1e-14))

def rmse(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return float(np.sqrt(np.mean((a-b)**2)))

def load(root,region):
    z=ak.load_A(root,region)
    ak.provenance_check(root,region,z)
    ds=aa.load_data(root,region)
    xyz=np.asarray(ds['coordinates'],dtype=np.float64)
    ips=np.asarray(ds['ip_coordinates'],dtype=np.float64)
    conn=np.asarray(ds['element_node_indices'],dtype=np.int64)
    for key,reference in [('coordinates',xyz),('ip_coordinates',ips)]:
        given=np.asarray(z[key],dtype=np.float64)
        if given.shape!=reference.shape or np.max(np.abs(given-reference))>TOL:
            raise RuntimeError(f'Geometry/order mismatch between Step 65AK and original 64J for {region}/{key}')
    if conn.ndim!=2 or conn.shape[1]!=8 or int(conn.max())>=len(xyz) or int(conn.min())<0:
        raise RuntimeError('Invalid C3D8R connectivity for '+region)
    for k in ('U_pred','U_ref'):
        if z[k].shape!=(6,len(xyz),3):
            raise RuntimeError('Unexpected six-case displacement shape '+k+' '+str(z[k].shape))
    dn=aa.element_dN_numpy(ds).astype(np.float64)
    geom=np.einsum('eic,eij->ecj',xyz[conn],dn,optimize=True)
    aff=float(np.max(np.abs(geom-np.eye(3)[None,:,:])))
    if aff>5e-3:raise RuntimeError(f'Non-affine FE geometry/shape derivatives {region}: {aff:.4g}')
    if z['LE_ref'].shape!=(6,len(ips),6):raise RuntimeError('Unexpected FEM LE shape')
    return z,ds,conn,dn,aff

def outdir(root,stage):
    p=Path(root)/'results'/BASE/stage
    if p.exists() and any(p.iterdir()):
        raise RuntimeError('65AL never overwrites existing output. To rerun manually delete ONLY:\n'+str(p))
    p.mkdir(parents=True,exist_ok=True)
    return p

def summary(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def near_mask(coords,width=.6):
    x=np.asarray(coords)[:,0]
    lo,hi=float(np.min(x)),float(np.max(x))
    near=np.minimum(x-lo,hi-x)<=width+1e-8
    if not (np.any(near) and np.any(~near)):raise RuntimeError('No near/interior region under given 0.6 mm criterion')
    return near

def gradient(U,conn,dn):
    U=np.asarray(U,dtype=np.float64)
    return np.einsum('beic,eij->becj',U[:,conn,:],dn,optimize=True)

def shape_norm(a,axis=-1):
    return np.sqrt(np.sum(np.asarray(a,dtype=np.float64)**2,axis=axis))
