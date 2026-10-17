#!/usr/bin/env python3
"""Step 65AM-A helpers. Frozen, diagnostic-only FEM/NO gradient attribution.

ALL inputs are original locally generated 65AA/65AI, 65AJ, 65AK data.
Six AJ cases are previously examined EXTERNAL cases: explanation only, NO training.
"""
from pathlib import Path
import json
import numpy as np

REGIONS = ('inner_left', 'inner_right')
OUTROOT = '65AM_A_hermite_attribution'
COMPONENTS = ('U1','U2','U3')
DIRECTIONS = ('X','Y','Z')


def output(root, stage):
    folder = Path(root)/'results'/OUTROOT/stage
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Existing Step 65AM result protected. In Finder, remove ONLY this generated stage folder if deliberately rerunning:\n'+str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def relpct(x, ref):
    x=np.asarray(x, np.float64);ref=np.asarray(ref,np.float64)
    if x.shape!=ref.shape:raise RuntimeError('Relative error shape mismatch')
    return float(100*np.linalg.norm((x-ref).ravel())/max(np.linalg.norm(ref.ravel()),1e-14))


def rmse(x, ref):
    x=np.asarray(x,np.float64);ref=np.asarray(ref,np.float64)
    return float(np.sqrt(np.mean((x-ref)**2)))


def rms(x):
    x=np.asarray(x,np.float64)
    return float(np.sqrt(np.mean(x*x)))


def saved_json(path, v):
    Path(path).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def gradient(U, conn, dN):
    U=np.asarray(U,dtype=np.float64)
    if U.ndim!=3 or U.shape[-1]!=3:
        raise RuntimeError('Expected [case,node,3] displacement')
    return np.einsum('beai,eaj->beij',U[:,conn,:],dN,optimize=True)


def zone_masks(ip_coordinates, width=0.6):
    x=np.asarray(ip_coordinates,np.float64)[:,0]
    lo, hi = x.min(),x.max()
    left=x-lo<=width+1e-8
    right=hi-x<=width+1e-8
    interior=~(left|right)
    if lo>=hi or not left.any() or not right.any() or not interior.any() or np.any(left & right):
        raise RuntimeError('Cannot separate two interfaces/interior from integration point X coordinates')
    return {'left_end':left,'interior':interior,'right_end':right,'both_ends':left|right}


def check_gradient_geometry(xyz,conn,dn):
    xyz=np.asarray(xyz,np.float64)
    assert conn.shape[1]==8 and dn.shape[-2:]==(8,3)
    affine=np.einsum('eai,eaj->eij',xyz[conn],dn,optimize=True)
    maxerr=float(np.max(np.abs(affine-np.eye(3)[None,:,:])))
    const=float(np.max(np.abs(np.sum(dn,axis=1))))
    if maxerr>5e-3 or const>2e-5:raise RuntimeError('C3D8R affine/constant check failed')
    return {'affine_error':maxerr,'constant_gradient_error':const}


def summary_case(ground,raw,hard,pca,disp_ref=None,disp_raw=None,disp_hard=None):
    # Gradient convention: [elements, displacement component i, derivative j].
    a=rmse(raw,ground);b=rmse(hard,ground)
    r={
        'grad_raw_error_pct':relpct(raw,ground),
        'grad_hard_error_pct':relpct(hard,ground),
        'grad_pca_only_error_pct':relpct(pca,ground),
        'grad_raw_RMSE':a,
        'grad_hard_RMSE':b,
        'grad_pca_only_RMSE':rmse(pca,ground),
        'hard_minus_raw_RMSE':b-a,
        'hard_gradient_effect_pct':100*(1-b/max(a,1e-16)),
        'hermite_correction_gradient_RMS':rms(hard-raw),
        'pca_only_correction_gradient_RMS':rms(pca-ground),
        'raw_residual_dot_correction_mean':float(np.mean((raw-ground)*(hard-raw))),
        'gradient_correction_energy_mean':float(np.mean((hard-raw)**2)),
    }
    if disp_ref is not None:
        r.update(U_raw_error_pct=relpct(disp_raw,disp_ref),U_hard_error_pct=relpct(disp_hard,disp_ref))
    return r


def xz_reduce(points, values, absolute_rms=False):
    """Aggregate overlapping Y layers before X-Z visualization (no Y projection alias)."""
    points=np.asarray(points,np.float64)
    v=np.asarray(values,np.float64)
    if v.ndim!=1 or len(v)!=len(points):raise ValueError('XZ reduce shape mismatch')
    key=np.round(points[:,[0,2]],6)
    pair,group=np.unique(key,axis=0,return_inverse=True)
    count=np.bincount(group,minlength=len(pair))
    if absolute_rms:
        result=np.sqrt(np.bincount(group,weights=v*v,minlength=len(pair))/count)
    else:
        result=np.bincount(group,weights=v,minlength=len(pair))/count
    return pair,result,count
