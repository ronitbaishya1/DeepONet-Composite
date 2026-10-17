#!/usr/bin/env python3
"""65AM-B9 Windows Abaqus Python: export a VERIFIED saved 65V state and force basis.

No Abaqus jobs. Does not change 65V or original models. Requires:
  ODBS/scripts/hybrid_runtime_7region.py
  ODBS/interfaces_7region/interface_*.npz
  ODBS/online_results_7region_65V_middle/65V_D_lm_convergence/rom_solution_7region.npz

From ODBS Abaqus Command:
  abaqus python scripts\65AM_B9_W_export_65V_anchor.py --stage check
  abaqus python scripts\65AM_B9_W_export_65V_anchor.py --stage export

Create ODBS/B9_anchor in File Explorer BEFORE export.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NAMES=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
EXPECTED_COUNTS=(4,4,5,5,4,4)  # checked against stored 65V source
DEFAULT_SOURCE=ROOT/'online_results_7region_65V_middle'/'65V_D_lm_convergence'/'rom_solution_7region.npz'
OUT=ROOT/'B9_anchor'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()


def arrays(source):
    if not source.is_file():raise FileNotFoundError('Saved 65V solution not found: '+str(source))
    with np.load(str(source),allow_pickle=False) as z:
        must=('q','normalized_residual','c_scale','rms_residual','E1','E2','G12')+tuple('c_'+n for n in NAMES)
        if set(must)-set(z.files):raise RuntimeError('Saved solution missing '+str(set(must)-set(z.files)))
        o={k:np.asarray(z[k],dtype=np.float64).copy() for k in must}
    for k in ('q','normalized_residual','c_scale'):
        if o[k].shape!=(26,) or not np.isfinite(o[k]).all():raise RuntimeError('Invalid 65V vector '+k)
    if np.any(o['c_scale']<=0):raise RuntimeError('Nonpositive interface scaling')
    if tuple(len(o['c_'+n].ravel()) for n in NAMES)!=EXPECTED_COUNTS:raise RuntimeError('Unexpected 65V PCA sizes; stop and inspect')
    if not np.allclose(np.concatenate([o['c_'+n].ravel() for n in NAMES]),o['q']*o['c_scale'],atol=1e-8,rtol=1e-6):
        raise RuntimeError('65V stored q and physical interface coefficients differ')
    mat=np.array([float(o[n].ravel()[0]) for n in ('E1','E2','G12')])
    if not np.allclose(mat,[46500.,12500.,4750.],atol=1e-5,rtol=0):raise RuntimeError('65V is not the documented middle material')
    rms=float(np.sqrt(np.mean(o['normalized_residual']**2)))
    if abs(rms-float(o['rms_residual'].ravel()[0]))>1e-8:raise RuntimeError('65V stored RMS mismatch')
    if rms>.02000001:raise RuntimeError('Saved 65V is not converged (RMS %.6f)'%rms)
    o['material']=mat
    return o,rms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('check','export'))
    p.add_argument('--solution',default=str(DEFAULT_SOURCE))
    a=p.parse_args()
    path=Path(a.solution).resolve()
    d,rms=arrays(path)
    # Import exactly the original Windows interface/force runtime; no solve calls.
    import sys
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    from hybrid_runtime_7region import evaluate_neural_operators,get_residual_force_scales
    scales=get_residual_force_scales()
    fs=np.concatenate([np.asarray(scales[n],float).ravel() for n in NAMES])
    if fs.shape!=(26,) or np.any(fs<=0) or not np.isfinite(fs).all():raise RuntimeError('Invalid 65V force scales')
    coeff={n:d['c_'+n].ravel() for n in NAMES}
    old=evaluate_neural_operators(float(d['material'][0]),float(d['material'][1]),float(d['material'][2]),coeff)
    oldg=np.concatenate([np.asarray(old[n],float).ravel() for n in NAMES])
    if oldg.shape!=(26,) or not np.isfinite(oldg).all():raise RuntimeError('Invalid old NO force predictions')
    # At the SAME saved q, old verified residual was (actual FE force + old NO force) / scale.
    fef=fs*d['normalized_residual']-oldg
    # Export literal PCA coordinates/bases to prevent silent mismatch between Windows and Mac copies.
    for n in NAMES:
        fp=ROOT/'interfaces_7region'/('interface_'+n+'.npz')
        if not fp.is_file():raise FileNotFoundError('PCA interface file missing: '+str(fp))
        with np.load(str(fp),allow_pickle=False) as z:
            for key in ('mean','basis','coordinates'):
                if key not in z:raise RuntimeError(str(fp)+' lacks '+key)
                d['pca_'+key+'_'+n]=np.asarray(z[key],dtype=np.float64).copy()
        k=len(coeff[n]);ba=d['pca_basis_'+n];nodes=d['pca_coordinates_'+n]
        if ba.ndim!=2 or ba.shape[1]!=k or ba.shape[0]!=nodes.shape[0]*3:
            raise RuntimeError('Windows PCA layout mismatch '+n)
        if d['pca_mean_'+n].size!=ba.shape[0]:raise RuntimeError('PCA mean size mismatch '+n)
    print('B9 WINDOWS CHECK PASS | saved 65V RMS = %.7f | old NO and FE force coefficients ready'%rms)
    if a.stage=='check':
        print('CHECK ONLY: no files created')
        return
    if not OUT.is_dir():raise RuntimeError('In Windows File Explorer, create ODBS\\B9_anchor first')
    dest=OUT/'B9_65V_anchor.npz';meta=OUT/'B9_65V_anchor_manifest.json'
    if dest.exists() or meta.exists():raise RuntimeError('B9 anchor already exists. Leave earlier outputs unchanged.')
    d.update(old_NO_g=oldg,FE_g_inferred=fef,force_scale=fs)
    np.savez_compressed(str(dest),**d)
    with open(meta,'w') as f:
        json.dump(dict(step='65AM-B9',method='saved-state FE force + NO operator audit',
                       original_verified_RMS=rms,source_65V=str(path),source_65V_sha256=sha(path),
                       anchor_sha256=sha(dest),material_MPa=d['material'].tolist(),
                       no_new_abaqus_jobs=True,full_B5_coupling_verified=False,
                       assumptions=['Inferred FE generalized force from saved verified 65V residual, assuming the current old NO force model is unchanged; not independently extracted from Abaqus ODB',
                         'Saved PCA mean/basis/coordinate arrays copied for Windows-Mac alignment']),f,indent=2)
    print('B9 EXPORTED:',dest)
    print('B9 EXPORTED:',meta)

if __name__=='__main__':main()
