"""Step 65W-A: independent 3-Abaqus-patch verification of saved Step65V convergence.

Run from ODBS root: abaqus python scripts\65W_A_reproduce.py --cpus 4
Requires saved Step65V convergence solution, old working 65L and 65O modules.
No existing experiment directories are modified. Outputs only in 65W_validation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'online_results_7region_65V_middle' / '65V_D_lm_convergence' / 'rom_solution_7region.npz'
MANIFEST = SOURCE.parent / 'rom_solution_manifest.json'
OUT = ROOT / 'online_results_7region_65W_middle' / 'A_independent_reproduction'
MAT = np.array([46500., 12500., 4750.])


def loadmod(name, alias):
    p = HERE / name
    if not p.is_file():
        raise FileNotFoundError('Missing existing Windows runtime: %s' % p)
    spec = importlib.util.spec_from_file_location(alias, str(p))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def summary_npz(path):
    if not path.is_file():
        raise FileNotFoundError('Missing converged 65V solution: %s' % path)
    with np.load(str(path), allow_pickle=False) as f:
        q = np.asarray(f['q'], float).ravel().copy()
        r = np.asarray(f['normalized_residual'], float).ravel().copy()
        scale = np.asarray(f['c_scale'], float).ravel().copy()
        m = np.array([float(np.asarray(f[k]).ravel()[0]) for k in ('E1', 'E2', 'G12')])
        saved = float(np.asarray(f['rms_residual']).ravel()[0])
    if any(v.shape != (26,) for v in (q, r, scale)):
        raise RuntimeError('Saved state needs 26 q, residual and coefficient scales')
    if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)) or
        not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise RuntimeError('Saved state has invalid values')
    if not np.allclose(m, MAT, atol=1e-8, rtol=0):
        raise RuntimeError('Wrong material; this validator is for middle material only')
    check = float(np.sqrt(np.mean(r*r)))
    if abs(check - saved) > 1e-9:
        raise RuntimeError('Saved residual vector RMS disagrees with scalar RMS')
    if MANIFEST.is_file():
        with open(str(MANIFEST)) as f:
            manifest = json.load(f)
        if abs(float(manifest['best_verified_rms'])-check) > 1e-9:
            raise RuntimeError('65V manifest and saved solution disagree')
        if not bool(manifest['converged']):
            raise RuntimeError('65V manifest does not report convergence')
    return q,r,scale,check


def sha256(path):
    h = hashlib.sha256()
    with open(str(path),'rb') as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(str(path),'w') as f:json.dump(value,f,indent=2)


def perform(args):
    q,r,scale,saved = summary_npz(SOURCE)
    if args.stage == 'audit':
        print(json.dumps({'source':str(SOURCE),'saved_rms':saved,
                          'saved_converged':bool(saved <= args.target),
                          'material_MPa':MAT.tolist(),
                          'workdir':str(OUT),'abaqus_jobs_for_reproduction':3},indent=2))
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('65W reproduction directory is not empty; refusing to overwrite: '+str(OUT))
    OUT.mkdir(parents=True)
    old=loadmod('65L_A_to_D_complete.py','W_65L')
    adapt=loadmod('65O_adaptive_coupling.py','W_65O')
    ctx=old.Coupling(MAT.copy(),str(OUT))
    if not np.allclose(ctx.scale,scale,rtol=1e-12,atol=1e-14):
        raise RuntimeError('Runtime coefficient scaling differs from saved 65V state')
    prior=ROOT / 'rom_models_7region' / '65N_center_correction.npz'
    oracle=adapt.Oracle(ctx,str(OUT),prior_path=str(prior),job_budget=3,
                        cpus=args.cpus,adaptive_fidelity=False,online_correction=True)
    t=time.perf_counter()
    result=oracle.evaluate(q)  # no incumbent: full independent 3-patch evaluation
    if not result['verified'] or set(result['files']) != {'left','center','right'} or oracle.jobs != 3:
        raise RuntimeError('Independent verification did not complete all 3 FEM patches')
    rr=np.asarray(result['r'],float).ravel()
    gap=float(np.sqrt(np.mean((rr-r)**2)))
    maxgap=float(np.max(np.abs(rr-r)))
    status=bool(result['rms'] <= args.target and gap <= args.repro_tol)
    details={
        'saved_65V_rms':saved, 'reproduced_hf_rms':float(result['rms']),
        'target_rms':args.target,'reproduces_saved_vector':bool(gap <= args.repro_tol),
        'reproduced_full_equilibrium_converged':bool(result['rms'] <= args.target),
        'independent_validation_pass':status,'vector_rms_gap':gap,
        'max_abs_component_gap':maxgap,'vector_gap_tolerance':args.repro_tol,
        'max_abs_new_residual':float(np.max(np.abs(rr))),
        'material_MPa':MAT.tolist(), 'total_abaqus_patch_jobs':int(oracle.jobs),
        'wall_seconds':time.perf_counter()-t,
        'source_sha256':sha256(SOURCE),
        'source':str(SOURCE),
        'note':'Three NEW Abaqus patches; no partial or surrogate-only acceptance. Interface RMS is not full-field accuracy.'
    }
    file_report={}
    # Copy actual verified ODBs/INPs for fully reproducible downstream extraction.
    final=OUT/'final';final.mkdir()
    for patch in ('left','center','right'):
        file_report[patch]={}
        for key,extension in (('odb_file','.odb'),('input_file','.inp')):
            p=Path(result['files'][patch][key])
            if not p.is_file():
                raise FileNotFoundError('Missing new verified FEM output: %s' % p)
            copied=final/(patch+extension)
            shutil.copy2(str(p),str(copied))
            file_report[patch][key]=str(copied)
            file_report[patch][key+'_sha256']=sha256(copied)
    details['verified_files']=file_report
    np.savez(str(OUT/'independently_verified_state.npz'),q=q,
             saved_residual=r,reproduced_residual=rr,c_scale=scale,
             rms_residual=np.array([result['rms']]),material=MAT)
    write_json(OUT/'65W_A_reproduction.json',details)
    print(json.dumps(details,indent=2))
    if not status:
        print('WARNING: verification failed threshold and/or reproducibility',file=sys.stderr)
        raise SystemExit(2)


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'a.npz'
        q=np.zeros(26);r=np.full(26,0.019);s=np.ones(26)
        np.savez(str(p),q=q,normalized_residual=r,c_scale=s,rms_residual=np.array([0.019]),
                 E1=np.array([46500.]),E2=np.array([12500.]),G12=np.array([4750.]))
        # Same shape/numerical invariant as actual save, no Abaqus.
        with np.load(str(p)) as f:
            assert abs(np.sqrt(np.mean(f['normalized_residual']**2))-.019)<1e-14
    print('65W-A selftest PASS (numerical/file shape; no Abaqus run)')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['selftest','audit','verify'],default='audit')
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--target',type=float,default=.02)
    p.add_argument('--repro-tol',type=float,default=1e-5)
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    else:perform(a)

if __name__=='__main__':main()
