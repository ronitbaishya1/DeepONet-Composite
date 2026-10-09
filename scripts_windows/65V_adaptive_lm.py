"""65V - target RMS <= 0.02: measured center/left Jacobian + reused right Jacobian,
      safeguarded full-system LM, patchwise HF verification and secant updates.

Put in Windows ODBS\\scripts. Requires WORKING, PREVIOUSLY TESTED:
    scripts\\65L_A_to_D_complete.py
    scripts\\65O_adaptive_coupling.py
    online_results_7region_65U_middle\\65U_C_lm_continuation\\
        rom_solution_7region.npz
        right_refresh_01.npz

Stages from ODBS (university Windows server):
    abaqus python scripts\\65V_adaptive_lm.py --stage selftest
    abaqus python scripts\\65V_adaptive_lm.py --stage audit
    abaqus python scripts\\65V_adaptive_lm.py --stage calibrate --cpus 4
    abaqus python scripts\\65V_adaptive_lm.py --stage plan
    abaqus python scripts\\65V_adaptive_lm.py --stage solve --cpus 4 --jobs 60
    abaqus python scripts\\65V_adaptive_lm.py --stage review

Calibration = 12 center patch + 2 left patch jobs, separate from --jobs.
Solver = at most --jobs patch jobs including 3-patch reproduction and refreshes.
No original file is overwritten. NO guarantee of numerical convergence.
Exit code 2 means the verified RMS target was not achieved, but results were saved.
"""
import argparse
import csv
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASE = np.array([46500., 12500., 4750.])
PREV = ROOT / 'online_results_7region_65U_middle' / '65U_C_lm_continuation'
SOURCE = PREV / 'rom_solution_7region.npz'
RIGHT_SOURCE = PREV / 'right_refresh_01.npz'
OUT = ROOT / 'online_results_7region_65V_middle'
CALIB = OUT / '65V_B_initial_calibration.npz'
CENTER_ROWS = np.arange(8, 18)
RIGHT_ROWS = np.arange(18, 26)
LEFT_ROWS = np.arange(0, 8)
CENTER_COLS = np.array([8, 11, 13, 14, 15, 16], int)
RIGHT_COLS = np.array([20, 21], int)
LEFT_COLS = np.array([1], int)
ACTIVE = np.r_[CENTER_COLS, RIGHT_COLS, LEFT_COLS]
NAMES = ('left_outer', 'left_inner', 'center_left', 'center_right',
         'right_inner', 'right_outer')
ENDS = (0, 4, 8, 13, 18, 22, 26)
PATCH_KEYS = {
    'left': ('left_outer', 'left_inner'),
    'center': ('center_left', 'center_right'),
    'right': ('right_inner', 'right_outer'),
}
PATCH_ROWS = {'left': LEFT_ROWS, 'center': CENTER_ROWS, 'right': RIGHT_ROWS}
PATCH_COLS = {'left': LEFT_COLS, 'center': CENTER_COLS, 'right': RIGHT_COLS}


def rms(r):
    return float(np.sqrt(np.mean(np.asarray(r, float)**2)))


def objective(r):
    r = np.asarray(r, float)
    return 0.5 * float(r @ r)


def write_json(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(x, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def v(data, key, length):
    a = np.asarray(data[key], float).ravel().copy()
    if a.shape != (length,) or not np.all(np.isfinite(a)):
        raise ValueError('%s must be %d finite values' % (key, length))
    return a


def load_source():
    for x in (SOURCE, RIGHT_SOURCE):
        if not x.is_file():
            raise FileNotFoundError('Missing previous Windows file: '+str(x))
    with np.load(SOURCE, allow_pickle=False) as f:
        q = v(f, 'q', 26)
        r = v(f, 'normalized_residual', 26)
        scale = v(f, 'c_scale', 26)
        material = np.array([float(np.asarray(f[x]).ravel()[0]) for x in ('E1', 'E2', 'G12')])
        stored = float(np.asarray(f['rms_residual']).ravel()[0])
    with np.load(RIGHT_SOURCE, allow_pickle=False) as f:
        qr = v(f, 'q0', 26)
        rr = v(f, 'r0', 26)
        cols = np.asarray(f['selected_idx'], int).ravel()
        Jr = np.asarray(f['j_true_right'], float).copy()
        right_mat = v(f, 'material', 3)
    if not np.allclose(material, CASE, rtol=0, atol=1e-8) or not np.allclose(right_mat, CASE, rtol=0, atol=1e-8):
        raise RuntimeError('Material does not match middle case')
    if np.max(abs(q)) > 3 or np.any(scale <= 0) or abs(rms(r)-stored) > 1e-9:
        raise RuntimeError('Saved solution invalid/inconsistent')
    if not np.array_equal(cols, RIGHT_COLS) or Jr.shape != (8, 2) or not np.all(np.isfinite(Jr)):
        raise RuntimeError('Expected right refresh shape (8,2) at indices [20,21]')
    if np.linalg.norm(q[RIGHT_ROWS]-qr[RIGHT_ROWS]) > .16:
        raise RuntimeError('Saved right Jacobian is too far in local coordinates; remeasure before using')
    return dict(q=q, r=r, scale=scale, qr=qr, Jr=Jr, right_ref_r=rr)


def load_module(filename, label):
    path = HERE / filename
    if not path.is_file():
        raise FileNotFoundError('Missing existing Windows coupling script: '+str(path))
    spec = importlib.util.spec_from_file_location(label, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_oracle(directory, jobs, cpus):
    old = load_module('65L_A_to_D_complete.py', 'v_65L')
    adapt = load_module('65O_adaptive_coupling.py', 'v_65O')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ctx = old.Coupling(CASE.copy(), str(directory))
    prior = ROOT / 'rom_models_7region' / '65N_center_correction.npz'
    oracle = adapt.Oracle(ctx, str(directory), prior_path=str(prior),
                          job_budget=jobs, cpus=cpus,
                          adaptive_fidelity=True, online_correction=True)
    return ctx, oracle


def context_check(ctx, d, force_scale=None):
    if not np.allclose(ctx.scale, d['scale'], atol=1e-14, rtol=1e-12):
        raise RuntimeError('PCA scaling differs from saved verified state')
    fs = np.asarray(ctx.force_scale, float)
    if fs.shape != (26,) or not np.all(np.isfinite(fs)) or np.any(fs <= 0):
        raise RuntimeError('Invalid force normalization')
    if force_scale is not None and not np.allclose(fs, force_scale, atol=1e-14, rtol=1e-12):
        raise RuntimeError('Force normalization differs from initial calibration')


def patch_residual(ctx, oracle, patch, q, serial):
    """One actual Abaqus patch. Does NOT declare complete convergence."""
    _, _, _, no_g = ctx.rom(q, detailed=True)
    record, fe_g = oracle._patch(patch, q, serial, ctx.coeffs(q))
    fe = np.concatenate([np.asarray(fe_g[n], float).ravel() for n in PATCH_KEYS[patch]])
    no = np.concatenate([np.asarray(no_g[n], float).ravel() for n in PATCH_KEYS[patch]])
    rr = (fe + no) / np.asarray(ctx.force_scale, float)[PATCH_ROWS[patch]]
    if rr.shape != (len(PATCH_ROWS[patch]),) or not np.all(np.isfinite(rr)):
        raise RuntimeError('Invalid high-fidelity residual for patch '+patch)
    return rr, record


def probe(ctx, oracle, patch, q, idx, h, serial, r_anchor):
    if q[idx]+h > 3 or q[idx]-h < -3:
        raise RuntimeError('q perturbation exceeds [-3,3] bound; index %d' % idx)
    qp = q.copy(); qm = q.copy()
    qp[idx] += h; qm[idx] -= h
    rp, fplus = patch_residual(ctx, oracle, patch, qp, serial)
    rm, fminus = patch_residual(ctx, oracle, patch, qm, serial+1)
    true_col = (rp-rm)/(2*h)
    curvature = rms(rp+rm-2*np.asarray(r_anchor)[PATCH_ROWS[patch]])
    return true_col, curvature, fplus, fminus


def numeric_jacobian(fun, q, epsilon=1e-3):
    q = np.asarray(q, float)
    f0 = np.asarray(fun(q), float)
    if f0.shape != (26,):
        raise RuntimeError('ROM residual shape should be (26,)')
    J = np.empty((26, 26), float)
    for i in range(26):
        x = q.copy(); h=epsilon*max(1., abs(q[i]))
        x[i] = min(3., q[i]+h)
        if x[i] == q[i]: x[i] = max(-3., q[i]-h)
        if x[i] == q[i]: raise RuntimeError('Cannot finite-difference q index %d' % i)
        J[:,i] = (np.asarray(fun(x),float)-f0)/(x[i]-q[i])
    if not np.all(np.isfinite(J)):
        raise RuntimeError('Nonfinite ROM Jacobian')
    return f0,J


def load_calibration(d):
    if not CALIB.is_file():
        raise FileNotFoundError('Run --stage calibrate first: '+str(CALIB))
    with np.load(CALIB, allow_pickle=False) as f:
        q0=v(f,'q0',26); r0=v(f,'r0',26)
        Jc=np.asarray(f['j_true_center'],float).copy()
        Jl=np.asarray(f['j_true_left'],float).copy()
        fs=v(f,'force_scale',26)
        material=v(f,'material',3)
        ci=np.asarray(f['center_selected'],int).ravel()
        li=np.asarray(f['left_selected'],int).ravel()
    if not np.allclose(q0,d['q'],atol=1e-12,rtol=0) or not np.allclose(r0,d['r'],atol=1e-12,rtol=0):
        raise RuntimeError('Calibration not at the saved 65U best solution')
    if not np.array_equal(ci,CENTER_COLS) or not np.array_equal(li,LEFT_COLS):
        raise RuntimeError('Calibrated column indices do not match this solver')
    if Jc.shape!=(10,6) or Jl.shape!=(8,1) or not np.all(np.isfinite(Jc)) or not np.all(np.isfinite(Jl)):
        raise RuntimeError('Invalid center or left high-fidelity Jacobian')
    if not np.allclose(material,CASE,atol=1e-8,rtol=0):
        raise RuntimeError('Calibration material mismatch')
    return dict(Jc=Jc,Jl=Jl,fs=fs)


def patch_drift(q, ref, patch):
    rows = PATCH_ROWS[patch]
    return float(np.linalg.norm(q[rows]-ref[rows]))


def audit(args):
    d=load_source();r=d['r'];q=d['q']
    groups=[];tot=float(r@r)
    for name,a,b in zip(NAMES,ENDS[:-1],ENDS[1:]):
        groups.append(dict(interface=name,rms=rms(r[a:b]),
                           squared_share_pct=100*float(r[a:b]@r[a:b])/tot))
    result=dict(start_verified_rms=rms(r),target=args.tol,
                missing_reduction_pct=100*(1-args.tol/rms(r)),
                current_right_local_drift=patch_drift(q,d['qr'],'right'),
                current_right_global_drift=float(np.linalg.norm(q-d['qr'])),
                planned_center_columns=CENTER_COLS.tolist(),
                planned_right_columns=RIGHT_COLS.tolist(),
                planned_left_columns=LEFT_COLS.tolist(),
                calibration_jobs_center=12,calibration_jobs_left=2,
                group_breakdown=groups,
                note='Right reuse does not guarantee validity away from calibration; full Abaqus verifies each accepted state.')
    write_json(OUT/'65V_A_audit.json',result)
    print(json.dumps(result,indent=2))


def calibrate(args):
    if CALIB.exists():
        raise RuntimeError('Calibration exists, refusing to overwrite: '+str(CALIB))
    directory=OUT/'65V_B_calibration_runs'
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Previous calibration runs exist; inspect before retrying: '+str(directory))
    d=load_source(); q=d['q']; r=d['r']; ctx,oracle=new_oracle(directory,14,args.cpus)
    context_check(ctx,d)
    _,Jm = numeric_jacobian(oracle.corrected,q)
    Jc=np.empty((10,6));Jl=np.empty((8,1))
    rows=[]
    count=0
    for patch,cols,out in (('center',CENTER_COLS,Jc),('left',LEFT_COLS,Jl)):
        for k,idx in enumerate(cols):
            true_col, curvature, fp, fm=probe(ctx,oracle,patch,q,int(idx),args.h,
                                              3000+2*count,r)
            count += 1
            out[:,k]=true_col
            model_col=Jm[np.ix_(PATCH_ROWS[patch],[int(idx)])].ravel()
            n_true=float(np.linalg.norm(true_col));n_model=float(np.linalg.norm(model_col))
            rows.append(dict(patch=patch,index=int(idx),
                  true_norm=n_true,rom_norm=n_model,
                  relative_error=float(np.linalg.norm(true_col-model_col)/max(n_true,1e-12)),
                  cosine=float(true_col@model_col/max(n_true*n_model,1e-14)),
                  second_difference_rms=curvature,
                  plus_odb=fp.get('odb_file',''),minus_odb=fm.get('odb_file','')))
    if oracle.jobs!=14:
        raise RuntimeError('Expected 14 Abaqus patch jobs, got %d' % oracle.jobs)
    np.savez(CALIB,q0=q,r0=r,material=CASE,c_scale=d['scale'],
             force_scale=np.asarray(ctx.force_scale,float),
             center_selected=CENTER_COLS,left_selected=LEFT_COLS,
             j_true_center=Jc,j_true_left=Jl,h=np.array([args.h]))
    write_csv(OUT/'65V_B_column_audit.csv',rows)
    write_json(OUT/'65V_B_manifest.json',dict(
        starting_rms=rms(r),patch_jobs=oracle.jobs,
        center_columns=CENTER_COLS.tolist(),left_columns=LEFT_COLS.tolist(),
        right_columns_reused=RIGHT_COLS.tolist(),
        center_singular_values=np.linalg.svd(Jc,compute_uv=False).tolist(),
        note='Finite difference sensitivity is at the saved verified starting state.'))
    print('65V calibration complete: 12 center + 2 left Abaqus patch jobs.')
    for item in rows:
        print('%-7s q[%2d] ROM column error %.2f; cosine %.3f' %
              (item['patch'],item['index'],item['relative_error'],item['cosine']))


def corrected_jacobian(oracle,q,cal,H):
    f,Jm = numeric_jacobian(oracle.corrected,q)
    B=np.zeros((26,26))
    for patch, jac in (('center',cal['Jc']),('left',cal['Jl']),('right',cal['Jr'])):
        idx=PATCH_ROWS[patch];cols=PATCH_COLS[patch]
        B[np.ix_(idx,cols)] = jac - Jm[np.ix_(idx,cols)]
    # Update only unmeasured columns using secants, preserving measured entries.
    return f,Jm+B+H,B


def lm(J,r,damping,radius):
    G=J.T@J
    reg=np.maximum(np.diag(G),1e-5)
    step=-np.linalg.solve(G+damping*np.diag(reg),J.T@r)
    n=float(np.linalg.norm(step))
    if n>radius:step *= radius/n
    return step


def direction_pool(J,r,radius,damping):
    options=[]
    groups=[('Full26-LM',np.arange(26)),('Active9-LM',ACTIVE),
            ('Center6-LM',CENTER_COLS),('Right2-LM',RIGHT_COLS),
            ('Left1-LM',LEFT_COLS)]
    for name,cols in groups:
        for factor in (0.35,1.0,3.0):
            try:
                p=np.zeros(26)
                p[cols]=lm(J[:,cols],r,damping*factor,radius)
                if np.linalg.norm(p)>1e-11:
                    options.append(('%s@%.2g'%(name,factor),p))
            except np.linalg.LinAlgError:
                continue
    # Steepest-descent fallback in the measured center subspace.
    g=J[np.ix_(CENTER_ROWS,CENTER_COLS)].T@r[CENTER_ROWS]
    if np.linalg.norm(g)>1e-10:
        p=np.zeros(26);p[CENTER_COLS]=-radius*g/np.linalg.norm(g)
        options.append(('Center-Cauchy',p))
    return options


def propose(oracle,q,r,cal,H,radius,damping,args):
    f,J,B=corrected_jacobian(oracle,q,cal,H)
    pool=[]
    for method,p0 in direction_pool(J,r,radius,damping):
        for frac in (1.0,.5,.25):
            step=p0*frac
            x=q+step
            if np.max(np.abs(x))>3:continue
            drifts={patch:patch_drift(x,cal['q'+patch[0]],patch) for patch in ('center','right','left')}
            global_drifts={patch:float(np.linalg.norm(x-cal['q'+patch[0]]))
                           for patch in ('center','right','left')}
            if (drifts['center']>args.center_hard or
                drifts['right']>args.right_hard or
                drifts['left']>args.left_hard or
                any(value>args.global_hard for value in global_drifts.values())):continue
            prediction=r + (np.asarray(oracle.corrected(x),float)-f)+(B+H)@step
            gain=objective(r)-objective(prediction)
            if gain<=1e-12 or not np.isfinite(gain):continue
            # Favor calibrated coordinates, but allow the 26-variable LM to
            # act on other modes if the model predicts worthwhile decrease.
            frac_unmeasured=float(np.linalg.norm(step[np.setdiff1d(np.arange(26),ACTIVE)]) /
                                  max(np.linalg.norm(step),1e-14))
            extra=float(drifts['center']>=args.center_refresh_at)*12 + float(drifts['right']>=args.right_refresh_at)*4
            utility=gain/(1+args.unmeasured_penalty*frac_unmeasured+args.refresh_penalty*extra)
            pool.append(dict(method=method,step=step,trial=x, predicted=float(gain),
                             predicted_rms=rms(prediction),utility=float(utility),
                             drifts=drifts,global_drifts=global_drifts))
    pool.sort(key=lambda x:x['utility'],reverse=True)
    return pool,J


def secant_update(H,J,s,y,blend):
    # Regularized Broyden correction in columns NOT currently calibrated.
    unmeasured=np.ones(26,bool);unmeasured[ACTIVE]=False
    su=s.copy();su[~unmeasured]=0.
    ss=float(su@su)
    if ss<1e-6:return H,False
    update=blend*np.outer(y-J@s,su)/(ss+.001)
    cap=max(1e-8,.25*float(np.linalg.norm(J,'fro')))
    scale=float(np.linalg.norm(update,'fro'))
    if scale>cap:update*=cap/scale
    Hnew=H+update
    cap2=max(1e-8,.6*float(np.linalg.norm(J,'fro')))
    if np.linalg.norm(Hnew,'fro')>cap2:
        Hnew*=cap2/np.linalg.norm(Hnew,'fro')
    Hnew[:,~unmeasured]=0
    return Hnew,True


def plan(args):
    d=load_source();c=load_calibration(d)
    ctx,oracle=new_oracle(OUT/'65V_C_model_only',0,args.cpus)
    context_check(ctx,d,c['fs'])
    cal={'Jc':c['Jc'],'Jl':c['Jl'],'Jr':d['Jr'],
         'qc':d['q'],'ql':d['q'],'qr':d['qr']}
    pool,J=propose(oracle,d['q'],d['r'],cal,np.zeros((26,26)),
                   args.radius,args.damping,args)
    report=dict(start_verified_rms=rms(d['r']),target=args.tol,
                singular_values=np.linalg.svd(J,compute_uv=False).tolist(),
                top_candidates=[dict(method=x['method'],predicted_rms=x['predicted_rms'],
                                     predicted_gain=x['predicted'],step=x['step'].tolist(),
                                     local_distances=x['drifts']) for x in pool[:12]],
                warning='ROM + measured local Jacobian prediction ONLY; Abaqus verification required.')
    write_json(OUT/'65V_C_plan.json',report)
    print('Step 65V model proposals (unverified):')
    for x in pool[:8]: print('  %-23s %.8f'%(x['method'],x['predicted_rms']))


def save_best(folder,best,ctx,scale):
    kw=dict(q=best['q'],normalized_residual=best['r'],c_scale=scale,
            rms_residual=np.array([best['rms']]),
            E1=np.array([CASE[0]]),E2=np.array([CASE[1]]),G12=np.array([CASE[2]]))
    for name,value in ctx.coeffs(best['q']).items():
        if name in NAMES:kw['c_'+name]=value
    np.savez(folder/'rom_solution_7region.npz',**kw)
    final=folder/'final';final.mkdir(exist_ok=True)
    for patch in ('left','center','right'):
        for key,suffix in (('odb_file','.odb'),('input_file','.inp')):
            source=Path(best['files'][patch][key])
            if not source.is_file():
                raise FileNotFoundError('Verified patch file missing: '+str(source))
            shutil.copy2(str(source),str(final/(patch+suffix)))


def checkpoint(folder,best,rows,oracle,phase,refreshes):
    np.savez(folder/'best_verified_checkpoint.npz',q=best['q'],
             normalized_residual=best['r'],rms_residual=np.array([best['rms']]))
    write_csv(folder/'65V_D_history.csv',rows)
    write_json(folder/'checkpoint.json',dict(phase=phase,best_verified_rms=float(best['rms']),
              jobs=int(oracle.jobs),accepted=sum(int(x['accepted']) for x in rows),refreshes=refreshes))


def refresh(ctx,oracle,cal,q,r,patch,args,count,folder):
    cols=PATCH_COLS[patch]
    cost=2*len(cols)
    if oracle.jobs+cost+3>args.jobs:return False
    J=np.empty((len(PATCH_ROWS[patch]),len(cols)))
    for k,j in enumerate(cols):
        derivative, curvature,_,_=probe(ctx,oracle,patch,q,int(j),args.h,
                                         20000+1000*({'center':1,'right':2,'left':3}[patch])+100*count+2*k,r)
        J[:,k]=derivative
    path=folder/('%s_refresh_%02d.npz'%(patch,count))
    if path.exists():raise RuntimeError('Existing refresh is protected: '+str(path))
    np.savez(path,q0=q.copy(),r0=r.copy(),selected_idx=cols.copy(),
             j_true=J,material=CASE,h=np.array([args.h]))
    cal['J'+patch[0]]=J
    cal['q'+patch[0]]=q.copy()
    return True


def solve(args):
    d=load_source();c=load_calibration(d)
    folder=OUT/args.tag
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Refusing to overwrite prior run: '+str(folder)+'. Choose a new --tag.')
    if args.jobs<6:raise ValueError('--jobs must allow at least two full Abaqus evaluations')
    ctx,oracle=new_oracle(folder,args.jobs,args.cpus)
    context_check(ctx,d,c['fs'])
    started=time.perf_counter()
    start=oracle.evaluate(d['q'])
    if not start['verified'] or oracle.jobs!=3:
        raise RuntimeError('Starting 65U full Abaqus verification failed')
    gap=rms(start['r']-d['r'])
    if gap>args.repro_tol or abs(start['rms']-rms(d['r']))>args.repro_tol:
        write_json(folder/'reproduction_failed.json',dict(saved=rms(d['r']),
             reproduced=start['rms'],residual_vector_gap=gap))
        raise RuntimeError('Saved 65U state not reproducible by current real FEM/NO runtime')
    best=current=start
    cal={'Jc':c['Jc'].copy(),'Jl':c['Jl'].copy(),'Jr':d['Jr'].copy(),
         'qc':d['q'].copy(),'ql':d['q'].copy(),'qr':d['qr'].copy()}
    refreshes=dict(center=0,right=0,left=0)
    H=np.zeros((26,26))
    radius=args.radius;damping=args.damping
    accepted=0;poor=0;history=[];stop='iteration_limit';partial=0
    checkpoint(folder,best,history,oracle,'starting_state',refreshes)
    for iteration in range(args.max_iter):
        if best['rms']<=args.tol:stop='converged';break
        if oracle.jobs+3>args.jobs:stop='job_budget';break
        q=np.asarray(current['q'],float).copy();r=np.asarray(current['r'],float).copy()
        # Prefer refreshing locally stale patches. Under a constrained budget,
        # preserve enough jobs for one complete verification of any trial.
        for patch, trigger, hard, maxrefresh in (
            ('center',args.center_refresh_at,args.center_hard,args.max_center_refresh),
            ('right',args.right_refresh_at,args.right_hard,args.max_right_refresh),
            ('left',args.left_refresh_at,args.left_hard,args.max_left_refresh)):
            drift=patch_drift(q,cal['q'+patch[0]],patch)
            global_drift=float(np.linalg.norm(q-cal['q'+patch[0]]))
            due=((drift>trigger or global_drift>args.global_refresh_at) and
                 poor>=args.poor_before_refresh) or drift>hard*.85 or global_drift>args.global_hard*.85
            if not due: continue
            if refreshes[patch]>=maxrefresh or not refresh(ctx,oracle,cal,q,r,patch,args,refreshes[patch]+1,folder):
                if drift>=hard*.90 or global_drift>=args.global_hard*.90:
                    stop=patch+'_recalibration_required'
                    break
            else:
                refreshes[patch]+=1; H[:,:]=0;poor=0
                print('65V measured %s Jacobian again; used %d/%d patch jobs'%(patch,oracle.jobs,args.jobs))
        if stop.endswith('_recalibration_required'):break
        if oracle.jobs+3>args.jobs:stop='job_budget';break
        pool,J=propose(oracle,q,r,cal,H,radius,damping,args)
        if not pool:
            # A no-descent model might need a refresh. If unable, stop safely.
            poor+=1; radius*=.75;damping=min(1e4,damping*1.7)
            history.append(dict(iteration=iteration,method='no_descent',accepted=False,
                verified=False,predicted_gain=0.,actual_gain='',gain_ratio='',trial_rms='',
                best_rms=best['rms'],jobs_this_trial=0,jobs_total=oracle.jobs,
                radius=radius,damping=damping,poor=poor))
            checkpoint(folder,best,history,oracle,'no_descent',refreshes)
            if radius<args.min_radius:stop='trust_region_small';break
            continue
        choice=pool[0];start_jobs=oracle.jobs
        trial=oracle.evaluate(choice['trial'],incumbent_r=r)
        verified=bool(trial['verified'])
        actual_gain='';gain_ratio='';good=False;secant=False
        if verified:
            actual_gain=objective(r)-objective(trial['r'])
            gain_ratio=actual_gain/max(choice['predicted'],1e-14)
            if actual_gain>args.min_actual_gain and gain_ratio>=args.min_gain_ratio:
                good=True
                svec=trial['q']-q; yvec=trial['r']-r
                H,secant=secant_update(H,J,svec,yvec,args.secant_blend)
                current=trial
                if trial['rms']<best['rms']:best=trial
                accepted+=1
                damping=max(1e-5,damping*(.75 if gain_ratio>.5 else .9))
                radius=min(args.max_radius,radius*(1.20 if gain_ratio>.7 else 1.06))
                poor=0 if gain_ratio>.3 else poor+1
        else: partial+=1
        if not good:
            poor+=1
            radius=max(args.min_radius*.8,radius*.65)
            damping=min(1e4,damping*2)
        drifts={p:patch_drift(current['q'],cal['q'+p[0]],p) for p in ('center','right','left')}
        history.append(dict(iteration=iteration,method=choice['method'],accepted=bool(good),
            verified=verified,trial_rms=trial.get('rms',''),best_rms=float(best['rms']),
            predicted_rms=choice['predicted_rms'],predicted_gain=choice['predicted'],
            actual_gain=actual_gain,gain_ratio=gain_ratio,
            jobs_this_trial=oracle.jobs-start_jobs,jobs_total=oracle.jobs,
            radius=radius,damping=damping,center_drift=drifts['center'],
            right_drift=drifts['right'],left_drift=drifts['left'],
            center_global_drift=float(np.linalg.norm(current['q']-cal['qc'])),
            right_global_drift=float(np.linalg.norm(current['q']-cal['qr'])),
            left_global_drift=float(np.linalg.norm(current['q']-cal['ql'])),secant_updated=secant,
            refresh_center=refreshes['center'],refresh_right=refreshes['right'],refresh_left=refreshes['left']))
        checkpoint(folder,best,history,oracle,'trial_%03d'%iteration,refreshes)
        print('65V %02d %-19s %-6s RMS %.8f  jobs %d/%d'%(
            iteration,choice['method'],'ACCEPT' if good else 'REJECT',
            best['rms'],oracle.jobs,args.jobs))
        if radius<args.min_radius:stop='trust_region_small';break
    else:stop='iteration_limit'
    # Always save the BEST verified complete three-patch state.
    save_best(folder,best,ctx,d['scale'])
    manifest=dict(method='65V center6/right2/left1 HF-calibrated LM + safe secants',
        initial_verified_rms=float(start['rms']),best_verified_rms=float(best['rms']),
        tolerance=args.tol,converged=bool(best['rms']<=args.tol),stop_reason=stop,
        accepted_steps=accepted,full_verified_evaluations=sum(bool(x['verified']) for x in oracle.log),
        partial_rejections=partial,initial_calibration_jobs=14,
        solver_patch_jobs=int(oracle.jobs),total_65V_patch_jobs=14+int(oracle.jobs),
        refreshes=refreshes,source=str(SOURCE),best_serial=int(best['serial']),
        runtime_seconds=time.perf_counter()-started,
        note='Abaqus verification of ALL three patches is mandatory before acceptance.')
    write_json(folder/'rom_solution_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:raise SystemExit(2)


def review(args):
    path=OUT/args.tag/'rom_solution_7region.npz'
    if not path.is_file():raise FileNotFoundError('Run --stage solve first: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        q=v(f,'q',26);r=v(f,'normalized_residual',26)
    total=float(r@r)
    groups=[]
    for name,a,b in zip(NAMES,ENDS[:-1],ENDS[1:]):
        groups.append(dict(interface=name,rms=rms(r[a:b]),
            squared_share_pct=100*float(r[a:b]@r[a:b])/total))
    modes=[dict(index=int(i),q=float(q[i]),normalized_residual=float(r[i]),
                squared_share_pct=100*float(r[i]**2)/total)
           for i in np.argsort(-np.abs(r))]
    result=dict(best_verified_rms=rms(r),converged=bool(rms(r)<=args.tol),
                remaining_reduction_pct=max(0.,100*(1-args.tol/rms(r))),
                interface_breakdown=groups,top_eight=modes[:8])
    write_json(OUT/(args.tag+'_review.json'),result)
    write_csv(OUT/(args.tag+'_modes.csv'),modes)
    print(json.dumps(result,indent=2))


def selftest():
    d=load_source()
    assert abs(rms(d['r'])-.02595398258943553)<1e-6
    assert d['Jr'].shape==(8,2)
    assert len(set(ACTIVE.tolist()))==9
    fun=lambda q:np.eye(26)@q
    _,J=numeric_jacobian(fun,np.zeros(26))
    assert np.allclose(J,np.eye(26),atol=1e-10)
    assert np.linalg.norm(lm(J,np.ones(26),.08,.1))<=.1000001
    H=np.zeros((26,26));s=np.ones(26)*.025
    Hnew,updated=secant_update(H,J,s,J@s,.7)
    assert updated and np.allclose(Hnew,H)
    assert np.allclose(Hnew[:,ACTIVE],0)
    print('65V selftest PASS: actual file schemas, Jacobian, LM, secant masks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('selftest','audit','calibrate','plan','solve','review'))
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--h',type=float,default=.02)
    p.add_argument('--jobs',type=int,default=60)
    p.add_argument('--tag',default='65V_D_lm_convergence')
    p.add_argument('--tol',type=float,default=.02)
    p.add_argument('--radius',type=float,default=.08)
    p.add_argument('--max-radius',type=float,default=.15)
    p.add_argument('--min-radius',type=float,default=.004)
    p.add_argument('--damping',type=float,default=.07)
    p.add_argument('--max-iter',type=int,default=55)
    p.add_argument('--repro-tol',type=float,default=.001)
    p.add_argument('--min-actual-gain',type=float,default=1e-11)
    p.add_argument('--min-gain-ratio',type=float,default=.025)
    p.add_argument('--secant-blend',type=float,default=.60)
    p.add_argument('--unmeasured-penalty',type=float,default=.25)
    p.add_argument('--refresh-penalty',type=float,default=.03)
    p.add_argument('--center-refresh-at',type=float,default=.12)
    p.add_argument('--right-refresh-at',type=float,default=.11)
    p.add_argument('--left-refresh-at',type=float,default=.13)
    p.add_argument('--center-hard',type=float,default=.19)
    p.add_argument('--right-hard',type=float,default=.19)
    p.add_argument('--left-hard',type=float,default=.20)
    p.add_argument('--global-refresh-at',type=float,default=.27)
    p.add_argument('--global-hard',type=float,default=.33)
    p.add_argument('--max-center-refresh',type=int,default=1)
    p.add_argument('--max-right-refresh',type=int,default=1)
    p.add_argument('--max-left-refresh',type=int,default=1)
    p.add_argument('--poor-before-refresh',type=int,default=2)
    a=p.parse_args()
    if not (a.cpus>=1 and a.jobs>=6 and .0<a.h<=.1 and
            .0<a.radius<=a.max_radius and a.tol>0 and a.damping>0 and
            0<a.center_refresh_at<a.center_hard and
            0<a.right_refresh_at<a.right_hard and
            0<a.left_refresh_at<a.left_hard and
            0<a.global_refresh_at<a.global_hard):
        raise ValueError('Invalid solver parameters')
    if not a.tag or '..' in a.tag or '/' in a.tag or '\\' in a.tag:
        raise ValueError('Use a simple output --tag')
    if a.stage=='selftest':selftest()
    elif a.stage=='audit':audit(a)
    elif a.stage=='calibrate':calibrate(a)
    elif a.stage=='plan':plan(a)
    elif a.stage=='solve':solve(a)
    elif a.stage=='review':review(a)

if __name__=='__main__':main()
