"""Step 65S: mode-adaptive center Abaqus Jacobian calibration and verified retry.

Windows commands (run from ODBS):
  abaqus python scripts\65S_A_to_E.py --stage selftest
  abaqus python scripts\65S_A_to_E.py --stage audit
  abaqus python scripts\65S_A_to_E.py --stage missing --cpus 4
  abaqus python scripts\65S_A_to_E.py --stage calibrate --cpus 4
  abaqus python scripts\65S_A_to_E.py --stage plan
  abaqus python scripts\65S_A_to_E.py --stage solve --cpus 4 --jobs 24

This uses the working 65L Coupling, 65O Oracle and 65R center patch adapter.
All stored historical experiments remain untouched.
Only COMPLETE 3-patch Abaqus evaluations count for convergence.
"""
import argparse
import csv
import importlib.util
import json
import os
import shutil
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASE = np.array([46500., 12500., 4750.])
SOURCE = os.path.join(ROOT, 'online_results_7region_65R_middle', '65R_D_retry',
                      'rom_solution_7region.npz')
OUT = os.path.join(ROOT, 'online_results_7region_65S_middle')
MISSING = os.path.join(OUT, '65S_A_missing_mode11.npz')
CALIB = os.path.join(OUT, '65S_B_five_column_jacobian.npz')
PLAN = os.path.join(OUT, '65S_C_plan.json')
CENTER = np.arange(8, 18)
IDX = np.array([8, 11, 13, 15, 16], dtype=int)
NAMES = ('left_outer', 'left_inner', 'center_left', 'center_right',
         'right_inner', 'right_outer')


def mkdir(p):
    os.makedirs(p, exist_ok=True)


def dump(p, obj):
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)


def tab(p, rows):
    if not rows:
        return
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a*a)))


def phi(a):
    a = np.asarray(a, dtype=float)
    return .5 * float(a @ a)


def getmod(filename, name):
    f = os.path.join(HERE, filename)
    if not os.path.isfile(f):
        raise FileNotFoundError('Required existing script: '+f)
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def state():
    if not os.path.isfile(SOURCE):
        raise FileNotFoundError('Step 65R best state is missing: '+SOURCE)
    with np.load(SOURCE) as f:
        q = np.asarray(f['q'], float).reshape(-1).copy()
        r = np.asarray(f['normalized_residual'], float).reshape(-1).copy()
        s = np.asarray(f['c_scale'], float).reshape(-1).copy()
        mu = np.array([float(np.asarray(f[k]).ravel()[0])
                       for k in ('E1', 'E2', 'G12')])
    if q.shape != (26,) or r.shape != (26,) or s.shape != (26,):
        raise RuntimeError('Saved state must contain 26 q, residual and scales.')
    if not np.allclose(mu, CASE, atol=1e-8, rtol=0):
        raise RuntimeError('Material mismatch with middle case.')
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)):
        raise RuntimeError('Invalid saved state.')
    return q, r, s


def context(tag, jobs, cpus):
    base = getmod('65L_A_to_D_complete.py', 'working_65L')
    mod = getmod('65O_adaptive_coupling.py', 'working_65O')
    work = os.path.join(OUT, tag)
    mkdir(work)
    ctx = base.Coupling(CASE, work)
    prior = os.path.join(ROOT, 'rom_models_7region',
                         '65N_center_correction.npz')
    oracle = mod.Oracle(ctx, work, prior_path=prior, job_budget=jobs,
                        cpus=cpus, adaptive_fidelity=True,
                        online_correction=True)
    return ctx, oracle


def scales_ok(ctx, coeff_scale, force_scale=None):
    if not np.allclose(ctx.scale, coeff_scale, rtol=1e-12, atol=1e-14):
        raise RuntimeError('Current model PCA scales differ from 65R.')
    if force_scale is not None and not np.allclose(
            ctx.force_scale, force_scale, rtol=1e-12, atol=1e-14):
        raise RuntimeError('Current force normalization differs from calibration.')


def center_real(ctx, oracle, q, serial):
    """One center Abaqus patch, with V4 NO forces evaluated at the SAME q."""
    _, _, _, no_g = ctx.rom(q, detailed=True)
    rec, fe = oracle._patch('center', q, serial, ctx.coeffs(q))
    actual = np.r_[fe['center_left'], fe['center_right']]
    no = np.r_[no_g['center_left'], no_g['center_right']]
    rr = (actual + no) / np.asarray(ctx.force_scale)[CENTER]
    if rr.shape != (10,) or not np.all(np.isfinite(rr)):
        raise RuntimeError('Center force block not finite 10-vector')
    return rr, rec


def numerical_jac(fun, q, step=1e-3):
    f0 = np.asarray(fun(q), float)
    J = np.empty((len(f0), len(q)), float)
    for j in range(len(q)):
        h = step * max(1., abs(float(q[j])))
        x = q.copy()
        x[j] = min(3., q[j]+h)
        if x[j] == q[j]:
            x[j] = max(-3., q[j]-h)
        if x[j] == q[j]:
            raise RuntimeError('Immovable interface coordinate '+str(j))
        J[:, j] = (np.asarray(fun(x))-f0)/(x[j]-q[j])
    return J


def probe(ctx, oracle, q, index, step, serial):
    if q[index]-step < -3. or q[index]+step > 3.:
        raise RuntimeError('Finite-difference step exceeds q bounds')
    qa = q.copy(); qa[index] += step
    qb = q.copy(); qb[index] -= step
    rp, fp = center_real(ctx, oracle, qa, serial)
    rm, fm = center_real(ctx, oracle, qb, serial+1)
    return (rp-rm)/(2*step), rp, rm, fp, fm


def audit():
    mkdir(OUT)
    q,r,s = state()
    frac = 100.*float(r[CENTER]@r[CENTER])/float(r@r)
    rows = []
    for j in range(26):
        rows.append({'index':j, 'q':q[j], 'residual':r[j],
                     'squared_residual':r[j]**2, 'selected':int(j in IDX)})
    tab(os.path.join(OUT, '65S_A_residual_modes.csv'),rows)
    summary = {'start_from':'65R_D_retry', 'saved_true_rms':rms(r),
               'center_squared_residual_pct':frac,
               'selected_q_indices':IDX.tolist(),
               'mode11_residual':float(r[11]),
               'sensitivity_patch_cost_for_missing_mode':2,
               'sensitivity_patch_cost_for_remaining_four':8}
    dump(os.path.join(OUT, '65S_A_audit.json'),summary)
    print(json.dumps(summary, indent=2))


def missing(args):
    """2 actual center Abaqus jobs for previously unmeasured center-left mode4."""
    if os.path.exists(MISSING):
        raise RuntimeError('65S-A existing output; not overwriting '+MISSING)
    mkdir(OUT)
    q,r,s = state()
    ctx,oracle = context('65S_A_missing_runs', 2, args.cpus)
    scales_ok(ctx,s)
    j, plus, minus, plusfile, minusfile = probe(
        ctx, oracle, q, 11, args.h, 200)
    # Cheap model derivative; diagnostic only, not full Jacobian.
    Jm = numerical_jac(oracle.corrected,q)[CENTER,11]
    sec = rms(plus + minus - 2*r[CENTER])
    cos = float(j@Jm / max(1e-14, np.linalg.norm(j)*np.linalg.norm(Jm)))
    proj = float(j@r[CENTER])
    # Model optimum in one direction; deliberately limited to diagnostic step.
    p = -proj/max(float(j@j), 1e-12)
    p = float(np.clip(p, -args.radius, args.radius))
    report = {'q_index':11,'mode':'center_left_m4','h':args.h,
              'true_jacobian_norm':float(np.linalg.norm(j)),
              'rom_jacobian_norm':float(np.linalg.norm(Jm)),
              'cosine':cos,'relative_column_error':float(np.linalg.norm(j-Jm)/max(np.linalg.norm(j),1e-12)),
              'center_second_difference_rms':sec,
              'predicted_q11_step':p,
              'center_rms_before':rms(r[CENTER]),
              'center_rms_after_linear_prediction':rms(r[CENTER]+j*p),
              'plus_odb':plusfile.get('odb_file'),
              'minus_odb':minusfile.get('odb_file'),
              'Abaqus_center_patch_jobs':oracle.jobs,
              'warning':'One-column local linear prediction is NOT a verified full-FEM improvement.'}
    np.savez(MISSING,q0=q,r0=r,c_scale=s,
             force_scale=np.asarray(ctx.force_scale),
             material=CASE,mode_index=np.array([11]),
             j_true=j,j_model=Jm,plus=plus,minus=minus,
             h=np.array([args.h]))
    dump(os.path.join(OUT,'65S_A_mode11_diagnostic.json'),report)
    print(json.dumps(report, indent=2))


def calibrate(args):
    """8 additional Abaqus center patch jobs; combines with saved mode 11."""
    if not os.path.isfile(MISSING):
        raise FileNotFoundError('Run --stage missing first.')
    if os.path.exists(CALIB):
        raise RuntimeError('65S-B existing output; not overwriting '+CALIB)
    q,r,s=state()
    with np.load(MISSING) as d:
        if not np.allclose(d['q0'],q,rtol=0,atol=1e-12):
            raise RuntimeError('Mode-11 diagnostic has different reference state.')
        if not np.isclose(float(d['h'][0]),args.h,rtol=0,atol=1e-10):
            raise RuntimeError('Use SAME --h as missing stage: '+str(d['h'][0]))
        j11 = d['j_true'].copy()
        force_scale = d['force_scale'].copy()
    ctx,oracle = context('65S_B_four_more_runs', 8, args.cpus)
    scales_ok(ctx,s,force_scale)
    Jtrue=np.empty((10,5),float)
    Jtrue[:,1]=j11
    Jmodel=numerical_jac(oracle.corrected,q)[np.ix_(CENTER,IDX)]
    rows=[]
    for col,index in enumerate(IDX):
        if index != 11:
            j,plus,minus,fp,fm = probe(ctx,oracle,q,int(index),args.h,300+2*col)
            Jtrue[:,col] = j
            second = rms(plus+minus-2*r[CENTER])
            odbplus,odbminus=fp.get('odb_file'),fm.get('odb_file')
        else:
            with np.load(MISSING) as d:
                second = rms(d['plus']+d['minus']-2*r[CENTER])
            odbplus=odbminus='(see 65S_A_mode11_diagnostic.json)'
        jt=Jtrue[:,col]; jm=Jmodel[:,col]
        rows.append({'q_index':int(index),'mode':'center_left_m4' if index==11 else 'previous_mode',
                     'true_J_norm':float(np.linalg.norm(jt)),
                     'model_J_norm':float(np.linalg.norm(jm)),
                     'relative_error':float(np.linalg.norm(jt-jm)/max(np.linalg.norm(jt),1e-12)),
                     'cosine':float(jt@jm/max(1e-14,np.linalg.norm(jt)*np.linalg.norm(jm))),
                     'second_difference_rms':second,
                     'plus_odb':odbplus,'minus_odb':odbminus})
    np.savez(CALIB,q0=q,r0=r,c_scale=s,force_scale=force_scale,
             material=CASE,selected_idx=IDX,center_idx=CENTER,
             j_true_center=Jtrue,j_model_center=Jmodel,
             h=np.array([args.h]))
    tab(os.path.join(OUT,'65S_B_Jacobian_column_comparison.csv'),rows)
    info={'measured_q_indices':IDX.tolist(),'jobs_missing_stage':2,
          'jobs_additional_stage':int(oracle.jobs),
          'total_center_patch_jobs_for_5_columns':2+oracle.jobs,
          'singular_values':np.linalg.svd(Jtrue,compute_uv=False).tolist(),
          'note':'Derivative measurements at 65R BEST q, no full re-verification in diagnostic stages.'}
    dump(os.path.join(OUT,'65S_B_manifest.json'),info)
    print(json.dumps(info,indent=2))


def load_calibration():
    if not os.path.isfile(CALIB):
        raise FileNotFoundError('Run --stage calibrate before plan/solve.')
    q,r,s=state()
    with np.load(CALIB) as d:
        if not np.allclose(d['q0'],q,atol=1e-12,rtol=0):
            raise RuntimeError('Calibrated Jacobian belongs to another starting q.')
        if not np.allclose(d['r0'],r,atol=1e-12,rtol=0):
            raise RuntimeError('Calibrated residual differs from saved 65R q.')
        if not np.array_equal(d['selected_idx'],IDX):
            raise RuntimeError('Expected 5 selected q indices.')
        return q,r,s,d['j_true_center'].copy(),d['force_scale'].copy()


def center_lm(J, r, radius, lam=0.1):
    H=J.T@J
    p=-np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-5)),J.T@r)
    return p*min(1.,radius/max(float(np.linalg.norm(p)),1e-14))


def plan(args):
    q,r,s,J,fs=load_calibration()
    d=center_lm(J,r[CENTER],args.radius)
    d_full=np.zeros(26); d_full[IDX]=d
    predicted=r.copy();predicted[CENTER]+=J@d
    report={'current_true_full_rms':rms(r),'current_center_rms':rms(r[CENTER]),
            'selected_q_indices':IDX.tolist(),'proposed_dq':d.tolist(),
            'radius':args.radius,
            'linearized_center_rms':rms(predicted[CENTER]),
            'full_rms_if_unchanged_other_patch_residuals':rms(predicted),
            'singular_values':np.linalg.svd(J,compute_uv=False).tolist(),
            'noncenter_only_rms_floor':float(np.sqrt(np.sum(r[np.r_[0:8,18:26]]**2)/26.)),
            'warning':'Linearized estimate only; no FEM verification, cross-patch effects ignored.'}
    dump(PLAN,report)
    print(json.dumps(report,indent=2))


def gmres(Aop, b, max_iter=20, tolerance=1e-4):
    """Small matrix-free GMRES for the 26-variable Newton correction."""
    b=np.asarray(b,float)
    beta=float(np.linalg.norm(b))
    if beta<1e-14:
        return np.zeros_like(b)
    n=len(b);kmax=min(max_iter,n)
    V=np.zeros((n,kmax+1));H=np.zeros((kmax+1,kmax))
    V[:,0]=b/beta
    x=np.zeros_like(b)
    for k in range(kmax):
        w=np.asarray(Aop(V[:,k]),float).copy()
        for j in range(k+1):
            H[j,k]=V[:,j]@w
            w-=H[j,k]*V[:,j]
        H[k+1,k]=np.linalg.norm(w)
        if H[k+1,k]>1e-13:
            V[:,k+1]=w/H[k+1,k]
        rhs=np.zeros(k+2);rhs[0]=beta
        y=np.linalg.lstsq(H[:k+2,:k+1],rhs,rcond=None)[0]
        x=V[:,:k+1]@y
        err=np.linalg.norm(rhs-H[:k+2,:k+1]@y)
        if err<=tolerance*beta or H[k+1,k]<=1e-13:
            break
    return x


def candidates(J,r,Jb,radius,damping):
    """Return candidate directions (method,p), including center-only fallback."""
    possibilities=[]
    # Fit full 26-dimensional model and restricted 10x5 local model.
    for method in ('Center-LM','Center-Cauchy','LM','Broyden','Newton-Krylov'):
        try:
            if method=='Center-LM':
                p=np.zeros(26);p[IDX]=center_lm(J[np.ix_(CENTER,IDX)],r[CENTER],radius,damping)
            elif method=='Center-Cauchy':
                p=np.zeros(26);p[IDX]=-J[np.ix_(CENTER,IDX)].T@r[CENTER]
            elif method=='LM':
                G=J.T@J
                p=-np.linalg.solve(G+damping*np.diag(np.maximum(np.diag(G),1e-5)),J.T@r)
            elif method=='Broyden':
                p=-np.linalg.lstsq(Jb+1e-4*np.eye(26),r,rcond=1e-7)[0]
            else:
                # Preconditioned GMRES applied to the locally calibrated
                # 26x26 Jacobian, with no additional Abaqus calls.
                P=np.linalg.pinv(J,rcond=1e-3)
                p=gmres(lambda v:P@(J@v),-P@r,max_iter=18)
            if not np.all(np.isfinite(p)) or np.linalg.norm(p)<1e-10:
                continue
            for mult in (1.,0.5,0.25,0.1):
                step=p*min(1.,radius*mult/max(np.linalg.norm(p),1e-14))
                possibilities.append((method,step))
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue
    return possibilities


def save_best(directory,best,ctx,scale):
    arrays={'q':best['q'],'c_scale':scale,'normalized_residual':best['r'],
            'rms_residual':np.array([best['rms']]),
            'E1':np.array([CASE[0]]),'E2':np.array([CASE[1]]),'G12':np.array([CASE[2]])}
    for name,arr in ctx.coeffs(best['q']).items():
        if name in NAMES:
            arrays['c_'+name]=arr
    np.savez(os.path.join(directory,'rom_solution_7region.npz'),**arrays)
    final=os.path.join(directory,'final')
    mkdir(final)
    for patch in ('left','center','right'):
        for key,ext in (('odb_file','.odb'),('input_file','.inp')):
            source=best['files'][patch][key]
            if not os.path.isfile(source):
                raise FileNotFoundError('Cannot archive verified FEM file: '+source)
            shutil.copy2(source,os.path.join(final,patch+ext))


def solve(args):
    q,r,s,Jtrue,force_scale=load_calibration()
    output=os.path.join(OUT,'65S_D_retry')
    manifest_path=os.path.join(output,'rom_solution_manifest.json')
    if os.path.isfile(manifest_path):
        raise RuntimeError('65S result exists: will not overwrite '+manifest_path)
    ctx,oracle=context('65S_D_retry',args.jobs,args.cpus)
    scales_ok(ctx,s,force_scale)
    if args.jobs<3:
        raise ValueError('Require budget for full 3-patch starting verification')
    start=time.perf_counter()
    first=oracle.evaluate(q)
    if not first['verified'] or rms(first['r']-r)>args.repro_tol:
        raise RuntimeError('65R best starting state was NOT reproducible with Abaqus')
    current=best=first
    qcal=q.copy()
    Jcal=Jtrue.copy()
    radius=args.radius
    damp=.1
    Jb=None
    logs=[]
    n_refresh=0
    stop_reason='iteration_limit'
    for iteration in range(args.max_iter):
        if best['rms']<=args.tol:
            stop_reason='converged';break
        if oracle.jobs+3>args.jobs:
            stop_reason='job_budget';break
        qk=current['q'].copy()
        drift=float(np.linalg.norm(qk-qcal))
        # Update high-fidelity calibration only if enough Abaqus budget
        # remains for ten center probes AND at least one full verification.
        if drift>=args.recalibrate_at:
            if n_refresh>=args.max_recalibrations or oracle.jobs+13>args.jobs:
                stop_reason='requires_recalibration_budget';break
            rows=[]
            Jnew=np.zeros((10,5))
            for j,index in enumerate(IDX):
                col,plus,minus,fp,fm=probe(ctx,oracle,qk,int(index),args.h,900+20*n_refresh+2*j)
                Jnew[:,j]=col
                rows.append({'index':int(index),'center_second_difference_rms':
                             rms(plus+minus-2*current['r'][CENTER])})
            # Save actual sensitivity calibration and its valid reference state.
            n_refresh+=1
            qcal=qk.copy();Jcal=Jnew
            np.savez(os.path.join(output,'refresh_%02d.npz'%n_refresh),
                     q0=qcal,r0=current['r'],j_true_center=Jcal,selected_idx=IDX)
            tab(os.path.join(output,'refresh_%02d.csv'%n_refresh),rows)
            Jb=None
            print('Recalibrated five center Jacobian columns at accepted state; total jobs:',oracle.jobs)
        basefun=oracle.corrected
        fhere=basefun(qk)
        J0=numerical_jac(basefun,qk)
        corr=np.zeros((26,26))
        corr[np.ix_(CENTER,IDX)]=Jcal-J0[np.ix_(CENTER,IDX)]
        Jeff=J0+corr
        def predictor(x):
            return current['r']+(basefun(x)-fhere)+corr@(x-qk)
        if Jb is None:
            Jb=Jeff.copy()
        pool=[]
        for method,p in candidates(Jeff,current['r'],Jb,radius,damp):
            trial=np.clip(qk+p,-3.,3.)
            if np.linalg.norm(trial-qk)<1e-8:
                continue
            if np.linalg.norm(trial-qcal)>args.max_calibration_distance:
                continue
            improvement=phi(current['r'])-phi(predictor(trial))
            if improvement>1e-10:
                pool.append((improvement,method,trial))
        if not pool:
            logs.append({'iteration':iteration,'method':'no_descent','accepted':False,
                         'verified':False,'trial_rms':'','best_rms':best['rms'],
                         'jobs_this_trial':0,'jobs_total':oracle.jobs,
                         'radius':radius,'damping':damp,'predicted_gain':0.,
                         'actual_gain':'','model_ratio':'','drift_from_calibration':drift})
            radius*=.65;damp=min(1e6,damp*2.)
            if radius<args.min_radius:
                stop_reason='trust_region_small';break
            continue
        # Select max predicted gain; ALWAYS Abaqus-verify before accepting.
        predicted,method,trial=max(pool,key=lambda x:x[0])
        used0=oracle.jobs
        attempt=oracle.evaluate(trial,incumbent_r=current['r'])
        accepted=False
        actual_gain='';rho=''
        if attempt['verified']:
            actual=phi(current['r'])-phi(attempt['r'])
            actual_gain=actual
            rho=actual/max(predicted,1e-12)
            if actual>1e-11 and rho>args.min_ratio:
                accepted=True
                svec=attempt['q']-current['q']
                yvec=attempt['r']-current['r']
                Jb+=np.outer(yvec-Jb@svec,svec)/max(1e-14,float(svec@svec))
                current=attempt
                if attempt['rms']<best['rms']:
                    best=attempt
                radius=min(args.max_radius, radius*1.2)
                damp=max(1e-6,damp*.8)
            else:
                radius*=.7;damp=min(1e6,damp*2.)
        else:
            radius*=.7;damp=min(1e6,damp*2.)
        logs.append({'iteration':iteration,'method':method,'accepted':bool(accepted),
                     'verified':bool(attempt['verified']),
                     'trial_rms':attempt.get('rms',''),'best_rms':best['rms'],
                     'jobs_this_trial':oracle.jobs-used0,'jobs_total':oracle.jobs,
                     'radius':radius,'damping':damp,'predicted_gain':predicted,
                     'actual_gain':actual_gain,'model_ratio':rho,
                     'drift_from_calibration':drift})
        tab(os.path.join(output,'65S_D_history.csv'),logs)
        print('Trial %02d %-15s %-8s best RMS %.6f jobs %d'%(
              iteration,method, 'ACCEPT' if accepted else 'REJECT',best['rms'],oracle.jobs))
        if radius<args.min_radius:
            stop_reason='trust_region_small';break
    else:
        stop_reason='iteration_limit'
    # Archive genuine best full-FEM solution even when convergence fails.
    tab(os.path.join(output,'65S_D_history.csv'),logs)
    save_best(output,best,ctx,s)
    manifest={'method':'five-center-mode true-Jacobian calibrated adaptive FE-NO',
              'material_MPa':CASE.tolist(),
              'initial_verified_rms':float(first['rms']),
              'best_verified_rms':float(best['rms']),
              'converged':bool(best['rms']<=args.tol),
              'tolerance':args.tol,
              'Abaqus_jobs_missing_mode':2,
              'Abaqus_jobs_four_more_columns':8,
              'Abaqus_jobs_solve':int(oracle.jobs),
              'Abaqus_jobs_total_65S':10+int(oracle.jobs),
              'refreshes':n_refresh,
              'accepted_steps':sum(int(row['accepted']) for row in logs),
              'partial_rejections':sum(not e['verified'] for e in oracle.log),
              'fully_verified_evaluations':sum(bool(e['verified']) for e in oracle.log),
              'best_fem_evaluation_serial':int(best['serial']),
              'stop_reason':stop_reason,'run_seconds':time.perf_counter()-start,
              'notes':'Includes 10 center patch sensitivity jobs. True convergence requires complete 3-patch FEM verification.'}
    dump(manifest_path,manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:
        raise SystemExit(2)


def review():
    """65S-E: offline diagnosis of the actual best 3-patch verified residual."""
    result=os.path.join(OUT,'65S_D_retry','rom_solution_7region.npz')
    if not os.path.isfile(result):
        raise FileNotFoundError('Run --stage solve before --stage review.')
    with np.load(result) as d:
        q=d['q'].copy();r=d['normalized_residual'].copy()
    bounds=(0,4,8,13,18,22,26)
    labels=NAMES
    total=float(r@r)
    groups=[]
    for i,name in enumerate(labels):
        a,b=bounds[i:i+2]
        groups.append({'interface':name,'rms':rms(r[a:b]),
                       'share_squared_pct':100*float(r[a:b]@r[a:b])/max(total,1e-14)})
    rows=sorted([{'q_index':i,'q':float(q[i]),'normalized_residual':float(v),
                  'share_squared_pct':100*float(v*v)/max(total,1e-14)}
                 for i,v in enumerate(r)],key=lambda x:abs(x['normalized_residual']),reverse=True)
    summary={'best_verified_rms':rms(r),'converged':rms(r)<=.02,
             'interface_breakdown':groups,'top_five_modes':rows[:5],
             'right_inner_modes_3_4_pct':sum(row['share_squared_pct'] for row in rows if row['q_index'] in (20,21)),
             'caution':'Cannot infer a successful true interface solve from ROM residual only.'}
    dump(os.path.join(OUT,'65S_E_review.json'),summary)
    tab(os.path.join(OUT,'65S_E_mode_ranking.csv'),rows)
    print(json.dumps(summary,indent=2))


def selftest():
    rng=np.random.default_rng(24)
    J=rng.standard_normal((10,5))*.2
    r=rng.standard_normal(10)*.05
    p=center_lm(J,r,.1)
    assert np.linalg.norm(p)<=.1000001
    assert phi(r+J@p)<phi(r)
    q=np.zeros(26)
    M=np.eye(26)
    assert np.allclose(numerical_jac(lambda x:M@x,q),M,atol=1e-9)
    full=np.eye(26)
    assert any(method=='Center-LM' for method,_ in candidates(full,np.ones(26),full,.1,.1))
    print('65S selftest PASSED: center LM, 26D Jacobian and proposals')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,
                   choices=('selftest','audit','missing','calibrate','plan','solve','review'))
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--h',type=float,default=.02)
    p.add_argument('--radius',type=float,default=.08)
    p.add_argument('--jobs',type=int,default=24,help='Abaqus patch-job budget for solve stage only')
    p.add_argument('--tol',type=float,default=.02)
    p.add_argument('--repro-tol',type=float,default=.01)
    p.add_argument('--max-iter',type=int,default=26)
    p.add_argument('--max-radius',type=float,default=.14)
    p.add_argument('--min-radius',type=float,default=.008)
    p.add_argument('--min-ratio',type=float,default=.01)
    p.add_argument('--recalibrate-at',type=float,default=.14,
                   help='Accepted state distance triggering optional new 10-job Jacobian measurement')
    p.add_argument('--max-calibration-distance',type=float,default=.20)
    p.add_argument('--max-recalibrations',type=int,default=1)
    args=p.parse_args()
    if not 0<args.h<=.1:raise ValueError('Require 0 < --h <= 0.1')
    if not 0<args.radius<=args.max_radius:raise ValueError('Invalid trust-region radius')
    if not args.max_calibration_distance>args.recalibrate_at:raise ValueError('Calibration domain must exceed refresh distance')
    if args.stage=='selftest':selftest()
    elif args.stage=='audit':audit()
    elif args.stage=='missing':missing(args)
    elif args.stage=='calibrate':calibrate(args)
    elif args.stage=='plan':plan(args)
    elif args.stage=='solve':solve(args)
    elif args.stage=='review':review()


if __name__=='__main__':
    main()
