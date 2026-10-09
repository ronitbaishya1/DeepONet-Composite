"""Step 65T: verified two-patch (CENTER + RIGHT) Jacobian coupling.

Place next to 65S_A_to_E.py, 65O_adaptive_coupling.py,
65L_A_to_D_complete.py inside Windows ODBS\\scripts.

Commands from ODBS:
  abaqus python scripts\\65T_two_patch_adaptive.py --stage selftest
  abaqus python scripts\\65T_two_patch_adaptive.py --stage audit
  abaqus python scripts\\65T_two_patch_adaptive.py --stage right --cpus 4
  abaqus python scripts\\65T_two_patch_adaptive.py --stage plan
  abaqus python scripts\\65T_two_patch_adaptive.py --stage solve --cpus 4 --jobs 24
  abaqus python scripts\\65T_two_patch_adaptive.py --stage review

Right calibration takes exactly FOUR right-patch Abaqus jobs. The solver
performs a new THREE-patch start reproducibility check; --jobs includes it.
All outputs go to a NEW 65T directory; no earlier output is modified.
Only THREE-patch full residuals may be accepted or marked converged.
"""
import argparse
import csv
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREV = os.path.join(ROOT, 'online_results_7region_65S_middle', '65S_D_from_verified')
SOURCE = os.path.join(PREV, 'rom_solution_7region.npz')
REFRESH = os.path.join(PREV, 'jacobian_refresh_01.npz')
OUT = os.path.join(ROOT, 'online_results_7region_65T_middle')
RIGHT_NPZ = os.path.join(OUT, '65T_B_right_jacobian.npz')
SOLVE = os.path.join(OUT, '65T_D_two_patch_retry')
CASE = np.array([46500.0, 12500.0, 4750.0])
CROW = np.arange(8, 18)
RROW = np.arange(18, 26)
CIDX = np.array([8, 11, 13, 15, 16], int)
RIDX = np.array([20, 21], int)
ACTIVE = np.r_[CIDX, RIDX]


def module(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError('Cannot import existing runtime: ' + path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def upstream():
    path = os.path.join(HERE, '65S_A_to_E.py')
    if not os.path.isfile(path):
        raise FileNotFoundError('Required existing 65S_A_to_E.py: ' + path)
    return module(path, 'step65S_runtime')


def put_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def put_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x*x)))


def objective(r):
    r = np.asarray(r, float)
    return 0.5*float(r@r)


def vector(d, key, n):
    a = np.asarray(d[key], float).ravel().copy()
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise RuntimeError('%s must be %d finite entries' % (key, n))
    return a


def load_inputs():
    for path in [SOURCE, REFRESH]:
        if not os.path.isfile(path):
            raise FileNotFoundError('Required Step 65S file missing: ' + path)
    with np.load(SOURCE, allow_pickle=False) as f:
        q = vector(f, 'q', 26)
        r = vector(f, 'normalized_residual', 26)
        cscale = vector(f, 'c_scale', 26)
        mat = np.array([float(f[k].ravel()[0]) for k in ['E1','E2','G12']])
    with np.load(REFRESH, allow_pickle=False) as f:
        qc = vector(f, 'q0', 26)
        rc = vector(f, 'r0', 26)
        indices = np.asarray(f['selected_idx'], int).ravel()
        Jc = np.asarray(f['j_true_center'], float).copy()
    if not np.allclose(mat, CASE, rtol=0, atol=1e-8):
        raise RuntimeError('65S file is not the required middle material')
    if not np.array_equal(indices, CIDX) or Jc.shape != (10, 5):
        raise RuntimeError('Refresh must contain five CENTER columns [8,11,13,15,16]')
    if not np.all(np.isfinite(Jc)) or not np.all(np.isfinite(cscale)) or np.any(cscale <= 0):
        raise RuntimeError('Invalid Jacobian or PCA scale')
    if not np.all(np.abs(q) <= 3):
        raise RuntimeError('Interface values exceed the existing [-3,3] bounds')
    distance = float(np.linalg.norm(q - qc))
    if distance > 0.2:
        raise RuntimeError('Saved CENTER Jacobian too far from verified solution: %.5f' % distance)
    return dict(q=q, r=r, scale=cscale, qc=qc, rc=rc, Jc=Jc, distance=distance)


def new_oracle(s, tag, budget, cpus):
    """Instantiate original 65L/65O runtimes with all 65T jobs under 65T OUT.

    Do NOT call s.context(), since that hardcodes the older 65S folder.
    """
    workdir = os.path.join(OUT, tag)
    os.makedirs(workdir, exist_ok=True)
    legacy = module(os.path.join(HERE, '65L_A_to_D_complete.py'), 'step65T_65L')
    adapt = module(os.path.join(HERE, '65O_adaptive_coupling.py'), 'step65T_65O')
    ctx = legacy.Coupling(CASE.copy(), workdir)
    prior = os.path.join(ROOT, 'rom_models_7region', '65N_center_correction.npz')
    oracle = adapt.Oracle(ctx, workdir, prior_path=prior, job_budget=budget,
                          cpus=cpus, adaptive_fidelity=True, online_correction=True)
    return ctx, oracle


def check_scale(s, ctx, d, extra_force_scale=None):
    s.scales_ok(ctx, d['scale'], extra_force_scale)
    if np.asarray(ctx.force_scale).shape != (26,) or np.any(np.asarray(ctx.force_scale) <= 0):
        raise RuntimeError('Unexpected 26-component force normalization')


def right_true(ctx, oracle, q, serial):
    """One real RIGHT patch; NO traction supplied by existing learned operator."""
    _, _, _, no_g = ctx.rom(q, detailed=True)
    record, fe_g = oracle._patch('right', q, serial, ctx.coeffs(q))
    names = ('right_inner', 'right_outer')
    fe = np.concatenate([np.asarray(fe_g[name], float).ravel() for name in names])
    no = np.concatenate([np.asarray(no_g[name], float).ravel() for name in names])
    y = (fe+no)/np.asarray(ctx.force_scale, float)[RROW]
    if y.shape != (8,) or not np.all(np.isfinite(y)):
        raise RuntimeError('Invalid true right-patch residual')
    return y, record


def right_probe(ctx, oracle, q, idx, h, serial):
    if q[idx]-h < -3 or q[idx]+h > 3:
        raise RuntimeError('Right sensitivity exceeds interface coordinate bounds')
    a = q.copy(); b = q.copy()
    a[idx] += h; b[idx] -= h
    yp, rp = right_true(ctx, oracle, a, serial)
    ym, rm = right_true(ctx, oracle, b, serial+1)
    return (yp-ym)/(2*h), yp, ym, rp, rm


def load_right(d):
    if not os.path.isfile(RIGHT_NPZ):
        raise FileNotFoundError('Run --stage right first: '+RIGHT_NPZ)
    with np.load(RIGHT_NPZ, allow_pickle=False) as f:
        qr = vector(f, 'q0', 26)
        rr = vector(f, 'r0', 26)
        m = np.asarray(f['material'], float).ravel()
        idx = np.asarray(f['selected_idx'], int).ravel()
        Jr = np.asarray(f['j_true_right'], float).copy()
        fs = vector(f, 'force_scale', 26)
    if not np.allclose(qr, d['q'], atol=1e-12, rtol=0):
        raise RuntimeError('Right Jacobian is not calibrated at saved 65S best q')
    if not np.allclose(rr, d['r'], atol=1e-12, rtol=0):
        raise RuntimeError('Right Jacobian reference residual mismatch')
    if not np.allclose(m, CASE, atol=1e-8, rtol=0):
        raise RuntimeError('Right Jacobian material mismatch')
    if not np.array_equal(idx, RIDX) or Jr.shape != (8, 2) or not np.all(np.isfinite(Jr)):
        raise RuntimeError('Right Jacobian must be 8x2 on modes 20 and 21')
    return Jr, fs


def audit():
    d=load_inputs()
    rows=[]; ends=(0,4,8,13,18,22,26)
    for k,name in enumerate(['left_outer','left_inner','center_left','center_right','right_inner','right_outer']):
        v=d['r'][ends[k]:ends[k+1]]
        rows.append(dict(interface=name, rms=rms(v),
                         squared_share_pct=100*float(v@v)/float(d['r']@d['r'])))
    report=dict(start_rms=rms(d['r']), required_rms=.02,
                center_reference_rms=rms(d['rc']), center_reference_distance=d['distance'],
                center_columns=CIDX.tolist(), right_new_columns=RIDX.tolist(),
                right_probe_cost=4, full_verification_cost=3, interface_breakdown=rows)
    put_json(os.path.join(OUT,'65T_A_audit.json'),report)
    print(json.dumps(report,indent=2))


def calibrate_right(args):
    right_dir=os.path.join(OUT, '65T_B_right_patch_runs')
    if (os.path.exists(RIGHT_NPZ) or
            (os.path.isdir(right_dir) and os.listdir(right_dir))):
        raise RuntimeError('Refusing to overwrite or reuse an existing right-patch calibration. Inspect: '+right_dir)
    d=load_inputs(); s=upstream()
    ctx,oracle=new_oracle(s,'65T_B_right_patch_runs',4,args.cpus)
    check_scale(s,ctx,d)
    # Freeze the ROM prediction BEFORE collecting any real right patch samples.
    Jmodel=s.numerical_jac(oracle.corrected,d['q'])[np.ix_(RROW,RIDX)]
    Jtrue=np.zeros((8,2)); lines=[]; paths=[]
    for col,idx in enumerate(RIDX):
        j, yp, ym, fp, fm=right_probe(ctx,oracle,d['q'],int(idx),args.h,500+2*col)
        Jtrue[:,col]=j
        cos=float(j@Jmodel[:,col]/max(np.linalg.norm(j)*np.linalg.norm(Jmodel[:,col]),1e-14))
        lines.append(dict(q_index=int(idx),interface_mode='right_inner_m%d'%(int(idx)-17),
                          norm_true=float(np.linalg.norm(j)), norm_model=float(np.linalg.norm(Jmodel[:,col])),
                          relative_error=float(np.linalg.norm(j-Jmodel[:,col])/max(np.linalg.norm(j),1e-14)),
                          cosine=cos, second_difference_rms=rms(yp+ym-2*d['r'][RROW])))
        paths.append(dict(q_index=int(idx),plus_odb=fp.get('odb_file'),minus_odb=fm.get('odb_file')))
    if oracle.jobs != 4:
        raise RuntimeError('Expected exactly four right Abaqus patch jobs')
    os.makedirs(OUT,exist_ok=True)
    np.savez(RIGHT_NPZ, q0=d['q'],r0=d['r'],material=CASE,c_scale=d['scale'],
             force_scale=np.asarray(ctx.force_scale),selected_idx=RIDX,
             right_idx=RROW,j_true_right=Jtrue,j_model_right=Jmodel,step=np.array([args.h]))
    put_csv(os.path.join(OUT,'65T_B_right_column_audit.csv'),lines)
    put_json(os.path.join(OUT,'65T_B_right_manifest.json'),
             dict(Abaqus_patch_jobs=oracle.jobs,sensitivities=lines,odb_paths=paths,
                  limitation='Two right-patch Jacobian columns only, at current verified 65S state.'))
    print('RIGHT PATCH calibration complete: four Abaqus jobs')
    print(json.dumps(lines,indent=2))


def corrected_model(s,oracle,q,center, right):
    raw=np.asarray(oracle.corrected(q),float)
    Jrom=s.numerical_jac(oracle.corrected,q)
    C=np.zeros((26,26))
    C[np.ix_(CROW,CIDX)]=center-Jrom[np.ix_(CROW,CIDX)]
    C[np.ix_(RROW,RIDX)]=right-Jrom[np.ix_(RROW,RIDX)]
    return raw,Jrom+C,C


def descent_directions(s,J,r,B,radius,damping):
    out=[]
    def lm(M,y,lam):
        G=M.T@M
        diag=np.maximum(np.diag(G),1e-5)
        return -np.linalg.solve(G+lam*np.diag(diag),M.T@y)
    try:
        z=lm(J[:,ACTIVE],r,damping)
        p=np.zeros(26);p[ACTIVE]=z;out.append(('Joint7-LM',p))
        z=lm(J[np.ix_(CROW,CIDX)],r[CROW],damping)
        p=np.zeros(26);p[CIDX]=z;out.append(('Center5-LM',p))
        z=lm(J[np.ix_(RROW,RIDX)],r[RROW],damping)
        p=np.zeros(26);p[RIDX]=z;out.append(('Right2-LM',p))
        p=lm(J,r,damping);out.append(('Full26-LM',p))
        p=-np.linalg.lstsq(B+1e-4*np.eye(26),r,rcond=1e-7)[0]
        out.append(('Broyden',p))
        Pinv=np.linalg.pinv(J,rcond=1e-3)
        p=s.gmres(lambda v:Pinv@(J@v),-Pinv@r,max_iter=18)
        out.append(('Newton-Krylov',p))
        for rows,cols,name in [(CROW,CIDX,'Center5-gradient'),(RROW,RIDX,'Right2-gradient')]:
            p=np.zeros(26);p[cols]=-J[np.ix_(rows,cols)].T@r[rows]
            out.append((name,p))
    except np.linalg.LinAlgError:
        return []
    selected=[]
    for name,p in out:
        if not np.all(np.isfinite(p)) or np.linalg.norm(p)<1e-12:
            continue
        for factor in [1.,.5,.25]:
            scale=min(1.,radius*factor/max(np.linalg.norm(p),1e-14))
            selected.append((name,p*scale))
    return selected


def build_plan(args):
    d=load_inputs(); Jr,_=load_right(d);s=upstream()
    ctx,oracle=new_oracle(s,'65T_C_rom_only',0,args.cpus)
    check_scale(s,ctx,d)
    f0,J,C=corrected_model(s,oracle,d['q'],d['Jc'],Jr)
    options=[]
    for method,p in descent_directions(s,J,d['r'],J,args.radius,.1):
        trial=d['q']+p
        if np.any(np.abs(trial)>3):continue
        if np.linalg.norm(trial-d['qc'])>args.max_center_distance:continue
        predicted=d['r']+(oracle.corrected(trial)-f0)+C@p
        gain=objective(d['r'])-objective(predicted)
        if gain>0:
            options.append(dict(method=method, predicted_gain=float(gain),
                                predicted_full_rms=rms(predicted),norm_step=float(np.linalg.norm(p)),
                                step=p.tolist()))
    options.sort(key=lambda t:t['predicted_gain'],reverse=True)
    report=dict(start_verified_rms=rms(d['r']),center_distance=d['distance'],
                center_jacobian_shape=[10,5],right_jacobian_shape=[8,2],
                right_selected=RIDX.tolist(),center_selected=CIDX.tolist(),
                top_candidates=options[:7],warning='Only a model prediction. Three-patch Abaqus verification needed.')
    put_json(os.path.join(OUT,'65T_C_plan.json'),report)
    print(json.dumps(report,indent=2))


def broyden(B,p,y):
    denom=float(p@p)
    if denom<1e-14:return B
    return B+np.outer(y-B@p,p)/denom


def checkpoint(best,oracle,history,phase):
    os.makedirs(SOLVE,exist_ok=True)
    np.savez(os.path.join(SOLVE,'best_verified_checkpoint.npz'),
             q=best['q'],normalized_residual=best['r'],rms_residual=np.array([best['rms']]))
    put_csv(os.path.join(SOLVE,'65T_D_history.csv'),history)
    put_json(os.path.join(SOLVE,'checkpoint.json'),
             dict(phase=phase,verified_best_rms=float(best['rms']),jobs=int(oracle.jobs),
                  verified_serial=int(best['serial'])))


def solve(args):
    if os.path.isdir(SOLVE) and os.listdir(SOLVE):
        raise RuntimeError('Existing Step 65T solver output will NOT be overwritten: '+SOLVE)
    if args.jobs<6:raise ValueError('Need at least six patch jobs: 3 starting + 3 trial')
    d=load_inputs(); Jr,fs=load_right(d); s=upstream()
    ctx,oracle=new_oracle(s,'65T_D_two_patch_retry',args.jobs,args.cpus)
    check_scale(s,ctx,d,fs)
    t0=time.perf_counter()
    initial=oracle.evaluate(d['q'],incumbent_r=None)
    if not initial['verified'] or oracle.jobs!=3:
        raise RuntimeError('Could not reproduce fully verified starting point')
    err=rms(initial['r']-d['r'])
    if err>args.repro_tol or abs(initial['rms']-rms(d['r']))>args.repro_tol:
        put_json(os.path.join(SOLVE,'reproduction_failed.json'),
                 dict(reference_rms=rms(d['r']),actual_rms=initial['rms'],vector_gap_rms=err))
        raise RuntimeError('Start FEM verification disagrees with saved 65S; aborted.')
    best=current=initial
    qcenter=d['qc'].copy(); Jcenter=d['Jc'].copy()
    qright=d['q'].copy(); Jright=Jr.copy()
    B=None; accepted=0; right_refreshes=0; center_refreshes=0
    radius=args.radius; damping=args.damping
    history=[]; reason='iteration_limit'; bad_predictions=0
    checkpoint(best,oracle,history,'reproduced_start')
    for iteration in range(args.max_iter):
        if best['rms']<=args.tol:reason='converged';break
        if oracle.jobs+3>args.jobs:reason='job_budget';break
        qk=np.asarray(current['q'],float); rk=np.asarray(current['r'],float)
        dc=float(np.linalg.norm(qk-qcenter)); dr=float(np.linalg.norm(qk-qright))
        # Refresh only if moved appreciably AND previously observed gain ratios
        # were poor, or the local Jacobian is about to leave its allowed region.
        need_right=(dr>=args.recalibrate_at and (bad_predictions>0 or dr>=args.max_right_distance*.90))
        need_center=(dc>=args.recalibrate_at and (bad_predictions>0 or dc>=args.max_center_distance*.90))
        if need_right:
            if right_refreshes<args.max_right_refreshes and oracle.jobs+4+3<=args.jobs:
                Jnew=np.zeros((8,2))
                for col,idx in enumerate(RIDX):
                    Jnew[:,col]=right_probe(ctx,oracle,qk,int(idx),args.h,
                                            2000+20*right_refreshes+2*col)[0]
                Jright=Jnew;qright=qk.copy();right_refreshes+=1;B=None;bad_predictions=0
                np.savez(os.path.join(SOLVE,'right_refresh_%02d.npz'%right_refreshes),
                         q0=qright,r0=rk,selected_idx=RIDX,j_true_right=Jright)
            elif dr>=args.max_right_distance*.95:
                reason='right_refresh_required_or_budget';break
        if need_center:
            if args.allow_center_refresh and center_refreshes<args.max_center_refreshes and oracle.jobs+10+3<=args.jobs:
                Jnew=np.zeros((10,5))
                for col,idx in enumerate(CIDX):
                    Jnew[:,col]=s.probe(ctx,oracle,qk,int(idx),args.h,
                                        3000+20*center_refreshes+2*col)[0]
                Jcenter=Jnew;qcenter=qk.copy();center_refreshes+=1;B=None;bad_predictions=0
                np.savez(os.path.join(SOLVE,'center_refresh_%02d.npz'%center_refreshes),
                         q0=qcenter,r0=rk,selected_idx=CIDX,j_true_center=Jcenter)
            elif dc>=args.max_center_distance*.95:
                reason='center_refresh_required_or_budget';break
        f0,J,C=corrected_model(s,oracle,qk,Jcenter,Jright)
        if B is None:
            B=J.copy()
            # Existing full HF secant from center refresh anchor to current q:
            # compatible only if it is not an intervening recalibration.
            if center_refreshes==0 and right_refreshes==0:
                delta=d['q']-d['qc'];dy=d['r']-d['rc']
                if np.linalg.norm(delta)>1e-10:
                    B=broyden(B,delta,dy)
        proposals=[]
        for method,p in descent_directions(s,J,rk,B,radius,damping):
            trial=qk+p
            if np.max(np.abs(trial))>3:continue
            dc_next=float(np.linalg.norm(trial-qcenter))
            dr_next=float(np.linalg.norm(trial-qright))
            if dc_next>args.max_center_distance or dr_next>args.max_right_distance:continue
            estimate=rk+(np.asarray(oracle.corrected(trial),float)-f0)+C@p
            gain=objective(rk)-objective(estimate)
            if not np.isfinite(gain) or gain<=1e-10:continue
            # Do not reward a proposal that moves close to expensive refresh limits.
            # Every complete trial costs 3 patches; high drift adds a conservative
            # expected calibration cost used ONLY for ranking, not actual jobs.
            overhead=(10 if dc_next>=args.recalibrate_at else 0)+(
                4 if dr_next>=args.recalibrate_at else 0)
            utility=gain/(3+args.refresh_cost_weight*overhead)
            proposals.append((float(utility),float(gain),method,p,trial,dc_next,dr_next))
        if not proposals:
            radius*=.7;damping=min(1e6,damping*1.8)
            history.append(dict(iteration=iteration,method='no_descent',accepted=False,
                verified=False,trial_rms='',best_rms=best['rms'],jobs_total=oracle.jobs,
                jobs_trial=0,predicted_gain=0.,actual_gain='',gain_ratio='',
                center_drift=dc,right_drift=dr,radius=radius,damping=damping))
            checkpoint(best,oracle,history,'no_descent')
            if radius<args.min_radius:reason='trust_region_small';break
            continue
        utility,predicted,method,p,trial,dc_next,dr_next=max(proposals,key=lambda x:x[0])
        before=oracle.jobs
        outcome=oracle.evaluate(trial,incumbent_r=rk)
        okay=False;actual='';ratio=''
        if outcome['verified']:
            actual=objective(rk)-objective(outcome['r'])
            ratio=actual/max(predicted,1e-14)
            if actual>1e-11 and ratio>=args.min_ratio:
                okay=True
                B=broyden(B,outcome['q']-qk,outcome['r']-rk)
                current=outcome
                if outcome['rms']<best['rms']:best=outcome
                accepted+=1
                radius=min(args.max_radius,radius*1.15)
                damping=max(1e-6,damping*.8)
                bad_predictions=0 if ratio>=args.low_ratio else bad_predictions+1
            else:
                bad_predictions+=1;radius*=.72;damping=min(1e6,damping*1.7)
        else:
            bad_predictions+=1;radius*=.72;damping=min(1e6,damping*1.7)
        history.append(dict(iteration=iteration,method=method,accepted=okay,
            verified=bool(outcome['verified']),trial_rms=outcome.get('rms',''),
            best_rms=best['rms'],jobs_total=oracle.jobs,jobs_trial=oracle.jobs-before,
            predicted_gain=predicted,actual_gain=actual,gain_ratio=ratio,
            center_drift=dc,right_drift=dr,radius=radius,damping=damping))
        checkpoint(best,oracle,history,'trial_%d'%iteration)
        print('65T %02d %-16s %-6s RMS %.8f  Abaqus jobs %d/%d'%(
            iteration,method,'ACCEPT' if okay else 'REJECT',best['rms'],oracle.jobs,args.jobs))
        if radius<args.min_radius:reason='trust_region_small';break
    else:
        reason='iteration_limit'
    s.save_best(SOLVE,best,ctx,d['scale'])
    manifest=dict(method='65T center/right Jacobian guided verified FE-NO coupling',
      start_source=SOURCE,initial_verified_rms=float(initial['rms']),best_verified_rms=float(best['rms']),
      converged=bool(best['rms']<=args.tol),target_rms=args.tol,stop_reason=reason,
      jobs_right_calibration=4,jobs_solver=int(oracle.jobs),
      total_new_patch_jobs=4+int(oracle.jobs),full_verified_evaluations=sum(
          int(t['verified']) for t in oracle.log),
      partial_rejections=sum(int(not t['verified']) for t in oracle.log),
      accepted_steps=accepted,right_recalibrations=right_refreshes,
      center_recalibrations=center_refreshes,run_seconds=time.perf_counter()-t0,
      best_evaluation_serial=int(best['serial']),
      note='Only fully verified 3-patch Abaqus states count toward convergence. Right initial calibration was performed separately.')
    put_json(os.path.join(SOLVE,'rom_solution_manifest.json'),manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:
        raise SystemExit(2)


def review():
    path=os.path.join(SOLVE,'rom_solution_7region.npz')
    if not os.path.isfile(path):raise FileNotFoundError('Run Step65T --stage solve first')
    with np.load(path,allow_pickle=False) as f:
        q=vector(f,'q',26); r=vector(f,'normalized_residual',26)
    ends=(0,4,8,13,18,22,26)
    names=['left_outer','left_inner','center_left','center_right','right_inner','right_outer']
    groups=[]
    for i,name in enumerate(names):
        y=r[ends[i]:ends[i+1]]
        groups.append(dict(interface=name,rms=rms(y),squared_share_pct=100*float(y@y)/float(r@r)))
    top=sorted([dict(q_index=int(i),q=float(q[i]),residual=float(r[i]),
                     squared_share_pct=100*float(r[i]*r[i])/float(r@r)) for i in range(26)],
               key=lambda x:-abs(x['residual']))
    report=dict(best_verified_rms=rms(r),converged=bool(rms(r)<=.02),
                interface_breakdown=groups,top_eight=top[:8])
    put_json(os.path.join(OUT,'65T_E_review.json'),report)
    put_csv(os.path.join(OUT,'65T_E_mode_ranking.csv'),top)
    print(json.dumps(report,indent=2))


def selftest():
    rng=np.random.default_rng(65)
    J=rng.normal(size=(26,26));r=rng.normal(size=26)
    choices=descent_directions(upstream(),J,r,J,.1,.1)
    assert choices
    assert all(np.linalg.norm(p)<=.100000001 for _,p in choices)
    assert np.linalg.norm(broyden(J,np.ones(26),np.ones(26))@np.ones(26)-np.ones(26))<1e-9
    d=load_inputs()
    assert d['distance']<.2
    print('65T numerical selftest PASSED; saved start RMS %.8f'%rms(d['r']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['selftest','audit','right','plan','solve','review'],required=True)
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--h',type=float,default=.02)
    p.add_argument('--jobs',type=int,default=24)
    p.add_argument('--tol',type=float,default=.02)
    p.add_argument('--radius',type=float,default=.06)
    p.add_argument('--max-radius',type=float,default=.12)
    p.add_argument('--min-radius',type=float,default=.006)
    p.add_argument('--damping',type=float,default=.1)
    p.add_argument('--min-ratio',type=float,default=.01)
    p.add_argument('--low-ratio',type=float,default=.25)
    p.add_argument('--repro-tol',type=float,default=.001)
    p.add_argument('--max-iter',type=int,default=30)
    p.add_argument('--recalibrate-at',type=float,default=.14)
    p.add_argument('--max-center-distance',type=float,default=.20)
    p.add_argument('--max-right-distance',type=float,default=.20)
    p.add_argument('--max-right-refreshes',type=int,default=1)
    p.add_argument('--allow-center-refresh',action='store_true',default=False)
    p.add_argument('--max-center-refreshes',type=int,default=1)
    p.add_argument('--refresh-cost-weight',type=float,default=.5)
    args=p.parse_args()
    if not (0<args.h<=.1 and 0<args.radius<=args.max_radius and .0<args.tol and
            .0<args.recalibrate_at<min(args.max_center_distance,args.max_right_distance)):
        raise ValueError('Invalid h, radius, tolerance, or distance thresholds')
    if args.jobs<6 or args.cpus<1:raise ValueError('Invalid Abaqus budget or cpus')
    actions={'selftest':lambda:selftest(),'audit':lambda:audit(),
             'right':lambda:calibrate_right(args),'plan':lambda:build_plan(args),
             'solve':lambda:solve(args),'review':lambda:review()}
    actions[args.stage]()


if __name__=='__main__':
    main()
