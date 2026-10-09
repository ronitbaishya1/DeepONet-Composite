"""Step 65S-D2: continue an Abaqus-calibrated FE–NO solver from the
verified Step 65S-C five-mode trial, NOT from the older 65R reference.

Place this file in ODBS/scripts next to:
    65S_A_to_E.py
    65L_A_to_D_complete.py
    65O_adaptive_coupling.py

Windows, from ODBS:
    abaqus python scripts\65S_D_continue_verified.py --stage audit
    abaqus python scripts\65S_D_continue_verified.py --stage selftest
    abaqus python scripts\65S_D_continue_verified.py --stage solve --cpus 4 --jobs 24
    abaqus python scripts\65S_D_continue_verified.py --stage review

All files are written under
    online_results_7region_65S_middle/65S_D_from_verified/
Old 65R, 65S-A/B/C, and 65S-D_retry results are never overwritten.
A restarted state is verified with all three Abaqus patches; only FULL
high-fidelity trials can be accepted or marked converged.
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
SFILE = os.path.join(HERE, '65S_A_to_E.py')
SOUT = os.path.join(ROOT, 'online_results_7region_65S_middle')
START_NPZ = os.path.join(SOUT, '65S_C_exact_step_verification', 'verified_trial.npz')
START_JSON = os.path.join(SOUT, '65S_C_exact_step_verification', 'verification.json')
OUT = os.path.join(SOUT, '65S_D_from_verified')
MANIFEST = os.path.join(OUT, 'rom_solution_manifest.json')
HISTORY = os.path.join(OUT, '65S_D2_history.csv')


def load_65s():
    if not os.path.isfile(SFILE):
        raise FileNotFoundError('Required existing code missing: '+SFILE)
    spec = importlib.util.spec_from_file_location('step_65s_original', SFILE)
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    return s


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_verified(s, atol=1e-10):
    """Validate provenance: 65R anchor -> 65S-C planned step -> FEM truth."""
    if not os.path.isfile(START_NPZ) or not os.path.isfile(START_JSON):
        raise FileNotFoundError('Need the original 65S_C_exact_step_verification files on Windows.')
    if not os.path.isfile(s.PLAN) or not os.path.isfile(s.CALIB):
        raise FileNotFoundError('Need the original 65S-B calibration and 65S-C plan on Windows.')

    qref, rref, scale, Jcenter, fscale = s.load_calibration()
    with np.load(START_NPZ, allow_pickle=False) as d:
        required = ('q', 'normalized_residual', 'q0', 'r0')
        missing = [k for k in required if k not in d.files]
        if missing:
            raise RuntimeError('Verified trial NPZ missing keys '+str(missing))
        q = np.asarray(d['q'], float).ravel().copy()
        r = np.asarray(d['normalized_residual'], float).ravel().copy()
        prev_q = np.asarray(d['q0'], float).ravel().copy()
        prev_r = np.asarray(d['r0'], float).ravel().copy()
    if any(x.shape != (26,) for x in (q, r, prev_q, prev_r)):
        raise RuntimeError('Expected all verified and reference vectors to have length 26.')
    if not all(np.all(np.isfinite(x)) for x in (q, r, prev_q, prev_r)):
        raise RuntimeError('Invalid NaN/Inf in verified state.')
    if np.max(np.abs(q)) > 3.0:
        raise RuntimeError('Verified state exceeds interface coordinate bounds.')
    if not np.allclose(prev_q, qref, atol=atol, rtol=0):
        raise RuntimeError('Verified trial q0 does not match the 65S-B calibration q0.')
    if not np.allclose(prev_r, rref, atol=atol, rtol=0):
        raise RuntimeError('Verified trial r0 does not match the 65S-B calibration r0.')

    with open(s.PLAN) as f:
        plan = json.load(f)
    with open(START_JSON) as f:
        result = json.load(f)
    indices = np.asarray(plan['selected_q_indices'], dtype=int)
    dq = np.asarray(plan['proposed_dq'], dtype=float)
    if not np.array_equal(indices, s.IDX) or dq.shape != (5,):
        raise RuntimeError('The verification does not match the five calibrated modes.')
    expected = prev_q.copy()
    expected[indices] += dq
    if not np.allclose(q, expected, atol=atol, rtol=0):
        raise RuntimeError('Verified q does not equal 65S-C planned q0 + dq.')
    if int(result['abaqus_patch_jobs']) != 3:
        raise RuntimeError('65S-C did not report three FEM patch jobs.')
    if not np.isclose(s.rms(r), float(result['actual_three_patch_full_rms']), atol=atol, rtol=0):
        raise RuntimeError('65S-C verification.json does not agree with verified NPZ.')
    if s.rms(r) >= s.rms(rref):
        raise RuntimeError('65S-C trial was not a verified improvement; inspect source files.')
    return dict(q=q, r=r, qref=qref, rref=rref, scale=scale,
                Jcenter=Jcenter, fscale=fscale,
                verification=result, distance=float(np.linalg.norm(q-qref)))


def groups(s, r):
    ends = (0,4,8,13,18,22,26)
    total = float(r @ r)
    return [dict(interface=name,
                 rms=s.rms(r[ends[i]:ends[i+1]]),
                 squared_share_pct=100.0*float(r[ends[i]:ends[i+1]] @ r[ends[i]:ends[i+1]])/max(total,1e-20))
            for i,name in enumerate(s.NAMES)]


def audit():
    s = load_65s()
    d = load_verified(s)
    r = d['r']
    k = np.argsort(-np.abs(r))[:8]
    info = dict(start_type='Step 65S-C fully FEM-verified',
                start_rms=s.rms(r), previous_rms=s.rms(d['rref']),
                improvement_percent=100*(1-s.rms(r)/s.rms(d['rref'])),
                distance_from_calibration=d['distance'],
                jacobian_center_shape=list(d['Jcenter'].shape),
                selected_indices=list(map(int,s.IDX)),
                interface_shares=groups(s,r),
                top_modes=[dict(index=int(i),residual=float(r[i]),
                                squared_share_pct=100*float(r[i]**2)/float(r@r)) for i in k],
                current_center_rms=s.rms(r[s.CENTER]),
                noncenter_full_rms_floor=float(np.sqrt(np.sum(np.r_[r[:8],r[18:]]**2)/26.)),
                note='The 65S-C state is existing FEM-verified evidence; the solve will reproduce it with 3 NEW patch jobs.')
    print(json.dumps(info,indent=2))
    # Audit writes its own file, but does NOT reserve the expensive solve directory.
    save_json(os.path.join(SOUT,'65S_D2_start_audit.json'), info)


def update_broyden(Jb, svec, yvec):
    denom = float(svec@svec)
    if denom <= 1e-14:
        return Jb.copy()
    return Jb + np.outer(yvec-Jb@svec, svec)/denom


def selftest():
    rng = np.random.default_rng(54)
    B = rng.standard_normal((26,26))
    v = rng.standard_normal(26)
    y = rng.standard_normal(26)
    B2 = update_broyden(B,v,y)
    assert np.linalg.norm(B2@v-y)<1e-10
    assert np.all(np.isfinite(B2))
    s = load_65s()
    s.selftest()
    print('65S-D2 continuation selftest PASSED')


def save_checkpoint(s, best, oracle, rows, phase):
    """Readable recovery record, not a claim of convergence."""
    np.savez(os.path.join(OUT,'best_verified_checkpoint.npz'),
             q=np.asarray(best['q'],float),
             normalized_residual=np.asarray(best['r'],float),
             rms_residual=np.array([best['rms']]))
    save_csv(HISTORY, rows)
    save_json(os.path.join(OUT,'checkpoint.json'),
              dict(phase=phase, best_rms=float(best['rms']),
                   high_fidelity_patch_jobs=int(oracle.jobs),
                   evaluations=int(oracle.evaluation),
                   last_fully_verified_serial=int(best['serial']),
                   warning='Only complete Abaqus evaluations are accepted.'))


def solve(args):
    s = load_65s()
    d = load_verified(s)
    if os.path.isdir(OUT) and os.listdir(OUT):
        raise RuntimeError('The continuation directory already contains output: '+OUT+'\nRefusing to overwrite any existing Abaqus work. Review/rename it before a deliberate rerun.')
    if args.jobs < 6:
        raise ValueError('Need at least 6 patch jobs (3 to reproduce + at least 3 for a trial).')
    if d['distance'] >= args.recalibrate_at:
        raise RuntimeError('Starting state is already beyond recalibration trigger; remeasure Jacobian first.')
    if args.max_calibration_distance <= args.recalibrate_at:
        raise ValueError('max calibration distance must exceed recalibration trigger.')
    start_time = time.perf_counter()
    ctx, oracle = s.context('65S_D_from_verified', args.jobs, args.cpus)
    s.scales_ok(ctx, d['scale'], d['fscale'])
    print('Reproducing the previous fully verified 65S-C starting state (3 Abaqus patches).')
    first = oracle.evaluate(d['q'], incumbent_r=None)
    if not first['verified'] or oracle.jobs != 3:
        raise RuntimeError('Starting solution could not be fully reproduced.')
    vector_gap = s.rms(first['r']-d['r'])
    if vector_gap > args.repro_vector_tol or abs(first['rms']-s.rms(d['r'])) > args.repro_rms_tol:
        save_json(os.path.join(OUT,'start_reproduction_failure.json'),
                  dict(expected_rms=s.rms(d['r']),actual_rms=float(first['rms']),
                       normalized_residual_vector_gap=vector_gap,
                       message='Aborted: do not accept an unrepeatable reference state.'))
        raise RuntimeError('Abaqus failed to reproduce the previous verified solution; review start_reproduction_failure.json')

    current = best = first
    qcal = d['qref'].copy()       # J was measured at 65R, NOT at the new verified state
    Jcal = d['Jcenter'].copy()
    Jb = None
    radius = args.radius
    damping = args.damping
    accepted = 0
    refreshes = 0
    history = []
    stop_reason = 'iteration_limit'
    save_checkpoint(s,best,oracle,history,'reproduced_start')

    for iteration in range(args.max_iter):
        if best['rms'] <= args.tol:
            stop_reason = 'converged';break
        if oracle.jobs + 3 > args.jobs:
            stop_reason = 'job_budget';break

        qk = np.asarray(current['q']).copy()
        rk = np.asarray(current['r']).copy()
        drift = float(np.linalg.norm(qk-qcal))

        # Refresh is optional and may not exceed total retry job budget.
        # If refresh is needed but cannot be afforded, STOP rather than
        # extrapolating an old FEM Jacobian beyond its intended neighborhood.
        if drift >= args.recalibrate_at:
            if (refreshes >= args.max_recalibrations
                    or oracle.jobs + 10 + 3 > args.jobs):
                stop_reason = 'needs_jacobian_refresh_or_budget';break
            Jfresh = np.empty((10,5))
            rows = []
            # Unique serial range; normal trial evaluations use 0,1,...
            for col,idx in enumerate(s.IDX):
                serial = 800 + refreshes*30 + 2*col
                j,plus,minus,fp,fm = s.probe(ctx,oracle,qk,int(idx),args.h,serial)
                Jfresh[:,col] = j
                rows.append(dict(index=int(idx),
                                 second_difference_rms=s.rms(plus+minus-2*rk[s.CENTER]),
                                 plus_odb=fp.get('odb_file'),minus_odb=fm.get('odb_file')))
            refreshes += 1
            qcal = qk.copy()
            Jcal = Jfresh.copy()
            np.savez(os.path.join(OUT,'jacobian_refresh_%02d.npz'%refreshes),
                     q0=qcal, r0=rk,selected_idx=s.IDX,
                     j_true_center=Jcal)
            save_csv(os.path.join(OUT,'jacobian_refresh_%02d.csv'%refreshes),rows)
            Jb = None
            save_checkpoint(s,best,oracle,history,'jacobian_refreshed')

        basefun = oracle.corrected
        f0 = np.asarray(basefun(qk), float)
        Jrom = s.numerical_jac(basefun,qk)
        correction = np.zeros((26,26))
        correction[np.ix_(s.CENTER,s.IDX)] = Jcal-Jrom[np.ix_(s.CENTER,s.IDX)]
        Jeff = Jrom+correction
        if Jb is None:
            Jb = Jeff.copy()
            # Reuse a real 65R->65S-C full-FEM secant that cost ZERO new jobs.
            # It informs only the separate Broyden proposal, and does not
            # overwrite the five measured center-Jacobian columns in Jeff.
            if refreshes == 0:
                Jb = update_broyden(Jb,d['q']-d['qref'],d['r']-d['rref'])

        def predictor(x):
            return rk + (np.asarray(basefun(x),float)-f0) + correction@(x-qk)

        pool = []
        for method,step in s.candidates(Jeff,rk,Jb,radius,damping):
            trial = qk+step
            # Do not clip here: clipping changes the searched direction.
            if np.any(trial < -3.) or np.any(trial > 3.):
                continue
            if float(np.linalg.norm(step)) < 1e-9:
                continue
            if np.linalg.norm(trial-qcal) > args.max_calibration_distance:
                continue
            estimate = s.phi(rk)-s.phi(predictor(trial))
            if np.isfinite(estimate) and estimate>1e-10:
                pool.append((float(estimate),method,trial))

        if not pool:
            history.append(dict(iteration=iteration,method='no_descent',verified=False,
                                accepted=False,trial_rms='',best_rms=float(best['rms']),
                                jobs_this_trial=0,jobs_total=int(oracle.jobs),
                                radius=radius,damping=damping,drift=drift,
                                predicted_phi_decrease=0.,actual_phi_decrease='',
                                gain_ratio=''))
            radius *= 0.7
            damping = min(1e6,damping*1.8)
            save_checkpoint(s,best,oracle,history,'no_descent')
            if radius < args.min_radius:
                stop_reason = 'trust_region_small';break
            continue

        predicted,method,trial = max(pool,key=lambda x:x[0])
        n0 = oracle.jobs
        attempt = oracle.evaluate(trial,incumbent_r=rk)
        gain = ''
        rho = ''
        ok = False
        if attempt['verified']:
            gain = s.phi(rk)-s.phi(attempt['r'])
            rho = gain/max(predicted,1e-12)
            if gain>1e-11 and rho>args.min_ratio:
                ok = True
                Jb = update_broyden(Jb,attempt['q']-qk,attempt['r']-rk)
                current = attempt
                if attempt['rms'] < best['rms']:
                    best = attempt
                accepted += 1
                radius = min(args.max_radius,radius*1.2)
                damping = max(1e-6,damping*0.8)
            else:
                radius *= 0.7
                damping = min(1e6,damping*1.8)
        else:
            radius *= 0.7
            damping = min(1e6,damping*1.8)

        history.append(dict(iteration=iteration,method=method,
                            verified=bool(attempt['verified']),accepted=bool(ok),
                            trial_rms=float(attempt['rms']) if attempt['verified'] else '',
                            best_rms=float(best['rms']),jobs_this_trial=int(oracle.jobs-n0),
                            jobs_total=int(oracle.jobs),radius=radius,damping=damping,
                            drift=drift,predicted_phi_decrease=predicted,
                            actual_phi_decrease=gain,gain_ratio=rho))
        save_checkpoint(s,best,oracle,history,'iteration_%d'%iteration)
        print('65S-D2 %02d %-15s %-6s best FEM RMS %.8f patch jobs %d/%d'%
              (iteration,method,'ACCEPT' if ok else 'REJECT',best['rms'],oracle.jobs,args.jobs))
        if radius < args.min_radius:
            stop_reason='trust_region_small';break
    else:
        stop_reason='iteration_limit'

    # Always archive the best FULLY VERIFIED ODB + input files.
    s.save_best(OUT,best,ctx,d['scale'])
    manifest = dict(method='65S-D2 verified continuation with five-column Abaqus-calibrated Jacobian',
                    initial_verified_rms=float(first['rms']),
                    best_verified_rms=float(best['rms']),
                    converged=bool(best['rms']<=args.tol),
                    tolerance=args.tol,
                    material_MPa=s.CASE.tolist(),
                    start_source=START_NPZ,
                    baseline_verified_reference_rms=float(s.rms(d['r'])),
                    starting_calibration_distance=d['distance'],
                    accepted_steps=int(accepted),
                    patch_jobs_this_continuation=int(oracle.jobs),
                    full_verified_evaluations=sum(bool(x['verified']) for x in oracle.log),
                    partial_rejections=sum(not bool(x['verified']) for x in oracle.log),
                    jacobian_refresh_count=refreshes,
                    stop_reason=stop_reason,
                    runtime_seconds=float(time.perf_counter()-start_time),
                    best_evaluation_serial=int(best['serial']),
                    note='The first 3 patch jobs reproduce the previously verified 65S-C state; 65S-A/B/C jobs are not charged again. All accepted states underwent full 3-patch Abaqus verification.')
    save_json(MANIFEST,manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:
        raise SystemExit(2)


def review():
    s=load_65s()
    path=os.path.join(OUT,'rom_solution_7region.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError('Run --stage solve first: '+path)
    with np.load(path,allow_pickle=False) as d:
        q=np.asarray(d['q'],float)
        r=np.asarray(d['normalized_residual'],float)
    rows = [dict(q_index=int(i), q=float(q[i]), normalized_residual=float(r[i]),
                 squared_share_pct=100*float(r[i]**2)/float(r@r))
            for i in np.argsort(-np.abs(r))]
    info=dict(best_verified_rms=s.rms(r),converged=bool(s.rms(r)<=0.02),
              interface_breakdown=groups(s,r),top_eight=rows[:8],
              caution='All claimed improvements are based on fully verified Abaqus patch evaluations.')
    save_json(os.path.join(SOUT,'65S_D2_review.json'),info)
    save_csv(os.path.join(SOUT,'65S_D2_mode_ranking.csv'),rows)
    print(json.dumps(info,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('audit','selftest','solve','review'),required=True)
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--jobs',type=int,default=24,help='NEW Abaqus patch-job budget for continuation')
    p.add_argument('--tol',type=float,default=.02)
    p.add_argument('--radius',type=float,default=.06)
    p.add_argument('--min-radius',type=float,default=.007)
    p.add_argument('--max-radius',type=float,default=.12)
    p.add_argument('--damping',type=float,default=.1)
    p.add_argument('--min-ratio',type=float,default=.01)
    p.add_argument('--max-iter',type=int,default=30)
    p.add_argument('--h',type=float,default=.02)
    p.add_argument('--recalibrate-at',type=float,default=.14)
    p.add_argument('--max-calibration-distance',type=float,default=.20)
    p.add_argument('--max-recalibrations',type=int,default=1)
    p.add_argument('--repro-vector-tol',type=float,default=1e-3)
    p.add_argument('--repro-rms-tol',type=float,default=5e-4)
    args=p.parse_args()
    if args.stage=='audit': audit();return
    if args.stage=='selftest': selftest();return
    if args.stage=='review': review();return
    if not (0<args.radius<=args.max_radius and args.recalibrate_at>0 and
            args.max_calibration_distance>args.recalibrate_at):
        raise ValueError('Invalid trust-radius or calibration distance settings.')
    if args.jobs<6 or not(0<args.h<=.1) or args.max_iter<1:
        raise ValueError('Invalid Abaqus job budget, finite-difference step, or iteration count.')
    solve(args)


if __name__=='__main__':
    main()
