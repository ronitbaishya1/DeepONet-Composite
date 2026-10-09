"""STEP 65U: LM-dominant, secant-assisted two-patch FE-NO continuation.

Windows ODBS layout:
  scripts/65U_lm_secant_continuation.py    <-- this file
  scripts/65S_A_to_E.py
  scripts/65T_two_patch_adaptive.py
  scripts/65L_A_to_D_complete.py
  scripts/65O_adaptive_coupling.py
  online_results_7region_65T_middle/65T_D_two_patch_retry/
      rom_solution_7region.npz
      center_refresh_01.npz
  online_results_7region_65T_middle/65T_B_right_jacobian.npz

Run (from ODBS):
  abaqus python scripts\\65U_lm_secant_continuation.py --stage selftest
  abaqus python scripts\\65U_lm_secant_continuation.py --stage audit
  abaqus python scripts\\65U_lm_secant_continuation.py --stage plan
  abaqus python scripts\\65U_lm_secant_continuation.py --stage solve --cpus 4 --jobs 36
  abaqus python scripts\\65U_lm_secant_continuation.py --stage review

Stage audit/plan/selftest have NO FEM patch jobs. The solver ALWAYS runs
3 full FEM patches to reproduce the saved state; only full FEM evaluations
can be accepted. An optional right 4-patch remeasurement may be triggered
by poor prediction or local displacement drift. Expensive center refresh is
opt-in (--allow-center-refresh). No old 65K-65T result is overwritten.

Exit status 2 means the RMS=0.02 tolerance was not attained; output files
are still saved. Synthetic tests do not guarantee real Abaqus convergence.
"""
import argparse
import csv
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASE = np.array([46500., 12500., 4750.])
PREV = ROOT/'online_results_7region_65T_middle'
SOLVE_T = PREV/'65T_D_two_patch_retry'
SOURCE = SOLVE_T/'rom_solution_7region.npz'
CENTER_FILE = SOLVE_T/'center_refresh_01.npz'
RIGHT_FILE = PREV/'65T_B_right_jacobian.npz'
OUT = ROOT/'online_results_7region_65U_middle'
CROW = np.arange(8, 18)
RROW = np.arange(18, 26)
CIDX = np.array([8, 11, 13, 15, 16])
RIDX = np.array([20, 21])
ACTIVE = np.r_[CIDX, RIDX]
NAMES = ('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
ENDS = (0, 4, 8, 13, 18, 22, 26)


def rms(r):
    return float(np.sqrt(np.mean(np.asarray(r, float)**2)))


def objective(r):
    r=np.asarray(r,float)
    return float(r@r)/2.


def save_json(path, d):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f:json.dump(d,f,indent=2)


def save_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def require_vector(f,key,size):
    a=np.asarray(f[key],dtype=float).ravel().copy()
    if a.shape!=(size,) or not np.all(np.isfinite(a)):
        raise RuntimeError('Invalid %s: expected %d finite values'%(key,size))
    return a


def load_inputs():
    for path in (SOURCE,CENTER_FILE,RIGHT_FILE):
        if not path.is_file():
            raise FileNotFoundError('Required saved Windows file missing: '+str(path))
    with np.load(SOURCE,allow_pickle=False) as f:
        q=require_vector(f,'q',26);r=require_vector(f,'normalized_residual',26)
        scale=require_vector(f,'c_scale',26)
        stored=float(np.asarray(f['rms_residual']).ravel()[0])
        mat=np.array([float(np.asarray(f[x]).ravel()[0]) for x in ('E1','E2','G12')])
    with np.load(CENTER_FILE,allow_pickle=False) as f:
        qc=require_vector(f,'q0',26);rc=require_vector(f,'r0',26)
        jc=np.asarray(f['j_true_center'],float).copy()
        ci=np.asarray(f['selected_idx'],int).ravel()
    with np.load(RIGHT_FILE,allow_pickle=False) as f:
        qr=require_vector(f,'q0',26);rr=require_vector(f,'r0',26)
        jr=np.asarray(f['j_true_right'],float).copy()
        ri=np.asarray(f['selected_idx'],int).ravel()
        mat_r=require_vector(f,'material',3)
        scale_r=require_vector(f,'c_scale',26)
        fs=require_vector(f,'force_scale',26)
    if not np.allclose(mat,CASE,rtol=0,atol=1e-8) or not np.allclose(mat_r,CASE,rtol=0,atol=1e-8):
        raise RuntimeError('Material does not match 65U middle case')
    if not np.allclose(scale,scale_r,rtol=1e-12,atol=1e-14):
        raise RuntimeError('PCA coefficient scales differ across files')
    if np.any(scale<=0) or np.any(fs<=0) or np.max(np.abs(q))>3:
        raise RuntimeError('Invalid scale, force scale or q bounds')
    if not np.array_equal(ci,CIDX) or jc.shape!=(10,5) or not np.all(np.isfinite(jc)):
        raise RuntimeError('Expected five-column center Jacobian on [8,11,13,15,16]')
    if not np.array_equal(ri,RIDX) or jr.shape!=(8,2) or not np.all(np.isfinite(jr)):
        raise RuntimeError('Expected two-column right Jacobian on [20,21]')
    if abs(rms(r)-stored)>1e-9:
        raise RuntimeError('Saved best residual and rms_residual disagree')
    return dict(q=q,r=r,scale=scale,stored=stored,qc=qc,rc=rc,Jc=jc,
                qr=qr,rr=rr,Jr=jr,force_scale=fs)


def distances(q, qc, qr):
    return dict(center_global=float(np.linalg.norm(q-qc)),
                center_local=float(np.linalg.norm(q[CROW]-qc[CROW])),
                right_global=float(np.linalg.norm(q-qr)),
                right_local=float(np.linalg.norm(q[RROW]-qr[RROW])),
                right_selected=float(np.linalg.norm(q[RIDX]-qr[RIDX])))


def import_script(file,name):
    p=HERE/file
    if not p.is_file():raise FileNotFoundError('Missing existing Windows runtime: '+str(p))
    spec=importlib.util.spec_from_file_location(name,str(p))
    if spec is None or spec.loader is None:raise ImportError(str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def new_oracle(tag,budget,cpus):
    """Use original, previously tested 65L/65O runtime; new 65U workdir."""
    old=import_script('65L_A_to_D_complete.py','step65U_original65L')
    adapt=import_script('65O_adaptive_coupling.py','step65U_original65O')
    folder=OUT/tag
    folder.mkdir(parents=True,exist_ok=True)
    ctx=old.Coupling(CASE.copy(),str(folder))
    prior=ROOT/'rom_models_7region'/'65N_center_correction.npz'
    oracle=adapt.Oracle(ctx,str(folder),prior_path=str(prior),job_budget=budget,
                        cpus=cpus,adaptive_fidelity=True,online_correction=True)
    return ctx,oracle


def check_context(ctx,d):
    if not np.allclose(ctx.scale,d['scale'],rtol=1e-12,atol=1e-14):
        raise RuntimeError('Unexpected PCA coefficient scales in Coupling')
    if not np.allclose(ctx.force_scale,d['force_scale'],rtol=1e-12,atol=1e-14):
        raise RuntimeError('Unexpected generalized force normalization')


def numeric_jacobian(fun,q,eps=1e-3):
    q=np.asarray(q,float);f=np.asarray(fun(q),float)
    if f.shape!=(26,):raise RuntimeError('ROM residual must be (26,)')
    J=np.empty((26,26))
    for j in range(26):
        h=eps*max(1.,abs(q[j]));x=q.copy()
        x[j]=min(3.,q[j]+h)
        if abs(x[j]-q[j])<1e-12:x[j]=max(-3.,q[j]-h)
        if abs(x[j]-q[j])<1e-12:raise RuntimeError('Unperturbable q coordinate')
        J[:,j]=(np.asarray(fun(x),float)-f)/(x[j]-q[j])
    if not np.all(np.isfinite(J)):raise RuntimeError('Nonfinite ROM Jacobian')
    return f,J


def measured_model(oracle,q,Jc,Jr,H):
    f,J0=numeric_jacobian(oracle.corrected,q)
    C=np.zeros((26,26))
    C[np.ix_(CROW,CIDX)]=Jc-J0[np.ix_(CROW,CIDX)]
    C[np.ix_(RROW,RIDX)]=Jr-J0[np.ix_(RROW,RIDX)]
    J=J0+C+H
    return f,J,C


def lm(A,r,lam,rad):
    g=A.T@r
    G=A.T@A
    reg=np.maximum(np.diag(G),1e-5)
    step=-np.linalg.solve(G+lam*np.diag(reg),g)
    n=float(np.linalg.norm(step))
    if n>rad:step*=rad/n
    return step


def make_directions(J,r,radius,damping):
    choices=[]
    for name,idx in [('Full26-LM',np.arange(26)),('Joint7-LM',ACTIVE),
                     ('Center5-LM',CIDX),('Right2-LM',RIDX)]:
        try:
            p=np.zeros(26)
            p[idx]=lm(J[:,idx],r,damping,radius)
            if np.linalg.norm(p)>1e-12:choices.append((name,p))
        except np.linalg.LinAlgError:pass
    # Center-only Cauchy direction uses measured center sensitivities.
    g=J[np.ix_(CROW,CIDX)].T@r[CROW]
    ng=np.linalg.norm(g)
    if ng>1e-12:
        p=np.zeros(26);p[CIDX]=-radius*g/ng
        choices.append(('Center-Cauchy',p))
    return choices


def propose(oracle,q,r,J,C,H,fun_current,qc,qr,args,radius,damping):
    pool=[]
    for method,p0 in make_directions(J,r,radius,damping):
        for fraction in (1.0,0.5,0.25):
            p=p0*fraction;trial=q+p
            if np.max(np.abs(trial))>3:continue
            drifts=distances(trial,qc,qr)
            if (drifts['center_global']>args.center_max_global or
                drifts['right_global']>args.right_max_global or
                drifts['center_local']>args.center_max_local or
                drifts['right_local']>args.right_max_local):continue
            prediction=r+(np.asarray(oracle.corrected(trial),float)-fun_current)+(C+H)@p
            gain=objective(r)-objective(prediction)
            if not np.isfinite(gain) or gain<=1e-11:continue
            # Penalize reliance on regions where a new sensitivity measurement
            # could become necessary. This only ranks proposals; it does not
            # claim a rigorous uncertainty bound or measured runtime speedup.
            surcharge=0.0
            if drifts['right_local']>=args.right_refresh_at:surcharge+=4.0
            if drifts['center_local']>=args.center_refresh_at:surcharge+=10.0
            utility=gain/(3.0+args.cost_weight*surcharge)
            pool.append(dict(method=method,p=p,trial=trial,predicted_gain=float(gain),
                             predicted_rms=rms(prediction),utility=float(utility),
                             center_local=drifts['center_local'],
                             right_local=drifts['right_local']))
    pool.sort(key=lambda x:x['utility'],reverse=True)
    return pool


def apply_unmeasured_secant(H,J,s,y,blend=0.7):
    """Update ONLY non-calibrated columns; keep measured FEM blocks exact."""
    s=np.asarray(s,float);y=np.asarray(y,float)
    su=s.copy();su[ACTIVE]=0.0
    norm2=float(su@su)
    if norm2<1e-5:return H.copy(),False
    missing=y-(J@s)
    delta=blend*np.outer(missing,su)/(norm2+1e-4)
    # Cap updates to prevent one poorly conditioned secant from dominating.
    bound=max(1e-8,float(np.linalg.norm(J,'fro'))*0.5)
    norm=float(np.linalg.norm(delta,'fro'))
    if norm>bound:delta*=bound/norm
    updated=H+delta
    hbound=max(1e-8,float(np.linalg.norm(J,'fro')))
    hn=float(np.linalg.norm(updated,'fro'))
    if hn>hbound:updated*=hbound/hn
    return updated,True


def right_probe(ctx,oracle,q,idx,h,serial):
    # Exactly the same patch-residual normalization as the verified 65T code.
    t=import_script('65T_two_patch_adaptive.py','step65U_65T_helper')
    return t.right_probe(ctx,oracle,q,idx,h,serial)


def center_probe(ctx,oracle,q,idx,h,serial):
    s=import_script('65S_A_to_E.py','step65U_65S_helper')
    return s.probe(ctx,oracle,q,idx,h,serial)


def refresh_patch(ctx,oracle,patch,q,r,args,index,reference_dir):
    if patch=='right':
        idx=RIDX;J=np.empty((8,2));cost=4
        fn=right_probe
    else:
        idx=CIDX;J=np.empty((10,5));cost=10
        fn=center_probe
    if oracle.jobs+cost+3>args.jobs:
        return None
    if np.any(np.abs(q[idx]) + args.h > 3):
        raise RuntimeError('FD sensitivity outside bounded q range')
    for col,j in enumerate(idx):
        J[:,col]=fn(ctx,oracle,q,int(j),args.h,5000+100*index+2*col)[0]
    folder=OUT/reference_dir
    folder.mkdir(parents=True,exist_ok=True)
    fnp=folder/('%s_refresh_%02d.npz'%(patch,index))
    if fnp.exists():raise RuntimeError('Refusing to overwrite '+str(fnp))
    kw=dict(q0=q.copy(),r0=r.copy(),selected_idx=idx,j_true=J,
            h=np.array([args.h]),material=CASE)
    kw['j_true_right' if patch=='right' else 'j_true_center']=J
    np.savez(fnp,**kw)
    return J


def audit(args):
    d=load_inputs();dist=distances(d['q'],d['qc'],d['qr'])
    totals=float(d['r']@d['r'])
    groups=[]
    for name,a,b in zip(NAMES,ENDS[:-1],ENDS[1:]):
        y=d['r'][a:b]
        groups.append(dict(interface=name,rms=rms(y),squared_share_pct=100*float(y@y)/totals))
    modes=[dict(index=int(i),residual=float(d['r'][i]),share_pct=100*float(d['r'][i]**2)/totals)
           for i in np.argsort(-abs(d['r']))[:10]]
    result=dict(start_verified_rms=rms(d['r']),target=args.tol,
                material_MPa=CASE.tolist(),center_selected=CIDX.tolist(),
                right_selected=RIDX.tolist(),distances=dist,groups=groups,
                largest_components=modes,
                decision='Reuse center/right Jacobians; refresh selectively after bad gain ratios or patch-local drift',
                warning='A global displacement criterion is still enforced; patch-local drift alone is not a validated uncertainty measure.')
    save_json(OUT/'65U_A_audit.json',result)
    print(json.dumps(result,indent=2))


def plan(args):
    d=load_inputs();ctx,oracle=new_oracle('65U_B_plan_no_Abaqus',0,args.cpus)
    check_context(ctx,d)
    H=np.zeros((26,26))
    f,J,C=measured_model(oracle,d['q'],d['Jc'],d['Jr'],H)
    proposals=propose(oracle,d['q'],d['r'],J,C,H,f,d['qc'],d['qr'],args,args.radius,args.damping)
    top=[dict(method=p['method'],predicted_gain=p['predicted_gain'],
              predicted_rms=p['predicted_rms'],utility=p['utility'],
              step_norm=float(np.linalg.norm(p['p'])),
              center_local=p['center_local'],right_local=p['right_local'],
              step=p['p'].tolist()) for p in proposals[:8]]
    report=dict(start_verified_rms=rms(d['r']),candidates=top,
                distances=distances(d['q'],d['qc'],d['qr']),
                note='ROM/linear-model estimates only; no Abaqus jobs and no new verified result.')
    save_json(OUT/'65U_B_plan.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))
    for row in top[:5]:print('%-17s predicted RMS %.8f (step %.5f)'%
                             (row['method'],row['predicted_rms'],row['step_norm']))


def checkpoint(folder,best,history,oracle,tag):
    np.savez(folder/'best_verified_checkpoint.npz',q=best['q'],
             normalized_residual=best['r'],rms_residual=np.array([best['rms']]))
    save_csv(folder/'65U_C_history.csv',history)
    save_json(folder/'checkpoint.json',dict(phase=tag,patch_jobs=int(oracle.jobs),
                 best_rms=float(best['rms']),serial=int(best['serial'])))


def solve(args):
    if args.jobs<6:raise ValueError('At least 6 patch jobs required')
    d=load_inputs()
    folder=OUT/args.tag
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Existing run protected: '+str(folder)+'. Use another --tag for a new run.')
    ctx,oracle=new_oracle(args.tag,args.jobs,args.cpus)
    check_context(ctx,d)
    tic=time.perf_counter()
    initial=oracle.evaluate(d['q'])
    if not initial['verified'] or oracle.jobs!=3:
        raise RuntimeError('Full starting Abaqus reproduction failed')
    gap=rms(np.asarray(initial['r'])-d['r'])
    if gap>args.repro_tol or abs(initial['rms']-rms(d['r']))>args.repro_tol:
        save_json(folder/'reproduction_failed.json',dict(vector_gap_rms=gap,
                  computed_rms=initial['rms'],saved_rms=rms(d['r'])))
        raise RuntimeError('65T verified state not reproducible within threshold')
    best=current=initial
    qc=d['qc'].copy();qr=d['qr'].copy()
    Jc=d['Jc'].copy();Jr=d['Jr'].copy()
    H=np.zeros((26,26))
    radius=args.radius;damping=args.damping
    accepted=0;center_n=0;right_n=0;poor=0
    history=[];stop='iteration_limit';partial_count=0
    checkpoint(folder,best,history,oracle,'reproduced_start')
    for it in range(args.max_iter):
        if best['rms']<=args.tol:stop='converged';break
        if oracle.jobs+3>args.jobs:stop='job_budget';break
        q=np.asarray(current['q'],float).copy()
        r=np.asarray(current['r'],float).copy()
        dr=distances(q,qc,qr)
        # Do not refresh a patch just because other patches have changed.
        right_due=(dr['right_local']>=args.right_refresh_at and poor>=args.poor_before_refresh) or dr['right_local']>=args.right_max_local*0.85 or dr['right_global']>=args.right_max_global*0.9
        center_due=(dr['center_local']>=args.center_refresh_at and poor>=args.poor_before_refresh) or dr['center_local']>=args.center_max_local*0.85 or dr['center_global']>=args.center_max_global*0.9
        if right_due:
            if right_n>=args.max_right_refreshes:
                if dr['right_local']>=args.right_max_local*0.95 or dr['right_global']>=args.right_max_global*0.95:
                    stop='right_recalibration_limit';break
            else:
                new=refresh_patch(ctx,oracle,'right',q,r,args,right_n+1,args.tag)
                if new is not None:
                    Jr=new;qr=q.copy();right_n+=1;H[:,:]=0;poor=0
                    print('65U right two-column Abaqus refresh, four patch jobs')
                elif dr['right_local']>=args.right_max_local*0.95 or dr['right_global']>=args.right_max_global*0.95:
                    stop='right_refresh_budget';break
        if center_due:
            if args.allow_center_refresh and center_n<args.max_center_refreshes:
                new=refresh_patch(ctx,oracle,'center',q,r,args,center_n+1,args.tag)
                if new is not None:
                    Jc=new;qc=q.copy();center_n+=1;H[:,:]=0;poor=0
                    print('65U center five-column Abaqus refresh, ten patch jobs')
                elif dr['center_local']>=args.center_max_local*0.95 or dr['center_global']>=args.center_max_global*0.95:
                    stop='center_refresh_budget';break
            elif dr['center_local']>=args.center_max_local*0.95 or dr['center_global']>=args.center_max_global*0.95:
                stop='center_recalibration_required';break
        if oracle.jobs+3>args.jobs:stop='job_budget';break
        f,J,C=measured_model(oracle,q,Jc,Jr,H)
        pool=propose(oracle,q,r,J,C,H,f,qc,qr,args,radius,damping)
        if not pool:
            radius*=.68;damping=min(1e5,damping*1.7);poor+=1
            history.append(dict(iteration=it,method='no_descent',accepted=False,verified=False,
                predicted_gain=0.,actual_gain='',gain_ratio='',trial_rms='',best_rms=best['rms'],
                jobs_this_trial=0,jobs_total=oracle.jobs,radius=radius,damping=damping,
                center_local_distance=dr['center_local'],right_local_distance=dr['right_local'],
                secant_updated=False))
            checkpoint(folder,best,history,oracle,'no_descent')
            if radius<args.min_radius:stop='trust_region_small';break
            continue
        winner=pool[0]
        before=oracle.jobs
        candidate=oracle.evaluate(winner['trial'],incumbent_r=r)
        verified=bool(candidate['verified'])
        ratio='';actual='';okay=False;secant=False
        if verified:
            actual=objective(r)-objective(candidate['r'])
            ratio=actual/max(winner['predicted_gain'],1e-14)
            if actual>1e-11 and ratio>=args.min_ratio:
                okay=True
                svec=np.asarray(candidate['q'])-q
                yvec=np.asarray(candidate['r'])-r
                H,secant=apply_unmeasured_secant(H,J,svec,yvec,args.secant_blend)
                current=candidate
                if candidate['rms']<best['rms']:best=candidate
                accepted+=1
                radius=min(args.max_radius,radius*1.12)
                damping=max(1e-5,damping*.85)
                poor=0 if ratio>=args.low_ratio else poor+1
        else:partial_count+=1
        if not okay:
            radius*=.72;damping=min(1e5,damping*1.8);poor+=1
        history.append(dict(iteration=it,method=winner['method'],accepted=okay,
            verified=verified,predicted_gain=winner['predicted_gain'],
            actual_gain=actual,gain_ratio=ratio,trial_rms=candidate.get('rms',''),
            best_rms=best['rms'],jobs_this_trial=oracle.jobs-before,jobs_total=oracle.jobs,
            radius=radius,damping=damping,center_local_distance=dr['center_local'],
            right_local_distance=dr['right_local'],secant_updated=secant))
        checkpoint(folder,best,history,oracle,'trial_%03d'%it)
        print('65U %02d %-16s %-6s best RMS %.8f  patch jobs %d/%d'%
              (it,winner['method'],'ACCEPT' if okay else 'REJECT',best['rms'],oracle.jobs,args.jobs))
        if radius<args.min_radius:stop='trust_region_small';break
    else:stop='iteration_limit'
    s=import_script('65S_A_to_E.py','step65U_save_best_helper')
    s.save_best(str(folder),best,ctx,d['scale'])
    manifest=dict(method='65U LM-dominant mode-aware Jacobian, accepted HF secants',
        material_MPa=CASE.tolist(),start_source=str(SOURCE),
        initial_verified_rms=float(initial['rms']),best_verified_rms=float(best['rms']),
        tolerance=args.tol,converged=bool(best['rms']<=args.tol),stop_reason=stop,
        abaqus_patch_jobs=int(oracle.jobs),full_verified_evaluations=sum(bool(x['verified']) for x in oracle.log),
        partial_rejections=partial_count,accepted_steps=accepted,
        right_refreshes=right_n,center_refreshes=center_n,best_serial=int(best['serial']),
        runtime_seconds=time.perf_counter()-tic,
        warning='Only three-patch Abaqus evaluations establish convergence. This single trajectory is not a general benchmark.')
    save_json(folder/'rom_solution_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:raise SystemExit(2)


def review(args):
    path=OUT/args.tag/'rom_solution_7region.npz'
    if not path.is_file():raise FileNotFoundError('Run 65U --stage solve first: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        q=require_vector(f,'q',26);r=require_vector(f,'normalized_residual',26)
    total=float(r@r)
    groups=[]
    for name,a,b in zip(NAMES,ENDS[:-1],ENDS[1:]):
        v=r[a:b]
        groups.append(dict(interface=name,rms=rms(v),squared_share_pct=100*float(v@v)/total))
    rows=[dict(index=i,q=float(q[i]),residual=float(r[i]),
               squared_share_pct=100*float(r[i]**2)/total) for i in range(26)]
    rows.sort(key=lambda v:-abs(v['residual']))
    result=dict(best_verified_rms=rms(r),converged=rms(r)<=args.tol,
                interface_breakdown=groups,top_eight=rows[:8])
    save_json(OUT/(args.tag+'_review.json'),result)
    save_csv(OUT/(args.tag+'_modes.csv'),rows)
    print(json.dumps(result,indent=2))


def selftest():
    d=load_inputs()
    assert np.isclose(rms(d['r']),0.027654954317618403,atol=1e-7)
    assert d['Jc'].shape==(10,5) and d['Jr'].shape==(8,2)
    assert len(ACTIVE)==7 and len(set(ACTIVE.tolist()))==7
    I=np.eye(26);q=np.zeros(26);_,J=numeric_jacobian(lambda z:I@z,q)
    assert np.allclose(J,I,atol=1e-10)
    rr=np.ones(26)
    directions=make_directions(I,rr,.06,.1)
    assert directions and all(np.linalg.norm(p)<=.06000001 for _,p in directions)
    H=np.zeros_like(I);s=np.ones(26)*.02;y=I@s
    H2,ok=apply_unmeasured_secant(H,I,s,y)
    assert ok and np.allclose(H2,H)
    assert np.allclose(H2[:,ACTIVE],0)
    print('65U selftest PASS: real data schemas, numerical Jacobian, LM, constrained secants')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('selftest','audit','plan','solve','review'))
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--jobs',type=int,default=36,help='Includes starting 3 FEM patches and refreshes')
    p.add_argument('--tag',default='65U_C_lm_continuation')
    p.add_argument('--tol',type=float,default=.02)
    p.add_argument('--h',type=float,default=.02)
    p.add_argument('--radius',type=float,default=.055)
    p.add_argument('--max-radius',type=float,default=.10)
    p.add_argument('--min-radius',type=float,default=.005)
    p.add_argument('--damping',type=float,default=.12)
    p.add_argument('--repro-tol',type=float,default=.001)
    p.add_argument('--min-ratio',type=float,default=.01)
    p.add_argument('--low-ratio',type=float,default=.25)
    p.add_argument('--cost-weight',type=float,default=.6)
    p.add_argument('--secant-blend',type=float,default=.7)
    p.add_argument('--poor-before-refresh',type=int,default=2)
    p.add_argument('--max-iter',type=int,default=30)
    p.add_argument('--right-refresh-at',type=float,default=.075)
    p.add_argument('--center-refresh-at',type=float,default=.14)
    p.add_argument('--right-max-local',type=float,default=.16)
    p.add_argument('--center-max-local',type=float,default=.18)
    p.add_argument('--right-max-global',type=float,default=.26)
    p.add_argument('--center-max-global',type=float,default=.20)
    p.add_argument('--max-right-refreshes',type=int,default=1)
    p.add_argument('--max-center-refreshes',type=int,default=1)
    p.add_argument('--allow-center-refresh',action='store_true')
    a=p.parse_args()
    if not (a.jobs>=6 and a.cpus>=1 and 0<a.h<=.1 and 0<a.tol and
            0<a.radius<=a.max_radius and 0<a.secant_blend<=1):
        raise ValueError('Invalid job budget, step, tolerance or radius')
    if not a.tag or '/' in a.tag or '\\' in a.tag or '..' in a.tag:
        raise ValueError('Use a simple run tag')
    if a.stage=='selftest':selftest()
    elif a.stage=='audit':audit(a)
    elif a.stage=='plan':plan(a)
    elif a.stage=='solve':solve(a)
    elif a.stage=='review':review(a)

if __name__=='__main__':main()
