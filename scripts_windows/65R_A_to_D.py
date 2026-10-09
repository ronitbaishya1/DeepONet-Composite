"""Step 65R: center-interface sensitivity audit and guarded adaptive retry.

Windows: abaqus python scripts\\65R_A_to_D.py --stage selftest
         abaqus python scripts\\65R_A_to_D.py --stage audit
         abaqus python scripts\\65R_A_to_D.py --stage sensitivity --cpus 4
         abaqus python scripts\\65R_A_to_D.py --stage plan
         abaqus python scripts\\65R_A_to_D.py --stage solve --cpus 4 --jobs 24

This keeps Step65H/65K/65M/65P unchanged. Does NOT claim convergence
from partially evaluated trials. Requires the existing working 65L and
65O modules in the same Windows scripts directory.
"""
from __future__ import print_function
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
CASE = (46500.0, 12500.0, 4750.0)
SOURCE = os.path.join(ROOT, 'online_results_7region_65P_middle_adaptive_selective')
SOLUTION = os.path.join(SOURCE, 'rom_solution_7region.npz')
PRIOR = os.path.join(ROOT, 'rom_models_7region', '65N_center_correction.npz')
OUT = os.path.join(ROOT, 'online_results_7region_65R_middle')
DIAG = os.path.join(OUT, '65R_B_sensitivity.npz')
NAMES = ('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
CENTER_IDX = np.arange(8,18)
# Center-left mode 1; center-right modes 1, 3, 4 (zero-based q indices).
SELECTED = np.array([8,13,15,16], dtype=int)
SELECTED_NAMES = ('center_left_m1','center_right_m1','center_right_m3','center_right_m4')


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def csv_out(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def norm_rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x*x)))


def obj(r):
    return 0.5 * float(np.dot(r,r))


def module(filename, name):
    path = os.path.join(HERE, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError('Required working module missing: '+path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_state():
    if not os.path.isfile(SOLUTION):
        raise FileNotFoundError('Cannot find Step65P best solution: '+SOLUTION)
    with np.load(SOLUTION) as s:
        q = np.asarray(s['q'],float).ravel().copy()
        r = np.asarray(s['normalized_residual'],float).ravel().copy()
        mat = np.array([float(np.asarray(s[k]).ravel()[0])
                        for k in ('E1','E2','G12')])
        c_scale = np.asarray(s['c_scale'],float).ravel().copy()
    if q.shape != (26,) or r.shape != (26,) or c_scale.shape != (26,):
        raise RuntimeError('65P NPZ must have 26 q, normalized_residual and c_scale entries')
    if not np.allclose(mat, CASE, rtol=0, atol=1e-8):
        raise RuntimeError('Step65P material does not match the middle case')
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)):
        raise RuntimeError('Nonfinite values in original 65P solution')
    return q,r,c_scale


def build_context(tag, jobs, cpus):
    old = module('65L_A_to_D_complete.py', 'base65L')
    adapt = module('65O_adaptive_coupling.py', 'oracle65O')
    directory = os.path.join(OUT, tag)
    mkdir(directory)
    ctx = old.Coupling(np.array(CASE), directory)
    oracle = adapt.Oracle(ctx, directory, prior_path=PRIOR,
                          job_budget=jobs, cpus=cpus,
                          adaptive_fidelity=True, online_correction=True)
    return ctx, oracle


def check_scale(ctx, c_scale):
    if not np.allclose(ctx.scale, c_scale, rtol=1e-12, atol=1e-14):
        raise RuntimeError('Current 65L runtime differs from saved Step65P PCA scaling')
    if np.asarray(ctx.force_scale).shape != (26,):
        raise RuntimeError('Expected 26 force normalization scales')


def prior_report(oracle, q):
    p = oracle.prior
    answer = {'file_found':bool(os.path.isfile(PRIOR)),
              'loaded_and_material_approved':bool(p.available),
              'reference_state_applicability':False,
              'nearest_standardized_distance':None,
              'nonzero_correction_rms':0.0}
    if p.available:
        x = (q[8:18]*np.asarray(oracle.ctx.scale)[8:18]-p.mean_x)/p.std_x
        dist = np.sqrt(np.sum((p.x_train-x)**2,axis=1))
        near = float(np.min(dist))
        v = p.predict(q)
        answer.update({'reference_state_applicability':bool(near<=8.0),
                       'nearest_standardized_distance':near,
                       'nonzero_correction_rms':norm_rms(v)})
    return answer


def audit():
    mkdir(OUT)
    q, r, scale = load_state()
    rows = []
    for i in range(26):
        rows.append({'q_index':i,'q':float(q[i]),
                     'normalized_residual':float(r[i]),
                     'squared_residual':float(r[i]*r[i]),
                     'selected_for_true_jacobian':bool(i in SELECTED)})
    csv_out(os.path.join(OUT,'65R_A_interface_audit.csv'),rows)
    info = {'input_npz':SOLUTION,'material_MPa':list(CASE),
            'initial_saved_rms':norm_rms(r),
            'center_share_squared_residual_pct':float(100*np.dot(r[8:18],r[8:18])/np.dot(r,r)),
            'selected_q_indices':SELECTED.tolist(),
            'selected_modes':list(SELECTED_NAMES),
            'center_prior_file_exists':os.path.isfile(PRIOR),
            'no_Abaqus_jobs':True}
    dump(os.path.join(OUT,'65R_A_audit.json'),info)
    print(json.dumps(info,indent=2))


def center_true_residual(ctx, oracle, q, serial):
    """Exactly one real Abaqus center patch, plus existing V4 NO forces."""
    _,_,_,no_g = ctx.rom(q, detailed=True)
    record, fe_g = oracle._patch('center', q, serial, ctx.coeffs(q))
    no = np.concatenate([np.asarray(no_g[k],float).ravel()
                         for k in ('center_left','center_right')])
    fe = np.concatenate([np.asarray(fe_g[k],float).ravel()
                         for k in ('center_left','center_right')])
    r = (fe+no)/np.asarray(ctx.force_scale)[8:18]
    if r.shape != (10,) or not np.all(np.isfinite(r)):
        raise RuntimeError('Bad high-fidelity center interface force vector')
    return r, record


def finite_jac(fun,q, step=1e-3):
    f0=np.asarray(fun(q),float)
    J=np.zeros((len(f0),len(q)))
    for j in range(len(q)):
        x=q.copy(); x[j]=min(3.,q[j]+step*max(1,abs(q[j])))
        if x[j]==q[j]:x[j]=max(-3.,q[j]-step*max(1,abs(q[j])))
        if x[j]==q[j]:raise RuntimeError('q variable at immovable bound')
        J[:,j]=(np.asarray(fun(x))-f0)/(x[j]-q[j])
    return J


def sensitivities(args):
    mkdir(OUT)
    if os.path.isfile(DIAG):
        raise RuntimeError('65R-B sensitivity already exists; leave existing result intact')
    q, saved_r, scale=load_state()
    # Three jobs to reverify starting state + eight center-only perturbations.
    ctx, oracle=build_context('65R_B_patch_runs',11,args.cpus)
    check_scale(ctx,scale)
    prior=prior_report(oracle,q)
    print('65N prior diagnostics:',json.dumps(prior,indent=2))
    if args.require_prior and not prior['reference_state_applicability']:
        raise RuntimeError('65N correction inactive at starting state. See prior diagnostics.')
    first=oracle.evaluate(q)  # all three real Abaqus patches
    if not first['verified']:
        raise RuntimeError('Starting point not fully verified')
    gap=norm_rms(first['r']-saved_r)
    if gap>args.repro_tol:
        raise RuntimeError('65P reproducibility check failed: RMS difference=%g'%gap)
    # Baseline ROM+online correction model, frozen before probe observations.
    J_base=finite_jac(oracle.corrected,q)[8:18,:][:,SELECTED]
    J_true=np.zeros((10,4))
    plus_res=np.zeros((10,4));minus_res=np.zeros((10,4))
    probe_rows=[];odb_paths=[]
    for j,col in enumerate(SELECTED):
        h=float(args.h)
        if q[col]-h < -3 or q[col]+h > 3:
            raise RuntimeError('Selected finite difference would leave q bounds')
        q_plus=q.copy();q_plus[col]+=h
        q_minus=q.copy();q_minus[col]-=h
        a,arec=center_true_residual(ctx,oracle,q_plus,100+2*j)
        b,brec=center_true_residual(ctx,oracle,q_minus,101+2*j)
        plus_res[:,j]=a;minus_res[:,j]=b
        J_true[:,j]=(a-b)/(2*h)
        nonlinearity=norm_rms(a+b-2*first['r'][8:18])
        cos=float(np.dot(J_true[:,j],J_base[:,j]) / max(
            1e-12,np.linalg.norm(J_true[:,j])*np.linalg.norm(J_base[:,j])))
        probe_rows.append({'mode':SELECTED_NAMES[j], 'q_index':int(col),
                           'h_q':h,'true_jacobian_norm':float(np.linalg.norm(J_true[:,j])),
                           'model_jacobian_norm':float(np.linalg.norm(J_base[:,j])),
                           'relative_column_error':float(np.linalg.norm(J_true[:,j]-J_base[:,j])/max(np.linalg.norm(J_true[:,j]),1e-12)),
                           'direction_cosine':cos,'center_second_difference_rms':nonlinearity})
        odb_paths.append({'q_index':int(col), 'plus_center_odb':arec.get('odb_file',''),
                          'minus_center_odb':brec.get('odb_file','')})
        print('%s: cosine %.3f, relative Jacobian error %.3f'%(SELECTED_NAMES[j],cos,probe_rows[-1]['relative_column_error']))
    np.savez(DIAG,q0=q,material=np.array(CASE),center_idx=CENTER_IDX,selected_idx=SELECTED,
             c_scale=scale,force_scale=np.asarray(ctx.force_scale),
             r0=first['r'],j_true_center=J_true,j_model_center=J_base,
             plus_center_residuals=plus_res,minus_center_residuals=minus_res,
             center_probe_h=np.array([args.h]))
    csv_out(os.path.join(OUT,'65R_B_center_jacobian_audit.csv'),probe_rows)
    dump(os.path.join(OUT,'65R_B_manifest.json'),
         {'initial_reproduced_rms':float(first['rms']),'reproduction_gap_rms':gap,
          'center_prior':prior,'sensitivity_step':args.h,
          'Abaqus_patch_jobs':int(oracle.jobs),'expected_jobs':11,
          'probe_center_odb_paths':odb_paths,
          'limitation':'Center derivatives only for four q coordinates at the 65P best state. No full Jacobian claimed.'})
    print('65R-B completed: %d Abaqus patch jobs; saved %s'%(oracle.jobs,DIAG))


def plan():
    if not os.path.isfile(DIAG):
        raise FileNotFoundError('Run --stage sensitivity first: '+DIAG)
    with np.load(DIAG) as d:
        q=d['q0'];r=d['r0'];J=d['j_true_center'];Jrom=d['j_model_center']
        indices=d['selected_idx']
    # Diagnostic 4-parameter LM; does NOT claim a full mechanics solve.
    R=r[8:18]
    lam=0.1
    G=J.T@J+lam*np.diag(np.maximum(np.diag(J.T@J),1e-5))
    p=-np.linalg.solve(G,J.T@R)
    p=p*min(1.0,0.1/max(float(np.linalg.norm(p)),1e-14))
    rows=[]
    for i in range(10):
        rows.append({'center_row_global_q_index':i+8,
                     'current_real_residual':float(R[i]),
                     'predicted_new_residual_linearized':float(R[i]+(J@p)[i]),
                     **{'Jtrue_'+SELECTED_NAMES[j]:float(J[i,j]) for j in range(4)},
                     **{'Jmodel_'+SELECTED_NAMES[j]:float(Jrom[i,j]) for j in range(4)}})
    csv_out(os.path.join(OUT,'65R_C_true_vs_model_jacobian.csv'),rows)
    result={'diagnostic_center_four_variable_step':p.tolist(),
            'selected_q_indices':indices.tolist(),
            'current_center_rms':norm_rms(R),
            'linearized_proposed_center_rms':norm_rms(R+J@p),
            'condition_jacobian_singular_values':np.linalg.svd(J,compute_uv=False).tolist(),
            'warning':'This is a linear diagnostic prediction; an Abaqus solve is required to verify any improvement.'}
    dump(os.path.join(OUT,'65R_C_direction_plan.json'),result)
    print(json.dumps(result,indent=2))


def gmres(A,b,maxit=20,rtol=1e-5):
    """Small Krylov solver; A maps vectors to vectors (not necessarily explicit matrix)."""
    b=np.asarray(b,float)
    norm=np.linalg.norm(b)
    if norm<1e-14:return np.zeros_like(b)
    n=len(b);m=min(int(maxit),n)
    V=np.zeros((n,m+1));H=np.zeros((m+1,m));V[:,0]=b/norm
    x=np.zeros(n)
    for j in range(m):
        w=np.asarray(A(V[:,j]),float)
        for i in range(j+1):
            H[i,j]=np.dot(V[:,i],w);w-=H[i,j]*V[:,i]
        H[j+1,j]=np.linalg.norm(w)
        if H[j+1,j]>1e-12: V[:,j+1]=w/H[j+1,j]
        e=np.zeros(j+2);e[0]=norm
        y=np.linalg.lstsq(H[:j+2,:j+1],e,rcond=None)[0]
        x=V[:,:j+1]@y
        if np.linalg.norm(e-H[:j+2,:j+1]@y)<rtol*norm or H[j+1,j]<1e-12:break
    return x


def step_lm(J,r,lam):
    g=J.T@J
    return -np.linalg.solve(g+lam*np.diag(np.maximum(np.diag(g),1e-5)),J.T@r)


def step_candidates(J,r,Jb, radius, damping, methods):
    out=[]
    for method in methods:
        try:
            if method=='LM': p=step_lm(J,r,damping)
            elif method=='Broyden': p=-np.linalg.lstsq(Jb+1e-4*np.eye(26),r,rcond=1e-7)[0]
            elif method=='Newton-Krylov':
                # Matrix-vector action from the corrected local Jacobian.
                P=np.linalg.pinv(J,rcond=1e-3)
                p=gmres(lambda v:P@(J@v),-P@r,maxit=18)
            elif method=='center-Cauchy':
                p=np.zeros(26)
                v=-J[np.ix_(CENTER_IDX,SELECTED)].T@r[CENTER_IDX]
                p[SELECTED]=v
            else:raise ValueError(method)
            if not np.all(np.isfinite(p)) or np.linalg.norm(p)<1e-12:continue
            for mult in (1.,0.5,0.25,0.1):
                d=p * min(1., radius*mult/max(np.linalg.norm(p),1e-14))
                out.append((method,d))
        except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:
            print('Method %s skipped: %s'%(method,exc))
    return out


def solve(args):
    if not os.path.isfile(DIAG):
        raise FileNotFoundError('Run --stage sensitivity and --stage plan first')
    q_saved, r_saved, scale=load_state()
    with np.load(DIAG) as d:
        q_cal=d['q0'].copy(); J_true=d['j_true_center'].copy()
        stored_scale=d['c_scale'].copy();force_scale=d['force_scale'].copy()
    if not np.allclose(q_cal,q_saved,atol=1e-12,rtol=0):
        raise RuntimeError('65R sensitivity calibration no longer matches 65P saved starting state')
    if not np.allclose(stored_scale,scale,atol=1e-12,rtol=0):
        raise RuntimeError('Changed coefficient scaling')
    run_dir=os.path.join(OUT,'65R_D_retry')
    manifest_file=os.path.join(run_dir,'rom_solution_manifest.json')
    if os.path.isfile(manifest_file):
        raise RuntimeError('65R-D result already exists; archive old folder before another run')
    ctx,oracle=build_context('65R_D_retry',args.jobs,args.cpus)
    check_scale(ctx,scale)
    if not np.allclose(force_scale,ctx.force_scale,atol=1e-12,rtol=1e-12):
        raise RuntimeError('Force normalization differs from sensitivity stage')
    prior=prior_report(oracle,q_saved)
    print('65N correction status:',json.dumps(prior,indent=2))
    if args.require_prior and not prior['reference_state_applicability']:
        raise RuntimeError('Expected an active center prior, but it is inactive')
    t0=time.perf_counter()
    first=oracle.evaluate(q_saved)
    if not first['verified'] or norm_rms(first['r']-r_saved)>args.repro_tol:
        raise RuntimeError('65P best verified state could not be reproduced on Abaqus')
    current=first;best=first;radius=float(args.radius);damping=0.1;Jb=None
    methods=('Broyden','LM','Newton-Krylov','center-Cauchy')
    rows=[]
    for iteration in range(args.max_iter):
        if best['rms']<=args.tol or oracle.jobs+3>args.jobs:break
        q=current['q'].copy()
        if np.linalg.norm(q-q_cal)>args.max_calibration_distance:
            print('Outside local sensitivity calibration trust ball. Stop; recalibrate center Jacobian.');break
        base_fun=oracle.corrected
        base_here=base_fun(q)
        Jbase=finite_jac(base_fun,q)
        correction=np.zeros((26,26))
        correction[np.ix_(CENTER_IDX,SELECTED)]=J_true-Jbase[np.ix_(CENTER_IDX,SELECTED)]
        Jeff=Jbase+correction
        # Exact at current VERIFIED state, and correct four measured center derivatives.
        def model(x):
            x=np.asarray(x,float)
            return current['r']+(base_fun(x)-base_here)+correction@(x-q)
        if Jb is None:Jb=Jeff.copy()
        pool=[]
        for method,p in step_candidates(Jeff,current['r'],Jb,radius,damping,methods):
            x=np.clip(q+p,-3.,3.)
            if np.linalg.norm(x-q)<1e-7:continue
            if np.linalg.norm(x-q_cal)>args.max_calibration_distance:continue
            predicted=obj(current['r'])-obj(model(x))
            if predicted>1e-10:
                pool.append((predicted,method,x))
        if not pool:
            rows.append({'iteration':iteration,'method':'no_predicted_descent','fully_verified':False,
                         'accepted':False,'trial_rms':'','best_rms':best['rms'],
                         'jobs_total':oracle.jobs,'jobs_this_trial':0,
                         'radius':radius,'damping':damping,
                         'partial_sse':'','predicted_improvement':''})
            radius*=0.6;damping*=2.
            if radius<args.min_radius:break
            continue
        # Select strongest cheap predicted objective reduction, one Abaqus trial.
        predicted,method,x=max(pool,key=lambda t:t[0])
        previous=oracle.jobs
        result=oracle.evaluate(x,incumbent_r=current['r'])
        accepted=False
        if result['verified'] and obj(result['r'])+1e-12<obj(current['r']):
            accepted=True
            s=result['q']-current['q'];y=result['r']-current['r']
            Jb+=np.outer(y-Jb@s,s)/max(float(s@s),1e-14)
            current=result
            if result['rms']<best['rms']:best=result
            radius=min(args.max_radius,radius*1.2)
            damping=max(1e-5,damping*0.7)
        else:
            radius*=0.7;damping=min(1e6,damping*2.)
        rows.append({'iteration':iteration,'method':method,
                     'fully_verified':bool(result['verified']), 'accepted':bool(accepted),
                     'trial_rms':result.get('rms',''), 'best_rms':best['rms'],
                     'jobs_total':oracle.jobs,'jobs_this_trial':oracle.jobs-previous,
                     'radius':radius,'damping':damping,
                     'partial_sse':result['partial_sse_lower_bound'],
                     'predicted_improvement':predicted})
        csv_out(os.path.join(run_dir,'65R_D_history.csv'),rows)
        print('Iter %02d %-15s %-7s HF best=%.6f, used jobs %d'%(iteration,method,
              'ACCEPT' if accepted else 'REJECT',best['rms'],oracle.jobs))
        if radius<args.min_radius:break
    csv_out(os.path.join(run_dir,'65R_D_history.csv'),rows)
    arrays={'q':best['q'],'c_scale':scale,'normalized_residual':best['r'],
            'rms_residual':np.array([best['rms']]),
            'E1':np.array([CASE[0]]),'E2':np.array([CASE[1]]),'G12':np.array([CASE[2]])}
    for name,value in ctx.coeffs(best['q']).items():
        if name in NAMES:arrays['c_'+name]=value
    np.savez(os.path.join(run_dir,'rom_solution_7region.npz'),**arrays)
    final=os.path.join(run_dir,'final')
    mkdir(final)
    for patch in ('left','center','right'):
        record=best['files'][patch]
        for key,ext in (('odb_file','.odb'),('input_file','.inp')):
            source=record[key]
            if not os.path.isfile(source):raise FileNotFoundError(source)
            shutil.copy2(source,os.path.join(final,patch+ext))
    manifest={'method':'65R four-column real center Jacobian corrected adaptive search',
              'initial_verified_rms':float(first['rms']),
              'best_verified_rms':float(best['rms']),
              'converged':bool(best['rms']<=args.tol),
              'tolerance':args.tol,
              'Abaqus_patch_jobs_diagnostic':11,
              'Abaqus_patch_jobs_retry':int(oracle.jobs),
              'total_Abaqus_patch_jobs_in_65R':11+int(oracle.jobs),
              'full_verified_evaluations':sum(bool(x['verified']) for x in oracle.log),
              'partial_rejections':sum(not x['verified'] for x in oracle.log),
              'center_prior':prior,
              'accepted_steps':sum(bool(x.get('accepted')) for x in rows),
              'best_evaluation_serial':int(best['serial']),
              'runtime_retry_seconds':time.perf_counter()-t0,
              'warning':'Local Jacobian measured only at starting 65P state; do not extrapolate it. Only complete three-patch evaluations establish convergence.'}
    dump(manifest_file,manifest)
    print(json.dumps(manifest,indent=2))
    if not manifest['converged']:raise SystemExit(2)


def selftest():
    # Synthetic linear least-squares problem: Jacobian in four calibrated columns.
    J=np.eye(26);r=np.ones(26)*0.07
    assert np.linalg.norm(r+J@step_lm(J,r,0.1))<np.linalg.norm(r)
    sol=gmres(lambda x:J@x,-r)
    assert np.linalg.norm(J@sol+r)<1e-6
    q=np.zeros(26)
    F=lambda x:0.1+J@x
    assert np.allclose(finite_jac(F,q),J,atol=1e-8)
    c=np.zeros((26,26));D=np.eye(10,4)*0.2
    c[np.ix_(CENTER_IDX,SELECTED)]=D
    assert np.allclose(c[np.ix_(CENTER_IDX,SELECTED)],D)
    print('65R synthetic algebra, Jacobian, LM and GMRES selftests PASSED')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['selftest','audit','sensitivity','plan','solve'],required=True)
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--h',type=float,default=0.02,help='center finite-difference perturbation in normalized q units')
    p.add_argument('--repro-tol',type=float,default=0.015)
    p.add_argument('--require-prior',action='store_true',help='stop if trained 65N correction is not active')
    p.add_argument('--jobs',type=int,default=24,help='separate retry-stage Abaqus job budget')
    p.add_argument('--radius',type=float,default=0.12)
    p.add_argument('--max-radius',type=float,default=0.25)
    p.add_argument('--min-radius',type=float,default=0.008)
    p.add_argument('--max-calibration-distance',type=float,default=0.35)
    p.add_argument('--max-iter',type=int,default=28)
    p.add_argument('--tol',type=float,default=0.02)
    args=p.parse_args()
    if args.h<=0 or args.h>0.1:raise ValueError('Use 0 < --h <= 0.1')
    if args.jobs<3:raise ValueError('Need at least 3 jobs for a verified 65R-D retry state')
    if args.stage=='selftest':selftest()
    elif args.stage=='audit':audit()
    elif args.stage=='sensitivity':sensitivities(args)
    elif args.stage=='plan':plan()
    elif args.stage=='solve':solve(args)


if __name__=='__main__':main()
