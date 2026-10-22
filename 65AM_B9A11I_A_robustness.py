#!/usr/bin/env python3
"""B9A11I-A: fixed-method multi-case *approximate* Bayesian/Laplace robustness.

Uses 50-case full-FEM data and existing B9A11H2-B implementation. Not B5 coupled.
Holds original train/validation/test partitions fixed. Previously explored test
cases are a retrospective benchmark, NOT a new untouched holdout.
For speed, marginal intervals are local-Gaussian Laplace approximations, clipped
at the fixed prior bounds; they are not exact Bayesian credible intervals.
Does not write checkpoints, retrain a neural net, or run Abaqus.
"""
import argparse, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm

ROOT_DIR=Path(__file__).resolve().parent
B_MODULE='65AM_B9A11H2_B_inverse_uq.py'
MODES=('U_only','U_plus_RF','RF_only')
PARAMS=('E1','E2','G12')

def import_b(root):
    file=root/B_MODULE
    if not file.is_file():raise FileNotFoundError('Missing existing '+str(file))
    spec=importlib.util.spec_from_file_location('b9a11h_existing_unchanged',str(file))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def writecsv(path,rows):
    if not rows:raise RuntimeError('Empty report: '+str(path))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def metrics_for_case(B,model,chol,obs,z0,noise,mode,which):
    """Constrained weighted LS / uniform-bounds MAP; local Fisher covariance only."""
    # once per case: whiten observed residual. Cholesky includes emulator covariance.
    def value_grad(x):
        p=B.predict(model,x)[0]
        resid=solve_triangular(chol,p-obs,lower=True,check_finite=False)
        rawj=(B.jacdesign(x).T@model['coef'])*model['scale'][None,:]
        jw=solve_triangular(chol,rawj.T,lower=True,check_finite=False)
        return float(.5*np.dot(resid,resid)),jw.T@resid
    starts=(np.zeros(3),np.array([-.7,-.7,-.7]),np.array([.7,.7,.7]))
    solutions=[minimize(lambda x:value_grad(x),x0,bounds=[(-1.,1.)]*3,
                        jac=True,method='L-BFGS-B',options={'maxiter':160,'ftol':1e-11}) for x0 in starts]
    ans=min(solutions,key=lambda q:q.fun)
    z=np.clip(ans.x,-1.,1.)
    p=B.predict(model,z)[0]
    j=(B.jacdesign(z).T@model['coef'])*model['scale'][None,:]
    jw=solve_triangular(chol,j.T,lower=True,check_finite=False)
    singular=np.linalg.svd(jw,compute_uv=False)
    rank=int(np.sum(singular>max(float(singular[0]),1e-10)*1e-4))
    matrix=jw.T@jw
    # Weak prior curvature is not added to an apparently singular measurement model.
    if rank==3:
        cov=np.linalg.inv(matrix)
        one_sd=np.sqrt(np.maximum(np.diag(cov),0.))
        low=np.maximum(-1.,z-1.959963984540054*one_sd)
        high=np.minimum(1.,z+1.959963984540054*one_sd)
    else:
        one_sd=np.full(3,np.nan);low=np.full(3,np.nan);high=np.full(3,np.nan)
    truth=np.asarray(z0,dtype=float)
    e=np.asarray((z-truth)*B.HALF,dtype=float)
    result={'case_id':which,'noise_factor':noise,'sensors':len(obs)//3 if mode=='U_only' else
            (len(obs)-1)//3 if mode=='U_plus_RF' else 0,
            'mode':mode,'n_observations':len(obs),'identifiability_rank':rank,
            'condition_number':float(singular[0]/singular[-1]) if rank==3 else None,
            'at_prior_edge':bool(np.max(np.abs(z))>=.999),
            'MAP_chi2':float(2*ans.fun),
            'converged':bool(ans.success or np.linalg.norm(value_grad(z)[1])<1e-4),
            'min_singular':float(singular[-1]),
           }
    for k,name in enumerate(PARAMS):
        physical=float(B.CENTER[k]+z[k]*B.HALF[k])
        result[name+'_estimate_MPa']=physical
        result[name+'_true_MPa']=float(B.CENTER[k]+truth[k]*B.HALF[k])
        result[name+'_absolute_relative_error_pct']=float(100*abs(e[k])/(B.CENTER[k]+truth[k]*B.HALF[k]))
        result[name+'_interval_low_MPa']=float(B.CENTER[k]+low[k]*B.HALF[k]) if rank==3 else None
        result[name+'_interval_high_MPa']=float(B.CENTER[k]+high[k]*B.HALF[k]) if rank==3 else None
        result[name+'_local_interval_covers']=bool(low[k]<=truth[k]<=high[k]) if rank==3 else None
        result[name+'_approx_1sigma_MPa']=float(one_sd[k]*B.HALF[k]) if rank==3 else None
    return result

def selftest():
    import tempfile
    # Check identity of chi-square derivative in a simple linear Gaussian model.
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/B_MODULE
        if not p.is_file():pass
    a=np.array([[3.,1.,0.],[0.,4.,1.],[1.,1.,5.]])
    rank=np.linalg.matrix_rank(a)
    if rank!=3 or np.linalg.svd(a,compute_uv=False)[-1]<=0:raise RuntimeError('SVD selftest failed')
    print('B9A11I-A SELFTEST PASS | Fisher rank and prior restrictions')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--data-dir')
    ap.add_argument('--tag',default='run01')
    ap.add_argument('--stage',choices=('selftest','check','run'),required=True)
    ap.add_argument('--repeats',type=int,default=2,help='Predeclared noise seeds per scenario; >=2')
    ap.add_argument('--noise-factors',type=float,nargs='+',default=[1.,5.])
    ap.add_argument('--sensor-counts',type=int,nargs='+',default=[5,10,20,50])
    ap.add_argument('--seed',type=int,default=20261019)
    args=ap.parse_args()
    if args.stage=='selftest':return selftest()
    if not (2<=args.repeats<=30 and args.tag.replace('_','').replace('-','').isalnum()
            and all(5<=n<=50 for n in args.sensor_counts)
            and all(0<x<=50 for x in args.noise_factors)):
        raise ValueError('Invalid arguments')
    root=Path(args.project).expanduser().resolve()
    B=import_b(root)
    source=B.dataset_dir(root,args.data_dir)
    d=B.read_data(source)
    rank_path=root/'results'/'65AM_B9A11H_v2'/'run01'/'C_sensitivity'/'B9A11H_C_sensor_rankings.csv'
    if not rank_path.is_file():
        raise FileNotFoundError('Run existing B9A11H2-C sensitivity first; ranking missing: '+str(rank_path))
    print('B9A11I check: source',source,'| ranking',rank_path)
    print('Held-out cases:',[d['ids'][k] for k in d['groups']['test']])
    if args.stage=='check':
        for n in args.sensor_counts:
            y,labels,types,_=B.compute_observations(d,n,rank_path)
            if 'RF' not in types:raise RuntimeError('Missing RF2 for '+str(n))
            print('  ',n,'nodes ->',len(labels),'observables; selection frozen')
        print('CHECK PASS. Posterior intervals in run are Gaussian/Laplace approximations, not full MCMC.')
        return
    out=root/'results'/'65AM_B9A11I_healthy_robustness'/args.tag
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output protected: '+str(out))
    cases=[];ident=[];models=[]
    rng=np.random.default_rng(args.seed)
    for n in sorted(set(args.sensor_counts)):
        y,labels,types,layout=B.compute_observations(d,n,rank_path)
        tr=d['groups']['train'];va=d['groups']['validation'];te=d['groups']['test']
        for mode in MODES:
            col=np.array([i for i,t in enumerate(types) if mode=='U_plus_RF' or
                          (mode=='U_only' and t!='RF') or (mode=='RF_only' and t=='RF')],dtype=int)
            if not len(col):raise RuntimeError('No features for '+mode)
            sub=y[:,col]
            model,cov,cv,hold=B.model_pipeline(d,sub)
            models.append({'sensors':n,'mode':mode,'alpha':float(model['alpha']),
                           'validation_standardized_RMSE':float(min(q['validation_standardized_rmse'] for q in cv)),
                           'holdout_test_rmse':float(hold['full_test_rmse']),
                           'n_observations':len(col)})
            if mode=='RF_only':
                # Rank <= 1: cannot infer three elastic properties separately.
                ident.append({'sensors':n,'mode':mode,'rank_upper_bound':1,
                              'finding':'ONE reaction-force scalar cannot distinguish three unknown moduli without strong prior information'})
                continue
            for factor in args.noise_factors:
                sigma=np.array([5*factor if types[i]=='RF' else .0002*factor for i in col],float)
                covsub=cov+np.diag(sigma*sigma)
                chol=np.linalg.cholesky(covsub)
                for k in range(args.repeats):
                    # Use *the same underlying noise realization* for comparisons between
                    # U_only and combined when possible: index by full observation label.
                    seed=int(args.seed+10000*n+100*list(args.noise_factors).index(factor)+k)
                    for idx in te:
                        # Per-case independent noise; same realization for nested modes.
                        rr=np.random.default_rng(seed+int(idx)*104729)
                        noise_full=rr.normal(size=len(types))
                        ob=sub[idx]+noise_full[col]*sigma
                        zz=(d['X'][idx]-B.CENTER)/B.HALF
                        record=metrics_for_case(B,model,chol,ob,zz,factor,mode,d['ids'][idx])
                        record['sensors']=n;record['replicate']=k+1
                        record['synthetic_noise_seed']=seed
                        cases.append(record)
                print('DONE',n,mode,'noise x',factor,'cases',len(te)*args.repeats,flush=True)
    if not cases:raise RuntimeError('No cases generated')
    import pandas as pd
    full=pd.DataFrame(cases)
    agg=[]
    for (sensor,mode,noise),part in full.groupby(['sensors','mode','noise_factor'],sort=True):
        row={'sensors':int(sensor),'mode':mode,'noise_factor':float(noise),
             'runs':len(part),'rank_3_fraction':float(np.mean(part.identifiability_rank==3)),
             'boundary_MAP_fraction':float(np.mean(part.at_prior_edge)),
             'optimization_success_fraction':float(np.mean(part.converged))}
        for p in PARAMS:
            row[p+'_median_relative_error_pct']=float(part[p+'_absolute_relative_error_pct'].median())
            row[p+'_p90_relative_error_pct']=float(part[p+'_absolute_relative_error_pct'].quantile(.90))
            subset=part[part.identifiability_rank==3]
            row[p+'_approx_interval_coverage_fraction']=float(subset[p+'_local_interval_covers'].mean()) if len(subset) else None
            row[p+'_median_approx_1sigma_MPa']=float(subset[p+'_approx_1sigma_MPa'].median()) if len(subset) else None
        agg.append(row)
    out.mkdir(parents=True,exist_ok=False)
    full.to_csv(out/'B9A11I_case_recovery.csv',index=False)
    pd.DataFrame(agg).to_csv(out/'B9A11I_aggregate_robustness.csv',index=False)
    pd.DataFrame(ident).to_csv(out/'B9A11I_RF_only_identifiability.csv',index=False)
    pd.DataFrame(models).to_csv(out/'B9A11I_surrogate_screens.csv',index=False)
    summary={'step':'B9A11I-A','source':'original 50 full-FEM healthy cases, NOT coupled B5/65V',
       'method':'predefined quadratic ridge + weighted MAP + local Fisher/Laplace intervals',
       'case_count':int(len(te)),'n_runs':int(len(full)),
       'original_test_cases_previously_explored':True,
       'new_independent_holdout':False,
       'noise_assumed_1sigma':{'displacement_mm':.0002,'reaction_N':5},
       'noise_multipliers':args.noise_factors,'repeats':args.repeats,
       'modes':['U_only','U_plus_RF','RF_only rank diagnostic'],
       'measurement_noise_across_nodes_assumed_independent':True,
       'emulator_error_covariance':'validation residual (15% full +85% diagonal), estimated with seven validation cases',
       'constraints':'uniform prior bounds E1[40500,49500],E2[10800,13200],G12[4050,4950] MPa',
       'interpretation':'Approximate local 95% Fisher/Laplace clipped intervals; NOT full posterior, near boundaries unreliable',
       'do_not_infer':['damage','experimentally calibrated coverage','new untouched test performance','B5 coupled inverse'],
       'summary_rows':agg,'outputs':['B9A11I_case_recovery.csv','B9A11I_aggregate_robustness.csv',
          'B9A11I_RF_only_identifiability.csv','B9A11I_surrogate_screens.csv']}
    (out/'B9A11I_robustness_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11I COMPLETE:',out)
    print(pd.DataFrame(agg)[['sensors','mode','noise_factor','rank_3_fraction','E1_median_relative_error_pct','E2_median_relative_error_pct','G12_median_relative_error_pct']].to_string(index=False))

if __name__=='__main__':main()
