#!/usr/bin/env python3
"""B9A11P-B: frozen B5 KNOWN-INTERFACE subdomain inverse proof-of-concept.

CAUTION: Uses FEM-prescribed PCA interface q from ORIGINAL TRAIN internal tune cases.
This is not a new fully-coupled seven-region forward solve, an external validation,
a deployment digital twin, or comparable directly to full-beam nine-property FEM ranking.
Outputs are explicitly labelled as conditional/diagnostic.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

KEYS=('E1','E2','G12')


def select_sensor_nodes(xyz,free,n):
    xyz=np.asarray(xyz,float);free=np.asarray(free,int)
    if len(free)<n:raise ValueError('Too few free nodes')
    # Spatial coverage inside inner subdomain; deterministic, no FEM U used.
    loc=xyz[free]
    scores=loc[:,0] + .00071*loc[:,1] + .000019*loc[:,2]
    ids=free[np.argsort(scores)]
    chosen=ids[np.round(np.linspace(0,len(ids)-1,n)).astype(int)]
    if len(np.unique(chosen))!=n:raise RuntimeError('Repeated sensor node')
    return chosen


def sensitivity(model, q, lo, hi, center, half, sigma=.0002, rel=.05):
    vals=[]
    for j in range(3):
        pmin=center.copy();pmax=center.copy()
        pmin[j]*=(1-rel);pmax[j]*=(1+rel)
        if pmin[j]<lo[j] or pmax[j]>hi[j]:
            # Stay in-fit-range and state actual applied perturbation.
            step=min(.95*(center[j]-lo[j]),.95*(hi[j]-center[j]),rel*center[j]);
            if step<=0:raise RuntimeError('Zero valid perturbation')
            pmin[j]=center[j]-step;pmax[j]=center[j]+step
        diff=model(pmax,q)-model(pmin,q)
        actual=(pmax[j]-pmin[j])/(2*center[j])
        score=float(np.sqrt(np.mean((diff/(2*sigma))**2)))
        vals.append((KEYS[j],score,actual))
    return vals


def selftest():
    f=lambda p,q: np.array([.1*p[0]+.01*p[1],p[1]+3*p[2]],float)
    lo=np.array([1.,1.,1.]);hi=np.array([3.,3.,3.]);center=np.array([2.,2.,2.])
    v=sensitivity(f,np.zeros(2),lo,hi,center,(hi-lo)/2,1.,.05)
    assert len(v)==3 and v[0][1]>0 and abs(v[0][2]-.05)<1e-12
    ix=select_sensor_nodes(np.arange(120).reshape(40,3),np.arange(40),20)
    assert len(set(ix))==20
    print('B9A11P-B SELFTEST PASS')


def process(root,args):
    import torch, pandas as pd
    from scipy.optimize import least_squares
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    audit_module=importlib.import_module('65AM_B9A11P_A_readiness')
    audit=audit_module.audit(root,args.variant,args.loss,None)
    if args.stage=='check':
        print('B9A11P-B CHECK PASS: frozen checkpoints, 12-input branch, and material-independent interface lifting checked. No optimizer run.')
        return
    c=importlib.import_module('65AM_B5_common')
    dev=torch.device('cpu');torch.set_num_threads(min(torch.get_num_threads(),2))
    data=[];rows=[];sens=[];report={}; rng=np.random.default_rng(args.seed)
    for region in ('inner_left','inner_right'):
        d,mu,sd=c.prepare(root,region)
        model,ck=c.load_model(root,args.variant,args.loss,region,d,mu,sd,dev)
        ds=d['ds'];phys=np.asarray(ds['branch_inputs'],float)
        lo=phys[d['fit'],:3].min(0);hi=phys[d['fit'],:3].max(0)
        center=(lo+hi)/2;half=(hi-lo)/2
        tune=[int(cid) for cid in d['tune'] if np.all(phys[int(cid),:3]>=lo) and np.all(phys[int(cid),:3]<=hi)]
        if not tune:raise RuntimeError('No in-fit-range tune cases '+region)
        tune=tune[:args.cases]
        # free = nodes strictly inside trained inner subdomain, excluding prescribed boundaries
        nsel=select_sensor_nodes(ds['coordinates'], np.setdiff1d(np.arange(len(ds['coordinates'])),d['fixed']),args.sensors)
        B=torch.as_tensor(np.asarray(d['basis'][:,nsel,:],np.float32).reshape(d['rank'],-1),device=dev)
        branch_mean=torch.as_tensor(np.asarray(d['branch_mean'],np.float32),device=dev)
        branch_std=torch.as_tensor(np.asarray(d['branch_std'],np.float32),device=dev)
        lifted_lookup={int(v):i for i,v in enumerate(np.r_[d['fit'],d['tune']])}
        def pred_np(p,q,lift):
            x=np.r_[p,q]
            xx=(torch.as_tensor(x,dtype=torch.float32,device=dev)-branch_mean)/branch_std
            with torch.no_grad(): coeff=model(xx[None,:])[0]
            return lift+np.asarray((coeff@B).cpu().numpy(),float)
        # Fixed interface PCA q is *prescribed test case input*, NOT a model-predicted boundary state.
        anchor=int(tune[0]);q0=phys[anchor,3:].copy();lift0=np.asarray(d['lift'][lifted_lookup[anchor]][nsel,:],float).reshape(-1)
        def forward(p,q):return pred_np(p,q,lift0)
        rank=sensitivity(forward,q0,lo,hi,center,half,args.sigma_u,.05)
        for param,value,applied in rank:
            sens.append(dict(region=region,property=param,importance_local_U_only=value,
                applied_fractional_shift=applied,n_sensor_nodes=int(args.sensors),
                trained_parameter_range_only=True,conditional_on_prescribed_q=True))
        for cid in tune:
            case=phys[cid];truth=case[:3];q=case[3:].copy()
            lift=np.asarray(d['lift'][lifted_lookup[cid]][nsel,:],float).reshape(-1)
            y_fem=np.asarray(ds['U'][cid,nsel,:],float).reshape(-1)
            f=lambda p: pred_np(p,q,lift)
            y_nominal=f(truth); rmse=float(np.sqrt(np.mean((y_nominal-y_fem)**2)))
            if cid==tune[0]:
                # The fast observation operator must reproduce original frozen B5 output.
                original,_=c.predict(model,d,np.array([cid]),dev)
                gap=float(np.max(np.abs(original[0,nsel,:].reshape(-1)-y_nominal)))
                if gap>1e-5:raise RuntimeError('Forward wrapper does not match B5.predict: '+str(gap))
            # Prohibit label leakage: truth used only for metric, not optimizer, priors or initial guesses.
            for rep in range(args.trials):
                obs=y_fem+rng.normal(0,args.sigma_u,len(y_fem))
                def residual(z):return np.r_[(f(center+half*z)-obs)/args.sigma_u,0.05*z]
                starts=(np.zeros(3),np.array([.35,-.30,.20]))
                opt=[]
                for z0 in starts:
                    r=least_squares(residual,z0,bounds=(-np.ones(3),np.ones(3)),
                        max_nfev=90,ftol=1e-8,xtol=1e-8,gtol=1e-8,diff_step=1e-3)
                    opt.append(r)
                best=min(opt,key=lambda v:float(np.sum(residual(v.x)**2)))
                est=center+half*best.x
                row=dict(region=region,case_id=cid,rep=rep,scope='KNOWN_PCA_INTERFACE',
                    source='64J original TRAIN internal TUNE',
                    oracle_interface_boundary_from_dataset=True,n_sensor_nodes=args.sensors,
                    forward_rmse_mm=rmse,forward_rmse_noise_ratio=rmse/args.sigma_u,
                    obs_sigma_mm=args.sigma_u,optimization_success=bool(best.success),
                    data_fit_whitened_RMS=float(np.sqrt(np.mean(((f(est)-obs)/args.sigma_u)**2))),
                    **{k+'_true_MPa':float(truth[j]) for j,k in enumerate(KEYS)},
                    **{k+'_estimated_MPa':float(est[j]) for j,k in enumerate(KEYS)},
                    **{k+'_absolute_relative_error_pct':float(100*abs(est[j]-truth[j])/truth[j]) for j,k in enumerate(KEYS)})
                rows.append(row)
        report[region]=dict(tune_cases=len(tune),selected_node_count=len(nsel),
                            training_ranges={k:[float(lo[j]),float(hi[j])] for j,k in enumerate(KEYS)})
    df=pd.DataFrame(rows);sc=pd.DataFrame(sens)
    if args.stage=='check':
        print('CHECK PASS conditional forward & inverse input structure; no output written')
        print('Internal tune inversion attempted:',df.shape[0])
        print('Median forward mismatch / assumed sensor noise:',float(df.forward_rmse_noise_ratio.median()))
        return
    out=root/'results'/'65AM_B9A11P_conditional_inverse'/args.tag
    if out.exists():raise FileExistsError('Choose a new --tag; refusing to overwrite '+str(out))
    out.mkdir(parents=True)
    df.to_csv(out/'B9A11P_B_conditional_inverse_cases.csv',index=False)
    sc.to_csv(out/'B9A11P_B_conditional_sensitivities.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for j,k in enumerate(KEYS):
        ax=axs[j];xx=df[k+'_true_MPa'];yy=df[k+'_estimated_MPa']
        ax.scatter(xx,yy,alpha=.65,s=15)
        low=min(xx.min(),yy.min());high=max(xx.max(),yy.max());ax.plot([low,high],[low,high],'--',lw=1)
        ax.set_xlabel(k+' FEM true (MPa)');ax.set_ylabel(k+' B5 inverse (MPa)')
        ax.set_title(k+' | median error %.2f%%'%df[k+'_absolute_relative_error_pct'].median())
    fig.suptitle('B9A11P | KNOWN interface, internal tune only — NOT full hybrid twin')
    fig.tight_layout();fig.savefig(out/'B9A11P_B_inverse_recovery.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    agg=sc.groupby('property').importance_local_U_only.mean().reindex(KEYS)
    ax.bar(range(3),agg.values)
    ax.set_xticks(range(3),KEYS);ax.set_ylabel('RMS response change / sensor noise')
    ax.set_title('B5 local material sensitivity | inner U only, fixed interface q')
    fig.tight_layout();fig.savefig(out/'B9A11P_B_local_forward_importance.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    errs=[df[k+'_absolute_relative_error_pct'].median() for k in KEYS]
    ax.bar(range(3),errs)
    ax.set_xticks(range(3),KEYS);ax.set_ylabel('Median absolute inverse relative error (%)')
    ax.set_title('Conditional B5 material recovery error (internal tune only)')
    fig.tight_layout();fig.savefig(out/'B9A11P_B_inverse_error.png',dpi=170);plt.close(fig)
    # Optional side-by-side RANK diagnostic. The full-beam FEM observations differ
    # from the B5 inner U-only known-interface observations: numeric score agreement
    # cannot be interpreted as a validated matched-operator comparison.
    fem_file=root/'results/65AM_B9A11N_nine_property_importance/run01/B9A11N_all_importance.csv'
    if fem_file.is_file():
        fem=pd.read_csv(fem_file)
        fem=fem[(fem['configuration']=='20_sensors')&fem['parameter'].isin(KEYS)]
        if len(fem)==3:
            ref=fem.set_index('parameter').loc[list(KEYS)]['importance_RMS_5pct_in_measurement_noise_sigma']
            val=agg.reindex(list(KEYS))
            fig,ax=plt.subplots(figsize=(8.7,4.6))
            xx=np.arange(3)
            ax.bar(xx-.19,ref.values/ref.max(),width=.38,label='Abaqus full beam (U+strain)')
            ax.bar(xx+.19,val.values/val.max(),width=.38,label='B5 known-interface inner U-only')
            ax.set_xticks(xx,list(KEYS));ax.set_ylabel('Importance / maximum within each method')
            ax.legend(loc='upper right',fontsize=8)
            ax.set_title('Exploratory rank comparison: different observation operators')
            fig.text(.5,.015,'NOT a matched-sensor or full-coupling validation. Compare RANK only, not importance amplitude.',
                     ha='center',fontsize=8)
            fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(out/'B9A11P_B_exploratory_FEM_vs_B5_rank.png',dpi=170);plt.close(fig)
            print('Saved EXPLORATORY rank comparison; different sensors/boundary conditions.')
    med=float(df.forward_rmse_noise_ratio.median())
    summary=dict(stage='B9A11P-B',scope='known-interface INNER B5 conditional inverse',
        seed=args.seed,regions=report,noise_sigma_U_mm=args.sigma_u,
        number_of_fem_tune_cases_by_region=args.cases,number_of_noisy_fits=len(df),
        median_forward_model_rmse_over_sensor_noise=med,
        median_inverse_absolute_pct={k:float(df[k+'_absolute_relative_error_pct'].median()) for k in KEYS},
        b5_conditional_sensitivity_rank=sc.groupby('property').importance_local_U_only.mean().sort_values(ascending=False).index.tolist(),
        forward_error_below_noise_preliminary_gate=bool(med<1),
        not_suitable_for_digital_twin_claim=True,
        note='If forward mismatch exceeds noise, inverse may be badly biased. Truth is not used as optimizer input. Prescribed q comes from original FEM test dataset.',
        fem_nine_property_importance_not_directly_comparable='B9A11N evaluated FULL beam 20 near-surface strain+U sensors; B9A11P uses INNER regions only, U at nodes, with known q.',
        not_external_validation=True,not_a_B5_coupled_full_beam_solution=True)
    (out/'B9A11P_B_summary.json').write_text(json.dumps(summary,indent=2))
    print('RUN COMPLETE',out)
    print('WARNING: this does not validate fully coupled FE-NO inversion or digital twin.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','run'))
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='pilot01')
    p.add_argument('--variant',default='centered',choices=('centered','standardized'))
    p.add_argument('--loss',default='grad',choices=('U','grad'))
    p.add_argument('--sensors',type=int,default=20)
    p.add_argument('--cases',type=int,default=6)
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--sigma-u',type=float,default=.0002)
    p.add_argument('--seed',type=int,default=65)
    a=p.parse_args()
    if a.stage=='selftest':selftest();return
    if not(5<=a.sensors<=50 and 1<=a.cases<=12 and 1<=a.trials<=20 and a.sigma_u>0):raise ValueError('Invalid settings')
    process(Path(a.project).expanduser().resolve(),a)
if __name__=='__main__':main()
