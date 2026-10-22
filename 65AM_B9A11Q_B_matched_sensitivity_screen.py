#!/usr/bin/env python3
"""B9A11Q-B: known-q B5 vs FIT-only empirical FEM regression sensitivity SCREEN.

Same 20 inner-node U123 sensors and fixed 9 interface descriptors, but the FEM
reference is only a FIRST-ORDER REGRESSION over the 44 original fitting cases.
This is NOT an exact Abaqus fixed-interface central difference, not a heldout
independent sensitivity validation, and not a full-beam B9A11N comparison.
If regression error exceeds sensor sigma, a matched FEM reference still needs
paired Abaqus subdomain perturbations at fixed q.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

KEYS=('E1','E2','G12')
REGIONS=('inner_left','inner_right')


def ridge_fit_predict(Xfit,Yfit,Xtest,lam=0.01):
    Xfit=np.asarray(Xfit,float);Yfit=np.asarray(Yfit,float);Xtest=np.asarray(Xtest,float)
    xm=Xfit.mean(axis=0);xs=np.maximum(Xfit.std(axis=0),1e-9)
    ym=Yfit.mean(axis=0)
    Z=(Xfit-xm)/xs
    A=Z.T@Z+lam*np.eye(Z.shape[1]);W=np.linalg.solve(A,Z.T@(Yfit-ym))
    Ypred=ym+(Xtest-xm)/xs@W
    return Ypred,W,xm,xs


def selftest():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(60,12));w=rng.normal(size=(12,14));y=x@w
    pred,B,mean,std=ridge_fit_predict(x,y,x,lam=1e-9)
    assert np.max(abs(pred-y))<1e-6
    assert B.shape==(12,14) and len(mean)==12 and len(std)==12
    print('B9A11Q-B SELFTEST PASS: fit-only linear surrogate, standardized derivatives')


def process(args):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    Q=importlib.import_module('65AM_B9A11Q_A_inverse_diagnostics')
    audit=importlib.import_module('65AM_B9A11P_A_readiness')
    audit.audit(root,args.variant,args.loss,None)
    material=[];quality=[]
    for region in REGIONS:
        C=Q.load_region(root,region,args.variant,args.loss,args.sensors)
        ds=C['ds'];d=C['d'];phys=C['phys'];sensor=C['ids']
        fit=np.asarray(d['fit'],int);tune=np.asarray(d['tune'],int)
        xf=phys[fit];xt=phys[tune]
        yf=np.asarray(ds['U'][fit][:,sensor,:],float).reshape(len(fit),-1)
        yt=np.asarray(ds['U'][tune][:,sensor,:],float).reshape(len(tune),-1)
        pred,W,xm,xs=ridge_fit_predict(xf,yf,xt,args.ridge)
        # Critical: these are independently prescribed interface BCs on each
        # FEM training subdomain; regression conditions on all q descriptors.
        rmse=float(np.sqrt(np.mean((pred-yt)**2)))
        whitened=float(rmse/args.sigma_u)
        train_pred,_,_,_=ridge_fit_predict(xf,yf,xf,args.ridge)
        fit_rmse=float(np.sqrt(np.mean((train_pred-yf)**2)))
        rank=int(np.linalg.matrix_rank((xf-xm)/xs, tol=1e-9))
        qlist=[]
        for cid in tune:
            if np.all(phys[cid,:3]>=C['lo']) and np.all(phys[cid,:3]<=C['hi']):
                qlist.append(int(cid))
        if not qlist:raise RuntimeError('No tune cases inside material training envelope for '+region)
        b5_by_property={k:[] for k in KEYS}
        for cid in qlist[:args.cases]:
            op=Q.ConditionalOperator(C,cid)
            for j,k in enumerate(KEYS):
                dz=.05*C['mid'][j]/C['half'][j]
                if dz>=1:
                    raise RuntimeError('Cannot perform +/-5%% B5 in training range: '+region+' '+k)
                zp=np.zeros(3);zm=np.zeros(3);zp[j]=dz;zm[j]=-dz
                change=.5*(op.pred_and_jac(zp,False)-op.pred_and_jac(zm,False))
                b5_by_property[k].append(float(np.sqrt(np.mean((change/args.sigma_u)**2))))
        for j,k in enumerate(KEYS):
            # Single-property +5% response, computed from the FIT-only FEM
            # regression while HOLDING ALL q DESCRIPTORS constant.
            fem_shift=np.asarray(W[j]*(.05*C['mid'][j]/xs[j]),float)
            fem_score=float(np.sqrt(np.mean((fem_shift/args.sigma_u)**2)))
            b5_score=float(np.mean(b5_by_property[k]))
            material.append(dict(region=region,property=k,trained_center_MPa=float(C['mid'][j]),
                                 input_perturbation_fraction=.05,
                                 FEM_fit_only_linear_sensitivity=fem_score,
                                 B5_conditional_sensitivity=b5_score,
                                 model_ratio_to_FEM_regression=b5_score/max(fem_score,1e-12),
                                 FEM_regression_tune_RMSE_over_sigma=whitened,
                                 FEM_regression_reliable_at_assumed_sigma=bool(whitened<=1),
                                 number_of_B5_conditional_tune_q=len(qlist[:args.cases])))
        quality.append(dict(region=region,fit_cases=len(fit),tune_cases=len(tune),
                            sensor_nodes=len(sensor),outputs_per_case=len(sensor)*3,
                            input_features=xf.shape[1],input_matrix_rank=rank,
                            ridge_penalty=args.ridge,fit_rmse_mm=fit_rmse,
                            tune_rmse_mm=rmse,tune_rmse_over_sigma=whitened,
                            pass_approximate_reference_gate=bool(whitened<=1)))
        print('Q-B',region,'fit-only FEM regression TUNE RMSE / measurement sigma =',round(whitened,4),flush=True)
    df=pd.DataFrame(material);val=pd.DataFrame(quality)
    if args.stage=='check':
        print('B9A11Q-B CHECK PASS. No sensitivity outputs written.')
        print(val.to_string(index=False))
        return
    out=root/'results'/'65AM_B9A11Q_conditional_sensitivity'/args.tag
    if out.exists():raise FileExistsError('Output exists; choose new --tag: '+str(out))
    out.mkdir(parents=True)
    df.to_csv(out/'B9A11Q_B_conditional_sensitivity.csv',index=False)
    val.to_csv(out/'B9A11Q_B_FEM_regression_quality.csv',index=False)
    grouped=df.groupby('property')[['FEM_fit_only_linear_sensitivity','B5_conditional_sensitivity']].mean().reindex(KEYS)
    ax=grouped.plot.bar(figsize=(9.2,5),rot=0)
    ax.set_ylabel('RMS response shift / assumed U sensor sigma (5% material shift)')
    ax.set_title('B9A11Q-B: known-q INNER U, B5 vs FIT-only FEM regression')
    ax.legend(['FEM fit-only linear proxy (not exact Abaqus derivative)','B5 frozen operator'],fontsize=8)
    fig=ax.figure
    gate=bool(val.pass_approximate_reference_gate.all())
    subtitle=('FEM regression passed provisional U-noise gate' if gate else
              'FEM regression failed U-noise gate: DO NOT CLAIM numerical sensitivity agreement')
    fig.text(.5,.015,subtitle,ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,1])
    fig.savefig(out/'B9A11Q_B_exploratory_matched_sensitivity.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(val.region,val.tune_rmse_over_sigma)
    ax.axhline(1,color='black',ls='--',lw=1,label='Assumed single-state displacement noise')
    ax.set_ylabel('FEM regression tune RMSE / sensor sigma')
    ax.set_title('Can an empirical FEM fit support a matched derivative comparison?')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'B9A11Q_B_FEM_regression_quality.png',dpi=190);plt.close(fig)
    summary=dict(stage='B9A11Q-B',scope='INNER B5 conditional q, same sensor nodes, U123 only',
                 rows=len(df),gate_all_regions=gate,
                 FEM_regression_is_not_exact_FEM_derivative=True,
                 FEM_regression_training='Original 64J TRAIN 44 FIT cases per region, linear 12-feature ridge; 12 internal TRAIN tune cases for diagnostic only',
                 B5_checkpoint_unchanged=True,legacy_65V_unchanged=True,
                 no_new_FEM_solutions=True, B9A11N_full_beam_importance_is_not_matched=True,
                 quality=quality,
                 next_if_fails='Obtain paired +/-5% Abaqus subdomain ODBs at identical prescribed PCA q for independent FEM finite-difference reference; use NEW holdouts thereafter',
                 warning='Never claim matched FEM sensitivity agreement from this regression if forward error or parameter confounding is material.')
    (out/'B9A11Q_B_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11Q-B COMPLETE',out)
    print('REFERENCE GATE', 'PASS (provisional)' if gate else 'FAIL (do NOT claim numeric FEM agreement)')
    print(grouped.to_string())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','run'))
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='pilot01')
    p.add_argument('--variant',default='centered',choices=('centered','standardized'))
    p.add_argument('--loss',default='grad',choices=('U','grad'))
    p.add_argument('--sensors',type=int,default=20)
    p.add_argument('--cases',type=int,default=6)
    p.add_argument('--sigma-u',type=float,default=.0002)
    p.add_argument('--ridge',type=float,default=.01)
    a=p.parse_args()
    if a.stage=='selftest':selftest();return
    if not(3<=a.sensors<=50 and 1<=a.cases<=12 and a.sigma_u>0 and a.ridge>=0):raise ValueError('Invalid settings')
    process(a)
if __name__=='__main__':main()
