#!/usr/bin/env python3
"""B9A11Q-A: noise-free B5->B5 and FEM->B5 conditional material inverse.

Read-only: original B5 checkpoints, 64J TRAIN internal tune data, interface q.
No new Abaqus, NO full 7-region re-equilibration. Synthetic operator oracle is
not independent FEM validation or a digital twin.

Mac usage:
  python3 65AM_B9A11Q_A_inverse_diagnostics.py --stage selftest
  python3 65AM_B9A11Q_A_inverse_diagnostics.py --stage check --project .
  python3 65AM_B9A11Q_A_inverse_diagnostics.py --stage run --project . --tag pilot01 --cases 6 --sensors 20
"""
from __future__ import annotations
import argparse, importlib, sys, json, hashlib
from pathlib import Path
import numpy as np

PKEYS=('E1','E2','G12')
REGIONS=('inner_left','inner_right')
OUT='65AM_B9A11Q_conditional_inverse'


def select_sensor_nodes(xyz,free,n):
    xyz=np.asarray(xyz,float);free=np.asarray(free,int)
    if not (3<=n<=len(free)): raise ValueError('Not enough free nodes')
    scores=xyz[free,0]+.00071*xyz[free,1]+.000019*xyz[free,2]
    ordered=free[np.argsort(scores)]
    ids=ordered[np.round(np.linspace(0,len(free)-1,n)).astype(int)]
    if len(set(ids.tolist()))!=n:raise RuntimeError('Duplicate sensor selection')
    return ids


def load_region(root,region,variant,loss,sensors):
    import torch
    C=importlib.import_module('65AM_B5_common')
    d,mu,sd=C.prepare(root,region)
    model,_=C.load_model(root,variant,loss,region,d,mu,sd,torch.device('cpu'))
    model.eval()
    ds=d['ds'];phys=np.asarray(ds['branch_inputs'],float)
    if phys.shape[1]!=12:raise RuntimeError('Expected [E1,E2,G12,9 interface coefficients] input')
    lo=phys[d['fit'],:3].min(0);hi=phys[d['fit'],:3].max(0)
    mid=(lo+hi)/2;half=(hi-lo)/2
    if np.any(half<=0):raise RuntimeError('Invalid parameter training intervals')
    free=np.setdiff1d(np.arange(len(ds['coordinates']),dtype=int),d['fixed'])
    ids=select_sensor_nodes(ds['coordinates'],free,sensors)
    lookup={int(v):i for i,v in enumerate(np.r_[d['fit'],d['tune']])}
    B=torch.as_tensor(np.asarray(d['basis'][:,ids,:],dtype=np.float32).reshape(d['rank'],-1))
    m=torch.as_tensor(d['branch_mean'],dtype=torch.float32)
    s=torch.as_tensor(d['branch_std'],dtype=torch.float32)
    tmid=torch.as_tensor(mid,dtype=torch.float32)
    thalf=torch.as_tensor(half,dtype=torch.float32)
    return dict(C=C,ds=ds,d=d,model=model,phys=phys,lo=lo,hi=hi,mid=mid,half=half,
                ids=ids,lookup=lookup,B=B,mean=m,std=s,tmid=tmid,thalf=thalf,region=region)


class ConditionalOperator:
    def __init__(self,config,case_id):
        import torch
        x=config
        self.x=x;self.case_id=int(case_id)
        self.q=torch.as_tensor(x['phys'][case_id,3:].copy(),dtype=torch.float32)
        self.lift=torch.as_tensor(np.asarray(x['d']['lift'][x['lookup'][case_id]][x['ids'],:],
                                              dtype=np.float32).reshape(-1))
        self.base_FEM=np.asarray(x['ds']['U'][case_id,x['ids'],:],float).reshape(-1)
        self.true=np.asarray(x['phys'][case_id,:3],float)

    def _coeff(self,z):
        import torch
        x=self.x
        p=x['tmid']+x['thalf']*z
        branch=torch.cat((p,self.q))
        return x['model'](((branch-x['mean'])/x['std'])[None,:])[0]

    def pred_and_jac(self,z,jac=True):
        import torch
        z=np.asarray(z,float)
        if z.shape!=(3,):raise ValueError('z must contain 3 normalized material variables')
        tz=torch.tensor(z,dtype=torch.float32,requires_grad=jac)
        if jac:
            coeff=self._coeff(tz)
            with torch.enable_grad():
                Jc=torch.autograd.functional.jacobian(self._coeff,tz,vectorize=False)
            vals=(self.lift+coeff@self.x['B']).detach().numpy().astype(float)
            J=(self.x['B'].T@Jc).detach().numpy().astype(float)
            return vals,J
        with torch.no_grad():
            coeff=self._coeff(tz)
            vals=(self.lift+coeff@self.x['B']).numpy().astype(float)
        return vals

    def normalized_true(self):
        return (self.true-self.x['mid'])/self.x['half']


def inversion(op,y,sigma=0.0002,max_nfev=150):
    from scipy.optimize import least_squares
    def fun(z):return (op.pred_and_jac(z,False)-y)/sigma
    def jac(z):return op.pred_and_jac(z,True)[1]/sigma
    starts=[np.zeros(3),np.array([.55,-.4,.3]),np.array([-.35,.4,-.55])]
    out=[]
    for z0 in starts:
        r=least_squares(fun,z0,jac=jac,bounds=(-np.ones(3),np.ones(3)),
                        max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        out.append(r)
    best=min(out,key=lambda r:float(np.dot(fun(r.x),fun(r.x))))
    return best, float(np.sqrt(np.mean(fun(best.x)**2)))


def diagnostics(J,sigma):
    x=np.asarray(J,float)/sigma
    s=np.linalg.svd(x,compute_uv=False)
    norm=np.linalg.norm(x,axis=0)
    cos=x.T@x/np.maximum(norm[:,None]*norm[None,:],1e-20)
    np.fill_diagonal(cos,0.0)
    return dict(singular_min=float(s[-1]),singular_max=float(s[0]),
                condition=float(s[0]/max(s[-1],1e-15)),
                max_abs_column_cosine=float(np.max(np.abs(cos))),
                jacobian_rank=int(np.sum(s>s[0]*1e-5)))


def selftest():
    rng=np.random.default_rng(19)
    A=rng.normal(size=(15,3));z=np.array([.2,-.35,.45]);y=A@z
    from scipy.optimize import least_squares
    result=least_squares(lambda w:A@w-y,np.zeros(3),jac=lambda _:A,bounds=(-np.ones(3),np.ones(3)))
    assert np.linalg.norm(result.x-z)<1e-7
    ix=select_sensor_nodes(np.arange(180).reshape(60,3),np.arange(60),20)
    assert len(ix)==20 and len(set(ix))==20
    r=diagnostics(A,0.0002)
    assert r['jacobian_rank']==3 and r['singular_min']>0
    print('B9A11Q-A SELFTEST PASS: inverse, rank, sensor uniqueness')


def run(args):
    import torch, pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    torch.set_num_threads(min(torch.get_num_threads(),2))
    audit=importlib.import_module('65AM_B9A11P_A_readiness')
    audit.audit(root,args.variant,args.loss,None)
    configs={region:load_region(root,region,args.variant,args.loss,args.sensors) for region in REGIONS}
    if args.stage=='check':
        for reg,c in configs.items():
            valid=[int(k) for k in c['d']['tune'] if np.all(c['phys'][k,:3]>=c['lo']) and np.all(c['phys'][k,:3]<=c['hi'])]
            if not valid:raise RuntimeError('No tuning cases within training range: '+reg)
            op=ConditionalOperator(c,valid[0])
            vals,J=op.pred_and_jac(op.normalized_true(),True)
            if vals.shape!=(3*args.sensors,) or J.shape!=(3*args.sensors,3) or not np.isfinite(J).all():
                raise RuntimeError('B5 response/Jacobian shape bad')
            # Non-leaking verification against official B5 original prediction.
            orig,_=c['C'].predict(c['model'],c['d'],np.array([valid[0]]),torch.device('cpu'))
            err=np.max(np.abs(orig[0,c['ids'],:].reshape(-1)-vals))
            if err>2e-5:raise RuntimeError('B5 wrapper discrepancy '+str(err))
            print('CHECK',reg,'valid tuning cases',len(valid),'outputs',len(vals),'B5 fidelity max',err)
        print('B9A11Q-A CHECK PASS. No optimizations or outputs written.')
        return
    rng=np.random.default_rng(args.seed)
    rows=[];jrows=[]
    for reg,c in configs.items():
        valid=[int(k) for k in c['d']['tune'] if np.all(c['phys'][k,:3]>=c['lo']) and np.all(c['phys'][k,:3]<=c['hi'])]
        for cid in valid[:args.cases]:
            op=ConditionalOperator(c,cid)
            ztrue=op.normalized_true()
            y_b5,J=op.pred_and_jac(ztrue,True)
            y_fem=op.base_FEM
            info=diagnostics(J,args.sigma_u)
            jrows.append(dict(region=reg,case_id=cid,**info,
                              b5_vs_fem_forward_rmse_mm=float(np.sqrt(np.mean((y_b5-y_fem)**2))),
                              b5_vs_fem_forward_rmse_over_sigma=float(np.sqrt(np.mean(((y_b5-y_fem)/args.sigma_u)**2)))))
            jobs=[('B5_to_B5_zero_noise',y_b5,0),('FEM_to_B5_zero_noise',y_fem,0)]
            for rep in range(args.noise_reps):
                jobs.append(('FEM_to_B5_noisy',y_fem+rng.normal(0,args.sigma_u,len(y_fem)),rep))
            for mode,y,rep in jobs:
                opt,residual_rms=inversion(op,y,args.sigma_u,args.max_nfev)
                est=c['mid']+c['half']*opt.x
                rec=dict(region=reg,case_id=cid,mode=mode,repeat=int(rep),optimizer_success=bool(opt.success),
                         optimizer_message=str(opt.message),nfev=int(opt.nfev),
                         selected_sensor_nodes=args.sensors,sensor_components='U1 U2 U3',
                         fit_whitened_rms=float(residual_rms),known_interface_oracle_from_FEM=True,
                         **{k+'_true_MPa':float(op.true[i]) for i,k in enumerate(PKEYS)},
                         **{k+'_estimate_MPa':float(est[i]) for i,k in enumerate(PKEYS)},
                         **{k+'_abs_rel_pct':float(100*abs(est[i]-op.true[i])/op.true[i]) for i,k in enumerate(PKEYS)},
                         **{k+'_hit_bound':bool(abs(opt.x[i])>.99) for i,k in enumerate(PKEYS)})
                rows.append(rec)
            print('Q-A',reg,'case',cid,'B5/FEM rmse',round(jrows[-1]['b5_vs_fem_forward_rmse_mm'],7),flush=True)
    df=pd.DataFrame(rows); jj=pd.DataFrame(jrows)
    out=root/'results'/OUT/args.tag
    if out.exists():raise FileExistsError('Output exists. Use a fresh --tag: '+str(out))
    out.mkdir(parents=True)
    df.to_csv(out/'B9A11Q_A_inverse_modes.csv',index=False)
    jj.to_csv(out/'B9A11Q_A_conditioning.csv',index=False)
    meds=df.groupby('mode')[[k+'_abs_rel_pct' for k in PKEYS]].median()
    ax=meds.plot.bar(rot=0,figsize=(10,5))
    ax.set_ylabel('Median absolute material recovery error (%)')
    ax.set_title('B9A11Q-A: noise-free operator oracle vs FEM-driven inverse | KNOWN q')
    ax.legend([k for k in PKEYS]);plt.tight_layout()
    plt.savefig(out/'B9A11Q_A_inverse_error_modes.png',dpi=190);plt.close()
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for k,ax in zip(PKEYS,axs):
        for mode,group in df.groupby('mode'):
            ax.scatter(group[k+'_true_MPa'],group[k+'_estimate_MPa'],s=17,alpha=.68,label=mode)
        lo=min(df[k+'_true_MPa'].min(),df[k+'_estimate_MPa'].min())
        hi=max(df[k+'_true_MPa'].max(),df[k+'_estimate_MPa'].max())
        ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set_title(k);ax.set_xlabel('FEM true (MPa)');ax.set_ylabel('Recovered (MPa)')
    axs[-1].legend(fontsize=6,loc='best');fig.suptitle('B9A11Q-A | Conditional B5; TRAIN internal tuning only')
    fig.tight_layout();fig.savefig(out/'B9A11Q_A_inverse_recovery.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.2))
    for reg,group in jj.groupby('region'):
        ax.semilogy(np.arange(len(group)),group['condition'].to_numpy(),'o-',label=reg)
    ax.set_xlabel('Internal tuning case index');ax.set_ylabel('Whitened local Jacobian condition number')
    ax.set_title('Conditional inverse sensitivity conditioning')
    ax.legend();fig.tight_layout();fig.savefig(out/'B9A11Q_A_identifiability.png',dpi=190);plt.close(fig)
    summary={'stage':'B9A11Q-A','scope':'B5 known-PCA-interface, internal original 64J TRAIN tune',
             'fit_cases_total':len(df),'tune_cases_total':len(jj),'noise_std_mm':args.sigma_u,
             'median_absolute_pct_by_mode':meds.to_dict(orient='index'),
             'median_b5_fem_rmse_mm':float(jj['b5_vs_fem_forward_rmse_mm'].median()),
             'median_jacobian_condition':float(jj['condition'].median()),
             'minimum_singular_value_over_cases':float(jj['singular_min'].min()),
             'boundary_hits_by_mode':{mode:int(group[[k+'_hit_bound' for k in PKEYS]].to_numpy().sum()) for mode,group in df.groupby('mode')},
             'not_an_independent_validation':True,'not_full_beam_coupling':True,
             'q_from_FEM_tuning_cases':True,'checkpoint_and_65V_unchanged':True,
             'interpretation':'B5->B5 zero-noise tests optimizer / conditional local observability; FEM->B5 tests surrogate mismatch and inverse bias.'}
    (out/'B9A11Q_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11Q-A COMPLETE',out)
    print('Median error by mode:\n',meds.to_string())
    print('NOT a full-coupled digital twin; known q is supplied from FEM.')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--stage',choices=('selftest','check','run'),required=True)
    a.add_argument('--project',default='.')
    a.add_argument('--tag',default='pilot01')
    a.add_argument('--variant',choices=('centered','standardized'),default='centered')
    a.add_argument('--loss',choices=('U','grad'),default='grad')
    a.add_argument('--cases',type=int,default=6)
    a.add_argument('--sensors',type=int,default=20)
    a.add_argument('--noise-reps',type=int,default=2)
    a.add_argument('--sigma-u',type=float,default=.0002)
    a.add_argument('--max-nfev',type=int,default=150)
    a.add_argument('--seed',type=int,default=651021)
    args=a.parse_args()
    if args.stage=='selftest':selftest();return
    if not (3<=args.sensors<=50 and 1<=args.cases<=12 and 0<=args.noise_reps<=8 and args.sigma_u>0):
        raise ValueError('Invalid experiment configuration')
    run(args)
if __name__=='__main__':main()