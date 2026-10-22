#!/usr/bin/env python3
"""B9A11H-B: lightweight material inverse problem and Bayesian UQ from REAL 50-case
full-beam Abaqus response data (not a single 65V material point).

Mac / Python 3: numpy, pandas, scipy, matplotlib. No Abaqus / PyTorch needed.
Outputs are PROOF-OF-WORKFLOW, not experimental parameter identification.
A predefined polynomial response surface is fitted on original TRAIN cases;
regularization is selected using VALIDATION; TEST is used only after fitting.
Posterior demo uses a selected TEST case as noisy synthetic observations, and
remains conditional on Gaussian observation noise and validation-estimated
surrogate discrepancy. Neither B5 nor the 65V solver is used as the variable-
material forward operator. Do not call this a validated B5 Bayesian inverse.
"""
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

NAMES=('E1_MPa','E2_MPa','G12_MPa')
CENTER=np.array([45000.,12000.,4500.],dtype=float)
HALF=np.array([4500.,1200.,450.],dtype=float)
LOW=CENTER-HALF;HIGH=CENTER+HALF
LAMBDAS=(1e-8,1e-6,1e-4,1e-3,.01,.1,1.)
REQUIRED=('parameters.npy','coordinates.npy','U1.npy','U2.npy','U3.npy','split_assignment.csv','case_ids.csv','global_outputs.csv')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
    return h.hexdigest()


def dataset_dir(root,chosen):
    if chosen:return Path(chosen).expanduser().resolve()
    for p in (root/'data',root/'Unused_Files'/'DeepONet_Dataset_50'):
        if all((p/f).is_file() for f in REQUIRED):return p
    raise FileNotFoundError('No complete 50-case full-beam dataset. Expected '+', '.join(REQUIRED)+
                            ' in data/ or Unused_Files/DeepONet_Dataset_50. Pass --data-dir explicitly.')


def read_data(folder):
    for k in REQUIRED:
        if not (folder/k).is_file():raise FileNotFoundError(str(folder/k))
    X=np.asarray(np.load(folder/'parameters.npy',allow_pickle=False),dtype=float)
    xyz=np.asarray(np.load(folder/'coordinates.npy',allow_pickle=False),dtype=float)
    u1=np.asarray(np.load(folder/'U1.npy',allow_pickle=False),dtype=float)
    u2=np.asarray(np.load(folder/'U2.npy',allow_pickle=False),dtype=float)
    u3=np.asarray(np.load(folder/'U3.npy',allow_pickle=False),dtype=float)
    df=pd.read_csv(folder/'case_ids.csv');spl=pd.read_csv(folder/'split_assignment.csv')
    glob=pd.read_csv(folder/'global_outputs.csv')
    if X.shape!=(50,3) or xyz.shape!=(4592,3) or u1.shape!=(50,4592) or u2.shape!=(50,4592) or u3.shape!=(50,4592):
        raise RuntimeError('Expected 50 parameters and 50x4592 U1/U2 with 4592x3 coordinates')
    if list(df.columns)!=['Index','CaseID'] or not {'Index','CaseID','Split'}.issubset(spl.columns):
        raise RuntimeError('Missing Index/CaseID/Split mapping')
    df=df.sort_values('Index').reset_index(drop=True)
    spl=spl.sort_values('Index').reset_index(drop=True)
    if df.Index.tolist()!=list(range(50)) or not np.array_equal(df.CaseID.astype(str),spl.CaseID.astype(str)) or not np.array_equal(df.Index,spl.Index):
        raise RuntimeError('Case/split rows not in exact array order')
    ids=df.CaseID.astype(str).to_list()
    if len(set(ids))!=50 or len(glob)!=50 or glob.CaseID.nunique()!=50:
        raise RuntimeError('CaseID mismatch or duplicate')
    glob=glob.set_index('CaseID').loc[ids].reset_index()
    for k in (*NAMES,'Final_RF2_N','Final_U2_mm'):
        if k not in glob:raise RuntimeError('Missing global output '+k)
    G=glob[list(NAMES)].to_numpy(float)
    if np.max(np.abs(G-X))>5e-3:raise RuntimeError('Global CSV and parameter.npy do not match by CaseID')
    if any(not np.isfinite(z).all() for z in (X,xyz,u1,u2,u3,G,glob.Final_RF2_N.to_numpy(float))):
        raise RuntimeError('Nonfinite training data')
    if np.max(np.abs(X-CENTER)/HALF)>1.00001:
        raise RuntimeError('Some materials outside original 64E +/-10% prior bounds')
    if np.max(abs(glob.Final_U2_mm.to_numpy(float)+.2))>.004:
        raise RuntimeError('This does not appear to be the known displacement-controlled -0.2mm experiment')
    groups={g:np.flatnonzero(spl.Split.astype(str).str.lower().to_numpy()==g) for g in ('train','validation','test')}
    if any(len(v)<3 for v in groups.values()) or sum(len(v) for v in groups.values())!=50:
        raise RuntimeError('Missing/invalid train/validation/test split')
    if max(len(set(x)) for x in (groups['train'],groups['validation'],groups['test']))!=max(len(x) for x in groups.values()):
        raise RuntimeError('Duplicate split indices')
    return {'X':X,'xyz':xyz,'u1':u1,'u2':u2,'u3':u3,'rf':glob.Final_RF2_N.to_numpy(float),
            'ids':ids,'groups':groups,'folder':folder,'glob':glob}


def select_sensors(xyz,count=20,sensor_file=None):
    """Geometry-only default or pre-ranked training-derived full-field sensitivity sensors.
    Sensor selections are made without inspecting TEST outputs.
    """
    if not (5<=count<=50):raise ValueError('Use between 5 and 50 spatial sensor nodes')
    if sensor_file is not None:
        tab=pd.read_csv(sensor_file)
        if 'NodeIndex' not in tab:raise RuntimeError('Sensitivity sensor layout lacks NodeIndex')
        ids=tab.NodeIndex.astype(int).to_numpy()[:count]
        if len(ids)!=count or len(set(ids))!=count or np.any(ids<0) or np.any(ids>=len(xyz)):
            raise RuntimeError('Invalid sensitivity sensor ranking')
        labels=['sensitivity_%02d'%i for i in range(1,count+1)]
        return ids,[dict(sensor=labels[k],index=int(j),x_mm=float(xyz[j,0]),
                y_mm=float(xyz[j,1]),z_mm=float(xyz[j,2]),distance_from_target_mm=0.)
            for k,j in enumerate(ids)]
    nlower=int(np.ceil(count/2));nupper=count-nlower
    xmin,xmax=xyz[:,0].min(),xyz[:,0].max()
    ymin,ymax=xyz[:,1].min(),xyz[:,1].max()
    zm=(xyz[:,2].min()+xyz[:,2].max())/2
    idx=[];info=[]
    for surface,y,n in (('lower',ymin,nlower),('upper',ymax,nupper)):
        for k,frac in enumerate(np.linspace(-.87,.87,n)):
            pt=np.array([(xmin+xmax)/2+frac*(xmax-xmin)/2,y,zm])
            dist=np.linalg.norm(xyz-pt,axis=1)
            j=int(np.argmin(dist))
            if j in idx:raise RuntimeError('Duplicate geometry sensor location')
            idx.append(j)
            info.append(dict(sensor='%s_%02d'%(surface,k+1),index=j,
                  x_mm=float(xyz[j,0]),y_mm=float(xyz[j,1]),z_mm=float(xyz[j,2]),
                  distance_from_target_mm=float(dist[j])))
    return np.asarray(idx,dtype=int),info


def design(z):
    z=np.atleast_2d(np.asarray(z,float))
    a,b,c=z[:,0],z[:,1],z[:,2]
    return np.column_stack((np.ones(len(z)),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c))


def jacdesign(z):
    a,b,c=np.asarray(z,float)
    return np.array([[0.,1.,0.,0.,2*a,0.,0.,b,c,0.],
                     [0.,0.,1.,0.,0.,2*b,0.,a,0.,c],
                     [0.,0.,0.,1.,0.,0.,2*c,0.,a,b]],float).T


def compute_observations(d,sensor_count=20,sensor_file=None):
    inds,info=select_sensors(d['xyz'],sensor_count,sensor_file);train=d['groups']['train']
    y=[];labels=[];types=[];layout=[]
    for j,r in zip(inds,info):
        for comp,values in (('U1',d['u1']),('U2',d['u2']),('U3',d['u3'])):
            outname=comp+'_'+r['sensor']
            y.append(values[:,j]);labels.append(outname);types.append(comp)
            layout.append(dict(observable=outname,unit='mm',**r))
    y.append(d['rf']);labels.append('RF2_nose_N');types.append('RF')
    layout.append(dict(observable='RF2_nose_N',unit='N',sensor='nose_reaction',index=-1,
                       x_mm=float('nan'),y_mm=float('nan'),z_mm=float('nan'),distance_from_target_mm=float('nan')))
    y=np.array(y,float).T
    keep=np.asarray([np.std(y[train,j])>(.5 if types[j]=='RF' else 1e-6) for j in range(len(labels))])
    if np.sum(keep)<6 or not keep[-1]:raise RuntimeError('Too few parameter-varying responses; inspect sensors and force')
    return y[:,keep], [v for i,v in enumerate(labels) if keep[i]], [v for i,v in enumerate(types) if keep[i]], [v for i,v in enumerate(layout) if keep[i]]


def fit_model(z,y,alpha,indices):
    A=design(z[indices]);yt=y[indices]
    ym=yt.mean(axis=0)
    ys=np.maximum(yt.std(axis=0),1e-7)
    yy=(yt-ym)/ys
    penal=np.diag([0.]+[float(alpha)]*(A.shape[1]-1))
    coef=np.linalg.solve(A.T@A+penal,A.T@yy)
    return {'coef':coef,'mean':ym,'scale':ys,'alpha':float(alpha)}


def predict(model,z):
    return design(z)@model['coef']*model['scale']+model['mean']


def standardized_rmse(a,b,scale):
    return float(np.sqrt(np.mean(((a-b)/scale)**2)))


def model_pipeline(d,y):
    z=(d['X']-CENTER)/HALF
    tr,va,te=(d['groups'][g] for g in ('train','validation','test'))
    scale=np.maximum(y[tr].std(axis=0),1e-6)
    report=[]
    for alpha in LAMBDAS:
        model=fit_model(z,y,alpha,tr)
        p=predict(model,z[va])
        report.append({'alpha':alpha,'validation_standardized_rmse':standardized_rmse(p,y[va],scale)})
    best=min(report,key=lambda a:(a['validation_standardized_rmse'],a['alpha']))
    initial=fit_model(z,y,best['alpha'],tr)
    residual=y[va]-predict(initial,z[va])
    enlarged=fit_model(z,y,best['alpha'],np.r_[tr,va])
    truth=y[te];pred=predict(enlarged,z[te])
    # Finite validation cohort: shrink error covariance heavily towards diagonal.
    C=np.cov(residual,rowvar=False,ddof=1)
    if C.ndim!=2:C=np.atleast_2d(C)
    if not np.isfinite(C).all():raise RuntimeError('Invalid validation error covariance')
    C=.15*C+.85*np.diag(np.diag(C))
    # Add small positive floor; real experiment noise is added separately.
    C=C+np.eye(len(C))*1e-18
    return enlarged,C,report,{'test_case_ids':[d['ids'][i] for i in te],
        'n_train':len(tr),'n_validation':len(va),'n_test':len(te),
        'best_regularization':best,
        'full_test_rmse':float(np.sqrt(np.mean((truth-pred)**2))),
        'holdout_per_case':[{'case_id':d['ids'][i],
          'RF2_error_N':float(pred[j,-1]-truth[j,-1]),
          'observation_rmse':float(np.sqrt(np.mean((pred[j]-truth[j])**2)))}
          for j,i in enumerate(te)]}


def run_mcmc(model,z_map,obs,chol,jac,draws,burn,thin,seed):
    rng=np.random.default_rng(seed)
    H=jac.T@jac
    ev=np.linalg.eigvalsh(H)
    local=np.linalg.inv(H+np.eye(3)*max(1e-4,1e-7*max(ev)))
    P=np.linalg.cholesky(local+np.eye(3)*1e-10)*(2.38/np.sqrt(3.))
    n_chains=4
    allchains=[];rates=[]
    def loss(z):
        if np.any(z < -1) or np.any(z > 1):return math.inf
        r=solve_triangular(chol,predict(model,z)[0]-obs,lower=True)
        return .5*float(r@r)
    for i in range(n_chains):
        z=np.clip(z_map+rng.normal(size=3)@P.T*.25,-.999,.999)
        ll=loss(z);accepted=0;keep=[]
        for step in range(draws+burn):
            prop=z+rng.normal(size=3)@P.T
            l2=loss(prop)
            if math.log(max(rng.random(),1e-15)) < min(0.,ll-l2):
                z=prop;ll=l2;accepted+=1
            if step>=burn and (step-burn)%thin==0:keep.append(z.copy())
        rates.append(accepted/(draws+burn));allchains.append(np.asarray(keep))
    allchains=np.stack(allchains)
    m,n,_=allchains.shape
    W=np.mean(np.var(allchains,axis=1,ddof=1),axis=0)
    B=n*np.var(np.mean(allchains,axis=1),axis=0,ddof=1)
    rhat=np.sqrt(np.maximum(((n-1)/n*W+B/n)/np.maximum(W,1e-14),0))
    return allchains,np.asarray(rates),rhat


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--data-dir',help='Override dataset directory')
    ap.add_argument('--stage',choices=('check','run'),default='check')
    ap.add_argument('--tag',default='run01')
    ap.add_argument('--num-sensors',type=int,choices=(5,10,20,50),default=20)
    ap.add_argument('--sensor-layout',help='Optional C sensitivity-ranked sensor CSV. Uses top --num-sensors and never inspects test responses.')
    ap.add_argument('--observation',help='Optional external measurements CSV: observable,value,noise_std. Omit for held-out pseudo-observation demo.')
    ap.add_argument('--noise-u-mm',type=float,default=.0002,help='Assumed measurement 1 sigma, displacement mm, demo only')
    ap.add_argument('--noise-rf-N',type=float,default=5.,help='Assumed measurement 1 sigma, reaction force N, demo only')
    ap.add_argument('--draws',type=int,default=4000)
    ap.add_argument('--burn',type=int,default=1200)
    ap.add_argument('--thin',type=int,default=5)
    ap.add_argument('--seed',type=int,default=20261017)
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    if not (args.tag.replace('_','').replace('-','').isalnum() and args.noise_u_mm>0 and args.noise_rf_N>0 and
            args.draws>=500 and args.burn>=200 and 1<=args.thin<=args.draws):
        raise ValueError('Invalid output tag, noise, or MCMC size')
    folder=dataset_dir(root,args.data_dir)
    d=read_data(folder)
    ranked=Path(args.sensor_layout).expanduser().resolve() if args.sensor_layout else root/'results'/'65AM_B9A11H_v2'/args.tag/'C_sensitivity'/'B9A11H_C_sensor_rankings.csv'
    if args.sensor_layout and not ranked.is_file():raise FileNotFoundError('Sensor ranking missing: '+str(ranked))
    ranked=ranked if ranked.is_file() else None
    y,labels,types,layout=compute_observations(d,args.num_sensors,ranked)
    mats=d['X'];z=(mats-CENTER)/HALF
    print('DATA:',folder,'| 50 cases','| train/validation/test:',*[len(d['groups'][g]) for g in ('train','validation','test')])
    print('Parameter-varying observations:',len(labels),'including RF:',labels[-1])
    print('Sensor positions:',('train/validation sensitivity-ranked from '+str(ranked)) if ranked else 'geometry-only (no test labels)')
    if args.stage=='check':
        print('PRECHECK PASS. Source physical model/units still require ODB provenance review.')
        return
    model,cov,cv,holdout=model_pipeline(d,y)
    obs_true=None;known=None
    if args.observation:
        m=pd.read_csv(args.observation)
        if not {'observable','value','noise_std'}.issubset(m.columns):
            raise RuntimeError('Measurements CSV needs observable,value,noise_std')
        if len(m)==0 or m.observable.duplicated().any():raise RuntimeError('Empty/duplicated measurements')
        index={s:i for i,s in enumerate(labels)}
        bad=set(m.observable)-set(index)
        if bad:raise RuntimeError('Unknown observable(s): '+str(bad))
        chosen=np.array([index[s] for s in m.observable],dtype=int)
        obs=m.value.to_numpy(float);sigma=m.noise_std.to_numpy(float)
        if not np.isfinite(obs).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise RuntimeError('Invalid measurement/noise values')
        obs_names=m.observable.astype(str).tolist()
        kind='user_supplied_measurements'
    else:
        case=int(d['groups']['test'][0]);chosen=np.arange(len(labels))
        sigma=np.array([args.noise_rf_N if k=='RF' else args.noise_u_mm for k in types],float)
        rng=np.random.default_rng(args.seed)
        obs_true=y[case].copy()
        obs=obs_true+rng.normal(size=len(sigma))*sigma
        known=mats[case].tolist()
        obs_names=labels
        kind='synthetic_holdout_case_with_assumed_noise'
    if len(chosen)<3:raise RuntimeError('Need >=3 informative observables to identify 3 material parameters')
    covsub=cov[np.ix_(chosen,chosen)]+np.diag(sigma*sigma)
    chol=np.linalg.cholesky(covsub)
    def chi2(v):
        pred=predict(model,v)[0][chosen]
        r=solve_triangular(chol,pred-obs,lower=True)
        return float(r@r)
    initial=[np.zeros(3),*list(z[d['groups']['train'][:3]])]
    candidates=[minimize(lambda w:.5*chi2(w),np.asarray(w),bounds=[(-1,1)]*3,method='L-BFGS-B',
                         options={'maxiter':600,'ftol':1e-12}) for w in initial]
    ans=min(candidates,key=lambda v:v.fun)
    zhat=np.clip(ans.x,-1,1)
    # J[z_unit] -> whitened observation derivative. Singular values measure identifiability.
    J=(jacdesign(zhat).T@model['coef']*model['scale'])[:,chosen]
    JW=solve_triangular(chol,J.T,lower=True)
    sv=np.linalg.svd(JW,compute_uv=False)
    rank=int((sv>max(sv[0],1e-10)*1e-4).sum())
    chains,accept,rhat=run_mcmc(model,zhat,obs,chol,JW,args.draws,args.burn,args.thin,args.seed+1)
    posterior=chains.reshape(-1,3)*HALF+CENTER
    lo,median,hi=np.percentile(posterior,[2.5,50,97.5],axis=0)
    MAP=CENTER+zhat*HALF
    out=root/'results'/'65AM_B9A11H_v2'/args.tag/('B_inverse_UQ_%02d'%args.num_sensors)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(layout).to_csv(out/'B9A11H_sensor_layout.csv',index=False)
    pd.DataFrame(cv).to_csv(out/'B9A11H_regularization_validation.csv',index=False)
    pd.DataFrame(posterior,columns=NAMES).to_csv(out/'B9A11H_posterior_samples.csv',index=False)
    pd.DataFrame({'observable':obs_names,'value':obs,'noise_std':sigma}).to_csv(out/'B9A11H_inference_observations.csv',index=False)
    covpred=predict(model,(posterior[:min(300,len(posterior))]-CENTER)/HALF)[:,chosen]
    pred_ci=np.percentile(covpred,[2.5,50,97.5],axis=0)
    pd.DataFrame({'observable':obs_names,'observed':obs,'pred_low':pred_ci[0],
                  'pred_median':pred_ci[1],'pred_high':pred_ci[2]}).to_csv(out/'B9A11H_posterior_predictive.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4.5))
        # Comparable scale: percent of initial material reference.
        percent=100*(median-CENTER)/CENTER
        percent_lo=100*(lo-CENTER)/CENTER
        percent_hi=100*(hi-CENTER)/CENTER
        ax.errorbar(range(3),percent,yerr=[percent-percent_lo,percent_hi-percent],fmt='o',capsize=5)
        if known:ax.scatter(range(3),100*(np.asarray(known)-CENTER)/CENTER,marker='x',label='Known synthetic truth')
        ax.set_xticks(range(3),NAMES);ax.set_ylabel('Change from nominal material (%)');ax.set_title('Inverse UQ parameter intervals (conditional)')
        if known:ax.legend()
        fig.tight_layout();fig.savefig(out/'B9A11H_parameter_intervals.png',dpi=170);plt.close(fig)
    except ImportError:pass
    summary={'stage':'B9A11H-B','forward_model':'quadratic ridge full-FEM response surface; NOT coupled 65V or B5',
             'type':kind,'sensor_count':args.num_sensors,'sensor_selection':str(ranked) if ranked else 'geometry_only','components':'U1,U2,U3,RF2','source_data_directory':str(folder),'source_checksums':{k:sha(folder/k) for k in REQUIRED},
             'cohort':{'train_case_ids':[d['ids'][i] for i in d['groups']['train']],
                       'validation_case_ids':[d['ids'][i] for i in d['groups']['validation']],
                       'test_case_ids':holdout['test_case_ids']},
             'regularization_selection':holdout['best_regularization'],
             'surrogate_heldout_test':holdout,
             'measurement_assumptions':{'sigma_U_mm_demo':args.noise_u_mm,'sigma_RF2_N_demo':args.noise_rf_N,
                'covariance':'validation residual covariance 15% + diagonal 85%, plus measurement variance'},
             'inverse':{'known_test_material_MPa':known,'map_MPa':dict(zip(NAMES,MAP.tolist())),
                        'posterior_median_MPa':dict(zip(NAMES,median.tolist())),
                        'posterior_95pct_intervals_MPa':{k:[float(lo[i]),float(hi[i])] for i,k in enumerate(NAMES)},
                        'posterior_acceptance_by_chain':accept.tolist(),
                        'rhat_by_parameter':dict(zip(NAMES,rhat.tolist())),
                        'whitened_jacobian_singular_values':sv.tolist(),
                        'local_identifiability_rank':rank,
                        'local_condition_number':float(sv[0]/max(sv[-1],1e-14)),
                        'map_at_prior_boundary':bool(np.any(abs(zhat)>.999)),
                        'n_observables':len(chosen),'map_chi2':chi2(zhat),
                        'posterior_diagnostics_preliminary_pass':bool(np.max(rhat)<1.10 and np.all((accept>.08)&(accept<.75)) and rank==3),
                        'warning_if_failed':'Do not interpret narrow intervals as reliable when chains fail to mix or local identifiability is poor.'},
             'truth_constraints':'Demo truth selected from ORIGINAL held-out test; training/validation fitting never sees that case.',
             'limitations':['This is simulation-based Bayesian UQ, not posterior experimental identification.',
                'Approximate emulator uncertainty from limited validation data; posterior coverage is not guaranteed.',
                'The experiment controls final displacement; Final_U2 is almost constant and intentionally excluded as information.',
                'Full-FEM historical dataset provenance, boundary conditions, orientation and reaction-set definition need physical audit.',
                'No B5 coupling/discrepancy is propagated; add a distinct coupled-model discrepancy once that model is validated.'],
             'outputs':['B9A11H_sensor_layout.csv','B9A11H_regularization_validation.csv',
                       'B9A11H_posterior_samples.csv','B9A11H_posterior_predictive.csv','B9A11H_parameter_intervals.png']}
    (out/'B9A11H_inverse_UQ_summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE | B9A11H inverse/UQ:',out)
    print('MAP MPa:',dict(zip(NAMES,MAP.round(3))))
    print('95% intervals MPa:',summary['inverse']['posterior_95pct_intervals_MPa'])
    print('Identifiability rank',rank,'of 3 | Rhat',rhat.round(3).tolist(),'| accept',accept.round(3).tolist())
    print('WARNING: synthetic test demonstration, not experimental calibration.')

if __name__=='__main__':main()
