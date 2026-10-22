#!/usr/bin/env python3
"""B9A11M-B: controlled grid Bayesian damage inference with healthy null model.

Reads B9A11M-A prefit model, LOCKED 10/20 virtual sensor positions, and
four previously held-out FEM cases. Healthy-vs-damage prior is 50:50, and
continuous location/severity use uniform quadrature grid. The decision
threshold is calibrated to ~5% synthetic HEALTHY false alarms, without
using ANY holdout responses to fit or calibrate.

Results are a FEM-based simulated inference proof of concept, not experimental
UQ or a joint inverse over E1/E2/G12. Discrepancy floor is a heuristic from
train-only leave-one-location-out and not a calibrated stochastic-process model.
"""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp,expit

TRAIN=('healthy','I_DMG_X01_D01','I_DMG_X01_D02','I_DMG_X02_D01','I_DMG_X02_D02','I_DMG_X03_D01','I_DMG_X03_D02','I_DMG_X04_D01','I_DMG_X04_D02')
HOLD={'I_DMG_X05_D03':(-6.,.20),'I_DMG_X05_D04':(-6.,.30),
      'I_DMG_X06_D03':(6.,.20),'I_DMG_X06_D04':(6.,.30)}
SIGMA_STR=(10.,25.,50.)
NAMES=('LE22','LE12','U1','U2','U3')

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load(root,tag,read_holdout):
 A=module(root/'65AM_B9A11M_A_continuous_forward.py','b9a11m_A_import')
 K=A.load_k(root)
 p=root/'results'/'65AM_B9A11M_continuous'/tag/'B9A11M_training_model.npz'
 if not p.is_file():raise FileNotFoundError('Run B9A11M-A fit first: '+str(p))
 pkg=np.load(p,allow_pickle=False)
 tags=TRAIN+tuple(HOLD) if read_holdout else TRAIN
 cases=K.load_cases(root,root/'B9A11J_exports',tags)
 sites,groups,nodes=K.design_sites(cases['healthy'])
 obs=K.read_virtual_observables(cases,groups,nodes)
 lock=root/'results'/'65AM_B9A11L_balanced_design'/'design01'/'B9A11L_locked_sensor_design.csv'
 if A.sha(lock)!=str(pkg['locked_sha256']):raise RuntimeError('Frozen layout sha mismatch')
 d=pd.read_csv(lock)
 for r in d.itertuples():
  row=sites.iloc[int(r.site_id)]
  if int(row.displacement_node_label)!=int(r.displacement_node_label):raise RuntimeError('Sensor location changed')
 return A,pkg,obs,d

def feature(obs,name,ids):
 a=obs[name];b=obs['healthy']
 return np.concatenate([a['strain'][ids]-b['strain'][ids],a['u'][ids]-b['u'][ids]],axis=1)

def grid(A,pkg,n,x_step=.25,d_step=.01):
 # Model supports x within trained [-8,8], damage severity within [.15,.35]
 X=np.linspace(-8,8,int(round(16/x_step))+1)
 D=np.linspace(.15,.35,int(round(.20/d_step))+1)
 anchorx=np.asarray(pkg['anchor_x']);anchord=np.asarray(pkg['anchor_d']);tens=np.asarray(pkg['delta_'+str(n)])
 z=np.stack([A.model_predict(x,d,anchorx,anchord,tens) for x in X for d in D])
 return X,D,z,np.asarray(pkg['cv_floor_'+str(n)])

def maha_matrix(y,preds,noise_scale,cv_floor,rho):
 """Correct Gaussian Mahalanobis for each component's equicorrelated sensor errors.

 C_component=(2 sigma^2*(1-rho) + model_floor^2)*I + (2 sigma^2*rho)*11^T.
 Heuristic model floor treated independent across sites (uncertainty NOT calibrated).
 """
 y=np.asarray(y);P=np.asarray(preds)
 if y.ndim==2:y=y[None,...]
 T,ns,C=y.shape
 assert P.shape[1:]==(ns,C)
 Q=np.zeros((T,len(P)))
 for c in range(C):
  sig=float(noise_scale[c]);floor=float(cv_floor[c]);
  a=2*sig*sig*(1-rho)+floor*floor
  b=2*sig*sig*rho
  z=y[:,:,c];p=P[:,:,c]
  zsq=np.sum(z*z,axis=1)[:,None]
  psq=np.sum(p*p,axis=1)[None,:]
  q=(zsq+psq-2*z@p.T)/a
  if b>0:
   sums=np.sum(z,axis=1)[:,None]-np.sum(p,axis=1)[None,:]
   q-=b*sums*sums/(a*(a+ns*b))
  Q+=np.maximum(q,0)
 return Q

def posterior(y,templates,noise_scale,cv_floor,rho):
 # Use equal 50:50 total prior mass on 'healthy' and on 'damaged'.
 # Uniform discrete-grid quadrature for continuous x and severity.
 qd=maha_matrix(y,templates,noise_scale,cv_floor,rho)
 qh=maha_matrix(y,np.zeros_like(templates[:1]),noise_scale,cv_floor,rho)[:,0]
 l=-.5*qd
 ld=logsumexp(l,axis=1)-math.log(len(templates))
 ph=expit(ld+.5*qh)
 # Conditional damage-location/severity distribution over quadrature grid.
 cond=np.exp(l-logsumexp(l,axis=1,keepdims=True))
 return ph,cond

def simulate(base,ns,noise_scale,rho,T,rng):
 # Difference between TWO physical noisy states => sqrt2 times one-state noise.
 iid=rng.normal(size=(T,ns,5));shared=rng.normal(size=(T,1,5))
 out=(math.sqrt(1-rho)*iid+math.sqrt(rho)*shared)*noise_scale[None,None,:]*math.sqrt(2)
 return base[None,:,:]+out

def evaluate(root,tag,stage,trials,calib,seed):
 A,pkg,obs,design=load(root,tag,stage=='run')
 holdout_path=root/'results'/'65AM_B9A11M_continuous'/tag/'B9A11M_A_holdout_summary.json'
 if not holdout_path.is_file():raise FileNotFoundError('Run B9A11M-A --stage validate before inversion: '+str(holdout_path))
 holdout_summary=json.loads(holdout_path.read_text())
 holdout_rms=float(holdout_summary['max_whitened_RMS_at_25ue'])
 print('M-B HOLDOUT FORWARD max whitened RMS at 25ue:',round(holdout_rms,4), '| provisional pass <1:',holdout_rms<1)
 print('M-B CHECK PASS: preserved training model, locked sites and healthy null; heldouts read only by RUN')
 if stage=='check':return
 out=root/'results'/'65AM_B9A11M_inverse'/tag
 if out.exists():raise RuntimeError('Refusing overwrite '+str(out))
 out.mkdir(parents=True)
 rng=np.random.default_rng(seed)
 rows=[];diagnostics=[]
 for n in (10,20):
  ids=design[design.sensor_count==n].sort_values('sensor_rank').site_id.to_numpy(dtype=int)
  xgrid,dgrid,templates,cv_floor=grid(A,pkg,n)
  xx=np.repeat(xgrid,len(dgrid));dd=np.tile(dgrid,len(xgrid))
  for strain_noise in SIGMA_STR:
   sigma=np.array([strain_noise*1e-6,strain_noise*1e-6,.0002,.0002,.0002])
   for rho in (0.,.25):
    # Calibration uses HEALTHY ONLY and never reads heldout for fitting.
    healthy=np.zeros((n,5))
    yh=simulate(healthy,n,sigma,rho,calib,rng)
    pnull,_=posterior(yh,templates,sigma,cv_floor,rho)
    threshold=float(np.quantile(pnull,.95))
    diagnostics.append({'sensor_count':n,'sigma_strain_ue':strain_noise,'common_noise_rho':rho,
       'healthy_calibration_trials':calib,'detection_threshold_for_nominal_5pct_FPR':threshold,
       'observed_healthy_calibration_FPR':float(np.mean(pnull>threshold)),
       'CV_floor_LE22_ue':cv_floor[0]*1e6,'CV_floor_LE12_ue':cv_floor[1]*1e6,
       'CV_floor_U_RMS_mm':float(np.sqrt(np.mean(cv_floor[2:]**2)))})
    for name in ('healthy',)+tuple(HOLD):
     y0=feature(obs,name,ids)
     y=simulate(y0,n,sigma,rho,trials,rng)
     pdmg,w=posterior(y,templates,sigma,cv_floor,rho)
     detected=pdmg>threshold
     dx=np.sum(w*xx[None,:],axis=1);dv=np.sum(w*dd[None,:],axis=1)
     side=np.sum(w[:,xx>0],axis=1)>.5
     truex,truedd=HOLD.get(name,(None,None))
     r={'case_id':name,'sensor_count':n,'sigma_strain_ue':strain_noise,'common_noise_rho':rho,
        'trials':trials,'threshold':threshold,'damage_positive_rate':float(np.mean(detected)),
        'p_damage_median':float(np.median(pdmg)),'p_damage_p10':float(np.percentile(pdmg,10)),
        'p_damage_p90':float(np.percentile(pdmg,90)),
        'true_x_mm':truex,'true_severity':truedd,
        'mean_location_abs_error_mm_when_detected':None,
        'mean_severity_abs_error_when_detected':None,
        'joint_detection_correct_side_rate':None}
     if truex is not None:
      r['joint_detection_correct_side_rate']=float(np.mean(detected & (side==(truex>0))))
      if detected.any():
       r['mean_location_abs_error_mm_when_detected']=float(np.mean(abs(dx[detected]-truex)))
       r['mean_severity_abs_error_when_detected']=float(np.mean(abs(dv[detected]-truedd)))
     rows.append(r)
 df=pd.DataFrame(rows);cal=pd.DataFrame(diagnostics)
 # 2 counts * 3 noises * 2 correlation models * 5 states = 60 rows.
 if len(df)!=60 or set(df.case_id)!=set(('healthy',)+tuple(HOLD)):
  raise RuntimeError('Incomplete case coverage')
 df.to_csv(out/'B9A11M_B_detection_localization.csv',index=False)
 cal.to_csv(out/'B9A11M_B_healthy_calibration.csv',index=False)
 report={'stage':'B9A11M-B','result_rows':len(df),'number_of_heldout_cases':4,
  'heldout_forward_max_whitened_RMS_at_25ue':holdout_rms,
  'provisional_forward_screen_pass':bool(holdout_rms<1),
  'if_forward_screen_fails':'Interpret all inverse results as exploratory; do not claim reliable posterior',
  'trials_per_state':trials,'healthy_calibration_per_configuration':calib,
  'forward_model':'B9A11M-A fixed piecewise-bilinear interpolation; trained only on 8 old damage cases',
  'Bayesian_prior':'50% healthy, 50% damaged; uniform damage x [-8,8], severity [.15,.35]',
  'threshold':'95th percentile of separately simulated healthy probability using the assumed noise law',
  'modeled_correlations':'Matched independent rho=0 and common-site rho=0.25 Gaussian observation errors',
  'model_error':'Train-only leave-inner-location-out residual RMS; heuristic per-component independent floor, not calibrated',
  'key_warnings':['Four FEM holdouts do not establish clinical/real-world damage calibration',
   'Material properties fixed at healthy nominal; no E1/E2/G12 UQ',
   'No real surface strains or strain-gauge calibration',
   'Posterior x,d conditional on damage is descriptive and can be biased if damage is missed',
   'Use prior-locked test results only: do not tune on these holdouts'],
  'files':['B9A11M_B_detection_localization.csv','B9A11M_B_healthy_calibration.csv']}
 (out/'B9A11M_B_summary.json').write_text(json.dumps(report,indent=2))
 print('M-B DONE',out)
 print(df[(df.sensor_count==20)&(df.sigma_strain_ue==25)&(df.common_noise_rho==0)][['case_id','damage_positive_rate','joint_detection_correct_side_rate','mean_location_abs_error_mm_when_detected']].to_string(index=False))

def selftest():
 from scipy.special import expit
 v=np.zeros((3,2,5));v[1,:,:]=.01;v[2,:,:]=.02
 sig=np.ones(5)*.01;fl=np.ones(5)*.002
 z=maha_matrix(np.zeros((5,2,5)),v,sig,fl,0)
 assert z.shape==(5,3) and np.min(z)>=0
 p,c=posterior(np.zeros((2,2,5)),v,sig,fl,0.25)
 assert len(p)==2 and np.all((p>=0)&(p<=1)) and np.allclose(c.sum(axis=1),1)
 s=simulate(np.zeros((2,5)),2,sig,.25,12,np.random.default_rng(7))
 assert s.shape==(12,2,5)
 print('M-B SELFTEST PASS: Gaussian noise, covariance, mixture prior, grid posterior')

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['selftest','check','run']);p.add_argument('--project',default='.');p.add_argument('--tag',default='run01');p.add_argument('--trials',type=int,default=150);p.add_argument('--healthy-calibration',type=int,default=1000);p.add_argument('--seed',type=int,default=52431);p.add_argument('--healthy-material-confirmed',action='store_true')
 a=p.parse_args()
 if a.stage=='selftest':return selftest()
 if not a.healthy_material_confirmed:raise RuntimeError('Recheck matched healthy/damage material and loading')
 if a.trials<20 or a.healthy_calibration<300:raise ValueError('Too few trials/calibration samples')
 evaluate(Path(a.project).resolve(),a.tag,a.stage,a.trials,a.healthy_calibration,a.seed)
if __name__=='__main__':main()
