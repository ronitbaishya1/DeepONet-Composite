#!/usr/bin/env python3
"""B9A11M-A: continuous damage-response interpolation at LOCKED virtual sensors.

Fit ONLY on eight B9A11I damage templates + healthy. Holdout x=+/-6 and
severity 20/30% is read ONLY by --stage validate; never used for fitting.
Physical features: [LE22,LE12,U1,U2,U3], strains dimensionless; U in mm.
This is a sparse grid interpolation, not a trained neural operator.
"""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
TRAIN={
 'I_DMG_X01_D01':(-8.,.15),'I_DMG_X01_D02':(-8.,.35),
 'I_DMG_X02_D01':(-4.2,.15),'I_DMG_X02_D02':(-4.2,.35),
 'I_DMG_X03_D01':(4.2,.15),'I_DMG_X03_D02':(4.2,.35),
 'I_DMG_X04_D01':(8.,.15),'I_DMG_X04_D02':(8.,.35)}
HOLD={'I_DMG_X05_D03':(-6.,.20),'I_DMG_X05_D04':(-6.,.30),
      'I_DMG_X06_D03':(6.,.20),'I_DMG_X06_D04':(6.,.30)}
COMP=['LE22','LE12','U1','U2','U3']
X=np.array([-8.,-4.2,4.2,8.]);D=np.array([.15,.35])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_k(root):
 p=root/'65AM_B9A11K_surface_damage_design.py'
 if not p.is_file():raise FileNotFoundError('Missing required B9A11K script '+str(p))
 spec=importlib.util.spec_from_file_location('KforM',str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m

def model_predict(x,d,anchors,levels,tensor):
 """tensor shape (Nx,Nd,Nsites,5); piecewise bilinear, bounded domain."""
 x=float(x);d=float(d);xx=np.asarray(anchors);dd=np.asarray(levels)
 if not (xx[0]-1e-10<=x<=xx[-1]+1e-10 and 0<=d<=dd[-1]+1e-10):
  raise ValueError('Out of fitted domain x [%g,%g], severity [0,%g]'%(xx[0],xx[-1],dd[-1]))
 if d==0:return np.zeros_like(tensor[0,0])
 i=int(np.clip(np.searchsorted(xx,x,side='right')-1,0,len(xx)-2))
 t=float(np.clip((x-xx[i])/(xx[i+1]-xx[i]),0,1))
 byseverity=(1-t)*tensor[i]+t*tensor[i+1]
 if d<=dd[0]:return byseverity[0]*(d/dd[0])
 j=int(np.clip(np.searchsorted(dd,d,side='right')-1,0,len(dd)-2))
 s=float(np.clip((d-dd[j])/(dd[j+1]-dd[j]),0,1))
 return (1-s)*byseverity[j]+s*byseverity[j+1]

def inputs(root,include_hold=False):
 K=load_k(root);tags=('healthy',)+tuple(TRAIN)+(tuple(HOLD) if include_hold else ())
 cases=K.load_cases(root,root/'B9A11J_exports',tags)
 sites,groups,nodes=K.design_sites(cases['healthy'])
 obs=K.read_virtual_observables(cases,groups,nodes)
 lock=root/'results'/'65AM_B9A11L_balanced_design'/'design01'/'B9A11L_locked_sensor_design.csv'
 if not lock.is_file():raise FileNotFoundError('Missing frozen design '+str(lock))
 frame=pd.read_csv(lock)
 assert set(frame.sensor_count)=={10,20}
 for row in frame.itertuples():
  match=sites.iloc[int(row.site_id)]
  if (int(match.displacement_node_label)!=int(row.displacement_node_label) or
      abs(float(match.center_x_mm)-float(row.center_x_mm))>1e-5 or
      match.surface!=row.surface):raise RuntimeError('Frozen sensor layout does not match FE exports')
 arrays={};rows={}
 for n,g in frame.groupby('sensor_count'):
  sel=g.sort_values('sensor_rank').site_id.astype(int).to_numpy()
  if len(sel)!=n or len(set(sel))!=n:raise RuntimeError('Invalid sensor group')
  if len(set(g.displacement_node_label))!=n:raise RuntimeError('Duplicate displacement nodes')
  def vals(name):
   a=obs[name];b=obs['healthy']
   return np.concatenate([a['strain'][sel]-b['strain'][sel],a['u'][sel]-b['u'][sel]],axis=1)
  train=np.zeros((4,2,n,5),float)
  for name,(x,d) in TRAIN.items():
   i=list(X).index(x);j=list(D).index(d);train[i,j]=vals(name)
  arrays[str(n)]=train
  if include_hold:rows[str(n)]={name:vals(name) for name in HOLD}
 return arrays,rows,sha(lock)

def loo_floors(tensor):
 # Leave out an ENTIRE interior x location (both severities) for
 # out-of-location interpolation diagnostics; limited four-location CV.
 v=[]
 for drop in [1,2]:
  keep=[i for i in range(4) if i!=drop]
  for j,d in enumerate(D):
   guess=model_predict(X[drop],d,X[keep],D,tensor[keep])
   v.append(guess-tensor[drop,j])
 residual=np.asarray(v)
 return np.sqrt(np.mean(residual**2,axis=(0,1))) # 5 physical-component floors

def output(root,tag):return root/'results'/'65AM_B9A11M_continuous'/tag

def fit(root,tag,stage):
 arr,_,lockhash=inputs(root,False)
 print('M-A FIT CHECK PASS: 9 FE states; locked 10 and 20 virtual sensor locations; holdouts NOT read')
 if stage=='check':return
 out=output(root,tag)
 if out.exists():raise RuntimeError('Refusing overwrite '+str(out))
 out.mkdir(parents=True)
 floors={};bundle={'anchor_x':X,'anchor_d':D,'component_names':np.array(COMP),'locked_sha256':np.array(lockhash)}
 for n in (10,20):
  bundle['delta_'+str(n)]=arr[str(n)]
  floors[str(n)]=loo_floors(arr[str(n)]).tolist()
  bundle['cv_floor_'+str(n)]=np.array(floors[str(n)])
 np.savez(out/'B9A11M_training_model.npz',**bundle)
 report={'stage':'B9A11M-A-fit','train_cases':['healthy']+list(TRAIN),
   'heldout_cases_not_loaded':list(HOLD),'sensor_design_sha256':lockhash,
   'model':'piecewise bilinear response-difference interpolation in x and severity',
   'anchor_x_mm':X.tolist(),'anchor_severity':D.tolist(),
   'leave_inner_location_out_RMS_per_component_LE_dimless_U_mm':floors,
   'caveats':['Only four x anchors and two severities','CV floor is not calibrated predictive uncertainty',
     'No unknown healthy material parameters','No real surface gauge validation','No neural operator training']}
 (out/'B9A11M_A_fit_summary.json').write_text(json.dumps(report,indent=2))
 print('M-A FIT DONE:',out)

def validate(root,tag):
 out=output(root,tag);f=out/'B9A11M_training_model.npz'
 if not f.is_file():raise FileNotFoundError('First run --stage fit: '+str(f))
 mdl=np.load(f,allow_pickle=False)
 arrays,holds,lockhash=inputs(root,True)
 if str(mdl['locked_sha256'])!=lockhash:raise RuntimeError('Locked design changed since fitting')
 rows=[]
 for n in (10,20):
  raw=np.asarray(mdl['delta_'+str(n)])
  if not np.allclose(raw,arrays[str(n)],rtol=0,atol=1e-12):raise RuntimeError('Training features changed')
  for name,(x,d) in HOLD.items():
   pred=model_predict(x,d,X,D,raw);truth=holds[str(n)][name];delta=pred-truth
   z={'case_id':name,'true_x_mm':x,'true_severity':d,'sensor_count':n}
   for c,comp in enumerate(COMP):
    rmse=np.sqrt(np.mean(delta[:,c]**2))
    scale=25e-6 if c<2 else .0002
    z['RMSE_'+comp]=float(rmse)
    z['RMSE_over_25ue_or_0p0002mm_'+comp]=float(rmse/scale)
   # Difference of the measured healthy and damaged states has sqrt2 noise.
   sigma=np.array([25e-6,25e-6,.0002,.0002,.0002])
   z['whitened_RMS_at_25ue']=float(np.sqrt(np.mean((delta/(np.sqrt(2)*sigma))**2)))
   z['signal_RMS_over_difference_noise']=float(np.sqrt(np.mean((truth/(np.sqrt(2)*sigma))**2)))
   rows.append(z)
 df=pd.DataFrame(rows)
 dest=out/'B9A11M_A_holdout_forward_errors.csv'
 if dest.exists():raise RuntimeError('Refusing overwrite of validation output '+str(dest))
 df.to_csv(dest,index=False)
 summary={'stage':'B9A11M-A-validate','holdout_cases':list(HOLD),'training_unchanged':True,
   'n_rows':len(df),'mean_whitened_RMS_at_25ue_by_sensor_count':df.groupby('sensor_count')['whitened_RMS_at_25ue'].mean().to_dict(),
   'max_whitened_RMS_at_25ue':float(df.whitened_RMS_at_25ue.max()),
   'provisional_per_component_gate':'Require errors < assumed difference noise before trusting downstream inverse; no gate automatically asserted',
   'warning':'Holdout used ONLY to evaluate, never to choose interpolator, prior, discrepancy scale or sensor positions'}
 (out/'B9A11M_A_holdout_summary.json').write_text(json.dumps(summary,indent=2))
 print('M-A HOLDOUT EVALUATION DONE:',dest,'\n',df[['case_id','sensor_count','whitened_RMS_at_25ue']].to_string(index=False))

def selftest():
 zz=np.zeros((4,2,3,5));
 for i,x in enumerate(X):
  for j,d in enumerate(D):zz[i,j,:,:]=d*(2.+.25*x)
 assert np.allclose(model_predict(6.,.20,X,D,zz),.2*(2+1.5))
 assert np.allclose(model_predict(4.2,.35,X,D,zz),zz[2,1])
 assert np.array_equal(model_predict(0.,0,X,D,zz),np.zeros((3,5)))
 assert np.all(np.isfinite(loo_floors(zz)))
 print('M-A SELFTEST PASS: anchor recovery, interpolation, healthy zero, LOO')

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['selftest','check','fit','validate']);p.add_argument('--project',default='.');p.add_argument('--tag',default='run01');p.add_argument('--healthy-material-confirmed',action='store_true')
 a=p.parse_args()
 if a.stage=='selftest':return selftest()
 if not a.healthy_material_confirmed:raise RuntimeError('Physically confirm same healthy material, geometry, loading and mesh before this flag')
 root=Path(a.project).resolve()
 if a.stage in ('check','fit'):fit(root,a.tag,a.stage)
 else:validate(root,a.tag)
if __name__=='__main__':main()
