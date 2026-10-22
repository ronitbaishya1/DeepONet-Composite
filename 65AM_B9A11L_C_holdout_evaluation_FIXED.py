#!/usr/bin/env python3
"""B9A11L-C: frozen-gauge TEST on four NEW FEM damage positions/severities.

Template-based baseline, NOT a trained damage operator or continuous Bayesian
posterior. The original 8 + healthy are known templates. The 4 newly solved
FEM cases at x=+-6 mm and severity 20%,30% are not used in gauge selection.
Assumes the original healthy material, mesh, output orientation, contacts,
and source coordinates are unchanged. LE22/LE12 are near-surface C3D8R
centroid averaged proxies, NOT measured surface gauges.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN=('healthy','I_DMG_X01_D01','I_DMG_X01_D02','I_DMG_X02_D01','I_DMG_X02_D02',
       'I_DMG_X03_D01','I_DMG_X03_D02','I_DMG_X04_D01','I_DMG_X04_D02')
HOLD=('I_DMG_X05_D03','I_DMG_X05_D04','I_DMG_X06_D03','I_DMG_X06_D04')
TRUTH={'I_DMG_X01_D01':(-8.,.15),'I_DMG_X01_D02':(-8.,.35),'I_DMG_X02_D01':(-4.2,.15),
       'I_DMG_X02_D02':(-4.2,.35),'I_DMG_X03_D01':(4.2,.15),'I_DMG_X03_D02':(4.2,.35),
       'I_DMG_X04_D01':(8.,.15),'I_DMG_X04_D02':(8.,.35),
       'I_DMG_X05_D03':(-6.,.20),'I_DMG_X05_D04':(-6.,.30),
       'I_DMG_X06_D03':(6.,.20),'I_DMG_X06_D04':(6.,.30)}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def softmax(z):
 z=np.asarray(z,float);q=np.exp(z-np.max(z,axis=-1,keepdims=True));return q/q.sum(axis=-1,keepdims=True)

def get_k(root):
 import importlib.util
 path=root/'65AM_B9A11K_surface_damage_design.py'
 spec=importlib.util.spec_from_file_location('b9a11k_holdout',path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod

def posterior_experiment(K,obs,selected,noise_s,noise_u,rng,trials,noise_model,case_tag):
 # Normalized by the independent difference noise of two states (sqrt2 sigma).
 def f(tag):return K.features(obs,tag,'LE22_LE12_U123',noise_s,noise_u)[selected].reshape(-1)/math.sqrt(2.)
 templates=np.stack([f(t) for t in TRAIN]);nsite=len(selected)
 data=[]
 tag=case_tag
 true=f(tag)
 nfeat=len(true)
 iid=rng.standard_normal((trials,nsite,5))
 if noise_model=='common_mode_25pct':
  # 25% sensor-common variation by component; model fitting still assumes independent.
  z=rng.standard_normal((trials,1,5))
  noise=math.sqrt(.75)*iid+math.sqrt(.25)*z
 else:noise=iid
 y=true.reshape((1,nfeat))+noise.reshape(trials,nfeat)
 logits=-.5*np.sum((y[:,None,:]-templates[None,:,:])**2,axis=-1)
 P=softmax(logits)
 pred_indices=np.argmax(P,axis=1)
 pred_labels=[TRAIN[i] for i in pred_indices]
 pdmg=1-P[:,0]
 chosen_damaged=np.argmax(P[:,1:],axis=1)+1
 xest=np.array([TRUTH[TRAIN[i]][0] for i in chosen_damaged])
 sest=np.array([TRUTH[TRAIN[i]][1] for i in chosen_damaged])
 isdam=tag!='healthy'
 realx,real_s=(TRUTH[tag] if isdam else (float('nan'),float('nan')))
 noisefreeD=float(np.min(np.linalg.norm(templates-true[None,:],axis=1)))
 out={'case_id':tag,'sensor_count':nsite,'sigma_strain_ue':float(noise_s),
    'noise_model':noise_model,'trials':trials,'p_damage_mean':float(pdmg.mean()),
    'p_damage_p05':float(np.percentile(pdmg,5)),'p_damage_p95':float(np.percentile(pdmg,95)),
    'decision_damage_rate_at_0.5':float(np.mean(pdmg>0.5)),
    'nearest_template_distance_at_no_noise':noisefreeD,
    'most_selected_template':max(set(pred_labels),key=pred_labels.count),
    'truth_x_mm':None if not isdam else realx,
    'truth_severity':None if not isdam else real_s,
    'mean_loc_abs_error_mm_given_detection':None,
    'correct_damage_side_rate_given_detection':None,
    'mean_severity_abs_error_given_detection':None}
 if isdam:
  detected=pdmg>0.5
  out['detected_trials']=int(np.sum(detected))
  out['joint_detection_and_correct_side_rate']=float(np.mean(detected & (np.sign(xest)==np.sign(realx))))
  if np.any(detected):
   out['mean_loc_abs_error_mm_given_detection']=float(np.mean(abs(xest[detected]-realx)))
   out['correct_damage_side_rate_given_detection']=float(np.mean(np.sign(xest[detected])==np.sign(realx)))
   out['mean_severity_abs_error_given_detection']=float(np.mean(abs(sest[detected]-real_s)))
 return out

def run(root,tag,design_tag,manifest_tag,trials,sigma_u,confirmed,stage):
 if not confirmed:raise RuntimeError('Verify healthy/damaged mesh/loading/material match before confirming')
 K=get_k(root)
 export=root/'B9A11J_exports'
 cases=K.load_cases(root,export,TRAIN+HOLD)
 sites,groups,nodes=K.design_sites(cases['healthy'])
 obs=K.read_virtual_observables(cases,groups,nodes)
 lock=root/'results'/'65AM_B9A11L_balanced_design'/design_tag/'B9A11L_locked_sensor_design.csv'
 manifest=root/'results'/'65AM_B9A11L_validation_design'/manifest_tag/'B9A11L_new_damage_manifest.csv'
 if not lock.is_file() or not manifest.is_file():raise FileNotFoundError('Missing locked design or frozen validation manifest')
 d=pd.read_csv(lock);m=pd.read_csv(manifest)
 if set(m.case_id)!=set(HOLD):raise RuntimeError('Unexpected holdout manifest cases')
 for _,r in m.iterrows():
  ex=TRUTH[r.case_id]
  if not (abs(float(r.x_mm)-ex[0])<1e-8 and abs(float(r.damage_E2_fraction)-ex[1])<1e-9):
   raise RuntimeError('Validation manifest changed: '+str(r.case_id))
 for _,row in d.iterrows():
  old=sites.iloc[int(row.site_id)]
  if (int(row.displacement_node_label)!=int(old.displacement_node_label) or
      abs(float(row.center_x_mm)-float(old.center_x_mm))>1e-5 or
      row.surface!=old.surface):raise RuntimeError('Locked sensor site changed')
 print('L-C CHECK PASS:',len(cases),'matched FEM cases; 4 off-grid holdouts; frozen design unchanged.')
 if stage=='check':return
 out=root/'results'/'65AM_B9A11L_holdout'/tag
 if out.exists():raise RuntimeError('Refusing overwrite '+str(out))
 out.mkdir(parents=True)
 rows=[]
 for n,sub in d.groupby('sensor_count'):
  indices=sub.sort_values('sensor_rank')['site_id'].astype(int).tolist()
  if n!=len(indices) or len(set(indices))!=n:raise RuntimeError('Broken locked sensor count')
  if len(set(d.loc[(d.sensor_count==n),'displacement_node_label']))!=n:
   raise RuntimeError('Duplicated displacement sensor nodes')
  for sig in (10.,25.,50.,100.):
   for model in ('independent','common_mode_25pct'):
    for k in ('healthy',)+HOLD:
     # deterministic unique streams independent of Python hash randomization
     import zlib
     s=4917+zlib.crc32((str(n)+str(sig)+model+k).encode('ascii'))
     rng=np.random.default_rng(s)
     result=posterior_experiment(K,obs,indices,sig,sigma_u,rng,trials,model,k)
     rows.append(result)
 report=pd.DataFrame(rows)
 # Critical safety check: each of the five test states must appear for every
 # (sensor_count, strain_noise, noise_model) configuration.  A previous
 # version returned after the healthy case and silently mislabeled results.
 expected_cases={'healthy',*HOLD}
 if len(report)!=80 or set(report['case_id'])!=expected_cases:
  raise RuntimeError('HOLDOUT COVERAGE ERROR: expected five test states in 80 rows; got %s' % report['case_id'].value_counts().to_dict())
 coverage=report.groupby(['sensor_count','sigma_strain_ue','noise_model'])['case_id'].agg(lambda x:set(x))
 if not all(x==expected_cases for x in coverage):
  raise RuntimeError('HOLDOUT COVERAGE ERROR: a configuration is missing one or more damaged cases')
 if len(report[report.case_id!='healthy'])!=64:
  raise RuntimeError('HOLDOUT COVERAGE ERROR: expected 64 rows from new damaged FEM cases')
 report.to_csv(out/'B9A11L_holdout_template_results.csv',index=False)

 summary={'stage':'B9A11L-C','result_rows':len(rows),'new_Abaqus_cases':list(HOLD),
  'original_training_templates':list(TRAIN),'locked_sensor_design_sha256':sha(lock),
  'validation_manifest_sha256':sha(manifest),'trials_per_case':trials,'case_row_counts':report['case_id'].value_counts().to_dict(),
  'test_uses_data_not_seen_in_sensor_design':True,
  'algorithm':'nearest-discrete-FEM-template probability (Gaussian likelihood over 9 precomputed response hypotheses); NOT continuous Bayesian inversion',
  'limits':['Four off-grid cases only; constant healthy parameters and simplified damage law',
     'Posterior template probabilities are miscalibrated if true state is outside the discrete library',
     'VIRTUAL near-surface element-centroid strain, not experimental gauge readings',
     'Common-mode noise stress test intentionally exposes diagonal-likelihood misspecification',
     'No contact/surface accessibility audit, no material uncertainty, no trained damage-dependent forward operator']}
 (out/'B9A11L_C_summary.json').write_text(json.dumps(summary,indent=2))
 print('L-C DONE',out,'; holdout cases=',report['case_id'].value_counts().to_dict())
 print('Send B9A11L_C_summary.json and B9A11L_holdout_template_results.csv')

def selftest():
 a=softmax(np.array([[0.,0.],[100.,-100.]]))
 assert np.allclose(a.sum(axis=1),1.) and a[1,0]>0.999
 # Regression: every requested class (healthy and four unseen damaged
 # cases) must be scored independently rather than repeatedly scoring healthy.
 class MockK:
  @staticmethod
  def features(obs,tag,kind,noise_s,noise_u):
   f={name:i for i,name in enumerate(TRAIN+HOLD)}
   a=np.zeros((2,5));a[:,0]=f[tag]*.1
   return a
 m=MockK()
 for tag in ('healthy',)+HOLD:
  out=posterior_experiment(m,None,[0,1],25.,.0002,np.random.default_rng(42),20,'independent',tag)
  assert out['case_id']==tag
  if tag=='healthy':assert out['truth_x_mm'] is None
  else:assert out['truth_x_mm']==TRUTH[tag][0]
 print('B9A11L-C FIXED SELFTEST PASS | all four off-grid damaged states separately scored')

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',required=True,choices=('selftest','check','run'))
 p.add_argument('--project',default='.')
 p.add_argument('--tag',default='holdout01')
 p.add_argument('--design-tag',default='design01');p.add_argument('--manifest-tag',default='holdout01')
 p.add_argument('--trials',type=int,default=200)
 p.add_argument('--sigma-u-mm',type=float,default=.0002)
 p.add_argument('--healthy-material-confirmed',action='store_true')
 a=p.parse_args()
 if a.stage=='selftest':return selftest()
 if a.trials<20 or a.trials>10000 or a.sigma_u_mm<=0:raise ValueError('Invalid trial count or displacement noise')
 run(Path(a.project).expanduser().resolve(),a.tag,a.design_tag,a.manifest_tag,a.trials,a.sigma_u_mm,a.healthy_material_confirmed,a.stage)
if __name__=='__main__':main()
