#!/usr/bin/env python3
"""Evaluate FROZEN models on genuinely new Abaqus-labelled, geometry-matched cases.

No fitting, model selection, recalibration or retraining occurs. This is an
offline ORACLE-INTERFACE test: new PCA inputs extracted from actual full-FEM
interface displacement; it is not a coupled FE-NO solution or equilibrium test.
"""
import argparse,json,csv,hashlib,importlib
from pathlib import Path
import numpy as np,pandas as pd
aj=importlib.import_module('65AJ_common')

def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.');p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
 p.add_argument('--check-only',action='store_true');a=p.parse_args()
 root=Path(a.project).resolve();dev=aj.ai.co.device_from_option(a.device)
 folder=root/'results'/aj.ROOT/'E_external_npz'
 manifest=root/'results'/aj.ROOT/'D_new_material_manifest'/'65AJ_D_new_cases.csv'
 if not manifest.is_file():raise FileNotFoundError(str(manifest))
 with open(manifest) as f:cases=list(csv.DictReader(f))
 names=np.asarray([r['case_id'] for r in cases],dtype=str)
 mats=np.asarray([[float(r[c]) for c in ('E1_MPa','E2_MPa','G12_MPa')] for r in cases],np.float32)
 loaded={};sha={}
 for reg in aj.REGIONS:
  f=folder/(reg+'.npz')
  if not f.is_file():raise FileNotFoundError('Missing new Abaqus reference: '+str(f))
  sha[reg]=filehash(f)
  with np.load(f,allow_pickle=False) as z:loaded[reg]={k:z[k] for k in z.files}
  ext=loaded[reg]
  for key in ('case_ids','materials','U','LE','S','coordinates','ip_coordinates','source_odb_paths'):
   if key not in ext:raise RuntimeError('Missing required external array '+reg+': '+key)
  if not np.array_equal(ext['case_ids'].astype(str),names):raise RuntimeError('New case names mismatch manifest')
  if not np.allclose(ext['materials'],mats,rtol=1e-5,atol=1e-3):raise RuntimeError('New case materials mismatch manifest')
  if len(ext['source_odb_paths'])!=len(names):raise RuntimeError('Missing source ODB provenance')
  st=aj.ai.member_source(root,reg,1,dev,False)
  aj.load_control(root,st,dev)
 if a.check_only:
  print('65AJ-F CHECK PASS: file shapes/provenance fields and checkpoints exist (FE physics not yet audited)');return
 out=aj.new_folder(root,'F_external_evaluation')
 rows=[];quality=[]
 for reg in aj.REGIONS:
  st=aj.ai.member_source(root,reg,1,dev,True)
  ext=loaded[reg]
  ids,branch,pca_error,distances=aj.attach_external(root,st,ext,dev)
  # Physically mandatory verification: new FEM U must give compatible LE at
  # existing C3D8R integration points (same orientation + engineering shear).
  L_from_truth=aj.ai.ag.ac.logstrain_from_gradient(
      aj.ai.ag.ac.aa.gradient_numpy(ext['U'],st['ds']))
  LEerr=aj.ai.ag.f.relative(L_from_truth,ext['LE'])
  LE22err=aj.ai.ag.f.relative(L_from_truth[...,1],ext['LE'][...,1])
  refC=aj.ai.ag.f.stiffness(st['ds'],ids)
  source_constitutive=aj.ai.ag.f.C_LE(ext['LE'],refC)
  S22Cerr=aj.ai.ag.f.relative(source_constitutive[...,1],ext['S'][...,1])
  quality.append(dict(region=reg,cases=len(ids),truth_U_to_LE_vector_pct=LEerr,
                      truth_U_to_LE22_pct=LE22err,
                      truth_C_LE_to_S22_pct=S22Cerr,
                      min_new_material_distance=float(min(distances)),**pca_error))
  if LE22err>2.0 or LEerr>2.0:
   raise RuntimeError('New Abaqus reference strain inconsistent with C3D8R FE U (%s): %.3f%% LE22, %.3f%% all; check sorting/engineering shear/orientation'%(reg,LE22err,LEerr))
  # Constitutive reference is approximate; report deviation, do not silently
  # use an inaccurate C@LE surrogate to reject real FEM data.
  dat=aj.ai.base_features(st,ids,dev)
  lscale,sscale=aj.ai.train_scales(st['ds'])
  outputs={'65AG_uncoupled':(dat['L0'],dat['S0'])}
  models=aj.models_for_comparison(root,st,dev)
  for name,mdl in models.items():
   outputs[name]=aj.ai.predict(mdl,dat,lscale,sscale,dev)
  for name,(L,S) in outputs.items():
   vals=aj.ai.error_metrics(L,S,dat)
   rows.append(dict(region=reg,model=name,external_cases=len(ids),
       external_validation_only=True,oracle_full_FE_interface_PCA=True,
       source_package_SHA256=sha[reg],**vals))
   print('65AJ-F',reg,name,'S22 %.3f%% LE22 %.3f%%'%(vals['S22_pct'],vals['LE22_pct']),flush=True)
 pd.DataFrame(rows).to_csv(out/'65AJ_F_external_metrics.csv',index=False)
 pd.DataFrame(quality).to_csv(out/'65AJ_F_reference_integrity.csv',index=False)
 (out/'65AJ_F_summary.json').write_text(json.dumps(dict(step='65AJ-F',
   new_case_count=len(names),case_ids=names.tolist(),package_sha256=sha,
   not_training_or_selecting_parameters=True,
   conditional_generalization_with_full_FEM_oracle_interface_coefficients=True,
   not_full_seven_region_coupled_equilibrium_test=True,
   source_ODB_paths={reg:loaded[reg]['source_odb_paths'].astype(str).tolist() for reg in aj.REGIONS},
   caution='Provenance information comes from external exports; original Abaqus ODB files must be independently inspected.',
   next='Only AFTER report, consider independently verifying full coupling, reaction and traction equilibrium'),indent=2))
 print('65AJ-F DONE:',out)
if __name__=='__main__':main()
