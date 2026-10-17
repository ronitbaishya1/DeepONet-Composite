#!/usr/bin/env python3
"""Generate NEW in-range material cases for independent Abaqus simulations.

This only creates a manifest. It does NOT run Abaqus or invent FEM labels.
To validate a frozen surrogate, solve every case with the SAME full-model
geometry/mesh/contact/loading and export real U/LE/S plus interface data.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import importlib
aj=importlib.import_module('65AJ_common')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.');p.add_argument('--count',type=int,default=6)
 p.add_argument('--seed',type=int,default=65011010)
 a=p.parse_args();root=Path(a.project).resolve()
 if not 4<=a.count<=30:raise ValueError('4 to 30 independent cases recommended')
 old={r:aj.ai.ag.ac.aa.load_data(root,r) for r in aj.REGIONS}
 all_mat=np.concatenate([d['branch_inputs'][:,:3] for d in old.values()],axis=0)
 train_mat=np.concatenate([d['branch_inputs'][d['splits']['train'],:3] for d in old.values()],axis=0)
 low=train_mat.min(axis=0);high=train_mat.max(axis=0)
 if (high<=low).any():raise RuntimeError('No material variation in one input')
 rng=np.random.default_rng(a.seed)
 cases=[];attempts=0;seen=[]
 while len(cases)<a.count and attempts<15000:
  attempts+=1
  t=rng.uniform(.10,.90,size=3)
  mat=low+t*(high-low)
  nd=np.linalg.norm((all_mat-mat)/(high-low),axis=1).min()
  if nd<.035:continue
  if seen and min(np.linalg.norm((np.asarray(seen)-mat)/(high-low),axis=1))<.15:continue
  seen.append(mat)
  cases.append(dict(case_id=f'AJ_new_{len(cases)+1:02d}',E1_MPa=float(mat[0]),
                    E2_MPa=float(mat[1]),G12_MPa=float(mat[2]),
                    nearest_old_material_normalized=float(nd)))
 if len(cases)<a.count:raise RuntimeError('Could not produce required unique cases; decrease count')
 out=aj.new_folder(root,'D_new_material_manifest')
 with open(out/'65AJ_D_new_cases.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cases[0].keys());w.writeheader();w.writerows(cases)
 (out/'65AJ_D_summary.json').write_text(json.dumps(dict(step='65AJ-D',cases=cases,
   material_bounds_train_and_existing= {'min':low.tolist(),'max':high.tolist()},
   status='CASE PLAN ONLY; NO NEW ABAQUS DATA HAS BEEN GENERATED',
   requirements='Full 3-point-bending Abaqus solves with verified original mesh, load, contact, material orientation; export original region-specific nodal U and integration-point LE/S',
   old_test_and_65W_not_used=True),indent=2))
 print('Material plan only (NOT FEM simulations):',out/'65AJ_D_new_cases.csv')
if __name__=='__main__':main()
