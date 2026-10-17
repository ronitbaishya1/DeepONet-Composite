#!/usr/bin/env python3
"""A: audit actual 65AA V4 and zero-residual unified start, all three targets.

python3 65AH_A_audit.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ah=importlib.import_module('65AH_common')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.')
 p.add_argument('--members',type=int,default=1)
 p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
 p.add_argument('--check-only',action='store_true')
 a=p.parse_args();root=Path(a.project).expanduser().resolve()
 if not 1<=a.members<=5:raise ValueError('members must be 1 to 5')
 dev=ah.co.device_from_option(a.device);rows=[]
 for region in ah.SEGMENTS:
  for m in range(1,a.members+1):
   st=ah.load_source(root,region,m,dev)
   fit,tune=ah.split_in_train(st['ds'],region,m)
   if a.check_only:
    print('CHECK',region,m,'branch',int(st['t']['branch'].shape[1]),'fit',len(fit),'tune',len(tune),'device',dev)
    continue
   ids=st['ds']['splits']['validation']
   base=ah.baseline_numpy(st,ids,dev)
   net=ah.get_model(st,dev,'local').eval()
   init=ah.predict_numpy(net,st['t'],ids)
   equal=[float(np.max(abs(x-y))) for x,y in zip(base,init)]
   if max(equal)>1e-5:raise RuntimeError('Initial unified model DOES NOT reproduce original V4: '+str(equal))
   err=ah.boundary_error(init[0],st,ids,dev)
   if err>2e-5:raise RuntimeError('Hard interface not preserved; max %.9g mm'%err)
   if not ah.source_force_equal(net,st):raise RuntimeError('Force head changed before training')
   metrics=ah.metrics(base,ah.reference(st,ids))
   rows.append(dict(segment=region,member=m,train_cases=len(st['ds']['splits']['train']),
       internal_fit_cases=len(fit),internal_tune_cases=len(tune),
       original_validation_cases=len(ids),interface_max_abs_mm=err,
       source_sha=st['source_sha'],branch_dim=st['ck']['branch_dim'],
       **metrics))
   print('65AH-A',region,'U2 %.3f%% LE22 %.3f%% S22 %.3f%%'%(metrics['U2_pct'],metrics['LE22_pct'],metrics['S22_pct']),flush=True)
 if a.check_only:
  print('65AH-A check PASS');return
 out=ah.output_dir(root,'A_audit')
 pd.DataFrame(rows).to_csv(out/'65AH_A_source_metrics.csv',index=False)
 (out/'65AH_A_summary.json').write_text(json.dumps(dict(step='65AH-A',members=a.members,
  model='Trainable V4 U/LE/S + joint branch/trunk + local adaptive 8-node stencil',
  branch='E1,E2,G12 plus interface PCA coefficients',train_policy='original 64J TRAIN only with internal split',
  original_validation_status='reused historical development data; NOT independent',
  old_test_and_65W_untouched=True,source_force_head_frozen=True,zero_residual_initialization_verified=True,
  rows=rows),indent=2))
 print('65AH-A DONE',out)

if __name__=='__main__':main()
