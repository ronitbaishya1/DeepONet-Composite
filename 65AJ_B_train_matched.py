#!/usr/bin/env python3
"""Train capacity-matched (50,310 params) pointwise 65AI control.

Reuses the EXACT 65AI training loop, seed, losses, fit/tune split, optimizer,
patience, and epoch budget. Only changes pointwise width 96 -> 166 and the
output folder. Does not retrain or replace the original 65AI adapters.
"""
import argparse,json,importlib
from pathlib import Path
import pandas as pd
aj=importlib.import_module('65AJ_common')
old=importlib.import_module('65AI_B_train_local')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.');p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
 p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=4)
 p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.epochs!=200:raise ValueError('Use exactly 200 epochs for fair comparison with existing LGNO')
 if a.batch_size!=4:raise ValueError('Use original 65AI batch size 4')
 a.root=Path(a.project).resolve();a.arm='pointwise';a.hidden=aj.CONTROL_HIDDEN
 a.lr=4e-4;a.tune_cases=12;a.stress_weight=1.0;a.anchor_weight=.04
 a.patience=40;a.cap=.8;a.members=1
 dev=aj.ai.co.device_from_option(a.device)
 source_summary=a.root/'results'/aj.ai.ROOT/'A_audit'/'65AI_A_summary.json'
 if not source_summary.is_file():raise FileNotFoundError('65AI-A summary required: '+str(source_summary))
 for reg in aj.REGIONS:
  st=aj.ai.member_source(a.root,reg,1,dev,False)
  aj.inspect_control(st,a.root,reg,1,dev)
 if a.check_only:print('65AJ-B CHECK PASS. No models trained.');return
 out=aj.new_folder(a.root,'B_matched_mlp')
 # This module-local patch changes ONLY ai.ck_path's output root. The source
 # 65AI frozen baseline loading and 65AG sources remain at original paths.
 original_root=aj.ai.ROOT
 original_ck_path=aj.ai.ck_path
 rows=[]
 try:
  aj.ai.ROOT=aj.ROOT
  aj.ai.ck_path=lambda root,arm,region,member: (aj.control_path(root,region,member) if arm=='pointwise' else original_ck_path(root,arm,region,member))
  for reg in aj.REGIONS:
   st=aj.ai.member_source(a.root,reg,1,dev,True)
   record=old.train_one(st,reg,1,dev,a)
   record['matched_to_65AI_LGNO_parameters']=50316
   rows.append(record)
 finally:
  aj.ai.ROOT=original_root
  aj.ai.ck_path=original_ck_path
 pd.DataFrame(rows).to_csv(out/'65AJ_B_training_summary.csv',index=False)
 (out/'65AJ_B_manifest.json').write_text(json.dumps(dict(step='65AJ-B',width=166,
   arm='pointwise',epochs=200,batch_size=4,losses_and_seed='exact original 65AI',
   old_validation_and_test_excluded_from_new_weights=True,
   upstream_pretraining_had_seen_original_validation=True,
   original_65AI_LGNO_not_retrained=True,rows=rows),indent=2))
 print('65AJ-B DONE',out)
if __name__=='__main__':main()
