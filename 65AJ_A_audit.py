#!/usr/bin/env python3
"""Preflight and freeze the parameter-matched experiment specification."""
import argparse,json
from pathlib import Path
import importlib
aj=importlib.import_module('65AJ_common')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.');p.add_argument('--members',type=int,default=1)
 p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
 p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.members!=1:raise ValueError('This step is deliberately limited to member 01 in both regions.')
 root=Path(a.project).resolve();dev=aj.ai.co.device_from_option(a.device)
 rows=[]
 for region in aj.REGIONS:
  st=aj.ai.member_source(root,region,1,dev,False)
  local,matched=aj.inspect_control(st,root,region,1,dev)
  rows.append(dict(region=region,member=1,local_lgno_parameters=local,
                   matched_pointwise_parameters=matched,matched_hidden=aj.CONTROL_HIDDEN,
                   original_65AI_lgno_checkpoint=str(aj.ai.ck_path(root,'local_lgno',region,1)),
                   prior_65AG_sha=st['prior_sha'],source_65AA_sha=st['source_sha']))
  print(region,'LGNO',local,'matched MLP',matched)
 if a.check_only:
  print('65AJ-A CHECK PASS');return
 out=aj.new_folder(root,'A_audit')
 (out/'65AJ_A_summary.json').write_text(json.dumps(dict(step='65AJ-A',rows=rows,
    training='identical original 65AI fitting/safety/objective, 200 epochs',
    historical_validation_reused=True,old_test_prohibited=True,
    original_source_checkpoints_unchanged=True),indent=2))
 print('65AJ-A DONE',out)
if __name__=='__main__':main()
