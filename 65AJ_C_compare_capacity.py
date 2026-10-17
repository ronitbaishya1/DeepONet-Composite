#!/usr/bin/env python3
"""Capacity-matched development comparison; DOES NOT select new hyperparameters."""
import argparse,json,importlib
from pathlib import Path
import pandas as pd
aj=importlib.import_module('65AJ_common')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.');p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
 p.add_argument('--check-only',action='store_true');a=p.parse_args()
 root=Path(a.project).resolve();dev=aj.ai.co.device_from_option(a.device)
 for reg in aj.REGIONS:
  st=aj.ai.member_source(root,reg,1,dev,False)
  aj.load_control(root,st,dev)
 if a.check_only:print('65AJ-C CHECK PASS');return
 out=aj.new_folder(root,'C_matched_comparison')
 rows=[];dec=[]
 for reg in aj.REGIONS:
  st=aj.ai.member_source(root,reg,1,dev,True)
  ids=st['ds']['splits']['validation']
  dat=aj.ai.base_features(st,ids,dev)
  ls,ss=aj.ai.train_scales(st['ds'])
  preds={'65AG_uncoupled':(dat['L0'],dat['S0'])}
  models=aj.models_for_comparison(root,st,dev)
  counts={}
  for name,model in models.items():
   counts[name]=aj.parameter_count(model)
   preds[name]=aj.ai.predict(model,dat,ls,ss,dev)
  ms={}
  for name,(L,S) in preds.items():
   m=aj.ai.error_metrics(L,S,dat);ms[name]=m
   rows.append(dict(region=reg,model=name,parameter_count=counts.get(name,0),cases=len(ids),**m))
   print(reg,name,'LE22 %.3f%% S22 %.3f%%'%(m['LE22_pct'],m['S22_pct']))
  lg=ms['65AI_local_lgno'];ctrl=ms['65AJ_matched_mlp'];prior=ms['65AG_uncoupled']
  dec.append(dict(region=reg,matched_parameters=counts['65AJ_matched_mlp'],
     lgno_parameters=counts['65AI_local_lgno'],
     lgno_S22_pct=lg['S22_pct'],matched_mlp_S22_pct=ctrl['S22_pct'],
     lgno_LE22_pct=lg['LE22_pct'],matched_mlp_LE22_pct=ctrl['LE22_pct'],
     gain_lgno_over_matched_S22_pct=100*(1-lg['S22_pct']/max(ctrl['S22_pct'],1e-8)),
     both_improve_vs_65AG=bool(lg['S22_pct']<prior['S22_pct'] and ctrl['S22_pct']<prior['S22_pct']),
     local_better_by_5pct=bool(lg['S22_pct']<=.95*ctrl['S22_pct']),
     local_other_components_safe=bool(aj.ai.safety(lg,prior)),
     historical_only=True))
 pd.DataFrame(rows).to_csv(out/'65AJ_C_all_metrics.csv',index=False)
 pd.DataFrame(dec).to_csv(out/'65AJ_C_decisions.csv',index=False)
 (out/'65AJ_C_summary.json').write_text(json.dumps(dict(step='65AJ-C',arms=list(preds),
     historical_validation_repeated=True,new_training_only_original_64J_train=True,
     parameter_matched_1pct=True,not_independent_generalization=True,
     caveat='A parameter-matched single-seed ablation cannot establish causal superiority; repeat seeds on fresh cases.',
     decisions=dec),indent=2))
 print('65AJ-C DONE',out)
if __name__=='__main__':main()
