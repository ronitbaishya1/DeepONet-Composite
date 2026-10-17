#!/usr/bin/env python3
"""65AG-A: inspect 65AF baseline and 64J train-only selection split.
Run: python3 65AG_A_audit.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
ag=importlib.import_module('65AG_common')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--members',type=int,default=1)
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args()
    if not 1<=a.members<=5:raise ValueError('members must be in 1..5')
    root=Path(a.project).expanduser().resolve()
    dev=ag.ac.aa.co.device_from_option(a.device)
    for reg in ag.REGIONS:
        for m in range(1,a.members+1):
            st=ag.attach_st(ag.member_sources(root,reg,m,dev,False),reg,m)
            tr,tune=ag.train_tune_split(st['ds'],reg,m)
            if len(tr)<5:raise RuntimeError('Too few fit examples')
    if a.check_only:
        print('65AG-A CHECK PASS: 65AF frozen checkpoints, 65AF selection policy and 64J split')
        return
    out=ag.safe_output(root,'A_baseline_audit')
    rows=[]
    for reg in ag.REGIONS:
        for m in range(1,a.members+1):
            st=ag.attach_st(ag.member_sources(root,reg,m,dev),reg,m)
            ds=st['ds'];fit,tune=ag.train_tune_split(ds,reg,m)
            for cohort,ids in [('fit_train',fit),('tune_train',tune)]:
                data=ag.make_dataset(st,ids,dev)
                C=data['C'];reference_gap=ag.component_percent(ag.f.C_LE(data['LEref'],C),data['Sref'],1)
                if reference_gap>2.0:raise RuntimeError('Constitutive approximation disagrees with FEM reference for '+reg)
                for name,(L,S) in ag.baselines_from_data(data).items():
                    mm=ag.report(L,S,data['LEref'],data['Sref'],C)
                    rows.append(dict(region=reg,member=m,cohort=cohort,model=name,
                                     cases=len(ids),reference_CLE_S22_error_pct=reference_gap,**mm))
                print('65AG-A',reg,cohort,'cases',len(ids),'reference-CLE S22 error %.3f%%'%reference_gap,flush=True)
    pd.DataFrame(rows).to_csv(out/'65AG_A_baselines.csv',index=False)
    info=dict(step='65AG-A',members=a.members,source='65AF pointwise and local LGNO, frozen 65AA heads',
              new_weights_fit='Original 64J train only; internal deterministic fit/tune',
              inherited_training_note='65AF pretrained on all 56 old train cases and selected checkpoints on previously reused old 12 validation cases',
              original_test_and_65W_not_used=True,
              formula='S ~= C @ material Hencky LE for the studied cases; approximation only',
              source_heads_and_interfaces_unchanged=True)
    (out/'65AG_A_summary.json').write_text(json.dumps(info,indent=2))
    print('65AG-A DONE:',out)

if __name__=='__main__':main()
