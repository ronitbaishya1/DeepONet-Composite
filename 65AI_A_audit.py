#!/usr/bin/env python3
"""65AI-A: audit physical inputs, source models, frozen U and local neighborhoods.

Commands from Mac project root:
 python3 65AI_A_audit.py --project . --members 1 --check-only
 python3 65AI_A_audit.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
ai=importlib.import_module('65AI_common')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.members<=5 or not 1<=a.batch_size<=16:raise ValueError('Invalid member/batch')
    root=Path(a.project).expanduser().resolve()
    device=ai.co.device_from_option(a.device)
    for region in ai.REGIONS:
        for member in range(1,a.members+1):
            st=ai.member_source(root,region,member,device,False)
            ds=st['ds']
            original=set(map(int,ds['splits']['train']))
            val=set(map(int,ds['splits']['validation']))
            tst=set(map(int,ds['splits']['test']))
            if original&val or original&tst or val&tst:raise RuntimeError('Split overlap')
            if st['ck']['branch_dim']<=3:raise RuntimeError('Missing boundary coefficient inputs')
            if len(st['geometry']['conn'])!=ds['LE'].shape[1]:
                raise RuntimeError('Incorrect mesh/integration-point shape')
    if a.check_only:
        print('65AI-A CHECK PASS: prior models, geometry, dataset splits and provenance');return
    out=ai.no_overwrite(root,'A_audit')
    rows=[]
    for region in ai.REGIONS:
        for member in range(1,a.members+1):
            st=ai.member_source(root,region,member,device)
            ids=np.asarray(st['ds']['splits']['train'],np.int64)
            fit,tune=ai.internal_split(st['ds'],region,member)
            data=ai.base_features(st,ids,device,a.batch_size)
            m=ai.error_metrics(data['L0'],data['S0'],data)
            if data['features'].shape[-1] != ai.FEATURES_BASE + 12 + int(st['ck']['branch_dim']):
                raise RuntimeError('Unexpected displacement+material+branch feature count')
            row=dict(region=region,member=member,train_cases=len(ids),fit_cases=len(fit),
                     internal_tune_cases=len(tune),nodes=data['U0'].shape[1],
                     elements=data['Lref'].shape[1],branch_dim=st['ck']['branch_dim'],
                     input_features=data['features'].shape[-1],
                     frozen_65AG_uncoupled_sha256=st['prior_sha'],
                     frozen_65AA_sha256=st['source_sha'],
                     upstream_validation_inherited=True,**m)
            rows.append(row)
            print('65AI-A %s member %d: frozen U2 %.3f%%; 65AG S22 %.3f%%, LE22 %.3f%%'%(
                region,member,m['U2_pct'],m['S22_pct'],m['LE22_pct']),flush=True)
    pd.DataFrame(rows).to_csv(out/'65AI_A_source_metrics.csv',index=False)
    (out/'65AI_A_summary.json').write_text(json.dumps(dict(step='65AI-A',members=a.members,
        method='Predicted displacement-conditioned local joint strain/stress refinement',
        train_only=True,original_test_and_65W_not_used=True,
        upstream_65AG_and_65AF_pretrained_using_historical_validation=True,
        source_displacement_and_force_frozen=True,
        reference_fields_never_used_as_network_inputs=True,
        warnings=['All 12 historical validation cases are reused development diagnostics, not independent tests',
                  'Predicted strain is not forced to match frozen U gradients or exact Abaqus constitutive update'],
        rows=rows),indent=2))
    print('65AI-A DONE',out)

if __name__=='__main__':main()
