#!/usr/bin/env python3
"""65AE-A: identify which material strain errors drive S22.

Run: python3 65AE_A_S22_diagnosis.py --project . --members 1
TRAIN cases only; no tuning or old test data. Uses 65AA + 65AC + 65AD.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ae=importlib.import_module('65AE_common')


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='.');p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5:raise ValueError('--members must be 1 to 5')
    dev=ae.ac.aa.co.device_from_option(a.device)
    for reg in ae.REGIONS:
        for m in range(1,a.members+1):ae.prepare(root,reg,m,dev)
    if a.check_only:print('65AE-A PASS: checkpoints, meshes, interfaces and sparse projection ready');return
    out=ae.output(root,'A_S22_diagnosis');rows=[];recaps=[]
    for reg in ae.REGIONS:
        for m in range(1,a.members+1):
            d=ae.prepare(root,reg,m,dev);ds=d['ds'];ids=ds['splits']['train']
            U0,G,heads,C=ae.batch_inputs(d,ids,dev)
            UP,GP,_=ae.project(d,U0,G)
            L,stress=ae.stress_from_grad(GP,C)
            LE_ref=np.asarray(ds['LE'][ids],float);S_ref=np.asarray(ds['S'][ids],float)
            # All three diagonal Hencky terms contribute to orthotropic S22.
            for j,name in enumerate(('LE11','LE22','LE33')):
                contrib=C[:,None,1,j]*(L[...,j]-LE_ref[...,j])
                rows.append(dict(region=reg,member=m,strain_term=name,
                     signed_mean_S22_error_contribution_MPa=float(np.mean(contrib)),
                     contribution_RMSE_MPa=float(np.sqrt(np.mean(contrib**2))),
                     contribution_abs_mean_MPa=float(np.mean(np.abs(contrib))),
                     strain_rel_L2_pct=ae.relative(L[...,j],LE_ref[...,j]),
                     train_cases=len(ids)))
            ref_const=np.einsum('bec,bdc->bed',LE_ref,C,optimize=True)
            recaps.append(dict(region=reg,member=m,train_cases=len(ids),
                 S22_head_error_pct=ae.relative(heads[...,1],S_ref[...,1]),
                 S22_projected_error_pct=ae.relative(stress[...,1],S_ref[...,1]),
                 S22_reference_constitutive_error_pct=ae.relative(ref_const[...,1],S_ref[...,1]),
                 gradient_projected_error_pct=ae.relative(GP,ae.ac.gradient_from_u(ds['U'][ids],ds))))
            print('65AE-A',reg,'head S22 %.2f%% projected S22 %.2f%%'%(
                  recaps[-1]['S22_head_error_pct'],recaps[-1]['S22_projected_error_pct']),flush=True)
    pd.DataFrame(rows).to_csv(out/'65AE_A_S22_contributions.csv',index=False)
    pd.DataFrame(recaps).to_csv(out/'65AE_A_training_diagnosis.csv',index=False)
    (out/'65AE_A_summary.json').write_text(json.dumps(dict(step='65AE-A',
        cohort='Original 64J TRAIN only',diagnostics=recaps,
        note='Contributions to S22 error from C21*LE11 + C22*LE22 + C23*LE33; nonlinearity and stress law approximations noted.'),indent=2))
    print('65AE-A DONE:',out)
if __name__=='__main__':main()
