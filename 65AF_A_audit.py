#!/usr/bin/env python3
"""65AF-A: check strain and stress baselines on original 64J TRAIN only.
python3 65AF_A_audit.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
f=importlib.import_module('65AF_common')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5:raise ValueError('Expected 1 to 5 members')
    dev=f.ac.aa.co.device_from_option(a.device)
    for reg in f.REGIONS:
        for member in range(1,a.members+1):f.cases(root,reg,member,dev,need_predictions=False)
    if a.check_only:
        print('65AF-A CHECK PASS: source models, LGNO, 64J meshes and material data found')
        return
    out=f.safe_output(root,'A_baselines')
    rows=[];checks=[]
    for reg in f.REGIONS:
        for m in range(1,a.members+1):
            st=f.cases(root,reg,m,dev)
            ds=st['ds'];idx=ds['splits']['train']
            _,_,headL,headS,gl=f.build_features(st,idx,dev)
            lgnoL=f.ac.logstrain_from_gradient(gl)
            C=f.stiffness(ds,idx)
            ref=ds['LE'][idx];Sref=ds['S'][idx]
            models={'original_LE_head':headL,'65AC_LGNO_strain':lgnoL,'Abaqus_reference_LE':ref}
            sref=f.C_LE(ref,C)
            sanity=f.relative(sref[...,1],Sref[...,1])
            if sanity>2:raise RuntimeError('Abaqus LE-to-S approximation failed for '+reg+' (%.2f%%)'%sanity)
            for name,L in models.items():
                for comp,label in enumerate(f.LABELS):
                    rows.append(dict(region=reg,member=m,model=name,component='LE'+label,
                        rel_L2_pct=f.relative(L[...,comp],ref[...,comp]),train_cases=len(idx)))
            for label,S in [('original_S_head',headS),('C_original_LE_head',f.C_LE(headL,C)),
                            ('C_LGNO_LE',f.C_LE(lgnoL,C)),('C_reference_LE',sref)]:
                for comp,name in enumerate(f.LABELS):
                    rows.append(dict(region=reg,member=m,model=label,component='S'+name,
                        rel_L2_pct=f.relative(S[...,comp],Sref[...,comp]),train_cases=len(idx)))
            checks.append(dict(region=reg,member=m,train_cases=len(idx),
                reference_C_LE_S22_pct=sanity,
                original_S_head_S22_pct=f.relative(headS[...,1],Sref[...,1]),
                original_LE_head_LE22_pct=f.relative(headL[...,1],ref[...,1]),
                LGNO_gradient_LE22_pct=f.relative(lgnoL[...,1],ref[...,1])))
            print('65AF-A',reg,'head LE22 %.2f%%, head S22 %.2f%%, reference constitutive gap %.3f%%'%(
                checks[-1]['original_LE_head_LE22_pct'],checks[-1]['original_S_head_S22_pct'],sanity),flush=True)
    pd.DataFrame(rows).to_csv(out/'65AF_A_baseline_components.csv',index=False)
    (out/'65AF_A_summary.json').write_text(json.dumps(dict(step='65AF-A',cohort='Original 64J TRAIN ONLY',
       source='Frozen 65AA strain/stress and 65AC local gradients',members=a.members,checks=checks,
       warning='C @ Hencky LE approximates Cauchy stress, not an exact finite-strain constitutive update.'),indent=2))
    print('65AF-A DONE',out)
if __name__=='__main__':main()
