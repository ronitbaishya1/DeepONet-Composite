#!/usr/bin/env python3
"""65AC-A: audit element stencil and source 65AA displacement gradients.
Run on Mac from project root:
 python3 65AC_A_stencil_audit.py --project . --members 1 --check-only
 python3 65AC_A_stencil_audit.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import torch
import importlib
ac=importlib.import_module('65AC_common')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('members must be 1 through 5')
    root=Path(args.project).expanduser().resolve()
    device=ac.aa.co.device_from_option(args.device)
    for seg in ac.SEGMENTS:
        for member in range(1,args.members+1):
            ac.prepare(root,seg,member,device,need_predictions=False)
    if args.check_only:
        print('65AC-A CHECK PASS: source checkpoints, data, element connectivity, C3D8R gradients')
        return
    out=root/'results'/ac.ROOT_NAME/'A_stencil_audit'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing output protected. Manually delete ONLY:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[];components=[]
    for seg in ac.SEGMENTS:
        for m in range(1,args.members+1):
            st=ac.prepare(root,seg,m,device)
            ds=st['ds'];ids=ds['splits']['validation']
            g=ac.split_predict(st,None,ids,device)
            gt=ac.gradient_from_u(ds['U'][ids],ds)
            d=ac.score(g,gt,ds['LE'][ids])
            d.update(segment=seg,member=m,validation_cases=len(ids),
                     elements=int(len(st['geometry']['conn'])),
                     max_affine_error=float(st['geometry']['affine_error']),
                     source_sha256=st['source_sha'])
            rows.append(d)
            for c in range(3):
                for ax in range(3):
                    name='dU%d_d%s'%(c+1,'XYZ'[ax])
                    components.append(dict(segment=seg,member=m,component=name,
                        rel_l2_pct=ac.rel2(g[...,c,ax],gt[...,c,ax]),
                        rmse=float(np.sqrt(np.mean((g[...,c,ax]-gt[...,c,ax])**2)))))
            print('65AC-A %-11s member %d: gradient %.3f%%, logLE22 %.2f%%'%(seg,m,d['gradient_rel_l2_pct'],d['logU_LE22_rel_l2_pct']),flush=True)
    pd.DataFrame(rows).to_csv(out/'65AC_A_baseline_metrics.csv',index=False)
    pd.DataFrame(components).to_csv(out/'65AC_A_component_diagnostics.csv',index=False)
    summary=dict(step='65AC-A',source='65AA gradient-model checkpoint',
        cohort='64J development validation only',num_members=args.members,
        validation_test_separation='Original 64J test and 65W FEM excluded',
        affine_consistency_verified=True,
        architecture='C3D8R local eight-node nonperiodic differential stencils',
        result_rows=rows,
        warning='Learned element-center gradient estimate may be non-integrable; source displacement is unchanged.')
    (out/'65AC_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('65AC-A DONE:',out)

if __name__=='__main__':main()
