#!/usr/bin/env python3
"""65AA-A: Diagnose inaccurate displacement gradients.

Compare raw U, hard-interface-corrected U, and reference FEM U on ORIGINAL
64J development-validation cases. No training and no held-out test cases.

python3 65AA_A_gradient_audit.py --project . --check-only
python3 65AA_A_gradient_audit.py --project . --members 5
"""
import argparse,json,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import pandas as pd
import numpy as np
import torch

spec=spec_from_file_location('step65aa',Path(__file__).with_name('65AA_common.py'))
aa=module_from_spec(spec);sys.modules[spec.name]=aa;spec.loader.exec_module(aa)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=5)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('--members must be 1 to 5')
    root=Path(args.project).expanduser().resolve()
    device=aa.co.device_from_option(args.device)
    for seg in aa.SEGMENTS:
        ds=aa.load_data(root,seg)
        for member in range(1,args.members+1):
            f=aa.source_checkpoint(root,seg,member)
            if not f.is_file():raise FileNotFoundError(str(f))
    if args.check_only:
        print('65AA-A CHECK PASS: dataset and %d constitutive checkpoints ready'% (len(aa.SEGMENTS)*args.members))
        print('Device:',device,'; original test and 65W reference will not be accessed')
        return
    out=root/'results'/aa.ROOT_NAME/'A_baseline_gradient_audit'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY:\n'+str(out)+'\nthen rerun 65AA-A')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seg in aa.SEGMENTS:
        for member in range(1,args.members+1):
            ds,ck,model,ctx,path=aa.init_case(root,seg,member,device)
            indices=ds['splits']['validation']
            true=ds['U'][indices]
            le=ds['LE'][indices]
            raw=aa.predict_numpy(model,ctx,indices,correct=False)
            corrected=aa.predict_numpy(model,ctx,indices,correct=True)
            for label,pred in [('raw',raw),('hard_corrected',corrected)]:
                d=aa.errors(pred,true,ds,le)
                d.update(segment=seg,member=member,variant=label,validation_cases=len(indices),
                         source_sha256=aa.co.sha256(path))
                rows.append(d)
                print('%-11s member %d %-14s U2 %.3f%% grad %.3f%% logLE22 %.2f%%'%(
                    seg,member,label,d['U2_rel_l2_pct'],d['gradient_rel_l2_pct'],
                    d['logU_LE22_rel_l2_pct']),flush=True)
            del model
            if device.type=='mps':torch.mps.empty_cache()
    table=pd.DataFrame(rows)
    table.to_csv(out/'65AA_A_member_metrics.csv',index=False)
    numeric=[c for c in table.columns if c not in ('segment','member','variant','validation_cases','source_sha256')]
    means=table.groupby(['segment','variant'],as_index=False)[numeric].mean()
    means.to_csv(out/'65AA_A_region_means.csv',index=False)
    findings=[]
    for seg in aa.SEGMENTS:
        r=means[(means.segment==seg)&(means.variant=='raw')].iloc[0]
        c=means[(means.segment==seg)&(means.variant=='hard_corrected')].iloc[0]
        findings.append(dict(segment=seg,
            hard_correction_gradient_error_ratio=float(c.gradient_rel_l2_pct/max(r.gradient_rel_l2_pct,1e-12)),
            hard_correction_U2_error_ratio=float(c.U2_rel_l2_pct/max(r.U2_rel_l2_pct,1e-12)),
            hard_correction_worsens_gradient=bool(c.gradient_rel_l2_pct>r.gradient_rel_l2_pct)))
    report=dict(step='65AA-A',cases='Original 64J validation only',reference='Original Abaqus U1/U2/U3 arrays',
        source_models='65Z constitutive fine-tuned checkpoints; displacement head frozen since 65V',
        model_count=int(len(aa.SEGMENTS)*args.members),findings=findings,
        note='Gradient error uses element-center B matrices; logU uses material frame and audited engineering shear. Do not use original test/65W middle for optimization.')
    (out/'65AA_A_summary.json').write_text(json.dumps(report,indent=2))
    print('65AA-A DONE:',out)

if __name__=='__main__':main()
