#!/usr/bin/env python3
"""65AC-C: compare original FE derivative, pointwise MLP and LGNO stencil.

All comparisons on original 64J development validation. NEVER report this as
fresh independent test data; do not silently replace the converged 65V solver.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ac=importlib.import_module('65AC_common')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('members must be 1-5')
    root=Path(args.project).expanduser().resolve();dev=ac.aa.co.device_from_option(args.device)
    for seg in ac.SEGMENTS:
        for m in range(1,args.members+1):
            ac.prepare(root,seg,m,dev,need_predictions=False)
            for arm in ac.ARMS:
                f=ac.checkpoint_path(root,arm,seg,m)
                if not f.is_file():raise FileNotFoundError('Missing 65AC-B candidate: '+str(f))
    if args.check_only:
        print('65AC-C CHECK PASS: both trained arm checkpoints + frozen source models exist')
        return
    out=root/'results'/ac.ROOT_NAME/'C_validation_comparison'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing output protected. Manually delete ONLY:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[];derivatives=[]
    for seg in ac.SEGMENTS:
        for m in range(1,args.members+1):
            st=ac.prepare(root,seg,m,dev)
            val=st['ds']['splits']['validation'];truth=ac.gradient_from_u(st['ds']['U'][val],st['ds'])
            preds={'65AA_source':ac.split_predict(st,None,val,dev,args.batch_size)}
            for arm in ac.ARMS:
                path=ac.checkpoint_path(root,arm,seg,m)
                mod,ck=ac.load_adapter(path,arm,seg,m,st['source_sha'],dev)
                preds[arm]=ac.split_predict(st,mod,val,dev,args.batch_size)
            for label,g in preds.items():
                d=ac.score(g,truth,st['ds']['LE'][val])
                d.update(segment=seg,member=m,model=label,validation_cases=len(val),
                         source_sha256=st['source_sha'])
                rows.append(d)
                for c in range(3):
                    for j in range(3):
                        derivatives.append(dict(segment=seg,member=m,model=label,
                            derivative='dU%d_d%s'%(c+1,'XYZ'[j]),
                            rel_l2_pct=ac.rel2(g[...,c,j],truth[...,c,j]),
                            rmse=float(np.sqrt(np.mean((g[...,c,j]-truth[...,c,j])**2)))))
                print('65AC-C %-11s member %d %-12s gradient %.3f%% logLE22 %.2f%%'%(
                      seg,m,label,d['gradient_rel_l2_pct'],d['logU_LE22_rel_l2_pct']),flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(out/'65AC_C_all_member_metrics.csv',index=False)
    pd.DataFrame(derivatives).to_csv(out/'65AC_C_derivative_metrics.csv',index=False)
    metrics=['gradient_rel_l2_pct','gradient_rmse','logU_LE11_rel_l2_pct',
             'logU_LE22_rel_l2_pct','logU_LE12_rel_l2_pct']
    aggregate=table.groupby(['segment','model'],as_index=False)[metrics].mean()
    aggregate.to_csv(out/'65AC_C_comparison.csv',index=False)
    decisions=[]
    for seg in ac.SEGMENTS:
        s=aggregate[aggregate.segment==seg].set_index('model')
        baseline=s.loc['65AA_source'];control=s.loc['pointwise'];lg=s.loc['lgno']
        g0=float(baseline.gradient_rel_l2_pct);gc=float(control.gradient_rel_l2_pct);gl=float(lg.gradient_rel_l2_pct)
        le0=float(baseline.logU_LE22_rel_l2_pct);lel=float(lg.logU_LE22_rel_l2_pct)
        passed=bool(gl<=.95*g0 and gl<=.95*gc and lel<=.95*le0 and
                    float(lg.logU_LE11_rel_l2_pct)<=1.05*float(baseline.logU_LE11_rel_l2_pct))
        decisions.append(dict(segment=seg,lgno_gradient_gain_vs_source_pct=100*(1-gl/max(g0,1e-12)),
                  lgno_gradient_gain_vs_pointwise_pct=100*(1-gl/max(gc,1e-12)),
                  lgno_logLE22_gain_vs_source_pct=100*(1-lel/max(le0,1e-12)),
                  preliminary_gate_passed=passed))
    report=dict(step='65AC-C',architecture=ac.ARCH,
        source_reference='baiming-zhang/LGNO 2D Navier-Stokes local stencil MLP -> kernel contraction',
        cohort='Original 64J validation, not an independent test',members=args.members,
        decisions=decisions,all_preliminary_gates_passed=all(x['preliminary_gate_passed'] for x in decisions),
        source_displacement_unchanged=True,source_interface_displacements_unchanged=True,
        source_force_head_unchanged=True,
        limitation='Gradient estimate is not necessarily the spatial derivative of any compatible displacement field.',
        next='If better, test on fresh Abaqus cases and design compatible U/gradient architecture before 65V coupling.')
    (out/'65AC_C_summary.json').write_text(json.dumps(report,indent=2))
    print('65AC-C DONE:',out)

if __name__=='__main__':main()
