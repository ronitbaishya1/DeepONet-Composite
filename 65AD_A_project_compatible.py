#!/usr/bin/env python3
"""65AD-A: Project LGNO gradients onto a C3D8R-compatible displacement field.

Preflight:
 python3 65AD_A_project_compatible.py --project . --check-only
Run:
 python3 65AD_A_project_compatible.py --project . --members 1
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ad=importlib.import_module('65AD_common')
ac=ad.ac


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--lambda-u',type=float,default=.05,help='Fixed projection regularization; do not tune on the previously inspected validation set')
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('members must be between 1 and 5')
    root=Path(args.project).expanduser().resolve();dev=ac.aa.co.device_from_option(args.device)
    for seg in ad.SEGMENTS:
        for member in range(1,args.members+1):
            st=ac.prepare(root,seg,member,dev,need_predictions=False)
            if not ac.checkpoint_path(root,'lgno',seg,member).is_file():
                raise FileNotFoundError('Missing Step65AC LGNO model for '+seg+' member '+str(member))
            ad.interface_nodes(root,st['ds'])
            ad.CompatibleProjector(st['ds'],ad.interface_nodes(root,st['ds']),args.lambda_u)
    if args.check_only:
        print('65AD-A CHECK PASS: original interfaces, C3D8R geometry, LGNO checkpoints, scipy')
        return
    out=ad.safe_output(root,'A_compatible_projection')
    all_rows=[]
    for seg in ad.SEGMENTS:
        for member in range(1,args.members+1):
            st=ac.prepare(root,seg,member,dev)
            ckfile=ac.checkpoint_path(root,'lgno',seg,member)
            lgno,meta=ac.load_adapter(ckfile,'lgno',seg,member,st['source_sha'],dev)
            ds=st['ds'];ids=ds['splits']['validation']
            boundary=ad.interface_nodes(root,ds)
            solver=ad.CompatibleProjector(ds,boundary,args.lambda_u)
            u0=np.asarray(st['pred'][[st['index_map'][int(i)] for i in ids]],float)
            base_gradient=ac.split_predict(st,None,ids,dev)
            target_gradient=ac.split_predict(st,lgno,ids,dev)
            if np.max(abs(base_gradient-ad.ac.gradient_from_u(u0,ds)))>2e-5:
                raise RuntimeError('Different baseline gradient paths; stop')
            projected_U=[];projected_G=[];notes=[]
            for k in range(len(ids)):
                u,g,note=solver.reconstruct(u0[k],target_gradient[k])
                projected_U.append(u);projected_G.append(g);notes.append(note)
            projected_U=np.stack(projected_U).astype(np.float32)
            projected_G=np.stack(projected_G).astype(np.float32)
            if np.max(np.abs(projected_U[:,boundary,:]-u0[:,boundary,:]))>1e-9:
                raise RuntimeError('Prescribed interface displacement changed!')
            cases={
                '65AA_source':(u0,base_gradient),
                '65AC_LGNO_gradient_only':(u0,target_gradient),
                '65AD_compatible_projection':(projected_U,projected_G)}
            for name,(u,g) in cases.items():
                metrics=ad.errors_from_values(u,g,ds,ids)
                metrics.update(segment=seg,member=member,model=name,cases=len(ids),lambda_u=args.lambda_u,
                               max_interface_change_mm=float(np.max(np.abs(u[:,boundary,:]-u0[:,boundary,:]))))
                all_rows.append(metrics)
                print(seg,member,name,'gradient %.3f%%'%metrics['gradient_rel_l2_pct'],
                      'LE22 %.2f%%'%metrics['LE22_rel_l2_pct'],
                      'U2 %.2f%%'%metrics['U2_rel_l2_pct'],flush=True)
            folder=out/seg/('member_%02d'%member);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'projection_validation.npz',
                indices=np.asarray(ids,dtype=np.int64),
                baseline_U=u0.astype(np.float32),compatible_U=projected_U,
                baseline_G=base_gradient.astype(np.float32),lgno_G=target_gradient.astype(np.float32),
                compatible_G=projected_G,lambda_u=np.array(args.lambda_u),
                source_sha256=np.array(st['source_sha']),
                lgno_checkpoint_sha256=np.array(ac.aa.co.sha256(ckfile)),
                fixed_boundary=np.asarray(boundary,dtype=np.int64))
            print('  projection residual / LGNO correction RMS: %.5g / %.5g'%(
                np.mean([r['unintegrable_gradient_residual_rms'] for r in notes]),
                np.mean([r['LGNO_correction_abs_rms'] for r in notes])),flush=True)
    df=pd.DataFrame(all_rows)
    df.to_csv(out/'65AD_A_projection_metrics.csv',index=False)
    summary={'step':'65AD-A','cases':'Original 64J validation only; no old test or 65W full FEM',
       'lambda_u':args.lambda_u,'models_per_region':args.members,'interfaces_preserved_exactly':True,
       'C3D8R_compatible_gradients':True,
       'projection_definition':'min ||D(Unew)-G_LGNO||^2 + lambda * mean_diag(Df.T Df) ||Unew-U0||^2, with Unew=U0 at interface nodes',
       'selected_on_validation':False,
       'warning':'LGNO gradient target is not necessarily integrable. Projected gradients are integrable but may lose LGNO accuracy. Stress/strain must be checked separately.'}
    (out/'65AD_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('65AD-A DONE:',out)

if __name__=='__main__':main()
