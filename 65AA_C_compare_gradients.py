#!/usr/bin/env python3
"""65AA-C: Decide if gradient-aware training beats ordinary extra U training.

Compare original 65Z, 65AA-B displacement-only control, and gradient arm
on original 64J VALIDATION cases only. This is NOT a fresh test set.

python3 65AA_C_compare_gradients.py --project . --members 1 --check-only
python3 65AA_C_compare_gradients.py --project . --members 1
"""
import argparse,json,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import pandas as pd
import torch

spec=spec_from_file_location('step65aa',Path(__file__).with_name('65AA_common.py'))
aa=module_from_spec(spec);sys.modules[spec.name]=aa;spec.loader.exec_module(aa)


def candidate_path(root,arm,seg,m):
    return (root/'results'/aa.ROOT_NAME/('B_'+arm)/seg/('member_%02d'%m)/'best_v4_7region.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('Expected 1..5 members')
    root=Path(args.project).expanduser().resolve()
    device=aa.co.device_from_option(args.device)
    for seg in aa.SEGMENTS:
        aa.load_data(root,seg)
        for m in range(1,args.members+1):
            files=[aa.source_checkpoint(root,seg,m)]+[candidate_path(root,arm,seg,m) for arm in ('displacement','gradient')]
            for f in files:
                if not f.is_file():raise FileNotFoundError(str(f))
    if args.check_only:
        print('65AA-C CHECK PASS: 65Z source + displacement control + gradient candidates present')
        return
    out=root/'results'/aa.ROOT_NAME/'C_validation_comparison'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY:\n'+str(out)+'\nthen rerun 65AA-C')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seg in aa.SEGMENTS:
        for m in range(1,args.members+1):
            ds,ck,source_model,ctx,source_path=aa.init_case(root,seg,m,device)
            val=ds['splits']['validation'];utrue=ds['U'][val];LEtrue=ds['LE'][val]
            baseline_ck=ck
            models={'65Z_baseline':(source_model,ck,source_path)}
            for arm in ('displacement','gradient'):
                path=candidate_path(root,arm,seg,m)
                model_ck=aa.co.load_checkpoint(path)
                if not aa.frozen_heads_equal(baseline_ck,model_ck):
                    raise RuntimeError('Stress/strain/force parameters changed unexpectedly: '+str(path))
                info=model_ck.get('65AA_gradient_experiment',{})
                if info.get('arm')!=arm or info.get('initialization_sha256')!=aa.co.sha256(source_path):
                    raise RuntimeError('Candidate checkpoint not linked to the original baseline')
                models[arm]=(aa.co.make_model(model_ck,device),model_ck,path)
            for label,(model,current,path) in models.items():
                # Same source normalization and boundary correction for all arms.
                pred=aa.predict_numpy(model,ctx,val,correct=True)
                row=aa.errors(pred,utrue,ds,LEtrue)
                row.update(segment=seg,member=m,model=label,
                           validation_cases=len(val),checkpoint_sha256=aa.co.sha256(path))
                rows.append(row)
                print('%-11s member %d %-13s U2 %.3f%% grad %.2f%% logLE22 %.1f%%'%(
                     seg,m,label,row['U2_rel_l2_pct'],row['gradient_rel_l2_pct'],
                     row['logU_LE22_rel_l2_pct']),flush=True)
            del models,source_model
            if device.type=='mps':torch.mps.empty_cache()
    table=pd.DataFrame(rows)
    table.to_csv(out/'65AA_C_member_metrics.csv',index=False)
    metrics=['U_vector_rel_l2_pct','U2_rel_l2_pct','U2_rmse_mm',
             'gradient_rel_l2_pct','gradient_rmse',
             'logU_LE11_rel_l2_pct','logU_LE22_rel_l2_pct','logU_LE12_rel_l2_pct']
    aggregate=table.groupby(['segment','model'],as_index=False)[metrics].mean()
    aggregate.to_csv(out/'65AA_C_aggregate_metrics.csv',index=False)
    decisions=[]
    for seg in aa.SEGMENTS:
        d=aggregate[aggregate.segment==seg].set_index('model')
        base=d.loc['65Z_baseline'];control=d.loc['displacement'];grad=d.loc['gradient']
        ratio=lambda x,y:float(x/max(y,1e-12))
        pass_gradient=(grad['gradient_rel_l2_pct']<=.95*base['gradient_rel_l2_pct'] and
                       grad['gradient_rel_l2_pct']<=.95*control['gradient_rel_l2_pct'])
        pass_displacement=(grad['U2_rel_l2_pct']<=1.05*base['U2_rel_l2_pct']+.02 and
                           grad['U_vector_rel_l2_pct']<=1.05*base['U_vector_rel_l2_pct']+.02)
        pass_log22=(grad['logU_LE22_rel_l2_pct']<=.95*base['logU_LE22_rel_l2_pct'])
        decisions.append(dict(segment=seg,
           gradient_gain_vs_65Z_pct=100*(1-ratio(grad['gradient_rel_l2_pct'],base['gradient_rel_l2_pct'])),
           gradient_gain_vs_control_pct=100*(1-ratio(grad['gradient_rel_l2_pct'],control['gradient_rel_l2_pct'])),
           better_gradient_than_baseline_and_control_5pct=bool(pass_gradient),
           nodal_displacement_safe=bool(pass_displacement),
           log_strain_LE22_improved_5pct=bool(pass_log22),
           preliminary_gate_passed=bool(pass_gradient and pass_displacement and pass_log22)))
    report=dict(step='65AA-C',cohort='Original 64J development-validation only',
        source_models='65Z constitutive checkpoints',members=args.members,
        learned_stress_strain_and_force_heads_unchanged=True,
        decisions=decisions,all_preliminary_gates_passed=bool(all(x['preliminary_gate_passed'] for x in decisions)),
        conclusion='Validation diagnostic only. Requires fresh independent Abaqus material case and new interface verification before any deployment.',
        note='Improved predicted gradient does not automatically improve frozen LE/S head outputs.')
    (out/'65AA_C_summary.json').write_text(json.dumps(report,indent=2))
    print('65AA-C DONE:',out)

if __name__=='__main__':main()
