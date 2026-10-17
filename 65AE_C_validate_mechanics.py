#!/usr/bin/env python3
"""65AE-C: evaluate training-selected stress-aware projection on 64J VALIDATION.

The saved beta is frozen. Test the original learned S22 head, beta=0 (65AD)
and stress-aware beta without using reference Abaqus S at inference.
This is STILL development validation, not a fresh independent holdout.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
ae=importlib.import_module('65AE_common')


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5:raise ValueError('members must be 1 to 5')
    choice_file=root/'results'/ae.ROOT/'B_train_selection'/'65AE_B_selected.json'
    if not choice_file.is_file():raise FileNotFoundError('Run Step 65AE-B first')
    selection=json.loads(choice_file.read_text())
    if not (selection.get('training_only') and selection.get('original_validation_not_used_in_selection')):
        raise RuntimeError('Invalid source selection protocol')
    lookup={(x['region'],int(x['member'])):x for x in selection['selected']}
    dev=ae.ac.aa.co.device_from_option(a.device)
    for reg in ae.REGIONS:
        for m in range(1,a.members+1):
            if (reg,m) not in lookup:raise RuntimeError('No training-selected beta for '+reg+' member '+str(m))
            ae.prepare(root,reg,m,dev)
    if a.check_only:print('65AE-C CHECK PASS: selected beta and sources are ready');return
    out=ae.output(root,'C_validation_decision');allrows=[];decisions=[]
    for reg in ae.REGIONS:
        for m in range(1,a.members+1):
            data=ae.prepare(root,reg,m,dev);sel=lookup[(reg,m)]
            if sel['source_sha256']!=data['source_sha'] or sel['lgno_sha256']!=data['lgno_sha']:
                raise RuntimeError('Upstream checkpoints changed; cannot validate training selection')
            ds=data['ds'];ids=ds['splits']['validation']
            U0,G0,head,C=ae.batch_inputs(data,ids,dev)
            metrics={}
            paths={}
            for name,beta in [('65AD_baseline',0.0),('65AE_stress_aware',float(sel['chosen_beta']))]:
                goal,changes=ae.stress_target(G0,C,head,beta)
                U,G,notes=ae.project(data,U0,goal)
                measurements=ae.evaluate(data,ids,U,G,C,head)
                measurements.update(model=name,region=reg,member=m,beta=beta,
                                    validation_cases=len(ids),
                                    interface_max_change_mm=float(np.max(abs(U[:,data['boundary'],:]-U0[:,data['boundary'],:]))),
                                    projection_residual_RMS=float(np.mean([x['unintegrable_gradient_residual_rms'] for x in notes])))
                allrows.append(measurements);metrics[name]=measurements;paths[name]=(U,G)
                print('65AE-C',reg,name,'S22 %.2f%% U2 %.3f%% gradient %.3f%%'%(
                    measurements['S22_pct'],measurements['U2_pct'],measurements['gradient_pct']),flush=True)
            old=metrics['65AD_baseline'];new=metrics['65AE_stress_aware']
            head_error=old['learned_head_S22_pct']
            safer=ae.evaluation_safe(new,old)
            actual_gain=100*(old['S22_pct']-new['S22_pct'])/max(old['S22_pct'],1e-12)
            full_pass=bool(safer and actual_gain>=5 and new['S22_pct']<=head_error*.95 and
                           new['interface_max_change_mm']<1e-8)
            decision=dict(region=reg,member=m,selected_beta=float(sel['chosen_beta']),
                S22_before_pct=old['S22_pct'],S22_after_pct=new['S22_pct'],
                original_learned_head_S22_pct=head_error,
                S22_error_reduction_vs_65AD_pct=actual_gain,
                U2_before_pct=old['U2_pct'],U2_after_pct=new['U2_pct'],
                gradient_before_pct=old['gradient_pct'],gradient_after_pct=new['gradient_pct'],
                improvement_safe=bool(safer),beats_65AD_S22_by_5pct=bool(actual_gain>=5),
                beats_original_learned_head_S22_by_5pct=bool(new['S22_pct']<=.95*head_error),
                preserved_interfaces=bool(new['interface_max_change_mm']<1e-8),
                preliminary_gate_passed=full_pass)
            decisions.append(decision)
            # Saved candidate can be independently replayed with its provenance metadata.
            folder=out/reg/('member_%02d'%m);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'65AE_projection_validation.npz',
                  indices=np.asarray(ids,dtype=np.int64),selected_beta=np.array(sel['chosen_beta']),
                  compatible_U=np.asarray(paths['65AE_stress_aware'][0],dtype=np.float32),
                  compatible_G=np.asarray(paths['65AE_stress_aware'][1],dtype=np.float32),
                  base_U=np.asarray(U0,dtype=np.float32),
                  source_sha256=np.array(data['source_sha']),lgno_sha256=np.array(data['lgno_sha']),
                  fixed_boundary=np.asarray(data['boundary'],int))
    pd.DataFrame(allrows).to_csv(out/'65AE_C_validation_metrics.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'65AE_C_comparison.csv',index=False)
    (out/'65AE_C_summary.json').write_text(json.dumps(dict(
        step='65AE-C',data_cohort='Original 64J development validation only',
        original_test_and_65W_untouched=True,model_selection='beta fixed on train, evaluated validation only',
        stress_target_in_prediction='existing 65AA learned stress S22 head',
        baseline='65AD compatible LGNO gradient projection with beta=0',
        selected_candidates=decisions,
        all_preliminary_gates_passed=bool(all(x['preliminary_gate_passed'] for x in decisions)),
        warning='Stress approximation C @ Hencky strain is not exact finite deformation; no fresh case or coupled force equilibrium measured. The original 65V result is untouched.'),indent=2))
    print('65AE-C DONE:',out)
if __name__=='__main__':main()
