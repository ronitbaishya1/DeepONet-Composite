#!/usr/bin/env python3
"""65AE-B: select stress-aware correction strength on TRAIN cases only.

This is training of ONE scalar parameter, beta (not new NN weights).
Inference uses only the pre-trained LGNO gradients, S22 predicted by existing
65AA stress head, and material stiffness. Training Abaqus S22 is used ONLY
for selecting beta; never needed for prediction.

python3 65AE_B_train_stress_weight.py --project . --members 1
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
    args=p.parse_args();root=Path(args.project).expanduser().resolve()
    if not 1<=args.members<=5:raise ValueError('members must be 1..5')
    diag=root/'results'/ae.ROOT/'A_S22_diagnosis'/'65AE_A_summary.json'
    if not diag.is_file():raise FileNotFoundError('Run Step 65AE-A first')
    dev=ae.ac.aa.co.device_from_option(args.device)
    for region in ae.REGIONS:
        for member in range(1,args.members+1):ae.prepare(root,region,member,dev)
    if args.check_only:print('65AE-B CHECK PASS: TRAIN cases + LGNO + stress head + scipy solver');return
    out=ae.output(root,'B_train_selection');allrows=[];choices=[]
    for reg in ae.REGIONS:
        for member in range(1,args.members+1):
            data=ae.prepare(root,reg,member,dev)
            ds=data['ds'];ids=ds['splits']['train']
            U0,GL,head,C=ae.batch_inputs(data,ids,dev)
            train_baseline=None;best=None;best_score=np.inf
            for beta in ae.BETAS:
                target,changes=ae.stress_target(GL,C,head,beta)
                U,G,proj_notes=ae.project(data,U0,target)
                scores=ae.evaluate(data,ids,U,G,C,head)
                if train_baseline is None:
                    if beta!=0:raise RuntimeError('First candidate must be beta=0')
                    train_baseline=scores;best=0.;best_score=ae.selection_objective(scores)
                safe=beta==0 or ae.evaluation_safe(scores,train_baseline)
                objective=ae.selection_objective(scores)
                if safe and objective<best_score-1e-8:
                    best=float(beta);best_score=objective
                row=dict(region=reg,member=member,beta=beta,train_cases=len(ids),
                         passed_training_safety_gate=bool(safe),objective=objective,
                         source_sha256=data['source_sha'],lgno_sha256=data['lgno_sha'],
                         projection_mean_gap_RMS=float(np.mean([x['unintegrable_gradient_residual_rms'] for x in proj_notes])),
                         **changes,**scores)
                allrows.append(row)
                print('65AE-B %-11s beta=%4.2f train S22 %.2f%% grad %.2f%% U2 %.2f%% %s'%(
                    reg,beta,scores['S22_pct'],scores['gradient_pct'],scores['U2_pct'],
                    'safe' if safe else 'reject'),flush=True)
            # Selection based on original TRAIN only, no validation and no test.
            chosen=dict(region=reg,member=member,chosen_beta=best,
                        baseline_training_S22_pct=train_baseline['S22_pct'],
                        baseline_training_head_S22_pct=train_baseline['learned_head_S22_pct'],
                        chosen_training_objective=best_score,
                        source_sha256=data['source_sha'],lgno_sha256=data['lgno_sha'],
                        projection_lambda=.05,training_cases=len(ids),
                        policy='train_only_beta_selection; no new neural network')
            choices.append(chosen)
            print('CHOSEN',reg,'member',member,'beta=',best,flush=True)
    pd.DataFrame(allrows).to_csv(out/'65AE_B_train_grid.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'65AE_B_selected.csv',index=False)
    (out/'65AE_B_selected.json').write_text(json.dumps(dict(
        step='65AE-B',selected=choices,all_beta_candidates=list(ae.BETAS),
        training_only=True,original_validation_not_used_in_selection=True,
        original_test_and_65W_not_used=True,
        inference_stress_target='65AA learned stress head S22, not Abaqus reference S22',
        no_original_network_weights_modified=True),indent=2))
    print('65AE-B DONE:',out)
if __name__=='__main__':main()
