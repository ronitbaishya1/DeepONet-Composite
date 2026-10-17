#!/usr/bin/env python3
"""65AI-C: historical development comparison (NOT independent holdout).

Compares original DeepONet/65AF/65AG against the two 65AI correction arms.
Reports U2, LE22, S22, S11, S12, stress near interfaces and approximate
kinematic/constitutive disagreement. No new model weights fitted here.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
ai=importlib.import_module('65AI_common')


def around_interface_error(S,ref,mask):
    if not np.any(mask):return float('nan')
    return float(ai.ag.f.relative(S[:,mask,1],ref[:,mask,1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5 or not 1<=a.batch_size<=16:raise ValueError('Invalid members/batch size')
    device=ai.co.device_from_option(a.device)
    for reg in ai.REGIONS:
        for m in range(1,a.members+1):
            st=ai.member_source(root,reg,m,device,False)
            for arm in ai.ARMS:
                ai.load_candidate(root,arm,st,device)
    if a.check_only:
        print('65AI-C CHECK PASS: all source and comparison model files found');return
    out=ai.no_overwrite(root,'C_development_comparison')
    rows=[];decisions=[]
    for reg in ai.REGIONS:
        for m in range(1,a.members+1):
            st=ai.member_source(root,reg,m,device,True)
            ids=np.asarray(st['ds']['splits']['validation'],np.int64)
            if set(map(int,ids))&set(map(int,st['ds']['splits']['train'])):
                raise RuntimeError('Unexpected dataset overlap')
            data=ai.base_features(st,ids,device,a.batch_size)
            ls,ss=ai.train_scales(st['ds'])
            curves={'original_65AA':(data['head_L'],data['head_S']),
                    'prior_65AF_pointwise':(data['af_L'],data['af_S']),
                    'prior_65AG_uncoupled':(data['L0'],data['S0'])}
            for arm in ai.ARMS:
                model,_=ai.load_candidate(root,arm,st,device)
                curves['65AI_'+arm]=ai.predict(model,data,ls,ss,device,a.batch_size)
                del model
            geom=np.asarray(st['geometry']['position'][:,0],np.float64)
            near=np.abs(geom)>=0.7
            if not near.any():near=np.ones_like(geom,dtype=bool)
            scores={}
            for name,(L,S) in curves.items():
                metric=ai.error_metrics(L,S,data)
                metric['interface_band_S22_pct']=around_interface_error(S,data['Sref'],near)
                row=dict(region=reg,member=m,model=name,validation_cases=len(ids),
                         reference_model_sha256=st['prior_sha'],**metric)
                rows.append(row);scores[name]=metric
                print('65AI-C %-11s %-23s S22 %.3f%% LE22 %.3f%% U2 %.3f%%'%(
                    reg,name,metric['S22_pct'],metric['LE22_pct'],metric['U2_pct']),flush=True)
            old=scores['prior_65AG_uncoupled'];ctrl=scores['65AI_pointwise'];new=scores['65AI_local_lgno']
            local_better_prior=new['S22_pct']<=.95*old['S22_pct']
            local_better_ctrl=new['S22_pct']<=.95*ctrl['S22_pct']
            strain_gain=new['LE22_pct']<=.98*old['LE22_pct']
            safe=ai.safety(new,old)
            passed=bool(local_better_prior and local_better_ctrl and strain_gain and safe)
            decisions.append(dict(region=reg,member=m,
                original_65AG_S22_pct=old['S22_pct'],pointwise_S22_pct=ctrl['S22_pct'],
                local_lgno_S22_pct=new['S22_pct'],
                prior_65AG_LE22_pct=old['LE22_pct'],local_lgno_LE22_pct=new['LE22_pct'],
                lgno_gain_vs_prior_S22_pct=100*(1-new['S22_pct']/max(old['S22_pct'],1e-8)),
                lgno_gain_vs_pointwise_S22_pct=100*(1-new['S22_pct']/max(ctrl['S22_pct'],1e-8)),
                lgno_interface_band_S22_pct=new['interface_band_S22_pct'],
                prior_interface_band_S22_pct=old['interface_band_S22_pct'],
                prior_U2_pct=old['U2_pct'],local_U2_pct=new['U2_pct'],
                S22_better_than_65AG_by_5pct=bool(local_better_prior),
                S22_better_than_pointwise_by_5pct=bool(local_better_ctrl),
                LE22_better_than_65AG_by_2pct=bool(strain_gain),
                other_components_safe=bool(safe),
                interface_U_unchanged=True,source_generalized_force_unchanged=True,
                preliminary_gate_passed=passed))
    pd.DataFrame(rows).to_csv(out/'65AI_C_all_validation_metrics.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'65AI_C_decisions.csv',index=False)
    report=dict(step='65AI-C',model='frozen 65AA U conditioned local U -> (LE,S) joint residuals',
        arms=['pointwise local feature MLP','LGNO-inspired local U neighbor stencils'],
        historical_validation='Original 64J validation repeatedly examined, not independent',
        upstream_models_preselected_using_historical_validation=True,
        new_weights_selected_on='internal split of 64J original TRAIN only',
        original_64J_test_and_65W_not_used=True,
        U_and_force_unchanged_by_architecture=True,
        reference_true_U_not_used_as_model_input=True,
        displacement_gradient_strain_compatibility_not_guaranteed=True,
        constitutive_gap_uses_approximate_C_Hencky_for_Abaqus_stresses=True,
        parameter_count_matched_between_arms=False,
        no_seven_region_coupled_evaluation=True,
        decisions=decisions,all_preliminary_gates_passed=all(x['preliminary_gate_passed'] for x in decisions),
        next='No deployment. Independent new Abaqus FE cases and consistent mechanics/coupled forces required.')
    (out/'65AI_C_summary.json').write_text(json.dumps(report,indent=2))
    print('65AI-C DONE:',out)

if __name__=='__main__':main()
