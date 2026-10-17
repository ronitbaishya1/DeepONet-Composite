#!/usr/bin/env python3
"""65AG-C: compare frozen 65AF baselines, uncoupled and coupled joint heads.

Produces DEVELOPMENT validation diagnostics only; original 12-case 64J
validation has been seen repeatedly upstream, so results are NOT independent.
No existing FE-NO solver, force heads, U fields or interfaces are modified.

python3 65AG_C_compare.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import pandas as pd
import importlib
ag=importlib.import_module('65AG_common')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.members<=5:raise ValueError('Expected member count 1-5')
    root=Path(a.project).expanduser().resolve()
    dev=ag.ac.aa.co.device_from_option(a.device)
    for reg in ag.REGIONS:
        for member in range(1,a.members+1):
            st=ag.attach_st(ag.member_sources(root,reg,member,dev,False),reg,member)
            for arm in ag.ARMS:
                file=ag.model_path(root,arm,reg,member)
                if not file.is_file():raise FileNotFoundError('Train both 65AG-B arms: '+str(file))
    if a.check_only:
        print('65AG-C CHECK PASS: uncoupled and coupled models and frozen baselines found');return
    out=ag.safe_output(root,'C_validation_comparison')
    rows=[];decisions=[]
    for reg in ag.REGIONS:
        for member in range(1,a.members+1):
            st=ag.attach_st(ag.member_sources(root,reg,member,dev),reg,member)
            ds=st['ds'];ids=ds['splits']['validation']
            data=ag.make_dataset(st,ids,dev,a.batch_size)
            vals={}
            for name,(L,S) in ag.baselines_from_data(data).items():
                vals[name]=ag.report(L,S,data['LEref'],data['Sref'],data['C'])
            for arm in ag.ARMS:
                mdl,ck=ag.load_model(root,arm,st,dev)
                L,S=ag.forward_numpy(mdl,data,ag.f.strain_scales(ds),ag.f.stress_scales(ds),dev,a.batch_size)
                vals['65AG_'+arm]=ag.report(L,S,data['LEref'],data['Sref'],data['C'])
                del mdl
            for name,metr in vals.items():
                rows.append(dict(region=reg,member=member,model=name,cases=len(ids),**metr))
                print('65AG-C %-11s %-24s S22 %.2f%% LE22 %.2f%% S11 %.2f%% consistency %.2f%%'%(
                    reg,name,metr['S22_pct'],metr['LE22_pct'],metr['S11_pct'],metr['constitutive_S22_gap_pct']),flush=True)
            orig=vals['original_LE_S_heads'];base=vals['65AF_pointwise_blend'];lgno=vals['65AF_lgno_blend']
            free=vals['65AG_uncoupled'];phys=vals['65AG_coupled']
            safety=ag.safe_gates(phys,base,orig)
            wins=bool(phys['S22_pct'] <= .95*base['S22_pct'] and
                      phys['S22_pct'] <= .95*lgno['S22_pct'] and
                      phys['S22_pct'] <= .95*orig['S22_pct'])
            le_gain=bool(phys['LE22_pct'] <= .98*base['LE22_pct'])
            improved_physics=bool(phys['constitutive_S22_gap_pct'] <= .95*free['constitutive_S22_gap_pct'])
            decisions.append(dict(region=reg,member=member,
                baseline_65AF_pointwise_S22_pct=base['S22_pct'],
                baseline_65AF_lgno_S22_pct=lgno['S22_pct'],
                original_head_S22_pct=orig['S22_pct'],
                joint_uncoupled_S22_pct=free['S22_pct'],
                joint_coupled_S22_pct=phys['S22_pct'],
                pointwise_LE22_pct=base['LE22_pct'],joint_coupled_LE22_pct=phys['LE22_pct'],
                uncoupled_consistency_S22_gap_pct=free['constitutive_S22_gap_pct'],
                coupled_consistency_S22_gap_pct=phys['constitutive_S22_gap_pct'],
                coupled_improvement_over_pointwise_S22_pct=100*(1-phys['S22_pct']/max(base['S22_pct'],1e-9)),
                S22_gain_5pct_vs_all_existing=bool(wins),
                LE22_gain_2pct_vs_pointwise=bool(le_gain),
                mechanics_safety_pass=bool(safety),
                consistency_gap_improved_5pct=bool(improved_physics),
                source_displacement_and_interface_force_frozen=True,
                preliminary_gate_passed=bool(wins and le_gain and safety and improved_physics)))
    pd.DataFrame(rows).to_csv(out/'65AG_C_all_validation_metrics.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'65AG_C_decisions.csv',index=False)
    summary=dict(step='65AG-C',members=a.members,model='joint shared local strain-stress correction',
        comparison='Frozen original heads + 65AF pointwise + 65AF LGNO + 65AG uncoupled + 65AG coupled',
        validation_cohort='Previously repeatedly examined original 64J validation; NOT independent',
        new_network_weights_and_epoch_selection='only an internal split of original 64J train',
        upstream_baselines_previous_validation_used=True,
        original_test_and_65W_untouched=True,
        source_U_LE_S_force_heads_unchanged=True,source_displacement_and_interface_displacements_unchanged=True,
        approximate_constitutive_consistency_only=True,
        displacement_gradient_strain_compatibility_not_enforced=True,
        independent_abaqus_test_required=True,
        decisions=decisions,all_preliminary_gates_passed=all(r['preliminary_gate_passed'] for r in decisions),
        next_action='Do NOT deploy, even if passed, until independent FEM cases and coupled equilibrium are verified')
    (out/'65AG_C_summary.json').write_text(json.dumps(summary,indent=2))
    print('65AG-C DONE',out)

if __name__=='__main__':main()
