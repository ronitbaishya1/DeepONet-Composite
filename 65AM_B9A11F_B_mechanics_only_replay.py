#!/usr/bin/env python3
"""B9A11F-B: DIAGNOSTIC 65Y mechanics-head-only replay at FROZEN 65V q.

Requires 65Y-C holdout gate, five original/candidate checkpoint pairs for BOTH
inner regions, and complete saved 65W-H middle-material exports on Mac.

No Abaqus, no retraining, no changes to original models. Output to NEW tag only.
Before replay, proves all non-mechanics checkpoint tensors unchanged exactly.
After replay, verifies every assembled nodal displacement equals old 65W baseline,
and outer/FE mechanics is unchanged, so only inner mechanics heads are compared.

DO NOT use the middle FEM reference to re-tune a model selected by 65Y-C.
DO NOT call this a coupled B5 model: the V4 force/displacement is unchanged.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

INNER=('inner_left','inner_right')
SEGMENTS=('outer_left','inner_left','inner_right','outer_right')
REGION_LABELS={'inner_left':'NO_L','inner_right':'NO_R'}
FOCUS=('LE11','LE12','LE22','S11','S12','S22','S33')
MECH_PREFIX=('strain_operator.','stress_operator.')

def norm_metric(reference,prediction):
    ref=np.asarray(reference,float); pred=np.asarray(prediction,float)
    assert ref.shape==pred.shape and np.isfinite(ref).all() and np.isfinite(pred).all()
    return {'relative_l2_percent':float(100*np.linalg.norm((pred-ref).ravel())/max(np.linalg.norm(ref.ravel()),1e-15)),
            'rmse':float(np.sqrt(np.mean((pred-ref)**2))),
            'squared_error_sum':float(np.sum((pred-ref)**2))}

def check_models(root):
    import torch
    out={}
    for region in INNER:
        out[region]={}
        for m in range(1,6):
            key='member_%02d'%m
            a=root/'results'/'v4_7region_direct'/region/key/'best_v4_7region.pt'
            b=root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'/region/key/'best_v4_7region.pt'
            if not a.is_file() or not b.is_file():raise FileNotFoundError('Missing pair: '+str(a)+' / '+str(b))
            baseline=torch.load(a,map_location='cpu',weights_only=False)
            trial=torch.load(b,map_location='cpu',weights_only=False)
            meta=trial.get('65Y_finetune',{})
            if not meta.get('train_split_only') or meta.get('constitutive_loss_used') or meta.get('kinematic_loss_used'):
                raise RuntimeError(region+'/'+key+': unexpected training metadata')
            if meta.get('segment')!=region or int(meta.get('member',-1))!=m:
                raise RuntimeError(region+'/'+key+': candidate attribution mismatch')
            x=baseline['model_state_dict'];y=trial['model_state_dict']
            if x.keys()!=y.keys():raise RuntimeError('Incompatible checkpoint tensors: '+region+'/'+key)
            changed=[]
            for name in x:
                if not torch.equal(x[name],y[name]):
                    if not name.startswith(MECH_PREFIX):
                        raise RuntimeError('Nonmechanics tensor changed! '+region+'/'+key+'/'+name)
                    changed.append(name)
            # Require branch and field normalization to be the same for both models.
            for field in ('branch_mean','branch_std','LE_mean','LE_std','S_mean','S_std',
                          'coordinate_min','coordinate_max'):
                if not np.array_equal(np.asarray(baseline[field]),np.asarray(trial[field])):
                    raise RuntimeError('Checkpoint input/output scaling changed: '+field)
            out[region][key]={'changed_mechanics_tensors':len(changed), 'non_mechanics_identical':True,
                             'origin':str(a),'candidate':str(b)}
    return out

def prepare(root,tag):
    a=root/'results'/'65Y_middle_mechanics'/'C_holdout_comparison'/'65Y_C_summary.json'
    if not a.is_file():raise FileNotFoundError('Missing holdout gate: '+str(a))
    gate=json.loads(a.read_text())
    if gate.get('all_preliminary_gates_passed') is not True or gate.get('displacement_and_force_heads_identical') is not True:
        raise RuntimeError('65Y-C test gate failed. DO NOT use its models as replacement.')
    for region in INNER:
        if gate.get('gates',{}).get(region,{}).get('preliminary_gate_passed') is not True:
            raise RuntimeError('Holdout gate failed for '+region)
    baseline=root/'results'/'hybrid_online_7region_65W_middle_validation'
    src=root/'64V_reconstruct_validate_7region_direct.py'
    inp=root/'data'/'hybrid_online_7region_65W_middle'
    expected=[src,inp/'rom_solution_7region.npz',baseline/'assembled_hybrid_nodes.csv',
              baseline/'assembled_hybrid_ip.csv',baseline/'65W_H_validation_summary.json']
    expected += [inp/'validation_fields'/x for x in ('full_reference_nodes.csv','full_reference_ip.csv',
               'left_nodes.csv','left_ip.csv','center_nodes.csv','center_ip.csv','right_nodes.csv','right_ip.csv')]
    for seg in SEGMENTS:
        for m in range(1,6):
            expected.append(root/'results'/'v4_7region_direct'/seg/('member_%02d'%m)/'best_v4_7region.pt')
    for p in expected:
        if not p.is_file():raise FileNotFoundError('Missing original 65W/64V inputs: '+str(p))
    with (baseline/'65W_H_validation_summary.json').open() as f: summary=json.load(f)
    if summary.get('full_fem_nodes')!=4592 or not np.allclose(summary.get('material_MPa'),[46500,12500,4750],atol=1e-6,rtol=0):
        raise RuntimeError('Original 65W baseline source inconsistent')
    out=root/'results'/'65AM_B9A11F_mechanics_only' / tag
    if out.exists():raise FileExistsError('Destination already exists. Use new --tag: '+str(out))
    return src,baseline,out

def make_links(root,out):
    models=out/'checkpoints'
    for seg in SEGMENTS:
        for m in range(1,6):
            member='member_%02d'%m
            source=(root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'/seg/member/'best_v4_7region.pt'
                   if seg in INNER else
                   root/'results'/'v4_7region_direct'/seg/member/'best_v4_7region.pt')
            d=models/seg/member
            d.mkdir(parents=True,exist_ok=False)
            (d/'best_v4_7region.pt').symlink_to(source)
    return models

def run_original_reconstruction(root,src,out):
    source=src.read_text(encoding='utf-8')
    required={'hybrid_online_7region_direct_validation':1,'v4_7region_direct':1,
              'online_solution_7region.npz':1,'# REACTION SUMMARY':1}
    for name,count in required.items():
        if source.count(name)!=count:raise RuntimeError('Original 64V code changed; review adapter for '+name)
    # Root output relative to /results; we isolate the candidate in a separate tag.
    output_token='65AM_B9A11F_mechanics_only/'+out.name+'/evaluation'
    model_token='65AM_B9A11F_mechanics_only/'+out.name+'/checkpoints'
    adapted=source.replace('hybrid_online_7region_direct_validation',output_token)
    adapted=adapted.replace('hybrid_online_7region_direct','hybrid_online_7region_65W_middle')
    adapted=adapted.replace('online_solution_7region.npz','rom_solution_7region.npz')
    adapted=adapted.replace('v4_7region_direct',model_token)
    cutoff=adapted.index('# REACTION SUMMARY')
    adapted=adapted[:adapted.rfind('# ============================================================',0,cutoff)]
    if any(s in adapted for s in ('hybrid_online_7region_direct','online_solution_7region.npz','v4_7region_direct')):
        raise RuntimeError('Old output checkpoint references remain in adapted source')
    os.chdir(root)
    scope={'__name__':'__main__','__file__':str(src)}
    exec(compile(adapted,str(src)+' [B9A11F isolated mechanics replay]', 'exec'),scope)
    return out/'evaluation',scope

def validate_and_compare(baseline,candidate,checkpoint_audit,out):
    base_node=pd.read_csv(baseline/'assembled_hybrid_nodes.csv')
    new_node=pd.read_csv(candidate/'assembled_hybrid_nodes.csv')
    base_ip=pd.read_csv(baseline/'assembled_hybrid_ip.csv')
    new_ip=pd.read_csv(candidate/'assembled_hybrid_ip.csv')
    if len(base_node)!=4592 or len(new_node)!=4592 or len(base_ip)!=3640 or len(new_ip)!=3640:
        raise RuntimeError('Wrong full-field point counts')
    for df in (base_node,new_node):
        if df['NodeLabel'].duplicated().any():raise RuntimeError('Duplicate node labels')
    for df in (base_ip,new_ip):
        if df['ElementLabel'].duplicated().any():raise RuntimeError('Duplicate element labels')
    # Compare only after stable key alignment, never blindly by row.
    base_node=base_node.sort_values('NodeLabel').reset_index(drop=True)
    new_node=new_node.sort_values('NodeLabel').reset_index(drop=True)
    base_ip=base_ip.sort_values('ElementLabel').reset_index(drop=True)
    new_ip=new_ip.sort_values('ElementLabel').reset_index(drop=True)
    for x,y,keys in ((base_node,new_node,['NodeLabel','Owner','X','Y','Z']),
                     (base_ip,new_ip,['ElementLabel','Owner','X','Y','Z'])):
        for k in keys:
            if k=='Owner' or k.endswith('Label'):
                if not x[k].equals(y[k]):raise RuntimeError('Rows misaligned in '+k)
            elif np.max(np.abs(x[k].to_numpy(float)-y[k].to_numpy(float)))>1e-7:
                raise RuntimeError('Coordinates changed unexpectedly: '+k)
    displacement_diff={}
    for k in ('U1','U2','U3'):
        delta=new_node['Hybrid_'+k].to_numpy(float)-base_node['Hybrid_'+k].to_numpy(float)
        displacement_diff[k]=float(np.max(np.abs(delta)))
        if displacement_diff[k]>1e-7:
            raise RuntimeError('CRITICAL: Candidate changed predicted displacement '+k)
        if not np.allclose(new_node['FEM_'+k].to_numpy(float),base_node['FEM_'+k].to_numpy(float),atol=1e-9,rtol=0):
            raise RuntimeError('FEM displacement reference changed: '+k)
    rows=[]
    for region in ['NO_OL','FE_L','NO_L','FE_C','NO_R','FE_R','NO_OR']:
        b=base_ip[base_ip.Owner==region].sort_values('ElementLabel').reset_index(drop=True)
        n=new_ip[new_ip.Owner==region].sort_values('ElementLabel').reset_index(drop=True)
        if len(b)==0 or len(n)!=len(b) or not b.ElementLabel.equals(n.ElementLabel):
            raise RuntimeError('Missing/misaligned mechanics region '+region)
        for comp in FOCUS:
            ref=b['FEM_'+comp].to_numpy(float)
            if not np.allclose(ref,n['FEM_'+comp].to_numpy(float),rtol=0,atol=1e-9):
                raise RuntimeError('FEM mechanics reference changed in '+region+'/'+comp)
            old=b['Hybrid_'+comp].to_numpy(float)
            new=n['Hybrid_'+comp].to_numpy(float)
            before=norm_metric(ref,old);after=norm_metric(ref,new)
            change=float(np.max(np.abs(old-new)))
            if region not in ('NO_L','NO_R') and change>1e-8:
                raise RuntimeError('Unexpected change to frozen FE/outer model: '+region+'/'+comp)
            gain=100*(before['squared_error_sum']-after['squared_error_sum'])/max(before['squared_error_sum'],1e-15)
            rows.append({'region':region,'component':comp,'n_points':len(ref),
                         'baseline_rel_l2_pct':before['relative_l2_percent'],
                         'candidate_rel_l2_pct':after['relative_l2_percent'],
                         'baseline_rmse':before['rmse'],'candidate_rmse':after['rmse'],
                         'sse_gain_pct':gain,'max_abs_new_minus_old':change})
    result={'step':'B9A11F-B','experiment':'65Y mechanics-head-only on frozen V4/65V, not B5 coupled',
            'baseline_65W':str(baseline),'candidate_result':str(candidate),
            'node_count':len(new_node),'ip_count':len(new_ip),
            'max_abs_displacement_change_mm':displacement_diff,
            'nonmechanics_checkpoint_tensors_identical':True,
            'checkpoint_verification':checkpoint_audit,
            'physical_material_loading_and_reaction_audits_remain_separate':True,
            'selection_warning':'65W middle FEM reference is an additional diagnostic only; never tune using these metrics and then claim an untouched test.'}
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'B9A11F_B_regional_mechanics.csv',index=False)
    (out/'B9A11F_B_summary.json').write_text(json.dumps(result,indent=2))
    print(frame[frame.region.isin(['NO_L','NO_R']) & frame.component.isin(['LE22','S22','S33'])].to_string(index=False))
    print('B9A11F-B complete. Saved:',out/'B9A11F_B_regional_mechanics.csv')

def selftest():
    assert norm_metric(np.array([1.,2.]),np.array([1.,2.]))['squared_error_sum']==0
    assert abs(norm_metric(np.array([1.,2.]),np.array([2.,2.]))['relative_l2_percent']-100/np.sqrt(5))<1e-12
    print('B9A11F-B numerical SELFTEST PASS')

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--project',default='.')
    pa.add_argument('--tag',default='run01')
    pa.add_argument('--check-only',action='store_true')
    pa.add_argument('--selftest',action='store_true')
    a=pa.parse_args()
    if a.selftest:selftest();return
    root=Path(a.project).expanduser().resolve()
    src,baseline,out=prepare(root,a.tag)
    audit=check_models(root)
    print('65Y heldout gate PASSED; verified 10 candidate checkpoints only modified mechanics heads.')
    if a.check_only:
        print('PREFLIGHT PASS. Nothing created.')
        return
    out.mkdir(parents=True)
    make_links(root,out)
    candidate,_=run_original_reconstruction(root,src,out)
    validate_and_compare(baseline,candidate,audit,out)

if __name__=='__main__':main()
