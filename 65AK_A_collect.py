#!/usr/bin/env python3
"""65AK-A: collect FROZEN predictions, per-case LE/S errors, and FE-U kinematics.

macOS in project root:
    python3 65AK_A_collect.py --project . --check-only
    python3 65AK_A_collect.py --project .

Never trains, retunes or alters pretrained checkpoints or any Abaqus reference.
Uses full-FEM ORACLE interface PCA coefficients, just as 65AJ-F did.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import importlib
ak=importlib.import_module('65AK_common')


def evaluate_region(root,region,device,ext):
    ai=ak.ai; aj=ak.aj
    st=ai.member_source(root,region,1,device,True)
    # This is the exact frozen externally conditioned evaluation path in 65AJ-F.
    ids,branch,pca_error,distances=aj.attach_external(root,st,ext,device)
    data=ai.base_features(st,ids,device)
    scales=ai.train_scales(st['ds'])
    out={'65AG_uncoupled':(data['L0'],data['S0'])}
    for name,model in aj.models_for_comparison(root,st,device).items():
        out[name]=ai.predict(model,data,*scales,device)
    # Original Abaqus C3D8R kinematic definitions: log(E) from grad U.
    grad=ai.ag.ac.aa.gradient_numpy
    to_log=ai.ag.ac.logstrain_from_gradient
    truth_kin=to_log(grad(data['Uref'],st['ds']))
    pred_kin=to_log(grad(data['U0'],st['ds']))
    C=np.asarray(data['C'],np.float32)
    z=dict(case_ids=np.asarray(ext['case_ids']).astype(str),
           materials=np.asarray(ext['materials'],dtype=np.float32),
           coordinates=np.asarray(ext['coordinates'],dtype=np.float32),
           ip_coordinates=np.asarray(ext['ip_coordinates'],dtype=np.float32),
           source_npz_sha256=np.array(ak.sha256(ak.input_npz(root,region))),
           U_ref=np.asarray(data['Uref'],dtype=np.float32),
           U_pred=np.asarray(data['U0'],dtype=np.float32),
           LE_ref=np.asarray(data['Lref'],dtype=np.float32),
           S_ref=np.asarray(data['Sref'],dtype=np.float32),
           LE_from_FEM_U=np.asarray(truth_kin,dtype=np.float32),
           LE_from_pred_U=np.asarray(pred_kin,dtype=np.float32),
           S_from_FEM_LE=np.asarray(ai.ag.f.C_LE(data['Lref'],C),dtype=np.float32),
           S_from_pred_U=np.asarray(ai.ag.f.C_LE(pred_kin,C),dtype=np.float32))
    for name,(L,S) in out.items():
        z['LE_'+name]=np.asarray(L,dtype=np.float32)
        z['S_'+name]=np.asarray(S,dtype=np.float32)
        z['S_constitutive_'+name]=np.asarray(ai.ag.f.C_LE(L,C),dtype=np.float32)
    k=len(ids)
    for key in ('U_ref','U_pred'):
        if z[key].shape[0]!=k:raise RuntimeError('Case count mismatch in '+key)
    if ak.relpct(z['LE_from_FEM_U'],z['LE_ref'])>2.0:
        raise RuntimeError('Reference FEM U->LE mismatch; do not analyze unverified exports: '+region)
    return z,pca_error,distances


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--device',default='auto',choices=('auto','cpu','mps'))
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=Path(args.project).resolve()
    device=ak.ai.co.device_from_option(args.device)
    summary=root/'results'/ak.aj.ROOT/'F_external_evaluation'/'65AJ_F_summary.json'
    if not summary.is_file():raise FileNotFoundError('Run 65AJ-F first: '+str(summary))
    fsummary=json.loads(summary.read_text())
    if not fsummary.get('conditional_generalization_with_full_FEM_oracle_interface_coefficients'):
        raise RuntimeError('65AJ-F summary does not confirm oracle FEM interface protocol')
    inp={}
    for reg in ak.REGIONS:
        e=ak.open_external(root,reg)
        if len(e['case_ids'])!=6:raise RuntimeError('Expected exactly six reference FEM cases: '+reg)
        if len(set(e['case_ids'].astype(str)))!=6:raise RuntimeError('Repeated IDs in '+reg)
        expected=fsummary['package_sha256'][reg]
        if ak.sha256(ak.input_npz(root,reg))!=expected:
            raise RuntimeError('65AJ-E NPZ hash does not match previously evaluated 65AJ-F summary: '+reg)
        st=ak.ai.member_source(root,reg,1,device,False)
        ak.aj.load_control(root,st,device)
        ak.ai.load_candidate(root,'local_lgno',st,device)
        inp[reg]=e
    if args.check_only:
        print('65AK-A CHECK PASS: six-case NPZ hashes, checkpoints and provenance verified; no models were trained.');return
    # Cross-check the new prediction path against independently saved 65AJ-F totals.
    fmetrics_path=root/'results'/ak.aj.ROOT/'F_external_evaluation'/'65AJ_F_external_metrics.csv'
    if not fmetrics_path.is_file():raise FileNotFoundError('Missing completed 65AJ-F metrics: '+str(fmetrics_path))
    fmetrics=pd.read_csv(fmetrics_path)
    out=ak.new_output(root,'A_predictions')
    per=[];wins=[];quality=[]
    for reg in ak.REGIONS:
        z,pca_error,dist=evaluate_region(root,reg,device,inp[reg])
        for model in ak.ARMS:
            ref=fmetrics[(fmetrics.region==reg)&(fmetrics.model==model)]
            if len(ref)!=1:raise RuntimeError('65AJ-F comparison row missing/duplicated for '+reg+'/'+model)
            for fld in ('LE22','S22'):
                estimate=ak.relpct(z[fld[:2]+'_'+model][...,1] if fld=='LE22' else z['S_'+model][...,1],
                                   z[fld[:2]+'_ref'][...,1] if fld=='LE22' else z['S_ref'][...,1])
                saved=float(ref.iloc[0][fld+'_pct'])
                if abs(estimate-saved)>.15:
                    raise RuntimeError('65AK prediction differs from validated 65AJ-F (%s %s %s): %.5f vs %.5f pct. Stop; do not interpret.'%(reg,model,fld,estimate,saved))
        np.savez_compressed(out/(reg+'.npz'),**z)
        for i,case in enumerate(z['case_ids'].tolist()):
            base=z['LE_ref'][i];refS=z['S_ref'][i]
            for model in ak.ARMS:
                le,s=ak.per_case_fields(z,model,i)
                row=dict(region=reg,case_id=case,model=model,
                         E1_MPa=float(z['materials'][i,0]),E2_MPa=float(z['materials'][i,1]),G12_MPa=float(z['materials'][i,2]),
                         LE_vector_pct=ak.relpct(le,base),S_vector_pct=ak.relpct(s,refS),
                         U_vector_pct=ak.relpct(z['U_pred'][i],z['U_ref'][i]))
                for j,suff in enumerate(ak.COMPONENTS):
                    row['LE'+suff+'_pct']=ak.relpct(le[:,j],base[:,j])
                    row['S'+suff+'_pct']=ak.relpct(s[:,j],refS[:,j])
                per.append(row)
            lg=next(x for x in per if x['region']==reg and x['case_id']==case and x['model']=='65AI_local_lgno')
            ml=next(x for x in per if x['region']==reg and x['case_id']==case and x['model']=='65AJ_matched_mlp')
            wins.append(dict(region=reg,case_id=case,
                LGNO_S22_pct=lg['S22_pct'],MLP_S22_pct=ml['S22_pct'],
                LGNO_LE22_pct=lg['LE22_pct'],MLP_LE22_pct=ml['LE22_pct'],
                LGNO_wins_S22=bool(lg['S22_pct']<ml['S22_pct']),
                LGNO_wins_LE22=bool(lg['LE22_pct']<ml['LE22_pct']),
                S22_gain_over_MLP_pct=100*(1-lg['S22_pct']/max(ml['S22_pct'],1e-14)),
                LE22_gain_over_MLP_pct=100*(1-lg['LE22_pct']/max(ml['LE22_pct'],1e-14))))
        quality.append(dict(region=reg,cases=len(z['case_ids']),
            reference_LE22_from_FEM_U_pct=ak.relpct(z['LE_from_FEM_U'][...,1],z['LE_ref'][...,1]),
            reference_constitutive_S22_pct=ak.relpct(z['S_from_FEM_LE'][...,1],z['S_ref'][...,1]),
            **pca_error))
        print('65AK-A processed',reg,flush=True)
    pd.DataFrame(per).to_csv(out/'65AK_A_per_case_metrics.csv',index=False)
    pd.DataFrame(wins).to_csv(out/'65AK_A_case_winners.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'65AK_A_reference_checks.csv',index=False)
    ak.atomic_summary(out/'65AK_A_summary.json',dict(step='65AK-A',cases_per_region=6,
        uses_oracle_full_FEM_interface_PCA=True,no_retraining=True,
        notes='Per-case results are descriptive repeated looks at the same six external cases, not an additional independent holdout.',
        winners={reg:{'S22':int(sum(x['LGNO_wins_S22'] for x in wins if x['region']==reg)),
                      'LE22':int(sum(x['LGNO_wins_LE22'] for x in wins if x['region']==reg))} for reg in ak.REGIONS}))
    print('65AK-A DONE:',out)

if __name__=='__main__':main()
