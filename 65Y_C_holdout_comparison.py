#!/usr/bin/env python3
"""65Y-C: final untouched-test comparison of fine-tuned mechanics-only V4 models.

No Abaqus, no retraining. Run ONLY after 65Y-B training/hyperparameters are fixed.
From DeepONet_Composite root:
  python3 65Y_C_holdout_comparison.py --project . --check-only
  python3 65Y_C_holdout_comparison.py --project .

IMPORTANT: test set is evaluated now; if you tune after seeing it, it ceases
being a clean independent test set. Comparison does not deploy candidate models.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from importlib import import_module

REGIONS=('inner_left','inner_right')
COMPS=('11','22','33','12','13','23')
HEADS=('displacement_operator.','force_coefficient_head.')


def metrics(pred,truth):
    a=np.asarray(pred,float);b=np.asarray(truth,float)
    return {'relative_L2_pct':float(100*np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)),
            'RMSE':float(np.sqrt(np.mean((a-b)**2))),
            'ref_RMS':float(np.sqrt(np.mean(b**2))),
            'squared_error_sum':float(np.sum((a-b)**2))}


def read_model(ckpt_path,device):
    x=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    mod=import_module('src.hybrid_bulk_operator_v4_7region').HybridBulkOperatorV4SevenRegion(
        branch_dim=int(x['branch_dim']),number_force_coefficients=int(x['number_force_coefficients']),
        hidden_dim=int(x.get('hidden_dim',128)),latent_dim=int(x.get('latent_dim',128)),depth=int(x.get('depth',4)))
    mod.load_state_dict(x['model_state_dict'],strict=True)
    mod=mod.to(device).eval()
    return x,mod


def member_predictions(model,ck,X,ip,device,batch=8):
    branch=(X-np.asarray(ck['branch_mean'],float))/np.asarray(ck['branch_std'],float)
    cmin=np.asarray(ck['coordinate_min'],float);cmax=np.asarray(ck['coordinate_max'],float)
    xyz=2*(ip-cmin)/np.maximum(cmax-cmin,1e-8)-1
    xyzt=torch.tensor(xyz,dtype=torch.float32,device=device)
    le=[];s=[]
    with torch.no_grad():
        for start in range(0,len(X),batch):
            b=torch.tensor(branch[start:start+batch],dtype=torch.float32,device=device)
            le.append(model.strain_operator(b,xyzt).cpu().numpy())
            s.append(model.stress_operator(b,xyzt).cpu().numpy())
    le=np.concatenate(le,axis=0)*np.asarray(ck['LE_std'],float)+np.asarray(ck['LE_mean'],float)
    s=np.concatenate(s,axis=0)*np.asarray(ck['S_std'],float)+np.asarray(ck['S_mean'],float)
    return le,s


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    dev=torch.device('mps' if args.device=='auto' and torch.backends.mps.is_available() else ('cpu' if args.device=='auto' else args.device))
    src=root/'results'/'v4_7region_direct'
    new=root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'
    manifest=new/'65Y_B_manifest.json'
    if not manifest.is_file():raise FileNotFoundError('65Y-B training not completed: '+str(manifest))
    details=json.loads(manifest.read_text())
    if not details.get('test_split_unused') or not details.get('frozen_displacement_and_force_heads'):
        raise RuntimeError('Training provenance does not establish clean test and frozen heads')
    stats=pd.read_csv(new/'65Y_B_validation_training_summary.csv')
    rows=[];gates={};verified_frozen=True
    for seg in REGIONS:
        local=root/'data'/'hybrid_bulk_7region'/seg
        split=pd.read_csv(local/'split_assignment.csv')
        ix=split.loc[split['Split']=='test','Index'].to_numpy(dtype=int)
        if not len(ix):raise RuntimeError(seg+' has no untouched test split')
        X=np.load(local/'branch_inputs.npy')[ix];ip=np.load(local/'ip_coordinates.npy')
        truth={'LE':np.load(local/'LE.npy')[ix],'S':np.load(local/'S.npy')[ix]}
        orig=[];cand=[]
        members=sorted(stats.loc[stats.segment==seg,'member'].astype(int).tolist())
        if not members:raise RuntimeError(seg+' missing candidate ensemble')
        for i in members:
            f1=src/seg/('member_%02d'%i)/'best_v4_7region.pt'
            f2=new/seg/('member_%02d'%i)/'best_v4_7region.pt'
            if not f1.is_file() or not f2.is_file():raise FileNotFoundError('Missing checkpoint '+str(f1 if not f1.is_file() else f2))
            if args.check_only:continue
            c0,m0=read_model(f1,dev);c1,m1=read_model(f2,dev)
            p=c1.get('65Y_finetune',{})
            if not p.get('train_split_only') or p.get('constitutive_loss_used') or p.get('kinematic_loss_used'):
                raise RuntimeError('Unverified candidate training provenance for '+seg)
            for k,v in c0['model_state_dict'].items():
                if k.startswith(HEADS) and not torch.equal(v,c1['model_state_dict'][k]):
                    verified_frozen=False
                    raise RuntimeError('Frozen displacement/interface force head differs: '+seg+' '+k)
            a,b=member_predictions(m0,c0,X,ip,dev)
            c,d=member_predictions(m1,c1,X,ip,dev)
            orig.append((a,b));cand.append((c,d))
            del m0,m1
        if args.check_only:continue
        original={'LE':np.mean(np.stack([k[0] for k in orig]),axis=0),
                  'S':np.mean(np.stack([k[1] for k in orig]),axis=0)}
        candidate={'LE':np.mean(np.stack([k[0] for k in cand]),axis=0),
                   'S':np.mean(np.stack([k[1] for k in cand]),axis=0)}
        for typ in ('LE','S'):
            for j,component in enumerate(COMPS):
                a=metrics(original[typ][...,j],truth[typ][...,j]);b=metrics(candidate[typ][...,j],truth[typ][...,j])
                rows.append({'region':seg,'field':typ,'component':typ+component,
                             'test_cases':len(ix),'n_members':len(members),
                             'baseline_rel_l2_pct':a['relative_L2_pct'],
                             'candidate_rel_l2_pct':b['relative_L2_pct'],
                             'baseline_RMSE':a['RMSE'],'candidate_RMSE':b['RMSE'],
                             'baseline_SSE':a['squared_error_sum'],'candidate_SSE':b['squared_error_sum'],
                             'error_SSE_reduction_pct':100*(a['squared_error_sum']-b['squared_error_sum'])/max(a['squared_error_sum'],1e-12)})
    if args.check_only:
        print('65Y-C CHECK PASS: both source and candidate ensembles, original held-out test split')
        return
    df=pd.DataFrame(rows)
    for seg in REGIONS:
        t=df[df.region==seg].set_index('component')
        target=['S22','LE22','S12','LE12']
        reductions=[float(t.loc[k,'error_SSE_reduction_pct']) for k in target]
        main_improved=sum(reductions)>0 and sum(v>0 for v in reductions)>=3
        protect=all(float(t.loc[k,'candidate_rel_l2_pct'])<=1.10*max(float(t.loc[k,'baseline_rel_l2_pct']),1e-8)
                    for k in ('S11','LE11'))
        gates[seg]={'target_SSE_reductions_pct':dict(zip(target,reductions)),
                    'three_of_four_targets_improved':sum(v>0 for v in reductions)>=3,
                    'target_sum_positive':sum(reductions)>0,
                    'S11_and_LE11_no_more_than_10pct_relative_regression':protect,
                    'preliminary_gate_passed':bool(main_improved and protect)}
    out=root/'results'/'65Y_middle_mechanics'/'C_holdout_comparison'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY this folder to rerun:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'65Y_C_test_comparison.csv',index=False)
    summary={'step':'65Y-C','cohort':'Original 64J untouched test cases, not 65W middle FEM reference',
             'candidate_models_are_offline_only':True,'baseline_models_untouched':True,
             'displacement_and_force_heads_identical':verified_frozen,'gates':gates,
             'all_preliminary_gates_passed':all(x['preliminary_gate_passed'] for x in gates.values()),
             'note':'This is a single comparison on previously unused test split. Do not tune another candidate based on its score and still call this a clean independent test.',
             'next_step':'If gates pass, evaluate candidate mechanics on a new independent material case, then preserve/verify interface coupling before deployment.'}
    (out/'65Y_C_summary.json').write_text(json.dumps(summary,indent=2))
    print('65Y-C COMPLETE (offline; no coupling weights changed):')
    print(df[df.component.isin(['S11','S12','S22','LE11','LE12','LE22'])][['region','component','baseline_rel_l2_pct','candidate_rel_l2_pct','error_SSE_reduction_pct']].to_string(index=False))
    print('PRELIMINARY PASS:',summary['all_preliminary_gates_passed'])
    print('Report:',out)

if __name__=='__main__':main()
