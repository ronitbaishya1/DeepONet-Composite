#!/usr/bin/env python3
"""65Z-C: Compare 65Y and 65Z candidate V4 ensembles using ORIGINAL VALIDATION ONLY.

The 65Y-C original test split has already been inspected. It is NOT used again
here for selecting the 65Z physics weight or candidate. No Abaqus/full 65W fields.

  python3 65Z_C_validation_decision.py --project . --arm constitutive --check-only
  python3 65Z_C_validation_decision.py --project . --arm constitutive

Output: results/65Z_nonlinear_mechanics/C_constitutive_validation/
"""
import argparse,json,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import pandas as pd
import torch

sp=spec_from_file_location('s65zcommon',Path(__file__).with_name('65Z_common.py'))
co=module_from_spec(sp);sys.modules[sp.name]=co;sp.loader.exec_module(co)


def predictions(root,ds,ck,model,device,idx,batch=8):
    b,_,ip=co.scaled_inputs(ck,ds)
    ip_t=torch.tensor(ip,dtype=torch.float32,device=device)
    LE=[];S=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(idx),batch):
            ids=idx[start:start+batch]
            bn=torch.tensor(b[ids],dtype=torch.float32,device=device)
            LE.append(model.strain_operator(bn,ip_t).cpu().numpy())
            S.append(model.stress_operator(bn,ip_t).cpu().numpy())
    le=np.concatenate(LE,axis=0)*np.asarray(ck['LE_std'])+np.asarray(ck['LE_mean'])
    stress=np.concatenate(S,axis=0)*np.asarray(ck['S_std'])+np.asarray(ck['S_mean'])
    return {'LE':le.astype(np.float64),'S':stress.astype(np.float64)}


def report_metrics(pred,truth):
    d=pred-truth
    return {'sse':float(np.sum(d*d)),
            'rel_pct':co.relative_l2(pred,truth),
            'rmse':float(np.sqrt(np.mean(d*d)))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    parser.add_argument('--arm',choices=['constitutive','combined'],default='constitutive')
    parser.add_argument('--members',type=int,default=5)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.members<=5:raise ValueError('Members must be between 1 and 5')
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    device=co.device_from_option(args.device)
    ck65y=root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'
    ck65z=root/'results'/'65Z_nonlinear_mechanics'/('B_'+args.arm)
    if not (ck65z/'65Z_B_manifest.json').is_file():
        raise FileNotFoundError('65Z-B arm not finished: '+str(ck65z))
    manifest=json.loads((ck65z/'65Z_B_manifest.json').read_text())
    if (manifest.get('arm')!=args.arm or not manifest.get('train_and_validation_only')
        or not manifest.get('displacement_and_force_heads_frozen_and_equal')):
        raise RuntimeError('Unverified 65Z-B manifest')
    plans=[]
    for seg in co.SEGMENTS:
        ds=co.segment_data(root,seg)
        for i in range(1,args.members+1):
            a=ck65y/seg/('member_%02d'%i)/'best_v4_7region.pt'
            b=ck65z/seg/('member_%02d'%i)/'best_v4_7region.pt'
            for x in (a,b):
                if not x.is_file():raise FileNotFoundError('Missing '+str(x))
            plans.append((seg,i,a,b))
    if args.check_only:
        print('65Z-C CHECK PASS: %d 65Y/65Z checkpoint pairs and untouched validation tables'%len(plans))
        print('Original TEST split is intentionally NOT evaluated')
        return
    rows=[];gates={};consistent=[]
    for seg in co.SEGMENTS:
        ds=co.segment_data(root,seg)
        ix=ds['splits']['validation']
        C=co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][ix,:3]).astype(np.float64)
        predictions={'65Y':[],'65Z':[]}
        for _seg,m,path65y,path65z in [p for p in plans if p[0]==seg]:
            old=co.load_checkpoint(path65y);new=co.load_checkpoint(path65z)
            prov=new.get('65Z_physics_finetune',{})
            if prov.get('arm')!=args.arm or prov.get('source_65Y_sha256')!=co.sha256(path65y):
                raise RuntimeError('Source checkpoint differs or provenance missing for '+str(path65z))
            for key,tensor in old['model_state_dict'].items():
                if key.startswith(co.FREEZE) and not torch.equal(tensor,new['model_state_dict'][key]):
                    raise RuntimeError('Frozen U/interface-force head changed: '+key)
            # Same head statistics must be used for meaningful comparisons.
            for field in ('branch_mean','branch_std','LE_mean','LE_std','S_mean','S_std','U_mean','U_std'):
                if not np.array_equal(np.asarray(old[field]),np.asarray(new[field])):
                    raise RuntimeError('Changed normalization '+field)
            mod_old=co.make_model(old,device);mod_new=co.make_model(new,device)
            predictions['65Y'].append(predictions_for_segment(root,ds,old,mod_old,device,ix))
            predictions['65Z'].append(predictions_for_segment(root,ds,new,mod_new,device,ix))
            del mod_old,mod_new
            if device.type=='mps':torch.mps.empty_cache()
        ensemble={which:{name:np.mean(np.stack([z[name] for z in members]),axis=0)
                         for name in ('LE','S')} for which,members in predictions.items()}
        for field in ('LE','S'):
            truth=ds[field][ix].astype(np.float64)
            for j,c in enumerate(co.COMPONENTS):
                a=report_metrics(ensemble['65Y'][field][:,:,j],truth[:,:,j])
                b=report_metrics(ensemble['65Z'][field][:,:,j],truth[:,:,j])
                rows.append({'region':seg,'field':field,'component':field+c,
                             'validation_cases':len(ix),'ensemble_members':args.members,
                             '65Y_rel_l2_pct':a['rel_pct'],'65Z_rel_l2_pct':b['rel_pct'],
                             '65Y_RMSE':a['rmse'],'65Z_RMSE':b['rmse'],
                             '65Y_SSE':a['sse'],'65Z_SSE':b['sse'],
                             '65Z_SSE_reduction_pct':100*(a['sse']-b['sse'])/max(a['sse'],1e-12)})
        # Compare self-consistency independently of reference goodness.
        for which in ('65Y','65Z'):
            le=ensemble[which]['LE'];s=ensemble[which]['S'];constitutive=np.einsum('nec,ndc->ned',le,C)
            r=report_metrics(constitutive,s)
            consistent.append({'region':seg,'model':which,
                               'constitutive_head_disagreement_rel_pct':r['rel_pct'],
                               'constitutive_head_disagreement_RMSE_MPa':r['rmse'],
                               'constitutive_head_disagreement_SSE':r['sse']})
    df=pd.DataFrame(rows)
    for seg in co.SEGMENTS:
        t=df[df.region==seg].set_index('component')
        target=['S22','LE22','S12','LE12']
        vals=[float(t.loc[k,'65Z_SSE_reduction_pct']) for k in target]
        protects={k:float(t.loc[k,'65Z_rel_l2_pct'])<=1.10*max(float(t.loc[k,'65Y_rel_l2_pct']),1e-9)
                  for k in ('S11','LE11')}
        gates[seg]={'target_error_reduction_pct':dict(zip(target,vals)),
            'three_of_four_targets_improved':sum(v>0 for v in vals)>=3,
            'total_target_error_reduction_positive':sum(vals)>0,
            'S11_LE11_protected':all(protects.values()),
            'preliminary_validation_gate_passed':bool(sum(v>0 for v in vals)>=3 and sum(vals)>0 and all(protects.values()))}
    out=root/'results'/'65Z_nonlinear_mechanics'/('C_'+args.arm+'_validation')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Protected output. Manually delete ONLY:\n'+str(out)+'\nThen rerun.')
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'65Z_C_validation_comparison.csv',index=False)
    pd.DataFrame(consistent).to_csv(out/'65Z_C_constitutive_consistency.csv',index=False)
    summary={'step':'65Z-C','arm':args.arm,'evaluated':'Original 64J development validation only',
       'original_65Y_test_split_already_used_before_65Z':True,
       'new_independent_test_still_required':True,
       'baseline_65Y_models_untouched':True,
       'displacement_and_force_weights_identical':True,
       'number_compared_models_per_region':args.members,
       'regions':gates,'all_preliminary_validation_gates_passed':all(x['preliminary_validation_gate_passed'] for x in gates.values()),
       'note':'A reduced constitutive discrepancy does NOT prove a reduced FEM-reference stress error. Do not deploy before fresh independent material and coupled-interface verification.'}
    (out/'65Z_C_summary.json').write_text(json.dumps(summary,indent=2))
    print('\n65Z-C VALIDATION COMPARISON:')
    print(df[df.component.isin(['S11','S12','S22','LE11','LE12','LE22'])][
          ['region','component','65Y_rel_l2_pct','65Z_rel_l2_pct','65Z_SSE_reduction_pct']].to_string(index=False))
    print('Regional gates:',{k:v['preliminary_validation_gate_passed'] for k,v in gates.items()})
    print('Do not claim independent test or new 65V equilibrium verification.')
    print('Report:',out/'65Z_C_summary.json')


def predictions_for_segment(root,ds,ck,model,device,ix):
    return predictions(root,ds,ck,model,device,ix)

if __name__=='__main__':main()
