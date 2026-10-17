#!/usr/bin/env python3
"""B5 internal development evaluation. NO model selection on external AJ cases.

Compare 65AA, old B1/B2 and four B5 normalized-output variants, both regions.
B5 evaluations use ONLY 12 internal development cases within original 64J TRAIN.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib
c=importlib.import_module('65AM_B5_common')
b=c.b
oldcompare=importlib.import_module('65AM_B3_compare')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    dev=b.CO.device_from_option(a.device)
    data={}
    for region in b.REGIONS:
        d,mu,sd=c.prepare(root,region)
        for variant in c.VARIANTS:
            for loss in c.LOSSES:
                c.load_model(root,variant,loss,region,d,mu,sd,dev)
        for arm in ('B1','B2'):
            oldfile=b.candidate_path(root,arm,region)
            if not oldfile.is_file():raise FileNotFoundError('Missing previous checkpoint: '+str(oldfile))
        b.check_reference(d['ds'],d['tune'])
        data[region]=(d,mu,sd)
        print('B5 COMPARE CHECK',region,'six models + baseline ready',flush=True)
    if a.check_only:
        print('65AM-B5 COMPARISON CHECK ONLY PASS. No outputs created.')
        return
    out=c.guarded_directory(c.report_path(root))
    rows=[];zones=[];cases=[];modes=[];decisions=[]
    for region,(d,mu,sd) in data.items():
        ids=d['tune'];ds=d['ds'];truth=ds['U'][ids];LE=ds['LE'][ids]
        preds={'65AA_frozen':b.baseline_predictions(root,region,ds,ids,dev)}
        for arm in ('B1','B2'):
            net,_=b.load_candidate(root,arm,region,d,dev)
            preds['old_'+arm]=b.predict_candidate(net,d,ids,dev)
        for variant in c.VARIANTS:
            for loss in c.LOSSES:
                net,ck=c.load_model(root,variant,loss,region,d,mu,sd,dev)
                U,coefs=c.predict(net,d,ids,dev)
                key=variant+'_'+loss
                preds[key]=U
                cref=c.coefficient_targets(d,ids)
                for k in range(d['rank']):
                    modes.append(dict(region=region,model=key,mode=k+1,
                        coeff_rmse=float(np.sqrt(np.mean((coefs[:,k]-cref[:,k])**2))),
                        rmse_div_fit_sd=float(np.sqrt(np.mean((coefs[:,k]-cref[:,k])**2))/sd[k]),
                        coeff_reference_mean=float(np.mean(cref[:,k])),
                        coeff_predicted_mean=float(np.mean(coefs[:,k])),fit_coeff_std=float(sd[k])))
        # Verify old B3 report, if present, to catch wrong checkpoints or data.
        old_csv=root/'results'/b.ROOT_NAME/'B3_comparison'/'B3_pooled_metrics.csv'
        previous=pd.read_csv(old_csv) if old_csv.is_file() else None
        masks=oldcompare.local_zones(ds)
        lift=d['lift'][len(d['fit']):]
        for name,U in preds.items():
            if np.max(np.abs(U[:,d['fixed'],:]-lift[:,d['fixed'],:]))>1.e-5:
                raise RuntimeError('Interface displacement mismatch: '+region+' '+name)
            m=b.metrics(U,truth,ds,LE)
            if previous is not None and name in ('old_B1','old_B2','65AA_frozen'):
                prevname={'old_B1':'B1_FE_residual','old_B2':'B2_FE_residual',
                          '65AA_frozen':'65AA_frozen_existing'}[name]
                old=previous[(previous.region==region)&(previous.model==prevname)]
                if len(old)!=1 or abs(float(old.iloc[0].grad_rel_pct)-m['grad_rel_pct'])>0.08:
                    raise RuntimeError('B5 baseline is not reproducing prior B3: '+region+' '+name)
            rows.append(dict(region=region,model=name,cohort='original64J_train_internal_tune',
                             n_cases=len(ids),**m))
            for k,cid in enumerate(ids):
                mcase=b.metrics(U[k:k+1],truth[k:k+1],ds,LE[k:k+1])
                cases.append(dict(region=region,model=name,case_index=int(cid),**mcase))
            Gp=b.AA.gradient_numpy(U,ds);Gr=b.AA.gradient_numpy(truth,ds)
            for zone,mask in masks.items():
                if not np.any(mask):continue
                zones.append(dict(region=region,model=name,zone=zone,
                    grad_rel_pct=b.CO.relative_l2(Gp[:,mask],Gr[:,mask]),
                    dU2dY_rel_pct=b.CO.relative_l2(Gp[:,mask,1,1],Gr[:,mask,1,1]),
                    n_elements=int(np.sum(mask))))
        dff=pd.DataFrame(rows);dfz=pd.DataFrame(zones)
        base=dff[(dff.region==region)&(dff.model=='65AA_frozen')].iloc[0]
        inner='right' if region=='inner_left' else 'left'
        bz=dfz[(dfz.region==region)&(dfz.model=='65AA_frozen')&(dfz.zone==inner)].iloc[0]
        for key in [variant+'_'+loss for variant in c.VARIANTS for loss in c.LOSSES]:
            now=dff[(dff.region==region)&(dff.model==key)].iloc[0]
            zn=dfz[(dfz.region==region)&(dfz.model==key)&(dfz.zone==inner)].iloc[0]
            gain=100*(base.grad_rel_pct-now.grad_rel_pct)/max(base.grad_rel_pct,1e-10)
            innergain=100*(bz.grad_rel_pct-zn.grad_rel_pct)/max(bz.grad_rel_pct,1e-10)
            safeU=now.U_vec_rel_pct<=1.05*base.U_vec_rel_pct
            safeLE=now.LE22_rel_pct<base.LE22_rel_pct
            decisions.append(dict(region=region,model=key,
                baseline_grad_pct=float(base.grad_rel_pct),candidate_grad_pct=float(now.grad_rel_pct),
                overall_gradient_gain_pct=float(gain),inner_zone_gradient_gain_pct=float(innergain),
                baseline_U_pct=float(base.U_vec_rel_pct),candidate_U_pct=float(now.U_vec_rel_pct),
                baseline_LE22_pct=float(base.LE22_rel_pct),candidate_LE22_pct=float(now.LE22_rel_pct),
                provisional_dev_gate=bool(gain>=10 and innergain>=15 and safeU and safeLE)))
        plotrows=dff[dff.region==region]
        fig,ax=plt.subplots(figsize=(11,4.5))
        ax.bar(range(len(plotrows)),plotrows.grad_rel_pct.to_numpy())
        ax.set_xticks(range(len(plotrows)),plotrows.model.to_list(),rotation=30,ha='right')
        ax.set_ylabel('Relative full-gradient error (%)')
        ax.set_title(region+' | Internal development only')
        fig.tight_layout();fig.savefig(out/(region+'_B5_gradient_comparison.png'),dpi=155);plt.close(fig)
        print('B5',region,'gradient:', {q.model:round(q.grad_rel_pct,3) for _,q in plotrows.iterrows()},flush=True)
    pd.DataFrame(rows).to_csv(out/'B5_pooled_metrics.csv',index=False)
    pd.DataFrame(zones).to_csv(out/'B5_zone_metrics.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'B5_case_metrics.csv',index=False)
    pd.DataFrame(modes).to_csv(out/'B5_mode_errors.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'B5_decisions.csv',index=False)
    (out/'B5_summary.json').write_text(json.dumps(dict(step='65AM-B5',
        cohort='44 original64J TRAIN fit + 12 original64J TRAIN internal tune',
        new_models='centered_U, centered_grad, standardized_U, standardized_grad',
        original_B1_B2_and_65AA_untouched=True,source_65V_untouched=True,
        independent_test=False,original_64J_validation_test_unused=True,AJ_external_unused=True,
        note='Development comparison only. 65AA baseline was pretrained on all 56 original TRAIN cases, including tune.',
        decisions=decisions),indent=2))
    print('65AM-B5 COMPARE COMPLETE:',out)
    print('Send B5_pooled_metrics.csv, B5_zone_metrics.csv, B5_mode_errors.csv, B5_decisions.csv, B5_summary.json and 2 figures')

if __name__=='__main__':main()
