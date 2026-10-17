#!/usr/bin/env python3
"""65AM-B3: 12-case INTERNAL DEVELOPMENT comparison; no new external test.

 python3 65AM_B3_compare.py --project . --check-only
 python3 65AM_B3_compare.py --project .
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
b=importlib.import_module('65AM_B_common')


def local_zones(ds):
    xyz=np.asarray(ds['ip_coordinates'],float)
    xmin,xmax=float(xyz[:,0].min()),float(xyz[:,0].max())
    threshold=.6 # mm near axial ends; original 65AK convention
    return dict(left=xyz[:,0] <= xmin+threshold,right=xyz[:,0] >= xmax-threshold,
                interior=(xyz[:,0]>xmin+threshold)&(xyz[:,0]<xmax-threshold))


def metrics_by_case(U,gt,ds,ids,method,region):
    rows=[]
    for k,c in enumerate(ids):
        m=b.metrics(U[k:k+1],ds['U'][c:c+1],ds,gt[k:k+1])
        rows.append(dict(region=region,case_index=int(c),model=method,**m))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args();root=Path(a.project).expanduser().resolve()
    dev=b.CO.device_from_option(a.device)
    datas={}
    for region in b.REGIONS:
        d=b.load_saved(root,region)
        for arm in ('B1','B2'):
            p=b.candidate_path(root,arm,region)
            if not p.is_file():raise FileNotFoundError('First train '+arm+': '+str(p))
        datas[region]=d
        print('CHECK PASS B3',region,'internal tune cases',len(d['tune']),'rank',d['rank'],flush=True)
    if a.check_only:
        print('65AM-B3 CHECK ONLY complete; no outputs created')
        return
    out=b.output_dir(root,'B3_comparison')
    rows=[]; pooled=[];zone_rows=[]; summary=[]
    for region,d in datas.items():
        ds=d['ds'];ids=d['tune'];LE=ds['LE'][ids]
        candidates={'65AA_frozen_existing':b.baseline_predictions(root,region,ds,ids,dev)}
        for arm in ('B1','B2'):
            model,ck=b.load_candidate(root,arm,region,d,dev)
            candidates[arm+'_FE_residual']=b.predict_candidate(model,d,ids,dev)
        zone=local_zones(ds)
        nz=len(zone['interior']);fixed=d['fixed']
        for name,U in candidates.items():
            ref=ds['U'][ids]
            res=b.metrics(U,ref,ds,LE)
            inferred_lift=d['lift'][len(d['fit']):]
            interface_gap=float(np.max(abs(U[:,fixed,:]-inferred_lift[:,fixed,:])))
            if interface_gap>1e-5:
                raise RuntimeError('Interface mismatch for '+name+': '+str(interface_gap))
            pooled.append(dict(region=region,model=name,internal_development=True,
                               interface_abs_max_mm=interface_gap,**res))
            rows.extend(metrics_by_case(U,LE,ds,ids,name,region))
            GP=b.AA.gradient_numpy(U,ds);GR=b.AA.gradient_numpy(ref,ds)
            for z,m in zone.items():
                if not np.any(m):continue
                mobj=b.CO.relative_l2(GP[:,m],GR[:,m])
                zone_rows.append(dict(region=region,model=name,zone=z,grad_rel_pct=mobj,
                                      dU2dY_rel_pct=b.CO.relative_l2(GP[:,m,1,1],GR[:,m,1,1]),
                                      dU2dY_rmse=float(np.sqrt(np.mean((GP[:,m,1,1]-GR[:,m,1,1])**2))),
                                      elements=int(np.sum(m))))
        old=next(v for v in pooled if v['region']==region and v['model']=='65AA_frozen_existing')
        new=next(v for v in pooled if v['region']==region and v['model']=='B2_FE_residual')
        def near_inner(mod):
            side='right' if region=='inner_left' else 'left'
            return next(v for v in zone_rows if v['region']==region and v['model']==mod and v['zone']==side)
        oldnear=near_inner('65AA_frozen_existing');newnear=near_inner('B2_FE_residual')
        gain=100*(old['grad_rel_pct']-new['grad_rel_pct'])/max(old['grad_rel_pct'],1e-12)
        neargain=100*(oldnear['grad_rel_pct']-newnear['grad_rel_pct'])/max(oldnear['grad_rel_pct'],1e-12)
        safeU=new['U_vec_rel_pct']<=1.05*old['U_vec_rel_pct']
        safeLE=new['LE22_rel_pct']<old['LE22_rel_pct']
        gate=bool(gain>=10 and neargain>=15 and safeU and safeLE)
        summary.append(dict(region=region,baseline_grad_pct=old['grad_rel_pct'],
            candidate_B2_grad_pct=new['grad_rel_pct'],overall_grad_gain_pct=gain,
            inner_zone_grad_gain_pct=neargain,baseline_U_pct=old['U_vec_rel_pct'],
            B2_U_pct=new['U_vec_rel_pct'],baseline_LE22_pct=old['LE22_rel_pct'],
            B2_LE22_pct=new['LE22_rel_pct'],provisional_development_gate=gate,
            note='Baseline 65AA pretrained on all 56 original TRAIN cases, including 12 internal tuning cases. Not independent.'))
        plt.figure(figsize=(7.4,4.6))
        names=['65AA_frozen_existing','B1_FE_residual','B2_FE_residual']
        vals=[next(v for v in pooled if v['region']==region and v['model']==n)['grad_rel_pct'] for n in names]
        plt.bar(['65AA','B1 U-only','B2 Sobolev'],vals)
        plt.ylabel('Full displacement-gradient relative L2 (%)')
        plt.title(region+' | Original-TRAIN internal development only')
        plt.tight_layout();plt.savefig(out/(region+'_gradient_bar.png'),dpi=155);plt.close()
        print('B3',region,'gradient gain B2 vs frozen',round(gain,2),'%',
              'inner-facing gain',round(neargain,2),'%',
              'provisional gate',gate,flush=True)
    pd.DataFrame(rows).to_csv(out/'B3_case_metrics.csv',index=False)
    pd.DataFrame(pooled).to_csv(out/'B3_pooled_metrics.csv',index=False)
    pd.DataFrame(zone_rows).to_csv(out/'B3_zone_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'B3_decisions.csv',index=False)
    (out/'B3_summary.json').write_text(json.dumps(dict(step='65AM-B3',cohort='44 fit / 12 tune, all from ORIGINAL 64J TRAIN',
        selected_model_epochs_using_tune=True,
        independent_evaluation=False,original_validation_test_unused=True,
        external_AJ_cases_unused=True,original_65V_unchanged=True,
        note='Results are internal development only, with known pretraining overlap for 65AA baseline.',
        decisions=summary),indent=2))
    print('65AM-B3 COMPLETE:',out)

if __name__=='__main__':main()
