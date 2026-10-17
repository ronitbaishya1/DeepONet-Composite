#!/usr/bin/env python3
"""65AM-C: Write descriptive audit summary, not a new model selection.
python3 65AM_C_report.py --project .
"""
import argparse, importlib, json
from pathlib import Path
import numpy as np,pandas as pd
am=importlib.import_module('65AM_common')


def explanation(base,comp,zone):
    effect=base['mean_hard_effect_pct']
    if effect<-3.0:
        first='The Hermite correction increases frozen gradient RMSE on this already examined cohort.'
    elif effect>3.0:
        first='The Hermite correction reduces frozen gradient RMSE on this already examined cohort.'
    else:
        first='The Hermite correction changes total gradient RMSE little on this already examined cohort.'
    rel=base['mean_PCA_only_gradient_pct']
    if rel<0.5:
        second='The PCA-only control is small relative to FEM gradients, suggesting PCA truncation alone is not the dominant explanation.'
    else:
        second='The PCA-only gradient effect is non-negligible and needs separate interface-basis investigation.'
    return first+' '+second+' Results are descriptive and not causal proof.'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    args=p.parse_args();root=Path(args.project).expanduser().resolve()
    prev=root/'results'/am.OUTROOT/'A_attribution'
    req=['65AM_A_case_metrics.csv','65AM_A_component_metrics.csv','65AM_A_zone_metrics.csv','65AM_A_x_profiles.csv','65AM_A_summary.json']
    for f in req:
        if not (prev/f).is_file():raise FileNotFoundError(str(prev/f))
    cases=pd.read_csv(prev/req[0]);components=pd.read_csv(prev/req[1]);zones=pd.read_csv(prev/req[2]);profile=pd.read_csv(prev/req[3])
    out=am.output(root,'C_report')
    rows=[];notes=[]
    for region in am.REGIONS:
        case=cases[cases.region==region]; comp=components[components.region==region]; zone=zones[zones.region==region]
        if len(case)!=6:raise RuntimeError('Expected six cases for '+region)
        row=dict(region=region,n_cases=len(case),
                 mean_raw_gradient_pct=float(case.grad_raw_error_pct.mean()),
                 mean_hard_gradient_pct=float(case.grad_hard_error_pct.mean()),
                 mean_PCA_only_gradient_pct=float(case.grad_pca_only_error_pct.mean()),
                 mean_raw_gradient_RMSE=float(case.grad_raw_RMSE.mean()),
                 mean_hard_gradient_RMSE=float(case.grad_hard_RMSE.mean()),
                 mean_hard_effect_pct=float(case.hard_gradient_effect_pct.mean()),
                 Hermite_worsened_cases=int((case.hard_minus_raw_RMSE>0).sum()),
                 PCA_only_vs_Hermite_gradient_RMS_ratio=float(case.pca_only_correction_gradient_RMS.mean()/max(case.hermite_correction_gradient_RMS.mean(),1e-15)))
        for component in ('U2_dY','U1_dY','U2_dX'):
            compdf=comp[comp.component==component]
            if len(compdf)!=6:raise RuntimeError('Missing '+component+' metrics for '+region)
            row[component+'_raw_RMSE']=float(compdf.raw_RMSE.mean())
            row[component+'_hard_RMSE']=float(compdf.hard_RMSE.mean())
        for name in ('left_end','right_end','interior'):
            z=zone[zone.zone==name]
            row[name+'_raw_gradient_pct']=float(z.grad_raw_error_pct.mean())
            row[name+'_hard_gradient_pct']=float(z.grad_hard_error_pct.mean())
        rows.append(row)
        notes.append(dict(region=region,interpretation=explanation(row,comp,zone)))
    pd.DataFrame(rows).to_csv(out/'65AM_C_region_summary.csv',index=False)
    report=dict(step='65AM-C',status='offline frozen gradient attribution only',summary=notes,
        controls={'raw':'Original 65AA displacement network output without Hermite correction',
                  'hard':'Exactly reproduced Step 65AK frozen hard-compatible output',
                  'pca_only':'Ground-truth FEM U with the same PCA interface reconstruction; oracle control only'},
        caveats=['Previously studied six external 65AJ cases; not a new holdout',
                 'Gradient and PCA-only comparison do not independently validate online seven-region traction equilibrium',
                 'No retraining or hyperparameter choice using these six cases',
                 'Projection affects boundary displacement; it does not prove the raw model or Hermite extension is uniquely at fault'],
        next_decision='If correction amplifies errors: inspect Hermite lift/derivative representation on original training/development cases. Otherwise focus raw NO gradient or other field modeling, using original training/development cases for selection.')
    am.saved_json(out/'65AM_C_summary.json',report)
    lines=['# 65AM-A: Frozen interface-gradient attribution report','','Six AJ_new cases already studied in Steps 65AJ/65AK/65AL. This report is explanatory, NOT independent validation.','']
    for r,n in zip(rows,notes):
        lines.extend(['## '+r['region'],
                      '- Raw gradient error: %.3f%%'%r['mean_raw_gradient_pct'],
                      '- Hard-corrected gradient error: %.3f%%'%r['mean_hard_gradient_pct'],
                      '- PCA-only gradient error: %.4f%%'%r['mean_PCA_only_gradient_pct'],
                      '- Hermite gradient RMSE reduction (positive is improvement): %.2f%%'%r['mean_hard_effect_pct'],
                      '- Number of six cases whose gradient RMSE worsened: %d'%r['Hermite_worsened_cases'],
                      '- Left/right/interior: raw %.3f/%.3f/%.3f%%, hard %.3f/%.3f/%.3f%%'%(
                        r['left_end_raw_gradient_pct'],r['right_end_raw_gradient_pct'],r['interior_raw_gradient_pct'],
                        r['left_end_hard_gradient_pct'],r['right_end_hard_gradient_pct'],r['interior_hard_gradient_pct']),
                      '',n['interpretation'],''])
    lines.extend(['## Research safeguards','No new neural operator was trained; source checkpoints and working 65V were not changed.',
                  'Do not tune on AJ_new_01..06. Future model selection requires development data, and future generalization claims require freshly held-out FEM cases.'])
    (out/'65AM_C_report.md').write_text('\n'.join(lines)+'\n')
    print('65AM-C DONE:',out)
    for r in rows:
        print(r['region'],'raw %.3f%% -> Hermite %.3f%%; PCA only %.4f%%'%(
            r['mean_raw_gradient_pct'],r['mean_hard_gradient_pct'],r['mean_PCA_only_gradient_pct']))

if __name__=='__main__':main()
