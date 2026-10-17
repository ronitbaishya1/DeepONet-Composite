#!/usr/bin/env python3
"""65X-C / Mac: combine 65X-A spatial errors and 65X-B physics screens into a
readable next-experiment recommendation. No model modification and no training.

Run after A and B:
  python3 65X_C_decision_report.py --project .
Rerun: manually delete results/65X_middle_diagnostics/C_decision_report only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

NO=('NO_OL','NO_L','NO_R','NO_OR')
COMPS=('S11','S12','S22','LE11','LE12','LE22')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    args=p.parse_args()
    root=Path(args.project).expanduser().resolve()
    base=root/'results'/'65X_middle_diagnostics'
    x=base/'A_spatial_errors'/'65X_A_region_component_metrics.csv'
    y=base/'B_physics_consistency'/'65X_B_reference_error_comparison.csv'
    for f in (x,y):
        if not f.is_file():raise FileNotFoundError('Run previous stage first: '+str(f))
    a=pd.read_csv(x)
    b=pd.read_csv(y)
    out=base/'C_decision_report'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output exists. Manually delete only this folder before rerunning:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    focus_tables={}
    for field in ('S','LE'):
        subset=a[a.region.isin(NO)&a.component.isin([c for c in COMPS if c.startswith(field)])].copy()
        subset=subset.sort_values('squared_error_sum',ascending=False).reset_index(drop=True)
        # Only rank squared errors within the SAME units: stresses MPa^2, strains dimensionless^2.
        subset['share_selected_'+field+'_error_pct']=100*subset.squared_error_sum/max(subset.squared_error_sum.sum(),1e-30)
        subset.to_csv(out/('65X_C_prioritized_'+field+'_components.csv'),index=False)
        focus_tables[field]=subset.head(8)
    contenders=[]
    for region in NO:
        for comp in COMPS:
            sub=b[(b.region==region)&(b.component==comp)]
            if len(sub)==0:raise RuntimeError('Missing region or component '+region+'/'+comp)
            raw_method='learned_S' if comp.startswith('S') else 'learned_LE'
            raw=sub[sub.method==raw_method]
            if len(raw)!=1:raise RuntimeError('Cannot find original head')
            raw_val=float(raw.iloc[0]['rel_l2_percent'])
            for _,r in sub.iterrows():
                if r['method']==raw_method:continue
                contenders.append(dict(region=region,component=comp,method=r['method'],
                                       baseline_rel_l2_percent=raw_val,
                                       candidate_rel_l2_percent=float(r['rel_l2_percent']),
                                       candidate_minus_baseline_pp=float(r['rel_l2_percent'])-raw_val,
                                       advisory='DIAGNOSTIC ONLY: the physics approximation is not a validated replacement for NLGEOM learned head'))
    pd.DataFrame(contenders).sort_values('candidate_minus_baseline_pp').to_csv(out/'65X_C_physics_candidate_comparison.csv',index=False)
    md=['# Step 65X: Spatial and physics consistency decision report','',
        '**Case:** middle material E1=46500, E2=12500, G12=4750 MPa.',
        '**Baseline:** saved converged 65V interface q and 65W-H seven-region NO/FEM prediction (UNCHANGED).']
    for field in ('S','LE'):
        md += ['', '## Largest absolute squared '+('stress' if field=='S' else 'strain')+' errors (within consistent units)',
               '', '| Rank | Region | Component | Relative L2 (%) | RMSE | Squared error sum |',
               '|---:|---|---|---:|---:|---:|']
        for i,(_,r) in enumerate(focus_tables[field].iterrows(),1):
            md.append('| %d | %s | %s | %.3f | %.5g | %.5g |'%(i,r.region,r.component,r.relative_l2_percent,r.rmse,r.squared_error_sum))
    md+=['','## Interpretation safeguards','',
         '- Compare **absolute errors and reference RMS** alongside relative percentages. Relative percentages can explode for low-magnitude transverse stresses.',
         '- 65X-B computes C @ LE and B @ U in a **linearized material-frame approximation**. Abaqus uses NLGEOM and logarithmic strain; therefore large consistency residuals are not definitive proof of incorrect learned physics.',
         '- Do not tune model weights, physics-loss hyperparameters, or blend coefficients on the same middle-material full-FEM reference reported here. Use independent validation cases from the 64J/64N training split.',
         '- The V4 network predicts displacement, LE, S with independent heads. Its 64N script permits --lambda_kin and --lambda_const, but their default values are zero.',
         '- If you retrain the operator, the previous **0.019523 high-fidelity equilibrium guarantee no longer automatically applies**. Re-equilibrate and verify all three Abaqus patches with the changed NO model.',
         '',
         '## Next controlled experiment (after reviewing data)','',
         '1. Inspect the 65X-A spatial maps for hot spots near the six artificial interfaces and roller/nose contact areas.',
         '2. Inspect 65X-B to see whether approximate B@U or C@LE agrees with full FEM better than the independent strain/stress heads. This is a diagnostic, not a selection rule.',
         '3. If model inconsistency appears important, train a **separate** 64N physics-penalty ablation using its EXISTING train/val split and warm-start checkpoints, without overwriting results/v4_7region_direct.',
         '4. Evaluate all 64N held-out test cases and the new middle-material case. For any replacement neural operator, rerun the seven-region interface equilibrium, do not reuse 65V q unverified.',
         '5. Save original 65W results as baseline for the paper.']
    (out/'65X_C_decision_report.md').write_text('\n'.join(md)+'\n')
    prioritized={field:[dict(region=str(r.region),component=str(r.component),
                                 rel_l2_percent=float(r.relative_l2_percent),
                                 squared_error_sum=float(r.squared_error_sum))
                      for _,r in focus_tables[field].iterrows()]
                 for field in ('S','LE')}
    (out/'65X_C_summary.json').write_text(json.dumps({
        'step':'65X-C','baseline_preserved':True,'num_NO_comparison_rows':len(contenders),
        'rankings_by_physical_units':prioritized,
        'next_action':'Review 65X-B and choose held-out training ablations, not an unverified change to the 65V converged NO models',
        'reports':['65X_C_decision_report.md','65X_C_prioritized_S_components.csv','65X_C_prioritized_LE_components.csv','65X_C_physics_candidate_comparison.csv']},indent=2))
    print('65X-C PASS. Read:',out/'65X_C_decision_report.md')
    for field in ('S','LE'):
        print(('Stress' if field=='S' else 'Strain')+' high-error components (ranked separately, consistent units):')
        print(focus_tables[field][['region','component','relative_l2_percent','squared_error_sum']].head(5).to_string(index=False))

if __name__=='__main__':main()
