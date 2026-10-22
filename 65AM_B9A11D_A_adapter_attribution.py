#!/usr/bin/env python3
"""B9A11D-A: read-only internal comparison. Run from repository root.
Does not train, select new checkpoints, or overwrite existing results.
"""
import argparse, json, csv
from pathlib import Path

METRICS = ('displacement_pct','free_residual_pct','additional_free_force_pct','reference_interface_force_difference_pct')
ARMS = ('B5_baseline','supervised','floor','force_match')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--summary',default='results/65AM_B9_coupled_feasibility/B9A11C_weight_convergence/full01/B9A11C_summary.json')
    p.add_argument('--out',default='results/65AM_B9A11D_error_attribution')
    a=p.parse_args()
    root=Path(a.project).resolve(); source=root/a.summary
    if not source.is_file(): raise SystemExit('Missing summary: '+str(source))
    d=json.loads(source.read_text())
    if d.get('step')!='B9A11C' or d.get('independent_validation') is not False: raise SystemExit('Unexpected dataset/provenance')
    out=root/a.out
    out.mkdir(parents=True,exist_ok=True)
    result=out/'D1_internal_adapter_comparison.csv'
    if result.exists(): raise SystemExit('Protected output exists: '+str(result))
    rows=[]
    for region,s in d['results'].items():
        tune=s['internal_tune_metrics_diagnostic_only']; base=tune['B5_baseline']
        for arm in ARMS:
            for m in METRICS:
                val=float(tune[arm][m]); ref=float(base[m]);
                rows.append(dict(region=region,model=arm,metric=m,value_pct=val,
                                 B5_value_pct=ref,absolute_gain_percentage_points=ref-val,
                                 relative_gain_percent=100*(ref-val)/max(abs(ref),1e-12)))
    with result.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('INTERNAL TUNE ONLY — NOT INDEPENDENT TEST')
    for r in rows:
        if r['metric'] in ('displacement_pct','free_residual_pct'):
            print('{region:12} {model:12} {metric:20} {value_pct:9.5f}%  vs B5 {relative_gain_percent:+7.2f}%'.format(**r))
    print('Validation gates:',{k:v.get('screening_pass_on_fit_validation') for k,v in d['results'].items()})
    print('Coupled approval:',d.get('approved_for_coupled_use'))
    print('Saved:',result)

if __name__=='__main__':main()
