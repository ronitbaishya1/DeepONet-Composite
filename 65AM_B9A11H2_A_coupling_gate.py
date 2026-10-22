#!/usr/bin/env python3
"""Read-only deployment gate for B9A11G force and field results.

This script NEVER deploys B5, adjusts solver weights, or changes 65V.
G force-proxy mode RMS is not an identical metric to the 65V coupled RMS;
its size is a risk screen, not proof of a failed coupled solve.
"""
import argparse, json, math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='run01',help='Existing B9A11G run tag')
    p.add_argument('--out-tag',default='run01')
    a=p.parse_args()
    root=Path(a.project).expanduser().resolve()
    base=root/'results'/'65AM_B9A11G_fast_track'/a.tag
    fp=base/'rf'/'B9A11G_true_Abaqus_RF_summary.json'
    up=base/'preview'/'B9A11G_fixedq_summary.json'
    for f in (fp,up):
        if not f.is_file():raise FileNotFoundError('Missing existing result: '+str(f))
    force=json.loads(fp.read_text());preview=json.loads(up.read_text())
    if force.get('step')!='B9A11G-RF' or preview.get('step')!='B9A11G-preview':
        raise RuntimeError('Wrong stage output. Refuse a deployment decision.')
    if force.get('forces_are_actual_Abaqus_RF') is not True or force.get('calibrated_hourglass_proxy_still_approximate') is not True:
        raise RuntimeError('Expected actual Abaqus RF and approximate force proxy provenance')
    if preview.get('not_re_equilibrated') is not True or preview.get('not_full_B5_coupled') is not True:
        raise RuntimeError('Unexpected preview provenance. Inspect manually.')
    rows=[]
    for name in ('inner_left','inner_right'):
        d=force['result'][name]
        vals=[float(d[k]) for k in ('B5_force_RMS_mean','B5_force_RMS_worst','FEM_U_proxy_floor_mean')]
        if any(not math.isfinite(v) or v<0 for v in vals):raise RuntimeError('Nonfinite/negative RMS')
        rows.append({'region':name,'cases':int(d['cases']),'B5_force_RMS_mean':vals[0],
                     'B5_force_RMS_worst':vals[1],'FEM_U_proxy_floor_mean':vals[2],
                     'below_65V_target_0p02':vals[0]<=.02 and vals[1]<=.02})
    audit={'step':'B9A11H-A','B5_coupling_deployment_approved':False,
           'run_new_B5_coupled_jobs_now':False,'gate':'HOLD',
           'baseline_65V_coupled_RMS':float(preview['original_65V_residual_rms']),
           'force_mode_diagnostic':rows,
           'field_gain_fixed_q':{
              k:{'V4_rel_l2_pct':float(preview['global_metrics'][k]['original_v4']['rel_l2_pct']),
                 'B5_fixed_q_rel_l2_pct':float(preview['global_metrics'][k]['B5_frozen_q']['rel_l2_pct'])}
              for k in ('U','U2','LE22','S22')},
           'reason':'B5 force transfer was not independently validated to the original solver force model; fixed-q preview is not re-equilibrated. Do not start a costly coupled B5 solve yet.',
           'caveat':'G RMS compares approximate B5 projected forces to actual Abaqus RF on independent subdomains, not the identical 65V 26-mode residual. Source-force sign/frame and output equivalence must be audited before coupling.',
           'next':'Start material inverse/UQ with 50 full-FEM parameterized simulations; preserve 65V as verification baseline. Coupled B5 remains a future gate.'}
    target=root/'results'/'65AM_B9A11H_v2'/a.out_tag
    target.mkdir(parents=True,exist_ok=True)
    dest=target/'B9A11H_A_coupling_gate.json'
    if dest.exists():raise RuntimeError('Refusing overwrite: '+str(dest))
    dest.write_text(json.dumps(audit,indent=2))
    print(json.dumps({'gate':'HOLD','B5_coupling_deployment_approved':False,'force_rows':rows,'file':str(dest)},indent=2))

if __name__=='__main__':main()
