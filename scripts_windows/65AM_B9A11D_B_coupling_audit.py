#!/usr/bin/env python3
"""B9A11D-B: inspect saved 65V convergence records without running Abaqus.
Run on Windows server or Mac against a copy of the 65V results.
Never claims that additional solver iterations will reduce NO model error.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',required=True,help='65V output folder (online_results_7region_65V_middle)')
    p.add_argument('--out',default='B9A11D_65V_audit.json')
    a=p.parse_args(); root=Path(a.folder).resolve()
    if not root.is_dir():raise SystemExit('Missing 65V directory: '+str(root))
    found=[]
    for path in sorted(root.rglob('*.csv')):
        try:
            with path.open(newline='') as f:
                rd=csv.DictReader(f);fields=rd.fieldnames or []
                metric=[z for z in fields if any(k in z.lower() for k in ('rms','residual','objective'))]
                seq=[z for z in fields if any(k in z.lower() for k in ('iter','step','job'))]
                if not metric:continue
                rows=list(rd)
                if not rows:continue
                found.append(dict(file=str(path.relative_to(root)),rows=len(rows),iteration_fields=seq,metric_fields=metric,
                                  start={k:rows[0].get(k) for k in metric},end={k:rows[-1].get(k) for k in metric}))
        except (UnicodeError, OSError,ValueError):continue
    npz=[]
    for path in sorted(root.rglob('*.npz')):
        try:
            with np.load(path,allow_pickle=False) as f:
                keys=[k for k in f.files if any(z in k.lower() for z in ('rms','residual','converg'))]
                if keys: npz.append(dict(file=str(path.relative_to(root)),keys=keys))
        except (OSError,ValueError,KeyError):continue
    report={'source_directory':str(root),'csv_convergence_candidates':found,'npz_residual_candidates':npz,
            'interpretation':'Inventory only. Do not equate normalized 65V residual with global FEM field error.',
            'next_step':'Identify verified 65V iteration log and independent full-FEM comparison before changing tolerance.'}
    outfile=Path(a.out).resolve()
    if outfile.exists():raise SystemExit('Protected output exists: '+str(outfile))
    outfile.write_text(json.dumps(report,indent=2))
    print('Found',len(found),'CSV candidates and',len(npz),'NPZ candidates')
    for x in found:print(x['file'],x['metric_fields'],x['rows'])
    print('Saved:',outfile)

if __name__=='__main__':main()
