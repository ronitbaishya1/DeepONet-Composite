#!/usr/bin/env python3
"""B9A11D-C: read-only diagnosis of the already-solved 65V case.

Run from ODBS on Windows:
  abaqus python scripts\65AM_B9A11D_C_65V_diagnostics.py \
    --project "C:\\Users\\rrb225_lehigh\\Pictures\\ODBS"

No Abaqus simulation is launched. Does not change B5, 65V, or old results.
Uses ONLY saved verified solver residuals; these are NOT full-FEM field errors.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

NAMES = ('left_outer', 'left_inner', 'center_left', 'center_right',
         'right_inner', 'right_outer')
ENDS = (0, 4, 8, 13, 18, 22, 26)
DEFAULT_WIN_ROOT = r'C:\Users\rrb225_lehigh\Pictures\ODBS'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def as_float(val):
    try:
        if val is None or str(val).strip() == '':
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('Cannot write an empty table')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(project, out):
    folder = project / 'online_results_7region_65V_middle'
    run_folder = folder / '65V_D_lm_convergence'
    manifest = read_json(run_folder / 'rom_solution_manifest.json')
    history = read_csv(run_folder / '65V_D_history.csv')
    npz_path = run_folder / 'rom_solution_7region.npz'
    if not npz_path.is_file():
        raise FileNotFoundError('Saved 65V solved state not found: ' + str(npz_path))
    with np.load(str(npz_path), allow_pickle=False) as z:
        residual = np.asarray(z['normalized_residual'], dtype=float).ravel()
        recorded_rms = float(np.asarray(z['rms_residual']).ravel()[0])
    if residual.shape != (26,) or not np.all(np.isfinite(residual)):
        raise RuntimeError('Expected 26 finite interface-mode residuals')
    rms = float(np.sqrt(np.mean(residual**2)))
    if not np.isclose(rms, recorded_rms, rtol=1e-7, atol=1e-10):
        raise RuntimeError('Saved residual vector disagrees with stored RMS')
    if not manifest:
        raise FileNotFoundError('65V manifest not found: ' + str(run_folder / 'rom_solution_manifest.json'))
    declared_best = float(manifest['best_verified_rms'])
    if not np.isclose(rms, declared_best, rtol=1e-6, atol=1e-9):
        raise RuntimeError('Manifest best verified RMS differs from saved final state')
    initial = float(manifest['initial_verified_rms'])
    tol = float(manifest['tolerance'])
    total_sq = float(np.dot(residual, residual))
    groups=[]
    modes=[]
    for name, lo, hi in zip(NAMES, ENDS[:-1], ENDS[1:]):
        arr = residual[lo:hi]
        group_sq = float(np.dot(arr, arr))
        groups.append(dict(interface=name, mode_count=hi-lo,
                           rms=float(np.sqrt(np.mean(arr**2))),
                           squared_share_percent=100*group_sq/max(total_sq, 1e-30)))
        for j in range(lo,hi):
            modes.append(dict(index=j, interface=name,
                              normalized_residual=float(residual[j]),
                              squared_share_percent=100*float(residual[j]**2)/max(total_sq, 1e-30)))
    groups.sort(key=lambda row:row['squared_share_percent'], reverse=True)
    modes.sort(key=lambda row:row['squared_share_percent'], reverse=True)
    hist=[]
    for row in history:
        hist.append(dict(iteration=row.get('iteration',''), method=row.get('method',''),
                         accepted=row.get('accepted',''), verified=row.get('verified',''),
                         trial_rms=as_float(row.get('trial_rms')),
                         best_rms=as_float(row.get('best_rms')),
                         predicted_rms=as_float(row.get('predicted_rms')),
                         jobs_total=as_float(row.get('jobs_total'))))
    if hist and hist[-1]['best_rms'] is not None and not np.isclose(hist[-1]['best_rms'],rms,rtol=1e-6,atol=1e-9):
        raise RuntimeError('Final history best RMS does not match verified saved state')
    summary=dict(step='B9A11D-C',source=str(run_folder),
                 solver='65V safeguarded LM / secant update with Abaqus patch verification',
                 initial_verified_rms=initial,final_verified_rms=rms,
                 tolerance=tol,target_met=bool(rms<=tol),
                 residual_reduction_percent=100*(initial-rms)/initial,
                 history_rows=len(hist),
                 accepted_steps=manifest.get('accepted_steps'),
                 stop_reason=manifest.get('stop_reason'),
                 solver_patch_jobs=manifest.get('solver_patch_jobs'),
                 top_interfaces=groups[:3],top_modes=modes[:8],
                 reference_field_comparison_available=False,
                 caution='65V normalized modal interface RMS is not global FEM displacement, stress, strain, or reaction-force error.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite existing output: ' + str(out) + ' -- specify a fresh --out folder')
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out / 'D2_65V_history.csv',hist)
    write_csv(out / 'D2_65V_interface_breakdown.csv',groups)
    write_csv(out / 'D2_65V_mode_breakdown.csv',modes)
    (out / 'D2_65V_diagnostic.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    print('B9A11D-C: target met:',summary['target_met'], 'RMS:',rms)
    print('Dominant residual interface:',groups[0]['interface'],'contribution: %.2f%%'%groups[0]['squared_share_percent'])
    print('Written to',out)
    print('Send D2_65V_diagnostic.json and D2_65V_interface_breakdown.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=DEFAULT_WIN_ROOT)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    root=Path(a.project).expanduser().resolve()
    out=(Path(a.out).expanduser().resolve() if a.out else root/'results'/'65AM_B9A11D_C_65V_diagnostics')
    run(root,out)

if __name__=='__main__':
    main()
