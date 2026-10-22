#!/usr/bin/env python3
"""B9A11D-D: read-only inventory of middle-material full FEM reference and 65V patches.
Run from ODBS with: abaqus python scripts\\65AM_B9A11D_D_find_reference.py
No ODB modification, no Abaqus jobs, no inference from filenames alone.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = (46500., 12500., 4750.)  # E1 E2 G12 MPa
OUT_BASE = ROOT / 'results' / '65AM_B9A11D_D_reference_inventory'


def material_blocks(inp):
    """List ENGINEERING CONSTANTS blocks. This inspects INP, never infers from ODB name."""
    if not inp.is_file():
        return []
    try:
        with inp.open('r', encoding='utf-8-sig', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return []
    current = None
    found = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^\*material\b', line, re.I):
            m = re.search(r'\bname\s*=\s*([^,]+)', line, re.I)
            current = m.group(1).strip().strip('"\'') if m else None
        elif re.match(r'^\*elastic\b', line, re.I) and 'engineering constants' in line.lower():
            vals = []
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith('*'):
                content = lines[j].split('**', 1)[0].strip()
                if content:
                    for item in content.split(','):
                        if item.strip():
                            try:
                                vals.append(float(item.strip()))
                            except ValueError:
                                vals = []
                                break
                j += 1
            if len(vals) == 9:
                found.append({'name': current, 'E1': vals[0], 'E2': vals[1],
                              'G12': vals[6], 'all_nine': vals})
            i = j - 1
        i += 1
    return found


def matches_target(block):
    return all(abs(block[k] - t) < 0.001 for k, t in zip(('E1','E2','G12'), TARGET))


def case_files(root, max_depth, max_files):
    candidates = []
    seen = set()
    ignored = {'.git', '.venv', '__pycache__', 'scripts', 'results', 'research_reports', 'Unused_Files'}
    scanned_odb = 0
    truncated = False
    for base, dirs, files in os.walk(str(root)):
        p = Path(base)
        depth = len(p.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in ignored and depth < max_depth)
        for filename in files:
            if not filename.lower().endswith('.odb'):
                continue
            scanned_odb += 1
            if scanned_odb > max_files:
                truncated = True
                break
            f = p / filename
            if f in seen:
                continue
            seen.add(f)
            relative = str(f.relative_to(root)).replace('\\', '/')
            # List likely full FEM candidates and exact verified patch locations.
            name = filename.lower()
            likely_full = any(token in name for token in ('full','3point','reference','middle'))
            is_patch = filename.lower() in ('left.odb', 'center.odb', 'right.odb') and (
                '65v_d_lm_convergence' in relative.lower() or '65w_middle' in relative.lower())
            if not likely_full and not is_patch:
                continue
            paired = f.with_suffix('.inp')
            blocks = material_blocks(paired)
            matched = [b for b in blocks if matches_target(b)]
            st = f.stat()
            candidates.append({
                'role_guess': '65V/65W FEM patch' if is_patch else 'possible full FEM',
                'odb': str(f), 'size_MB': round(st.st_size / 1024**2, 2),
                'paired_inp': str(paired) if paired.is_file() else '',
                'matched_middle_INP': bool(matched) if paired.is_file() else None,
                'matched_material_name': '; '.join(str(b['name']) for b in matched),
                'engineering_constant_blocks': len(blocks),
                'note': ('INP confirms E1/E2/G12 only, not mesh/loads/BCs'
                         if matched else 'Material not established from paired INP'),
            })
        if truncated:
            break
    return sorted(candidates, key=lambda x: (
        0 if x['matched_middle_INP'] is True and x['role_guess'] == 'possible full FEM' else 1,
        0 if x['role_guess'] == 'possible full FEM' else 1,
        x['odb'].lower())), scanned_odb, truncated


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--run-tag', default='run01')
    pa.add_argument('--max-depth', type=int, default=8)
    pa.add_argument('--max-odb-files', type=int, default=3000)
    a = pa.parse_args()
    if not re.match(r'^[A-Za-z0-9_-]+$', a.run_tag):
        pa.error('Use simple run-tag, e.g. run01 or run02')
    out = OUT_BASE / a.run_tag
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Protected existing output: ' + str(out) + '. Use --run-tag run02.')
    cases, count, truncated = case_files(ROOT, a.max_depth, a.max_odb_files)
    prior = ROOT / 'online_results_7region_65W_middle'
    valid = prior / 'B_odb_exports'
    known = {
        'saved_65V_solution': ROOT / 'online_results_7region_65V_middle' / '65V_D_lm_convergence' / 'rom_solution_7region.npz',
        'saved_65V_left_odb': ROOT / 'online_results_7region_65V_middle' / '65V_D_lm_convergence' / 'final' / 'left.odb',
        'saved_65V_center_odb': ROOT / 'online_results_7region_65V_middle' / '65V_D_lm_convergence' / 'final' / 'center.odb',
        'saved_65V_right_odb': ROOT / 'online_results_7region_65V_middle' / '65V_D_lm_convergence' / 'final' / 'right.odb',
        '65W_independent_reproduction': prior / 'A_independent_reproduction' / '65W_A_reproduction.json',
        '65W_FullMiddle_inp': prior / '0_full_reference' / '65W_FullMiddle.inp',
        '65W_FullMiddle_odb_at_root': ROOT / '65W_FullMiddle.odb',
        '65W_FullMiddle_odb_in_reference_folder': prior / '0_full_reference' / '65W_FullMiddle.odb',
        '65W_export_full': valid / 'full' / 'full_manifest.json',
        '65W_export_left': valid / 'patch_left' / 'patch_left_manifest.json',
        '65W_export_center': valid / 'patch_center' / 'patch_center_manifest.json',
        '65W_export_right': valid / 'patch_right' / 'patch_right_manifest.json',
    }
    report = {
        'step': 'B9A11D-D', 'root': str(ROOT), 'target_MPa_E1_E2_G12': TARGET,
        'candidate_odb_count': len(cases), 'total_odb_seen': count,
        'inventory_truncated': truncated,
        'known_files': {k: {'path': str(v), 'exists': v.is_file()} for k,v in known.items()},
        'possible_full_reference': [c for c in cases if c['role_guess'] == 'possible full FEM'],
        'verified_FEM_patch_candidates': [c for c in cases if c['role_guess'] != 'possible full FEM'],
        'next': ('Choose full reference only AFTER confirming matching material, '
                 'loading, geometry, units, mesh/coordinates, step and frame. '
                 'An ODB filename or existing 65W INP alone is NOT verification.'),
        'existing_workflow': 'scripts/65W_B_export_odb.py -> 65W_C_compare_fields.py -> 65W_E_compare_mechanics.py',
        'no_existing_files_modified': True,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'D3_reference_inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fields = ['role_guess','odb','size_MB','paired_inp','matched_middle_INP',
              'matched_material_name','engineering_constant_blocks','note']
    with (out / 'D3_odb_candidates.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cases)
    print('B9A11D-D complete; candidate full references:', len(report['possible_full_reference']))
    for c in report['possible_full_reference'][:12]:
        print('  ', c['odb'], '| paired INP middle material:', c['matched_middle_INP'])
    print('Saved:', out)
    if truncated:
        print('WARNING: ODB scan truncated; repeat with larger --max-odb-files')


if __name__ == '__main__':
    main()
