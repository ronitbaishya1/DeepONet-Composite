#!/usr/bin/env python3
"""B9A11F-A: read-only inventory and provenance gates for B5/B6/B7 and 65Y.

Mac: python3 65AM_B9A11F_A_candidate_readiness.py --project .
A missing result is UNKNOWN/NOT READY, not a pass. Never edits source models.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REGIONS = ('inner_left', 'inner_right')

def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None

def csv_info(path):
    if not path.is_file():
        return {'exists': False, 'path': str(path)}
    with path.open(newline='') as f:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        rows = list(rdr)
    return {'exists': True, 'path': str(path), 'n_rows': len(rows), 'columns': headers}

def gate_set(summary, key='decisions'):
    if summary is None:
        return {'status': 'MISSING', 'passed': False}
    vals = summary.get(key, [])
    regions = {str(row.get('region')): row for row in vals if isinstance(row, dict)}
    passed = all(r in regions and regions[r].get('pass_gate') is True for r in REGIONS)
    return {'status': 'PASS' if passed else 'FAIL', 'passed': passed,
            'regional_details': {r: regions.get(r) for r in REGIONS}}

def inspect(root):
    base = root / 'results'
    b6 = base / '65AM_B6_independent' / 'D_independent_evaluation'
    b7 = base / '65AM_B7_stress_consistency' / 'A_constitutive_audit'
    yc = base / '65Y_middle_mechanics' / 'C_holdout_comparison'
    yb = base / '65Y_middle_mechanics' / 'B_mechanics_finetune'
    b9 = base / '65AM_B9_coupled_feasibility' / 'B9A11C_weight_convergence' / 'full01'
    b6s = load_json(b6 / 'B6_summary.json')
    b7s = load_json(b7 / 'B7_summary.json')
    ycs = load_json(yc / '65Y_C_summary.json')
    ybs = load_json(yb / '65Y_B_manifest.json')
    b9s = load_json(b9 / 'B9A11C_summary.json')
    old = base / 'hybrid_online_7region_65W_middle_validation'
    b6_gate = gate_set(b6s)
    b7_ok = bool(b7s and b7s.get('law_proxy_acceptable_for_both') is True)
    b7_status = 'MISSING' if b7s is None else ('PASS' if b7_ok else 'FAIL')
    y_pass = bool(ycs and ycs.get('all_preliminary_gates_passed') is True
                  and ycs.get('displacement_and_force_heads_identical') is True
                  and all(ycs.get('gates', {}).get(r, {}).get('preliminary_gate_passed') is True for r in REGIONS))
    yc_status = 'MISSING' if ycs is None else ('PASS' if y_pass else 'FAIL')
    yc_data = csv_info(yc / '65Y_C_test_comparison.csv')
    candidates = {}
    for r in REGIONS:
        origin_dir = base / 'v4_7region_direct' / r
        new_dir = yb / r
        members = {}
        for m in range(1, 6):
            key = 'member_%02d' % m
            orig = origin_dir / key / 'best_v4_7region.pt'
            new = new_dir / key / 'best_v4_7region.pt'
            members[key] = {'original_exists': orig.is_file(), 'candidate_exists': new.is_file()}
        candidates[r] = members
    models_full = all(all(v.values()) for r in candidates.values() for v in r.values())
    yb_ok = bool(ybs and ybs.get('test_split_unused') is True
                 and ybs.get('frozen_displacement_and_force_heads') is True)
    expected = [
        old / 'assembled_hybrid_nodes.csv',
        old / 'assembled_hybrid_ip.csv',
        old / '65W_H_validation_summary.json',
        root / 'data' / 'hybrid_online_7region_65W_middle' / 'rom_solution_7region.npz',
        root / '64V_reconstruct_validate_7region_direct.py',
    ]
    result = {
        'step': '65AM-B9A11F-A',
        'project': str(root),
        'B6': {'gate': b6_gate, 'summary_exists': b6s is not None,
               'metrics': csv_info(b6 / 'B6_pooled_metrics.csv'),
               'reference_checks': csv_info(b6 / 'B6_reference_checks.csv')},
        'B7': {'status': b7_status, 'proxy_acceptable_for_both': b7_ok,
               'not_deployment_approval': True,
               'component_metrics': csv_info(b7 / 'B7_stress_component_metrics.csv')},
        '65Y': {'holdout_status': yc_status, 'holdout_gate_pass': y_pass,
                'training_manifest_exists': ybs is not None,
                'training_provenance_pass': yb_ok,
                'candidate_ensemble_5_each_available': models_full,
                'holdout_comparison': yc_data, 'members': candidates,
                'gate_details': ycs.get('gates', {}) if ycs else {}},
        'B9A11C': {'summary_exists': b9s is not None,
                   'approved_for_coupled_use': bool(b9s and b9s.get('approved_for_coupled_use') is True)},
        '65W': {'requirements': {str(x): x.is_file() for x in expected},
                'baseline_present': all(x.is_file() for x in expected)},
    }
    result['readiness'] = {
        '65Y_middle_mechanics_diagnostic': bool(y_pass and yb_ok and models_full and all(x.is_file() for x in expected)),
        'B5_coupled_deployment': False,
        'why_B5_deployment_blocked': ('B6/B7 are offline FEM-oracle interface tests, '
                                      'and B9A11C has no approved coupled/independent deployment.'),
        'note': ('65Y mechanics-only replay is a comparison at the frozen 65V solution, '
                 'NOT approval of a new force/displacement hybrid or a selection test on the middle reference.'),
    }
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='.')
    p.add_argument('--tag', default='run01')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    root = Path(a.project).expanduser().resolve()
    result = inspect(root)
    print('B6 independent displacement/gradient gate:',result['B6']['gate']['status'])
    print('B7 stress-proxy check:',result['B7']['status'])
    print('65Y untouched test gate:',result['65Y']['holdout_status'])
    print('65Y 5-member checkpoint set:',result['65Y']['candidate_ensemble_5_each_available'])
    print('65W original assembled comparison:',result['65W']['baseline_present'])
    print('Can run 65Y middle-material mechanics-only diagnostic:',result['readiness']['65Y_middle_mechanics_diagnostic'])
    if a.check_only:
        print('CHECK ONLY; no output written')
        return
    out = root / 'results' / '65AM_B9A11F_readiness' / a.tag
    if out.exists():
        raise FileExistsError('Output protected; select new --tag: '+str(out))
    out.mkdir(parents=True)
    (out / 'B9A11F_A_readiness.json').write_text(json.dumps(result,indent=2))
    print('Saved:',out / 'B9A11F_A_readiness.json')
    if not result['readiness']['65Y_middle_mechanics_diagnostic']:
        print('STOP: do not deploy any new neural model. Review missing/failed gates first.')

if __name__ == '__main__':
    main()
