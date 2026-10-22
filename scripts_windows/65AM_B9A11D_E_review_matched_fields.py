#!/usr/bin/env python3
"""B9A11D-E: summarize 65W-C FEM-vs-patch displacement and 65W-E mechanics.
Never pretend patch-only fields cover seven FE-NO regions or that residual=field error.
Run AFTER 65W comparisons exist. No Abaqus jobs or checkpoint writes.
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'online_results_7region_65W_middle'
OUT = ROOT / 'results' / '65AM_B9A11D_E_matched_fields'
INTERFACES = {
    'FE_L': ('left_outer', 'left_inner'),
    'FE_C': ('center_left', 'center_right'),
    'FE_R': ('right_inner', 'right_outer'),
}


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def value(row, key):
    v = row.get(key)
    if v is None or v == '':
        return None
    return float(v)


def report(args):
    comparison = Path(args.field_json).expanduser().resolve()
    mechanics = Path(args.mechanics_csv).expanduser().resolve()
    residual = Path(args.residual_json).expanduser().resolve()
    if not comparison.is_file():
        raise FileNotFoundError('Missing 65W-C field comparison: ' + str(comparison) + '\nRun the 65W-C field comparison first.')
    fields = read_json(comparison)
    if not fields.get('physical_same_material_declared') or not fields.get('same_load_and_boundary_conditions_declared'):
        raise RuntimeError('65W-C comparison lacks explicit same-material/load confirmation.')
    mech_rows = read_csv(mechanics)
    residual_data = read_json(residual) if residual.is_file() else {}
    shares = {r['interface']: float(r['squared_share_percent'])
              for r in residual_data.get('top_interfaces', [])}
    # Prefer full per-interface list if a D2 CSV is available; D2 JSON often lists top three only.
    residual_csv = read_csv(Path(args.residual_csv).expanduser().resolve())
    if residual_csv:
        for r in residual_csv:
            name = r.get('interface')
            if name:
                for k in ('squared_share_percent','squared_share_pct'):
                    if r.get(k):
                        shares[name] = float(r[k]); break
    rows = []
    for region in ('FE_L','FE_C','FE_R'):
        d = fields.get('per_region', {}).get(region, {})
        metrics = {c: d.get(c, {}) for c in ('U_vector','U2')}
        # 65W-E uses 'left', 'center', 'right'.
        region_names = {'FE_L':'left','FE_C':'center','FE_R':'right'}
        mr = {comp: next((x for x in mech_rows if x.get('region') == region_names[region] and x.get('component') == comp), None)
              for comp in ('S22','LE22')}
        linked = INTERFACES[region]
        known = [shares.get(k) for k in linked]
        rows.append({
            'region': region,
            'matched_nodes': d.get('matched',0),
            'coverage_pct': d.get('coverage_pct',0),
            'U_vector_rel_l2_pct': metrics['U_vector'].get('relative_l2_percent') if metrics['U_vector'] else None,
            'U2_rel_l2_pct': metrics['U2'].get('relative_l2_percent') if metrics['U2'] else None,
            'S22_rel_l2_pct_FE_only': value(mr['S22'], 'rel_l2_percent') if mr['S22'] else None,
            'LE22_rel_l2_pct_FE_only': value(mr['LE22'], 'rel_l2_percent') if mr['LE22'] else None,
            'interfaces': '+'.join(linked),
            'linked_interface_squared_residual_share_pct': sum(known) if all(x is not None for x in known) else None,
            'has_direct_NO_field': False,
        })
    output = OUT / args.run_tag
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output exists, choose a new --run-tag: '+str(output))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'D4_FE_region_priority.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    data = {
        'step':'B9A11D-E',
        'field_comparison':str(comparison),
        'mechanics_comparison_csv':str(mechanics) if mechanics.is_file() else None,
        'solver_residual_json':str(residual) if residual.is_file() else None,
        'full_seven_region_claim_allowed':bool(fields.get('full_hybrid_validation_claim_allowed',False)),
        'regions':rows,
        'interpretation': ('Matched FE patch vs full-FEM field errors only. Missing NO region fields prevent full hybrid accuracy claims. '
                           'Normalized interface residual is a separate diagnostic and cannot be added to displacement/stress errors.'),
        'warning':('Matching by node coordinate, centroid, section and integration-point identifiers needs independent validation of '
                   'material orientation, mesh topology, units, time/frame and contact/load definition.'),
    }
    (output/'D4_FE_region_priority.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('D4 FE patch region comparison saved:',output)
    for r in rows:
        print('{:<5} cover={:>5.1f}% Uvec={} U2={} S22={} LE22={}'.format(
            r['region'],float(r['coverage_pct']),r['U_vector_rel_l2_pct'],r['U2_rel_l2_pct'],
            r['S22_rel_l2_pct_FE_only'],r['LE22_rel_l2_pct_FE_only']))
    if not data['full_seven_region_claim_allowed']:
        print('Partial FE-only comparison. Neural regions are not yet evaluated.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag', default='run01')
    p.add_argument('--field-json', default=str(BASE/'C_field_comparison'/'B9A11D_FE_only'/'65W_C_field_comparison.json'))
    p.add_argument('--mechanics-csv', default=str(BASE/'E_mechanics_comparison'/'65W_E_FE_only_mechanics.csv'))
    p.add_argument('--residual-json', default=str(ROOT/'results'/'65AM_B9A11D_C_65V_diagnostics'/'D2_65V_diagnostic.json'))
    p.add_argument('--residual-csv', default=str(ROOT/'results'/'65AM_B9A11D_C_65V_diagnostics'/'D2_65V_interface_breakdown.csv'))
    report(p.parse_args())

if __name__=='__main__':
    main()
