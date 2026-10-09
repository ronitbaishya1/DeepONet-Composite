#!/usr/bin/env python3
"""65W-H / Mac M1 Max: rerun EXACT original 64V five-model V4 inference
at the verified middle-material 65V solution, reconstruct seven-region fields,
then export four real NO nodal CSVs for the Windows 65W-C comparison.

Uses the trusted 64V file as the source of model architecture, hard-interface
compatibility, all field evaluations and spatial matching. The legacy 64V
baseline data/results are neither read nor changed. The original 64V source
is NEVER edited; the only changes applied in memory are input/output paths.

65W reaction evaluation remains separate and must use middle-material ODB.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

MAPPING={
    'outer_left':('NO_OL',-12.,-9.6),
    'inner_left':('NO_L',-6.6,-1.8),
    'inner_right':('NO_R',1.8,6.6),
    'outer_right':('NO_OR',9.6,12.),
}


def write_no_csv(out,segment,pred):
    name,a,b=MAPPING[segment]
    coords=np.asarray(pred['coordinates'],float)
    U=np.asarray(pred['U'],float)
    if coords.ndim!=2 or coords.shape[1]!=3 or coords.shape!=U.shape:
        raise RuntimeError('%s invalid NO coordinate/displacement shape: %s vs %s'%(segment,coords.shape,U.shape))
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(coords)):
        raise RuntimeError('%s contains nonfinite values'%segment)
    # Use same disjoint right-owner convention as the previously generated
    # 65W_C_compare_fields.py. The legacy 64V 7-region global assembly also
    # runs above and uses FE ownership at all six shared interfaces.
    inclusive_end=name=='NO_OR'
    mask=((coords[:,0]>=a-1e-7) & ((coords[:,0]<=b+1e-7) if inclusive_end else (coords[:,0]<b-1e-7)))
    if mask.sum()==0:
        raise RuntimeError('No coordinates inside region '+name)
    path=out/('65W_H_'+name+'.csv')
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['x','y','z','U1','U2','U3'])
        for xyz,u in zip(coords[mask],U[mask]):
            w.writerow([float(v) for v in np.r_[xyz,u]])
    return name,int(mask.sum()),str(path)


def preflight(root):
    required=[root/'64V_reconstruct_validate_7region_direct.py',
              root/'data'/'hybrid_online_7region_65W_middle'/'rom_solution_7region.npz',
              root/'src'/'hybrid_bulk_operator_v4_7region.py',
              root/'src'/'hard_interface_compatibility_7region.py']
    for seg in MAPPING:
        required.append(root/'data'/'hybrid_bulk_7region'/seg/'coordinates.npy')
        required.append(root/'data'/'hybrid_bulk_7region'/seg/'ip_coordinates.npy')
        for member in range(1,6):
            required.append(root/'results'/'v4_7region_direct'/seg/('member_%02d'%member)/'best_v4_7region.pt')
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing original Mac model dependencies (first 15):\n'+'\n'.join(missing[:15])+'\nTotal missing: '+str(len(missing)))
    vf=root/'data'/'hybrid_online_7region_65W_middle'/'validation_fields'
    expected=['full_reference_nodes.csv','full_reference_ip.csv','left_nodes.csv','left_ip.csv',
              'center_nodes.csv','center_ip.csv','right_nodes.csv','right_ip.csv']
    missing=[str(vf/n) for n in expected if not (vf/n).is_file()]
    if missing:raise FileNotFoundError('Run 65W_H_prepare_mac_inputs.py first:\n'+'\n'.join(missing))
    output=root/'results'/'hybrid_online_7region_65W_middle_validation'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output directory already contains data; protect prior results: '+str(output))
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',required=True)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    output=preflight(root)
    src=root/'64V_reconstruct_validate_7region_direct.py'
    source=src.read_text(encoding='utf-8')
    # Check exact original version before any model inference.
    for marker in ('hybrid_online_7region_direct','online_solution_7region.npz',
                   'hybrid_online_7region_direct_validation',
                   '# REACTION SUMMARY','NO_PREDICTIONS','predict_segment('):
        if marker not in source:
            raise RuntimeError('64V source does not have the expected original section: '+marker)
    # Path switch only, preserving exact 64V architecture and calculations.
    edited=source.replace('hybrid_online_7region_direct_validation',
                          'hybrid_online_7region_65W_middle_validation')
    edited=edited.replace('hybrid_online_7region_direct','hybrid_online_7region_65W_middle')
    edited=edited.replace('online_solution_7region.npz','rom_solution_7region.npz')
    # Cut off only the reaction report and final print section. Those refer to
    # old full-FEM nose reactions; reaction comparison is Step 65W-J.
    cutoff=edited.index('# REACTION SUMMARY')
    edited=edited[:edited.rfind('# ============================================================',0,cutoff)]
    # Verify the in-memory edit will not accidentally read/write legacy folders.
    forbidden=('hybrid_online_7region_direct','online_solution_7region.npz')
    if any(s in edited for s in forbidden):
        raise RuntimeError('Legacy 64V result path remains in adapted source')
    print('Model and input preflight PASS; 65W-H output is',output)
    if args.check_only:
        print('CHECK ONLY: zero model inference, zero new outputs')
        return
    import torch
    import pandas as pd
    os.chdir(root)
    scope={'__name__':'__main__','__file__':str(src)}
    # Executes trusted uploaded local 64V code; do not supply untrusted source.
    exec(compile(edited,str(src)+' [65W middle-material adapted in memory]','exec'),scope)
    out_no=output/'H_no_fields'
    out_no.mkdir(parents=True,exist_ok=True)
    details={}
    for segment,prediction in scope['NO_PREDICTIONS'].items():
        region,count,path=write_no_csv(out_no,segment,prediction)
        details[region]={'node_rows':count,'csv':path}
    metrics=scope['displacement_metrics']
    regions=scope['region_displacement_metrics']
    mechanics=scope['mechanics_metrics']
    assembled=scope['assembled_nodes']
    audit=scope['ip_matching_audit']
    if len(assembled)!=4592 or not audit['one_to_one'] or audit['maximum_coordinate_distance_mm']>1e-4:
        raise RuntimeError('FULL VALIDATION FAILED: wrong reference node count or IP matching')
    summary={
        'material_MPa':[float(scope[x]) for x in ('E1','E2','G12')],
        'final_state_verified_equilibrium_RMS':0.019522576349400432,
        'full_fem_nodes':int(len(assembled)),
        'mechanics_points':int(len(scope['assembled_ip'])),
        'no_csv':details,
        'region_nodes':{str(name):int((assembled['Owner']==name).sum()) for name in scope['REGION_ORDER']},
        'global_displacement':metrics.to_dict(orient='records'),
        'region_displacement':regions.to_dict(orient='records'),
        'mechanics':mechanics.to_dict(orient='records'),
        'ip_matching_audit':audit,
        'independent_reaction_audit':'Not part of this stage: see 65W-J',
        'physical_material_loading_equivalence_confirmed':False,
        'warning':'Numerical reconstruction is not certification of identical material, loading and mesh orientations; verify independently.'}
    with (output/'65W_H_validation_summary.json').open('w') as f:json.dump(summary,f,indent=2)
    print('65W-H COMPLETE: full seven-region assembled field,',len(assembled),'reference nodes')
    print('4 NO CSV files for Windows 65W-C:',out_no)
    print('Global displacement errors:')
    print(metrics.to_string(index=False))
    print('Send 65W_H_validation_summary.json and 65W_C_field_comparison.json for review.')

if __name__=='__main__':main()
