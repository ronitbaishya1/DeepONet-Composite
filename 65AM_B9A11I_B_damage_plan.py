#!/usr/bin/env python3
"""B9A11I-B: create a small controlled local E2-stiffness-loss pilot plan.

No Abaqus jobs are run. This is NOT real delamination or damage data.
Material parameters are fixed at the original healthy base reference values;
only the prescribed local E2 reduction varies. Need to audit mesh/locations.
"""
import argparse,csv,json
from pathlib import Path

def plan(out):
    centers=(-8.,-4.2,4.2,8.)
    losses=(.15,.35)
    rows=[]
    for i,x in enumerate(centers):
        for j,d in enumerate(losses):
            rows.append(dict(case_id='I_DMG_X%02d_D%02d'%(i+1,j+1),
                 x_mm=x,y_mm=2.,z_mm=4.,
                 radius_x_mm=1.5,radius_y_mm=1.5,radius_z_mm=2.5,
                 damage_E2_fraction=d,
                 E1_healthy_MPa=45000.,E2_healthy_MPa=12000.,G12_healthy_MPa=4500.,
                 design_partition='pilot_not_training_split'))
    if out.exists():raise RuntimeError('Refusing overwrite: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    info={'stage':'B9A11I-B','damage_mechanism':'localized E2-only reduction assigned per element',
      'not_delamination_or_crack':True,'cases':len(rows),'purpose':'mesh + response detectability pilot, NOT enough to train a general damage inverse surrogate',
      'next_dataset':'After pilot, design a substantially larger independently partitioned damage dataset varying location, severity, size and healthy properties; include healthy controls',
      'boundary':'Keep full 3Point geometry, load, contacts, mesh and other engineering constants unchanged',
      'field_outputs':'U1,U2,U3 at all specimen nodes, S/LE on specimen elements, RF2 at same loading reference point; verify Abaqus ODB convergence',
      'pilot_file':str(out)}
    out.with_name('B9A11I_damage_plan_summary.json').write_text(json.dumps(info,indent=2))
    print('DAMAGE PILOT MANIFEST:',out)
    print('Eight new INPs planned, zero Abaqus jobs run. Transfer CSV to Windows ODBS.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='pilot01')
    a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe tag')
    target=Path(a.project).expanduser().resolve()/'results'/'65AM_B9A11I_damage_plan'/a.tag/'B9A11I_damage_manifest.csv'
    plan(target)
