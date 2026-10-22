#!/usr/bin/env python3
"""B9A11L-B: create PREDECLARED validation manifests for B9A11I-C generator.

These new locations/severities are never used in B9A11L-A sensor selection.
Creates 4 new Abaqus input plans only, not .inp nor Abaqus jobs.
"""
import argparse,csv,hashlib,json
from pathlib import Path
CASES=[('I_DMG_X05_D03',-6.,.20),('I_DMG_X05_D04',-6.,.30),
       ('I_DMG_X06_D03', 6.,.20),('I_DMG_X06_D04',6.,.30)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',default='.');p.add_argument('--tag',default='holdout01')
 a=p.parse_args();root=Path(a.project).expanduser().resolve()
 out=root/'results'/'65AM_B9A11L_validation_design'/a.tag
 if out.exists():raise RuntimeError('Refusing overwrite existing '+str(out))
 out.mkdir(parents=True)
 rows=[{'case_id':name,'x_mm':x,'y_mm':2.,'z_mm':4.,'radius_x_mm':1.5,'radius_y_mm':1.5,'radius_z_mm':2.5,
    'damage_E2_fraction':damage,'E1_healthy_MPa':45000.,'E2_healthy_MPa':12000.,'G12_healthy_MPa':4500.,
    'design_partition':'holdout_fixed_before_FEM'} for name,x,damage in CASES]
 dest=out/'B9A11L_new_damage_manifest.csv'
 with dest.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'stage':'B9A11L-B','cases':rows,'manifest_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
 'source':'new x=+-6 and E2 reductions 20%,30%, using same mesh/geometry/ellipsoid radii as B9A11I',
 'holdout_not_seen_by_sensor_design':True,
 'WARNING':'Four FEM validation cases are limited and share the same healthy constants, mesh, contact and simplified damage law. Not a complete continuous damage-inference validation.'}
 (out/'B9A11L_B_validation_plan_summary.json').write_text(json.dumps(summary,indent=2))
 print('PLAN:',dest,'\nFOUR NEW cases, no Abaqus jobs; transfer CSV to Windows ODBS root')
if __name__=='__main__':main()
