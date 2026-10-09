#!/usr/bin/env python3
"""65W-H / Mac: adapt real 65W Abaqus CSV exports to the proven 64V field reconstruction inputs.

Run on M1 Max Mac. No Abaqus or PyTorch is required for this preparation step.
All writes are in a NEW 65W middle-material folder and existing files are protected.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np

FIELDS = ("11", "22", "33", "12", "13", "23")
INTERFACES = ("left_outer", "left_inner", "center_left", "center_right", "right_inner", "right_outer")
REGIONS = ("full", "patch_left", "patch_center", "patch_right")
NODAL = ("NodeLabel", "X", "Y", "Z", "U1", "U2", "U3", "RF1", "RF2", "RF3")
MECH = ("ElementLabel", "X", "Y", "Z") + tuple("LE"+j for j in FIELDS) + tuple("S"+j for j in FIELDS)


def load_rows(p):
    if not p.is_file():
        raise FileNotFoundError('Missing 65W-B export: '+str(p))
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(p, keys, rows):
    with p.open('w', newline='') as f:
        wr=csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def convert_node(source, target):
    raw=load_rows(source)
    out=[]
    for row in raw:
        if row.get('instance','').upper()!='COMPOSITE-1':
            continue
        out.append(dict(zip(NODAL, (
            row['node_label'], row['x'], row['y'], row['z'],
            row['U1'], row['U2'], row['U3'], row.get('RF1',''), row.get('RF2',''), row.get('RF3','')))))
    if not out or any(not all(r[k].strip() for k in ('X','Y','Z','U1','U2','U3')) for r in out):
        raise RuntimeError('Missing valid composite displacement data: '+str(source))
    labels=[r['NodeLabel'] for r in out]
    if len(labels)!=len(set(labels)):
        raise RuntimeError('Duplicate composite node labels: '+str(source))
    csv_write(target, NODAL, out)
    return len(out)


def convert_mech(source, target):
    raw=load_rows(source)
    per_field={'LE':{},'S':{}}
    for row in raw:
        if row.get('instance','').upper()!='COMPOSITE-1':
            continue
        fld=row.get('field','').strip().upper()
        if fld not in per_field:
            continue
        key=(row['element_label'],row.get('integration_point',''),row.get('section_point',''))
        if key in per_field[fld]:
            raise RuntimeError('Duplicate %s integration-point record for %s in %s'%(fld,key,source))
        per_field[fld][key]=row
    if not per_field['S'] or set(per_field['LE']) != set(per_field['S']):
        raise RuntimeError('S/LE integration-point keys do not match: '+str(source))
    elems={key[0] for key in per_field['S']}
    if len(elems)!=len(per_field['S']):
        raise RuntimeError('Multiple section/integration points per element; 64V expects one. Need explicitly map layer/IP data: '+str(source))
    out=[]
    for key in sorted(per_field['S'],key=lambda x: int(x[0])):
        S=per_field['S'][key]; LE=per_field['LE'][key]
        xyz=('centroid_x','centroid_y','centroid_z')
        if any(abs(float(S[k])-float(LE[k]))>1e-8 for k in xyz):
            raise RuntimeError('S and LE coordinates differ for element '+key[0])
        row={'ElementLabel':key[0], 'X':S[xyz[0]], 'Y':S[xyz[1]],'Z':S[xyz[2]]}
        for comp in FIELDS:
            for field,source_row in (('LE',LE),('S',S)):
                col=field+comp
                row[col]=source_row[col]
                if not row[col].strip():
                    raise RuntimeError('Missing %s for element %s'%(col,key[0]))
        out.append(row)
    csv_write(target, MECH, out)
    return len(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,help='Mac project root, containing 64V_reconstruct_validate_7region_direct.py')
    p.add_argument('--exports',required=True,help='Copied Windows B_odb_exports directory')
    p.add_argument('--solution',required=True,help='Final converged 65V rom_solution_7region.npz')
    args=p.parse_args()
    project=Path(args.project).expanduser().resolve()
    exports=Path(args.exports).expanduser().resolve()
    solution=Path(args.solution).expanduser().resolve()
    if not (project/'64V_reconstruct_validate_7region_direct.py').is_file():
        raise FileNotFoundError('Missing original 64V reconstruction script in '+str(project))
    dest=project/'data'/'hybrid_online_7region_65W_middle'
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Existing 65W Mac input directory is protected: '+str(dest))
    with np.load(solution,allow_pickle=False) as z:
        required={'q','E1','E2','G12','normalized_residual'}|{'c_'+x for x in INTERFACES}
        missing=required-set(z.files)
        if missing:raise RuntimeError('Saved solution lacks '+str(sorted(missing)))
        material=[float(np.asarray(z[k]).ravel()[0]) for k in ('E1','E2','G12')]
        if not np.allclose(material,[46500.,12500.,4750.],rtol=0,atol=1e-6):
            raise RuntimeError('Wrong material case: '+str(material))
        r=np.asarray(z['normalized_residual'],float).ravel()
        q=np.asarray(z['q'],float).ravel()
        if r.shape!=(26,) or q.shape!=(26,):raise RuntimeError('Expected 26 interface coordinates')
        RMS=float(np.sqrt(np.mean(r*r)))
        if RMS>0.020000001:raise RuntimeError('Saved 65V solution not converged: '+str(RMS))
    planned={}
    for tag in REGIONS:
        folder=exports/tag
        node_csv=folder/(tag+'_nodes.csv')
        mech_csv=folder/(tag+'_mechanics.csv')
        manifest=folder/(tag+'_manifest.json')
        for fp in (node_csv,mech_csv,manifest):
            if not fp.is_file():raise FileNotFoundError(str(fp))
        with manifest.open() as f:md=json.load(f)
        if list(md.get('instances',[])) != ['COMPOSITE-1']:
            raise RuntimeError('Unexpected Abaqus exported instances in '+str(manifest))
        planned[tag]=(node_csv,mech_csv,md)
    output_names={
        'full':('full_reference_nodes.csv','full_reference_ip.csv'),
        'patch_left':('left_nodes.csv','left_ip.csv'),
        'patch_center':('center_nodes.csv','center_ip.csv'),
        'patch_right':('right_nodes.csv','right_ip.csv')}
    vf=dest/'validation_fields';vf.mkdir(parents=True)
    report={'solution_rms':RMS,'material_MPa':material,'source_files':{},'rows':{}}
    shutil.copy2(solution,dest/'rom_solution_7region.npz')
    for tag,(node_csv,mech_csv,md) in planned.items():
        nname,mname=output_names[tag]
        n=convert_node(node_csv,vf/nname)
        m=convert_mech(mech_csv,vf/mname)
        if n!=int(md['node_rows']) or m!=int(md['mechanics_rows_per_field']['S']):
            raise RuntimeError('Converted CSV count disagrees with Abaqus manifest for '+tag)
        report['source_files'][tag]=[str(node_csv),str(mech_csv)]
        report['rows'][tag]={'nodes':n,'elements':m}
    if report['rows']['full']['nodes']!=4592 or report['rows']['full']['elements']!=3640:
        raise RuntimeError('Full reference mesh differs from previously inventoried middle case')
    with (vf/'65W_H_preparation_manifest.json').open('w') as f:json.dump(report,f,indent=2)
    print('65W-H PREPARATION PASS: Converted 65W-B CSVs for original 64V reconstruction.')
    print(json.dumps(report['rows'],indent=2))
    print('NEXT: python3 65W_H_reconstruct_on_mac.py --project "'+str(project)+'"')

if __name__=='__main__':main()
