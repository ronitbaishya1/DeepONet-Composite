#!/usr/bin/env python3
"""B9A11W1: geometrical surface-face/gauge candidates from original B5 C3D8R mesh.

No strain exports, no virtual gauge data, and no claim of physical accessibility.
Candidate faces must be reviewed for roller contact, fiber axes and accessibility.
Never substitute element-centroid LE for measured surface strain.
"""
from __future__ import annotations
import argparse, csv, importlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

FACE_LOCAL=((0,1,2,3),(4,5,6,7),(0,1,5,4),
            (1,2,6,5),(2,3,7,6),(3,0,4,7))


def exterior_faces(x,conn):
    x=np.asarray(x,float);conn=np.asarray(conn,int)
    if x.ndim!=2 or x.shape[1]!=3 or conn.ndim!=2 or conn.shape[1]!=8:
        raise ValueError('Expected node coordinates Nx3, HEX8 connectivity Ex8')
    if conn.min()<0 or conn.max()>=len(x):raise ValueError('Connectivity outside mesh')
    seen=defaultdict(list)
    for e,nodes in enumerate(conn):
        for face in FACE_LOCAL:
            inds=tuple(int(nodes[i]) for i in face)
            key=tuple(sorted(inds))
            seen[key].append((int(e),inds))
    bad=[key for key,val in seen.items() if len(val)>2]
    if bad:raise RuntimeError('Nonmanifold mesh: faces shared by >2 elements')
    out=[]
    for key,items in seen.items():
        if len(items)!=1:continue
        element,nodes=items[0]
        p=x[list(nodes)]
        normal=np.cross(p[1]-p[0],p[3]-p[0])
        length=np.linalg.norm(normal)
        if length<1e-12:raise RuntimeError('Degenerate exterior face')
        normal/=length
        center=p.mean(axis=0)
        # Orient the face normal away from its own element centroid.
        element_center=x[conn[element]].mean(axis=0)
        if np.dot(normal,center-element_center)<0:normal=-normal
        tangent=int(np.argmax(np.abs(normal)))
        # Map to a coordinate face (orientation sign is inferred only geometrically).
        out.append(dict(element=element,node_indices=nodes,
                        xyz=center,normal=normal,normal_axis=tangent,
                        geometric_area=float(length)))
    return out


def selftest():
    X=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    faces=exterior_faces(X,np.arange(8).reshape(1,8))
    assert len(faces)==6
    assert sorted(f['normal_axis'] for f in faces)==[0,0,1,1,2,2]
    assert sum(f['geometric_area'] for f in faces)==6
    print('B9A11W1 SELFTEST PASS | HEX8 exterior topology and surface normals')


def audit(root,tag,only_check=False):
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    B5=importlib.import_module('65AM_B5_common')
    reports={};rows=[]
    for region in ('inner_left','inner_right'):
        d,_,_=B5.prepare(root,region)
        ds=d['ds']
        X=np.asarray(ds['coordinates'],float)
        conn=np.asarray(ds['element_node_indices'],int)
        faces=exterior_faces(X,conn)
        lo=X.min(axis=0);hi=X.max(axis=0)
        span=hi-lo
        if min(span)<1e-6:raise RuntimeError('Flat/invalid 3D mesh')
        counts={k:sum(f['normal_axis']==k for f in faces) for k in range(3)}
        chosen=0
        for f in faces:
            axis=f['normal_axis'];coord=f['xyz'];normal=f['normal']
            signed_face='+' if normal[axis]>0 else '-'
            # DO NOT place strain gauges at x=end interfaces or at outermost end strips.
            interior_x=(lo[0]+.12*span[0] < coord[0] < hi[0]-.12*span[0])
            candidate=(axis in (1,2) and interior_x)
            # Surface accessibility must be checked: roller contacts may overlap y faces.
            # Normal axis=1 => tangent axes {0,2}; normal axis=2 => {0,1}.
            shear='LE13' if axis==1 else 'LE12' if axis==2 else 'NOT_SURFACE'
            axial='LE11' if axis in (1,2) else 'NOT_SURFACE'
            if candidate:chosen+=1
            rows.append(dict(region=region,element_index=f['element'],
                node_indices=';'.join(map(str,f['node_indices'])),
                center_x_mm=float(coord[0]),center_y_mm=float(coord[1]),
                center_z_mm=float(coord[2]),normal_axis='xyz'[axis],
                normal_sign=signed_face,area_mm2=f['geometric_area'],
                valid_geometrical_candidate=bool(candidate),
                candidate_axial_component=axial,candidate_shear_component=shear,
                physical_accessibility_verified=False,
                roller_contact_excluded=False, # geometric exclusion alone is insufficient
                material_axis_orientation_verified=False,
                FEM_surface_strain_exported=False,
                B5_surface_strain_exported=False))
        reports[region]=dict(nodes=len(X),elements=len(conn),exterior_faces=len(faces),
            exterior_face_axis_counts={'x':counts[0],'y':counts[1],'z':counts[2]},
            candidate_faces_excluding_interface_endstrips=chosen,
            xyz_bounds_mm={'min':lo.tolist(),'max':hi.tolist()})
    report={'stage':'B9A11W1','status':'GEOMETRY ONLY; NOT YET READY FOR STRAIN INVERSION',
        'data_source':'Original C3D8R B5 geometries; no new FEM/experimental strains',
        'limitations':['Exterior faces do not establish roller-contact-free surfaces',
            'Material axes/face tangent conventions still require Abaqus deck audit',
            'Abaqus element-center strains are not experimental surface-gauge values',
            'Gauge patch footprints/averaging and independent noise model not chosen',
            'Do not count strain derived from the same noisy U as independent'],
        'regions':reports}
    if only_check:
        print('W1 CHECK PASS: geometric exterior faces counted; no data written')
        print(json.dumps(report['regions'],indent=2));return
    out=root/'results'/'65AM_B9A11W1_surface_gauges'/tag
    if out.exists():raise FileExistsError('Previous W1 results protected: '+str(out))
    out.mkdir(parents=True)
    (out/'B9A11W1_surface_readiness.json').write_text(json.dumps(report,indent=2))
    with (out/'B9A11W1_exterior_face_candidates.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('W1 GEOMETRY-ONLY REPORT',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['selftest','check','run'])
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='geometry01')
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    else:audit(Path(a.project).resolve(),a.tag,only_check=a.stage=='check')

if __name__=='__main__':main()
