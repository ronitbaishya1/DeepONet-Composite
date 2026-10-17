#!/usr/bin/env python3
"""WINDOWS / Abaqus 2025 Python ONLY: export one NEW full-model ODB.

Example Windows command (run from a directory with the files below):
  abaqus python 65AJ_W_export_from_odb.py --odb C:\\...\\AJ_new_01.odb \
      --target-dir C:\\...\\new_FEM_exports\\inner_left --case AJ_new_01

Target dir MUST ALREADY contain the ORIGINAL 64J region coordinates.npy and
ip_coordinates.npy copied from Mac's data/hybrid_bulk_7region/<region>/.
No stress/strain reference values are fabricated or interpolated. Requires a
single C3D8R integration point per solid element. Match by coordinates only.
Do not run on Mac; uses Abaqus odbAccess.
"""
from __future__ import print_function
import os,sys,argparse
import numpy as np

COMP=('11','22','33','12','13','23')

def nearest_unique(target,source,tolerance=1e-4):
    target=np.asarray(target,float);source=np.asarray(source,float)
    if target.ndim!=2 or source.ndim!=2 or target.shape[1]!=3 or source.shape[1]!=3:
        raise RuntimeError('Expected N x 3 coordinate arrays')
    match=[];distances=[]
    for x in target:
        d2=np.sum((source-x[None,:])**2,axis=1)
        idx=int(np.argmin(d2));d=float(np.sqrt(d2[idx]))
        if d>tolerance:raise RuntimeError('No matching node/element IP near '+str(x)+'; gap='+str(d))
        if np.sum(d2<=tolerance**2)!=1:
            raise RuntimeError('Ambiguous mesh matching within tolerance '+str(tolerance)+' near '+str(x))
        match.append(idx);distances.append(d)
    if len(set(match))!=len(match):raise RuntimeError('Duplicate mapped node/element IP')
    return np.asarray(match,dtype=np.int64),float(max(distances) if distances else 0.)

def extract_field(field,inst,position,labels,expected_labels):
    from abaqusConstants import INTEGRATION_POINT
    sub=field.getSubset(region=inst,position=INTEGRATION_POINT)
    comps=list(sub.componentLabels)
    # Avoid using invariant or unknown tensor ordering silently.
    if not all(label in comps for label in expected_labels):
        raise RuntimeError('Field %s missing tensor components %s; found %s'%(field.name,expected_labels,comps))
    cols=[comps.index(lbl) for lbl in expected_labels]
    lookup={}
    for v in sub.values:
        label=int(v.elementLabel)
        if label in lookup:raise RuntimeError('Multiple integration point values per element %d; expected C3D8R'%label)
        lookup[label]=np.asarray(v.data,dtype=np.float64)[cols]
    missing=[int(lbl) for lbl in labels if int(lbl) not in lookup]
    if missing:raise RuntimeError('Missing %d element labels in %s, e.g. %s'%(len(missing),field.name,missing[:5]))
    return np.stack([lookup[int(lbl)] for lbl in labels]).astype(np.float32)

def main():
    from odbAccess import openOdb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--odb',required=True);p.add_argument('--target-dir',required=True)
    p.add_argument('--case',required=True);p.add_argument('--instance',default='COMPOSITE-1')
    p.add_argument('--step',default='Loading');p.add_argument('--tol',type=float,default=1e-4)
    a=p.parse_args();folder=os.path.abspath(a.target_dir)
    if not os.path.isfile(a.odb):raise IOError('ODB not found: '+a.odb)
    if not os.path.isdir(folder):raise IOError('Target directory not found: '+folder)
    requested=[os.path.join(folder,z+'.npy') for z in ('coordinates','ip_coordinates')]
    if not all(os.path.isfile(v) for v in requested):raise IOError('Original Mac template coordinates.npy and ip_coordinates.npy must be copied into '+folder)
    xyz_ref=np.load(requested[0],allow_pickle=False);ip_ref=np.load(requested[1],allow_pickle=False)
    outs=[os.path.join(folder,kind+'_'+a.case+'.npy') for kind in ('U','LE','S')]
    if any(os.path.exists(v) for v in outs):raise IOError('Refusing to overwrite '+str(outs))
    odb=openOdb(path=os.path.abspath(a.odb),readOnly=True)
    try:
        assembly=odb.rootAssembly
        if a.instance not in assembly.instances:raise RuntimeError('Composite ODB instance not found '+a.instance)
        inst=assembly.instances[a.instance]
        if a.step not in odb.steps:raise RuntimeError('Missing step '+a.step)
        frame=odb.steps[a.step].frames[-1]
        for fld in ('U','LE','S'):
            if fld not in frame.fieldOutputs:raise RuntimeError('Missing ODB output field '+fld)
        nodes=list(inst.nodes);elements=list(inst.elements)
        nxyz=np.asarray([n.coordinates for n in nodes],dtype=np.float64)
        node_ids=np.asarray([int(n.label) for n in nodes],dtype=int)
        idx_n,gn=nearest_unique(xyz_ref,nxyz,a.tol)
        nodes_by_label={int(n.label):np.asarray(n.coordinates,dtype=float) for n in nodes}
        elem_centers=[]
        for el in elements:
            if el.type.upper()!='C3D8R':
                raise RuntimeError('Non-C3D8R element in COMPOSITE-1: %s label %s'%(el.type,el.label))
            elem_centers.append(np.mean([nodes_by_label[int(j)] for j in el.connectivity],axis=0))
        idx_e,ge=nearest_unique(ip_ref,np.asarray(elem_centers),a.tol)
        el_labels=np.asarray([int(elements[i].label) for i in idx_e],dtype=int)
        # Nodal output U is gathered by node label (ODB field order may differ).
        nodal={int(v.nodeLabel):np.asarray(v.data,dtype=np.float32) for v in frame.fieldOutputs['U'].values if v.instance.name==inst.name}
        missing=[int(node_ids[i]) for i in idx_n if int(node_ids[i]) not in nodal]
        if missing:raise RuntimeError('Missing U labels: '+str(missing[:6]))
        U=np.stack([nodal[int(node_ids[i])][:3] for i in idx_n]).astype(np.float32)
        labels=['S'+x for x in COMP]
        S=extract_field(frame.fieldOutputs['S'],inst,None,el_labels,labels)
        LE=extract_field(frame.fieldOutputs['LE'],inst,None,el_labels,['LE'+x for x in COMP])
        if U.shape!=(len(xyz_ref),3) or S.shape!=(len(ip_ref),6) or LE.shape!=S.shape:
            raise RuntimeError('Exported arrays shape mismatch')
        if not all(np.isfinite(z).all() for z in (U,S,LE)):
            raise RuntimeError('Nonfinite Abaqus outputs')
        for path,data in zip(outs,(U,LE,S)):np.save(path,data)
        print('PASS',a.case,'nodal max geom gap',gn,'element IP max geom gap',ge)
        print('S and LE Abaqus material local-component conventions must be independently checked')
        print('Files:',outs)
    finally:odb.close()
if __name__=='__main__':main()
