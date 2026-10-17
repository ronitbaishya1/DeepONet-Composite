# -*- coding: utf-8 -*-
"""B9A4 Windows Abaqus: export REAL B9A3 nodal U/RF, C3D8R IP S/LE.

From ODBS:
  abaqus python scripts\65AM_B9A4_W_export_fields.py --stage check
  abaqus python scripts\65AM_B9A4_W_export_fields.py --stage export

Create ODBS/B9A4_exports with File Explorer before export.
The two B9A3 ODBs are reused. No new Abaqus jobs and no overwrites.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ANCHOR=ROOT/'B9_anchor'/'B9_65V_anchor.npz'
ANCHOR_META=ROOT/'B9_anchor'/'B9_65V_anchor_manifest.json'
REF=ROOT/'B9A3_exports'/'B9A3_NO_reference.npz'
REF_META=ROOT/'B9A3_exports'/'B9A3_windows_summary.json'
ODBS=ROOT/'B9A3_ODB'; OUT=ROOT/'B9A4_exports'
REGIONS=('inner_left','inner_right')
COMPONENTS=('11','22','33','12','13','23')

def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def load_json(path):
    with path.open() as f:return json.load(f)

def prerequisites():
    for f in (ANCHOR,ANCHOR_META,REF,REF_META):
        if not f.is_file():raise FileNotFoundError('Prior B9 file missing: '+str(f))
    anchor_hash=sha(ANCHOR)
    if load_json(ANCHOR_META).get('anchor_sha256')!=anchor_hash:raise RuntimeError('Anchor hash mismatch')
    meta=load_json(REF_META)
    if meta.get('anchor_sha256')!=anchor_hash or meta.get('export_sha256')!=sha(REF):
        raise RuntimeError('B9A3 reference hash mismatch')
    if not meta.get('geometry_and_displacement_checks') or len(meta.get('cases',[]))!=2:
        raise RuntimeError('Prior Abaqus displacement/reference checks were not passed')
    for reg in REGIONS:
        odb=ODBS/('B9A3_'+reg+'.odb')
        sta=ODBS/('B9A3_'+reg+'.sta')
        if not odb.is_file() or not sta.is_file():raise FileNotFoundError('Missing B9A3 ODB or STA for '+reg)
        if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
            raise RuntimeError('Abaqus B9A3 job not successful '+reg)
    return anchor_hash,meta

def field_tensor(frame,instance,name,elabels):
    from abaqusConstants import INTEGRATION_POINT
    field=frame.fieldOutputs[name]
    labels=tuple(str(c).upper() for c in field.componentLabels)
    needed=tuple(name+c for c in COMPONENTS)
    if set(labels)!=set(needed):raise RuntimeError('%s component labels unexpected: %r'%(name,labels))
    order=[labels.index(c) for c in needed]
    vals={}
    for v in field.getSubset(region=instance,position=INTEGRATION_POINT).values:
        lab=int(v.elementLabel)
        if lab in vals:raise RuntimeError('%s has multiple integration/section points for element %s'%(name,lab))
        raw=np.asarray(v.data,dtype=np.float64).ravel()
        if len(raw)!=6:raise RuntimeError('Unexpected tensor field length '+name)
        vals[lab]=raw[order]
    if set(vals)!=set(elabels):raise RuntimeError('%s element labels missing/excess: %s'%(name,len(vals)))
    return np.array([vals[int(k)] for k in elabels],dtype=np.float64)

def read_one(region):
    from odbAccess import openOdb
    f=ODBS/('B9A3_'+region+'.odb')
    odb=openOdb(path=str(f.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step')
        frame=odb.steps['Loading'].frames[-1]
        if abs(float(frame.frameValue)-1.)>1.e-7:raise RuntimeError('Incorrect final frame')
        for name in ('U','RF','S','LE'):
            if name not in frame.fieldOutputs:raise RuntimeError('Missing '+name+' ODB field')
        instance=None
        for name,obj in odb.rootAssembly.instances.items():
            if name.lower()=='block-1':instance=obj
        if instance is None:raise RuntimeError('NO-side BLOCK-1 instance not found')
        nodes=list(instance.nodes);elements=list(instance.elements)
        labels=np.array([int(v.label) for v in nodes],dtype=np.int64)
        xyz=np.array([np.asarray(v.coordinates,dtype=np.float64)[:3] for v in nodes],dtype=np.float64)
        elem_labels=np.array([int(v.label) for v in elements],dtype=np.int64)
        elem_conns=np.array([np.asarray(v.connectivity,dtype=np.int64) for v in elements],dtype=np.int64)
        if xyz.shape!=(1008,3) or elem_conns.shape!=(728,8):
            raise RuntimeError('Expected 1008 nodes, 728 C3D8R elements for '+region)
        nodal={}
        for fieldname in ('U','RF'):
            record={}
            for val in frame.fieldOutputs[fieldname].getSubset(region=instance).values:
                k=int(val.nodeLabel)
                if k in record:raise RuntimeError('Duplicate '+fieldname+' node '+str(k))
                record[k]=np.asarray(val.data,dtype=np.float64).ravel()[:3]
            missing=set(labels)-set(record)
            if fieldname=='U' and missing:
                raise RuntimeError('U field missing '+str(len(missing))+' nodes')
            if fieldname=='RF':
                xmin,xmax=float(np.min(xyz[:,0])),float(np.max(xyz[:,0]))
                boundary=set(int(labels[i]) for i in range(len(labels)) if abs(xyz[i,0]-xmin)<1e-5 or abs(xyz[i,0]-xmax)<1e-5)
                if boundary-set(record):raise RuntimeError('RF field missing '+str(len(boundary-set(record)))+' constrained interface nodes')
            nodal[fieldname]=np.array([record.get(int(x),np.zeros(3)) for x in labels],dtype=np.float64)
        S=field_tensor(frame,instance,'S',elem_labels)
        LE=field_tensor(frame,instance,'LE',elem_labels)
        index={int(l):i for i,l in enumerate(labels)}
        if len(index)!=len(labels) or len(set(elem_labels))!=len(elem_labels):raise RuntimeError('Duplicate nodes or elements')
        if not set(elem_conns.ravel()).issubset(index):raise RuntimeError('Unknown element nodes')
        ip_xyz=np.array([xyz[[index[int(n)] for n in con]].mean(axis=0) for con in elem_conns])
        vals=[xyz,nodal['U'],nodal['RF'],ip_xyz,S,LE]
        if not all(np.isfinite(a).all() for a in vals):raise RuntimeError('Nonfinite Abaqus fields')
        return dict(node_labels=labels,node_xyz=xyz,U=nodal['U'],RF=nodal['RF'],
                    element_labels=elem_labels,connectivity_labels=elem_conns,
                    ip_xyz=ip_xyz,S=S,LE=LE),dict(region=region,odb=str(f),sha256=sha(f),
                    frame=float(frame.frameValue),nodes=len(labels),elements=len(elem_labels),
                    fields=['U','RF','S','LE'],stress_components=list(COMPONENTS))
    finally:odb.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('check','export'));a=p.parse_args()
    h,meta=prerequisites()
    print('B9A4 Windows CHECK PASS | B9A3 jobs/anchor verified | no new jobs')
    if a.stage=='check':return
    if not OUT.is_dir():raise RuntimeError('In Windows File Explorer create ODBS\\B9A4_exports first')
    out=OUT/'B9A4_reference_fields.npz';manifest=OUT/'B9A4_windows_summary.json'
    if out.exists() or manifest.exists():raise RuntimeError('Existing B9A4 files protected: '+str(OUT))
    arrays={'anchor_sha256':np.array(h),'b9a3_reference_sha256':np.array(sha(REF))}
    audits=[]
    for name in REGIONS:
        d,au=read_one(name);arrays.update({name+'_'+k:v for k,v in d.items()});audits.append(au)
    np.savez_compressed(str(out),**arrays)
    rep=dict(step='65AM-B9A4',anchor_sha256=h,b9a3_reference_sha256=sha(REF),
             reference_fields_sha256=sha(out),cases=audits,
             note='Existing NO-side Abaqus ODB fields only; not B5 predictions; not equilibrium proof')
    with manifest.open('w') as f:json.dump(rep,f,indent=2)
    print('B9A4 EXPORT PASS |',out,'|',manifest)

if __name__=='__main__':main()
