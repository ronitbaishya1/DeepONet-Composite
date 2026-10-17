# -*- coding: utf-8 -*-
"""B9A10-L Windows/Abaqus 2025: read-only LE extraction for 24 B9A9 C3D8R jobs.

From C:\\Users\\rrb225_lehigh\\Pictures\\ODBS:
 abaqus python scripts\\65AM_B9A10L_W_strain_export.py --stage selftest
 abaqus python scripts\\65AM_B9A10L_W_strain_export.py --stage check
 abaqus python scripts\\65AM_B9A10L_W_strain_export.py --stage export

No new Abaqus jobs. Does not change ODB or prior exports.
"""
from __future__ import print_function
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ODBS=ROOT/'B9A9_ODB'
OLD=ROOT/'B9A10H_exports'/'B9A10H_windows_summary.json'
DEST=ROOT/'B9A10L_exports'
JOBS=['B9A9_%s%02d'%(r,i) for r in ('L','R') for i in range(1,13)]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def selftest():
    assert len(JOBS)==24 and len(set(JOBS))==24
    assert all(x.startswith('B9A9_') for x in JOBS)
    print('B9A10L WINDOWS SELFTEST PASS | 24 independent ODBs')

def check():
    if not OLD.is_file():raise FileNotFoundError('Missing B9A10H Windows summary: '+str(OLD))
    source=json.loads(OLD.read_text())
    previous={c['job']:c['odb_sha256'] for c in source['cases']}
    if set(previous)!=set(JOBS):raise RuntimeError('Wrong provenance in H Windows summary')
    for j in JOBS:
        p=ODBS/(j+'.odb');sta=ODBS/(j+'.sta')
        if not p.is_file() or not sta.is_file():raise FileNotFoundError('Missing '+j+'.odb / .sta')
        if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
            raise RuntimeError('Incomplete ODB '+j)
        if sha(p)!=previous[j]:raise RuntimeError('ODB SHA changed since B9A10H: '+j)
    print('B9A10L CHECK PASS | 24 verified unchanged C3D8R ODBs | no new Abaqus jobs')
    return previous

def read_odb(j):
    from odbAccess import openOdb
    p=ODBS/(j+'.odb')
    odb=openOdb(path=str(p.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('No Loading step '+j)
        f=odb.steps['Loading'].frames[-1]
        if abs(float(f.frameValue)-1)>1e-6:raise RuntimeError('Wrong final frame '+j)
        if 'LE' not in f.fieldOutputs:raise RuntimeError('LE was not saved in '+j+'; stop and report')
        inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
        if inst is None:raise RuntimeError('BLOCK-1 missing '+j)
        if len(inst.elements)!=728:raise RuntimeError('Wrong element count '+j)
        labels=sorted(int(e.label) for e in inst.elements)
        le={};qmap={};present=[]
        for v in f.fieldOutputs['LE'].getSubset(region=inst).values:
            key=(int(v.elementLabel),int(v.integrationPoint))
            if key in le:raise RuntimeError('Duplicate LE point '+str(key)+' '+j)
            data=np.asarray(v.data,dtype=float).reshape(-1)
            if data.shape!=(6,) or not np.isfinite(data).all():raise RuntimeError('Invalid LE '+str(key))
            le[key]=data
            try:
                Q=np.asarray(v.localCoordSystem,dtype=float).reshape(3,3)
                exists=True
            except Exception:
                Q=np.eye(3);exists=False
            if exists and (np.max(np.abs(Q@Q.T-np.eye(3)))>1e-3 or abs(np.linalg.det(Q)-1)>1e-3):
                raise RuntimeError('LE local orientation invalid '+j)
            qmap[key]=Q;present.append((key,exists))
        expected={(lab,1) for lab in labels}
        if set(le)!=expected:raise RuntimeError('Missing/extra integration points '+j)
        if len([a for a,flag in present if flag])!=728:
            raise RuntimeError('Missing local LE directions: '+j)
        return (np.asarray(labels,dtype=np.int64),
                np.asarray([le[(lab,1)] for lab in labels],dtype=float),
                np.asarray([qmap[(lab,1)] for lab in labels],dtype=float))
    finally:
        odb.close()

def export():
    prev=check()
    if not DEST.is_dir():raise RuntimeError('Create with File Explorer: '+str(DEST))
    f=DEST/'B9A10L_LE_fields.npz';meta=DEST/'B9A10L_windows_summary.json'
    if f.exists() or meta.exists():raise RuntimeError('B9A10L output already exists; do not overwrite')
    payload={};rows=[]
    for j in JOBS:
        lab,le,Q=read_odb(j)
        payload[j+'_element_labels']=lab
        payload[j+'_LE']=le
        payload[j+'_LE_Q_global_to_local']=Q
        rows.append({'job':j,'odb_sha256':prev[j],'n_elements':len(lab),'strain_field':'LE'})
        print('EXPORTED LE',j,'IP',len(lab))
    np.savez_compressed(str(f),**payload)
    meta.write_text(json.dumps(dict(step='65AM-B9A10-L',case_count=24,cases=rows,
       file_sha256=sha(f),source_H_manifest_sha256=sha(OLD),
       read_only_source=True,no_new_odb_jobs=True),indent=2))
    print('B9A10L EXPORT PASS:',f)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('selftest','check','export'))
    a=ap.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':check()
    else:export()
