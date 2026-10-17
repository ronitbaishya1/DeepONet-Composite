# -*- coding: utf-8 -*-
"""B9A10-G Windows Abaqus: read-only LE export from B9A10F ODBs.
Use Abaqus Python from ODBS root. Existing simulations are NEVER modified.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
ODBS=ROOT/'B9A10F_ODB'
SOURCE=ROOT/'B9A10F_exports'
DEST=ROOT/'B9A10G_exports'
ANGLES=('Z00','P17','M17','ISO')
AMPS=('A00','A10','A50','A100','LIN')
JOBS=['B9A10F_F_%s_%s'%(a,b) for a in ANGLES for b in AMPS]+['B9A10F_R_P17_A100']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()

def check():
    for d in (ODBS,SOURCE,DEST):
        if not d.is_dir():raise RuntimeError('Create or verify folder with File Explorer: '+str(d))
    original=SOURCE/'B9A10F_windows_summary.json'
    base=SOURCE/'B9A10F_patch_fields.npz'
    if not original.is_file() or not base.is_file():raise RuntimeError('Missing original B9A10F Windows exports')
    m=json.loads(original.read_text())
    if m.get('case_count')!=21 or m.get('exports_sha256')!=sha(base):
        raise RuntimeError('B9A10F original exports not verified')
    refs={x['job']:x for x in m['cases']}
    if set(refs)!=set(JOBS):raise RuntimeError('B9A10F case list mismatch')
    for j in JOBS:
        p=ODBS/(j+'.odb')
        if not p.is_file() or sha(p)!=refs[j]['odb_sha256']:
            raise RuntimeError('Missing/changed Abaqus ODB: '+str(p))
    print('B9A10G CHECK PASS | original B9A10F ODB SHA verified | 21 cases | no jobs needed')
    return m

def extract_case(j):
    from odbAccess import openOdb
    p=ODBS/(j+'.odb');odb=openOdb(str(p.resolve()),readOnly=True)
    try:
        frame=odb.steps['Loading'].frames[-1]
        if 'LE' not in frame.fieldOutputs:raise RuntimeError('LE not saved in '+j)
        inst=next((v for n,v in odb.rootAssembly.instances.items() if n.lower()=='patch-1'),None)
        if inst is None or len(inst.elements)!=1:raise RuntimeError('Wrong instance in '+j)
        field=frame.fieldOutputs['LE'].getSubset(region=inst)
        mapping={}
        for v in field.values:
            n=int(v.integrationPoint)
            if n in mapping:raise RuntimeError('Duplicate LE IP in '+j)
            data=np.asarray(v.data,float).reshape(-1)
            if data.shape!=(6,):raise RuntimeError('Expected LE11,22,33,12,13,23 in '+j)
            try:
                q=np.asarray(v.localCoordSystem,float).reshape(3,3)
                has=True
            except Exception:
                q=np.eye(3);has=False
            if abs(float(np.linalg.det(q))-1)>1e-3 or np.max(np.abs(q@q.T-np.eye(3)))>1e-3:
                raise RuntimeError('Invalid LE frame in '+j)
            mapping[n]=(data,q,has)
        nip=1 if '_R_' in j else 8
        if set(mapping)!=set(range(1,nip+1)):
            raise RuntimeError('Missing LE integration points in '+j)
        return (np.array([mapping[i][0] for i in range(1,nip+1)]),
                np.array([mapping[i][1] for i in range(1,nip+1)]),
                np.array([mapping[i][2] for i in range(1,nip+1)],dtype=np.uint8))
    finally:odb.close()

def export():
    original=check()
    a=DEST/'B9A10G_LE_fields.npz'; b=DEST/'B9A10G_windows_summary.json'
    if a.exists() or b.exists():raise RuntimeError('B9A10G outputs already exist; will not overwrite')
    data={};rows=[]
    for j in JOBS:
        le,q,flag=extract_case(j)
        data[j+'_LE']=le;data[j+'_LE_Q']=q;data[j+'_LE_Q_available']=flag
        rows.append({'job':j,'n_ip':len(le),'LE_frame_count':int(flag.sum())})
        print('LE EXPORTED',j,'IP',len(le),'LE frames',int(flag.sum()))
    np.savez_compressed(str(a),**data)
    b.write_text(json.dumps({'step':'65AM-B9A10-G','cases':rows,'case_count':21,
                             'LE_sha256':sha(a),'source_B9A10F_npz_sha256':original['exports_sha256'],
                             'old_models_untouched':True},indent=2))
    print('B9A10G DONE:',a)

def selftest():
    assert len(JOBS)==21 and len(set(JOBS))==21
    assert all(j.startswith('B9A10F_') for j in JOBS)
    assert sum('_R_' in j for j in JOBS)==1
    print('B9A10G WINDOWS SELFTEST PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('selftest','check','export'))
    s=p.parse_args().stage
    if s=='selftest':selftest()
    elif s=='check':check()
    else:export()
