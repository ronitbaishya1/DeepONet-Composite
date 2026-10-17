# -*- coding: utf-8 -*-
"""B9A10-M: READ-ONLY Abaqus/Standard ODB loading-history exporter.

Run in Abaqus Python 2025, from ODBS root:
  abaqus python scripts\65AM_B9A10M_W_history_export.py --stage selftest
  abaqus python scripts\65AM_B9A10M_W_history_export.py --stage inspect
  abaqus python scripts\65AM_B9A10M_W_history_export.py --stage export

Requires B9A9_ODB, B9A10H_exports/B9A10H_windows_summary.json,
and manually created B9A10M_exports. Does not modify previous jobs or input decks.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ODBS=ROOT/'B9A9_ODB'
SOURCE=ROOT/'B9A10H_exports'/'B9A10H_windows_summary.json'
OUT=ROOT/'B9A10M_exports'
JOBS=['B9A9_%s%02d'%(region,i) for region in ('L','R') for i in range(1,13)]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def provenance():
    if not SOURCE.is_file():raise FileNotFoundError(str(SOURCE))
    doc=json.loads(SOURCE.read_text())
    expected={row['job']:row['odb_sha256'] for row in doc['cases']}
    if set(expected)!=set(JOBS):raise RuntimeError('B9A10H reference has wrong cases')
    for job in JOBS:
        odb=ODBS/(job+'.odb');sta=ODBS/(job+'.sta')
        if not odb.is_file() or not sta.is_file():raise RuntimeError('Missing ODB/STA: '+job)
        if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
            raise RuntimeError('Incomplete Abaqus job '+job)
        if sha(odb)!=expected[job]:raise RuntimeError('Historical ODB changed: '+job)
    print('B9A10M PROVENANCE PASS | 24 unchanged ODBs')
    return expected


def inspect_one(job):
    from odbAccess import openOdb
    odb=openOdb(path=str((ODBS/(job+'.odb')).resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Loading step missing '+job)
        frames=odb.steps['Loading'].frames
        result=[]
        for n,f in enumerate(frames):
            result.append(dict(index=n,time=float(f.frameValue),
                       fields=[s for s in ('U','RF','S','LE') if s in f.fieldOutputs]))
        usable=[r for r in result if r['time']>1e-10 and all(t in r['fields'] for t in ('U','S','LE'))]
        sufficient=(len(usable)>=2 and abs(usable[-1]['time']-1.)<1e-6 and
                    all(usable[i+1]['time']>usable[i]['time'] for i in range(len(usable)-1)))
        print('%s: %d total frames, %d usable increments, %s'%(
              job,len(result),len(usable),'HISTORY PASS' if sufficient else 'INSUFFICIENT HISTORY'))
        return dict(job=job,frames=result,usable_count=len(usable),sufficient=sufficient)
    finally:odb.close()


def _node_field(frame,inst,name,labels):
    vals=frame.fieldOutputs[name].getSubset(region=inst).values
    d={}
    for v in vals:
        lab=int(v.nodeLabel)
        if lab in d:raise RuntimeError('Duplicate '+name+' node '+str(lab))
        d[lab]=np.asarray(v.data,dtype=float).reshape(-1)[:3]
    if set(d)!=set(labels):raise RuntimeError('Incomplete '+name+' node set')
    return np.asarray([d[lab] for lab in labels],dtype=np.float64)


def _ip_field(frame,inst,name,labels,require_orientation=False):
    vals=frame.fieldOutputs[name].getSubset(region=inst).values
    d={};q={}
    for v in vals:
        key=(int(v.elementLabel),int(v.integrationPoint))
        if key in d:raise RuntimeError('Duplicate '+name+' Gauss point '+str(key))
        data=np.asarray(v.data,dtype=np.float64).reshape(-1)
        if data.size<6:raise RuntimeError('Wrong '+name+' components '+str(key))
        d[key]=data[:6]
        if require_orientation:
            try:Q=np.asarray(v.localCoordSystem,dtype=np.float64).reshape(3,3)
            except Exception as exc:raise RuntimeError('Missing S local orientation '+str(key)+' '+str(exc))
            if np.max(np.abs(Q@Q.T-np.eye(3)))>1e-3 or abs(np.linalg.det(Q)-1)>1e-3:
                raise RuntimeError('Bad S local orientation '+str(key))
            q[key]=Q
    expected={(lab,1) for lab in labels}
    if set(d)!=expected:raise RuntimeError('Wrong integration point count for '+name)
    result=np.asarray([d[(lab,1)] for lab in labels],dtype=np.float64)
    if require_orientation:
        Qout=np.asarray([q[(lab,1)] for lab in labels],dtype=np.float64)
        return result,Qout
    return result


def extract_one(job,inspected):
    from odbAccess import openOdb
    if not inspected['sufficient']:
        raise RuntimeError('Not enough saved field history in '+job+'; do not interpolate or invent frames')
    odb=openOdb(path=str((ODBS/(job+'.odb')).resolve()),readOnly=True)
    try:
        inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
        if inst is None:raise RuntimeError('BLOCK-1 missing '+job)
        nodes=sorted(int(n.label) for n in inst.nodes)
        els=sorted(int(e.label) for e in inst.elements)
        if len(nodes)!=1008 or len(els)!=728:raise RuntimeError('Unexpected mesh '+job)
        times=[0.];u_hist=[np.zeros((len(nodes),3),np.float64)];S_hist=[];LE_hist=[];Q_hist=[]
        for row in inspected['frames']:
            if row['time']<=1e-10:continue
            if not all(x in row['fields'] for x in ('U','S','LE')):
                raise RuntimeError('Missing intermediate U/S/LE; no silent frame skipping in '+job)
            f=odb.steps['Loading'].frames[row['index']]
            times.append(float(f.frameValue))
            u_hist.append(_node_field(f,inst,'U',nodes))
            ss,qq=_ip_field(f,inst,'S',els,True)
            S_hist.append(ss)
            LE_hist.append(_ip_field(f,inst,'LE',els,False))
            Q_hist.append(qq)
        if len(times)<3 or abs(times[-1]-1.)>1e-6:
            raise RuntimeError('Missing useful loading history '+job)
        # At start of static load step, source B9A9 deck has no pre-stress.
        sh=np.concatenate([np.zeros((1,len(els),6)),np.asarray(S_hist)],axis=0)
        lh=np.concatenate([np.zeros((1,len(els),6)),np.asarray(LE_hist)],axis=0)
        qh=np.concatenate([np.tile(np.eye(3)[None,None],(1,len(els),1,1)),np.asarray(Q_hist)],axis=0)
        arrays=dict(times=np.asarray(times),U=np.asarray(u_hist),S=sh,LE=lh,Q=qh,
                    node_labels=np.asarray(nodes,dtype=np.int64),element_labels=np.asarray(els,dtype=np.int64))
        if not all(np.isfinite(x).all() for x in arrays.values()):raise RuntimeError('Nonfinite fields '+job)
        if max(np.linalg.norm(arrays['U'][0],axis=-1))>1e-12:
            raise RuntimeError('Nonzero artificial initial displacement')
        return arrays
    finally:odb.close()


def selftest():
    assert len(JOBS)==24 and len(set(JOBS))==24
    q=np.eye(3)
    assert np.allclose(q.T@q,np.eye(3))
    print('B9A10M WINDOWS SELFTEST PASS')


def inspect():
    hashes=provenance()
    rows=[inspect_one(j) for j in JOBS]
    print('HISTORY SUMMARY: %d/24 usable'%sum(x['sufficient'] for x in rows))
    if OUT.is_dir():
        p=OUT/'B9A10M_history_inspection.json'
        if not p.exists():p.write_text(json.dumps(dict(step='B9A10M',cases=rows,source_hashes=hashes),indent=2))
        else:print('Prior inspection protected; not overwritten')
    return rows


def export():
    provenance()
    if not OUT.is_dir():raise RuntimeError('Create with File Explorer: '+str(OUT))
    target=OUT/'B9A10M_history_fields.npz';metadata=OUT/'B9A10M_windows_summary.json'
    if target.exists() or metadata.exists():raise RuntimeError('B9A10M export already exists; will not overwrite')
    rows=[inspect_one(j) for j in JOBS]
    missing=[x['job'] for x in rows if not x['sufficient']]
    if missing:raise RuntimeError('Insufficient saved history in '+', '.join(missing)+'. Send inspection JSON; do not rerun all jobs.')
    blobs={};info=[]
    for row in rows:
        job=row['job'];a=extract_one(job,row)
        for key,value in a.items():blobs[job+'_'+key]=value
        info.append(dict(job=job,frames=len(a['times']),times=a['times'].tolist(),odb_sha256=sha(ODBS/(job+'.odb'))))
        print('EXPORTED',job,'frames',len(a['times']))
    np.savez_compressed(str(target),**blobs)
    metadata.write_text(json.dumps(dict(step='B9A10M',jobs=info,export_sha256=sha(target),
                    source_B9A10H_summary_sha256=sha(SOURCE),new_simulations=False,
                    fields_diagnostic_only=['S','LE','Q'],models_untouched=True),indent=2))
    print('B9A10M EXPORT COMPLETE:',target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('selftest','inspect','export'))
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='inspect':inspect()
    else:export()
