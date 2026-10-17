# -*- coding: utf-8 -*-
"""B9A10-H Windows Abaqus: READ-ONLY export of 24 existing B9A9 C3D8R cases.

All scripts run from ODBS root. Nothing in previous directories is changed.
Stages: --stage selftest | check | export
Requires B9A9_ODB and B9A9_exports/B9A9_windows_summary.json.
Make B9A10H_exports folder manually in Windows Explorer before export.
Compatible with Abaqus Python 3 and NumPy.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ODBS=ROOT/'B9A9_ODB'
MANIFEST=ROOT/'B9A9_exports'/'B9A9_windows_summary.json'
OUT=ROOT/'B9A10H_exports'
JOBS=['B9A9_%s%02d'%(r,i) for r in ('L','R') for i in range(1,13)]


def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def selftest():
    q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    s=np.array([[3.,.6,0.],[.6,2.,.2],[0.,.2,1.]])
    assert np.allclose(q.T@(q@s@q.T)@q,s)
    assert np.allclose(q@q.T,np.eye(3))
    print('B9A10H WINDOWS SELFTEST PASS | matrix orientation')


def check():
    if not MANIFEST.is_file(): raise RuntimeError('Missing '+str(MANIFEST))
    summary=json.loads(MANIFEST.read_text())
    prior={r['job']:r for r in summary['jobs']}
    if set(prior)!=set(JOBS) or not summary.get('all_jobs_pass',False):
        raise RuntimeError('Previous B9A9 manifest is incomplete or not successful')
    for name in JOBS:
        p=ODBS/(name+'.odb'); s=ODBS/(name+'.sta')
        if not p.is_file() or not s.is_file():raise RuntimeError('Missing B9A9 ODB/STA '+name)
        if 'COMPLETED SUCCESSFULLY' not in s.read_text(errors='ignore').upper():
            raise RuntimeError('Not completed successfully '+name)
        if sha(p)!=prior[name]['odb_sha256']:
            raise RuntimeError('Historical ODB checksum differs: '+name)
    print('B9A10H WINDOWS CHECK PASS | 24 original B9A9 ODB checksums verified | no new jobs')
    return prior


def fetch(frame,inst,name):
    if name not in frame.fieldOutputs:raise RuntimeError('Missing field '+name)
    values=frame.fieldOutputs[name].getSubset(region=inst).values
    out={}
    for v in values:
        label=int(v.nodeLabel)
        if label in out:raise RuntimeError('Duplicate '+name+' node '+str(label))
        out[label]=np.asarray(v.data,dtype=float).reshape(-1)[:3]
    return out


def read_job(name):
    from odbAccess import openOdb
    path=ODBS/(name+'.odb')
    odb=openOdb(path=str(path.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step '+name)
        frame=odb.steps['Loading'].frames[-1]
        if abs(float(frame.frameValue)-1.)>1e-6:raise RuntimeError('Unexpected last frame '+name)
        inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
        if inst is None:raise RuntimeError('Cannot find BLOCK-1 in '+name)
        nodes=sorted(inst.nodes,key=lambda x:int(x.label))
        els=sorted(inst.elements,key=lambda x:int(x.label))
        if len(nodes)!=1008 or len(els)!=728:raise RuntimeError('Mesh mismatch '+name)
        labels=np.asarray([int(v.label) for v in nodes],dtype=np.int64)
        x=np.asarray([v.coordinates for v in nodes],dtype=float)
        elabel=np.asarray([int(e.label) for e in els],dtype=np.int64)
        conn=np.asarray([list(e.connectivity) for e in els],dtype=np.int64)
        u=fetch(frame,inst,'U');rf=fetch(frame,inst,'RF')
        if set(u)!=set(labels.tolist()) or set(rf)!=set(labels.tolist()):
            raise RuntimeError('Incomplete nodal U/RF '+name)
        sfield=frame.fieldOutputs['S'].getSubset(region=inst).values
        smap={};qmap={}
        for v in sfield:
            key=(int(v.elementLabel),int(v.integrationPoint))
            if key in smap:raise RuntimeError('Duplicate integration-point stress '+str(key))
            arr=np.asarray(v.data,dtype=float).reshape(-1)
            if len(arr)<6:raise RuntimeError('Stress S lacks 6 components '+name)
            smap[key]=arr[:6]
            try: q=np.asarray(v.localCoordSystem,dtype=float).reshape(3,3)
            except Exception as ex:raise RuntimeError('Missing localCoordSystem (do not substitute identity): '+name+' '+str(key)+' '+str(ex))
            if np.max(np.abs(q@q.T-np.eye(3)))>1e-3 or abs(np.linalg.det(q)-1)>1e-3:
                raise RuntimeError('Non-rotation localCoordSystem '+name+' '+str(key))
            qmap[key]=q
        if len(smap)!=728:raise RuntimeError('C3D8R must have exactly one stress point per element: '+name)
        sarr=np.asarray([smap[(int(e),1)] for e in elabel],dtype=float)
        qarr=np.asarray([qmap[(int(e),1)] for e in elabel],dtype=float)
        arrays=dict(node_labels=labels,X=x,U=np.asarray([u[int(t)] for t in labels]),
                    RF=np.asarray([rf[int(t)] for t in labels]),element_labels=elabel,
                    connectivity=conn,S=sarr,Q_global_to_local=qarr)
        return arrays,dict(job=name,region='inner_left' if '_L' in name else 'inner_right',
                           odb_sha256=sha(path),n_nodes=len(nodes),n_elements=len(els),
                           orientations=len(qmap),frame=float(frame.frameValue))
    finally:odb.close()


def export():
    prior=check()
    if not OUT.is_dir():raise RuntimeError('Create folder using Windows Explorer: '+str(OUT))
    target=OUT/'B9A10H_independent_fields.npz'
    meta=OUT/'B9A10H_windows_summary.json'
    if target.exists() or meta.exists():raise RuntimeError('Outputs already exist; refusing overwrite')
    payload={};reports=[]
    first_mesh={}
    for name in JOBS:
        a,r=read_job(name)
        if r['odb_sha256']!=prior[name]['odb_sha256']:raise RuntimeError('ODB changed during reading '+name)
        side='L' if '_L' in name else 'R'
        if side not in first_mesh:
            first_mesh[side]=(a['node_labels'],a['X'],a['element_labels'],a['connectivity'])
        else:
            ref=first_mesh[side]
            for key,val,old in zip(('node_labels','X','element_labels','connectivity'),
                                    (a['node_labels'],a['X'],a['element_labels'],a['connectivity']),ref):
                if not np.allclose(val,old,rtol=0,atol=2e-5 if key=='X' else 0):
                    raise RuntimeError('Mesh changed from case to case: '+name+' '+key)
        for key,val in a.items():payload[name+'_'+key]=val
        reports.append(r);print('READ',name,'nodes',r['n_nodes'],'IP rotations',r['orientations'])
    # write only after all checks finish; never alter historical sources
    np.savez_compressed(str(target),**payload)
    meta.write_text(json.dumps({'step':'65AM-B9A10-H','source':'24 independent B9A9 Abaqus ODBs',
        'original_B9A9_manifest_sha256':sha(MANIFEST),'output_sha256':sha(target),
        'rotation_convention':'global_to_local','new_simulations_required':False,
        'cases':reports,'models_untouched':True},indent=2))
    print('B9A10H EXPORT PASS:',target,meta)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','export'))
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':check()
    else:export()
