# -*- coding: utf-8 -*-
"""B9A10-D (Windows, Abaqus Python): READ ONLY ODB local stress orientations.
Copy to ODBS/scripts. Outputs under ODBS/B9A10D_exports (create with Explorer).
Stages: --stage check, --stage export. Does not edit ODBs or older exports.
"""
from __future__ import print_function
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ODBDIR=ROOT/'B9A10C_ODB'
OUTDIR=ROOT/'B9A10D_exports'
OLDMETA=ROOT/'B9A10C_exports'/'B9A10C_windows_summary.json'
JOBS=['B9A10C_%s_%s_%s'%(reg,typ,bc)
      for reg in ('L','R') for typ in ('R','F') for bc in ('FREE','REPLAY')]

def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def check():
    if not OLDMETA.is_file():raise RuntimeError('Cannot find prior B9A10C_windows_summary.json')
    data=json.loads(OLDMETA.read_text())
    lookup={x['job']:x for x in data['jobs']}
    if set(lookup)!=set(JOBS):raise RuntimeError('B9A10C reference manifest does not contain all eight jobs')
    for job in JOBS:
        p=ODBDIR/(job+'.odb')
        if not p.is_file():raise RuntimeError('Missing original ODB: '+str(p))
        if sha(p)!=lookup[job]['odb_sha256']:
            raise RuntimeError('B9A10C ODB changed since original force comparison: '+job)
    print('B9A10D CHECK PASS: eight read-only reference ODBs match B9A10C checksums')
    return lookup

def export():
    lookup=check()
    if not OUTDIR.is_dir():
        raise RuntimeError('In Windows File Explorer, create '+str(OUTDIR))
    npzpath=OUTDIR/'B9A10D_orientations.npz'
    metapath=OUTDIR/'B9A10D_windows_summary.json'
    if npzpath.exists() or metapath.exists():
        raise RuntimeError('Existing B9A10D outputs protected; move the old results manually')
    from odbAccess import openOdb
    allarrays={};reports=[]
    for job in JOBS:
        odb=openOdb(path=str((ODBDIR/(job+'.odb')).resolve()),readOnly=True)
        try:
            if 'Loading' not in odb.steps:raise RuntimeError('Missing step Loading: '+job)
            fr=odb.steps['Loading'].frames[-1]
            inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
            if inst is None:raise RuntimeError('Missing BLOCK-1 instance: '+job)
            if 'S' not in fr.fieldOutputs:raise RuntimeError('Missing stress tensor S: '+job)
            values=fr.fieldOutputs['S'].getSubset(region=inst).values
            nip=1 if job.split('_')[2]=='R' else 8
            qmap={};missing=0
            for v in values:
                k=(int(v.elementLabel),int(v.integrationPoint))
                if k in qmap:raise RuntimeError('Duplicate stress IP: '+str(k))
                q=None
                for name in ('localCoordSystem','localCoordSystemDouble'):
                    try:
                        raw=getattr(v,name)
                        trial=np.asarray(raw,dtype=np.float64)
                        if trial.size==9:q=trial.reshape(3,3);break
                    except Exception:pass
                if q is None:missing+=1;continue
                if not np.isfinite(q).all():raise RuntimeError('Nonfinite orientation '+job+' '+str(k))
                if np.max(np.abs(q.dot(q.T)-np.eye(3)))>1e-3 or abs(np.linalg.det(q)-1.0)>1e-3:
                    raise RuntimeError('Nonorthogonal local orientation: '+job+' '+str(k))
                qmap[k]=q
            elements=sorted([int(e.label) for e in inst.elements])
            if len(elements)!=728 or len(values)!=728*nip:
                raise RuntimeError('Unexpected number of element stress points: '+job)
            missing_expected=728*nip-len(qmap)
            if missing!=missing_expected:raise RuntimeError('Orientation data count mismatch')
            qarr=np.full((728,nip,3,3),np.nan,dtype=np.float64)
            has=np.zeros((728,nip),dtype=np.uint8)
            for i,e in enumerate(elements):
                for ip in range(1,nip+1):
                    if (e,ip) in qmap:qarr[i,ip-1]=qmap[(e,ip)];has[i,ip-1]=1
            allarrays[job+'_element_labels']=np.asarray(elements,np.int64)
            allarrays[job+'_Q_global_to_local']=qarr
            allarrays[job+'_Q_available']=has
            Qgood=qarr[np.asarray(has,dtype=bool)]
            if Qgood.size:
                rotation=np.arccos(np.clip((np.trace(Qgood,axis1=1,axis2=2)-1)/2,-1.,1.))
                maxrot=float(np.rad2deg(np.max(rotation)))
                meanrot=float(np.rad2deg(np.mean(rotation)))
            else:maxrot=None;meanrot=None
            reports.append(dict(job=job,n_integration_points=728*nip,orientation_count=int(has.sum()),
                absent_count=int((~has.astype(bool)).sum()),max_rotation_deg=maxrot,
                mean_rotation_deg=meanrot,odb_sha256=lookup[job]['odb_sha256']))
            print(job,': orientations',int(has.sum()),'/',has.size,'max angle deg=',maxrot)
        finally:odb.close()
    np.savez_compressed(str(npzpath),**allarrays)
    metapath.write_text(json.dumps({'step':'65AM-B9A10-D','source':'read-only B9A10C ODB stress localCoordSystem',
      'localCoordSystem_convention':'rotation from global coordinates to local coordinates',
      'prior_reference_sha256':sha(OLDMETA),'export_sha256':sha(npzpath),'cases':reports,
      'orientation_absent_is_not_identity':True,'models_unchanged':True},indent=2))
    print('B9A10D EXPORT PASS:',str(npzpath),str(metapath))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('check','export'),required=True)
    a=p.parse_args()
    if a.stage=='check':check()
    else:export()
