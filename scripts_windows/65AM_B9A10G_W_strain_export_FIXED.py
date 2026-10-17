# -*- coding: utf-8 -*-
"""B9A10-G Windows extraction, corrected for NLGEOM=NO strain output.

Abaqus/Standard finite deformation -> LE, geometrically linear -> E.
Read-only existing B9A10F ODBs; optional four NEW E-output runs if E was not saved.
No existing B9A10F, B5, B9A7 or 65V file is modified.

From ODBS root:
  abaqus python scripts\65AM_B9A10G_W_strain_export_FIXED.py --stage selftest
  abaqus python scripts\65AM_B9A10G_W_strain_export_FIXED.py --stage check
  abaqus python scripts\65AM_B9A10G_W_strain_export_FIXED.py --stage inspect
  abaqus python scripts\65AM_B9A10G_W_strain_export_FIXED.py --stage prepare-linear
  [only if advised, run freshly generated B9A10G_F_*_LIN_E jobs]
  abaqus python scripts\65AM_B9A10G_W_strain_export_FIXED.py --stage export
"""
from __future__ import print_function
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ODBS = ROOT / 'B9A10F_ODB'
SOURCE = ROOT / 'B9A10F_exports'
DEST = ROOT / 'B9A10G_exports'
LINEAR = ROOT / 'B9A10G_linear_ODB'
ANGLES = ('Z00', 'P17', 'M17', 'ISO')
AMPS = ('A00', 'A10', 'A50', 'A100', 'LIN')
JOBS = ['B9A10F_F_%s_%s' % (a,b) for a in ANGLES for b in AMPS] + ['B9A10F_R_P17_A100']


def is_linear(job):
    return job.endswith('_LIN')


def expected_field(job):
    return 'E' if is_linear(job) else 'LE'


def corrected_job(job):
    return job.replace('B9A10F_', 'B9A10G_', 1) + '_E'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check():
    for d in (ODBS, SOURCE, DEST):
        if not d.is_dir():
            raise RuntimeError('Create/verify folder with File Explorer: '+str(d))
    mpath=SOURCE/'B9A10F_windows_summary.json'
    base=SOURCE/'B9A10F_patch_fields.npz'
    if not mpath.is_file() or not base.is_file():
        raise RuntimeError('Missing original B9A10F export(s)')
    m=json.loads(mpath.read_text())
    if m.get('case_count')!=21 or m.get('exports_sha256')!=sha(base):
        raise RuntimeError('Original B9A10F export failed SHA/metadata verification')
    refs={x['job']:x for x in m['cases']}
    if set(refs)!=set(JOBS):
        raise RuntimeError('B9A10F case list mismatch')
    for job in JOBS:
        p=ODBS/(job+'.odb')
        if not p.is_file() or sha(p)!=refs[job]['odb_sha256']:
            raise RuntimeError('Missing/changed B9A10F ODB: '+str(p))
    print('B9A10G CHECK PASS | 21 original ODB SHA verified | read-only')
    return m


def get_fields(path):
    from odbAccess import openOdb
    odb=openOdb(str(path.resolve()), readOnly=True)
    try:
        return list(odb.steps['Loading'].frames[-1].fieldOutputs.keys())
    finally:
        odb.close()


def inspect():
    check()
    missing=[]
    for job in JOBS:
        req=expected_field(job)
        p=ODBS/(job+'.odb')
        names=get_fields(p)
        have=req in names
        if not have and is_linear(job):
            newp=LINEAR/(corrected_job(job)+'.odb')
            newfields=get_fields(newp) if newp.is_file() else []
            have=req in newfields
        if not have:
            missing.append(job)
        print('%-27s expected=%-2s original=%-5s corrected=%-5s fields=%s' % (
            job,req,'YES' if req in names else 'NO',
            ('YES' if have and req not in names else '---'), ','.join(sorted(names))))
    print('MISSING REQUIRED STRAIN FIELDS:',len(missing))
    if missing:
        print('If only *_LIN cases are missing, run --stage prepare-linear, then the generated jobs.')
    return missing


def prepare_linear():
    check()
    if not LINEAR.is_dir():
        raise RuntimeError('In Windows File Explorer create this folder first: '+str(LINEAR))
    required=[]
    for job in JOBS:
        if not is_linear(job):
            continue
        if 'E' in get_fields(ODBS/(job+'.odb')):
            print('E ALREADY PRESENT in original', job)
            continue
        newjob=corrected_job(job)
        newinp=LINEAR/(newjob+'.inp')
        if newinp.exists() or (LINEAR/(newjob+'.odb')).exists():
            print('PROTECTED: existing new deck / ODB, inspect manually:',newjob)
            continue
        src=ODBS/(job+'.inp')
        if not src.is_file():
            raise RuntimeError('Missing original input: '+str(src))
        raw=src.read_text()
        if not re.search(r'^\*STEP\b[^\n]*NLGEOM=NO\b',raw,re.I|re.M):
            raise RuntimeError('Original not a geometrically linear step: '+job)
        # Modify only the exact field request, not the mechanics or boundary conditions.
        out,n=re.subn(r'^(\s*S\s*,\s*)LE(\s*,\s*COORD\s*)$',r'\g<1>E\g<2>',raw,flags=re.I|re.M)
        if n!=1:
            raise RuntimeError('Expected exactly one `S, LE, COORD` line in '+str(src))
        newinp.write_text('** B9A10G strain-output-only copy; original remains unchanged\n'+out)
        required.append(newjob)
        print('GENERATED',newinp)
    if required:
        print('\nRun only these new jobs from B9A10G_linear_ODB (one per line):')
        for j in required:
            print('abaqus job=%s input=%s.inp cpus=4 interactive'%(j,j))
    else:
        print('No new decks required; E is saved or corrective decks already exist.')


def get_values(job,source_path):
    from odbAccess import openOdb
    fld=expected_field(job)
    odb=openOdb(str(source_path.resolve()),readOnly=True)
    try:
        step=odb.steps['Loading']
        frame=step.frames[-1]
        if fld not in frame.fieldOutputs:
            raise RuntimeError('%s not saved in %s. Available: %s'%(
                fld,source_path.name,','.join(sorted(frame.fieldOutputs.keys()))))
        inst=next((v for name,v in odb.rootAssembly.instances.items() if name.lower()=='patch-1'),None)
        if inst is None or len(inst.elements)!=1:
            raise RuntimeError('Expected one element in PATCH-1: '+str(source_path))
        values=frame.fieldOutputs[fld].getSubset(region=inst).values
        mapping={};is_ortho=('_ISO_' not in job)
        for val in values:
            i=int(val.integrationPoint)
            if i in mapping:
                raise RuntimeError('Duplicate %s integration point in %s'%(fld,job))
            data=np.asarray(val.data,dtype=float).reshape(-1)
            if data.shape!=(6,) or not np.isfinite(data).all():
                raise RuntimeError('Invalid six-component %s in %s'%(fld,job))
            try:
                Q=np.asarray(val.localCoordSystem,float).reshape(3,3)
                has=True
            except Exception as exc:
                if is_ortho:
                    raise RuntimeError('Missing %s local material axes for %s: %s'%(fld,job,exc))
                Q=np.eye(3);has=False
            if not np.isfinite(Q).all() or abs(np.linalg.det(Q)-1)>1e-3 or np.max(np.abs(Q@Q.T-np.eye(3)))>1e-3:
                raise RuntimeError('Invalid strain local-axis rotation for %s'%(job))
            mapping[i]=(data,Q,has)
        nip=1 if '_R_' in job else 8
        if set(mapping)!=set(range(1,nip+1)):
            raise RuntimeError('Wrong number of %s integration points in %s'%(fld,job))
        return (np.array([mapping[i][0] for i in range(1,nip+1)]),
                np.array([mapping[i][1] for i in range(1,nip+1)]),
                np.array([mapping[i][2] for i in range(1,nip+1)],dtype=np.uint8))
    finally:
        odb.close()


def verify_corrected_job(original,corrected):
    """A correction run must agree with the same original U, RF, and S (unchanged physics)."""
    from odbAccess import openOdb
    def sample(path):
        odb=openOdb(str(path.resolve()),readOnly=True)
        try:
            frame=odb.steps['Loading'].frames[-1]
            inst=next((x for n,x in odb.rootAssembly.instances.items() if n.lower()=='patch-1'),None)
            ans={}
            for key in ('U','RF','S'):
                if key not in frame.fieldOutputs:raise RuntimeError('Missing '+key+' in '+path.name)
                records=[]
                for v in frame.fieldOutputs[key].getSubset(region=inst).values:
                    idx=int(v.nodeLabel) if key in ('U','RF') else int(v.integrationPoint)
                    records.append((idx,np.asarray(v.data,float)))
                ans[key]=np.array([x[1] for x in sorted(records,key=lambda t:t[0])])
            return ans
        finally:odb.close()
    a=sample(original);b=sample(corrected)
    for key in ('U','RF','S'):
        if a[key].shape!=b[key].shape:
            raise RuntimeError('Correction rerun has different '+key+' shape')
        rel=float(np.linalg.norm(a[key]-b[key])/max(np.linalg.norm(a[key]),1e-10))
        if rel>1e-4:
            raise RuntimeError('Correction rerun changed '+key+' values by relative %.5g; do not merge'%rel)
    return True


def export():
    original=check()
    a=DEST/'B9A10G_LE_fields.npz'  # legacy filename for existing Mac script
    b=DEST/'B9A10G_windows_summary.json'
    if a.exists() or b.exists():
        raise RuntimeError('Existing B9A10G results protected; do not overwrite')
    data={};rows=[]
    for job in JOBS:
        field=expected_field(job)
        p=ODBS/(job+'.odb')
        available=get_fields(p)
        used=p;rerun=False
        if field not in available:
            if not is_linear(job):
                raise RuntimeError('Required LE missing from original nonlinear ODB '+job)
            used=LINEAR/(corrected_job(job)+'.odb')
            if not used.is_file():
                raise RuntimeError('E not saved for '+job+'; create B9A10G_linear_ODB in File Explorer, '
                                   'run --stage prepare-linear, then the new four E-output jobs.')
            if field not in get_fields(used):
                raise RuntimeError('E still missing from new job '+used.name)
            verify_corrected_job(p,used)
            rerun=True
        strains,q,flag=get_values(job,used)
        # Retain original npz key schema for backward compatibility with the Mac audit.
        data[job+'_LE']=strains
        data[job+'_LE_Q']=q
        data[job+'_LE_Q_available']=flag
        data[job+'_STRAIN_KIND']=np.array(field)
        rows.append({'job':job,'strain_field':field,'source_odb':used.name,
                     'new_E_output_job':rerun,'n_ip':len(strains),
                     'strain_local_frame_count':int(flag.sum()),'source_odb_sha256':sha(used)})
        print('%s EXPORTED %-27s IP %d source %s'%(field,job,len(strains),used.name))
    np.savez_compressed(str(a),**data)
    b.write_text(json.dumps({'step':'65AM-B9A10-G','revision':'E_for_NLGEOM_NO_LE_for_NLGEOM_YES',
          'cases':rows,'case_count':21,'LE_sha256':sha(a),
          'source_B9A10F_npz_sha256':original['exports_sha256'],
          'old_models_untouched':True,'original_B9A10F_ODB_files_modified':False},indent=2))
    print('B9A10G EXPORT PASS: '+str(a))


def selftest():
    assert len(JOBS)==21 and len(set(JOBS))==21
    assert sum(is_linear(j) for j in JOBS)==4
    assert all(expected_field(j)==('E' if j.endswith('_LIN') else 'LE') for j in JOBS)
    original='*STEP, NAME=Loading, NLGEOM=NO\n*ELEMENT OUTPUT\nS, LE, COORD\n'
    patched,n=re.subn(r'^(\s*S\s*,\s*)LE(\s*,\s*COORD\s*)$',r'\g<1>E\g<2>',original,flags=re.I|re.M)
    assert n==1 and 'S, E, COORD' in patched and 'NLGEOM=NO' in patched
    print('B9A10G FIXED SELFTEST PASS | E linear, LE nonlinear, safe input cloning')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','inspect','prepare-linear','export'))
    mode=p.parse_args().stage
    if mode=='selftest':selftest()
    elif mode=='check':check()
    elif mode=='inspect':inspect()
    elif mode=='prepare-linear':prepare_linear()
    else:export()
