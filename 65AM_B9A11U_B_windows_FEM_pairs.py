# -*- coding: utf-8 -*-
"""B9A11U-B: verified Abaqus fixed-q sensitivity pairs for NEW B5 FIT cases.

From ODBS Windows CMD:
 abaqus python scripts\65AM_B9A11U_B_windows_FEM_pairs.py --stage selftest
 abaqus python scripts\65AM_B9A11U_B_windows_FEM_pairs.py --stage check
 abaqus python scripts\65AM_B9A11U_B_windows_FEM_pairs.py --stage generate
 [execute 48 direct Abaqus commands, not a .cmd batch]
 abaqus python scripts\65AM_B9A11U_B_windows_FEM_pairs.py --stage export

Requires scripts\65AM_B9A3_W_reference_NO.py and B9_anchor files.
Output ODBS\B9A11U_ODB and ODBS\B9A11U_exports.
"""
from __future__ import print_function
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PLAN=ROOT/'B9A11U_plan'
JOBS=ROOT/'B9A11U_ODB'
EXP=ROOT/'B9A11U_exports'
REGIONS=('inner_left','inner_right')
PROP=('E1','E2','G12')
IDS=('E1_M','E1_P','E2_M','E2_P','G12_M','G12_P')
EPS=.03

def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def b9a3():
    p=HERE/'65AM_B9A3_W_reference_NO.py'
    if not p.is_file():raise FileNotFoundError('Copy existing B9A3 Windows script next to this file '+str(p))
    spec=importlib.util.spec_from_file_location('B9A11U_B9A3_verified',str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def jobname(r,idx,shift):return 'U_'+('L' if r=='inner_left' else 'R')+str(idx+1).zfill(2)+'_'+shift

def load():
    m=b9a3();anchor,ahash=m.anchor()
    npz=PLAN/'B9A11U_plan.npz';meta=PLAN/'B9A11U_plan_manifest.json'
    if not npz.is_file() or not meta.is_file():raise FileNotFoundError('Copy BOTH Mac plan files into '+str(PLAN))
    man=json.loads(meta.read_text())
    if sha(npz)!=man['plan_sha256'] or ahash!=man['anchor_sha256']:
        raise RuntimeError('STOP: plan SHA or 65V anchor mismatch')
    with np.load(str(npz),allow_pickle=False) as z:
        d={k:np.asarray(z[k]).copy() for k in z.files}
    if float(d['fraction'].item())!=EPS or tuple(d['shift_ids'].astype(str))!=IDS:
        raise RuntimeError('Wrong fixed sensitivity grid')
    if str(d['anchor_sha256'].item())!=ahash:raise RuntimeError('Embedded anchor mismatch')
    for r in REGIONS:
        if d['branch_'+r].shape!=(4,12) or d['fit_ids_'+r].shape!=(4,) or tuple(d['roles_'+r].astype(str))!=('train',)*3+('dev',):
            raise RuntimeError('Unexpected FIT plan '+r)
        if len(set(d['fit_ids_'+r].tolist()))!=4:raise RuntimeError('Duplicate FIT ids')
        grid=m.grid(anchor,r)
        na=len(anchor[grid['interfaces'][0]]['coeff']);nb=len(anchor[grid['interfaces'][1]]['coeff'])
        if 3+na+nb!=12:raise RuntimeError('Different Windows/Mac PCA branch dimension')
    print('B9A11U WINDOWS CHECK PASS: 48 new jobs; original B9A11T kept out of training')
    return m,anchor,ahash,d,man

def write_one(m,a,region,branch,center,shift):
    grid=m.grid(a,region)
    name=jobname(region,center,shift)
    dest=JOBS/(name+'.inp')
    if dest.exists() or (JOBS/(name+'.odb')).exists():raise RuntimeError('Existing job protected '+name)
    mat=np.asarray(branch[:3],float).copy()
    j=PROP.index(shift.split('_')[0]);mat[j]*=1. + (EPS if shift.endswith('P') else -EPS)
    if np.any(mat<=0):raise RuntimeError('Nonpositive material')
    left,right=grid['interfaces'];nleft=len(a[left]['coeff'])
    coeffs={left:np.asarray(branch[3:3+nleft],float),right:np.asarray(branch[3+nleft:],float)}
    bc={nm:(a[nm]['mean'].ravel()+a[nm]['basis'].dot(coeffs[nm])).reshape(112,3) for nm in (left,right)}
    # Same orthotropic coefficients and C3D8R subdomain geometry as verified B9A3 reference.
    E1,E2,G12=mat.tolist()
    with dest.open('w') as f:
        def w(s):f.write(s+'\n')
        w('*HEADING');w('** B9A11U FIT-only gradient '+name)
        w('*PART, NAME=BLOCK');w('*NODE')
        for lab,x,y,z in grid['nodes']:w('%d, %.10f, %.10f, %.10f'%(lab,x,y,z))
        w('*ELEMENT, TYPE=C3D8R, ELSET=EALL')
        for row in grid['elements']:w(', '.join(map(str,row)))
        for side,label in (('left','LEFT'),('right','RIGHT')):
            w('*NSET, NSET='+label+'_FACE')
            for row in m.chunks(grid['faces'][side]):w(', '.join(map(str,row)))
        w('*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR');w('1.0, 0.0, 0.0, 0.0, 1.0, 0.0')
        w('*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI');w(',')
        w('*END PART');w('*ASSEMBLY, NAME=ASSEMBLY')
        w('*INSTANCE, NAME=BLOCK-1, PART=BLOCK');w('*END INSTANCE');w('*END ASSEMBLY')
        w('*MATERIAL, NAME=MAT');w('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
        w('%.12g, %.12g, %.12g, .28, .28, .40, %.12g, 4500.'%(E1,E2,12000.,G12))
        w('3500.')
        w('*STEP, NAME=Loading, NLGEOM=YES');w('*STATIC');w('0.05, 1.0, 1.0E-08, 0.1')
        w('*BOUNDARY')
        for k,(il,ir) in enumerate(grid['map']):
            for side,iface,ip in (('left',left,il),('right',right,ir)):
                lab=grid['faces'][side][k]
                for component in range(3):
                    w('BLOCK-1.%d, %d, %d, %.12e'%(lab,component+1,component+1,bc[iface][ip,component]))
        w('*OUTPUT, FIELD, FREQUENCY=1');w('*NODE OUTPUT');w('U, RF')
        w('*ELEMENT OUTPUT, DIRECTIONS=YES');w('S, LE');w('*END STEP')
    return dict(job=name,region=region,center_index=center,shift=shift,
                input_sha256=sha(dest),material_MPa=mat.tolist())

def generate():
    m,a,ahash,d,man=load()
    JOBS.mkdir(parents=True,exist_ok=True)
    target=JOBS/'B9A11U_job_manifest.json'
    if target.exists():raise RuntimeError('Original job manifest protected, use check/export instead')
    all_names=[jobname(r,k,s) for r in REGIONS for k in range(4) for s in IDS]
    for nm in all_names:
        if (JOBS/(nm+'.inp')).exists() or (JOBS/(nm+'.odb')).exists():raise RuntimeError('Job name already in use: '+nm)
    entries=[]
    for r in REGIONS:
        for k,branch in enumerate(d['branch_'+r]):
            for s in IDS:
                entries.append(write_one(m,a,r,branch,k,s))
                print('GENERATED',entries[-1]['job'])
    target.write_text(json.dumps(dict(stage='B9A11U-generate',anchor_sha256=ahash,
         plan_sha256=man['plan_sha256'],fraction=EPS,entries=entries),indent=2))
    # Plain-text instructions only. Never execute a .cmd batch (group-policy server).
    with (JOBS/'B9A11U_RUN_COMMANDS.txt').open('w') as f:
        for e in entries:
            nm=e['job'];f.write('abaqus job=%s input=%s.inp cpus=4 interactive\n'%(nm,nm))
    print('48 NEW INPUTS READY. Open B9A11U_RUN_COMMANDS.txt, paste lines directly into Windows CMD from',str(JOBS))

def export():
    from odbAccess import openOdb
    m,a,ahash,d,man=load()
    mf=JOBS/'B9A11U_job_manifest.json'
    if not mf.is_file():raise FileNotFoundError('Generate Windows jobs first')
    jm=json.loads(mf.read_text())
    if jm['plan_sha256']!=man['plan_sha256'] or jm['anchor_sha256']!=ahash or len(jm['entries'])!=48:
        raise RuntimeError('Job manifest differs from plan')
    EXP.mkdir(parents=True,exist_ok=True)
    target=EXP/'B9A11U_training_pairs.npz';meta=EXP/'B9A11U_windows_summary.json'
    if target.exists() or meta.exists():raise RuntimeError('Existing export protected; do not overwrite')
    payload={'plan_sha256':np.asarray(man['plan_sha256']), 'anchor_sha256':np.asarray(ahash),
             'fraction':np.asarray(EPS),'shift_ids':np.asarray(IDS)}
    checks=[]
    for r in REGIONS:
        grid=m.grid(a,r);expected={int(lab):np.asarray([x,y,z]) for lab,x,y,z in grid['nodes']}
        branch=np.asarray(d['branch_'+r]);fids=np.asarray(d['fit_ids_'+r]);roles=d['roles_'+r]
        ul=[];labels0=None;xyz0=None
        for k in range(4):
            six=[];left,right=grid['interfaces'];nleft=len(a[left]['coeff'])
            cf={left:branch[k,3:3+nleft],right:branch[k,3+nleft:]}
            bc={nm:(a[nm]['mean'].ravel()+a[nm]['basis'].dot(cf[nm])).reshape(112,3) for nm in (left,right)}
            for s in IDS:
                nm=jobname(r,k,s)
                entry=next((x for x in jm['entries'] if x['job']==nm),None)
                if entry is None or sha(JOBS/(nm+'.inp'))!=entry['input_sha256']:
                    raise RuntimeError('Input checksum mismatch '+nm)
                odbfile=JOBS/(nm+'.odb');stafile=JOBS/(nm+'.sta')
                if not odbfile.is_file() or not stafile.is_file():raise FileNotFoundError('Job incomplete '+nm)
                if 'COMPLETED SUCCESSFULLY' not in stafile.read_text(errors='ignore').upper():
                    raise RuntimeError('Job not complete '+nm)
                odb=openOdb(path=str(odbfile.resolve()),readOnly=True)
                try:
                    frame=odb.steps['Loading'].frames[-1]
                    if abs(frame.frameValue-1.)>1e-6:raise RuntimeError('Wrong ODB final frame '+nm)
                    inst=m.get_instance(odb)
                    coords={int(v.label):np.asarray(v.coordinates,float) for v in inst.nodes}
                    U={int(v.nodeLabel):np.asarray(v.data,float).reshape(-1)[:3] for v in frame.fieldOutputs['U'].getSubset(region=inst).values}
                    if len(coords)!=1008 or len(U)!=1008:raise RuntimeError('Unexpected nodes '+nm)
                    labels=np.asarray(sorted(coords),dtype=int)
                    xyz=np.stack([coords[int(i)] for i in labels]);disp=np.stack([U[int(i)] for i in labels])
                    if any(np.max(abs(coords[i]-expected[i]))>2e-5 for i in coords):raise RuntimeError('Geometry changed '+nm)
                    qgap=0.
                    for t,(il,ir) in enumerate(grid['map']):
                        for side,iface,idx in (('left',left,il),('right',right,ir)):
                            lab=grid['faces'][side][t]
                            qgap=max(qgap,float(np.max(abs(U[lab]-bc[iface][idx]))))
                    if qgap>1e-5:raise RuntimeError('Prescribed q differs '+nm+' gap '+str(qgap))
                    if labels0 is None:labels0=labels;xyz0=xyz
                    if not np.array_equal(labels,labels0) or not np.allclose(xyz,xyz0,atol=1e-7,rtol=0):
                        raise RuntimeError('Unexpected node ordering '+nm)
                    six.append(disp)
                    checks.append(dict(job=nm,region=r,role=str(roles[k]),qgap_mm=qgap,odb_sha256=sha(odbfile)))
                    print('EXPORTED',nm,'max interface displacement difference',qgap,flush=True)
                finally:odb.close()
            ul.append(np.stack(six))
        payload['branch_'+r]=branch;payload['fit_ids_'+r]=fids;payload['roles_'+r]=roles
        payload['U_'+r]=np.stack(ul);payload['labels_'+r]=labels0;payload['xyz_'+r]=xyz0
    np.savez_compressed(str(target),**payload)
    meta.write_text(json.dumps(dict(stage='B9A11U-export',plan_sha256=man['plan_sha256'],
         source_anchor_sha256=ahash,npz_sha256=sha(target),job_count=len(checks),
         all_jobs_pass=True,uses_ONLY_new_FIT_centers=True,perturb_dev_cases_excluded_from_training=True,
         not_B9A11T_benchmark=True,checks=checks),indent=2))
    print('B9A11U EXPORT PASS:',str(target),'\nTransfer ENTIRE B9A11U_exports folder to Mac project root.')

def selftest():
    assert jobname('inner_left',0,'E1_M')=='U_L01_E1_M'
    assert jobname('inner_right',3,'G12_P')=='U_R04_G12_P'
    print('B9A11U WINDOWS SELFTEST PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('selftest','check','generate','export'))
    args=ap.parse_args()
    if args.stage=='selftest':return selftest()
    if args.stage=='check':load();return
    if args.stage=='generate':generate();return
    export()

if __name__=='__main__':main()