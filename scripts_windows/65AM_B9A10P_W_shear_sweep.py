# -*- coding: utf-8 -*-
"""B9A10-P -- Abaqus/Standard single-element finite-shear sweep.
New independent C3D8 jobs; NO edits to earlier B9/B5/65V files.
Run from the ODBS directory (abaqus python).
Stages: selftest, check, generate, inspect, export.
Create B9A10P_ODB and B9A10P_exports in File Explorer first.
"""
from __future__ import print_function
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
JOBS=ROOT/'B9A10P_ODB'; EXPORT=ROOT/'B9A10P_exports'
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
DIMS=np.array([.6,.5,.62],float)
X=.5*(SIGNS+1)*DIMS[None,:]; I=np.eye(3)
AMPLITUDES=[('Q25',.25),('Q50',.5),('Q100',1.),('Q200',2.)]
MATERIALS=['O00','O17','ISO']
CASES=[('B9A10P_%s_%s'%(m,code),m,code,alpha) for m in MATERIALS for code,alpha in AMPLITUDES]

def shear_F(alpha):
    f=np.eye(3); f[0,1]=.016*alpha; f[1,2]=-.004*alpha
    return f

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for blk in iter(lambda:f.read(1<<20),b''):h.update(blk)
    return h.hexdigest()

def make_deck(mat,alpha):
    lines=['*HEADING','B9A10P finite shear constitutive test '+mat+' factor %.6g'%alpha,
           '*PART, NAME=PATCH','*NODE']
    for a,x in enumerate(X,1): lines.append('%d, %.12g, %.12g, %.12g'%((a,)+tuple(x)))
    lines+=['*ELEMENT, TYPE=C3D8, ELSET=EALL','1, 1, 2, 3, 4, 5, 6, 7, 8']
    if mat=='ISO':lines+=['*SOLID SECTION, ELSET=EALL, MATERIAL=MAT',',']
    else:
        angle=17. if mat=='O17' else 0.
        c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
        lines+=['*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR',
            '%.14g, %.14g, 0, %.14g, %.14g, 0'%(c,s,-s,c),
            '*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI',',']
    lines+=['*END PART','*ASSEMBLY, NAME=ASSEM','*INSTANCE, NAME=PATCH-1, PART=PATCH',
            '*END INSTANCE','*END ASSEMBLY','*MATERIAL, NAME=MAT']
    if mat=='ISO':lines+=['*ELASTIC','20000, 0.25']
    else:lines+=['*ELASTIC, TYPE=ENGINEERING CONSTANTS',
                '46500, 12500, 12000, 0.28, 0.28, 0.40, 4750, 4500','3500']
    lines+=['*STEP, NAME=Shear, NLGEOM=YES','*STATIC','0.02, 1.0, 1.0E-10, 0.05',
            '*BOUNDARY']
    U=X@(shear_F(alpha)-I).T
    for node,vec in enumerate(U,1):
        for dof,v in enumerate(vec,1):
            lines.append('PATCH-1.%d, %d, %d, %.14g'%(node,dof,dof,v))
    lines+=['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
            '*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES','S, LE',
            '*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT','ALLSE, ALLIE, ALLWK',
            '*END STEP']
    return '\n'.join(lines)+'\n'

def check():
    for d in (JOBS,EXPORT):
        if not d.is_dir():raise RuntimeError('Create this folder in File Explorer first: '+str(d))
    if len(CASES)!=12:raise RuntimeError('Case matrix error')
    print('B9A10P CHECK PASS | 12 C3D8 new shear cases | no changes to prior simulations')

def selftest():
    assert len(CASES)==12 and len(set(v[0] for v in CASES))==12
    assert np.linalg.det(shear_F(2.))>0
    assert np.allclose(shear_F(1.)[0,1],.016) and np.allclose(shear_F(1.)[1,2],-.004)
    for name,mat,code,fac in CASES:
        deck=make_deck(mat,fac)
        assert deck.count('*END STEP')==1
        assert deck.count('PATCH-1.')==24
        assert ('ORIENTATION=MAT_ORI' in deck)==(mat!='ISO')
        assert '*ELEMENT, TYPE=C3D8' in deck and 'NLGEOM=YES' in deck
        assert 'S, LE' in deck
    print('B9A10P WINDOWS SELFTEST PASS | known O00_Q100 shear path | 12 protected input decks')

def generate():
    check();man=JOBS/'B9A10P_input_manifest.json'
    expected=[JOBS/(key+'.inp') for key,_,_,_ in CASES]
    if man.exists() or any(p.exists() for p in expected):
        raise RuntimeError('Inputs already exist; refusing overwrite. Do not regenerate.')
    rows=[]
    for name,mat,code,fac in CASES:
        p=JOBS/(name+'.inp');p.write_text(make_deck(mat,fac))
        rows.append(dict(job=name,material=mat,amplitude=float(fac),code=code,
                         finalF=shear_F(fac).tolist(),input_sha256=sha(p)))
        print('GENERATED',name)
    man.write_text(json.dumps(dict(step='B9A10P',cases=rows,count=12),indent=2))
    print('B9A10P GENERATE PASS | 12 input decks | original B9 preserved')

def prepared():
    check();man=JOBS/'B9A10P_input_manifest.json'
    if not man.is_file():raise RuntimeError('Missing input manifest -- run --stage generate')
    m=json.loads(man.read_text())
    if len(m['cases'])!=12:raise RuntimeError('Wrong number of cases')
    for row in m['cases']:
        p=JOBS/(row['job']+'.inp')
        if not p.is_file() or sha(p)!=row['input_sha256']:
            raise RuntimeError('Missing or modified input: '+str(p))
    return m

def stress_orientation(v,mat):
    if mat=='ISO':return np.eye(3)
    try:q=np.asarray(v.localCoordSystem,float).reshape(3,3)
    except Exception as e:raise RuntimeError('Orthotropic S localCoordSystem missing: '+str(e))
    if not np.isfinite(q).all() or np.linalg.norm(q@q.T-np.eye(3))>1e-3:
        raise RuntimeError('Invalid Abaqus local orientation')
    return q

def extract_one(row,check_only=False):
    from odbAccess import openOdb
    job=row['job'];p=JOBS/(job+'.odb');sta=JOBS/(job+'.sta')
    if not p.is_file() or not sta.is_file():raise RuntimeError('ODB or STA missing: '+job)
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='replace').upper():
        raise RuntimeError('Job not successfully completed: '+job)
    odb=openOdb(str(p.resolve()),readOnly=True)
    try:
        inst=next((v for name,v in odb.rootAssembly.instances.items() if name.upper()=='PATCH-1'),None)
        if inst is None or len(inst.nodes)!=8 or len(inst.elements)!=1:raise RuntimeError('Unexpected mesh: '+job)
        coords=np.array([n.coordinates for n in sorted(inst.nodes,key=lambda x:x.label)],float)
        if np.max(abs(coords-X))>1e-7:raise RuntimeError('Mesh changed in '+job)
        if 'Shear' not in odb.steps:raise RuntimeError('Missing Shear step: '+job)
        frames=odb.steps['Shear'].frames
        if len(frames)<3:raise RuntimeError('Too few frames in '+job)
        arr={x:[] for x in ('U','RF','S','LE','Q','t')}
        for frame in frames:
            for fld in ('U','RF','S','LE'):
                if fld not in frame.fieldOutputs:raise RuntimeError('Missing '+fld+' field in '+job)
            for fld in ('U','RF'):
                vals=frame.fieldOutputs[fld].getSubset(region=inst).values
                d={int(v.nodeLabel):np.asarray(v.data,float)[:3] for v in vals}
                if set(d)!=set(range(1,9)):raise RuntimeError('Missing nodal '+fld+' '+job)
                arr[fld].append(np.array([d[i] for i in range(1,9)]))
            for fld in ('S','LE'):
                vals=frame.fieldOutputs[fld].getSubset(region=inst).values
                d={int(v.integrationPoint):v for v in vals}
                if set(d)!=set(range(1,9)):raise RuntimeError('Missing 8 integration points '+fld+' '+job)
                arr[fld].append(np.array([np.asarray(d[i].data,float)[:6] for i in range(1,9)]))
                if fld=='S':arr['Q'].append(np.array([stress_orientation(d[i],row['material']) for i in range(1,9)]))
            arr['t'].append(float(frame.frameValue))
        if np.max(abs(arr['U'][-1]-X@(np.array(row['finalF'])-I).T))>2e-5:
            raise RuntimeError('Final imposed displacement mismatch: '+job)
        report=dict(job=job,material=row['material'],amplitude=row['amplitude'],
                    nframes=len(frames),input_sha256=row['input_sha256'],odb_sha256=sha(p))
        if check_only:return None,report
        return {k:np.array(v) for k,v in arr.items()},report
    finally:odb.close()

def inspect():
    m=prepared()
    for row in m['cases']:
        _,summary=extract_one(row,True)
        print('COMPLETE',summary['job'],'frames',summary['nframes'])
    print('B9A10P INSPECT PASS | 12 jobs complete')

def export():
    m=prepared();out=EXPORT/'B9A10P_shear_fields.npz';summary=EXPORT/'B9A10P_windows_summary.json'
    if out.exists() or summary.exists():raise RuntimeError('Export already exists, refusing overwrite: '+str(out))
    data={};rows=[]
    for row in m['cases']:
        arrays,record=extract_one(row)
        key=row['job']
        for n,v in arrays.items():data[key+'__'+n]=v
        data[key+'__X']=X.copy()
        rows.append(record);print('EXPORTED',key,'frames',record['nframes'])
    np.savez_compressed(str(out),**data)
    summary.write_text(json.dumps(dict(step='B9A10P',count=12,cases=rows,
                 npz_sha256=sha(out),manifest_sha256=sha(JOBS/'B9A10P_input_manifest.json'),
                 read_only_originals=True),indent=2))
    print('B9A10P EXPORT PASS |',out)

def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,
                choices=('selftest','check','generate','inspect','export'));args=a.parse_args()
    globals()[args.stage]()
if __name__=='__main__':main()
