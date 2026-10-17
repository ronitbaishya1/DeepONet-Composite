# -*- coding: utf-8 -*-
"""B9A10-O: read-only source, NEW Abaqus/Standard C3D8 path experiments.
Abaqus Python, runs from C:\\Users\\rrb225_lehigh\\Pictures\\ODBS.
  abaqus python scripts\\65AM_B9A10O_W_path_probe.py --stage selftest
  abaqus python scripts\\65AM_B9A10O_W_path_probe.py --stage check
  abaqus python scripts\\65AM_B9A10O_W_path_probe.py --stage generate
  [run 10 Abaqus jobs from B9A10O_ODB; see README]
  abaqus python scripts\\65AM_B9A10O_W_path_probe.py --stage inspect
  abaqus python scripts\\65AM_B9A10O_W_path_probe.py --stage export
No original B5/65V/B9 files are changed or opened for writing.
"""
from __future__ import print_function
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT/'B9A10O_ODB'
EXPORT = ROOT/'B9A10O_exports'
SIGN = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
DIMS = np.array([.6,.5,.62],float)
X = .5*(SIGN+1)*DIMS[None,:]
I = np.eye(3)
S = np.diag([1.012,.996,1.004])
H = I.copy();H[0,1]=.016;H[1,2]=-.004
THETA = np.deg2rad(17)
R = np.array([[math.cos(THETA),-math.sin(THETA),0],[math.sin(THETA),math.cos(THETA),0],[0,0,1]],float)
CASES = [
    ('O00_ST','O00','ST'),('O00_SH','O00','SH'),
    ('O00_SR','O00','SR'),('O00_RS','O00','RS'),
    ('O00_HR','O00','HR'),('O00_RH','O00','RH'),
    ('O17_SR','O17','SR'),('O17_RS','O17','RS'),
    ('ISO_SR','ISO','SR'),('ISO_RS','ISO','RS'),
]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def targets(code):
    return {'ST':[('Stretch',S)], 'SH':[('Shear',H)],
       'SR':[('Stretch',S),('Rotate',R@S)],
       'RS':[('Rotate',R),('Stretch',R@S)],
       'HR':[('Shear',H),('Rotate',R@H)],
       'RH':[('Rotate',R),('Shear',R@H)]}[code]

def job_text(mat,code):
    lines=['*HEADING','B9A10O finite-rotation constitutive path control '+mat+' '+code,
           '*PART, NAME=PATCH','*NODE']
    for a,x in enumerate(X,1):
        lines.append('%d, %.12g, %.12g, %.12g'%((a,)+tuple(x)))
    lines += ['*ELEMENT, TYPE=C3D8, ELSET=EALL',
              '1, 1, 2, 3, 4, 5, 6, 7, 8']
    if mat!='ISO':
        angle=17. if mat=='O17' else 0.
        ca,sa=math.cos(math.radians(angle)),math.sin(math.radians(angle))
        lines += ['*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR',
                  '%.14g, %.14g, 0, %.14g, %.14g, 0'%(ca,sa,-sa,ca),
                  '*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI',',']
    else:
        lines += ['*SOLID SECTION, ELSET=EALL, MATERIAL=MAT',',']
    lines += ['*END PART','*ASSEMBLY, NAME=ASSEM','*INSTANCE, NAME=PATCH-1, PART=PATCH',
              '*END INSTANCE','*END ASSEMBLY','*MATERIAL, NAME=MAT']
    if mat=='ISO':
        lines += ['*ELASTIC','20000, 0.25']
    else:
        lines += ['*ELASTIC, TYPE=ENGINEERING CONSTANTS',
                  '46500, 12500, 12000, 0.28, 0.28, 0.40, 4750, 4500','3500']
    for sn,F in targets(code):
        lines += ['*STEP, NAME='+sn+', NLGEOM=YES', '*STATIC',
                  '0.02, 1.0, 1.0E-10, 0.05', '*BOUNDARY']
        u=X @ (F-I).T  # total displacement relative to reference configuration
        for a,vec in enumerate(u,1):
            for j,dd in enumerate(vec,1):
                lines.append('PATCH-1.%d, %d, %d, %.14g'%(a,j,j,dd))
        lines += ['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
                  '*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES','S, LE',
                  '*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT','ALLSE, ALLIE, ALLWK',
                  '*END STEP']
    return '\n'.join(lines)+'\n'

def check():
    if not JOBS.is_dir():raise RuntimeError('Create in File Explorer first: '+str(JOBS))
    if not EXPORT.is_dir():raise RuntimeError('Create in File Explorer first: '+str(EXPORT))
    if len(CASES)!=10 or len(set(c[0] for c in CASES))!=10:raise RuntimeError('Bad 10-case matrix')
    print('B9A10O CHECK PASS | 10 new C3D8 affine path experiments | original B9 unaffected')

def selftest():
    assert np.linalg.det(H)>0 and np.linalg.det(S)>0
    np.testing.assert_allclose(R.T@R,I,atol=1e-14)
    np.testing.assert_allclose(targets('SR')[-1][1],targets('RS')[-1][1],atol=1e-14)
    np.testing.assert_allclose(targets('HR')[-1][1],targets('RH')[-1][1],atol=1e-14)
    for _,mat,code in CASES:
        t=job_text(mat,code)
        assert t.count('*END STEP')==len(targets(code))
        assert t.count('PATCH-1.')==24*len(targets(code))
        assert ('ORIENTATION=MAT_ORI' in t)==(mat!='ISO')
        assert '*ELEMENT, TYPE=C3D8' in t and 'NLGEOM=YES' in t
    print('B9A10O WINDOWS SELFTEST PASS | same-final-F path pairs | 10 jobs | step BCs')

def generate():
    check();p=JOBS/'B9A10O_input_manifest.json'
    expected=[JOBS/('B9A10O_'+key+'.inp') for key,_,_ in CASES]
    if p.exists() or any(x.exists() for x in expected):
        raise RuntimeError('Existing B9A10O input protected, do not regenerate')
    rows=[]
    for key,mat,code in CASES:
        job='B9A10O_'+key;dest=JOBS/(job+'.inp')
        dest.write_text(job_text(mat,code))
        rows.append({'job':job,'material':mat,'path':code,'steps':[s for s,_ in targets(code)],
                     'final_F':targets(code)[-1][1].tolist(),'sha256':sha(dest)})
        print('GENERATED',job)
    p.write_text(json.dumps({'step':'B9A10O','count':len(rows),'cases':rows},indent=2))
    print('B9A10O GENERATE PASS | 10 NEW inputs written')

def prepared():
    check();manifest=JOBS/'B9A10O_input_manifest.json'
    if not manifest.exists():raise RuntimeError('Missing manifest, run generate')
    m=json.loads(manifest.read_text());rows=m['cases']
    if len(rows)!=10:raise RuntimeError('Wrong manifest')
    for row in rows:
        p=JOBS/(row['job']+'.inp')
        if not p.is_file() or sha(p)!=row['sha256']:
            raise RuntimeError('Input missing/modified: '+str(p))
    return m

def orient(v,mat):
    if mat=='ISO':return np.eye(3),False
    try:q=np.asarray(v.localCoordSystem,float).reshape(3,3)
    except Exception as ex:raise RuntimeError('Missing single-precision orthotropic Q: '+str(ex))
    if not np.isfinite(q).all() or np.linalg.norm(q@q.T-I)>1e-3:
        raise RuntimeError('Invalid Abaqus orientation')
    return q,True

def extract_one(row,only_check=False):
    from odbAccess import openOdb
    j=row['job'];p=JOBS/(j+'.odb');sta=JOBS/(j+'.sta')
    if not p.is_file() or not sta.is_file():raise RuntimeError('Missing ODB/STA '+j)
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='replace').upper():
        raise RuntimeError('Analysis incomplete '+j)
    odb=openOdb(str(p.resolve()),readOnly=True)
    try:
        inst=next((inst for name,inst in odb.rootAssembly.instances.items() if name.upper()=='PATCH-1'),None)
        if inst is None or len(inst.elements)!=1 or len(inst.nodes)!=8:raise RuntimeError('Wrong mesh '+j)
        nodes=sorted(inst.nodes,key=lambda x:x.label)
        if [n.label for n in nodes]!=list(range(1,9)) or np.max(np.abs(np.array([n.coordinates for n in nodes])-X))>1e-6:
            raise RuntimeError('Unexpected reference mesh '+j)
        allsteps=targets(row['path']);sample=[];seg=[];tt=[]
        for si,(sname,expectedF) in enumerate(allsteps):
            if sname not in odb.steps:raise RuntimeError('Missing step '+sname+' in '+j)
            step=odb.steps[sname]
            if len(step.frames)<3:raise RuntimeError('Too few saved frames in '+j+'/'+sname)
            for fi,frame in enumerate(step.frames):
                for k in ('U','RF','S','LE'):
                    if k not in frame.fieldOutputs:raise RuntimeError('Missing '+k+' at '+j+'/'+sname)
                def nodal(k):
                    vals=frame.fieldOutputs[k].getSubset(region=inst).values
                    d={int(v.nodeLabel):np.asarray(v.data,float)[:3] for v in vals}
                    if set(d)!=set(range(1,9)):raise RuntimeError('Missing nodes '+j+'/'+k)
                    return np.array([d[i] for i in range(1,9)])
                U=nodal('U');RF=nodal('RF')
                def ips(k):
                    vals=frame.fieldOutputs[k].getSubset(region=inst).values
                    d={int(v.integrationPoint):v for v in vals}
                    if set(d)!=set(range(1,9)):raise RuntimeError('Missing IP '+k+' '+j)
                    return [d[i] for i in range(1,9)]
                stresses=ips('S');les=ips('LE')
                sv=np.array([np.asarray(v.data,float)[:6] for v in stresses])
                lv=np.array([np.asarray(v.data,float)[:6] for v in les])
                q=[];ok=[]
                for v in stresses:
                    qi,flag=orient(v,row['material']);q.append(qi);ok.append(flag)
                if row['material']!='ISO' and not all(ok):raise RuntimeError('Orthotropic orientation missing '+j)
                sample.append((U,RF,sv,lv,np.array(q)))
                seg.append(si);tt.append(float(frame.frameValue))
            # verify last frame corresponds to current step's total target displacement
            expected=X@(expectedF-I).T
            if np.max(abs(sample[-1][0]-expected))>2e-5:
                raise RuntimeError('Incorrect final displacement for '+j+'/'+sname)
        if only_check:
            return {'job':j,'frames':len(sample),'steps':len(allsteps),'odb_sha256':sha(p),
                    'input_sha256':row['sha256']}
        return {'job':j,'U':np.stack([v[0] for v in sample]),'RF':np.stack([v[1] for v in sample]),
                'S':np.stack([v[2] for v in sample]),'LE':np.stack([v[3] for v in sample]),
                'Q':np.stack([v[4] for v in sample]),'segment':np.array(seg,dtype=np.int32),
                't':np.array(tt),'X':X,'finalF':np.array(row['final_F'])},\
               {'job':j,'frames':len(sample),'steps':len(allsteps),'odb_sha256':sha(p),
                'input_sha256':row['sha256']}
    finally:
        odb.close()

def inspect():
    m=prepared()
    for row in m['cases']:
        record=extract_one(row,only_check=True)
        print('COMPLETE',row['job'],'frames',record['frames'],'steps',record['steps'])
    print('B9A10O INSPECT PASS | all 10 ODBs complete')

def export():
    m=prepared()
    output=EXPORT/'B9A10O_stress_paths.npz';report=EXPORT/'B9A10O_windows_summary.json'
    if output.exists() or report.exists():raise RuntimeError('B9A10O outputs already exist; protected')
    data={};records=[]
    for row in m['cases']:
        record,rep=extract_one(row); key=rep['job']
        for k,v in record.items():
            if k!='job':data[key+'__'+k]=v
        records.append(rep);print('EXPORTED',key,'frames',rep['frames'])
    np.savez_compressed(str(output),**data)
    report.write_text(json.dumps({'step':'65AM-B9A10-O','count':10,'cases':records,
        'input_manifest_sha256':sha(JOBS/'B9A10O_input_manifest.json'),
        'npz_sha256':sha(output),'all_nodes_prescribed':True,
        'no_existing_models_modified':True},indent=2))
    print('B9A10O EXPORT PASS',output)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,
       choices=('selftest','check','generate','inspect','export'))
    args=p.parse_args()
    {'selftest':selftest,'check':check,'generate':generate,
     'inspect':inspect,'export':export}[args.stage]()
if __name__=='__main__':main()
