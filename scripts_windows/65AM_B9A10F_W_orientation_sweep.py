# -*- coding: utf-8 -*-
"""65AM-B9A10-F Windows/Abaqus | isolated orthotropic non-affine audit.

  abaqus python scripts\65AM_B9A10F_W_orientation_sweep.py --stage selftest
  abaqus python scripts\65AM_B9A10F_W_orientation_sweep.py --stage check
  abaqus python scripts\65AM_B9A10F_W_orientation_sweep.py --stage generate
  [run the 21 new input decks using Abaqus Standard]
  abaqus python scripts\65AM_B9A10F_W_orientation_sweep.py --stage extract

Never overwrites old B9A10E, B5, 65V, or current B9A10F outputs.
No folder modification: create B9A10F_ODB and B9A10F_exports manually.
"""
from __future__ import print_function
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / 'B9A10F_ODB'
OUT_DIR = ROOT / 'B9A10F_exports'
ANGLES = ('Z00', 'P17', 'M17', 'ISO')
AMPS = {'A00': 0.0, 'A10': 0.1, 'A50': 0.5, 'A100': 1.0}
# full integration C3D8, 4 materials x 4 warping strengths + 4 NLGEOM=NO + C3D8R control
CASES = [('F', ang, amp) for ang in ANGLES for amp in AMPS]
CASES += [('F', ang, 'LIN') for ang in ANGLES]
CASES += [('R', 'P17', 'A100')]
ETYPES = {'F': 'C3D8', 'R': 'C3D8R'}
NAT = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
DIM = np.array([0.6,0.5,0.62],float)
XYZ = (NAT+1.)*DIM[None,:]*.5
A = np.array([[.004,.002,.001],[-.001,-.002,.001],[.0005,-.0008,.003]],float)
SHIFT = np.array([.001,-.0005,.0002],float)


def case_name(t,ang,amp):
    return 'B9A10F_%s_%s_%s'%(t,ang,amp)


def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def config(t,ang,amp):
    if (t,ang,amp) not in CASES:
        raise ValueError('Unexpected test configuration: '+str((t,ang,amp)))
    alpha = 1.0 if amp=='LIN' else AMPS[amp]
    angle = {'Z00':0.0, 'P17':17.0, 'M17':-17.0, 'ISO':None}[ang]
    return {'t':t,'angle_code':ang,'amplitude_code':amp,'warping_amplitude':alpha,
            'angle_degrees':angle,'nlgeom':amp!='LIN','material':'ISO' if ang=='ISO' else 'ORTHO'}


def target_disp(alpha):
    u=XYZ.dot(A.T)+SHIFT[None,:]
    u[:,1] += alpha * .002 * NAT[:,0]*NAT[:,1]
    u[:,2] += alpha * .001 * NAT[:,1]*NAT[:,2]
    return u


def inp_text(t,ang,amp):
    cfg=config(t,ang,amp)
    u=target_disp(cfg['warping_amplitude'])
    j=case_name(t,ang,amp)
    lines=['*HEADING','B9A10F nonaffine orientation sweep: '+j,
           '*PART, NAME=PATCH','*NODE']
    for i,p in enumerate(XYZ):
        lines.append('%d, %.12g, %.12g, %.12g'%((i+1,)+tuple(p)))
    lines.extend(['*ELEMENT, TYPE='+ETYPES[t]+', ELSET=EALL',
                  '1, 1, 2, 3, 4, 5, 6, 7, 8'])
    if cfg['material']=='ORTHO':
        a=math.radians(cfg['angle_degrees']);c,s=math.cos(a),math.sin(a)
        lines.extend(['*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR',
                      '%.12g, %.12g, 0, %.12g, %.12g, 0'%(c,s,-s,c),
                      '*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI',','])
    else:
        lines.extend(['*SOLID SECTION, ELSET=EALL, MATERIAL=MAT',','])
    lines.extend(['*END PART','*ASSEMBLY, NAME=ASSEMBLY',
                  '*INSTANCE, NAME=PATCH-1, PART=PATCH','*END INSTANCE',
                  '*END ASSEMBLY','*MATERIAL, NAME=MAT'])
    if cfg['material']=='ORTHO':
        lines.extend(['*ELASTIC, TYPE=ENGINEERING CONSTANTS',
                      '46500, 12500, 12000, 0.28, 0.28, 0.40, 4750, 4500','3500'])
    else:
        lines.extend(['*ELASTIC','20000, 0.25'])
    lines.extend(['*STEP, NAME=Loading, NLGEOM='+('YES' if cfg['nlgeom'] else 'NO'),
                  '*STATIC','0.05, 1.0, 1.0E-10, 0.1','*BOUNDARY'])
    for i,p in enumerate(u):
        for c in range(3):
            lines.append('PATCH-1.%d, %d, %d, %.15g'%(i+1,c+1,c+1,p[c]))
    lines.extend(['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
                  '*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES',
                  'S, LE, COORD','*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT',
                  'ALLAE, ALLIE, ALLSE, ALLWK','*END STEP'])
    return '\n'.join(lines)+'\n'


def check():
    if not JOBS_DIR.is_dir():
        raise RuntimeError('In Windows File Explorer create folder: '+str(JOBS_DIR))
    if not OUT_DIR.is_dir():
        raise RuntimeError('In Windows File Explorer create folder: '+str(OUT_DIR))
    if len(CASES)!=21 or len(set(CASES))!=21:
        raise RuntimeError('Invalid 21-case matrix')
    print('B9A10F CHECK PASS | 21 isolated one-element cases, original B9A10E material and geometry')


def generate():
    check()
    manifest=JOBS_DIR/'B9A10F_input_manifest.json'
    targets=[JOBS_DIR/(case_name(*c)+'.inp') for c in CASES]
    if manifest.exists() or any(p.exists() for p in targets):
        raise RuntimeError('B9A10F inputs already exist; protected against overwrite')
    rows=[]
    for c in CASES:
        name=case_name(*c);p=JOBS_DIR/(name+'.inp')
        p.write_text(inp_text(*c))
        v=config(*c)
        rows.append(dict(job=name,element=ETYPES[c[0]],material=v['material'],
                         angle_code=c[1],angle_degrees=v['angle_degrees'],
                         amplitude_code=c[2],warping_amplitude=v['warping_amplitude'],
                         nlgeom=v['nlgeom'],input_sha256=sha(p),nodes=8,element_count=1,
                         prescribed_dofs=24))
        print('GENERATED',name)
    manifest.write_text(json.dumps({'step':'65AM-B9A10-F','cases':rows},indent=2))
    print('B9A10F GENERATED 21 decks, no old files modified')


def check_orient(value,material):
    if material=='ISO':
        return np.eye(3),False
    try:
        Q=np.asarray(value.localCoordSystem,float)
    except Exception as exc:
        raise RuntimeError('ORTHO stress orientation missing: '+str(exc))
    if Q.size!=9:
        raise RuntimeError('Expected 3x3 localCoordSystem; got '+str(Q.shape))
    Q=Q.reshape((3,3))
    if not np.isfinite(Q).all() or abs(np.linalg.det(Q)-1.)>1e-3 or np.max(abs(Q@Q.T-np.eye(3)))>1e-3:
        raise RuntimeError('Invalid stress-local orientation matrix')
    return Q,True


def read_history(step):
    h={}
    for key in ('ALLAE','ALLIE','ALLSE','ALLWK'):
        x=[]
        for region_name,region in step.historyRegions.items():
            if key in region.historyOutputs and region.historyOutputs[key].data:
                x.append((str(region_name),float(region.historyOutputs[key].data[-1][1])))
        h[key]=dict(status='available',value=x[0][1]) if len(x)==1 else dict(
            status='missing' if len(x)==0 else 'ambiguous',value=None)
    return h


def read_odb(c):
    from odbAccess import openOdb
    t,ang,amp=c
    name=case_name(*c);odbpath=JOBS_DIR/(name+'.odb');stapath=JOBS_DIR/(name+'.sta')
    if not odbpath.is_file() or not stapath.is_file():
        raise RuntimeError('Missing .odb or .sta for '+name)
    if 'COMPLETED SUCCESSFULLY' not in stapath.read_text(errors='ignore').upper():
        raise RuntimeError('Job did not complete successfully '+name)
    odb=openOdb(str(odbpath.resolve()),readOnly=True)
    try:
        step=odb.steps['Loading'];frame=step.frames[-1]
        if abs(float(frame.frameValue)-1.0)>1e-5:
            raise RuntimeError('Wrong final frame '+name)
        inst=next((v for n,v in odb.rootAssembly.instances.items() if n.lower()=='patch-1'),None)
        if inst is None or len(inst.nodes)!=8 or len(inst.elements)!=1:
            raise RuntimeError('Unexpected mesh '+name)
        for key in ('U','RF','S','COORD'):
            if key not in frame.fieldOutputs:
                raise RuntimeError('Output missing '+key+' '+name)
        ns=sorted(inst.nodes,key=lambda n:int(n.label))
        labels=np.array([int(n.label) for n in ns])
        X=np.asarray([n.coordinates for n in ns],float)
        if not np.array_equal(labels,np.arange(1,9)) or np.max(abs(X-XYZ))>1e-6:
            raise RuntimeError('Unexpected node labels/geometry '+name)
        conn=np.asarray(inst.elements[0].connectivity,dtype=int)
        if not np.array_equal(conn,np.arange(1,9)):
            raise RuntimeError('Unexpected element connectivity '+name)
        def nodefield(key):
            vals=frame.fieldOutputs[key].getSubset(region=inst).values
            mp={int(v.nodeLabel):np.asarray(v.data,float).reshape(-1)[:3] for v in vals}
            if set(mp)!=set(labels.tolist()):
                raise RuntimeError('Missing nodal '+key+' '+name)
            return np.array([mp[int(n)] for n in labels])
        U=nodefield('U'); RF=nodefield('RF')
        cfg=config(*c)
        if np.max(abs(U-target_disp(cfg['warping_amplitude'])))>2e-6:
            raise RuntimeError('Displacement mismatch '+name)
        smap={};qmap={};cmap={}
        for v in frame.fieldOutputs['S'].getSubset(region=inst).values:
            i=int(v.integrationPoint)
            if i in smap: raise RuntimeError('Duplicate integration point '+name)
            smap[i]=np.asarray(v.data,float).reshape(-1)
            qmap[i]=check_orient(v,cfg['material'])
        for v in frame.fieldOutputs['COORD'].getSubset(region=inst).values:
            i=int(v.integrationPoint)
            cmap[i]=np.asarray(v.data,float).reshape(-1)
        ipcount=1 if t=='R' else 8
        if set(smap)!=set(range(1,ipcount+1)) or set(cmap)!=set(smap):
            raise RuntimeError('Missing IP output '+name)
        Q=np.asarray([qmap[i][0] for i in range(1,ipcount+1)])
        flags=np.asarray([qmap[i][1] for i in range(1,ipcount+1)],np.uint8)
        if cfg['material']=='ORTHO' and not np.all(flags):
            raise RuntimeError('Missing assigned material frames '+name)
        s=np.asarray([smap[i] for i in range(1,ipcount+1)])
        xyz=np.asarray([cmap[i] for i in range(1,ipcount+1)])
        if s.shape!=(ipcount,6) or xyz.shape!=(ipcount,3):
            raise RuntimeError('Unexpected IP data shape '+name)
        data={'X':X,'U':U,'RF':RF,'S':s,'Q':Q,'Q_available':flags,'IP_COORD':xyz,'connectivity':conn}
        summary={'job':name,'odb_sha256':sha(odbpath),'n_ip':ipcount,
                 'rotation_count':int(flags.sum()),'energy':read_history(step),
                 'max_U_error_mm':float(np.max(abs(U-target_disp(cfg['warping_amplitude']))))}
        return data,summary
    finally:
        odb.close()


def extract():
    check()
    manifestpath=JOBS_DIR/'B9A10F_input_manifest.json'
    if not manifestpath.is_file(): raise RuntimeError('Missing input manifest; generate first')
    m=json.loads(manifestpath.read_text()); rows=m['cases']
    if len(rows)!=21 or set(r['job'] for r in rows)!=set(case_name(*c) for c in CASES):
        raise RuntimeError('Manifest does not match 21 cases')
    for r in rows:
        p=JOBS_DIR/(r['job']+'.inp')
        if not p.is_file() or sha(p)!=r['input_sha256']:
            raise RuntimeError('Input file modified or missing '+str(p))
    outfile=OUT_DIR/'B9A10F_patch_fields.npz'
    report=OUT_DIR/'B9A10F_windows_summary.json'
    if outfile.exists() or report.exists():
        raise RuntimeError('Existing B9A10F exports protected')
    data={};entries=[]
    for c in CASES:
        d,s=read_odb(c); j=case_name(*c)
        for k,v in d.items():data[j+'_'+k]=v
        entries.append(s)
        print('EXTRACTED',j,'IP',s['n_ip'])
    np.savez_compressed(str(outfile),**data)
    report.write_text(json.dumps({'step':'65AM-B9A10-F','case_count':21,
        'input_manifest_sha256':sha(manifestpath),'exports_sha256':sha(outfile),
        'cases':entries,'all_nodes_prescribed':True,
        'models_unchanged':True},indent=2))
    print('B9A10F EXTRACT SUCCESS',outfile)


def selftest():
    assert len(CASES)==21 and len(set(CASES))==21
    np.testing.assert_allclose(target_disp(1.)-target_disp(0.),
        np.column_stack((np.zeros(8),.002*NAT[:,0]*NAT[:,1],.001*NAT[:,1]*NAT[:,2])))
    for c in CASES:
        s=inp_text(*c)
        assert '*ELEMENT, TYPE='+ETYPES[c[0]] in s
        assert '*STEP, NAME=Loading, NLGEOM='+('YES' if config(*c)['nlgeom'] else 'NO') in s
        assert s.count('PATCH-1.')==24
        assert ('ORIENTATION=MAT_ORI' in s)==(c[1]!='ISO')
    class FakeSingle(object):
        @property
        def localCoordSystem(self):return np.eye(3)
        @property
        def localCoordSystemDouble(self):raise RuntimeError('single precision')
    assert check_orient(FakeSingle(),'ORTHO')[1]
    assert not check_orient(FakeSingle(),'ISO')[1]
    print('B9A10F WINDOWS SELFTEST PASS | 21 cases, amp/angle/linear modes, single precision S orientation')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','generate','extract'))
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':check()
    elif a.stage=='generate':generate()
    else:extract()
