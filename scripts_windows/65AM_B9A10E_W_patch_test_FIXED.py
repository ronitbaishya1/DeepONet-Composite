# -*- coding: utf-8 -*-
"""B9A10-E Abaqus (Windows): single-element force-assembly patch tests.

Run from ODBS:
  abaqus python scripts\65AM_B9A10E_W_patch_test.py --stage selftest
  abaqus python scripts\65AM_B9A10E_W_patch_test.py --stage check
  abaqus python scripts\65AM_B9A10E_W_patch_test.py --stage generate
  [run eight Abaqus jobs as described in README]
  abaqus python scripts\65AM_B9A10E_W_patch_test.py --stage extract

Two elements (C3D8R,C3D8), two materials (ISO,ORTHO), two displacement
patterns (AFF,NONAFF). All 8 nodes prescribed; no free-equilibrium claim.
Never changes B5, 65V, or older jobs. Outputs are protected.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = ROOT / 'B9A10E_ODB'
OUT_DIR = ROOT / 'B9A10E_exports'
ELEMENTS = ('R','F')
MATERIALS = ('ISO','ORTHO')
KINEMATICS = ('AFF','NONAFF')
KINDS = [(t,m,k) for t in ELEMENTS for m in MATERIALS for k in KINEMATICS]
C3 = {'R':'C3D8R','F':'C3D8'}
# Node ordering matches original C3D8R mesh in B9A3.
NAT = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
DIM = np.array([0.6,0.5,0.62],float)
XYZ = (NAT+1.)*DIM[None,:]*.5
# Fixed, known deformation; deliberately not tuned to the ODB results.
A = np.array([[.004,.002,.001],[-.001,-.002,.001],[.0005,-.0008,.003]],float)
SHIFT = np.array([.001,-.0005,.0002],float)


def job(t,m,k):return 'B9A10E_%s_%s_%s'%(t,m,k)

def sha(path):
    digest=hashlib.sha256()
    with open(str(path),'rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):digest.update(piece)
    return digest.hexdigest()

def field(k):
    u=XYZ.dot(A.T)+SHIFT[None,:]
    if k=='NONAFF':
        # Multiple trilinear cross terms; not representable as a linear field.
        h=NAT[:,0]*NAT[:,1]
        v=NAT[:,1]*NAT[:,2]
        u[:,1]+=0.002*h
        u[:,2]+=0.001*v
    return u

def geometry_check():
    if XYZ.shape!=(8,3):raise RuntimeError('Wrong node geometry')
    if not np.allclose(XYZ.min(axis=0),[0,0,0]):raise RuntimeError('Wrong geometry origin')
    if not np.allclose(XYZ.max(axis=0),DIM):raise RuntimeError('Wrong geometry extent')
    if np.linalg.det(A+np.eye(3))<=0:raise RuntimeError('Invalid affine F')
    if not JOB_DIR.is_dir():
        raise RuntimeError('In File Explorer create '+str(JOB_DIR))
    if not OUT_DIR.is_dir():
        raise RuntimeError('In File Explorer create '+str(OUT_DIR))
    print('B9A10E INPUT CHECK PASS | 8-node brick, two material systems, eight cases')


def inp_text(t,m,k):
    u=field(k)
    lines=['*HEADING','B9A10E independent one-element patch %s/%s/%s'%(t,m,k),
      '*PART, NAME=PATCH','*NODE']
    for i,p in enumerate(XYZ):
        lines.append('%d, %.12g, %.12g, %.12g'%((i+1,)+tuple(p)))
    lines.extend(['*ELEMENT, TYPE='+C3[t]+', ELSET=EALL','1, 1, 2, 3, 4, 5, 6, 7, 8'])
    if m=='ORTHO':
        theta=math.radians(17.)
        c,s=math.cos(theta),math.sin(theta)
        lines.extend(['*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR',
                      '%.12g, %.12g, 0.0, %.12g, %.12g, 0.0'%(c,s,-s,c),
                      '*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI',','])
    else:
        lines.extend(['*SOLID SECTION, ELSET=EALL, MATERIAL=MAT',','])
    lines.extend(['*END PART','*ASSEMBLY, NAME=ASSEMBLY',
      '*INSTANCE, NAME=PATCH-1, PART=PATCH','*END INSTANCE','*END ASSEMBLY',
      '*MATERIAL, NAME=MAT'])
    if m=='ORTHO':
        lines.extend(['*ELASTIC, TYPE=ENGINEERING CONSTANTS',
          '46500, 12500, 12000, 0.28, 0.28, 0.40, 4750, 4500','3500'])
    else:
        lines.extend(['*ELASTIC','20000, 0.25'])
    lines.extend(['*STEP, NAME=Loading, NLGEOM=YES','*STATIC','0.05, 1.0, 1.0E-10, 0.1',
                  '*BOUNDARY'])
    for i,p in enumerate(u):
        for c in range(3):
            lines.append('PATCH-1.%d, %d, %d, %.15g'%(i+1,c+1,c+1,p[c]))
    lines.extend(['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
       '*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES',
       'S, LE, COORD','*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT',
       'ALLAE, ALLIE, ALLSE, ALLWK','*END STEP'])
    return '\n'.join(lines)+'\n'


def generate():
    geometry_check()
    targets=[JOB_DIR/(job(*k)+'.inp') for k in KINDS]
    manifest=JOB_DIR/'B9A10E_inputs_manifest.json'
    if manifest.exists() or any(p.exists() for p in targets):
        raise RuntimeError('Existing B9A10E inputs protected; do not overwrite')
    rows=[]
    for t,m,k in KINDS:
        p=JOB_DIR/(job(t,m,k)+'.inp')
        p.write_text(inp_text(t,m,k))
        rows.append({'job':job(t,m,k),'element':C3[t], 'material':m,
                     'kinematics':k,'input_sha256':sha(p),'node_count':8,
                     'element_count':1,'prescribed_dofs':24})
        print('GENERATED',p.name)
    manifest.write_text(json.dumps({'step':'65AM-B9A10-E','jobs':rows},indent=2))
    print('B9A10E GENERATE PASS | eight new input decks')


def check_stress_orientation(value, material):
    """Read the single-precision Abaqus stress orientation safely.

    The generated ISO input has no *ORIENTATION and thus S is global:
    do not interrogate an absent localCoordSystem property. The generated
    ORTHO input has MAT_ORI, so every stress IP MUST supply a 3x3 local
    orientation. Abaqus's ODB field values are single precision: accessing
    localCoordSystemDouble raises OdbError (NOT AttributeError).
    """
    if material == 'ISO':
        return np.eye(3, dtype=float), False
    if material != 'ORTHO':
        raise RuntimeError('Unknown material flag for stress orientation: '+str(material))
    try:
        raw = value.localCoordSystem  # Single-precision ODB field property.
    except Exception as exc:
        raise RuntimeError('Missing MAT_ORI stress localCoordSystem: '+str(exc))
    if raw is None:
        raise RuntimeError('Missing MAT_ORI stress localCoordSystem (None)')
    q = np.asarray(raw, dtype=float)
    if q.size != 9:
        raise RuntimeError('MAT_ORI orientation must be a 3x3 matrix, got shape '+str(q.shape))
    q = q.reshape(3, 3)
    if (not np.isfinite(q).all() or
        np.max(np.abs(q.dot(q.T)-np.eye(3)))>1e-3 or
        abs(np.linalg.det(q)-1.)>1e-3):
        raise RuntimeError('Invalid ODB stress material orientation matrix')
    return q, True


def histories(step):
    ans={}
    for key in ('ALLAE','ALLIE','ALLSE','ALLWK'):
        entries=[]
        for name,region in step.historyRegions.items():
            if key in region.historyOutputs:
                data=region.historyOutputs[key].data
                if data:entries.append((str(name),float(data[-1][1])))
        ans[key]={'value':entries[0][1],'status':'available'} if len(entries)==1 else {
                 'status':'missing' if not entries else 'ambiguous','value':None}
    return ans


def read_odb(t,m,k):
    from odbAccess import openOdb
    j=job(t,m,k)
    path=JOB_DIR/(j+'.odb')
    sta=JOB_DIR/(j+'.sta')
    if not path.is_file() or not sta.is_file():
        raise RuntimeError('Missing completed .odb/.sta for '+j)
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
        raise RuntimeError('Abaqus job did not complete successfully: '+j)
    odb=openOdb(path=str(path.resolve()),readOnly=True)
    try:
        step=odb.steps['Loading'];fr=step.frames[-1]
        if abs(float(fr.frameValue)-1.)>1e-5:
            raise RuntimeError('Final time not 1.0 for '+j)
        inst=next((v for name,v in odb.rootAssembly.instances.items()
                   if name.lower()=='patch-1'),None)
        if inst is None or len(inst.nodes)!=8 or len(inst.elements)!=1:
            raise RuntimeError('Wrong single-element ODB mesh: '+j)
        for name in ('U','RF','S','COORD'):
            if name not in fr.fieldOutputs:raise RuntimeError('Missing output '+name+' '+j)
        nodes=sorted(inst.nodes,key=lambda n:int(n.label))
        labels=np.asarray([int(n.label) for n in nodes],np.int32)
        X=np.asarray([n.coordinates for n in nodes],float)
        if not np.array_equal(labels,np.arange(1,9)) or np.max(abs(X-XYZ))>1e-6:
            raise RuntimeError('Unexpected node ordering or coordinates: '+j)
        elem=inst.elements[0]
        conn=np.asarray(list(elem.connectivity),np.int32)
        if not np.array_equal(conn,np.arange(1,9)):
            raise RuntimeError('Unexpected element connectivity: '+j)
        def nodal(name):
            values=fr.fieldOutputs[name].getSubset(region=inst).values
            lookup={int(v.nodeLabel):np.asarray(v.data,dtype=float).reshape(-1)[:3] for v in values}
            if len(lookup)!=8 or set(lookup)!=set(labels.tolist()):
                raise RuntimeError('Missing nodal '+name+' '+j)
            return np.asarray([lookup[int(i)] for i in labels],float)
        U=nodal('U');RF=nodal('RF')
        if np.max(abs(U-field(k)))>2e-6:
            raise RuntimeError('Prescribed displacement mismatch: '+j)
        nips=1 if t=='R' else 8
        smap={};cmap={};qmap={}
        for v in fr.fieldOutputs['S'].getSubset(region=inst).values:
            ind=int(v.integrationPoint)
            if ind in smap:raise RuntimeError('Duplicate integration point stress: '+j)
            smap[ind]=np.asarray(v.data,float).reshape(-1)
            qmap[ind]=check_stress_orientation(v,m)
        for v in fr.fieldOutputs['COORD'].getSubset(region=inst).values:
            ind=int(v.integrationPoint)
            cmap[ind]=np.asarray(v.data,float).reshape(-1)
        if set(smap)!=set(range(1,nips+1)) or set(cmap)!=set(smap):
            raise RuntimeError('Incomplete stress or IP COORD: '+j)
        if any(len(smap[i])!=6 or len(cmap[i])!=3 for i in smap):
            raise RuntimeError('Incorrect tensor dimension: '+j)
        Q=np.asarray([qmap[i][0] for i in range(1,nips+1)],float)
        has=np.asarray([qmap[i][1] for i in range(1,nips+1)],bool)
        if m=='ORTHO' and not np.all(has):
            raise RuntimeError('Ortho material orientation missing in '+j)
        qstatus=int(np.sum(has))
        entry={'X':X,'U':U,'RF':RF,'element_connectivity':conn,
              'S':np.asarray([smap[i] for i in range(1,nips+1)],float),
              'IP_COORD':np.asarray([cmap[i] for i in range(1,nips+1)],float),
              'Q':Q,'Q_available':has.astype(np.uint8)}
        summ={'job':j,'odb_sha256':sha(path),'n_ip':nips,'material':m,'kinematics':k,
              'element':C3[t],'orientation_count':qstatus,'energy':histories(step),
              'max_U_error_mm':float(np.max(abs(U-field(k))))}
        return entry,summ
    finally:
        odb.close()


def extract():
    geometry_check()
    manifest=JOB_DIR/'B9A10E_inputs_manifest.json'
    if not manifest.is_file():raise RuntimeError('Missing generated input manifest')
    rows=json.loads(manifest.read_text())['jobs']
    if len(rows)!=8 or set(row['job'] for row in rows)!=set(job(*p) for p in KINDS):
        raise RuntimeError('Input manifest not for these 8 cases')
    for row in rows:
        inp=JOB_DIR/(row['job']+'.inp')
        if not inp.is_file() or sha(inp)!=row['input_sha256']:
            raise RuntimeError('Input changed: '+str(inp))
    npz=OUT_DIR/'B9A10E_patch_fields.npz'
    meta=OUT_DIR/'B9A10E_windows_summary.json'
    if npz.exists() or meta.exists():raise RuntimeError('Existing B9A10E exports protected')
    arrays={};audits=[]
    for t,m,k in KINDS:
        j=job(t,m,k)
        data,summary=read_odb(t,m,k)
        for key,val in data.items():arrays[j+'_'+key]=val
        audits.append(summary)
        print('EXTRACT',j,'IP=',summary['n_ip'],'localQ=',summary['orientation_count'])
    np.savez_compressed(str(npz),**arrays)
    meta.write_text(json.dumps({'step':'65AM-B9A10-E','cases':audits,
        'exports_sha256':sha(npz),'all_nodes_prescribed':True,
        'this_is_not_a_coupled_solution':True},indent=2))
    print('B9A10E EXPORT PASS:',npz)


def selftest():
    assert np.linalg.det(np.eye(3)+A)>0
    assert np.max(np.abs(field('AFF')-(XYZ.dot(A.T)+SHIFT)))<1e-14
    assert np.max(np.abs(field('NONAFF')-field('AFF')))>0.001
    assert len(KINDS)==8
    # Regression: ISO case must never access Abaqus's precision-specific
    # stress-orientation properties; ORTHO uses ONLY localCoordSystem.
    class IsoFieldValue(object):
        @property
        def localCoordSystem(self):
            raise AssertionError('ISO must not read localCoordSystem')
        @property
        def localCoordSystemDouble(self):
            raise AssertionError('ISO must not read localCoordSystemDouble')
    class OrthoSinglePrecisionValue(object):
        @property
        def localCoordSystem(self):
            return np.eye(3)
        @property
        def localCoordSystemDouble(self):
            raise RuntimeError('Data is single precision. Use localCoordSystem')
    q,flag=check_stress_orientation(IsoFieldValue(),'ISO')
    assert not flag and np.allclose(q,np.eye(3))
    q,flag=check_stress_orientation(OrthoSinglePrecisionValue(),'ORTHO')
    assert flag and np.allclose(q,np.eye(3))
    for t,m,k in KINDS:
        s=inp_text(t,m,k)
        assert '*ELEMENT, TYPE='+C3[t] in s
        assert s.count('PATCH-1.')==24
        assert s.count('*OUTPUT, HISTORY')==1
    print('B9A10E WINDOWS SELFTEST PASS | 8 decks, 24 displacement constraints each; single-precision orientation verified')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('selftest','check','generate','extract'),required=True)
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':geometry_check()
    elif a.stage=='generate':generate()
    else:extract()
