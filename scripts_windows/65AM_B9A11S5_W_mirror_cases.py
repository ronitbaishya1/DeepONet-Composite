# -*- coding: utf-8 -*-
"""B9A11-S5: NEW physically mirrored 7-region inner subdomain Abaqus tests.

Windows Abaqus Command, from C:\\Users\\rrb225_lehigh\\Pictures\\ODBS:
  abaqus python scripts\\65AM_B9A11S5_W_mirror_cases.py --stage selftest
  abaqus python scripts\\65AM_B9A11S5_W_mirror_cases.py --stage check
  abaqus python scripts\\65AM_B9A11S5_W_mirror_cases.py --stage generate
  cd B9A11S5_ODB
  for %M in (ISO OR0 OR17) do for %A in (H F) do for %S in (L R) do abaqus job=B9A11S5_%M_%A_%S input=B9A11S5_%M_%A_%S.inp cpus=4 interactive
  cd ..
  abaqus python scripts\\65AM_B9A11S5_W_mirror_cases.py --stage inspect
  abaqus python scripts\\65AM_B9A11S5_W_mirror_cases.py --stage export

Uses the same READ-ONLY 65V anchor and existing B9A3 mesh generator as B9A9.
Only creates NEW B9A11S5_ODB and B9A11S5_exports directories.
No old Abaqus ODB, B5 checkpoints, 65V, or Broyden outputs are modified.
"""
from __future__ import print_function
import argparse, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'
JOBS = ROOT / 'B9A11S5_ODB'
EXPORT = ROOT / 'B9A11S5_exports'
CASE_MATS = ('ISO', 'OR0', 'OR17')
CASE_AMPS = ('H', 'F')  # half, full
SIDES = ('L', 'R')
PAIRS = [(m, a) for m in CASE_MATS for a in CASE_AMPS]
STRESS_ORDER = ('11','22','33','12','13','23')
QREF = np.diag([-1.0,1.0,1.0])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def import_source():
    p = SCRIPTS / '65AM_B9A3_W_reference_NO.py'
    if not p.is_file():
        raise FileNotFoundError('Missing previously used B9A3 Windows mesh generator: '+str(p))
    spec = importlib.util.spec_from_file_location('s5_b9a3_mesh',str(p))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_mesh():
    m = import_source()
    anchor, source_sha = m.anchor()  # read-only validated 65V source
    grids = {side:m.grid(anchor, 'inner_left' if side=='L' else 'inner_right')
             for side in SIDES}
    # Match ALL nodes by exact reflected spatial coordinates, without relying on labels.
    lhs = np.array([n[1:] for n in grids['L']['nodes']],float)
    rhs = np.array([n[1:] for n in grids['R']['nodes']],float)
    refl = lhs @ QREF
    digits = lambda x:tuple(np.round(x,6).tolist())
    keys = {digits(r):i for i,r in enumerate(rhs)}
    if len(keys)!=len(rhs) or len(lhs)!=1008 or len(rhs)!=1008:
        raise RuntimeError('Unexpected or repeated mesh coordinates; B9A3 grid changed')
    mapping = []
    for p in refl:
        k = digits(p)
        if k not in keys:raise RuntimeError('Mesh not reflected across global x=0: '+str(k))
        mapping.append(keys[k])
    mapping = np.asarray(mapping,dtype=int)
    if len(set(mapping.tolist()))!=len(mapping):raise RuntimeError('Non-bijective mirror mapping')
    err = np.max(np.abs(refl-rhs[mapping]))
    if err>1e-7:raise RuntimeError('Mirror coordinate mismatch: '+str(err))
    # Element topology: unordered mirrored node sets, since reflection reverses handedness.
    left_labels = [int(n[0]) for n in grids['L']['nodes']]
    right_labels = [int(n[0]) for n in grids['R']['nodes']]
    LL = {lab:i for i,lab in enumerate(left_labels)}
    RR = {lab:i for i,lab in enumerate(right_labels)}
    mirror_elems={tuple(sorted(mapping[LL[int(l)]] for l in elem[1:])) for elem in grids['L']['elements']}
    target_elems={tuple(sorted(RR[int(l)] for l in elem[1:])) for elem in grids['R']['elements']}
    if len(mirror_elems)!=728 or mirror_elems!=target_elems:
        raise RuntimeError('Mirror element TOPOLOGY failed: must not run nonmatching meshes')
    for side in SIDES:
        g=grids[side]
        if len(g['elements'])!=728 or len(g['faces']['left'])!=112 or len(g['faces']['right'])!=112:
            raise RuntimeError('Unexpected B9A3 face/element counts on '+side)
        if set(g['faces']['left']) & set(g['faces']['right']):
            raise RuntimeError('Boundary faces overlap on '+side)
    return grids,source_sha,float(err)


def job(m,a,s):return 'B9A11S5_%s_%s_%s'%(m,a,s)


def prescribed(X,amp):
    """Globally defined even/odd displacement, EXACT symmetry U(R X)=R U(X).

    mm, global axes. These are controlled numerical symmetry experiments,
    NOT real three-point-bending loads or 65V PCA interface predictions.
    """
    x,y,z = [float(k) for k in X]
    xabs=abs(x)
    ytil=(y-2.)/2.
    ztil=(z-4.)/4.
    u1=amp*(0.0012*x*ztil+0.0006*x*ytil*ytil)
    u2=-amp*0.080*(1.0-0.20*xabs/8.)*(1.0+0.045*ytil-0.025*ztil*ztil)
    u3=amp*(0.0020*ztil*(1.0-0.15*xabs/8.)+0.0008*ytil*ytil)
    return np.array([u1,u2,u3],float)


def angle(m,side):
    return 17.0 if (m=='OR17' and side=='L') else (-17.0 if m=='OR17' else 0.0)


def input_text(g,m,a,side,source_sha):
    name=job(m,a,side)
    amp={'H':0.5,'F':1.0}[a]
    lines=['*HEADING',name+' controlled mirror test; source 65V SHA '+source_sha,
           '*PART, NAME=BLOCK','*NODE']
    for n in g['nodes']:
        lines.append('%d, %.12g, %.12g, %.12g'%tuple(n))
    lines.append('*ELEMENT, TYPE=C3D8R, ELSET=EALL')
    for e in g['elements']:lines.append(', '.join(str(int(k)) for k in e))
    for face,label in (('left','L_FACE'),('right','R_FACE')):
        lines.append('*NSET, NSET='+label)
        vals=list(g['faces'][face])
        for k in range(0,len(vals),16):
            lines.append(', '.join(str(int(t)) for t in vals[k:k+16]))
    if m!='ISO':
        rad=math.radians(angle(m,side))
        co,si=math.cos(rad),math.sin(rad)
        lines.extend(['*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR',
                      '%.14g, %.14g, 0., %.14g, %.14g, 0.'%(co,si,-si,co),
                      '*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI',','])
    else:
        lines.extend(['*SOLID SECTION, ELSET=EALL, MATERIAL=MAT',','])
    lines.extend(['*END PART','*ASSEMBLY, NAME=ASSEMBLY',
                  '*INSTANCE, NAME=BLOCK-1, PART=BLOCK','*END INSTANCE',
                  '*END ASSEMBLY','*MATERIAL, NAME=MAT'])
    if m=='ISO':
        lines.extend(['*ELASTIC','20000., 0.25'])
    else:
        lines.extend(['*ELASTIC, TYPE=ENGINEERING CONSTANTS',
                      '46500., 12500., 12000., 0.28, 0.28, 0.40, 4750., 4500.','3500.'])
    lines.extend(['*STEP, NAME=MirrorLoading, NLGEOM=YES','*STATIC',
                  '0.025, 1.0, 1.0E-09, 0.10', '*BOUNDARY'])
    node_dict={int(n[0]):np.asarray(n[1:],float) for n in g['nodes']}
    fixed=sorted(set(g['faces']['left'])|set(g['faces']['right']))
    if len(fixed)!=224:raise RuntimeError('Expected precisely 224 Dirichlet nodes')
    for label in fixed:
        u=prescribed(node_dict[int(label)],amp)
        for k in range(3):
            lines.append('BLOCK-1.%d, %d, %d, %.14e'%(label,k+1,k+1,u[k]))
    lines.extend(['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
                  '*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES','S, LE',
                  '*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT','ALLSE, ALLIE',
                  '*END STEP'])
    return '\n'.join(lines)+'\n'


def selftest():
    for m,a in PAIRS:
        amp={'H':0.5,'F':1.0}[a]
        for x in (-7.,-4.,-2.):
            xyz=np.array([x,1.75,5.25])
            np.testing.assert_allclose(prescribed(QREF@xyz,amp),QREF@prescribed(xyz,amp),atol=1e-14)
        if m=='OR17':assert angle(m,'L') == -angle(m,'R')
    assert len([job(m,a,s) for m,a in PAIRS for s in SIDES])==12
    # Test local to global and global to local rotation with a nontrivial tensor.
    th=math.radians(17.)
    rot=np.array([[math.cos(th),-math.sin(th),0],[math.sin(th),math.cos(th),0],[0,0,1.]])
    stress=np.diag([3.,2.,1.]);global_stress=rot@stress@rot.T
    np.testing.assert_allclose(np.linalg.eigvalsh(global_stress),np.linalg.eigvalsh(stress),atol=1e-12)
    print('B9A11-S5 SELFTEST PASS: 6 exact mirrored BC pairs; angle reflection; tensor transform')


def check():
    grids,h,err=get_mesh()
    print('B9A11-S5 CHECK PASS: 1008 nodes, 728 C3D8R, 224 Dirichlet nodes per job;',
          'mirrored geometry error %.3g mm'%err, 'anchor SHA',h)
    return grids,h


def generate():
    grids,h=check()
    JOBS.mkdir(parents=True,exist_ok=True)
    EXPORT.mkdir(parents=True,exist_ok=True)
    manifest=JOBS/'B9A11S5_manifest.json'
    destinations=[JOBS/(job(m,a,s)+'.inp') for m,a in PAIRS for s in SIDES]
    if manifest.exists() or any(p.exists() or p.with_suffix('.odb').exists() for p in destinations):
        raise RuntimeError('Existing S5 experiment protected; do not regenerate. Use a fresh folder.')
    records=[]
    for m,a in PAIRS:
        for s in SIDES:
            path=JOBS/(job(m,a,s)+'.inp')
            path.write_text(input_text(grids[s],m,a,s,h),encoding='utf-8')
            records.append(dict(job=path.stem,material=m,amplitude=a,side=s,
                    orientation_deg=angle(m,s),input_sha256=sha(path)))
            print('GENERATED',path.name)
    manifest.write_text(json.dumps({'step':'B9A11S5','anchor_sha256':h,
                                    'stress_frame':'Abaqus localCoordSystem global-to-local, converted as Q.T @ Sloc @ Q',
                                    'geometry_reflection_x0_mm':0.,'n_cases':12,'cases':records},indent=2))
    print('B9A11-S5 GENERATE PASS: 12 NEW inputs, run jobs before inspect')


def prepared():
    _,src=check()
    p=JOBS/'B9A11S5_manifest.json'
    if not p.is_file():raise RuntimeError('Run --stage generate first')
    m=json.loads(p.read_text())
    if len(m['cases'])!=12 or m['anchor_sha256']!=src:
        raise RuntimeError('Manifest mismatch')
    for rec in m['cases']:
        q=JOBS/(rec['job']+'.inp')
        if not q.is_file() or sha(q)!=rec['input_sha256']:
            raise RuntimeError('Input changed or missing: '+str(q))
    return m


def tensor(v):
    x=np.asarray(v,float)
    if x.shape[0]!=6:raise ValueError('Expected symmetric 3D stress Voigt [11,22,33,12,13,23]')
    return np.array([[x[0],x[3],x[4]],[x[3],x[1],x[5]],[x[4],x[5],x[2]]],float)


def voigt(t):return np.array([t[0,0],t[1,1],t[2,2],t[0,1],t[0,2],t[1,2]],float)


def read_field(value,mat,fallback_Q=None):
    """Abaqus localCoordSystem rows are GLOBAL-to-LOCAL for continuum tensor.
    Stress in global = Q.T @ stress_local @ Q. For no local system assume global.
    Rotated orthotropic requires local Q rather than silently assuming axes.
    """
    raw=np.asarray(value.data,float)
    try: Q=np.asarray(value.localCoordSystem,float)
    except Exception:Q=np.eye(3)
    if Q.shape!=(3,3):Q = (np.asarray(fallback_Q,float) if fallback_Q is not None else np.eye(3))
    if mat=='OR17' and np.allclose(Q,np.eye(3),atol=1e-10):
        raise RuntimeError('Missing expected rotated material localCoordSystem; global S unavailable')
    if np.max(abs(Q@Q.T-np.eye(3)))>1e-3:
        raise RuntimeError('Non-orthonormal Abaqus material directions')
    glob=Q.T@tensor(raw)@Q
    if abs(np.linalg.norm(glob)-np.linalg.norm(tensor(raw)))>1e-3*max(1,np.linalg.norm(tensor(raw))):
        raise RuntimeError('Stress invariants unexpectedly changed under rotation')
    return voigt(glob),Q


def extract_one(rec,summary_only=False):
    from odbAccess import openOdb
    p=JOBS/(rec['job']+'.odb')
    sta=JOBS/(rec['job']+'.sta')
    if not p.is_file() or not sta.is_file():
        raise FileNotFoundError('Missing ODB or STA for '+rec['job'])
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='replace').upper():
        raise RuntimeError('Job incomplete: '+rec['job']+'; inspect .dat/.msg')
    odb=openOdb(str(p.resolve()),readOnly=True)
    try:
        inst=None
        for name,v in odb.rootAssembly.instances.items():
            if name.upper()=='BLOCK-1':inst=v;break
        if inst is None or len(inst.nodes)!=1008 or len(inst.elements)!=728:
            raise RuntimeError('Unexpected ODB mesh '+rec['job'])
        if 'MirrorLoading' not in odb.steps:raise RuntimeError('Missing MirrorLoading step')
        frame=odb.steps['MirrorLoading'].frames[-1]
        if abs(float(frame.frameValue)-1.)>1e-7:raise RuntimeError('Not final time frame')
        for k in ('U','RF','S','LE'):
            if k not in frame.fieldOutputs:raise RuntimeError('Missing field '+k+' in '+rec['job'])
        nodes=sorted(inst.nodes,key=lambda x:x.label)
        labs=np.array([int(n.label) for n in nodes],np.int64)
        X=np.array([list(n.coordinates) for n in nodes],float)
        label2index={int(l):i for i,l in enumerate(labs)}
        def nodes_data(name):
            vs=frame.fieldOutputs[name].getSubset(region=inst).values
            d={int(v.nodeLabel):np.asarray(v.data,float)[:3] for v in vs}
            if any(k not in label2index for k in d):
                raise RuntimeError('Unexpected '+name+' node label '+rec['job'])
            if name=='U':
                if len(d)!=len(nodes) or any(int(i) not in d for i in labs):
                    raise RuntimeError('Missing U nodes '+rec['job'])
                return np.stack([d[int(i)] for i in labs]),len(d)
            # Abaqus can save RF on constrained nodes only; missing free-node
            # RF is not an interior equilibrium measurement and is set to zero.
            # Every prescribed boundary node MUST have a reaction value.
            bd=np.isclose(X[:,0],X[:,0].min(),atol=1e-7)|np.isclose(X[:,0],X[:,0].max(),atol=1e-7)
            if any(int(i) not in d for i in labs[bd]):
                raise RuntimeError('Missing constrained-node RF output '+rec['job'])
            return np.stack([d.get(int(i),np.zeros(3)) for i in labs]),len(d)
        U,_=nodes_data('U');RF,rf_recorded_count=nodes_data('RF')
        elems=sorted(inst.elements,key=lambda e:e.label)
        conn=np.array([[label2index[int(z)] for z in e.connectivity] for e in elems],int)
        elabs=[int(e.label) for e in elems]
        cen=X[conn].mean(axis=1)
        stress_vals=frame.fieldOutputs['S'].getSubset(region=inst).values
        le_vals=frame.fieldOutputs['LE'].getSubset(region=inst).values
        # C3D8R exactly one integration point per element.
        sdict={int(v.elementLabel):v for v in stress_vals}
        ldict={int(v.elementLabel):v for v in le_vals}
        if len(sdict)!=728 or len(ldict)!=728:
            raise RuntimeError('Expected one S/LE value per C3D8R element')
        S=[];LE=[];Q=[]
        for i in elabs:
            s,q=read_field(sdict[i],rec['material'])
            ll,_=read_field(ldict[i],rec['material'],fallback_Q=q)
            S.append(s);LE.append(ll);Q.append(q)
        S=np.asarray(S);LE=np.asarray(LE);Q=np.asarray(Q)
        amp={'H':0.5,'F':1.0}[rec['amplitude']]
        boundary=np.isclose(X[:,0],X[:,0].min(),atol=1e-7)|np.isclose(X[:,0],X[:,0].max(),atol=1e-7)
        if np.count_nonzero(boundary)!=224:raise RuntimeError('Fixed-node geometry count not 224')
        ub=np.array([prescribed(xx,amp) for xx in X[boundary]])
        bc_err=float(np.max(abs(U[boundary]-ub)))
        if bc_err>5e-5:raise RuntimeError('Abaqus prescribed displacement mismatch %.8g mm'%bc_err)
        data={'X':X,'node_labels':labs,'conn':conn,'centroids':cen,'U':U,'RF':RF,
              'S_global':S,'LE_global':LE,'Q_global_to_local':Q,
              'boundary_mask':boundary.astype(np.uint8)}
        info={'job':rec['job'],'material':rec['material'],'side':rec['side'],
              'amp':rec['amplitude'],'odb_sha256':sha(p),'num_nodes':len(nodes),
              'num_elems':len(elems),'max_boundary_error_mm':bc_err,
              'stress_frame_verified':True,'rf_recorded_node_count':rf_recorded_count}
        return (None if summary_only else data),info
    finally:odb.close()


def inspect():
    m=prepared()
    for rec in m['cases']:
        _,info=extract_one(rec,True)
        print('INSPECT PASS',rec['job'],'max BC error mm %.3g'%info['max_boundary_error_mm'])
    print('B9A11-S5 INSPECT PASS: all 12 ODBs, global stress conversion verified')


def export():
    m=prepared();EXPORT.mkdir(parents=True,exist_ok=True)
    target=EXPORT/'B9A11S5_mirrored_fem.npz'
    report=EXPORT/'B9A11S5_windows_summary.json'
    if target.exists() or report.exists():
        raise RuntimeError('Existing S5 exports protected: do not overwrite')
    payload={};status=[]
    for rec in m['cases']:
        d,info=extract_one(rec,False)
        for key,v in d.items():payload[rec['job']+'__'+key]=v
        status.append(info)
        print('EXPORTED',rec['job'])
    np.savez_compressed(str(target),**payload)
    report.write_text(json.dumps({'step':'B9A11S5','pair_count':6,'job_count':12,
                    'source_manifest_sha256':sha(JOBS/'B9A11S5_manifest.json'),
                    'export_sha256':sha(target),'symmetric_BC_constructed':True,
                    'global_stress_transformed':True,'jobs':status,
                    'b5_inference_performed':False,'original_models_modified':False},indent=2))
    print('B9A11-S5 EXPORT PASS: transfer whole B9A11S5_exports folder to Mac')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','check','generate','inspect','export'))
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':check()
    elif a.stage=='generate':generate()
    elif a.stage=='inspect':inspect()
    elif a.stage=='export':export()

if __name__=='__main__':main()
