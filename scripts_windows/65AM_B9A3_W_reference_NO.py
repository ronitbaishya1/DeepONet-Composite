# -*- coding: utf-8 -*-
"""B9A3 Windows (Abaqus Python): same-q 65V NO-side C3D8R reference.

Run from ODBS:
  abaqus python scripts\65AM_B9A3_W_reference_NO.py --stage check
  abaqus python scripts\65AM_B9A3_W_reference_NO.py --stage generate
  cd B9A3_ODB
  abaqus job=B9A3_inner_left input=B9A3_inner_left.inp cpus=4 interactive
  abaqus job=B9A3_inner_right input=B9A3_inner_right.inp cpus=4 interactive
  cd ..
  abaqus python scripts\65AM_B9A3_W_reference_NO.py --stage extract

Create ODBS/B9A3_ODB and ODBS/B9A3_exports manually before generating.
Will not change 65V, old force models, or B5 models.
"""
from __future__ import print_function
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ANCHOR=ROOT/'B9_anchor'/'B9_65V_anchor.npz'
ANCHOR_META=ROOT/'B9_anchor'/'B9_65V_anchor_manifest.json'
INP_DIR=ROOT/'B9A3_ODB'
EXPORT_DIR=ROOT/'B9A3_exports'
FACES={
    'inner_left': ('left_inner','center_left',-6.6,-1.8),
    'inner_right': ('center_right','right_inner',1.8,6.6),
}
STEPS=('check','generate','extract')
INTERFACE_ORDER=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
EXPECTED_COUNTS=(4,4,5,5,4,4)

def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def arr(z,k,shape=None):
    if k not in z:raise RuntimeError('Anchor missing '+k)
    a=np.asarray(z[k],dtype=np.float64)
    if shape is not None and a.shape!=shape:raise RuntimeError('%s shape %s, expected %s'%(k,str(a.shape),str(shape)))
    if not np.all(np.isfinite(a)):raise RuntimeError('Nonfinite anchor '+k)
    return a

def anchor():
    for p in [ANCHOR,ANCHOR_META]:
        if not p.is_file():raise FileNotFoundError('Missing prior B9 anchor: '+str(p))
    with open(str(ANCHOR_META)) as f: m=json.load(f)
    if sha(ANCHOR)!=m.get('anchor_sha256'):raise RuntimeError('B9 anchor SHA256 mismatch; do not continue')
    with np.load(str(ANCHOR),allow_pickle=False) as z:
        material=arr(z,'material',(3,))
        if not np.allclose(material,[46500,12500,4750],atol=1e-8,rtol=0):raise RuntimeError('Unexpected middle-material anchor')
        scale=arr(z,'force_scale',(26,))
        if (scale<=0).any():raise RuntimeError('Nonpositive original 65V force scale')
        old=arr(z,'old_NO_g',(26,))
        r=arr(z,'normalized_residual',(26,))
        if float(np.sqrt(np.mean(r**2)))>.020001:raise RuntimeError('Anchor not converged under original 65V')
        d=dict(material=material,force_scale=scale,old_NO_g=old,normalized_residual=r)
        for n,count in zip(INTERFACE_ORDER,EXPECTED_COUNTS):
            c=arr(z,'c_'+n,(count,))
            xyz=arr(z,'pca_coordinates_'+n)
            basis=arr(z,'pca_basis_'+n,(xyz.shape[0]*3,count))
            mean=arr(z,'pca_mean_'+n)
            if xyz.shape!=(112,3) or mean.size!=336:raise RuntimeError('Unexpected interface mesh for '+n)
            d[n]={'xyz':xyz,'basis':basis,'mean':mean.reshape(-1),'coeff':c,
                  'u':(mean.reshape(-1)+basis.dot(c)).reshape(-1,3)}
    return d,sha(ANCHOR)

def grid(d,name):
    left,right,xmin,xmax=FACES[name]
    A=d[left];B=d[right]
    xyzl=A['xyz'];xyzr=B['xyz']
    if max(abs(xyzl[:,0]-xmin))>2e-5 or max(abs(xyzr[:,0]-xmax))>2e-5:
        raise RuntimeError('Unexpected X face positions: '+name)
    yz0=xyzl[:,1:]; yz1=xyzr[:,1:]
    if np.max(abs(yz0-yz1))>1e-6:raise RuntimeError('Interface Y-Z ordering differs: '+name)
    # Reproduce the grid definition of the original 64F training generator.
    ys=np.unique(np.round(yz0[:,0],10));zs=np.unique(np.round(yz0[:,1],10))
    if len(ys)*len(zs)!=112:raise RuntimeError('Non-rectangular 112-node face')
    mapping=[]
    for y in ys:
        for z in zs:
            v=np.array([float(y),float(z)])
            il=int(np.argmin(np.max(np.abs(yz0-v[None,:]),axis=1)))
            ir=int(np.argmin(np.max(np.abs(yz1-v[None,:]),axis=1)))
            if max(abs(yz0[il]-v))>2e-6 or max(abs(yz1[ir]-v))>2e-6:
                raise RuntimeError('Missing interface Y-Z grid node '+str(v))
            mapping.append((il,ir))
    if len(set(i for i,j in mapping))!=112 or len(set(j for i,j in mapping))!=112:
        raise RuntimeError('Nonbijective interface coordinate mapping')
    nxe=int(round((xmax-xmin)/.6));xs=np.linspace(xmin,xmax,nxe+1)
    if nxe!=8 or max(abs(np.diff(xs)-.6))>1e-8:raise RuntimeError('Expected 8 C3D8R axial elements')
    nodes=[];edges={'left':[],'right':[]}
    nid=lambda ix,iy,iz: 1+ix*len(ys)*len(zs)+iy*len(zs)+iz
    for ix,x in enumerate(xs):
        for iy,y in enumerate(ys):
            for iz,z in enumerate(zs):
                n=nid(ix,iy,iz); nodes.append((n,float(x),float(y),float(z)))
                if ix==0:edges['left'].append(n)
                if ix==nxe:edges['right'].append(n)
    elements=[];eid=0
    for ix in range(nxe):
        for iy in range(len(ys)-1):
            for iz in range(len(zs)-1):
                eid+=1
                elements.append((eid,nid(ix,iy,iz),nid(ix+1,iy,iz),nid(ix+1,iy+1,iz),nid(ix,iy+1,iz),
                    nid(ix,iy,iz+1),nid(ix+1,iy,iz+1),nid(ix+1,iy+1,iz+1),nid(ix,iy+1,iz+1)))
    return dict(nodes=nodes,elements=elements,faces=edges,map=mapping,interfaces=(left,right),xs=xs)

def chunks(values,size=16):
    return [values[i:i+size] for i in range(0,len(values),size)]

def write_input(name,d,anchor_hash):
    desc=grid(d,name);left,right=desc['interfaces'];mat=d['material'];E1,E2,G12=mat
    if not INP_DIR.is_dir():raise RuntimeError('In File Explorer, create ODBS\\B9A3_ODB first')
    target=INP_DIR/('B9A3_'+name+'.inp')
    if target.exists():raise RuntimeError('Refuse overwrite existing input: '+str(target))
    E3=12000.;nu12=.28;nu13=.28;nu23=.40;G13=4500.;G23=3500.
    with target.open('w') as f:
        def w(s):f.write(s+'\n')
        w('*HEADING');w('B9A3 same-q reference for '+name);w('** Frozen saved 65V state anchor sha256: '+anchor_hash)
        w('*PART, NAME=BLOCK');w('*NODE')
        for label,x,y,z in desc['nodes']:w('%d, %.10f, %.10f, %.10f'%(label,x,y,z))
        w('*ELEMENT, TYPE=C3D8R, ELSET=EALL')
        for e in desc['elements']:w(', '.join(map(str,e)))
        for key,suffix in (('left','LEFT'),('right','RIGHT')):
            w('*NSET, NSET='+suffix+'_FACE')
            for row in chunks(desc['faces'][key]):w(', '.join(map(str,row)))
        w('*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR');w('1.0, 0.0, 0.0, 0.0, 1.0, 0.0')
        w('*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI');w(',')
        w('*END PART');w('*ASSEMBLY, NAME=ASSEMBLY');w('*INSTANCE, NAME=BLOCK-1, PART=BLOCK');w('*END INSTANCE');w('*END ASSEMBLY')
        w('*MATERIAL, NAME=MAT');w('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
        w('%.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g'%(E1,E2,E3,nu12,nu13,nu23,G12,G13));w('%.12g'%G23)
        w('*STEP, NAME=Loading, NLGEOM=YES');w('*STATIC');w('0.05, 1.0, 1.0E-08, 0.1')
        w('*BOUNDARY')
        for i,(jl,jr) in enumerate(desc['map']):
            for face,j in (('left',jl),('right',jr)):
                label=desc['faces'][face][i];iface=left if face=='left' else right
                for component in range(3):
                    w('BLOCK-1.%d, %d, %d, %.12e'%(label,component+1,component+1,d[iface]['u'][j,component]))
        w('*OUTPUT, FIELD, FREQUENCY=1');w('*NODE OUTPUT');w('U, RF');w('*ELEMENT OUTPUT, DIRECTIONS=YES');w('S, LE');w('*END STEP')
    return dict(name=name,inp=str(target),sha256=sha(target),nodes=len(desc['nodes']),elements=len(desc['elements']),
                faces={left:112,right:112},material=[float(x) for x in mat])

def get_instance(odb):
    for k,v in odb.rootAssembly.instances.items():
        if k.lower()=='block-1':return v
    raise RuntimeError('Reference NO ODB missing BLOCK-1 instance')

def extract(name,d,anchor_hash):
    from odbAccess import openOdb
    desc=grid(d,name)
    path=INP_DIR/('B9A3_'+name+'.odb')
    if not path.is_file():raise FileNotFoundError('Run the Abaqus job first: '+str(path))
    sta=path.with_suffix('.sta')
    if not sta.is_file():raise RuntimeError('Missing Abaqus .sta log for '+str(path))
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
        raise RuntimeError('Abaqus job not confirmed COMPLETED SUCCESSFULLY in '+str(sta))
    odb=openOdb(path=str(path.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step')
        frame=odb.steps['Loading'].frames[-1]
        if abs(float(frame.frameValue)-1.)>1e-7:raise RuntimeError('Final loading frame is not at time 1.0')
        for key in ('U','RF','S','LE'):
            if key not in frame.fieldOutputs:raise RuntimeError('ODB missing field '+key)
        instance=get_instance(odb)
        node_map={int(n.label):np.asarray(n.coordinates,dtype=np.float64) for n in instance.nodes}
        if len(node_map)!=len(desc['nodes']):raise RuntimeError('NO-side node count changed')
        maxgeo=max(float(np.linalg.norm(node_map[n]-np.asarray([x,y,z]))) for n,x,y,z in desc['nodes'])
        if maxgeo>2e-5:raise RuntimeError('ODB nodes differ from expected FE geometry %.6g'%maxgeo)
        RF={int(v.nodeLabel):np.asarray(v.data,dtype=np.float64).reshape(-1)[:3] for v in frame.fieldOutputs['RF'].getSubset(region=instance).values}
        U={int(v.nodeLabel):np.asarray(v.data,dtype=np.float64).reshape(-1)[:3] for v in frame.fieldOutputs['U'].getSubset(region=instance).values}
        actual={};full={};disp_gaps={};nodal={}
        for side,iface in (('left',desc['interfaces'][0]),('right',desc['interfaces'][1])):
            rf=[];ud=[];err=[]
            for i,pair in enumerate(desc['map']):
                j=pair[0 if side=='left' else 1];lab=desc['faces'][side][i]
                if lab not in RF or lab not in U:raise RuntimeError('Missing constrained U/RF label %d at %s'%(lab,iface))
                rf.append(RF[lab]);ud.append(U[lab]);err.append(U[lab]-d[iface]['u'][j]);
            rf=np.asarray(rf);ud=np.asarray(ud);err=np.asarray(err)
            ordered_rf=np.zeros((112,3));ordered_u=np.zeros((112,3))
            for i,pair in enumerate(desc['map']):
                j=pair[0 if side=='left' else 1]
                ordered_rf[j]=rf[i];ordered_u[j]=ud[i]
            g=d[iface]['basis'].T.dot(ordered_rf.reshape(-1))
            actual[iface]=g;full[iface]=ordered_rf
            nodal[iface]=ordered_u
            disp_gaps[iface]=float(np.max(np.abs(err)))
            if disp_gaps[iface]>1e-5:raise RuntimeError('Prescribed interface displacement mismatch '+iface)
        return actual,full,nodal,dict(region=name,odb=str(path.resolve()),odb_sha256=sha(path),
                        time=float(frame.frameValue),max_nodal_coordinate_error_mm=maxgeo,
                        max_prescribed_U_gap_mm=max(disp_gaps.values()),
                        nodes=len(node_map),elements_expected=len(desc['elements']),
                        RF_resultants_N={k:np.sum(v,axis=0).tolist() for k,v in full.items()},
                        projected_g_norm_N={k:float(np.linalg.norm(g)) for k,g in actual.items()})
    finally:odb.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=STEPS)
    a=p.parse_args()
    d,anchor_hash=anchor()
    for n in FACES:grid(d,n)
    print('B9A3 CHECK PASS | original 65V state %.8f | 2 NO regions, 4 interfaces'%float(np.sqrt(np.mean(d['normalized_residual']**2))))
    if a.stage=='check':
        print('No files created.');return
    if a.stage=='generate':
        if not INP_DIR.is_dir():raise RuntimeError('In File Explorer create ODBS\\B9A3_ODB first')
        created=[write_input(n,d,anchor_hash) for n in FACES]
        print('Generated exactly two new .inp files:');print(json.dumps(created,indent=2));return
    if not EXPORT_DIR.is_dir():raise RuntimeError('In File Explorer create ODBS\\B9A3_exports first')
    out=EXPORT_DIR/'B9A3_NO_reference.npz';summary=EXPORT_DIR/'B9A3_windows_summary.json'
    if out.exists() or summary.exists():raise RuntimeError('Refusing overwrite existing B9A3 export')
    forces={};nodalRF={};nodalU={};audits=[]
    for n in FACES:
        result,fr,uu,audit=extract(n,d,anchor_hash)
        forces.update(result);nodalRF.update(fr);nodalU.update(uu);audits.append(audit)
    g=np.full(26,np.nan);mask=np.zeros(26,dtype=np.int8);i=0
    for n,k in zip(INTERFACE_ORDER,EXPECTED_COUNTS):
        if n in forces:
            if forces[n].shape!=(k,):raise RuntimeError('Wrong projected force count at '+n)
            g[i:i+k]=forces[n];mask[i:i+k]=1
        i+=k
    payload=dict(anchor_sha256=np.array(anchor_hash),reference_NO_g=g,reference_mask=mask,material=d['material'])
    for n in forces:
        payload['nodal_RF_'+n]=nodalRF[n];payload['nodal_U_'+n]=nodalU[n]
    np.savez_compressed(str(out),**payload)
    report=dict(stage='65AM-B9A3',reference='Abaqus reaction RF on independently solved NO subdomains at SAVED 65V displacement BC',
       anchor_sha256=anchor_hash,export_sha256=sha(out),material_MPa=d['material'].tolist(),
       no_65V_change=True,no_B5_training=True,not_new_coupled_equilibrium=True,
       geometry_and_displacement_checks=True,cases=audits)
    with summary.open('w') as f:json.dump(report,f,indent=2)
    print('B9A3 EXTRACT PASS: real NO-side FEM reference for 4 interfaces')
    print('EXPORTED:',out);print('EXPORTED:',summary)

if __name__=='__main__':main()
