# -*- coding: utf-8 -*-
"""B9A6 Windows/Abaqus 2025: C3D8R force audit using all-nodal displacement BC.

Four independent Abaqus jobs: B5 and true FEM full-field displacements for each
inner region. Their RF yields a high-fidelity reference for the same displacement
state (including hourglass stabilization, with same material/element options).
These are NOT free equilibrium solves and NOT deployable neural force predictors.

From ODBS:
 abaqus python scripts\65AM_B9A6_W_kinematic_RF.py --stage check
 abaqus python scripts\65AM_B9A6_W_kinematic_RF.py --stage generate
 Manually run four jobs inside B9A6_ODB (see README).
 abaqus python scripts\65AM_B9A6_W_kinematic_RF.py --stage extract

Create folders ODBS/B9A6_inputs, ODBS/B9A6_ODB and ODBS/B9A6_exports
manually using File Explorer before starting. Old simulations are unmodified.
"""
from __future__ import print_function
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ANCHOR=ROOT/'B9_anchor'/'B9_65V_anchor.npz'
ANCHOR_META=ROOT/'B9_anchor'/'B9_65V_anchor_manifest.json'
INPUT=ROOT/'B9A6_inputs'/'B9A6_kinematic_fields.npz'
INPUT_META=ROOT/'B9A6_inputs'/'B9A6_mac_manifest.json'
BASE=ROOT/'B9A3_ODB'
JOBS=ROOT/'B9A6_ODB'
OUT=ROOT/'B9A6_exports'
REGIONS=('inner_left','inner_right')
IFACE={'inner_left':('left_inner','center_left'),'inner_right':('center_right','right_inner')}
COUNT={'left_inner':4,'center_left':5,'center_right':5,'right_inner':4}


def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_load(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    with path.open() as f:return json.load(f)


def arrays(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    with np.load(str(path),allow_pickle=False) as z:
        return {key:np.asarray(z[key]).copy() for key in z.files}


def inputs():
    for folder in (ROOT/'B9_anchor',ROOT/'B9A6_inputs',BASE):
        if not folder.is_dir():raise RuntimeError('Required folder missing: '+str(folder))
    am=json_load(ANCHOR_META);mm=json_load(INPUT_META)
    ah=sha(ANCHOR);th=sha(INPUT)
    if am.get('anchor_sha256')!=ah or mm.get('anchor_sha256')!=ah or mm.get('transfer_sha256')!=th:
        raise RuntimeError('Mac transfer/checkpoint SHA256 mismatch')
    a=arrays(ANCHOR);raw=arrays(INPUT)
    if str(raw['anchor_sha256'].item())!=ah:raise RuntimeError('Transfer NPZ anchor mismatch')
    for reg in REGIONS:
        label=np.asarray(raw[reg+'_node_labels'],int)
        xyz=np.asarray(raw[reg+'_node_xyz'],float)
        if label.shape!=(1008,) or xyz.shape!=(1008,3) or len(set(label.tolist()))!=1008:
            raise RuntimeError('Invalid full-field geometry for '+reg)
        for tag in ('B5','FEM'):
            u=np.asarray(raw[reg+'_U_'+tag],float)
            if u.shape!=(1008,3) or not np.isfinite(u).all():raise RuntimeError('Invalid U_'+tag+' '+reg)
        for iface in IFACE[reg]:
            c=np.asarray(a['pca_coordinates_'+iface],float)
            b=np.asarray(a['pca_basis_'+iface],float)
            if c.shape!=(112,3) or b.shape!=(336,COUNT[iface]):raise RuntimeError('Interface PCA mismatch '+iface)
    for reg in REGIONS:
        p=BASE/('B9A3_'+reg+'.inp')
        if not p.is_file():raise FileNotFoundError('Old original Abaqus deck missing '+str(p))
    return a,raw,ah,th


def parse_geometry(lines):
    keyword=[(i,line.strip().upper()) for i,line in enumerate(lines) if line.lstrip().startswith('*')]
    starts=[i for i,s in keyword if s=='*NODE' or s.startswith('*NODE,')]
    if len(starts)!=1:raise RuntimeError('Expected exactly one *NODE section')
    ni=starts[0];following=[i for i,s in keyword if i>ni]
    if not following:raise RuntimeError('No keyword after *NODE')
    nodes={}
    for s in lines[ni+1:following[0]]:
        if not s.strip() or s.lstrip().startswith('**'):continue
        x=[v.strip() for v in s.split(',')]
        if len(x)<4:raise RuntimeError('Bad *NODE row: '+s)
        label=int(x[0]);nodes[label]=np.array([float(z) for z in x[1:4]],float)
    if len(nodes)!=1008:raise RuntimeError('Expected 1008 B9A3 nodes, got '+str(len(nodes)))
    # B9A3 original generated decks have a single *BOUNDARY within Loading step.
    bounds=[i for i,s in keyword if s.startswith('*BOUNDARY')]
    if len(bounds)!=1:raise RuntimeError('Expected one and only one B9A3 *BOUNDARY block')
    bi=bounds[0];stop=next((i for i,s in keyword if i>bi and not s.startswith('**')),None)
    if stop is None or not lines[stop].lstrip().upper().startswith('*OUTPUT'):
        raise RuntimeError('Unexpected B9A3 deck; boundary must precede *OUTPUT')
    for req in ('*ELEMENT, TYPE=C3D8R','NLGEOM=YES','*ELASTIC, TYPE=ENGINEERING CONSTANTS','*NODE OUTPUT'):
        if not any(req in line.upper() for line in lines):raise RuntimeError('Original deck missing '+req)
    return nodes,bi,stop


def generate():
    a,raw,ah,th=inputs()
    if not JOBS.is_dir():raise RuntimeError('In Windows File Explorer create '+str(JOBS))
    planned=[]
    for reg in REGIONS:
        base=(BASE/('B9A3_'+reg+'.inp')).read_text().splitlines()
        geo,beg,end=parse_geometry(base)
        labels=np.asarray(raw[reg+'_node_labels'],int)
        xyz=np.asarray(raw[reg+'_node_xyz'],float)
        for label,pos in zip(labels,xyz):
            if label not in geo or np.linalg.norm(geo[label]-pos)>2e-5:
                raise RuntimeError('B9A3 inp/transfer mesh mismatch '+reg+' node '+str(label))
        # the original mesh/element/material/solver controls are copied verbatim.
        for tag in ('FEM','B5'):
            target=JOBS/('B9A6_'+reg+'_'+tag+'.inp')
            if target.exists():raise RuntimeError('Refusing to overwrite '+str(target))
            u=np.asarray(raw[reg+'_U_'+tag],float)
            bc=['*BOUNDARY']
            for label,vec in sorted(zip(labels,u),key=lambda q:int(q[0])):
                for j in range(3):bc.append('BLOCK-1.%d, %d, %d, %.14e'%(label,j+1,j+1,vec[j]))
            if len(bc)!=3025:raise RuntimeError('Expected all 3024 nodal translational DOFs constrained')
            target.write_text('\n'.join(base[:beg]+bc+base[end:])+'\n')
            planned.append({'region':reg,'field':tag,'input':str(target),'sha256':sha(target),'prescribed_dofs':3024})
            print('B9A6 generated:',target)
    print('B9A6 FOUR INPUTS READY. Run the four Abaqus jobs from B9A6_ODB before extraction.')


def nodal_map(field,instance):
    return {int(v.nodeLabel):np.asarray(v.data,dtype=float).ravel()[:3]
        for v in field.getSubset(region=instance).values}


def extract():
    from odbAccess import openOdb
    a,raw,ah,th=inputs()
    if not OUT.is_dir():raise RuntimeError('Create ODBS/B9A6_exports using File Explorer first')
    result={};cases=[]
    for reg in REGIONS:
        labels=raw[reg+'_node_labels'].astype(int)
        xyz=raw[reg+'_node_xyz'].astype(float)
        for tag in ('FEM','B5'):
            name='B9A6_'+reg+'_'+tag
            odbpath=JOBS/(name+'.odb');stapath=JOBS/(name+'.sta')
            inp=JOBS/(name+'.inp')
            if not odbpath.is_file() or not stapath.is_file() or not inp.is_file():
                raise FileNotFoundError('Missing Abaqus job output/inputs for '+name)
            if 'COMPLETED SUCCESSFULLY' not in stapath.read_text(errors='ignore').upper():
                raise RuntimeError('Abaqus .sta lacks successful completion for '+name)
            old=raw[reg+'_U_'+tag].astype(float)
            odb=openOdb(path=str(odbpath.resolve()),readOnly=True)
            try:
                if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step '+name)
                frame=odb.steps['Loading'].frames[-1]
                if abs(float(frame.frameValue)-1)>1e-6:raise RuntimeError('Last frame not time=1 '+name)
                inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
                if inst is None:raise RuntimeError('No BLOCK-1 instance '+name)
                geom={int(n.label):np.asarray(n.coordinates,float) for n in inst.nodes}
                if len(geom)!=1008:raise RuntimeError('Instance node count changed '+name)
                err_geom=max(float(np.linalg.norm(geom[k]-pos)) for k,pos in zip(labels,xyz))
                if err_geom>2e-5:raise RuntimeError('ODB geometry differs from original '+name)
                if 'RF' not in frame.fieldOutputs or 'U' not in frame.fieldOutputs:
                    raise RuntimeError('ODB missing U/RF field '+name)
                U=nodal_map(frame.fieldOutputs['U'],inst)
                RF=nodal_map(frame.fieldOutputs['RF'],inst)
                if len(U)!=1008 or len(RF)!=1008:raise RuntimeError('Missing some nodal U or RF '+name)
                err_disp=max(float(np.max(abs(U[int(k)]-value))) for k,value in zip(labels,old))
                if err_disp>2e-5:raise RuntimeError('Prescribed displacement not honored '+name+' gap '+str(err_disp))
                rf=np.asarray([RF[int(k)] for k in labels],float)
                if not np.isfinite(rf).all():raise RuntimeError('Nonfinite nodal reaction '+name)
                result[reg+'_'+tag+'_RF_nodal']=rf
                force_list=[];iface_details={}
                for iface in IFACE[reg]:
                    face=np.asarray(a['pca_coordinates_'+iface],float)
                    basis=np.asarray(a['pca_basis_'+iface],float)
                    f=[];unused=set()
                    for node in face:
                        d=np.linalg.norm(xyz-node[None,:],axis=1)
                        i=int(np.argmin(d))
                        if d[i]>2e-5 or i in unused:raise RuntimeError('PCA face mismatch '+iface)
                        unused.add(i);f.append(rf[i])
                    f=np.asarray(f,float)
                    projected=basis.T@f.ravel()
                    if projected.shape!=(COUNT[iface],):raise RuntimeError('Mode projection size incorrect')
                    force_list.extend(projected.tolist())
                    iface_details[iface]={'nodes':len(f),'force_resultant_N':np.sum(f,axis=0).tolist(),'g_norm_N':float(np.linalg.norm(projected))}
                result[reg+'_'+tag+'_g']=np.asarray(force_list,float)
                cases.append({'region':reg,'field':tag,'odb':str(odbpath),'odb_sha256':sha(odbpath),
                              'time':float(frame.frameValue),'max_U_error_mm':err_disp,'max_coord_error_mm':err_geom,
                              'total_RF_norm_N':float(np.linalg.norm(rf)),'interfaces':iface_details})
                print('B9A6 EXTRACTED:',name,'maximum displacement difference %.4g mm'%err_disp)
            finally:odb.close()
    out=OUT/'B9A6_kinematic_RF.npz';meta=OUT/'B9A6_windows_summary.json'
    if out.exists() or meta.exists():raise RuntimeError('Existing B9A6 exports protected; do not overwrite')
    result['anchor_sha256']=np.asarray(ah)
    np.savez(str(out),**result)
    summary={'step':'65AM-B9A6','anchor_sha256':ah,'transfer_sha256':th,'output_sha256':sha(out),
             'cases':cases,'purpose':'fully constrained C3D8R kinematic reaction audit; not online force model',
             'note':'Exact Abaqus RF from fully imposed B5/FEM displacement; control must reproduce original Abaqus reference before interpreting B5.'}
    meta.write_text(json.dumps(summary,indent=2))
    print('B9A6 EXPORT PASS:',out,meta)


def check():
    a,raw,ah,th=inputs()
    for reg in REGIONS:
        p=BASE/('B9A3_'+reg+'.inp')
        lines=p.read_text().splitlines()
        geo,beg,end=parse_geometry(lines)
        for lab,xyz in zip(raw[reg+'_node_labels'],raw[reg+'_node_xyz']):
            if int(lab) not in geo or np.linalg.norm(geo[int(lab)]-xyz)>2e-5:
                raise RuntimeError('B9A3 original input geometry differs from transfer '+reg)
    print('B9A6 CHECK PASS: anchor and 1008-node C3D8R inputs match; no output created')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('check','generate','extract'))
    args=p.parse_args()
    if args.stage=='check':check()
    elif args.stage=='generate':generate()
    else:extract()

if __name__=='__main__':main()
