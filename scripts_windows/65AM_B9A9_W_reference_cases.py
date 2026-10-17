# -*- coding: utf-8 -*-
"""65AM-B9A9 Windows Abaqus: independent NO-side reaction force benchmarks.

Copy next to existing scripts/65AM_B9A3_W_reference_NO.py. From ODBS:
  abaqus python scripts\65AM_B9A9_W_reference_cases.py --stage check
  abaqus python scripts\65AM_B9A9_W_reference_cases.py --stage generate
  (run 24 independent Abaqus jobs in B9A9_ODB)
  abaqus python scripts\65AM_B9A9_W_reference_cases.py --stage extract

Create ODBS/B9A9_inputs, ODBS/B9A9_ODB, ODBS/B9A9_exports MANUALLY using Explorer.
Input plan from Mac: B9A9_inputs/B9A9_case_plan.npz + B9A9_plan_manifest.json
No original files, saved 65V, or B5 checkpoints are modified.
"""
from __future__ import print_function
import argparse, importlib.util, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
INPUT=ROOT/'B9A9_inputs'
JOB=ROOT/'B9A9_ODB'
OUTPUT=ROOT/'B9A9_exports'
N=12
REGIONS=('inner_left','inner_right')
INTERFACES={'inner_left':('left_inner','center_left'),
            'inner_right':('center_right','right_inner')}


def imported_b9a3():
    p=HERE/'65AM_B9A3_W_reference_NO.py'
    if not p.is_file():raise FileNotFoundError('Retain previous B9A3 Windows script next to B9A9: '+str(p))
    spec=importlib.util.spec_from_file_location('B9A3_trusted_grid',str(p))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def data_check():
    m3=imported_b9a3()
    a,ahash=m3.anchor()
    fp=INPUT/'B9A9_case_plan.npz'
    fm=INPUT/'B9A9_plan_manifest.json'
    if not fp.is_file() or not fm.is_file():
        raise RuntimeError('Copy both B9A9 plan files from Mac into ODBS\\B9A9_inputs')
    man=json.loads(fm.read_text())
    if man.get('plan_sha256')!=sha(fp) or man.get('anchor_sha256')!=ahash:
        raise RuntimeError('Plan SHA256 or anchor SHA256 mismatch. Stop (do not generate).')
    with np.load(str(fp),allow_pickle=False) as z:
        data={k:np.asarray(z[k]).copy() for k in z.files}
    if str(data['anchor_sha256'].item())!=ahash:
        raise RuntimeError('NPZ saved anchor checksum changed')
    ids=[str(x) for x in data['case_ids']]
    classes=[str(x) for x in data['class_names']]
    if ids!=['B9A9_%02d'%j for j in range(1,N+1)]:
        raise RuntimeError('Case ID/order was changed')
    if len(classes)!=N or set(classes)!={'material','interface','combined'}:
        raise RuntimeError('Wrong experiment classes')
    for r in REGIONS:
        x=np.asarray(data['x_'+r],dtype=float)
        if x.shape!=(N,12) or not np.all(np.isfinite(x)) or np.any(x[:,:3]<=0):
            raise RuntimeError('Invalid 12-feature inputs '+r)
        left,right=INTERFACES[r]
        if 3+len(a[left]['coeff'])+len(a[right]['coeff'])!=12:
            raise RuntimeError('B9A9 branch features incompatible with 65V PCA')
        m3.grid(a,r)  # verifies geometry/mapping
    print('B9A9 WINDOWS CHECK PASS | 24 independent jobs | PCA and material plans valid')
    return m3,a,ahash,data,sha(fp)


def chunks(seq,n=16):return [seq[i:i+n] for i in range(0,len(seq),n)]


def job_name(region,i):return 'B9A9_'+('L' if region=='inner_left' else 'R')+'%02d'%(i+1)


def displacement(a,interface,coeff):
    z=a[interface]
    return (z['mean'].ravel()+np.asarray(z['basis'],float).dot(np.asarray(coeff,float))).reshape(-1,3)


def scenario(m3,anch,region,x):
    left,right=INTERFACES[region]
    nleft=len(anch[left]['coeff'])
    return dict(material=x[:3],u_left=displacement(anch,left,x[3:3+nleft]),
                u_right=displacement(anch,right,x[3+nleft:]),
                desc=m3.grid(anch,region))


def write_job(m3,anch,ahash,region,x,i):
    scen=scenario(m3,anch,region,x)
    desc=scen['desc']
    left,right=INTERFACES[region]
    path=JOB/(job_name(region,i)+'.inp')
    if path.exists():raise RuntimeError('Input already exists. Do not overwrite: '+str(path))
    E1,E2,G12=np.asarray(x[:3],dtype=float)
    # SAME constitutive constants as the previous successful B9A3 verification.
    E3=12000.;nu12=.28;nu13=.28;nu23=.40;G13=4500.;G23=3500.
    with path.open('w') as f:
        def w(s):f.write(s+'\n')
        w('*HEADING');w('B9A9 independent '+job_name(region,i)+' / previous 65V anchor '+ahash)
        w('*PART, NAME=BLOCK');w('*NODE')
        for lab,xx,yy,zz in desc['nodes']:w('%d, %.10f, %.10f, %.10f'%(lab,xx,yy,zz))
        w('*ELEMENT, TYPE=C3D8R, ELSET=EALL')
        for e in desc['elements']:w(', '.join(map(str,e)))
        for side,code in (('left','LEFT'),('right','RIGHT')):
            w('*NSET, NSET='+code+'_FACE')
            for ch in chunks(desc['faces'][side]):w(', '.join(map(str,ch)))
        w('*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR')
        w('1.0, 0.0, 0.0, 0.0, 1.0, 0.0')
        w('*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI');w(',')
        w('*END PART');w('*ASSEMBLY, NAME=ASSEMBLY')
        w('*INSTANCE, NAME=BLOCK-1, PART=BLOCK');w('*END INSTANCE');w('*END ASSEMBLY')
        w('*MATERIAL, NAME=MAT');w('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
        w('%.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g'%(E1,E2,E3,nu12,nu13,nu23,G12,G13))
        w('%.12g'%G23)
        w('*STEP, NAME=Loading, NLGEOM=YES');w('*STATIC');w('0.05, 1.0, 1.0E-08, 0.1')
        w('*BOUNDARY')
        for k,(il,ir) in enumerate(desc['map']):
            for side,j,uu in (('left',il,scen['u_left']),('right',ir,scen['u_right'])):
                lab=desc['faces'][side][k]
                for comp in range(3):
                    w('BLOCK-1.%d, %d, %d, %.12e'%(lab,comp+1,comp+1,uu[j,comp]))
        w('*OUTPUT, FIELD, FREQUENCY=1');w('*NODE OUTPUT');w('U, RF')
        w('*ELEMENT OUTPUT, DIRECTIONS=YES');w('S, LE');w('*END STEP')
    return str(path)


def generate():
    m3,a,h,d,plansha=data_check()
    if not JOB.is_dir():raise RuntimeError('Create ODBS\\B9A9_ODB using Windows File Explorer first')
    names=[job_name(r,i) for r in REGIONS for i in range(N)]
    for n in names:
        if (JOB/(n+'.inp')).exists() or (JOB/(n+'.odb')).exists():
            raise RuntimeError('Existing job protected: '+n)
    for r in REGIONS:
        for i,x in enumerate(d['x_'+r]):
            p=write_job(m3,a,h,r,x,i)
            print('GENERATED',p)
    print('B9A9 GENERATION DONE | Run 24 Abaqus jobs sequentially in B9A9_ODB')


def extract_one(m3,a,region,x,i):
    from odbAccess import openOdb
    desc=m3.grid(a,region)
    scen=scenario(m3,a,region,x)
    name=job_name(region,i)
    odbpath=JOB/(name+'.odb')
    status=JOB/(name+'.sta')
    if not odbpath.is_file() or not status.is_file():
        raise FileNotFoundError('ODB or STA missing for '+name)
    if 'COMPLETED SUCCESSFULLY' not in status.read_text(errors='ignore').upper():
        raise RuntimeError('Abaqus did not complete successfully: '+name)
    odb=openOdb(path=str(odbpath.resolve()),readOnly=True)
    try:
        frame=odb.steps['Loading'].frames[-1]
        if abs(float(frame.frameValue)-1.)>1e-7:
            raise RuntimeError('Final ODB frame not time 1.0: '+name)
        if 'RF' not in frame.fieldOutputs or 'U' not in frame.fieldOutputs:
            raise RuntimeError('Missing U/RF in ODB '+name)
        inst=m3.get_instance(odb)
        coords={int(node.label):np.asarray(node.coordinates,dtype=float) for node in inst.nodes}
        RF={int(v.nodeLabel):np.asarray(v.data,dtype=float).reshape(-1)[:3] for v in frame.fieldOutputs['RF'].getSubset(region=inst).values}
        U={int(v.nodeLabel):np.asarray(v.data,dtype=float).reshape(-1)[:3] for v in frame.fieldOutputs['U'].getSubset(region=inst).values}
        if len(coords)!=1008:raise RuntimeError('Unexpected C3D8R nodes: '+name)
        maximal_geo=0.;maximal_U=0.;results=[]
        for side,iface,ref_u in (('left',INTERFACES[region][0],scen['u_left']),
                                 ('right',INTERFACES[region][1],scen['u_right'])):
            rf=np.zeros((112,3),float)
            for j,pair in enumerate(desc['map']):
                v=pair[0 if side=='left' else 1]
                lab=desc['faces'][side][j]
                if lab not in RF or lab not in U or lab not in coords:
                    raise RuntimeError('Missing U/RF constrained node %d in %s'%(lab,name))
                rf[v]=RF[lab]
                maximal_geo=max(maximal_geo,float(np.linalg.norm(coords[lab]-a[iface]['xyz'][v])))
                maximal_U=max(maximal_U,float(np.linalg.norm(U[lab]-ref_u[v])))
            B=np.asarray(a[iface]['basis'],dtype=float)
            results.append(B.T.dot(rf.reshape(-1)))
        if maximal_geo>2e-5 or maximal_U>1e-5:
            raise RuntimeError('%s coordinate/U mismatch %.6g / %.6g'%(name,maximal_geo,maximal_U))
        return np.concatenate(results),dict(job=name,region=region,odb=str(odbpath),odb_sha256=sha(odbpath),
            max_coordinate_error_mm=maximal_geo,max_boundary_U_error_mm=maximal_U,
            frame=float(frame.frameValue),n_nodes=len(coords),n_elements=728,
            force_mode_count=len(results[0])+len(results[1]))
    finally:
        odb.close()


def extract():
    m3,a,ahash,d,plansha=data_check()
    if not OUTPUT.is_dir():raise RuntimeError('Create ODBS\\B9A9_exports using File Explorer first')
    outfile=OUTPUT/'B9A9_reference_forces.npz'
    summary=OUTPUT/'B9A9_windows_summary.json'
    if outfile.exists() or summary.exists():raise RuntimeError('B9A9 output exists; refusing to overwrite')
    payload=dict(anchor_sha256=np.asarray(ahash),plan_sha256=np.asarray(plansha),
        case_ids=d['case_ids'],class_names=d['class_names'])
    jobs=[]
    for region in REGIONS:
        preds=[]
        for i,x in enumerate(d['x_'+region]):
            g,r=extract_one(m3,a,region,x,i)
            preds.append(g)
            jobs.append(r)
            print('ODB PASS',r['job'],'max U gap mm %.3g'%r['max_boundary_U_error_mm'])
        payload['g_'+region]=np.asarray(preds,float)
        payload['x_'+region]=d['x_'+region]
    if len(jobs)!=24:raise RuntimeError('Incomplete case audit')
    np.savez_compressed(outfile,**payload)
    inf=dict(step='65AM-B9A9',anchor_sha256=ahash,plan_sha256=plansha,
        reference_sha256=sha(outfile),case_count=len(jobs),all_jobs_pass=True,
        jobs=jobs,no_models_retrained=True,not_a_coupled_solution=True)
    summary.write_text(json.dumps(inf,indent=2))
    print('B9A9 EXPORT COMPLETE:',outfile)
    print('Transfer the ENTIRE B9A9_exports folder to the Mac project root')


def selftest():
    M=imported_b9a3()
    v=np.array([2.,3.]);B=np.eye(2)
    assert np.allclose(B.T@v,v)
    assert job_name('inner_left',0)=='B9A9_L01'
    assert job_name('inner_right',11)=='B9A9_R12'
    print('WINDOWS B9A9 SELFTEST PASS | B9A3 geometry functions accessible')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('selftest','check','generate','extract'),required=True)
    a=ap.parse_args()
    if a.stage=='selftest':return selftest()
    if a.stage=='check':return data_check()
    if a.stage=='generate':return generate()
    if a.stage=='extract':return extract()

if __name__=='__main__':main()
