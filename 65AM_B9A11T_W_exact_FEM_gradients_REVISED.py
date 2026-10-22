# -*- coding: utf-8 -*-
"""B9A11T-REVISED Windows: matched fixed-q FEM material gradients and QC.

Place in ODBS/scripts next to 65AM_B9A3_W_reference_NO.py.
Stages selftest, audit, generate, export, qc. Use directly via `abaqus python ...` in ODBS.
Existing B9A3_inner_left/right.inp and successful ODB/STA required.
Generates 12 perturbed jobs (not the two already solved baselines).
No existing 65V, B9A3, or other jobs are overwritten.
"""
from __future__ import print_function
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
JOBS = ROOT / 'B9A11T_ODB'
EX = ROOT / 'B9A11T_exports'
REGIONS = ('inner_left','inner_right')
PROPS = ('E1','E2','G12')
ALL_IDS = ('base','E1_M','E1_P','E2_M','E2_P','G12_M','G12_P')
FRACTION = 0.03


def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for z in iter(lambda:f.read(1<<20), b''):h.update(z)
    return h.hexdigest()


def get_b9a3():
    p = HERE/'65AM_B9A3_W_reference_NO.py'
    if not p.is_file():raise FileNotFoundError('Missing original B9A3 Windows script: '+str(p))
    spec=importlib.util.spec_from_file_location('b9a11t_original_B9A3',str(p))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jobname(region, cid):
    return 'T_%s_%s'%('L' if region=='inner_left' else 'R',cid)


def original_path(mod,region):
    return mod.INP_DIR/('B9A3_'+region+'.inp')


def keywords(line):
    return line.strip().upper().replace(' ','')


def elastic_rows(lines):
    hits=[i for i,s in enumerate(lines) if keywords(s).startswith('*ELASTIC,TYPE=ENGINEERINGCONSTANTS')]
    if len(hits)!=1:raise RuntimeError('Expected one engineering-constants elastic card; got %d'%len(hits))
    i=hits[0]
    if i+2>=len(lines):raise RuntimeError('Incomplete engineering constants block')
    try:
        row0=[float(x.strip()) for x in lines[i+1].split(',') if x.strip()]
        row1=[float(x.strip()) for x in lines[i+2].split(',') if x.strip()]
    except ValueError:raise RuntimeError('Nonnumeric Abaqus engineering constants')
    if len(row0)!=8 or len(row1)!=1:raise RuntimeError('Expected 8+1 elastic constants')
    return i+1, i+2, np.asarray(row0+row1,dtype=float)


def verify_boundary(lines,desc,anchor,region):
    pos=[i for i,s in enumerate(lines) if keywords(s)=='*BOUNDARY']
    if len(pos)!=1:raise RuntimeError('Expected exactly one *BOUNDARY block')
    data=[]
    for line in lines[pos[0]+1:]:
        if line.lstrip().startswith('*'):break
        if line.strip():data.append(line.strip())
    if len(data)!=112*2*3:raise RuntimeError('Unexpected number of interface displacement constraints: %d'%len(data))
    vals={}
    for ln in data:
        cells=[s.strip() for s in ln.split(',')]
        if len(cells)!=4:raise RuntimeError('Unexpected boundary row: '+ln)
        key=(cells[0].upper(),int(cells[1]),int(cells[2]))
        if key in vals:raise RuntimeError('Duplicate prescribed degree: '+str(key))
        vals[key]=float(cells[3])
    left,right=desc['interfaces']
    largest=0.
    for i,(il,ir) in enumerate(desc['map']):
        for side,iface,j in (('left',left,il),('right',right,ir)):
            lab=desc['faces'][side][i]
            u=anchor[iface]['u'][j]
            for k in range(3):
                key=('BLOCK-1.%d'%lab,k+1,k+1)
                if key not in vals:raise RuntimeError('Missing imposed displacement '+str(key))
                largest=max(largest,abs(vals[key]-float(u[k])))
    if largest>2e-7:raise RuntimeError('Original input interface q not equal saved anchor: %.7g mm'%largest)
    return largest


def source_audit():
    m=get_b9a3()
    anchor,hash0=m.anchor()
    src={}
    nominal=np.asarray(anchor['material'],float)
    for region in REGIONS:
        desc=m.grid(anchor,region)
        fn=original_path(m,region)
        if not fn.is_file():raise FileNotFoundError('Run old B9A3 generate to produce '+str(fn))
        lines=fn.read_text(encoding='utf-8').splitlines(keepends=True)
        i,j,constants=elastic_rows(lines)
        if not np.allclose(constants[[0,1,6]],nominal,atol=1e-6,rtol=0):
            raise RuntimeError('Source B9A3 elastic constants differ from saved anchor')
        if not np.allclose(constants[[2,3,4,5,7,8]], [12000.,.28,.28,.40,4500.,3500.], atol=1e-6,rtol=0):
            raise RuntimeError('Other original B9A3 elastic constants changed; investigate first')
        boundarygap=verify_boundary(lines,desc,anchor,region)
        src[region]=dict(lines=lines,indices=(i,j),constants=constants,sha=sha(fn),path=fn,desc=desc,boundary_gap=boundarygap)
    return m,anchor,hash0,nominal,src


def expected_cases(nominal):
    ans={}
    for prop,index in zip(PROPS,(0,1,2)):
        for sign,code in ((-1,'M'),(1,'P')):
            x=nominal.copy()
            x[index]*=1.+sign*FRACTION
            ans[prop+'_'+code]=x
    return ans


def audit():
    m,a,h,nominal,src=source_audit()
    print('B9A11T AUDIT PASS: exact same 65V q in both old B9A3 input decks')
    print('Anchor hash:',h,' | nominal (MPa):',nominal.tolist())
    print('Perturbation fraction:',FRACTION)
    print('Expected 12 new FEM jobs; two existing B9A3 base ODBs are reused.')
    for region in REGIONS:
        print(region,'template',src[region]['path'],'SHA',src[region]['sha'])
    return m,a,h,nominal,src


def generate():
    m,a,h,nominal,src=audit()
    cases=expected_cases(nominal)
    JOBS.mkdir(parents=True,exist_ok=True)
    man=JOBS/'B9A11T_input_manifest.json'
    if man.exists():raise RuntimeError('Manifest already exists. Preserve earlier B9A11T results.')
    entries=[]
    for region in REGIONS:
        orig=src[region]
        for cid,x in cases.items():
            name=jobname(region,cid)
            dest=JOBS/(name+'.inp')
            if dest.exists() or (JOBS/(name+'.odb')).exists():raise RuntimeError('Old job protected: '+name)
            lines=orig['lines'][:]
            nine=orig['constants'].copy()
            nine[[0,1,6]]=x
            i,j=orig['indices']
            lines[i]=', '.join('%.12g'%v for v in nine[:8])+'\n'
            lines[j]='%.12g'%nine[8]+'\n'
            dest.write_text(''.join(lines),encoding='utf-8')
            ii,jj,after=elastic_rows(dest.read_text(encoding='utf-8').splitlines())
            if np.max(abs(after-nine))>1e-7:raise RuntimeError('Unexpected modified constants '+name)
            if any(lines[k]!=orig['lines'][k] for k in range(len(lines)) if k not in (i,j)):
                raise RuntimeError('Other input contents changed '+name)
            entries.append(dict(region=region,case=cid,job=name,material_MPa=x.tolist(),
                                input_sha256=sha(dest),original_B9A3_input_sha256=orig['sha']))
            print('GENERATED',name,'| material MPa',x.tolist())
    record=dict(step='B9A11T',anchor_sha256=h,fraction=FRACTION,baseline_material_MPa=nominal.tolist(),
                same_prescribed_interface_q=True,existing_B9A3_baselines_reused=True,
                n_jobs=12,entries=entries)
    man.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('DONE: 12 inputs in',str(JOBS))
    print('Run using direct Abaqus job commands in Windows CMD; do NOT call .cmd files.')


def get_fields(m,anchor,region,odbpath,stapath):
    from odbAccess import openOdb
    if not odbpath.is_file() or not stapath.is_file():raise FileNotFoundError('Missing ODB/STA: '+str(odbpath))
    if 'COMPLETED SUCCESSFULLY' not in stapath.read_text(errors='ignore').upper():
        raise RuntimeError('Abaqus not successful: '+str(stapath))
    odb=openOdb(path=str(odbpath.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step '+str(odbpath))
        frame=odb.steps['Loading'].frames[-1]
        if abs(float(frame.frameValue)-1.0)>1e-6:raise RuntimeError('Final frame is not t=1.0')
        inst=m.get_instance(odb)
        coords={int(n.label):np.asarray(n.coordinates,float) for n in inst.nodes}
        vals={int(u.nodeLabel):np.asarray(u.data,float).reshape(-1)[:3] for u in frame.fieldOutputs['U'].getSubset(region=inst).values}
        expected=m.grid(anchor,region)
        if len(coords)!=len(expected['nodes']) or len(vals)!=len(coords):raise RuntimeError('Unexpected U/node counts in '+str(odbpath))
        labels=np.asarray(sorted(coords),np.int32)
        xyz=np.stack([coords[int(k)] for k in labels])
        U=np.stack([vals[int(k)] for k in labels])
        expected_xyz={int(k):np.asarray([x,y,z]) for k,x,y,z in expected['nodes']}
        mx=max(np.max(abs(coords[k]-expected_xyz[k])) for k in coords)
        if mx>2e-5:raise RuntimeError('Node/geometry differs from nominal B9A3 grid')
        # Explicitly confirm each left/right prescribed boundary degree is equal to the original 65V q.
        gap=0.
        for i,(il,ir) in enumerate(expected['map']):
            for side,iface,j in (('left',expected['interfaces'][0],il),('right',expected['interfaces'][1],ir)):
                label=expected['faces'][side][i]
                gap=max(gap,float(np.max(abs(vals[label]-anchor[iface]['u'][j]))))
        if gap>1e-5:raise RuntimeError('ODB boundary q changed: '+str(odbpath)+' gap '+str(gap))
        return labels,xyz,U,dict(odb_sha256=sha(odbpath),boundary_max_error_mm=gap,geometry_max_error_mm=float(mx))
    finally:odb.close()


def export():
    m,a,h,nominal,src=audit()
    manfile=JOBS/'B9A11T_input_manifest.json'
    if not manfile.is_file():raise FileNotFoundError('First generate the 12 inputs')
    man=json.loads(manfile.read_text())
    if man['anchor_sha256']!=h or abs(man['fraction']-FRACTION)>1e-12:raise RuntimeError('Wrong/mismatched manifest')
    expected=expected_cases(nominal)
    for row in man['entries']:
        f=JOBS/(row['job']+'.inp')
        if not f.is_file() or sha(f)!=row['input_sha256']:raise RuntimeError('Input changed '+row['job'])
        if src[row['region']]['sha']!=row['original_B9A3_input_sha256']:raise RuntimeError('Original baseline input changed')
    EX.mkdir(parents=True,exist_ok=True)
    target=EX/'B9A11T_U_reference.npz';report=EX/'B9A11T_windows_summary.json'
    if target.exists() or report.exists():raise RuntimeError('Refusing overwrite existing exports')
    payload=dict(anchor_sha256=np.asarray(h),fraction=np.asarray(FRACTION),
                 case_ids=np.asarray(ALL_IDS),material_MPa=np.asarray([nominal]+[expected[k] for k in ALL_IDS[1:]],float))
    checks=[]
    for region in REGIONS:
        seq=[];labels0=None;coords0=None
        for cid in ALL_IDS:
            if cid=='base':
                f=m.INP_DIR/('B9A3_'+region+'.odb')
            else:f=JOBS/(jobname(region,cid)+'.odb')
            lab,xyz,U,d=get_fields(m,a,region,f,f.with_suffix('.sta'))
            if labels0 is None:labels0=lab;coords0=xyz
            if not np.array_equal(lab,labels0) or np.max(abs(xyz-coords0))>1e-8:
                raise RuntimeError('Mesh node order/coordinates do not match '+str(f))
            seq.append(U)
            checks.append(dict(region=region,case=cid,odb=str(f),**d))
            print('EXPORT CHECK PASS',region,cid,'max q gap mm',d['boundary_max_error_mm'])
        payload['labels_'+region]=labels0
        payload['xyz_'+region]=coords0
        payload['U_'+region]=np.stack(seq)
    np.savez_compressed(str(target),**payload)
    report.write_text(json.dumps(dict(step='B9A11T',anchor_sha256=h,source_manifest_sha256=sha(manfile),
          npz_sha256=sha(target),fraction=FRACTION,cases=14,comparisons='fixed q, material +/- 3%, two regions',
          not_full_beam_coupling=True,all_jobs_pass=True,checks=checks),indent=2))
    print('EXPORT COMPLETE: copy B9A11T_exports folder to Mac project:',str(target))



def compliance_eigenvalues(constants):
    """Check positive definiteness of orthotropic compliance in Abaqus engineering convention."""
    E1,E2,E3,n12,n13,n23,G12,G13,G23=[float(v) for v in constants]
    if min(E1,E2,E3,G12,G13,G23)<=0:
        raise RuntimeError('Nonpositive Young/shear modulus')
    S=np.array([[1/E1,-n12/E1,-n13/E1],
                [-n12/E1,1/E2,-n23/E2],
                [-n13/E1,-n23/E2,1/E3]],float)
    eig=np.linalg.eigvalsh(S)
    if np.min(eig)<=0:raise RuntimeError('Unstable orthotropic elastic tensor: '+str(eig))
    return eig


def quality():
    """Independent post-export finite-difference and provenance checks; no training."""
    m,a,h,nominal,src=source_audit()
    ref=EX/'B9A11T_U_reference.npz'
    meta=EX/'B9A11T_windows_summary.json'
    manifest=JOBS/'B9A11T_input_manifest.json'
    dest=EX/'B9A11T_FEM_quality.json'
    if dest.exists():raise RuntimeError('Existing QC report protected: '+str(dest))
    for f in (ref,meta,manifest):
        if not f.is_file():raise FileNotFoundError(str(f))
    md=json.loads(meta.read_text())
    man=json.loads(manifest.read_text())
    if md.get('npz_sha256')!=sha(ref) or man.get('anchor_sha256')!=h:
        raise RuntimeError('B9A11T export/anchor checksum mismatch')
    by_job={row['job']:row for row in man['entries']}
    checks=[]
    for region in REGIONS:
        for cid in ALL_IDS[1:]:
            job=jobname(region,cid)
            if job not in by_job:raise RuntimeError('Missing planned job '+job)
            f=JOBS/(job+'.inp')
            if not f.is_file() or sha(f)!=by_job[job]['input_sha256']:
                raise RuntimeError('Input SHA mismatch '+job)
            lines=f.read_text(encoding='utf-8').splitlines(keepends=True)
            _,_,constants=elastic_rows(lines)
            desired=np.asarray(src[region]['constants'],float).copy()
            desired[[0,1,6]]=expected_cases(nominal)[cid]
            if not np.allclose(constants,desired,atol=1e-6,rtol=0):
                raise RuntimeError('Incorrect constitutive constants '+job)
            bnd=verify_boundary(lines,src[region]['desc'],a,region)
            eig=compliance_eigenvalues(constants)
            checks.append(dict(job=job,min_compliance_eigenvalue=float(min(eig)),
                               input_sha256=sha(f),max_imposed_q_gap_mm=bnd))
    with np.load(ref,allow_pickle=False) as z:
        if tuple(z['case_ids'].astype(str))!=ALL_IDS:
            raise RuntimeError('Wrong case ordering')
        if not np.allclose(z['material_MPa'],np.asarray([nominal]+[
               expected_cases(nominal)[cid] for cid in ALL_IDS[1:]],float),rtol=0,atol=1e-6):
            raise RuntimeError('Incorrect material grid in NPZ')
        quality_rows=[]
        warnings=[]
        for region in REGIONS:
            U=np.asarray(z['U_'+region],dtype=float)
            if U.shape!=(7,1008,3) or not np.isfinite(U).all():
                raise RuntimeError('Invalid FEM displacement arrays '+region)
            for j,prop in enumerate(PROPS):
                um,up=U[1+2*j],U[2+2*j]
                u0=U[0]
                odd=(up-um)/2.
                even=(up+um)/2.-u0
                rms_odd=float(np.sqrt(np.mean(odd*odd)))
                rms_even=float(np.sqrt(np.mean(even*even)))
                ratio=rms_even/max(rms_odd,1e-15)
                if ratio>0.25:warnings.append('%s %s: finite-difference even/odd %.3g exceeds 0.25; consider step-size convergence'%(region,prop,ratio))
                quality_rows.append(dict(region=region,property=prop,
                    expected_fraction=FRACTION,
                    finite_difference_signal_rms_mm=rms_odd,
                    finite_difference_nonlinearity_rms_mm=rms_even,
                    even_over_odd_ratio=ratio,
                    signal_over_sigma_0p0002mm=rms_odd/0.0002))
    report=dict(step='B9A11T-REVISED',stage='FEM_finite_difference_QC',
        npz_sha256=sha(ref),anchor_sha256=h,material_baseline_MPa=nominal.tolist(),
        fraction=FRACTION,all_12_material_inputs_audited=True,
        identical_fixed_interface_q_on_all_inputs=True,
        ODB_provenance_from_original_export_metadata=True,
        finite_difference_even_odd_is_only_a_screen_not_convergence_proof=True,
        new_Abaqus_runs_required=0,case_checks=checks,quality=quality_rows,warnings=warnings)
    dest.write_text(json.dumps(report,indent=2))
    print('B9A11T QC PASS | 12 unchanged matched-q material inputs | '+str(dest))
    for s in warnings:print('QC WARNING:',s)
    print('No extra Abaqus solutions; no model fitting. If warnings appear, later verify with a smaller perturbation.')


def selftest():
    lines=['*Material, name=MAT\n','*Elastic, type=ENGINEERING CONSTANTS\n',
           '46500,12500,12000,.28,.28,.40,4750,4500\n','3500\n','*Step, name=Loading\n']
    i,j,v=elastic_rows(lines)
    assert (i,j)==(2,3) and np.allclose(v[[0,1,6]],[46500,12500,4750])
    cases=expected_cases(np.asarray([46500.,12500.,4750.]))
    assert len(cases)==6 and abs(cases['G12_P'][2]-4892.5)<1e-7
    assert jobname('inner_right','E2_M')=='T_R_E2_M'
    assert min(compliance_eigenvalues([46500,12500,12000,.28,.28,.40,4750,4500,3500]))>0
    try:
        compliance_eigenvalues([-1.,12500,12000,.28,.28,.40,4750,4500,3500])
        raise AssertionError('Nonstable tensor not rejected')
    except RuntimeError:pass
    print('B9A11T WINDOWS SELFTEST PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('selftest','audit','generate','export','qc'))
    stage=p.parse_args().stage
    if stage=='selftest':selftest()
    elif stage=='audit':audit()
    elif stage=='generate':generate()
    elif stage=='export':export()
    else:quality()