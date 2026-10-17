# -*- coding: utf-8 -*-
"""B9A10-J / Abaqus Standard 2025 / Windows: identical-kinematics hourglass-control experiments.
Source: B9A10C_{L,R}_R_REPLAY.inp and matching completed ODB; originals are read-only.
Creates 8 NEW decks: two regions x stiffness (0.5,1.0,2.0) and ENHANCED.
Stages: selftest, check, generate, extract. No source overwrite.
"""
from __future__ import print_function
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'B9A10C_ODB'
OUT=ROOT/'B9A10J_exports'
JOBS=ROOT/'B9A10J_ODB'
CASE={'S05':('STIFFNESS',0.5),'S10':('STIFFNESS',1.0),'S20':('STIFFNESS',2.0),'ENH':('ENHANCED',None)}
REGS=('L','R')

def digest(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def src(r):return BASE/('B9A10C_%s_R_REPLAY.inp'%r)
def name(r,setting):return 'B9A10J_%s_%s'%(r,setting)
def verify_deck(txt):
    upper=txt.upper()
    if len(re.findall(r'(?im)^\*SOLID SECTION\b',txt))!=1:raise RuntimeError('Expected exactly one *SOLID SECTION')
    if len(re.findall(r'(?im)^\*ELEMENT\s*,\s*TYPE=C3D8R\b',txt))!=1:raise RuntimeError('Expected C3D8R only')
    if len(re.findall(r'(?im)^\*END STEP\s*$',txt))!=1:raise RuntimeError('Expected one step')
    if re.search(r'(?im)^\*SECTION CONTROLS\b',txt):raise RuntimeError('Base input already uses section controls')
    section=re.search(r'(?im)^\*SOLID SECTION[^\n]*$',txt)
    if section is None or 'CONTROLS=' in section.group(0).upper():raise RuntimeError('Unexpected source section controls')
    if '*STEP, NAME=LOADING, NLGEOM=YES' not in upper:raise RuntimeError('Expected Abaqus Standard nonlinear step')
    m=re.search(r'(?ims)^\*BOUNDARY\s*\n(.*?)(?=^\*)',txt)
    if not m:raise RuntimeError('No full displacement block')
    lines=[x.strip() for x in m.group(1).splitlines() if x.strip() and not x.lstrip().startswith('**')]
    if len(lines)!=3024:raise RuntimeError('Expected 3024 prescribed DOFs; got %d'%len(lines))
    indices=[]
    for line in lines:
        p=[a.strip() for a in line.split(',')]
        if len(p)<4 or int(p[1])!=int(p[2]) or not 1<=int(p[1])<=3:raise RuntimeError('Invalid boundary row: '+line)
        indices.append((p[0].upper(),int(p[1])))
    if len(set(indices))!=3024:raise RuntimeError('Repeated prescribed DOFs')
    if len(set(x[0] for x in indices))!=1008:raise RuntimeError('Expected 1008 prescribed nodes')
    return len(lines)

def modify(inp,tag):
    typ,factor=CASE[tag]
    verify_deck(inp)
    control='B9J_'+tag
    if typ=='STIFFNESS':cmd='*SECTION CONTROLS, NAME=%s, HOURGLASS=STIFFNESS\n%.8g\n'%(control,factor)
    else:cmd='*SECTION CONTROLS, NAME=%s, HOURGLASS=ENHANCED\n'%(control)
    def rep(m):
        return cmd+m.group(0).rstrip()+', CONTROLS='+control
    new,n=re.subn(r'(?im)^\*SOLID SECTION[^\n]*',rep,inp)
    if n!=1:raise RuntimeError('Cannot safely insert controls')
    if new.count('CONTROLS='+control)!=1 or len(re.findall(r'(?im)^\*SECTION CONTROLS\b',new))!=1:
        raise RuntimeError('Invalid control insertion')
    # no other model/data changes, including field histories and boundary values
    return new

def selftest():
    s='*HEADING\n*ELEMENT, TYPE=C3D8R\n1, 1,2,3,4,5,6,7,8\n*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI\n,\n*STEP, NAME=Loading, NLGEOM=YES\n*BOUNDARY\n'
    s+=''.join('BLOCK-1.%d, %d, %d, 0.0\n'%(i,j,j) for i in range(1,1009) for j in (1,2,3))
    s+='*OUTPUT, FIELD, FREQUENCY=1\n*NODE OUTPUT\nU, RF\n*END STEP\n'
    for tag in CASE:
        m=modify(s,tag)
        assert ('CONTROLS=B9J_'+tag) in m
        assert ('HOURGLASS=ENHANCED' in m)==(tag=='ENH')
        assert m.split('*BOUNDARY')[1]==s.split('*BOUNDARY')[1]
    print('B9A10J WINDOWS SELFTEST PASS | correct section controls | identical BC data')

def check():
    for r in REGS:
        inp=src(r);odb=inp.with_suffix('.odb');sta=inp.with_suffix('.sta')
        for p in (inp,odb,sta):
            if not p.is_file():raise FileNotFoundError('Missing reference '+str(p))
        verify_deck(inp.read_text())
        if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():raise RuntimeError('Reference job not completed '+r)
    print('B9A10J WINDOWS CHECK PASS | two B9A10C full-field references | original source untouched')

def generate():
    check()
    if not JOBS.is_dir():raise RuntimeError('Create B9A10J_ODB in Windows File Explorer first')
    targets=[JOBS/(name(r,s)+'.inp') for r in REGS for s in CASE]
    meta=JOBS/'B9A10J_input_manifest.json'
    if meta.exists() or any(p.exists() for p in targets):raise RuntimeError('Existing B9A10J files protected; do not overwrite')
    entries=[]
    for r in REGS:
        original=src(r);base=original.read_text();verify_deck(base)
        for setting,(method,factor) in CASE.items():
            p=JOBS/(name(r,setting)+'.inp');new=modify(base,setting);p.write_text(new)
            entries.append(dict(job=name(r,setting),region=r,setting=setting,method=method,scale=factor,
                source_input_sha256=digest(original),input_sha256=digest(p),prescribed_dofs=3024))
            print('GENERATED',p.name,'control',setting)
    meta.write_text(json.dumps(dict(step='65AM-B9A10-J',cases=entries),indent=2))
    print('B9A10J GENERATED 8 decks. Run Abaqus jobs as in README.')

def energy(step):
    out={}
    for k in ('ALLAE','ALLIE','ALLSE','ALLWK'):
        samples=[(str(n),float(r.historyOutputs[k].data[-1][1])) for n,r in step.historyRegions.items()
                 if k in r.historyOutputs and len(r.historyOutputs[k].data)]
        out[k]={'status':'available','value':samples[0][1]} if len(samples)==1 else {
            'status':'missing' if not samples else 'ambiguous','value':None}
    a=out['ALLAE']['value'];b=out['ALLIE']['value']
    out['ALLAE_over_ALLIE_pct']=100*abs(a)/abs(b) if a is not None and b is not None and abs(b)>1e-12 else None
    return out

def read_odb(path,require_stress=True):
    from odbAccess import openOdb
    odb=openOdb(path=str(path.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('No Loading step '+str(path))
        step=odb.steps['Loading'];frame=step.frames[-1]
        if abs(float(frame.frameValue)-1.0)>1e-6:raise RuntimeError('Wrong final frame '+str(path))
        inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
        if inst is None:raise RuntimeError('Missing BLOCK-1 '+str(path))
        nodes=sorted(inst.nodes,key=lambda v:int(v.label))
        els=sorted(inst.elements,key=lambda v:int(v.label))
        if (len(nodes),len(els))!=(1008,728):raise RuntimeError('Mesh changed '+str(path))
        labels=np.asarray([int(v.label) for v in nodes],dtype=np.int64)
        xyz=np.asarray([v.coordinates for v in nodes],dtype=float)
        e_lab=np.asarray([int(v.label) for v in els],dtype=np.int64)
        conn=np.asarray([v.connectivity for v in els],dtype=np.int64)
        def nodal(key):
            if key not in frame.fieldOutputs:raise RuntimeError('Missing '+key+' in '+str(path))
            d={int(v.nodeLabel):np.asarray(v.data,dtype=float).reshape(-1)[:3] for v in frame.fieldOutputs[key].getSubset(region=inst).values}
            if set(d)!=set(labels.tolist()):raise RuntimeError('Incomplete '+key+' '+str(path))
            return np.asarray([d[int(l)] for l in labels],float)
        arr=dict(node_labels=labels,X=xyz,U=nodal('U'),RF=nodal('RF'),element_labels=e_lab,connectivity=conn)
        if require_stress:
            if 'S' not in frame.fieldOutputs:raise RuntimeError('Stress S not saved '+str(path))
            values=frame.fieldOutputs['S'].getSubset(region=inst).values
            data={};rot={}
            for v in values:
                key=(int(v.elementLabel),int(v.integrationPoint))
                if key in data:raise RuntimeError('Duplicate S at '+str(key))
                data[key]=np.asarray(v.data,float).reshape(-1)[:6]
                try:q=np.asarray(v.localCoordSystem,float).reshape(3,3)
                except Exception as ex:raise RuntimeError('No stress orientation for '+str(key)+' '+str(ex))
                if np.max(abs(q@q.T-np.eye(3)))>1e-3 or abs(np.linalg.det(q)-1)>1e-3:
                    raise RuntimeError('Invalid stress orientation at '+str(key))
                rot[key]=q
            if len(data)!=728:raise RuntimeError('Expected 728 one-point stresses')
            arr['S']=np.asarray([data[(int(e),1)] for e in e_lab],float)
            arr['Q_global_to_local']=np.asarray([rot[(int(e),1)] for e in e_lab],float)
        return arr,energy(step)
    finally:odb.close()

def extract():
    check()
    meta=JOBS/'B9A10J_input_manifest.json'
    if not meta.is_file():raise FileNotFoundError('Generate inputs first: '+str(meta))
    if not OUT.is_dir():raise RuntimeError('Create B9A10J_exports manually in File Explorer')
    dest=OUT/'B9A10J_stabilization_fields.npz';summ=OUT/'B9A10J_windows_summary.json'
    if dest.exists() or summ.exists():raise RuntimeError('B9A10J exports protected; do not overwrite')
    entries=json.loads(meta.read_text())['cases']
    payload={};reports=[]
    for entry in entries:
        r=entry['region'];job=entry['job'];p=JOBS/(job+'.inp');odb=JOBS/(job+'.odb');sta=JOBS/(job+'.sta')
        if digest(p)!=entry['input_sha256'] or digest(src(r))!=entry['source_input_sha256']:
            raise RuntimeError('Input provenance mismatch '+job)
        if not odb.is_file() or not sta.is_file() or 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
            raise RuntimeError('Missing or incomplete job '+job)
        a,h=read_odb(odb)
        baseline,_=read_odb(src(r).with_suffix('.odb'))
        for k in ('node_labels','element_labels','connectivity'):
            if not np.array_equal(a[k],baseline[k]):raise RuntimeError('Mesh labels mismatch '+job+' '+k)
        if np.max(abs(a['X']-baseline['X']))>1e-7:raise RuntimeError('Mesh coords changed '+job)
        err=float(np.max(abs(a['U']-baseline['U'])))
        if err>2e-5:raise RuntimeError('BC displacement mismatch '+job+' %g'%err)
        for k,v in a.items():payload[job+'_'+k]=v
        reports.append(dict(job=job,region=r,setting=entry['setting'],scale=entry['scale'],method=entry['method'],
            odb_sha256=digest(odb),input_sha256=digest(p),ref_odb_sha256=digest(src(r).with_suffix('.odb')),
            maximum_prescribed_U_difference_mm=err,energy=h))
        print('EXTRACTED',job,'max U delta',err,'ALLAE/ALLIE',h['ALLAE_over_ALLIE_pct'])
    for r in REGS:
        b,_=read_odb(src(r).with_suffix('.odb'))
        for k,v in b.items():payload['B9A10J_%s_REFERENCE_%s'%(r,k)]=v
    np.savez_compressed(str(dest),**payload)
    summ.write_text(json.dumps(dict(step='65AM-B9A10-J',cases=reports,
        source='Existing B9A10C full-field FEM replay',export_sha256=digest(dest),
        warning='Changing stiffness with fixed U identifies sensitivity; not itself recovered exact hourglass force.'),indent=2))
    print('B9A10J EXTRACTION PASS',dest,summ)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['selftest','check','generate','extract'])
    a=ap.parse_args();{'selftest':selftest,'check':check,'generate':generate,'extract':extract}[a.stage]()
