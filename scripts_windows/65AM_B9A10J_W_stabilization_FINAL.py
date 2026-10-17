# -*- coding: utf-8 -*-
"""B9A10-J / Abaqus Standard 2025 / Windows: identical-kinematics hourglass-control experiments.
Source: B9A10C_{L,R}_R_REPLAY.inp and matching completed ODB; originals are read-only.
Creates 8 NEW decks: two regions x stiffness (0.5,1.0,2.0) and ENHANCED.
Stages: selftest, check, generate, repair-enh, validate-enh, extract.
Repair edits only two failed ENH .inp files and synchronizes manifest; successful jobs are unchanged.
"""
from __future__ import print_function
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'B9A10C_ODB'
OUT=ROOT/'B9A10J_exports'
JOBS=ROOT/'B9A10J_FIXED_ODB'
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

def average_shear_modulus(inp):
    """ENGINEERING CONSTANTS: E1 E2 E3 nu12 nu13 nu23 G12 G13 / G23."""
    blocks=list(re.finditer(r'(?im)^\*ELASTIC\s*,\s*TYPE\s*=\s*ENGINEERING CONSTANTS[^\n]*$', inp))
    if len(blocks)!=1:raise RuntimeError('Expected one engineering-constants material block')
    rest=inp[blocks[0].end():]
    data=[]
    for line in rest.splitlines():
        s=line.strip()
        if s.startswith('*'):break
        if not s or s.startswith('**'):continue
        data.extend(float(v.strip()) for v in s.split(',') if v.strip())
    if len(data)!=9:raise RuntimeError('Expected 9 engineering constants; got %d'%len(data))
    g12,g13,g23=data[6],data[7],data[8]
    if min(g12,g13,g23)<=0:raise RuntimeError('Invalid shear moduli')
    return (g12+g13+g23)/3.

def enhanced_at_model_scope(text):
    """Move the single ENHANCED section-control definition outside *PART/*ASSEMBLY.

    In Abaqus/Standard, *SECTION CONTROLS is model-level; *SOLID SECTION
    may refer to that definition from part scope via CONTROLS=B9J_ENH.
    """
    pattern=r'(?im)^\*SECTION\s+CONTROLS[^\n]*$'
    matches=list(re.finditer(pattern,text))
    if len(matches)!=1:
        raise RuntimeError('Expected exactly one *SECTION CONTROLS in enhanced deck')
    line=matches[0].group(0)
    if not re.search(r'(?i)NAME\s*=\s*B9J_ENH\b',line):
        raise RuntimeError('Unexpected section controls name: '+line)
    if not re.search(r'(?i)HOURGLASS\s*=\s*ENHANCED\b',line):
        raise RuntimeError('Expected HOURGLASS=ENHANCED: '+line)
    rest=text[:matches[0].start()]+(text[matches[0].end()+1:] if text[matches[0].end():].startswith('\n') else text[matches[0].end():])
    # Do not modify any data line or material/BC/mesh fields.
    # Insert just before the first model-level part, or before model-level
    # *NODE for directly defined elements (no part blocks).
    insertion=re.search(r'(?im)^\*(?:PART\b|NODE(?=\s|,|$)|ELEMENT(?=\s|,|$)|ASSEMBLY\b)',rest)
    if not insertion:
        raise RuntimeError('Could not locate beginning of model data')
    out=rest[:insertion.start()]+line+'\n'+rest[insertion.start():]
    if len(list(re.finditer(pattern,out)))!=1:
        raise RuntimeError('Enhanced controls count invalid after repair')
    solid=re.findall(r'(?im)^\*SOLID SECTION[^\n]*$',out)
    if len(solid)!=1 or not re.search(r'(?i)CONTROLS\s*=\s*B9J_ENH\b',solid[0]):
        raise RuntimeError('The solid section must reference CONTROLS=B9J_ENH')
    if out==text:
        # If already at model level this is valid and needs no editing.
        return out
    if out.count('*STEP')!=text.count('*STEP'):
        raise RuntimeError('Unexpected step change')
    return out

def modify(inp,tag):
    """Abaqus/Standard 2025. S10 is actual unmodified source."""
    typ,factor=CASE[tag]
    verify_deck(inp)
    if tag=='S10':return inp
    if tag in ('S05','S20'):
        G=average_shear_modulus(inp)
        rfg=(0.005*G)*factor
        def after_section(m):
            return m.group(0).rstrip('\n')+'\n*HOURGLASS STIFFNESS\n%.15g\n'%rfg
        modified,n=re.subn(r'(?ims)^\*SOLID SECTION[^\n]*\n.*?(?=^\*)',after_section,inp)
        if n!=1 or modified.count('*HOURGLASS STIFFNESS')!=1:
            raise RuntimeError('Cannot insert Abaqus Standard *HOURGLASS STIFFNESS')
        return modified
    if tag=='ENH':
        control='B9J_ENH'
        modified,n=re.subn(r'(?im)^\*SOLID SECTION[^\n]*$',lambda m:m.group(0).rstrip()+', CONTROLS='+control,inp)
        if n!=1:raise RuntimeError('Cannot add enhanced section controls')
        # New decks start with model-scope controls, not inside *PART.
        # The repair-enh stage handles already generated flawed ENH decks.
        return enhanced_at_model_scope(insert_model_control(modified))
    raise ValueError(tag)

def insert_model_control(inp):
    """Insert model-level controls before first PART/NODE/ASSEMBLY."""
    pos=re.search(r'(?im)^\*(?:PART\b|NODE(?=\s|,|$)|ELEMENT(?=\s|,|$)|ASSEMBLY\b)',inp)
    if not pos:raise RuntimeError('No model scope anchor found')
    return inp[:pos.start()]+'*SECTION CONTROLS, NAME=B9J_ENH, HOURGLASS=ENHANCED\n'+inp[pos.start():]

def selftest():
    s='*HEADING\n*ELEMENT, TYPE=C3D8R\n1, 1,2,3,4,5,6,7,8\n*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI\n,\n*MATERIAL, NAME=MAT\n*ELASTIC, TYPE=ENGINEERING CONSTANTS\n46500, 12500, 12000, .28, .28, .40, 4750, 4500\n3500\n*STEP, NAME=Loading, NLGEOM=YES\n*BOUNDARY\n'
    s+=''.join('BLOCK-1.%d, %d, %d, 0.0\n'%(i,j,j) for i in range(1,1009) for j in (1,2,3))
    s+='*OUTPUT, FIELD, FREQUENCY=1\n*NODE OUTPUT\nU, RF\n*END STEP\n'
    assert average_shear_modulus(s)==4250.0
    expected={ 'S05':10.625,'S20':42.5 }
    for tag in CASE:
        m=modify(s,tag)
        assert m.split('*BOUNDARY')[1]==s.split('*BOUNDARY')[1]
        assert ('*HOURGLASS STIFFNESS' in m)==(tag in expected)
        assert ('*SECTION CONTROLS' in m)==(tag=='ENH')
        if tag=='S10':assert m==s
        if tag in expected:
            assert abs(float(m.split('*HOURGLASS STIFFNESS\n')[1].splitlines()[0])-expected[tag])<1e-12
            assert 'CONTROLS=' not in m
            assert m.index('*HOURGLASS STIFFNESS')<m.index('*MATERIAL')
    part='*HEADING\nGenerated test\n*PART, NAME=Block\n*NODE\n1,0,0,0\n*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, CONTROLS=B9J_ENH\n,\n*END PART\n*ASSEMBLY, NAME=Assembly\n*END ASSEMBLY\n*SECTION CONTROLS, NAME=B9J_ENH, HOURGLASS=ENHANCED\n*STEP, NAME=Loading, NLGEOM=YES\n*END STEP\n'
    repaired=enhanced_at_model_scope(part)
    assert repaired.index('*SECTION CONTROLS')<repaired.index('*PART')
    assert repaired.count('CONTROLS=B9J_ENH')==1
    assert repaired.count('*SECTION CONTROLS')==1
    assert '*STEP, NAME=Loading' in repaired
    print('B9A10J STANDARD FIX SELFTEST PASS | unmodified 1x | *HOURGLASS STIFFNESS half/double | ENHANCED controls')

def check():
    for r in REGS:
        inp=src(r);odb=inp.with_suffix('.odb');sta=inp.with_suffix('.sta')
        for p in (inp,odb,sta):
            if not p.is_file():raise FileNotFoundError('Missing reference '+str(p))
        verify_deck(inp.read_text())
        if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():raise RuntimeError('Reference job not completed '+r)
    print('B9A10J WINDOWS FIX CHECK PASS | two B9A10C full-field references | original source untouched')

def generate():
    check()
    if not JOBS.is_dir():raise RuntimeError('Create B9A10J_FIXED_ODB in Windows File Explorer first')
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
    print('B9A10J FIXED GENERATED 8 decks. Run Abaqus jobs as in README.')

def repair_enh(validate_only=False):
    """Repair ONLY L_ENH and R_ENH failed decks, preserving six successful jobs.

    Updates only two ENH inputs and their SHA-256 in the existing manifest.
    No folders are made, deleted or renamed. All source decks stay untouched.
    """
    check()
    manifest=JOBS/'B9A10J_input_manifest.json'
    if not manifest.is_file():raise FileNotFoundError(str(manifest))
    m=json.loads(manifest.read_text())
    edits=[]
    for r in REGS:
        job=name(r,'ENH');path=JOBS/(job+'.inp')
        if not path.is_file():raise FileNotFoundError(str(path))
        rows=[v for v in m['cases'] if v['job']==job and v['region']==r and v['setting']=='ENH']
        if len(rows)!=1:raise RuntimeError('Manifest missing unique ENH entry for '+r)
        entry=rows[0]
        old_hash=digest(path)
        if old_hash!=entry['input_sha256']:
            raise RuntimeError('ENH input already changed since generation: '+str(path))
        if digest(src(r))!=entry['source_input_sha256']:
            raise RuntimeError('Source deck changed: '+str(src(r)))
        sta=JOBS/(job+'.sta')
        if sta.exists() and 'COMPLETED SUCCESSFULLY' in sta.read_text(errors='ignore').upper():
            raise RuntimeError('ENH job already succeeded; will not modify '+job)
        old=path.read_text();new=enhanced_at_model_scope(old)
        if old==new:
            print('ALREADY PLACED',job)
            continue
        if old.count('*SECTION CONTROLS')!=new.count('*SECTION CONTROLS'):
            raise RuntimeError('Section control count changed unexpectedly')
        # Sanity-check source preserved, outside the relocated keyword.
        def strip_control(t):
            return re.sub(r'(?im)^\*SECTION\s+CONTROLS[^\n]*\n','',t)
        if strip_control(old)!=strip_control(new):
            raise RuntimeError('Unexpected data change, aborting '+job)
        edits.append((path,old_hash,new,entry))
        print(('READY TO REPAIR' if validate_only else 'REPAIRING'),job,
              'move enhanced controls from part scope to model scope')
    if validate_only:
        print('B9A10J ENH VALIDATE PASS | changes required:',len(edits))
        return
    if not edits:
        print('B9A10J ENH: no changes needed')
        return
    # Only the two failed ENH input decks are modified; update the manifest
    # fingerprints for the exporter, with original hashes recorded for audit.
    for path,old_hash,new,entry in edits:
        path.write_text(new)
        entry['pre_repair_input_sha256']=old_hash
        entry['input_sha256']=digest(path)
        entry['repair_note']='Moved section controls outside part for Abaqus Standard 2025'
    manifest.write_text(json.dumps(m,indent=2))
    print('B9A10J ENH REPAIR PASS | repaired:',len(edits),'| successful six jobs untouched')
    print('Now rerun ONLY B9A10J_L_ENH and B9A10J_R_ENH in B9A10J_FIXED_ODB')

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
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['selftest','check','generate','repair-enh','validate-enh','extract'])
    a=ap.parse_args();{'selftest':selftest,'check':check,'generate':generate,'repair-enh':repair_enh,'validate-enh':lambda:repair_enh(True),'extract':extract}[a.stage]()
