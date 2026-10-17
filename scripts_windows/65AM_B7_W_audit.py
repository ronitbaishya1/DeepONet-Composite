#!/usr/bin/env python
"""65AM-B7 Windows Abaqus 2025: READ-ONLY six-case material/ODB/solver audit.

Run from ODBS with:
 abaqus python scripts\65AM_B7_W_audit.py --base-inp 3Point.inp --manifest 65AM_B6_cases.csv --odb-dir B6_ODB

Creates ONLY B7_abaqus_audit/B7_windows_audit.json; rejects overwrite.
Check ODB material, section orientations and contact separately before publication.
"""
from __future__ import print_function
import os,sys,json,csv,hashlib,argparse,re,math

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def parse_material(path):
    text=open(path,'r').read();lines=text.splitlines()
    start=[i for i,v in enumerate(lines) if re.match(r'^\s*\*Material\s*,\s*name\s*=\s*GlassEpoxy\b',v,re.I)]
    if len(start)!=1:raise RuntimeError('Need unique GlassEpoxy material in '+path)
    i=start[0]+1
    while i<len(lines):
        s=lines[i].strip()
        if re.match(r'^\*Material\s*,',s,re.I):break
        if re.match(r'^\*Elastic\s*,',s,re.I):
            if 'type=engineeringconstants' not in s.lower().replace(' ',''):
                raise RuntimeError('GlassEpoxy must use engineering constants')
            vals=[];pos=[];j=i+1
            while j<len(lines):
                t=lines[j].strip()
                if t.startswith('*') and not t.startswith('**'):break
                if t and not t.startswith('**'):
                    try:vals.extend(float(v) for v in t.split(',') if v.strip())
                    except Exception:raise RuntimeError('Bad engineering constants '+path)
                    pos.append(j)
                j+=1
            if len(vals)!=9 or len(pos)!=2:
                raise RuntimeError('Expected nine orthotropic constants over two lines in '+path)
            if not all(math.isfinite(x) for x in vals):raise RuntimeError('Nonfinite material constants')
            return vals,pos,lines
        i+=1
    raise RuntimeError('GlassEpoxy engineering constants not found '+path)

def read_odb_summary(path):
    from odbAccess import openOdb
    odb=openOdb(path=os.path.abspath(path),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step')
        step=odb.steps['Loading']
        if len(step.frames)<2:raise RuntimeError('Loading has too few frames')
        frame=step.frames[-1]
        for field in ('U','S','LE'):
            if field not in frame.fieldOutputs:raise RuntimeError('Missing '+field+' in final frame')
        if 'COMPOSITE-1' not in odb.rootAssembly.instances:
            raise RuntimeError('COMPOSITE-1 missing')
        inst=odb.rootAssembly.instances['COMPOSITE-1']
        if not inst.elements:raise RuntimeError('No composite elements')
        if any(el.type.upper()!='C3D8R' for el in inst.elements):
            raise RuntimeError('Composite mesh contains non-C3D8R elements')
        return dict(loading_frames=len(step.frames),last_frame_value=float(frame.frameValue),
                    composite_elements=len(inst.elements),
                    material_names=sorted([str(k) for k in odb.materials.keys()]),
                    section_names=sorted([str(k) for k in odb.sections.keys()]),
                    contact_output_present=any('CSTRESS' in x or 'CPRESS' in x for x in frame.fieldOutputs.keys()),
                    fields_present=list(('U','LE','S')))
    finally: odb.close()

def perform(base,manifest,odbdir,inspect_odb=True):
    vals0,pos0,lines0=parse_material(base)
    with open(manifest,'r') as f:rows=list(csv.DictReader(f))
    names=['B6_new_%02d'%k for k in range(1,7)]
    if len(rows)!=6 or [v.get('case_id') for v in rows]!=names:
        raise RuntimeError('B6 manifest needs exactly six cases in fixed order')
    entries=[]
    for row in rows:
        case=row['case_id'];inp=os.path.join(odbdir,case+'.inp')
        odb=os.path.join(odbdir,case+'.odb')
        sta=os.path.join(odbdir,case+'.sta')
        for f in (inp,odb,sta):
            if not os.path.isfile(f):raise IOError('Missing B6 file '+str(f))
        vals,pos,lines=parse_material(inp)
        for j,k in ((0,'E1_MPa'),(1,'E2_MPa'),(6,'G12_MPa')):
            if not math.isclose(vals[j],float(row[k]),rel_tol=1.e-7,abs_tol=1.e-5):
                raise RuntimeError(case+' mismatch between manifest and INP '+k)
        if any(not math.isclose(vals[j],vals0[j],rel_tol=1.e-12,abs_tol=1.e-9)
               for j in (2,3,4,5,7,8)):
            raise RuntimeError(case+' changed fixed orthotropic constants')
        stripped=lambda ls,pp:'\n'.join(t for j,t in enumerate(ls) if j not in pp)
        if stripped(lines,pos)!=stripped(lines0,pos0):
            raise RuntimeError(case+' changed non-material INP lines; STOP')
        status=open(sta,'r',errors='replace').read().upper()
        if not ('THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in status or
                'THE ANALYSIS HAS BEEN COMPLETED SUCCESSFULLY' in status):
            raise RuntimeError(case+' has no successful-completion confirmation in .sta; review Abaqus .sta/.log')
        odbsummary=read_odb_summary(odb) if inspect_odb else {}
        entries.append(dict(case_id=case,inp_sha256=digest(inp),odb_sha256=digest(odb),
                            sta_sha256=digest(sta),engineer_constants=vals,
                            inp_match=True,sta_completed=True,odb_checked=inspect_odb,**odbsummary))
    return dict(step='65AM-B7',case_count=6,case_ids=names,
                manifest_sha256=digest(manifest),base_inp_sha256=digest(base),
                fixed_orthotropic_constants={key:vals0[i] for i,key in ((2,'E3'),(3,'nu12'),(4,'nu13'),(5,'nu23'),(7,'G13'),(8,'G23'))},
                all_cases_passed=True,
                caveat='Input and output files audited; ODB material orientation/contact quality and reaction balance still require independent engineering review.',
                cases=entries)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-inp',default='3Point.inp')
    p.add_argument('--manifest',default='65AM_B6_cases.csv')
    p.add_argument('--odb-dir',default='B6_ODB')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    result=perform(args.base_inp,args.manifest,args.odb_dir)
    if args.check_only:
        print('B7 WINDOWS AUDIT PASS all six; no files created')
        return
    out=os.path.join('B7_abaqus_audit','B7_windows_audit.json')
    if os.path.isfile(out):raise RuntimeError('Existing audit protected: '+out)
    if not os.path.isdir(os.path.dirname(out)):
        raise IOError('Create B7_abaqus_audit with File Explorer before running')
    with open(out,'w') as f:json.dump(result,f,indent=2)
    print('B7 WINDOWS AUDIT PASS; copy',os.path.abspath(out),'to Mac project folder B7_abaqus_audit/')
if __name__=='__main__':main()
