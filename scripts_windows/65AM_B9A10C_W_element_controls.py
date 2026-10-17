# -*- coding: utf-8 -*-
'''B9A10-C Windows Abaqus. Run stages check, generate, extract from ODBS.
Read-only B9A3_ODB and B9A6_ODB source decks. NEVER overwrite old outputs.
'''
from __future__ import print_function
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
JOBS=ROOT/'B9A10C_ODB'; OUT=ROOT/'B9A10C_exports'
REG={'L':'inner_left','R':'inner_right'}; TYPES={'R':'C3D8R','F':'C3D8'}
BCS=('FREE','REPLAY')
OUTPUT='''*OUTPUT, FIELD, FREQUENCY=1
*NODE OUTPUT
U, RF
*ELEMENT OUTPUT, POSITION=INTEGRATION POINTS, DIRECTIONS=YES
S, LE, COORD
*OUTPUT, HISTORY, FREQUENCY=1
*ENERGY OUTPUT
ALLAE, ALLIE, ALLSE, ALLWK
'''
def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def nm(r,t,bc):return 'B9A10C_%s_%s_%s'%(r,t,bc)
def src(r,bc):
    n=REG[r]
    if bc=='FREE':return ROOT/'B9A3_ODB'/('B9A3_'+n+'.inp')
    return ROOT/'B9A6_ODB'/('B9A6_'+n+'_FEM.inp')
def deck_check(text,bc):
    if len(re.findall(r'(?im)^\*ELEMENT\s*,\s*TYPE=C3D8R\b',text))!=1:raise RuntimeError('Expected one C3D8R section')
    if len(re.findall(r'(?im)^\*END STEP\s*$',text))!=1:raise RuntimeError('Expected one step')
    if len(re.findall(r'(?im)^\*OUTPUT\s*,\s*FIELD\b',text))!=1:raise RuntimeError('Expected one field-output block')
    for token in ('*ELASTIC, TYPE=ENGINEERING CONSTANTS','*STEP, NAME=LOADING, NLGEOM=YES'):
        if token not in text.upper():raise RuntimeError('Missing original parameter '+token)
    nodes=re.search(r'(?ims)^\*NODE\s*\n(.*?)(?=^\*)',text)
    elems=re.search(r'(?ims)^\*ELEMENT[^\n]*\n(.*?)(?=^\*)',text)
    if nodes is None or elems is None:raise RuntimeError('Missing mesh')
    nnode=sum(bool(s.strip()) and not s.lstrip().startswith('**') for s in nodes.group(1).splitlines())
    nelem=sum(bool(s.strip()) and not s.lstrip().startswith('**') for s in elems.group(1).splitlines())
    if (nnode,nelem)!=(1008,728):raise RuntimeError('Unexpected geometry: %d nodes %d elements'%(nnode,nelem))
    bcblock=text.upper().split('*BOUNDARY',1)[-1].split('*OUTPUT',1)[0]
    nbc=sum(bool(s.strip()) and not s.lstrip().startswith('**') for s in bcblock.splitlines())
    expected=3024 if bc=='REPLAY' else 672
    if nbc!=expected:raise RuntimeError('Incorrect prescribed DOFs: %s %d != %d'%(bc,nbc,expected))
    return nbc
def check():
    for r in REG:
        for bc in BCS:
            p=src(r,bc)
            if not p.is_file():raise FileNotFoundError('Missing original input '+str(p))
            deck_check(p.read_text(),bc)
    print('B9A10C CHECK PASS | 4 original decks ready; no output created')
def generate():
    check()
    if not JOBS.is_dir():raise RuntimeError('In Windows File Explorer create '+str(JOBS))
    expected=[JOBS/(nm(r,t,bc)+'.inp') for r in REG for t in TYPES for bc in BCS]
    if any(p.exists() for p in expected) or (JOBS/'B9A10C_input_manifest.json').exists():
        raise RuntimeError('Existing B9A10C generation protected; do not overwrite')
    entries=[]
    for r in REG:
        for bc in BCS:
            source=src(r,bc);text=source.read_text();dofs=deck_check(text,bc)
            for t in TYPES:
                s,n=re.subn(r'(?im)^(\*ELEMENT\s*,\s*TYPE=)C3D8R\b',
                    lambda m:m.group(1)+TYPES[t],text)
                if n!=1:raise RuntimeError('Failed to replace element type')
                s,n=re.subn(r'(?ims)^\*OUTPUT\s*,\s*FIELD\b[^\n]*\n.*?(?=^\*END STEP)',lambda m:OUTPUT,s)
                if n!=1:raise RuntimeError('Failed to replace output block')
                target=JOBS/(nm(r,t,bc)+'.inp');target.write_text(s)
                entries.append(dict(job=nm(r,t,bc),region=REG[r],element=TYPES[t],BC=bc,
                    original_input_sha256=sha(source),input_sha256=sha(target),
                    prescribed_dofs=dofs,nodes=1008,elements=728))
                print('GENERATED',target.name)
    (JOBS/'B9A10C_input_manifest.json').write_text(json.dumps({'jobs':entries},indent=2))
    print('Eight independent decks generated; see README run commands')
def nodal(field,inst):
    return {int(v.nodeLabel):np.asarray(v.data,float).reshape(-1)[:3] for v in field.getSubset(region=inst).values}
def elemfield(field,inst):
    d={}
    for v in field.getSubset(region=inst).values:
        key=(int(v.elementLabel),int(v.integrationPoint))
        if key in d:raise RuntimeError('Duplicate integration point '+str(key))
        d[key]=np.asarray(v.data,float).reshape(-1)
    return d
def energy(step):
    ans={}
    for key in ('ALLAE','ALLIE','ALLSE','ALLWK'):
        matched=[]
        for name,hr in step.historyRegions.items():
            if key in hr.historyOutputs:
                h=hr.historyOutputs[key].data
                if len(h):matched.append((str(name),float(h[-1][1])))
        ans[key]={'status':'available','value':matched[0][1],'region':matched[0][0]} if len(matched)==1 else {
            'status':'not_saved' if len(matched)==0 else 'ambiguous','value':None}
    a=ans['ALLAE']['value'];b=ans['ALLIE']['value']
    ratio=None if a is None or b is None or abs(b)<1e-15 else 100*abs(a)/abs(b)
    return ans,ratio
def read_case(r,t,bc):
    from odbAccess import openOdb
    job=nm(r,t,bc);odbpath=JOBS/(job+'.odb');inp=JOBS/(job+'.inp');sta=JOBS/(job+'.sta')
    if not (odbpath.is_file() and sta.is_file() and inp.is_file()):
        raise FileNotFoundError('Missing .inp/.odb/.sta for '+job)
    if 'COMPLETED SUCCESSFULLY' not in sta.read_text(errors='ignore').upper():
        raise RuntimeError('Not completed successfully: '+job)
    odb=openOdb(path=str(odbpath.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing step Loading '+job)
        step=odb.steps['Loading'];frame=step.frames[-1]
        if abs(float(frame.frameValue)-1)>1e-6:raise RuntimeError('Final frame time !=1 '+job)
        inst=next((v for k,v in odb.rootAssembly.instances.items() if k.lower()=='block-1'),None)
        if inst is None:raise RuntimeError('No BLOCK-1 '+job)
        for k in ('U','RF','S','COORD'):
            if k not in frame.fieldOutputs:raise RuntimeError('Missing output '+k+' '+job)
        nodes=sorted(inst.nodes,key=lambda v:int(v.label))
        elems=sorted(inst.elements,key=lambda v:int(v.label))
        if len(nodes)!=1008 or len(elems)!=728:raise RuntimeError('Incorrect mesh in '+job)
        labels=np.asarray([int(v.label) for v in nodes],int)
        xyz=np.asarray([v.coordinates for v in nodes],float)
        elabels=np.asarray([int(v.label) for v in elems],int)
        connectivity=np.asarray([list(v.connectivity) for v in elems],int)
        u=nodal(frame.fieldOutputs['U'],inst);rf=nodal(frame.fieldOutputs['RF'],inst)
        if len(u)!=1008 or len(rf)!=1008:raise RuntimeError('Incomplete U/RF '+job)
        S=elemfield(frame.fieldOutputs['S'],inst);C=elemfield(frame.fieldOutputs['COORD'],inst)
        nip=1 if t=='R' else 8
        if len(S)!=728*nip or len(C)!=728*nip:raise RuntimeError('Wrong integration point count '+job)
        Sarray=np.empty((728,nip,6));Carray=np.empty((728,nip,3))
        for k,el in enumerate(elabels):
            for ip in range(1,nip+1):
                key=(int(el),ip)
                if key not in S or key not in C or len(S[key])<6 or len(C[key])<3:
                    raise RuntimeError('Missing S/COORD at '+str(key)+' '+job)
                Sarray[k,ip-1]=S[key][:6];Carray[k,ip-1]=C[key][:3]
        h,ratio=energy(step)
        arrays=dict(node_labels=labels,X=xyz,U=np.asarray([u[int(k)] for k in labels],float),
            RF=np.asarray([rf[int(k)] for k in labels],float),element_labels=elabels,
            connectivity=connectivity,S=Sarray,IP_COORD=Carray)
        summary=dict(job=job,region=REG[r],element=TYPES[t],BC=bc,odb_sha256=sha(odbpath),
            input_sha256=sha(inp),history_energy=h,ALLAE_over_ALLIE_pct=ratio)
        return arrays,summary
    finally:odb.close()
def extract():
    check()
    if not JOBS.is_dir() or not OUT.is_dir():
        raise RuntimeError('Create B9A10C_ODB and B9A10C_exports manually in File Explorer')
    manifest=JOBS/'B9A10C_input_manifest.json'
    if not manifest.is_file():raise RuntimeError('Missing generated manifest')
    for info in json.loads(manifest.read_text())['jobs']:
        p=JOBS/(info['job']+'.inp')
        if sha(p)!=info['input_sha256']:raise RuntimeError('Input changed: '+str(p))
    dest=OUT/'B9A10C_fields.npz';js=OUT/'B9A10C_windows_summary.json'
    if dest.exists() or js.exists():raise RuntimeError('Existing results protected')
    data={};audits=[]
    for r in REG:
        for t in TYPES:
            for bc in BCS:
                case=nm(r,t,bc);a,b=read_case(r,t,bc)
                for k,v in a.items():data[case+'_'+k]=v
                audits.append(b)
                print('READ',case,'ALLAE/ALLIE %',b['ALLAE_over_ALLIE_pct'])
    np.savez_compressed(str(dest),**data)
    js.write_text(json.dumps({'step':'65AM-B9A10-C','output_sha256':sha(dest),'jobs':audits,
      'warning':'ALLAE is not a direct measure of nodal hourglass force; integration and stiffness also differ.'},indent=2))
    print('EXTRACT PASS:',dest,js)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('check','generate','extract'),required=True)
    a=p.parse_args()
    if a.stage=='check':check()
    elif a.stage=='generate':generate()
    else:extract()
