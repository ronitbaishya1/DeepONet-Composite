#!/usr/bin/env python3
"""B9A11I-C: safe PREPARE-only localized E2 reduction in existing 3Point.inp.

Run from ODBS on Windows with Abaqus Python or Python3:
 abaqus python scripts\65AM_B9A11I_C_W_damage_inp.py --stage check --base-inp 3Point.inp --manifest B9A11I_damage_manifest.csv
 abaqus python scripts\65AM_B9A11I_C_W_damage_inp.py --stage prepare --base-inp 3Point.inp --manifest B9A11I_damage_manifest.csv --out-dir B9A11I_damage_INP

No Abaqus jobs are launched. Never edits base or existing INPs. Refuses ambiguous
composite mesh/section assignments. New damaged material modifies E2 ONLY.
Composite elements selected by center-position ellipsoid; this is a pilot
*mesh-resolved stiffness-loss surrogate*, not a physical delamination.
"""
from __future__ import print_function
import argparse, csv, hashlib, json, math, re
from pathlib import Path


def keyword(line):
    s=line.lstrip()
    return s.startswith('*') and not s.startswith('**')

def name(line):return line.lstrip().split(',',1)[0].strip().upper().replace(' ','')

def keyval(line,key):
    m=re.search(r'(?:^|,)\s*'+re.escape(key)+r'\s*=\s*([^,\r\n]+)',line,re.I)
    return m.group(1).strip() if m else None

def next_keyword(lines,start):
    k=start+1
    while k<len(lines) and not keyword(lines[k]):k+=1
    return k

def numbers(line,cast=float):
    return [cast(q.strip()) for q in line.strip().split(',') if q.strip()]

def hashbytes(b):return hashlib.sha256(b).hexdigest()

def read_inp(path):
    raw=path.read_bytes()
    try:text=raw.decode('utf-8-sig');encoding='utf-8'
    except UnicodeDecodeError:text=raw.decode('cp1252');encoding='cp1252'
    return text.splitlines(keepends=True),hashbytes(raw),encoding

def find_composite(lines):
    parts=[]
    starts=[i for i,s in enumerate(lines) if name(s)=='*PART']
    for i in starts:
        j=next((k for k in range(i+1,len(lines)) if name(lines[k])=='*ENDPART'),None)
        if j is None:raise RuntimeError('Unclosed Part')
        sections=[k for k in range(i+1,j) if name(lines[k])=='*SOLIDSECTION' and
                  (keyval(lines[k],'material') or '').casefold()=='glassepoxy']
        if sections:parts.append((i,j,sections))
    if len(parts)!=1 or len(parts[0][2])!=1:
        raise RuntimeError('Expected exactly one composite part with ONE GlassEpoxy solid section. Found '+str(parts))
    start,end,secs=parts[0]; section=secs[0]
    nd={};els={};sets={}
    for i in range(start+1,end):
        key=name(lines[i])
        if key=='*NODE':
            for s in lines[i+1:next_keyword(lines,i)]:
                if not s.strip() or s.lstrip().startswith('**'):continue
                n=numbers(s,float)
                if len(n)!=4:raise RuntimeError('Unexpected composite *Node format: '+s.strip())
                ident=int(n[0]);nd[ident]=tuple(n[1:])
        elif key=='*ELEMENT':
            typ=(keyval(lines[i],'type') or '').upper()
            if typ!='C3D8R':
                raise RuntimeError('Composite part has non-C3D8R element block; explicit audit required: '+typ)
            labels=[]
            for s in lines[i+1:next_keyword(lines,i)]:
                if not s.strip() or s.lstrip().startswith('**'):continue
                a=numbers(s,int)
                if len(a)!=9:raise RuntimeError('Expected 8-node C3D8R on each *Element line')
                if a[0] in els:raise RuntimeError('Duplicate composite element labels')
                els[a[0]]=a[1:];labels.append(a[0])
            en=keyval(lines[i],'elset')
            if en:sets.setdefault(en.casefold(),set()).update(labels)
        elif key=='*ELSET':
            en=keyval(lines[i],'elset')
            if not en:raise RuntimeError('Elset missing name')
            gen='generate' in lines[i].casefold()
            entries=set()
            for s in lines[i+1:next_keyword(lines,i)]:
                if not s.strip() or s.lstrip().startswith('**'):continue
                try:v=numbers(s,int)
                except ValueError:raise RuntimeError('Composite *Elset uses nested named sets; cannot safely rewrite')
                if gen:
                    if len(v)!=3 or v[2]<=0:raise RuntimeError('Unrecognized *Elset generate row')
                    entries.update(range(v[0],v[1]+1,v[2]))
                else:entries.update(v)
            sets.setdefault(en.casefold(),set()).update(entries)
    if not nd or not els:raise RuntimeError('No composite part node/element records')
    original_elset=keyval(lines[section],'elset')
    if not original_elset:raise RuntimeError('GlassEpoxy section has no element set')
    orig=sets.get(original_elset.casefold())
    if not orig:
        raise RuntimeError('Cannot prove GlassEpoxy solid-section element membership: '+original_elset)
    if not orig.issubset(set(els)):raise RuntimeError('Composite section contains unknown/non-C3D8R elements')
    if len(orig)<50:raise RuntimeError('Unexpectedly small composite element section')
    for el in orig:
        for node in els[el]:
            if node not in nd:raise RuntimeError('Undefined node in element '+str(el))
    materials=[i for i,s in enumerate(lines) if name(s)=='*MATERIAL' and
               (keyval(s,'name') or '').casefold()=='glassepoxy']
    if len(materials)!=1 or not (materials[0]>end):
        raise RuntimeError('Expected unique GlassEpoxy material defined outside composite Part')
    mi=materials[0]
    # Material belongs to the model data; stop before unrelated interaction keywords.
    endmat=next((j for j in range(mi+1,len(lines)) if keyword(lines[j]) and
         name(lines[j]) not in ('*ELASTIC','*DENSITY')),len(lines))
    if endmat<len(lines) and name(lines[endmat]) in (
            '*PLASTIC','*VISCOELASTIC','*HYPERELASTIC','*DAMAGEINITIATION',
            '*DAMAGEEVOLUTION','*CREEP','*USERMATERIAL'):
        raise RuntimeError('Template uses additional composite constitutive behavior '+
                           name(lines[endmat])+'; damage material copy requires manual review')
    blocks=[i for i in range(mi+1,endmat) if name(lines[i])=='*ELASTIC']
    if len(blocks)!=1 or 'engineering constants' not in lines[blocks[0]].casefold().replace(' ',' '):
        # accept both ENGINEERING CONSTANTS and ENGINEERINGCONSTANTS
        if len(blocks)!=1 or 'engineeringconstants' not in lines[blocks[0]].casefold().replace(' ',''):
            raise RuntimeError('Expected exactly one engineering-constants *Elastic for GlassEpoxy')
    start_el=blocks[0]+1;stop_el=next_keyword(lines,blocks[0]);vv=[]
    for row in lines[start_el:stop_el]:
        if row.strip() and not row.lstrip().startswith('**'):vv.extend(numbers(row,float))
    if len(vv)!=9:raise RuntimeError('Expected 9 engineering constants, got '+str(len(vv)))
    if any(x<=0 for x in [vv[0],vv[1],vv[2],vv[6],vv[7],vv[8]]):raise RuntimeError('Invalid healthy material constants')
    if len(set(els))!=len(els):raise RuntimeError('Nonunique element IDs')
    return {'part':(start,end),'sec':section,'mat':mi,'endmat':endmat,'elastic':blocks[0],
            'elastic_rows':[k for k in range(start_el,stop_el) if lines[k].strip() and not lines[k].lstrip().startswith('**')],
            'nodes':nd,'elements':els,'original_set':orig,
            'original_elset':original_elset,'const':vv,'part_name':keyval(lines[start],'name')}

def block_elset(name,elems,nl):
    l=['*Elset, elset='+name+nl]
    a=sorted(elems)
    for k in range(0,len(a),12):l.append(', '.join(str(x) for x in a[k:k+12])+','+nl)
    return l

def selection(info,row):
    x,y,z=[float(row[q]) for q in ('x_mm','y_mm','z_mm')]
    rx,ry,rz=[float(row[q]) for q in ('radius_x_mm','radius_y_mm','radius_z_mm')]
    if min(rx,ry,rz)<=0:raise RuntimeError('Non-positive damage radii')
    center=[];selected=[]
    for e in sorted(info['original_set']):
        c=[sum(info['nodes'][v][j] for v in info['elements'][e])/8 for j in range(3)]
        if ((c[0]-x)/rx)**2+((c[1]-y)/ry)**2+((c[2]-z)/rz)**2<=1:
            selected.append(e);center.append(c)
    if len(selected)<4 or len(selected)>.15*len(info['original_set']):
        raise RuntimeError('Damage zone for '+row['case_id']+' selects '+str(len(selected))+
                           ' of '+str(len(info['original_set']))+' elements; review center/radii and mesh')
    return set(selected)

def modify(lines,info,chosen,loss):
    if not 0<loss<.70:raise RuntimeError('Unrealistic/unsafe E2 loss for pilot')
    newline='\r\n' if any(s.endswith('\r\n') for s in lines) else '\n'
    old=info['original_set'];healthy=old-chosen
    if not healthy or not chosen:raise RuntimeError('Incomplete healthy/damage partition')
    c=list(info['const']);c[1]*=1-loss
    # Section first: copy the original keyword and optional section data lines exactly.
    sec=info['sec'];after=next_keyword(lines,sec)
    original_header=lines[sec]
    h=re.sub(r'(?i)(\belset\s*=\s*)[^,\r\n]+',r'\g<1>B9I_HEALTHY',original_header,count=1)
    if h==original_header:raise RuntimeError('Could not redirect existing composite section')
    dam=re.sub(r'(?i)(\belset\s*=\s*)[^,\r\n]+',r'\g<1>B9I_DAMAGED',original_header,count=1)
    dam=re.sub(r'(?i)(\bmaterial\s*=\s*)[^,\r\n]+',r'\g<1>DamagedGlassEpoxy',dam,count=1)
    if dam==original_header:raise RuntimeError('Did not change damaged material')
    new_section=block_elset('B9I_HEALTHY',healthy,newline)+block_elset('B9I_DAMAGED',chosen,newline)
    new_section.extend([h]+lines[sec+1:after]+[dam]+lines[sec+1:after])
    # Clone ALL existing supported material behaviors (Elastic, optional Density).
    # Change only E2; retain other properties and every original nonmaterial line.
    ctext=list(lines[info['mat']:info['endmat']])
    ctext[0]=re.sub(r'(?i)(\bname\s*=\s*)[^,\r\n]+',r'\g<1>DamagedGlassEpoxy',ctext[0],count=1)
    er=info['elastic_rows']
    if len(er)!=2:raise RuntimeError('Expected 8+1 engineering constants rows for source material')
    a,b=[e-info['mat'] for e in er]
    ctext[a]=', '.join('%0.12g'%v for v in c[:8])+','+newline
    ctext[b]='%0.12g'%c[8]+','+newline
    # Handle offsets with descending splices so indices refer to original lines.
    result=list(lines)
    actions=[(info['mat'],info['mat'],ctext),(sec,after,new_section)]
    for left,right,payload in sorted(actions,reverse=True):result[left:right]=payload
    return result,c

def read_manifest(path):
    with path.open('r',newline='') as f:rows=list(csv.DictReader(f))
    cols={'case_id','x_mm','y_mm','z_mm','radius_x_mm','radius_y_mm','radius_z_mm','damage_E2_fraction',
          'E1_healthy_MPa','E2_healthy_MPa','G12_healthy_MPa'}
    if not rows or not cols.issubset(rows[0]):raise RuntimeError('Incomplete damage CSV manifest')
    if any(not re.fullmatch('I_DMG_X[0-9]{2}_D[0-9]{2}',r['case_id']) for r in rows):
        raise RuntimeError('Unexpected case IDs in manifest')
    if len({r['case_id'] for r in rows})!=len(rows):raise RuntimeError('Duplicate case IDs')
    return rows

def run(args):
    base=Path(args.base_inp).resolve();manifest=Path(args.manifest).resolve()
    if not base.is_file() or not manifest.is_file():raise FileNotFoundError('Base INP / manifest absent')
    lines,orig_hash,encoding=read_inp(base)
    info=find_composite(lines);rows=read_manifest(manifest)
    if (abs(info['const'][0]-45000)>1e-4 or abs(info['const'][1]-12000)>1e-4 or
        abs(info['const'][6]-4500)>1e-4):
        raise RuntimeError('Template is not the original 45000/12000/4500 healthy reference; audit first')
    planned=[]
    for r in rows:
        if any(abs(float(r[k])-ref)>1e-6 for k,ref in [('E1_healthy_MPa',45000),('E2_healthy_MPa',12000),('G12_healthy_MPa',4500)]):
            raise RuntimeError('Mismatch between manifest healthy material and template')
        d=float(r['damage_E2_fraction'])
        chosen=selection(info,r)
        candidate,newmat=modify(lines,info,chosen,d)
        # Full input round-trip parser: healthy assignment must contain exactly
        # the complement of damaged elements; untouched healthy constants remain.
        verification=find_composite(candidate)
        if verification['original_set']!=info['original_set']-chosen or verification['const']!=info['const']:
            raise RuntimeError('Generated section/material roundtrip differs from expected partition')
        # Verify count, independently roundtrip new material with exact selected elset.
        joined=''.join(candidate)
        if '*Material, name=DamagedGlassEpoxy' not in joined:raise RuntimeError('Damaged material missing')
        if joined.count('*Elset, elset=B9I_DAMAGED')!=1:raise RuntimeError('Damaged set missing/duplicate')
        planned.append({'case_id':r['case_id'],'damage_elements':len(chosen),
                        'healthy_elements':len(info['original_set'])-len(chosen),
                        'damage_E2_fraction':d,'new_E2_MPa':newmat[1],
                        'candidate':joined,'set':sorted(chosen)})
    print('PASS: unique composite part:',info['part_name'],'| nodes',len(info['nodes']),
          '| composite section elements',len(info['original_set']),'| cases',len(planned))
    for p in planned:print(p['case_id'],'damaged elements',p['damage_elements'],'E2',p['new_E2_MPa'])
    if args.stage=='check':
        print('CHECK ONLY: NO files written and NO Abaqus jobs submitted');return
    out=Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing overwrite nonempty output '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    for p in planned:
        dest=out/(p['case_id']+'.inp')
        dest.write_bytes(p['candidate'].encode(encoding))
        report.append({k:v for k,v in p.items() if k!='candidate'})
        report[-1]['input_sha256']=hashbytes(dest.read_bytes())
    (out/'B9A11I_damage_input_audit.json').write_text(json.dumps({
        'source_input':str(base),'source_sha256':orig_hash,'template_material_constants':info['const'],
        'source_mesh_section':info['original_elset'],
        'conditions':'Only GlassEpoxy composite solid-section assignment and new damaged material introduced',
        'no_jobs_submitted':True,'pilot_only':True,'cases':report},indent=2))
    print('PREPARE PASS:',out,'| files created:',len(planned))
    print('Submit Abaqus jobs manually and audit .sta/.odb before exporting fields.')

def selftest():
    # Abstract 8-brick element deck with >50 specimen elements (7x8x1).
    lines=['*Part, name=Composite\n','*Node\n']
    nodes=[];els=[];node_id=1;el_id=1
    ids={}
    for i in range(9):
      for j in range(2):
       for k in range(2):
        ids[(i,j,k)]=node_id;lines.append('%d, %g, %g, %g\n'%(node_id,float(i),float(j),float(k)));node_id+=1
    lines.append('*Element, type=C3D8R, elset=Body\n')
    for i in range(8):
        a=[ids[(i,0,0)],ids[(i+1,0,0)],ids[(i+1,1,0)],ids[(i,1,0)],
           ids[(i,0,1)],ids[(i+1,0,1)],ids[(i+1,1,1)],ids[(i,1,1)]]
        for _ in range(7):
            # Synthetic duplicated geometries per unique element ID, only to test parser.
            lines.append(str(el_id)+', '+', '.join(str(v) for v in a)+'\n');el_id+=1
    lines += ['*Solid Section, elset=Body, material=GlassEpoxy\n',',\n',
              '*End Part\n','*Assembly\n','*End Assembly\n','*Material, name=GlassEpoxy\n',
              '*Elastic, type=ENGINEERING CONSTANTS\n',
              '45000,12000,12000,.28,.28,.4,4500,4500,\n','3500,\n','*Step, name=Loading\n','*End Step\n']
    info=find_composite(lines)
    if len(info['original_set'])!=56:raise RuntimeError('Selftest expected 56 elements')
    r={'case_id':'I_DMG_X01_D01','x_mm':'4.5','y_mm':'.5','z_mm':'.5',
       'radius_x_mm':'.55','radius_y_mm':'1','radius_z_mm':'1','damage_E2_fraction':'.2'}
    selected=selection(info,r)
    modified,vals=modify(lines,info,selected,.2)
    s=''.join(modified)
    if abs(vals[1]-9600)>1e-6 or 'B9I_DAMAGED' not in s or s.count('DamagedGlassEpoxy')!=2:
        raise RuntimeError('Material and elset selftest failed')
    print('B9A11I-C SELFTEST PASS | parses composite input, creates partition + material')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['selftest','check','prepare'],required=True)
    ap.add_argument('--base-inp',default='3Point.inp')
    ap.add_argument('--manifest',default='B9A11I_damage_manifest.csv')
    ap.add_argument('--out-dir',default='B9A11I_damage_INP')
    args=ap.parse_args()
    if args.stage=='selftest':selftest()
    else:run(args)
