"""65W-0: create a NEW full-beam middle-material Abaqus INP safely.

Copies nine engineering elastic constants verbatim (as numeric values) from
Step65V verified center patch INP to a distinct full-beam reference INP.
Everything else in full model (nodes/elements/contact/loads/BCs) is preserved.
NO auto-Abaqus job launches, NO modification of either original INP.

From Windows ODBS:
  abaqus python scripts\65W_0_prepare_full_middle.py --stage audit --base-inp 3Point.inp
  abaqus python scripts\65W_0_prepare_full_middle.py --stage prepare --base-inp 3Point.inp
If the two models use different material names, supply --patch-material and --base-material.
Then run from ODBS with Abaqus solver executable:
  abaqus job=65W_FullMiddle input=online_results_7region_65W_middle\0_full_reference\65W_FullMiddle.inp cpus=4 interactive
Check the .sta/.dat and ODB completion before field extraction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent.parent
DEFAULT_PATCH=ROOT/'online_results_7region_65V_middle'/'65V_D_lm_convergence'/'final'/'center.inp'
DEFAULT_FULL=ROOT/'3Point.inp'
DEST=ROOT/'online_results_7region_65W_middle'/'0_full_reference'
MIDDLE=(46500.,12500.,4750.)


def decode(p):
    for enc in ('utf-8-sig','cp1252'):
        try:return Path(p).read_text(encoding=enc),enc
        except UnicodeDecodeError:pass
    raise RuntimeError('Unknown input file encoding: '+str(p))


def find_material_elastic(lines,material):
    current=None
    needle=material.lower().replace(' ','')
    matches=[]
    for i,line in enumerate(lines):
        s=line.strip()
        if re.match(r'^\*material\b',s,re.I):
            mm=re.search(r'\bname\s*=\s*([^,]+)',s,re.I)
            current=mm.group(1).strip().strip('"\'').lower().replace(' ','') if mm else None
        elif re.match(r'^\*elastic\b',s,re.I) and current==needle:
            if 'engineering constants' not in s.lower():
                raise RuntimeError('Selected %s elasticity must be ENGINEERING CONSTANTS: %s'%(material,s))
            j=i+1
            while j<len(lines) and not lines[j].strip().startswith('*'):
                j+=1
            content=lines[i+1:j]
            numbers=[]
            for ln in content:
                if not ln.strip() or ln.strip().startswith('**'):continue
                for item in ln.split(','):
                    item=item.strip()
                    if not item:continue
                    try:numbers.append(float(item))
                    except ValueError:raise RuntimeError('Nonnumeric elastic material entry: '+repr(item))
            if len(numbers)!=9:raise RuntimeError('%s expected 9 engineering constants, got %d'%(material,len(numbers)))
            matches.append((i+1,j,numbers))
    if len(matches)!=1:raise RuntimeError('Expected ONE matching %s engineering constants block; found %d'%(material,len(matches)))
    return matches[0]


def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()


def prepare(args):
    base=Path(args.base_inp).resolve();patch=Path(args.patch_inp).resolve()
    if not patch.exists() or not base.exists():
        raise FileNotFoundError('Require full base and 65V center patch INP files: %s / %s'%(base,patch))
    btext,benc=decode(base);ptext,penc=decode(patch)
    # splitlines retaining newlines to preserve source input outside material.
    b=btext.splitlines(True);p=ptext.splitlines(True)
    ba,bb,bvals=find_material_elastic(b,args.base_material)
    pa,pb,pvals=find_material_elastic(p,args.patch_material)
    # Typical E1/E2/G12 and E3/G13 mapping for the existing specimen.
    if not (abs(pvals[0]-MIDDLE[0])<1e-6 and abs(pvals[1]-MIDDLE[1])<1e-6 and
            abs(pvals[6]-MIDDLE[2])<1e-6):
        raise RuntimeError('Center patch constants do not match middle E1/E2/G12: %s'%pvals)
    if any(v<=0 for v in (pvals[0],pvals[1],pvals[2],pvals[6],pvals[7],pvals[8])):
        raise RuntimeError('Invalid elastic moduli/shear constants')
    if not args.allow_nonbaseline_base and not (abs(bvals[0]-45000)<1e-6 and
        abs(bvals[1]-12000)<1e-6 and abs(bvals[6]-4500)<1e-6):
        raise RuntimeError('Base input not original baseline (45000,12000,4500). Use explicit --allow-nonbaseline-base only after verifying the template.')
    report={
      'base_inp':str(base),'center_patch_inp':str(patch),
      'base_material_name':args.base_material,'patch_material_name':args.patch_material,
      'base_constants_E1_E2_E3_nu12_nu13_nu23_G12_G13_G23':bvals,
      'patch_constants_E1_E2_E3_nu12_nu13_nu23_G12_G13_G23':pvals,
      'changes':['Selected full-FEM material ENGINEERING CONSTANTS only'],
      'source_sha256':sha(btext),'patch_sha256':sha(ptext),
      'NOTE':'Must still verify geometry, loading, contact, mesh, frame/units and end-step convergence against the FE-NO specimen. Only material copying is automatic.'
    }
    if args.stage=='audit':
        print(json.dumps(report,indent=2));return
    if DEST.exists() and any(DEST.iterdir()):
        raise RuntimeError('Nonempty reference output folder protected: '+str(DEST))
    DEST.mkdir(parents=True)
    newline='\r\n' if '\r\n' in btext else '\n'
    insert=[','.join(('%0.12g'%v) for v in pvals[:8])+','+newline,
            ('%0.12g'%pvals[8])+','+newline]
    out=b[:ba]+insert+b[bb:]
    otext=''.join(out)
    oa,ob,oval=find_material_elastic(out,args.base_material)
    if oval!=pvals:raise RuntimeError('Material round-trip check failed')
    # No diffs outside the target elastic rows, by construction.
    assert b[:ba]==out[:ba] and b[bb:]==out[ba+len(insert):]
    target=DEST/'65W_FullMiddle.inp'
    with open(str(target),'w',encoding=benc if benc!='utf-8-sig' else 'utf-8') as f:f.write(otext)
    report['generated_full_inp']=str(target)
    report['generated_sha256']=sha(otext)
    report['generator_only_changes_material_block']=True
    with open(str(DEST/'65W_FullMiddle_material_audit.json'),'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    print('NEXT: abaqus job=65W_FullMiddle input="%s" cpus=4 interactive'%target)


def selftest():
    a='*Material, name=GlassEpoxy\n*Elastic, type=ENGINEERING CONSTANTS\n45000.,12000.,12000., .28, .28, .4, 4500., 4500.\n3500.,\n*Material, name=Steel\n*Elastic\n200000.,0.3\n'
    start,end,mat=find_material_elastic(a.splitlines(True),'GlassEpoxy')
    assert start==2 and end==4 and len(mat)==9 and mat[0]==45000.
    print('65W-0 selftest PASS: material selector and 9 engineering constants')


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--stage',choices=['selftest','audit','prepare'],default='audit')
    pa.add_argument('--base-inp',default=str(DEFAULT_FULL))
    pa.add_argument('--patch-inp',default=str(DEFAULT_PATCH))
    pa.add_argument('--base-material',default='GlassEpoxy')
    pa.add_argument('--patch-material',default='GlassEpoxy')
    pa.add_argument('--allow-nonbaseline-base',action='store_true')
    args=pa.parse_args()
    if args.stage=='selftest':selftest()
    else:prepare(args)

if __name__=='__main__':main()
