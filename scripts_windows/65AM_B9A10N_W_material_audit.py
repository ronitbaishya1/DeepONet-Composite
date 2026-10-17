# -*- coding: utf-8 -*-
"""B9A10-N Windows: read-only audit of all 24 original B9A9 Abaqus input files.

Nothing is changed in an Abaqus deck/ODB. Requires only standard Python 3,
so either `python` or `abaqus python` works. The optional export writes ONLY
B9A10N_material_audit/B9A10N_windows_materials.json.

Run from ODBS:
 abaqus python scripts\65AM_B9A10N_W_material_audit.py --stage selftest
 abaqus python scripts\65AM_B9A10N_W_material_audit.py --stage check
 abaqus python scripts\65AM_B9A10N_W_material_audit.py --stage export
"""
from __future__ import print_function
import argparse, hashlib, json, math
from pathlib import Path

SLOTS=('E1','E2','E3','nu12','nu13','nu23','G12','G13','G23')
REGIONS=('L','R')
TOL=5e-8

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def scan_deck(p):
    lines=p.read_text(encoding='utf-8',errors='replace').splitlines()
    blocks=[line.strip() for line in lines]
    el=[i for i,line in enumerate(blocks) if line.upper().startswith('*ELASTIC') and 'ENGINEERING CONSTANTS' in line.upper()]
    if len(el)!=1:raise RuntimeError('%s: expected one orthotropic *ELASTIC declaration'%(p.name))
    i=el[0]; nums=[]
    for line in blocks[i+1:]:
        if line.startswith('*'):break
        nums += [float(x.strip()) for x in line.split(',') if x.strip()]
    if len(nums)!=9:raise RuntimeError('%s: expected 9 engineering constants; got %d'%(p.name,len(nums)))
    if any(not math.isfinite(x) for x in nums):raise RuntimeError('Non-finite constants '+p.name)
    if any(nums[j]<=0 for j in (0,1,2,6,7,8)):
        raise RuntimeError('Nonpositive elastic modulus '+p.name)
    if len([x for x in blocks if x.upper().startswith('*ELEMENT,') and 'TYPE=C3D8R' in x.upper()])!=1:
        raise RuntimeError('C3D8R element declaration unexpected '+p.name)
    ori=[j for j,line in enumerate(blocks) if line.upper().startswith('*ORIENTATION,') and 'MAT_ORI' in line.upper()]
    if len(ori)!=1:raise RuntimeError('MAT_ORI declaration missing or duplicated '+p.name)
    orientation=[float(x.strip()) for x in blocks[ori[0]+1].split(',') if x.strip()]
    if len(orientation)!=6 or max(abs(a-b) for a,b in zip(orientation,[1,0,0,0,1,0]))>TOL:
        raise RuntimeError('MAT_ORI not identity '+p.name)
    if not any(line.upper().startswith('*STEP,') and 'NLGEOM=YES' in line.upper() for line in blocks):
        raise RuntimeError('Expected NLGEOM=YES '+p.name)
    if not any(line.upper().startswith('*SOLID SECTION,') and 'ORIENTATION=MAT_ORI' in line.upper() for line in blocks):
        raise RuntimeError('Solid section does not use MAT_ORI '+p.name)
    constants=dict(zip(SLOTS,nums))
    # Reciprocity for orthotropic engineering constants, definition nuji = nuij Ej/Ei.
    constants['nu21']=nums[3]*nums[1]/nums[0]
    constants['nu31']=nums[4]*nums[2]/nums[0]
    constants['nu32']=nums[5]*nums[2]/nums[1]
    return dict(job=p.stem,inp_sha256=digest(p),constants=constants,
                material_orientation='global identity',element='C3D8R',nlgeom=True)

def all_jobs(root):
    folder=root/'B9A9_ODB'
    if not folder.is_dir():raise FileNotFoundError('B9A9_ODB folder missing: '+str(folder))
    out=[]
    for r in REGIONS:
        for i in range(1,13):
            p=folder/('B9A9_%s%02d.inp'%(r,i))
            if not p.is_file():raise FileNotFoundError('Missing original input deck '+str(p))
            record=scan_deck(p)
            constants=record['constants']
            fixed={'E3':12000,'nu12':.28,'nu13':.28,'nu23':.40,'G13':4500,'G23':3500}
            for k,v in fixed.items():
                if abs(constants[k]-v)>TOL*max(1,abs(v)):
                    raise RuntimeError('%s: %s is %g, expected %g'%(p.name,k,constants[k],v))
            out.append(record)
    return out

def selftest():
    import tempfile
    sample='''*HEADING
*PART, NAME=BLOCK
*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR
1,0,0,0,1,0
*ELEMENT, TYPE=C3D8R, ELSET=EALL
1,1,2,3,4,5,6,7,8
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI
,
*END PART
*MATERIAL, NAME=MAT
*ELASTIC, TYPE=ENGINEERING CONSTANTS
46500,12500,12000,.28,.28,.4,4750,4500
3500
*STEP, NAME=Loading, NLGEOM=YES
*END STEP
'''
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'B9A9_L01.inp';p.write_text(sample)
        c=scan_deck(p)['constants']
        assert c['E1']==46500 and c['G23']==3500 and abs(c['nu21']-.28*12500/46500)<1e-12
    print('B9A10N WINDOWS SELFTEST PASS | 9 constants | orientation | reciprocity')

def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['selftest','check','export'],required=True)
    a.add_argument('--project',default=None,help='ODBS root; defaults to script parent folder')
    args=a.parse_args();root=Path(args.project).resolve() if args.project else Path(__file__).resolve().parent.parent
    if args.stage=='selftest':return selftest()
    cases=all_jobs(root)
    print('B9A10N WINDOWS CHECK PASS | 24 input decks | 9 constants and material orientations match')
    if args.stage=='check':return
    out=root/'B9A10N_material_audit'
    if not out.is_dir():raise RuntimeError('Create this folder using Windows File Explorer first: '+str(out))
    dst=out/'B9A10N_windows_materials.json'
    if dst.exists():raise RuntimeError('Existing report protected: '+str(dst))
    summary={'step':'65AM-B9A10-N','cases':cases,'count':len(cases),
             'only_read_original_inputs':True,'no_new_abaqus_runs':True,
             'warning':'Material agreement is not proof of an exact finite-strain constitutive update.'}
    dst.write_text(json.dumps(summary,indent=2))
    print('B9A10N WINDOWS EXPORT DONE:',dst)
if __name__=='__main__':main()
