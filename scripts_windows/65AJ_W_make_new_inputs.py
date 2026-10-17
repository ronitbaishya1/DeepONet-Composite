#!/usr/bin/env python3
"""Create six full-model Abaqus .inp files from the verified 3Point.inp.

RUN FROM WINDOWS CMD (inside ODBS):
    abaqus python scripts\65AJ_W_make_new_inputs.py --base-inp 3Point.inp --manifest 65AJ_D_new_cases.csv --out-dir new_ODB

Changes ONLY the composite E1, E2, G12 in the existing GlassEpoxy
*ELASTIC, TYPE=ENGINEERING CONSTANTS block. All other source lines are
preserved exactly. Does not modify the original input or submit jobs.
"""
from __future__ import print_function
import argparse
import csv
import hashlib
import math
import os
import re


def _keyword(line):
    return line.lstrip().startswith('*') and not line.lstrip().startswith('**')


def _keyword_name_and_args(line):
    return line.strip().lower().replace(' ', '')


def _material_is_glass_epoxy(line):
    if not _keyword(line):
        return False
    v=_keyword_name_and_args(line)
    return v.startswith('*material,') and bool(re.search(r'(?:^|,)name=glassepoxy(?:,|$)',v))


def locate_engineering_constants(lines):
    hits=[]
    for k,line in enumerate(lines):
        if not _material_is_glass_epoxy(line):
            continue
        block=[]
        j=k+1
        while j<len(lines) and not (_keyword(lines[j]) and lines[j].strip().lower().startswith('*material')):
            if _keyword(lines[j]) and _keyword_name_and_args(lines[j]).startswith('*elastic,'):
                key=_keyword_name_and_args(lines[j])
                if 'type=engineeringconstants' not in key:
                    raise RuntimeError('GlassEpoxy elastic block is not ENGINEERING CONSTANTS: '+lines[j].strip())
                if block:
                    raise RuntimeError('Several GlassEpoxy engineering elastic blocks found; cannot edit safely')
                p=j+1
                rows=[]
                while p<len(lines) and not _keyword(lines[p]):
                    if lines[p].strip() and not lines[p].lstrip().startswith('**'):
                        rows.append(p)
                    p+=1
                block=rows
            j+=1
        if not block:
            raise RuntimeError('GlassEpoxy material found but no engineering elastic rows')
        hits.append(block)
    if len(hits)!=1:
        raise RuntimeError('Expected exactly one GlassEpoxy material block, got '+str(len(hits)))
    indices=hits[0]
    if len(indices)!=2:
        raise RuntimeError('Expected 2 engineering constants data lines (8+1 constants); got '+str(len(indices)))
    row_values=[]
    for idx in indices:
        parts=[a.strip() for a in lines[idx].split(',') if a.strip()]
        try:row_values.append([float(a) for a in parts])
        except ValueError:raise RuntimeError('Invalid nonnumeric engineering constants line '+str(idx+1))
    if [len(x) for x in row_values]!=[8,1]:
        raise RuntimeError('Template expected 8+1 engineering constants, found '+str([len(x) for x in row_values]))
    return indices,row_values[0]+row_values[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-inp',default='3Point.inp')
    p.add_argument('--manifest',default='65AJ_D_new_cases.csv')
    p.add_argument('--out-dir',default='new_ODB')
    args=p.parse_args()
    if not os.path.isfile(args.base_inp):raise RuntimeError('Missing original model: '+os.path.abspath(args.base_inp))
    if not os.path.isfile(args.manifest):raise RuntimeError('Missing material CSV: '+os.path.abspath(args.manifest))
    if not os.path.isdir(args.out_dir):raise RuntimeError('Create the new_ODB folder manually before running')
    raw=open(args.base_inp,'rb').read()
    try:source=raw.decode('utf-8')
    except UnicodeDecodeError:source=raw.decode('latin-1')
    lines=source.splitlines(keepends=True)
    positions,base=locate_engineering_constants(lines)
    if not any(re.match(r'^\s*\*Step\s*,\s*name\s*=\s*Loading\b',v,re.I) for v in lines):
        raise RuntimeError('Validated Loading step missing from template')
    if not any('LE' in v and 'S' in v and v.lstrip()[:1]!='*' for v in lines[-100:]):
        raise RuntimeError('Check original model field output request contains LE and S')
    with open(args.manifest,'r',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)<3:raise RuntimeError('Expected at least 3 new cases in CSV')
    if not {'case_id','E1_MPa','E2_MPa','G12_MPa'}.issubset(rows[0]):
        raise RuntimeError('CSV must contain case_id,E1_MPa,E2_MPa,G12_MPa')
    all_cases=[]
    for row in rows:
        case=row['case_id'].strip()
        if not re.fullmatch(r'AJ_new_\d{2}',case):
            raise RuntimeError('Unexpected case name: '+case)
        try:e1,e2,g12=[float(row[x]) for x in ('E1_MPa','E2_MPa','G12_MPa')]
        except (ValueError,KeyError):raise RuntimeError('Invalid material numbers in '+case)
        if not all(math.isfinite(x) and x>0 for x in [e1,e2,g12]):raise RuntimeError('Nonpositive/unfinite elastic constants '+case)
        all_cases.append((case,e1,e2,g12))
    if len(set(x[0] for x in all_cases))!=len(all_cases):raise RuntimeError('Duplicate case ids')
    paths=[os.path.join(args.out_dir,case+'.inp') for case,_,_,_ in all_cases]
    if any(os.path.exists(x) for x in paths):
        raise RuntimeError('One or more new input files already exist; refusing overwrite')
    print('Verified base GlassEpoxy values [E1,E2,E3,nu12,nu13,nu23,G12,G13,G23]:')
    print('  '+str(base))
    print('The NEW input files will preserve E3, nu12, nu13, nu23, G13 and G23 from the original.')
    for (case,e1,e2,g12),path in zip(all_cases,paths):
        adjusted=list(base)
        adjusted[0]=e1;adjusted[1]=e2;adjusted[6]=g12
        new=list(lines)
        new[positions[0]]=(', '.join(format(v,'.12g') for v in adjusted[:8])+ '\n')
        new[positions[1]]=(format(adjusted[8],'.12g')+',\n')
        # Assert every other line remains identical.
        for j,(a,b) in enumerate(zip(lines,new)):
            if j not in positions and a!=b:raise RuntimeError('Unexpected non-material modification')
        with open(path,'wb') as f:f.write(''.join(new).encode('latin-1' if raw.decode('latin-1')==source else 'utf-8'))
        print('CREATED %s: E1=%.7f E2=%.7f G12=%.7f MPa'%(path,e1,e2,g12))
    print('SUCCESS: %d new ABAQUS INPUTS written. NO ODBs yet. Run Abaqus jobs next.'%len(all_cases))
    print('Source template SHA256:',hashlib.sha256(raw).hexdigest())


if __name__=='__main__':main()
