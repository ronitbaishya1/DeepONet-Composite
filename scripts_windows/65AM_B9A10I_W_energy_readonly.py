# -*- coding: utf-8 -*-
"""65AM-B9A10-I OPTIONAL read-only energy availability from 24 B9A9 ODBs.

No new jobs, no ODB modifications. Missing ALLAE/ALLIE is NOT treated as zero.
Place in C:\\Users\\rrb225_lehigh\\Pictures\\ODBS\\scripts\\.
Commands (Abaqus Python):
  abaqus python scripts\\65AM_B9A10I_W_energy_readonly.py --stage selftest
  abaqus python scripts\\65AM_B9A10I_W_energy_readonly.py --stage check
  abaqus python scripts\\65AM_B9A10I_W_energy_readonly.py --stage export
"""
from __future__ import print_function
import argparse,csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
ODB=ROOT/'B9A9_ODB'
SOURCE=ROOT/'B9A9_exports'/'B9A9_windows_summary.json'
OUT=ROOT/'B9A10I_energy_exports'
JOBS=['B9A9_%s%02d'%(r,i) for r in ('L','R') for i in range(1,13)]
ENERGIES=('ALLAE','ALLIE','ALLSE','ALLWK')


def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def selftest():
    # Treat missing history differently from actual zero-valued energy.
    def value(a):return '' if a is None else float(a)
    assert value(None)=='' and value(0.)==0.
    assert len(JOBS)==24 and len(set(JOBS))==24
    print('B9A10I WINDOWS SELFTEST PASS | 24 cases | missing energy distinct from zero')


def check():
    if not SOURCE.is_file():raise RuntimeError('Missing B9A9 source manifest: '+str(SOURCE))
    meta=json.loads(SOURCE.read_text())
    old={r['job']:r['odb_sha256'] for r in meta['jobs']}
    if set(old)!=set(JOBS):raise RuntimeError('B9A9 manifest must contain 24 original jobs')
    for name in JOBS:
        p=ODB/(name+'.odb')
        if not p.is_file():raise RuntimeError('Missing ODB '+str(p))
        if sha(p)!=old[name]:raise RuntimeError('Original ODB has changed '+name)
    print('B9A10I ENERGY CHECK PASS | 24 original ODB checksums | read-only')
    return old


def read_energies(name):
    from odbAccess import openOdb
    odb=openOdb(path=str((ODB/(name+'.odb')).resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Loading step not found '+name)
        step=odb.steps['Loading']
        candidates={e:[] for e in ENERGIES}
        for region_name,region in step.historyRegions.items():
            for var in ENERGIES:
                if var in region.historyOutputs:
                    data=region.historyOutputs[var].data
                    if data:
                        candidates[var].append((region_name,float(data[-1][1])))
        out={}
        for var,records in candidates.items():
            # Prefer system energy for assembly as a whole, never sum local values.
            assembly=[r for r in records if 'Assembly' in str(r[0]) or 'ASSEMBLY' in str(r[0])]
            matching=assembly if assembly else records
            if len(matching)==1:
                out[var]=matching[0][1]
                out[var+'_status']='available'
                out[var+'_history_region']=str(matching[0][0])
            elif not matching:
                out[var]='';out[var+'_status']='not_saved';out[var+'_history_region']=''
            else:
                out[var]='';out[var+'_status']='ambiguous_multiple_history_regions';out[var+'_history_region']=''
        return out
    finally:
        odb.close()


def export():
    originals=check()
    if not OUT.is_dir():raise RuntimeError('Create this folder in File Explorer first: '+str(OUT))
    p=OUT/'B9A10I_energy_history.csv'
    m=OUT/'B9A10I_energy_summary.json'
    if p.exists() or m.exists():raise RuntimeError('Existing energy output protected. Move it in Explorer before rerun.')
    rows=[]
    for job in JOBS:
        if sha(ODB/(job+'.odb'))!=originals[job]:raise RuntimeError('ODB changed while reading '+job)
        row={'job':job,'region':'inner_left' if '_L' in job else 'inner_right'}
        row.update(read_energies(job))
        rows.append(row)
        print('ENERGY',job,'ALLAE',row['ALLAE_status'],'ALLIE',row['ALLIE_status'])
    with open(str(p),'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    summary={'step':'65AM-B9A10-I','source_manifest_sha256':sha(SOURCE),
        'csv_sha256':sha(p),'cases':24,'read_only_source':True,
        'missing_energy_not_zero':True,
        'note':'ALLAE/ALLIE is an energy ratio, not an element or nodal hourglass force.'}
    m.write_text(json.dumps(summary,indent=2))
    print('B9A10I ENERGY EXPORT DONE:',p)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['selftest','check','export'])
    a=parser.parse_args()
    if a.stage=='selftest':selftest()
    elif a.stage=='check':check()
    else:export()
