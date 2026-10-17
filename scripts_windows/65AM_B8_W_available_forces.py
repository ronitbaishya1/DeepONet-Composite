#!/usr/bin/env python
"""65AM-B8 Windows Abaqus 2025: inspect available force outputs from six B6 ODBs.

READ ONLY ODB inspection: does not regenerate jobs or modify past files.
This is not a force-balance calculation. It records whether NFORC, RF,
CF, and contact-traction fields were saved in the final Loading frame.

From ODBS:
 abaqus python scripts\\65AM_B8_W_available_forces.py --odb-dir B6_ODB --check-only
 abaqus python scripts\\65AM_B8_W_available_forces.py --odb-dir B6_ODB

Create ODBS\\B8_windows_audit via File Explorer for the second command.
"""
from __future__ import print_function
import os, argparse, json

CASE_IDS=['B6_new_%02d'%i for i in range(1,7)]
KEYS=['NFORC','RF','CF','CSTRESS','CPRESS','CSHEAR1','CSHEAR2','CNORMF','CSHEARF','CTF','U','S','LE']


def one_odb(path):
    from odbAccess import openOdb
    odb=openOdb(path=os.path.abspath(path),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Loading step absent: '+path)
        step=odb.steps['Loading']
        if not step.frames:raise RuntimeError('Loading has no frames: '+path)
        frame=step.frames[-1]
        fields=frame.fieldOutputs
        present=sorted(str(k) for k in fields.keys())
        counts={}
        for key in KEYS:
            if key in fields:
                try:counts[key]=len(fields[key].values)
                except Exception:counts[key]='present but could not count'
            else:counts[key]=0
        for required in ('U','S','LE'):
            if not counts[required]:raise RuntimeError('Required field missing '+required+': '+path)
        return dict(name=os.path.basename(path),last_frame=float(frame.frameValue),
                    fields_present=present,counts=counts,
                    saved_nodal_internal_force=bool(counts['NFORC']),
                    saved_reaction_force=bool(counts['RF']),
                    saved_nodal_concentrated_force=bool(counts['CF']),
                    saved_contact_related_fields=any(k.startswith('C') and 'STRESS' in k or k.startswith('CPRESS')
                        for k in present))
    finally:odb.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--odb-dir',default='B6_ODB')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    items=[]
    for case in CASE_IDS:
        path=os.path.join(a.odb_dir,case+'.odb')
        if not os.path.isfile(path):raise RuntimeError('Missing '+path)
        data=one_odb(path);data['case_id']=case;items.append(data)
        print('B8',case,'NFORC:',data['counts']['NFORC'],'RF:',data['counts']['RF'],
              'CF:',data['counts']['CF'],'CSTRESS:',data['counts']['CSTRESS'])
    if a.check_only:
        print('B8 WINDOWS CHECK PASS; inspected 6 ODBs; no files created')
        return
    outdir='B8_windows_audit'
    if not os.path.isdir(outdir):
        raise RuntimeError('Create '+outdir+' inside ODBS using File Explorer before running')
    out=os.path.join(outdir,'B8_force_field_availability.json')
    if os.path.exists(out):raise RuntimeError('Existing audit protected: '+out)
    summary={'step':'65AM-B8-W', 'cases':items,
             'force_equilibrium_certified':False,
             'caveat':'Field availability alone does not establish FE-NO force balance or contact/reaction equilibrium.'}
    with open(out,'w') as f:json.dump(summary,f,indent=2)
    print('Saved:',os.path.abspath(out))

if __name__=='__main__':main()
