#!/usr/bin/env python
"""65AM B9A10-B: READ-ONLY Abaqus ODB energy-output availability audit.

From C:\\Users\\rrb225_lehigh\\Pictures\\ODBS (Abaqus Command):
  abaqus python scripts\\65AM_B9A10_B_W_energy_check.py --selftest
  abaqus python scripts\\65AM_B9A10_B_W_energy_check.py --stage check
  abaqus python scripts\\65AM_B9A10_B_W_energy_check.py --stage export

The original B9A3 inputs explicitly requested U/RF/S/LE. If ALLAE, ALLSE,
ALLIE were not requested/recorded, this tool CANNOT infer an hourglass energy.
It does not rerun any Abaqus job or modify any ODB, model, or solver.
"""
from __future__ import print_function
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'B9A10B_windows_audit'
CHECK=(('B9A3_ODB','B9A3_inner_left'),('B9A3_ODB','B9A3_inner_right'),
       ('B9A6_ODB','B9A6_inner_left_FEM'),('B9A6_ODB','B9A6_inner_left_B5'),
       ('B9A6_ODB','B9A6_inner_right_FEM'),('B9A6_ODB','B9A6_inner_right_B5'))
KEYS=('ALLAE','ALLSE','ALLIE','ALLWK','ETOTAL','ALLPD','ALLFD','ALLVD')

def last_value(obj):
    data=obj.data
    return float(data[-1][1]) if data else None

def audit_odb(path):
    from odbAccess import openOdb
    odb=openOdb(path=str(path.resolve()),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError('Missing Loading step: '+str(path))
        step=odb.steps['Loading']
        if len(step.frames)<1:raise RuntimeError('No ODB frames: '+str(path))
        final=step.frames[-1]
        fields=sorted([str(k) for k in final.fieldOutputs.keys()])
        history_matches={k:[] for k in KEYS}
        for region_name,history_region in step.historyRegions.items():
            for key in KEYS:
                if key in history_region.historyOutputs:
                    v=last_value(history_region.historyOutputs[key])
                    history_matches[key].append({'region':str(region_name),'value':v})
        def select_energy(name):
            matches=history_matches[name]
            if not matches:return {'status':'not_saved','value':None}
            if len(matches)==1:return {'status':'available','value':matches[0]['value'],'history_region':matches[0]['region']}
            assemblies=[m for m in matches if ('ASSEMBLY' in m['region'].upper() or 'WHOLE MODEL' in m['region'].upper())]
            if len(assemblies)==1:return {'status':'available','value':assemblies[0]['value'],'history_region':assemblies[0]['region']}
            return {'status':'ambiguous_multiple_regions','value':None,'matches':matches}
        energy={k:select_energy(k) for k in KEYS}
        ae=energy['ALLAE']['value'];ie=energy['ALLIE']['value'];se=energy['ALLSE']['value']
        ratio=(100.*abs(ae)/max(abs(ie),1e-12) if ae is not None and ie is not None else None)
        return {'file':str(path),'step':'Loading','last_frame_time':float(final.frameValue),
                'nodal_U_RF_available':('U' in fields and 'RF' in fields),
                'S_LE_available':('S' in fields and 'LE' in fields),
                'field_outputs':fields,'history_region_count':len(step.historyRegions),
                'available_history_keys':sorted(set(str(key) for reg in step.historyRegions.values() for key in reg.historyOutputs.keys())),
                'energy':energy,'ALLAE_over_ALLIE_pct_if_available':ratio,
                'ALLAE_over_ALLIE_definition':'100*abs(ALLAE)/abs(ALLIE) using final-frame assembly history, if unambiguous',
                'no_hourglass_attribution':True}
    finally:odb.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('check','export'),default='check')
    p.add_argument('--selftest',action='store_true')
    args=p.parse_args()
    if args.selftest:
        assert len(CHECK)==6 and len(set(x[1] for x in CHECK))==6
        print('B9A10-B WINDOWS SELFTEST PASS | expected six ODB names')
        return
    missing=[]
    for folder,job in CHECK:
        f=ROOT/folder/(job+'.odb')
        if not f.is_file():missing.append(str(f))
        else:print('FOUND:',f)
    if missing:raise RuntimeError('Missing prior ODB files: '+'; '.join(missing))
    if args.stage=='check':
        print('B9A10-B WINDOWS CHECK PASS | 6 ODB files present | no changes')
        return
    if not OUT.is_dir():
        raise RuntimeError('In Windows File Explorer create: '+str(OUT))
    target=OUT/'B9A10B_energy_availability.json'
    if target.exists():raise RuntimeError('Existing result protected: '+str(target))
    out=[]
    for folder,job in CHECK:
        file=ROOT/folder/(job+'.odb')
        values=audit_odb(file)
        values['job']=job
        values['replay_control_not_same_as_free_solution']=job.startswith('B9A6_')
        out.append(values)
        print(job,'ALLAE:',values['energy']['ALLAE']['status'],'ALLIE:',values['energy']['ALLIE']['status'])
    final={'step':'65AM-B9A10-B','purpose':'Energy availability ONLY from existing read-only ODBs',
           'cases':out,'ALLAE_available_count':sum(c['energy']['ALLAE']['status']=='available' for c in out),
           'energy_warnings':[
             'Missing ALLAE/ALLIE means energy was not stored; zero should not be assumed',
             'Nonzero ALLAE is not a measured nodal hourglass force contribution',
             'B9A6 replays prescribe all nodal displacements and are not free equilibrium solves',
             'No simulations changed, and no physics loss or solver integration approved']}
    target.write_text(json.dumps(final,indent=2))
    print('B9A10-B WINDOWS EXPORT DONE:',target)

if __name__=='__main__':main()
