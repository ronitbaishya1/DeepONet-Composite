"""65W-B: export actual Abaqus ODB fields for full-FEM or 65W-A patches.

Requires Abaqus Python with odbAccess. Never re-runs an FEM analysis.
From ODBS root:
  abaqus python scripts\65W_B_export_odb.py --stage inventory --odb "C:\\...\\full_middle.odb"
  abaqus python scripts\65W_B_export_odb.py --stage export --odb "C:\\...\\full_middle.odb" --tag full --instances BEAM-1
  abaqus python scripts\65W_B_export_odb.py --stage export --odb "C:\\...\\left.odb" --tag patch_left --instances BEAM-1
Use inventory to determine beam instance spelling.

No false approximation: S/LE are raw integration-point values (not nodal),
coordinate output for mechanics is ELEMENT CENTROID not true Gauss-point coordinate.
A mesh/section-matched field comparator is needed before claiming stress accuracy.
"""
from __future__ import print_function
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT,'online_results_7region_65W_middle','B_odb_exports')
COMPONENTS=('S11','S22','S33','S12','S13','S23',
            'LE11','LE22','LE33','LE12','LE13','LE23')


def mkdir(p):
    if not os.path.isdir(p):os.makedirs(p)


def dump(path,obj):
    with open(path,'w') as f:json.dump(obj,f,indent=2)


def xyzn(node):
    a=list(node.coordinates)
    return [float(a[i]) if i<len(a) else 0. for i in range(3)]


def to3(seq):
    a=list(seq)
    return [float(a[i]) if i<len(a) else 0. for i in range(3)]


def inventory(odb):
    step_names=list(odb.steps.keys())
    result={'steps':step_names, 'instances':{},'assembly_node_sets':list(odb.rootAssembly.nodeSets.keys()),
            'warning':'Choose only the beam/specimen instances in each ODB, not rollers or nose rigid bodies.'}
    for name,ins in odb.rootAssembly.instances.items():
        result['instances'][name]={'nodes':len(ins.nodes),'elements':len(ins.elements),
                                   'node_sets':list(ins.nodeSets.keys()),'element_sets':list(ins.elementSets.keys())}
    if step_names:
        step=odb.steps[step_names[-1]]
        result['last_step_frames']=len(step.frames)
        if step.frames:result['last_frame_field_names']=list(step.frames[-1].fieldOutputs.keys())
    return result


def export(odb,args):
    steps=list(odb.steps.keys())
    if not steps:raise RuntimeError('ODB has no analysis steps')
    stepname=args.step or steps[-1]
    if stepname not in odb.steps:raise RuntimeError('Step %s not found in ODB' % stepname)
    frames=odb.steps[stepname].frames
    if not frames:raise RuntimeError('No frames for selected analysis step')
    frame=frames[args.frame]
    available=list(frame.fieldOutputs.keys())
    if 'U' not in available:raise RuntimeError('No nodal displacement U in final ODB frame')
    names=[v.strip() for v in args.instances.split(',') if v.strip()] if args.instances else list(odb.rootAssembly.instances.keys())
    unknown=[n for n in names if n not in odb.rootAssembly.instances]
    if unknown:raise RuntimeError('Unknown instances %s. Run --stage inventory first' % unknown)
    folder=os.path.join(OUT,args.tag)
    if os.path.exists(folder) and os.listdir(folder):
        raise RuntimeError('Non-empty export directory protected: '+folder)
    mkdir(folder)
    coords={}; elem_center={}; instances=set(names)
    for name in names:
        ins=odb.rootAssembly.instances[name]
        loc={}
        for node in ins.nodes:
            c=xyzn(node)
            coords[(name,node.label)]=c
            loc[node.label]=c
        for elem in ins.elements:
            nodes=[loc[k] for k in elem.connectivity if k in loc]
            if nodes:
                n=len(nodes)
                elem_center[(name,elem.label)]=[sum(a[i] for a in nodes)/n for i in range(3)]
    nodal={}
    for key in ('U','RF'):
        if key not in available:continue
        for value in frame.fieldOutputs[key].values:
            name=value.instance.name if value.instance else ''
            if name not in instances or (name,value.nodeLabel) not in coords:continue
            record=nodal.setdefault((name,value.nodeLabel),{'U1':'','U2':'','U3':'',
                                                            'RF1':'','RF2':'','RF3':''})
            data=to3(value.data)
            for j in range(3):record['%s%d'%(key,j+1)]=data[j]
    nodal_path=os.path.join(folder,args.tag+'_nodes.csv')
    with open(nodal_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['instance','node_label','x','y','z','U1','U2','U3','RF1','RF2','RF3'])
        for name,label in sorted(nodal):
            row=nodal[(name,label)]
            w.writerow([name,label]+coords[(name,label)]+[row[k] for k in ('U1','U2','U3','RF1','RF2','RF3')])
    mech_path=os.path.join(folder,args.tag+'_mechanics.csv')
    counts={}
    with open(mech_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['instance','element_label','integration_point','section_point',
                    'centroid_x','centroid_y','centroid_z','field']+list(COMPONENTS))
        for field in ('S','LE'):
            if field not in available:continue
            raw=frame.fieldOutputs[field]
            labels=list(raw.componentLabels)
            for value in raw.values:
                name=value.instance.name if value.instance else ''
                key=(name,value.elementLabel)
                if name not in instances or key not in elem_center:continue
                coords3=elem_center[key]
                ip=getattr(value,'integrationPoint','')
                section=getattr(value,'sectionPoint',None)
                sec=(getattr(section,'number','') if section is not None else '')
                data=value.data
                cols={str(k).upper():float(v) for k,v in zip(labels,data)}
                w.writerow([name,value.elementLabel,ip,sec]+coords3+[field]+[cols.get(k,'') for k in COMPONENTS])
                counts[field]=counts.get(field,0)+1
    manifest={
        'role':args.tag, 'source_odb':os.path.abspath(args.odb),'step':stepname,
        'frame_index':args.frame,'frame_value':float(frame.frameValue),
        'instances':names,'fields_present':available,
        'node_rows':len(nodal),'mechanics_rows_per_field':counts,
        'node_csv':nodal_path,'mechanics_csv':mech_path,
        'region_bounds_mm':{'FE_L':[-9.6,-6.6],'FE_C':[-1.8,1.8],'FE_R':[6.6,9.6],
                            'NO_OL':[-12,-9.6],'NO_L':[-6.6,-1.8],
                            'NO_R':[1.8,6.6],'NO_OR':[9.6,12]},
        'IMPORTANT':'Exported fields are from FEM ODB instances only. Three patch ODBs do NOT contain the four neural-operator regions.',
        'mechanics_coordinate_warning':'centroid_xyz is ELEMENT CENTROID, NOT Gauss-point coordinate. No automatic nodal stress interpolation or layer averaging.',
        'node_coordinate_warning':'Uses instance node coordinates; validate global frame/orientation for translated or rotated instances.',
        'reaction_warning':'Nodal RF values are exported but nose/support force must be summed over appropriate named node sets. Do not sum all nodes as a nose reaction.'
    }
    dump(os.path.join(folder,args.tag+'_manifest.json'),manifest)
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['inventory','export'],required=True)
    parser.add_argument('--odb',required=True)
    parser.add_argument('--tag',default='full')
    parser.add_argument('--instances',help='Comma-separated beam instance names from inventory')
    parser.add_argument('--step',help='ODB step name; defaults last')
    parser.add_argument('--frame',type=int,default=-1)
    args=parser.parse_args()
    if not os.path.isfile(args.odb):raise FileNotFoundError(args.odb)
    try:
        from odbAccess import openOdb
    except ImportError:
        raise RuntimeError('This script MUST run using Abaqus Python with odbAccess')
    odb=openOdb(path=os.path.abspath(args.odb),readOnly=True)
    try:
        if args.stage=='inventory':print(json.dumps(inventory(odb),indent=2))
        else:export(odb,args)
    finally:odb.close()

if __name__=='__main__':main()
