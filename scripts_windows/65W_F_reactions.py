r"""65W-F: extract actual Abaqus reaction on an EXPLICIT known node set.

Use --stage inventory to list assembly node sets. Then run TWO extractions
(full-beam nose node set and hybrid center-patch nose node set), specifying
correct names. Do not guess a node set or sum arbitrary RF values.

 abaqus python scripts\65W_F_reactions.py --stage inventory --odb PATH\full.odb
 abaqus python scripts\65W_F_reactions.py --stage extract --odb PATH\full.odb --tag full --set NAME
 abaqus python scripts\65W_F_reactions.py --stage extract --odb PATH\center.odb --tag hybrid --set NAME
 python scripts\65W_F_reactions.py --stage compare --reference PATH\full.json --hybrid PATH\hybrid.json
"""
import argparse,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'online_results_7region_65W_middle'/'F_reaction_sets'


def save(path,value):
    if path.exists():raise RuntimeError('Refuse overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f:json.dump(value,f,indent=2)


def extract(args):
    from odbAccess import openOdb
    odb=openOdb(path=str(Path(args.odb).resolve()),readOnly=True)
    try:
        if args.stage=='inventory':
            print('Assembly node sets:',list(odb.rootAssembly.nodeSets.keys()))
            for name,inst in odb.rootAssembly.instances.items():
                print('Instance:',name,'instance node sets:',list(inst.nodeSets.keys()))
            return
        steps=list(odb.steps.keys())
        if not steps:raise RuntimeError('ODB has no steps')
        frame=odb.steps[args.step or steps[-1]].frames[args.frame]
        if 'RF' not in frame.fieldOutputs:raise RuntimeError('ODB lacks nodal RF field output')
        sets=odb.rootAssembly.nodeSets
        if args.set not in sets:raise RuntimeError('Assembly node set not found: '+str(args.set))
        vals=frame.fieldOutputs['RF'].getSubset(region=sets[args.set]).values
        if not vals:raise RuntimeError('Selected node set has no RF values; check RP and output requests')
        components=[0.,0.,0.]
        for v in vals:
            for i,x in enumerate(v.data[:3]):components[i]+=float(x)
        result={'ODB':str(Path(args.odb).resolve()),'assembly_node_set':args.set,
                'node_values':len(vals),'RF1_N':components[0],'RF2_N':components[1],
                'RF3_N':components[2],'frame_value':float(frame.frameValue),
                'note':'User identified a nose reaction node set; confirm same sign conventions and load definition before comparing.'}
        path=OUT/('65W_F_'+args.tag+'.json')
        save(path,result)
        print(json.dumps(result,indent=2))
    finally:odb.close()


def compare(args):
    with open(args.reference) as f:r=json.load(f)
    with open(args.hybrid) as f:h=json.load(f)
    a=float(r['RF2_N']);b=float(h['RF2_N'])
    diff=abs(a-b)
    out={'full_FEM_RF2_N':a,'hybrid_center_RF2_N':b,'absolute_difference_N':diff,
         'percent_difference':100*diff/abs(a) if abs(a)>1e-9 else None,
         'CAUTION':'Check whether hybrid reaction represents the SAME nose load and displacement before treating it as a physical comparison.'}
    save(OUT/'65W_F_reaction_comparison.json',out)
    print(json.dumps(out,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=('inventory','extract','compare'))
    p.add_argument('--odb');p.add_argument('--set');p.add_argument('--tag',default='full')
    p.add_argument('--step');p.add_argument('--frame',type=int,default=-1)
    p.add_argument('--reference');p.add_argument('--hybrid')
    args=p.parse_args()
    if args.stage=='compare':
        if not args.reference or not args.hybrid:p.error('Pass --reference and --hybrid')
        compare(args)
    else:
        if not args.odb:p.error('Pass --odb')
        if args.stage=='extract' and not args.set:p.error('Pass --set, from inventory')
        extract(args)

if __name__=='__main__':main()
