"""65W-E: conditional FEM-PATCH vs full-FEM S/LE integration-point comparisons.

Only compare at element centroids with SAME integration point and section point.
This is not a full hybrid comparison: NO fields are absent from FEM patch ODBs.
Material, units, fiber orientation, mesh and coordinate frames must match.
"""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'online_results_7region_65W_middle'/'E_mechanics_comparison'
FIELDS={'S':('S11','S22','S33','S12','S13','S23'),
        'LE':('LE11','LE22','LE33','LE12','LE13','LE23')}
ZONE={'left':(-9.6,-6.6),'center':(-1.8,1.8),'right':(6.6,9.6)}


def get(row,name):
    for k,v in row.items():
        if k.lower()==name.lower():return v
    raise ValueError('Missing mechanics CSV column '+name)


def load(file,region=None):
    data=[]
    with open(file,newline='') as f:
        for row in csv.DictReader(f):
            fld=str(get(row,'field')).upper()
            if fld not in FIELDS:continue
            try:
                xyz=np.array([float(get(row,k)) for k in ('centroid_x','centroid_y','centroid_z')])
                components=np.array([float(get(row,k)) if str(get(row,k)).strip() else np.nan for k in FIELDS[fld]])
            except ValueError:continue
            if region is not None:
                a,b=ZONE[region]
                if not (a-1e-7<=xyz[0]<=b+1e-7):continue
            ip=str(get(row,'integration_point'))
            sec=str(get(row,'section_point'))
            data.append(dict(pos=xyz,field=fld,ip=ip,sec=sec,values=components))
    return data


def metric(a,b):
    a=np.asarray(a);b=np.asarray(b)
    err=b-a
    return {'n':int(len(a)),'rmse':float(np.sqrt(np.mean(err**2))),
            'mae':float(np.mean(np.abs(err))),
            'rel_l2_percent':float(100*np.linalg.norm(err)/np.linalg.norm(a)) if np.linalg.norm(a)>1e-12 else None}


def match_key(r,tol):
    return (r['field'],r['ip'],r['sec'])+tuple(int(round(float(v)/tol)) for v in r['pos'])


def compare(a):
    if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('Existing mechanics output protected: '+str(OUT))
    OUT.mkdir(parents=True)
    if not a.material_confirmed or not a.mesh_orientation_confirmed:
        raise RuntimeError('Require --material-confirmed and --mesh-orientation-confirmed; geometry + integration section matching is essential')
    ref=load(a.full)
    rows=[];result={}
    for name,file in [('left',a.left),('center',a.center),('right',a.right)]:
        primary=load(file,name)
        reference=load(a.full,name)
        idx=defaultdict(list)
        for r in reference:idx[match_key(r,a.tol)].append(r)
        totals={};pairs=[];ambig=0;skipped=0
        for p in primary:
            k=match_key(p,a.tol)
            matches=[r for r in idx.get(k,[]) if np.linalg.norm(p['pos']-r['pos'])<=a.tol]
            if len(matches)!=1:
                if len(matches)>1:ambig+=1
                else:skipped+=1
                continue
            r=matches[0]
            for j,comp in enumerate(FIELDS[p['field']]):
                x,y=r['values'][j],p['values'][j]
                if not np.isfinite(x) or not np.isfinite(y):continue
                totals.setdefault(comp,[[],[]])[0].append(x)
                totals[comp][1].append(y)
                pairs.append((name,p['field'],comp,float(p['pos'][0]),float(p['pos'][1]),float(p['pos'][2]),
                              p['ip'],p['sec'],float(x),float(y)))
        for comp,(vals_ref,vals_hyb) in totals.items():
            z=metric(vals_ref,vals_hyb)
            rows.append(dict(region=name,component=comp,**z))
        result[name]={'patch_rows':len(primary),'reference_rows':len(reference),
                      'matched_component_values':len(pairs),'ambiguous':ambig,'unmatched':skipped,
                      'warning':'Only three FEM patches; stresses/strains in four NO regions absent.'}
        with open(OUT/('65W_E_pairs_'+name+'.csv'),'w',newline='') as f:
            w=csv.writer(f);w.writerow(['patch','field','component','x','y','z','ip','section_point','FEM','FEM_patch'])
            w.writerows(pairs)
    with open(OUT/'65W_E_FE_only_mechanics.csv','w',newline='') as f:
        cols=['region','component','n','rmse','mae','rel_l2_percent']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    out={'only_FE_patch_mechanics':True,'not_seven_region_mechanics':True,
         'same_material_asserted':a.material_confirmed,
         'same_mesh_and_orientation_asserted':a.mesh_orientation_confirmed,
         'match_tolerance_mm':a.tol,'regions':result,
         'caution':'Element centroid equality plus integration-point/section identifiers is a necessary screen, not proof of identical Gauss-point physical location. Compare actual integration-point global positions/element topology for publication-grade errors.'}
    with open(OUT/'65W_E_review.json','w') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


def selftest():
    r={'field':'S','ip':'1','sec':'1','pos':np.array([.1,.2,.3])}
    assert match_key(r,1e-4)==match_key(r,1e-4)
    assert metric([1,2],[1,2])['rmse']==0
    print('65W-E selftest PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['selftest','compare'],default='compare')
    for name in ('full','left','center','right'):p.add_argument('--'+name)
    p.add_argument('--tol',type=float,default=1e-4)
    p.add_argument('--material-confirmed',action='store_true')
    p.add_argument('--mesh-orientation-confirmed',action='store_true')
    a=p.parse_args()
    if a.stage=='selftest':selftest();return
    if any(getattr(a,n) is None for n in ('full','left','center','right')):p.error('Pass all four mechanics CSVs')
    if a.tol<=0 or a.tol>1e-2:p.error('0<--tol<=0.01')
    compare(a)

if __name__=='__main__':main()
