#!/usr/bin/env python3
"""B9A11J: read-only, full-FEM HEALTHY-vs-DAMAGED detectability diagnostic.

Requires outputs from unchanged GitHub scripts 65W_B_export_odb.py,
65W_F_reactions.py. Does NOT fit a damage model or claim damage localization.
All U/S/LE are compared only after one-to-one labels, coordinates, final
frame, step and instance match. Reaction-force comparison is optional, but
can only use an explicit validated NOSE assembly node set.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

TAGS = ('healthy', 'I_DMG_X02_D02', 'I_DMG_X03_D02', 'I_DMG_X02_D01')
TRUTH = {'I_DMG_X02_D02':(-4.2,.35), 'I_DMG_X03_D02':(4.2,.35),
         'I_DMG_X02_D01':(-4.2,.15)}
SCORES = (5,10,20,50)
COMP = ('11','22','33','12','13','23')
N_EXPECTED = 4592
E_EXPECTED = 3640


def finite(a, name):
    z = np.asarray(a,dtype=float)
    if not np.isfinite(z).all(): raise RuntimeError('Non-finite values: '+name)
    return z


def relpct(a,b):
    return float(100*np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1.e-12))


def summary_delta(delta,healthy):
    d=finite(delta,'change');h=finite(healthy,'healthy')
    return {'relative_to_healthy_norm_pct':relpct(d+h,h),
            'rms_change':float(np.sqrt(np.mean(d*d))),
            'max_abs_change':float(np.max(np.abs(d))),
            'mean_abs_change':float(np.mean(np.abs(d)))}


def load_one(folder,tag):
    d=folder/tag
    sources={
        'nodes':d/(tag+'_nodes.csv'),
        'mechanics':d/(tag+'_mechanics.csv'),
        'manifest':d/(tag+'_manifest.json')}
    for key,p in sources.items():
        if not p.is_file():raise FileNotFoundError('Missing '+key+': '+str(p))
    m=json.loads(sources['manifest'].read_text())
    if str(m.get('step','')).lower()!='loading':raise RuntimeError(tag+': expected Loading step')
    if not np.isclose(float(m['frame_value']),1.,atol=2.e-4,rtol=0):
        raise RuntimeError(tag+': final frame is not 1.0; inspect job convergence')
    if m.get('instances')!=['COMPOSITE-1']:
        raise RuntimeError(tag+': only COMPOSITE-1 is supported; got '+str(m.get('instances')))
    if tag=='healthy' and any(s in Path(str(m.get('source_odb',''))).name.lower() for s in ('65w','fullmiddle')):
        raise RuntimeError('65W middle-material ODB cannot be the healthy base (45000/12000/4500 MPa).')
    raw=pd.read_csv(sources['nodes'])
    need={'instance','node_label','x','y','z','U1','U2','U3'}
    if not need.issubset(raw):raise RuntimeError(tag+': node CSV missing '+str(need-set(raw)))
    if len(raw)!=N_EXPECTED or raw.node_label.duplicated().any():
        raise RuntimeError(tag+': expected exactly 4592 unique specimen nodes')
    if set(raw.instance.astype(str))!={'COMPOSITE-1'}:
        raise RuntimeError(tag+': extra or wrong nodal instance')
    raw=raw.sort_values('node_label').reset_index(drop=True)
    X=finite(raw[['x','y','z']].to_numpy(),tag+' node X')
    U=finite(raw[['U1','U2','U3']].to_numpy(),tag+' U')
    mech=pd.read_csv(sources['mechanics'])
    req={'instance','element_label','integration_point','section_point',
         'centroid_x','centroid_y','centroid_z','field'}
    if not req.issubset(mech):raise RuntimeError(tag+': mechanics missing '+str(req-set(mech)))
    fields={}
    for kind in ('S','LE'):
        sub=mech.loc[mech.field==kind].copy()
        cols=[kind+c for c in COMP]
        if len(sub)!=E_EXPECTED or not set(cols).issubset(sub):
            raise RuntimeError(tag+': expected 3640 IP records for '+kind)
        sub['section_point']=sub['section_point'].fillna(-1).astype(str)
        sub['integration_point']=sub['integration_point'].fillna(-1).astype(str)
        if set(sub.instance.astype(str))!={'COMPOSITE-1'}:
            raise RuntimeError(tag+': wrong mechanics instance')
        sort_cols=['instance','element_label','integration_point','section_point']
        if sub.duplicated(sort_cols).any():raise RuntimeError(tag+': duplicate '+kind+' IP record')
        sub=sub.sort_values(sort_cols).reset_index(drop=True)
        fields[kind]={'keys':sub[sort_cols].astype(str).to_numpy(),
             'coords':finite(sub[['centroid_x','centroid_y','centroid_z']].to_numpy(),tag+kind+' centroid'),
             'values':finite(sub[cols].to_numpy(),tag+kind)}
    return {'metadata':m,'labels':raw.node_label.to_numpy(), 'X':X,'U':U, 'fields':fields}


def verify_match(base,other,tag):
    if not np.array_equal(base['labels'],other['labels']):raise RuntimeError(tag+': node labels changed')
    if float(np.max(np.abs(base['X']-other['X'])))>1e-5:
        raise RuntimeError(tag+': nodal coordinates changed; cannot subtract different meshes')
    for kind in ('S','LE'):
        a=base['fields'][kind];b=other['fields'][kind]
        if not np.array_equal(a['keys'],b['keys']):raise RuntimeError(tag+': '+kind+' element/IP keys changed')
        if np.max(np.abs(a['coords']-b['coords']))>1.e-4:
            raise RuntimeError(tag+': '+kind+' IP element centroids changed')
        # Assumes unchanged section orientation / output-component convention.


def sensor_indices(root, base, tag):
    p=root/'results'/'65AM_B9A11H_v2'/tag/'C_sensitivity'/'B9A11H_C_sensor_rankings.csv'
    if not p.is_file():return {},'No healthy-material sensor ranking found: '+str(p)
    data=np.load(root/'data'/'coordinates.npy',allow_pickle=False)
    if data.shape!=(N_EXPECTED,3):raise RuntimeError('Healthy ranking dataset geometry is not 4592x3')
    ranks=pd.read_csv(p).sort_values('Rank')
    if len(ranks)<50 or not {'NodeIndex','X','Y','Z'}.issubset(ranks):
        raise RuntimeError('Unexpected healthy sensor ranking format')
    out=[];used=set()
    for _,record in ranks.iloc[:50].iterrows():
        k=int(record['NodeIndex']);test=np.array([record['X'],record['Y'],record['Z']])
        if k<0 or k>=len(data) or np.max(np.abs(data[k]-test))>1e-5:
            raise RuntimeError('Healthy sensor ranking conflicts with dataset coordinates')
        diff=np.linalg.norm(base['X']-test[None,:],axis=1)
        i=int(np.argmin(diff))
        if diff[i]>1e-5 or i in used:raise RuntimeError('Ambiguous or missing sensor in FEM node table')
        out.append(i);used.add(i)
    return {k:np.asarray(out[:k],int) for k in SCORES},None


def compare_reactions(folder):
    p=folder/'reactions'
    cases={}
    for tag in TAGS:
        f=p/('65W_F_'+tag+'.json')
        if not f.is_file():return None,'Nose RF files incomplete; compare displacement/mechanics only. Missing '+str(f)
        cases[tag]=json.loads(f.read_text())
    common={(v.get('assembly_node_set'),v.get('node_values')) for v in cases.values()}
    if len(common)!=1:raise RuntimeError('Nose reaction set labels/counts mismatch')
    for tag,v in cases.items():
        if not np.isclose(v.get('frame_value',float('nan')),1.,atol=2e-4):
            raise RuntimeError(tag+': nose RF is not from full final frame')
    baseline=float(cases['healthy']['RF2_N'])
    if abs(baseline)<1e-6:raise RuntimeError('Healthy nose RF2 nearly zero; check assembly node set')
    return {tag:{'RF2_N':float(v['RF2_N']),
                 'delta_from_healthy_N':float(v['RF2_N']-baseline),
                 'relative_change_pct':float(100*(v['RF2_N']-baseline)/abs(baseline))}
            for tag,v in cases.items()},None


def sensor_stats(delta, indices, sigma):
    a=finite(delta[indices],'sensor response')
    # Healthy and damaged independent physical measurement noise:
    pair_sigma=math.sqrt(2.)*sigma
    return {'nodes':len(indices),'components':int(a.size),
            'rms_delta_mm':float(np.sqrt(np.mean(a*a))),
            'rms_difference_over_pair_noise_sigma':float(np.sqrt(np.mean(a*a))/pair_sigma),
            'count_components_above_3sigma':int(np.count_nonzero(abs(a)>3*pair_sigma)),
            'fraction_components_above_3sigma':float(np.mean(abs(a)>3*pair_sigma))}


def spatial_mechanics(base,other,tag,xbins):
    detail=[]
    x=base['fields']['LE']['coords'][:,0]
    for j in range(len(xbins)-1):
        mask=(x>=xbins[j]) & (x<(xbins[j+1] if j<len(xbins)-2 else xbins[j+1]+1e-6))
        if not mask.any():continue
        row={'case_id':tag,'x_lower_mm':float(xbins[j]),'x_upper_mm':float(xbins[j+1]),'integration_points':int(mask.sum())}
        for kind,components in [('LE',('11','22','12')),('S',('11','22','12'))]:
            u=other['fields'][kind]['values'];v=base['fields'][kind]['values']
            for comp in components:
                ix=COMP.index(comp)
                d=u[mask,ix]-v[mask,ix]
                row[kind+comp+'_change_RMS']=float(np.sqrt(np.mean(d*d)))
        detail.append(row)
    return detail


def run(root, exports,tag,output,noise,confirmed,check_only=False):
    if not confirmed:
        raise RuntimeError('Provide --healthy-material-confirmed ONLY after verifying that the healthy ODB uses E1=45000, E2=12000, G12=4500 MPa, same geometry/load/mesh/orientation.')
    for name in TAGS:
        if not (exports/name).is_dir():raise FileNotFoundError(str(exports/name))
    data={tag:load_one(exports,tag) for tag in TAGS}
    for case in TAGS[1:]:verify_match(data['healthy'],data[case],case)
    print('B9A11J matched final Loading frame, 4592 node labels, 3640 S and 3640 LE IP keys in all four ODBs.')
    sensors,warning=sensor_indices(root,data['healthy'],tag)
    reactions,reaction_warning=compare_reactions(exports)
    if check_only:
        print('B9A11J CHECK PASS; no output written')
        if warning:print('SENSORS:',warning)
        if reaction_warning:print('RF:',reaction_warning)
        return
    if output.exists() and any(output.iterdir()):raise RuntimeError('Refusing overwrite: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    base=data['healthy'];rows=[]; sensor_rows=[];zone_rows=[];xedges=np.arange(-12,12.6,.6)
    warnings=[x for x in (warning,reaction_warning) if x]
    for case in TAGS[1:]:
        obj=data[case];delta=obj['U']-base['U'];loc,severity=TRUTH[case]
        row={'case_id':case,'true_x_mm':loc,'true_E2_reduction_fraction':severity,
             'U_vector':summary_delta(delta,base['U']),
             'U1':summary_delta(delta[:,0],base['U'][:,0]),
             'U2':summary_delta(delta[:,1],base['U'][:,1]),
             'U3':summary_delta(delta[:,2],base['U'][:,2]),
             'reactions':reactions.get(case) if reactions else None,
             'fields':{}}
        for kind in ('S','LE'):
            d=obj['fields'][kind]['values']-base['fields'][kind]['values']
            row['fields'][kind]={component:summary_delta(d[:,i],base['fields'][kind]['values'][:,i])
                                 for i,component in enumerate(COMP)}
        rows.append(row)
        for n,idx in sensors.items():
            rec={'case_id':case,'sensor_count':n,**sensor_stats(delta,idx,noise)}
            sensor_rows.append(rec)
        zone_rows.extend(spatial_mechanics(base,obj,case,xedges))
    # Pairwise discrimination uses signal differences between two DAMAGED states,
    # not between a damaged state and healthy state.
    discrimination=[]
    for ca,cb in ((TAGS[1],TAGS[2]),(TAGS[1],TAGS[3])):
        du=data[ca]['U']-data[cb]['U']
        which='location_at_fixed_severity' if cb==TAGS[2] else 'severity_at_fixed_location'
        comparison={'pair':ca+' vs '+cb,'meaning':which,
                    'full_field_rms_difference_mm':float(np.sqrt(np.mean(du*du))),
                    'rms_over_independent_pair_noise':float(np.sqrt(np.mean(du*du))/(math.sqrt(2)*noise))}
        for n,idx in sensors.items():
            comparison[str(n)+'_sensors']=sensor_stats(du,idx,noise)
        discrimination.append(comparison)
    pd.DataFrame(sensor_rows).to_csv(output/'B9A11J_sensor_detectability.csv',index=False)
    pd.DataFrame(zone_rows).to_csv(output/'B9A11J_spatial_mechanics.csv',index=False)
    (output/'B9A11J_summary.json').write_text(json.dumps({
        'stage':'B9A11J','healthy_reference_baseline_MPa':[45000,12000,4500],
        'healthy_material_and_loading_manually_confirmed':confirmed,
        'source_manifests':{name:data[name]['metadata']['source_odb'] for name in TAGS},
        'noise_model':'Two independent sets of displacement measurements, each Gaussian 1sigma; simulated FEM reference itself is deterministic.',
        'noise_U_mm_each_state_1sigma':noise,
        'noise_S_LE':'Not assumed. S/LE are FE diagnostics, not direct physical measurements.',
        'case_metrics':rows,'pairwise_discrimination':discrimination,
        'warnings':warnings,
        'limitations':['No statistical damage detector or inverse damage estimator is trained.',
                       'Only three damaged cases, insufficient for general inverse model.',
                       'Mechanics coordinates are element centroids, not physical Gauss-point locations.',
                       'Matching geometry and labels does not independently prove section orientation/loads/contact equivalence.',
                       'Healthy-material selected sensors may not be optimal for damage localization.',
                       'The exact damage location is disclosed for evaluation only, never inferred by this code.']},indent=2))
    print('B9A11J RESULTS',output)
    for row in rows:
        print(row['case_id'],'RMS |dU| component=',round(row['U_vector']['rms_change'],8),
              'mm; dU vector norm relative=',round(row['U_vector']['relative_to_healthy_norm_pct'],4),'%')
    if warnings:print('WARNINGS:',' | '.join(warnings))
    print('Send B9A11J_summary.json, B9A11J_sensor_detectability.csv, B9A11J_spatial_mechanics.csv')


def selftest():
    a=np.array([[1.,2.,3.],[-1.,2.,0.]])
    assert summary_delta(a,np.ones_like(a))['rms_change']>0
    assert sensor_stats(a,np.arange(2),0.0002)['nodes']==2
    assert relpct(a,a)==0.
    print('B9A11J SELFTEST PASS: norm calculations and noise thresholds.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--exports',default='B9A11J_exports')
    p.add_argument('--stage',choices=['selftest','check','run'],required=True)
    p.add_argument('--tag',default='run01',help='Existing C sensitivity run tag')
    p.add_argument('--out-tag',default='run01')
    p.add_argument('--noise-u-mm',type=float,default=0.0002)
    p.add_argument('--healthy-material-confirmed',action='store_true')
    a=p.parse_args()
    if not np.isfinite(a.noise_u_mm) or a.noise_u_mm<=0:raise RuntimeError('Positive measurement sigma required')
    if a.stage=='selftest':return selftest()
    root=Path(a.project).expanduser().resolve()
    export=Path(a.exports).expanduser();export=export if export.is_absolute() else root/export
    out=root/'results'/'65AM_B9A11J_damage_detectability'/a.out_tag
    run(root,export,a.tag,out,a.noise_u_mm,a.healthy_material_confirmed,a.stage=='check')

if __name__=='__main__':main()
