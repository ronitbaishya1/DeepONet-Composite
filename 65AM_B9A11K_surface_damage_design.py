#!/usr/bin/env python3
"""B9A11K: damage-specific near-surface virtual-gauge design and frozen-layout transfer.

READ-ONLY: existing B9A11J_exports. Requires existing 65AM_B9A11J_compare_damage.py
in the same directory. Simulated Abaqus integration-point strains near a surface are
VIRTUAL PROXIES, NOT actual surface strain gauge/DIC measurements.

PILOT: design against healthy plus three already examined damage cases. Exploratory;
       these same cases are used for design and cannot validate performance.
FULL:  keep pilot-selected virtual sensors FROZEN, evaluate all 8 known damage
       FE templates plus healthy. The new five simulations are not used to select
       sensors; this still is an oracle-template feasibility analysis, not trained
       or independently validated Bayesian damage localization.

No network training, model modification, or Abaqus solver calls.
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd

PILOT=('healthy','I_DMG_X02_D02','I_DMG_X03_D02','I_DMG_X02_D01')
ALL=('healthy','I_DMG_X01_D01','I_DMG_X01_D02','I_DMG_X02_D01',
     'I_DMG_X02_D02','I_DMG_X03_D01','I_DMG_X03_D02','I_DMG_X04_D01','I_DMG_X04_D02')
OUT='65AM_B9A11K_damage_sensors'
SIZES=(2,5,10,20)
MODES=('LE22','LE22_LE12','U123','LE22_LE12_U123')
STRAIN_COLS={'LE22':(1,), 'LE22_LE12':(1,3),'U123':(), 'LE22_LE12_U123':(1,3)}
TRUTH={'I_DMG_X01_D01':(-8.,.15),'I_DMG_X01_D02':(-8.,.35),
       'I_DMG_X02_D01':(-4.2,.15),'I_DMG_X02_D02':(-4.2,.35),
       'I_DMG_X03_D01':(4.2,.15),'I_DMG_X03_D02':(4.2,.35),
       'I_DMG_X04_D01':(8.,.15),'I_DMG_X04_D02':(8.,.35)}


def load_existing(root):
    p=root/'65AM_B9A11J_compare_damage.py'
    if not p.is_file():
        raise FileNotFoundError('Put the existing B9A11J Python script beside B9A11K: '+str(p))
    spec=importlib.util.spec_from_file_location('b9a11j_locked_reader',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def load_cases(root, exports, cases):
    J=load_existing(root)
    d={name:J.load_one(exports,name) for name in cases}
    for name in cases[1:]:J.verify_match(d['healthy'],d[name],name)
    # Field frame/labels/coordinate safety inherited from the existing audited reader.
    return d


def design_sites(healthy, min_abs_x=1.2, max_abs_x=9., faces='both'):
    ip=healthy['fields']['LE']['coords']
    xyz=healthy['X'];ids=healthy['labels']
    y_ip=np.round(ip[:,1],5)
    ymin=float(y_ip.min());ymax=float(y_ip.max())
    y_node_min=float(xyz[:,1].min());y_node_max=float(xyz[:,1].max())
    xvals=np.round(ip[:,0],5)
    # Detect genuine cross-section element layers; no hardcoded nodes/elements.
    if len(np.unique(y_ip))<2 or len(np.unique(xvals))<8:
        raise RuntimeError('Unexpected 3D specimen mesh for virtual sensor construction')
    candidates=[];groups=[];node_ids=[]
    for side in (('y_min','y_max') if faces=='both' else (faces,)):
        yy=ymin if side=='y_min' else ymax
        ny=y_node_min if side=='y_min' else y_node_max
        surf=np.flatnonzero(np.abs(xyz[:,1]-ny)<1e-4)
        if len(surf)<10:raise RuntimeError('Surface node set too small')
        zmid=0.5*(float(ip[:,2].min())+float(ip[:,2].max()))
        for xx in np.sort(np.unique(xvals)):
            if abs(xx)<min_abs_x-1e-6 or abs(xx)>max_abs_x+1e-6:continue
            # Center-of-width, about 2 mm wide, rather than averaging the entire
            # 8 mm specimen width into one unrealistically large physical gauge.
            g=np.flatnonzero((np.abs(xvals-xx)<1e-4)&(np.abs(y_ip-yy)<1e-4)&
                             (np.abs(ip[:,2]-zmid)<=1.0+1e-6))
            if len(g)<2:continue
            center=np.mean(ip[g],axis=0)
            # Match an accessible outer-surface NODE near the virtual strain patch.
            target=np.array([xx,ny,np.mean(ip[g,2])],float)
            delta=np.linalg.norm(xyz[surf]-target[None,:],axis=1)
            node=int(surf[np.argmin(delta)])
            candidates.append({'site_id':len(candidates),'surface':side,
                'center_x_mm':float(center[0]),'near_surface_ip_y_mm':float(center[1]),
                'target_surface_y_mm':ny,'center_z_mm':float(center[2]),
                'ip_count':len(g),'displacement_node_label':int(ids[node]),
                'displacement_node_x_mm':float(xyz[node,0]),
                'displacement_node_y_mm':float(xyz[node,1]),
                'displacement_node_z_mm':float(xyz[node,2]),
                'node_mapping_gap_mm':float(delta.min())})
            groups.append(g);node_ids.append(node)
    if len(candidates)<max(SIZES):raise RuntimeError('Too few candidate near-surface gauges')
    df=pd.DataFrame(candidates)
    return df,groups,np.array(node_ids,dtype=int)


def read_virtual_observables(cases,site_groups,node_ids):
    # Each site: averaged nearest-surface-layer IP strain, and nearest surface node U.
    answer={}
    for key,obj in cases.items():
        strain=obj['fields']['LE']['values']
        s=np.array([strain[g][:,[1,3]].mean(axis=0) for g in site_groups])
        u=obj['U'][node_ids,:]
        if not np.isfinite(s).all() or not np.isfinite(u).all():
            raise RuntimeError('Invalid virtual sensors '+key)
        answer[key]={'strain':s,'u':u}
    return answer


def features(obs,case,mode,sigma_strain_ue, sigma_U_mm,reference='healthy'):
    a=obs[case];b=obs[reference]
    features=[]
    if mode in ('LE22','LE22_LE12','LE22_LE12_U123'):
        cols=[0] if mode=='LE22' else [0,1]
        features.append((a['strain'][:,cols]-b['strain'][:,cols])/(sigma_strain_ue*1e-6))
    if mode in ('U123','LE22_LE12_U123'):
        features.append((a['u']-b['u'])/sigma_U_mm)
    return np.concatenate(features,axis=1)


def distance_mat(mats,known):
    # One state noisy measurement versus another independently noisy state; pair sigma sqrt(2).
    k=len(known);n=len(next(iter(mats.values())))
    d=np.zeros((k,k,n),float)
    for i,a in enumerate(known):
        for j in range(i+1,k):
            val=np.sum((mats[a]-mats[known[j]])**2,axis=1)/2.
            d[i,j]=d[j,i]=val
    return d


def greedy(d, count):
    # Maximize worst pairwise Mahalanobis separation; secondary mean distinguishes ties.
    pairs=np.triu_indices(d.shape[0],1)
    selected=[];remaining=list(range(d.shape[2]))
    totals=np.zeros((d.shape[0],d.shape[1]))
    for k in range(min(count,len(remaining))):
        score=[]
        for i in remaining:
            ds=(totals+d[:,:,i])[pairs]
            score.append((float(np.min(ds)),float(np.mean(ds)),-i,i))
        best=max(score)[-1]
        selected.append(best);remaining.remove(best);totals+=d[:,:,best]
    return selected


def pair_rows(mats,names,chosen,mode,sigma_s,tag):
    rows=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            # Whitened feature under single-state sigma. independent differential sigma = sqrt2.
            vec=(mats[a][chosen]-mats[b][chosen]).reshape(-1)/math.sqrt(2)
            D=float(np.linalg.norm(vec));
            rows.append({'cohort':tag,'mode':mode,'sigma_strain_ue':sigma_s,
                         'sensor_count':len(chosen),'case_a':a,'case_b':b,
                         'Mahalanobis_pair_distance':D,
                         'best_case_two_class_error_ideal':0.5*math.erfc(D/(2*math.sqrt(2))),
                         'case_a_is_new':a not in PILOT,'case_b_is_new':b not in PILOT})
    return rows


def pilot(root,exports,tag,noise_u,noise_strains,confirmed,check_only=False,face='both'):
    if not confirmed:raise RuntimeError('Use --healthy-material-confirmed only after checking matched healthy and damaged Abaqus models')
    cases=load_cases(root,exports,PILOT)
    sites,groups,node_ids=design_sites(cases['healthy'],faces=face)
    print('K PILOT inputs PASS:',len(cases),'matched FEM cases,',len(sites),'near-surface virtual-gauge candidates')
    if check_only:return
    out=root/'results'/OUT/tag
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing to overwrite existing results '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    sites.to_csv(out/'B9A11K_candidate_sites.csv',index=False)
    obs=read_virtual_observables(cases,groups,node_ids)
    selected_rows=[];pairs=[];overview=[]
    for mode in MODES:
        for sigma in ([float(noise_strains[0])] if mode=='U123' else noise_strains):
            mats={k:features(obs,k,mode,sigma,noise_u) for k in PILOT}
            d=distance_mat(mats,PILOT)
            selection=greedy(d,max(SIZES))
            for n in SIZES:
                chosen=selection[:n]
                group=pair_rows(mats,PILOT,chosen,mode,sigma,'pilot_design_in_sample')
                pairs.extend(group)
                overview.append({'mode':mode,'sigma_strain_ue':sigma,'sensor_count':n,
                                 'minimum_pair_distance':min(r['Mahalanobis_pair_distance'] for r in group),
                                 'median_pair_distance':float(np.median([r['Mahalanobis_pair_distance'] for r in group])),
                                 'chosen_site_ids':','.join(str(i) for i in chosen)})
                for rank,i in enumerate(chosen,1):
                    selected_rows.append({'mode':mode,'sigma_strain_ue':sigma,'sensor_count':n,
                                          'sensor_rank':rank,'site_id':i})
    pd.DataFrame(selected_rows).to_csv(out/'B9A11K_frozen_sensor_design.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'B9A11K_pilot_pairwise.csv',index=False)
    pd.DataFrame(overview).to_csv(out/'B9A11K_pilot_screen.csv',index=False)
    summary={'step':'B9A11K-pilot','source':'healthy + 3 historical damaged Abaqus FEM ODBs',
             'input_cases':list(PILOT),'candidate_site_count':len(sites),
             'selected_sensor_counts':list(SIZES),'measurement_types':list(MODES),
             'noise_strain_ue_single_state':list(noise_strains),'noise_u_mm_single_state':noise_u,
             'near_surface_approximation':'Virtual patch averages C3D8R centroid LE across an approximately 2mm center-of-width band in nearest interior surface layer; NOT actual surface strains or a calibrated physical strain gauge.',
             'contact_accessibility':'Top/bottom placements require experimental accessibility checks, including roller/nose contact exclusion.',
             'statistics':'Ideal Gaussian independent-noise pair-separation lower-bound error; assumes exactly known FE signals and no model error.',
             'validation':'IN-SAMPLE SENSOR DESIGN. Same 3 damaged cases select gauges and score them. NOT independently validated.',
             'output_files':['B9A11K_candidate_sites.csv','B9A11K_frozen_sensor_design.csv','B9A11K_pilot_pairwise.csv','B9A11K_pilot_screen.csv']}
    (out/'B9A11K_pilot_summary.json').write_text(json.dumps(summary,indent=2))
    print('K PILOT DONE',out,' | Send B9A11K_pilot_screen.csv + B9A11K_candidate_sites.csv + summary JSON')


def full(root,exports,pilot_tag,tag,noise_u,confirmed):
    if not confirmed:raise RuntimeError('Confirm matched FEM healthy and damaged models first')
    cases=load_cases(root,exports,ALL)
    sites,groups,nodes=design_sites(cases['healthy'])
    prior=root/'results'/OUT/pilot_tag
    paths=[prior/'B9A11K_candidate_sites.csv',prior/'B9A11K_frozen_sensor_design.csv']
    for p in paths:
        if not p.is_file():raise FileNotFoundError('Pilot run missing '+str(p))
    old=pd.read_csv(paths[0]);design=pd.read_csv(paths[1]);new=sites
    # Strict physical and index identity; detect any new mesh or changed settings.
    compare=('site_id','surface','center_x_mm','near_surface_ip_y_mm','center_z_mm','displacement_node_label')
    if len(old)!=len(new):raise RuntimeError('Candidate virtual sensor geometry changed')
    for k in compare:
        a=old[k].to_numpy();b=new[k].to_numpy()
        if k in ('surface',):
            if not np.array_equal(a,b):raise RuntimeError('Changed gauge positions: '+k)
        elif not np.allclose(a.astype(float),b.astype(float),atol=1e-7,rtol=0):
            raise RuntimeError('Changed gauge positions: '+k)
    obs=read_virtual_observables(cases,groups,nodes)
    out=root/'results'/OUT/tag
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing to overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[];overview=[]
    for (mode,sigma,n),tbl in design.groupby(['mode','sigma_strain_ue','sensor_count'],sort=False):
        tbl=tbl.sort_values('sensor_rank');chosen=[int(x) for x in tbl.site_id]
        if len(chosen)!=int(n) or len(set(chosen))!=int(n):raise RuntimeError('Broken locked gauge design')
        mats={k:features(obs,k,mode,float(sigma),noise_u) for k in ALL}
        items=pair_rows(mats,ALL,chosen,mode,float(sigma),'full_oracle_template_FE')
        rows.extend(items)
        by_new=[r for r in items if r['case_a_is_new'] or r['case_b_is_new']]
        overview.append({'mode':mode,'sigma_strain_ue':sigma,'sensor_count':n,
                         'minimum_all_pair_distance':min(r['Mahalanobis_pair_distance'] for r in items),
                         'minimum_pair_involving_new_case':min(r['Mahalanobis_pair_distance'] for r in by_new),
                         'minimum_strong_location_pair':min(r['Mahalanobis_pair_distance'] for r in items
                            if r['case_a'].endswith('D02') and r['case_b'].endswith('D02') and r['case_a']!='healthy'),
                         'locked_from_pilot_tag':pilot_tag})
    pd.DataFrame(rows).to_csv(out/'B9A11K_full_pairwise.csv',index=False)
    pd.DataFrame(overview).to_csv(out/'B9A11K_full_screen.csv',index=False)
    report={'step':'B9A11K-full','cases':list(ALL),'num_damaged_cases':8,
            'pilot_tag':pilot_tag,'gauge_design_locked':True,
            'source':'all 8 discrete Abaqus stiffness-loss templates (4 axial locations x 2 severities) plus healthy',
            'is_trained_damage_operator':False,'is_validated_Bayesian_damage_posterior':False,
            'statistical_interpretation':'Oracle-template pairwise Gaussian separability only. The FE signatures themselves are the hypotheses, so this is not unseen-forward-case validation.',
            'covariance_warning':'Independent sensor noise only. About 2mm-wide near-surface centroid band, not measured surface field. No correlated sensor errors, healthy-parameter uncertainty, contact discrepancy, or surrogate error.'}
    (out/'B9A11K_full_summary.json').write_text(json.dumps(report,indent=2))
    print('K FULL DONE',out,' | Send full summary JSON + full_screen CSV')


def selftest():
    raw={k:{'strain':np.array([[.001,0],[.002,0],[.003,0]],float),
            'u':np.zeros((3,3))} for k in PILOT}
    raw[PILOT[1]]['strain'][0,0]+=.0001
    raw[PILOT[2]]['strain'][1,0]+=.0001
    raw[PILOT[3]]['strain'][2,0]+=.00008
    mats={k:features(raw,k,'LE22',25.,.0002) for k in PILOT}
    d=distance_mat(mats,PILOT)
    g=greedy(d,3)
    assert len(set(g))==3 and set(g)=={0,1,2}
    p=pair_rows(mats,PILOT,g[:2],'LE22',25.,'test')
    assert len(p)==6 and all(np.isfinite(r['Mahalanobis_pair_distance']) for r in p)
    # Proper pair-noise sigma: 100ue / sqrt(2) / 25ue for one component.
    assert np.isclose(np.sqrt(d[0,1,0]),100/(math.sqrt(2)*25))
    print('B9A11K SELFTEST PASS | pair-noise | greedy objective | virtual-sensor comparison')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('selftest','check','pilot','full'))
    p.add_argument('--project',default='.')
    p.add_argument('--exports',default='B9A11J_exports')
    p.add_argument('--tag',default='pilot01')
    p.add_argument('--pilot-tag',default='pilot01')
    p.add_argument('--noise-strain-ue',default='10,25,50,100',help='One-state SD in microstrain; comma-separated assumptions')
    p.add_argument('--noise-u-mm',type=float,default=0.0002)
    p.add_argument('--faces',choices=('both','y_min','y_max'),default='both')
    p.add_argument('--healthy-material-confirmed',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if a.stage=='selftest':return selftest()
    noises=[float(s) for s in a.noise_strain_ue.split(',')]
    if not noises or not all(np.isfinite(n) and n>0 for n in noises):raise ValueError('Nonpositive strain noise')
    if not (a.noise_u_mm>0 and np.isfinite(a.noise_u_mm)):raise ValueError('Nonpositive U noise')
    export=Path(a.exports).expanduser();export=export if export.is_absolute() else root/export
    if a.stage in ('check','pilot'):
        pilot(root,export,a.tag,a.noise_u_mm,noises,a.healthy_material_confirmed,a.stage=='check',a.faces)
    elif a.stage=='full':
        if a.faces!='both':raise ValueError('FULL currently uses pilot default --faces both; do not change gauge geometry')
        full(root,export,a.pilot_tag,a.tag,a.noise_u_mm,a.healthy_material_confirmed)

if __name__=='__main__':main()
