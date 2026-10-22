#!/usr/bin/env python3
"""B9A11L-A: paired left/right, unique-node near-surface VIRTUAL gauge design.

Reads the nine already examined Abaqus FEM templates through B9A11J/B9A11K readers.
All nine are DEVELOPMENT data. Reports are IN-SAMPLE TEMPLATE separation only.
New held-out damaged Abaqus positions MUST NOT be used for selection.
No Abaqus jobs, no model training, no overwrites. Requires numpy/pandas.
"""
import argparse, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ALL=('healthy','I_DMG_X01_D01','I_DMG_X01_D02','I_DMG_X02_D01','I_DMG_X02_D02',
     'I_DMG_X03_D01','I_DMG_X03_D02','I_DMG_X04_D01','I_DMG_X04_D02')
SIZES=(10,20)
NOISES=(10.,25.,50.,100.)
MODE='LE22_LE12_U123'

def load_k(root):
    f=root/'65AM_B9A11K_surface_damage_design.py'
    if not f.is_file(): raise FileNotFoundError('Missing existing '+str(f))
    s=importlib.util.spec_from_file_location('b9a11k_for_l',f)
    mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
    return mod

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mirrored_candidates(df,minabs=2.1,maxabs=8.2):
    # Enforce physical x symmetry in candidate locations, same surface on each side.
    # Avoid gross contact zones near center and far ends; still inspect actual contact.
    rows=[]
    for face in ('y_min','y_max'):
        neg=df[(df.surface==face)&(df.center_x_mm<-minabs+1e-6)&(df.center_x_mm>=-maxabs-1e-6)]
        pos=df[(df.surface==face)&(df.center_x_mm>minabs-1e-6)&(df.center_x_mm<=maxabs+1e-6)]
        for _,r in neg.iterrows():
            match=pos.iloc[np.argmin(np.abs(pos.center_x_mm.to_numpy()+r.center_x_mm))] if len(pos) else None
            if match is None or abs(float(match.center_x_mm)+float(r.center_x_mm))>0.03:continue
            if int(r.displacement_node_label)==int(match.displacement_node_label):continue
            rows.append({'left_id':int(r.site_id),'right_id':int(match.site_id),'surface':face,
                         'abs_x_mm':float(abs(r.center_x_mm)),
                         'left_node':int(r.displacement_node_label),
                         'right_node':int(match.displacement_node_label)})
    if len(rows)<10:raise RuntimeError('Insufficient mirrored candidates with requested x/contact margin')
    return rows

def site_arrays(K,obs,noise_s,noise_u):
    return {k:K.features(obs,k,MODE,float(noise_s),float(noise_u)) for k in ALL}

def scores_by_pair(mats, pairs):
    ia,ib=np.triu_indices(len(ALL),1)
    x=np.stack([mats[k] for k in ALL],axis=0)
    d=np.empty((len(pairs),len(ia)),float)
    for j,p in enumerate(pairs):
        a=x[:,p['left_id'],:];b=x[:,p['right_id'],:]
        # Difference of TWO independently noisy physical states: variance x2.
        d[j]=(.5*((a[ia]-a[ib])**2+(b[ia]-b[ib])**2).sum(axis=1))
    return d,ia,ib

def select_pairs(pairs,pairscore,n_pairs,penalty_band=0.60):
    active=[];labels=set(); chosen=[];score=np.zeros(pairscore.shape[1],float)
    for _ in range(n_pairs):
        opt=[]
        for j,p in enumerate(pairs):
            if j in active:continue
            if p['left_node'] in labels or p['right_node'] in labels:continue
            # physically separate same-face gauges; x mirror is always exact.
            if any(p['surface']==pairs[z]['surface'] and abs(p['abs_x_mm']-pairs[z]['abs_x_mm'])<penalty_band for z in active):continue
            vals=score+pairscore[j]
            opt.append((float(vals.min()),float(vals.mean()),-j,j))
        if not opt:raise RuntimeError('Cannot select '+str(n_pairs)+' mirrored pairs without duplicate nodes')
        j=max(opt)[-1];p=pairs[j]
        active.append(j);chosen.extend((p['left_id'],p['right_id']))
        labels.update((p['left_node'],p['right_node']));score+=pairscore[j]
    return active,chosen

def distances(K,mats,ids,noise_s):
    x=K.pair_rows(mats,ALL,ids,MODE,float(noise_s),'development_oracle_template')
    return x

def run(root,tag,noise_u,minabs,maxabs,stage):
    K=load_k(root)
    exports=root/'B9A11J_exports'
    cases=K.load_cases(root,exports,ALL)
    sites,groups,nodes=K.design_sites(cases['healthy'])
    obs=K.read_virtual_observables(cases,groups,nodes)
    mirrors=mirrored_candidates(sites,minabs,maxabs)
    old=root/'results'/'65AM_B9A11K_damage_sensors'/'pilot01'/'B9A11K_frozen_sensor_design.csv'
    if not old.is_file():raise FileNotFoundError('Expected original pilot design: '+str(old))
    previous=pd.read_csv(old)
    print('L-A CHECK PASS: 9 matched FEM cases,',len(sites),'candidate gauges,',len(mirrors),'mirrored pairs')
    if stage=='check':return
    out=root/'results'/'65AM_B9A11L_balanced_design'/tag
    if out.exists():raise RuntimeError('Output exists; refusing overwrite: '+str(out))
    out.mkdir(parents=True)
    design=[];pairlog=[];screen=[]
    # Select using the locked design noise 25 microstrain; compare at other noise levels.
    seedmats=site_arrays(K,obs,25.,noise_u)
    pscore,_,_=scores_by_pair(seedmats,mirrors)
    for n in SIZES:
        pairidx,selected=select_pairs(mirrors,pscore,n//2)
        assert len(selected)==n and len(set(selected))==n
        select_nodes=[int(sites.iloc[j].displacement_node_label) for j in selected]
        if len(set(select_nodes))!=n:raise RuntimeError('Duplicated displacement nodes')
        if sum(float(sites.iloc[j].center_x_mm)<0 for j in selected)!=n//2:raise RuntimeError('Left/right imbalance')
        for i,j in enumerate(selected,1):
            a=sites.iloc[j]
            design.append({'sensor_count':n,'sensor_rank':i,'site_id':int(j),
                           'surface':a.surface,'center_x_mm':float(a.center_x_mm),
                           'target_surface_y_mm':float(a.target_surface_y_mm),
                           'displacement_node_label':int(a.displacement_node_label),
                           'near_surface_ip_y_mm':float(a.near_surface_ip_y_mm),
                           'center_z_mm':float(a.center_z_mm),'ip_count':int(a.ip_count)})
        for noise in NOISES:
            mats=site_arrays(K,obs,noise,noise_u)
            pr=distances(K,mats,selected,noise)
            for row in pr:pairlog.append({'design':'B9A11L_balanced','sensor_count':n,**row})
            screen.append({'design':'B9A11L_balanced','sensor_count':n,'sigma_strain_ue':noise,
                           'min_pair_distance':min(p['Mahalanobis_pair_distance'] for p in pr),
                           'min_healthy_vs_damaged_distance':min(p['Mahalanobis_pair_distance'] for p in pr if p['case_a']=='healthy'),
                           'min_strong_location_pair_distance':min(p['Mahalanobis_pair_distance'] for p in pr if p['case_a'].endswith('D02') and p['case_b'].endswith('D02'))})
            prior=previous[(previous['mode']==MODE)&(np.isclose(previous.sigma_strain_ue,noise))&(previous.sensor_count==n)].sort_values('sensor_rank')
            if len(prior)==n:
                oldids=[int(v) for v in prior.site_id]
                oldp=distances(K,mats,oldids,noise)
                screen.append({'design':'B9A11K_previous','sensor_count':n,'sigma_strain_ue':noise,
                    'min_pair_distance':min(p['Mahalanobis_pair_distance'] for p in oldp),
                    'min_healthy_vs_damaged_distance':min(p['Mahalanobis_pair_distance'] for p in oldp if p['case_a']=='healthy'),
                    'min_strong_location_pair_distance':min(p['Mahalanobis_pair_distance'] for p in oldp if p['case_a'].endswith('D02') and p['case_b'].endswith('D02'))})
    designfile=out/'B9A11L_locked_sensor_design.csv';pd.DataFrame(design).to_csv(designfile,index=False)
    pd.DataFrame(screen).to_csv(out/'B9A11L_in_sample_comparison.csv',index=False)
    pd.DataFrame(pairlog).to_csv(out/'B9A11L_in_sample_pairwise.csv',index=False)
    sites.to_csv(out/'B9A11L_site_inventory.csv',index=False)
    summary={'stage':'B9A11L-A','new_sensor_design':'Mirrored x pairs, balanced sides, unique U nodes',
       'selector_training':'ALL EIGHT damaged cases + healthy; IN-SAMPLE only',
       'validation_required':'Four NEW independently generated Abaqus cases at x=+-6 mm, severity 20%,30%',
       'locked_sensor_design_sha256':sha(designfile),
       'signal':'near-surface C3D8R centroid LE22/LE12 + nearest surface-node U1/U2/U3',
       'not_true_surface_measurements':True,
       'experimental_exclusion_not_certified':'Upper/lower gauge access and contact exclusion need a physical audit',
       'noise_model':'Independent Gaussian measurement errors; difference of two noisy states, sqrt2 scaling',
       'healthy_parameters_fixed_in_all_damage_design_cases':True,
       'prevents_duplicate_U_nodes':True,'sensor_counts':list(SIZES),'cases_used_for_design':list(ALL),
       'new_test_cases_not_seen':True}
    (out/'B9A11L_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('L-A DONE:',out,'\nSend B9A11L_in_sample_comparison.csv and B9A11L_locked_sensor_design.csv')

def selftest():
    # Direct selection invariants on ten disjoint mirrored pairs.
    p=[{'left_id':2*i,'right_id':2*i+1,'surface':'y_min','abs_x_mm':1.2*i,'left_node':2*i,'right_node':2*i+1} for i in range(12)]
    z=np.arange(len(p)*3,dtype=float).reshape(len(p),3)*.01+1
    ix,ids=select_pairs(p,z,5)
    assert len(ix)==5 and len(set(ids))==10
    print('B9A11L-A SELFTEST PASS | mirrored pairing, no reused IDs')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['selftest','check','run'])
    p.add_argument('--project',default='.')
    p.add_argument('--tag',default='design01')
    p.add_argument('--noise-u-mm',type=float,default=.0002)
    p.add_argument('--min-abs-x-mm',type=float,default=2.1)
    p.add_argument('--max-abs-x-mm',type=float,default=8.2)
    p.add_argument('--healthy-material-confirmed',action='store_true')
    a=p.parse_args()
    if a.stage=='selftest':return selftest()
    if not a.healthy_material_confirmed:raise RuntimeError('Check healthy/damage material and loading before using --healthy-material-confirmed')
    root=Path(a.project).expanduser().resolve()
    if not a.noise_u_mm>0:raise ValueError('Bad U noise')
    run(root,a.tag,a.noise_u_mm,a.min_abs_x_mm,a.max_abs_x_mm,a.stage)
if __name__=='__main__':main()
