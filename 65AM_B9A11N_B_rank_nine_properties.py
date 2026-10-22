#!/usr/bin/env python3
"""B9A11N-B: compute and plot normalized 9-property LOCAL forward sensitivity.

Requires same FEM mesh and loading across 18 perturbations and existing healthy.
Compares matched Abaqus nodes and C3D8R integration point fields using verified
B9A11J reader. Uses B9A11K's virtual strain site construction and B9A11L
locked 10/20-site selections. These are not actual physical surface gauges.
Not an inverse-model feature-importance measure, and not a claim of identifiability.
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARS=['E1','E2','E3','nu12','nu13','nu23','G12','G13','G23']
NOMINAL=[45000.,12000.,12000.,.28,.28,.40,4500.,4500.,3500.]
RESP=['U1','U2','U3','LE22','LE12']

def load_py(path,module):
    if not path.is_file():raise FileNotFoundError('Missing required original project script: '+str(path))
    spec=importlib.util.spec_from_file_location(module,str(path))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))

def make_virtual(obj,site_groups,node_ids,site_idxs):
    LE=obj['fields']['LE']['values']
    return {'U1':obj['U'][node_ids[site_idxs],0],
            'U2':obj['U'][node_ids[site_idxs],1],
            'U3':obj['U'][node_ids[site_idxs],2],
            'LE22':np.array([np.mean(LE[site_groups[i],1]) for i in site_idxs],float),
            'LE12':np.array([np.mean(LE[site_groups[i],3]) for i in site_idxs],float)}

def make_field(obj):
    LE=obj['fields']['LE']['values']
    return {'U1':obj['U'][:,0], 'U2':obj['U'][:,1], 'U3':obj['U'][:,2],
            'LE22':LE[:,1], 'LE12':LE[:,3]}

def compute_measures(base,plus,minus,h,noise_u,noise_s):
    if sorted(base)!=sorted(RESP):raise RuntimeError('Unexpected response names')
    out={};mix=[]
    for name in RESP:
        b=np.asarray(base[name]);p=np.asarray(plus[name]);m=np.asarray(minus[name])
        if b.shape!=p.shape or p.shape!=m.shape or not np.isfinite(p).all() or not np.isfinite(m).all():
            raise RuntimeError('Invalid or changed field shape for '+name)
        # Noise-whitened effect from a +5% fractional property shift, approximated
        # by central finite difference at +/-h. Dimensionless effect in sigma units.
        # For h=0.05, (plus-minus)/2 is the first-order +5% response change.
        sig=noise_s if name.startswith('LE') else noise_u
        deriv=(p-m)/(2*h)
        effect=h*deriv/sig
        # Relative sensitivity at nominal p, reference RMS field scale; no artificially
        # huge relative influence when healthy field is nearly zero.
        rel=h*rms(deriv)/max(rms(b),sig)
        out[name]={'RMS_effect_per_h_sigma':rms(effect), 'RMS_effect_physical':h*rms(deriv),
                   'local_log_sensitivity_rms':rms(deriv)/max(rms(b),sig),
                   'baseline_RMS':rms(b),'relative_h_response':rel,'noise_sigma':sig}
        mix.append(effect)
    # Equal unit weight for five observed components per site. Must keep
    # row counts consistent to avoid silent reweighting.
    return out,np.concatenate(mix)

def gather(root,export_dir,healthy,reader):
    cases={'healthy':reader.load_one(export_dir,'healthy')}
    for par in PARS:
        for sign in ('M','P'):
            key='N9_'+par+'_'+sign
            cases[key]=reader.load_one(export_dir,key)
            reader.verify_match(cases['healthy'],cases[key],key)
    print('Validated 18 perturbation exports + healthy; matching node/IP keys/coords and Loading t=1.0.')
    return cases

def get_locked_design(root,healthy,helper):
    site,groups,nodes=helper.design_sites(healthy)
    path=root/'results'/'65AM_B9A11L_balanced_design'/'design01'/'B9A11L_locked_sensor_design.csv'
    if not path.is_file():raise FileNotFoundError('Locked B9A11L design missing '+str(path))
    design=pd.read_csv(path)
    for col in ('sensor_count','site_id','displacement_node_label','center_x_mm','surface'):
        if col not in design:raise RuntimeError('Locked sensor design lacks '+col)
    indices={}
    for n in (10,20):
        part=design[design.sensor_count==n].sort_values('sensor_rank')
        if len(part)!=n:raise RuntimeError('Expected '+str(n)+' locked sensor rows')
        idx=part.site_id.to_numpy(dtype=int)
        if len(set(idx))!=n or min(idx)<0 or max(idx)>=len(site):raise RuntimeError('Invalid or duplicated site ids')
        for row in part.itertuples():
            ref=site.iloc[int(row.site_id)]
            if int(ref.displacement_node_label)!=int(row.displacement_node_label) or \
               abs(float(ref.center_x_mm)-float(row.center_x_mm))>1e-4 or \
               str(ref.surface)!=str(row.surface):
                raise RuntimeError('Sensor layout changed from B9A11L at site '+str(row.site_id))
        if len(set(int(site.iloc[i].displacement_node_label) for i in idx))!=n:
            raise RuntimeError('Duplicated displacement nodes in locked sensor design')
        indices[n]=idx
    return site,groups,nodes,indices

def calculate(cases,projection,h,noise_u,noise_s,config):
    baseline=projection(cases['healthy'])
    rows=[];mat=[];jac=[];full_comp=[]
    for par in PARS:
        plus=projection(cases['N9_'+par+'_P']);minus=projection(cases['N9_'+par+'_M'])
        metrics,col=compute_measures(baseline,plus,minus,h,noise_u,noise_s)
        mix=np.concatenate([(plus[k]-minus[k])/(2*({k:noise_s if k.startswith('LE') else noise_u}[k])) for k in RESP])
        score=rms(mix) # shift induced by h=5% fractional perturbation
        rows.append({'configuration':config,'parameter':par,'family':('Young' if par.startswith('E') else 'Poisson' if par.startswith('nu') else 'Shear'),
                     'nominal_value':NOMINAL[PARS.index(par)], 'fractional_perturbation':h,
                     'importance_RMS_5pct_in_measurement_noise_sigma':score,
                     'importance_note':'Whitened RMS response shift for fractional property change h; one-at-a-time local FEM sensitivity, not inverse posterior variable importance'})
        mat.append([metrics[k]['RMS_effect_per_h_sigma'] for k in RESP])
        jac.append(col)
        for k in RESP:
            full_comp.append({'configuration':config,'parameter':par,'response':k,
                              **metrics[k]})
    J=np.column_stack(jac) # rows stacked as [U1(all sites), U2(...), U3, LE22, LE12]
    sv=np.linalg.svd(J,full_matrices=False)[1]
    norm=np.sqrt(np.sum(J**2,axis=0))
    corr=np.divide(J.T@J,norm[:,None]*norm[None,:],out=np.zeros((9,9)),where=(norm[:,None]*norm[None,:])>1e-12)
    np.fill_diagonal(corr,1)
    summary={'configuration':config,'jacobian_shape':list(J.shape),'singular_values':list(map(float,sv)),
             'relative_singular_values':list(map(float,sv/sv[0])) if sv[0] else [],
             'numerical_rank_relative_1e-3':int(np.count_nonzero(sv>sv[0]*1e-3)) if len(sv) else 0,
             'condition_number':float(sv[0]/sv[-1]) if sv[-1]>1e-12 else None,
             'max_abs_offdiagonal_column_cosine':float(np.max(np.abs(corr-np.eye(9)))),
             'caution':'Rank/condition are conditional on these virtual or full-field observations and assumed independent component errors, not calibrated Bayesian identifiability.'}
    return pd.DataFrame(rows),pd.DataFrame(mat,index=PARS,columns=RESP),pd.DataFrame(full_comp),sv,corr,summary

def plots(out,importance,heat,sv,corr,tag):
    rank=importance.sort_values('importance_RMS_5pct_in_measurement_noise_sigma',ascending=True)
    fig,ax=plt.subplots(figsize=(9,5.3))
    ax.barh(rank['parameter'],rank['importance_RMS_5pct_in_measurement_noise_sigma'])
    ax.set_xlabel('RMS observable shift / single-state measurement noise σ (for +5% property change)')
    ax.set_ylabel('Orthotropic elastic constant')
    ax.set_title('Nine-property local importance — '+tag)
    ax.grid(axis='x',alpha=.25)
    fig.tight_layout();fig.savefig(out/('B9A11N_importance_'+tag+'.png'),dpi=220)
    fig.savefig(out/('B9A11N_importance_'+tag+'.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.6))
    im=ax.imshow(heat.to_numpy(),aspect='auto')
    ax.set_xticks(np.arange(5),RESP);ax.set_yticks(np.arange(9),PARS)
    fig.colorbar(im,ax=ax,label='RMS effect / σ for +5% material-property change')
    ax.set_title('Sensitivity by measured response — '+tag)
    fig.tight_layout();fig.savefig(out/('B9A11N_heatmap_'+tag+'.png'),dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    ax.semilogy(range(1,len(sv)+1),sv,marker='o')
    ax.set(xlabel='Singular-value index',ylabel='Singular value (noise-whitened response for 5% fractional change)',title='Local sensitivity-matrix spectrum — '+tag)
    ax.grid(True,alpha=.3);fig.tight_layout()
    fig.savefig(out/('B9A11N_identifiability_'+tag+'.png'),dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,6))
    im=ax.imshow(corr,vmin=-1,vmax=1)
    ax.set_xticks(np.arange(9),PARS,rotation=35,ha='right');ax.set_yticks(np.arange(9),PARS)
    fig.colorbar(im,ax=ax,label='Cosine similarity of local sensitivity vectors')
    ax.set_title('Parameter confounding diagnostic — '+tag)
    fig.tight_layout();fig.savefig(out/('B9A11N_parameter_confounding_'+tag+'.png'),dpi=220);plt.close(fig)

def selftest():
    y={k:np.array([1.,2.,3.]) for k in RESP}
    plus={k:y[k]*1.05 for k in RESP};minus={k:y[k]*.95 for k in RESP}
    met,vec=compute_measures(y,plus,minus,.05,.2,.1)
    assert len(vec)==15 and met['U1']['RMS_effect_per_h_sigma']>0
    assert abs(vec[0]-(plus['U1'][0]-minus['U1'][0])/(2*.2))<1e-10
    print('SELFTEST PASS: central sensitivities, mixed units and field-array shape')

def run(args):
    root=Path(args.project).resolve();exports=root/args.exports
    j=load_py(root/'65AM_B9A11J_compare_damage.py','b9n_j')
    helper=load_py(root/'65AM_B9A11K_surface_damage_design.py','b9n_k')
    cases=gather(root,exports,None,j)
    sites,groups,nodes,indices=get_locked_design(root,cases['healthy'],helper)
    # Audit material perturbation metadata if supplied; source audit on Windows remains authoritative.
    if args.stage=='check':
        print('CHECK PASS: 19 matched FEM states; locked 10/20-site layouts; no data used for scoring')
        return
    out=root/'results'/'65AM_B9A11N_nine_property_importance'/args.tag
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    all_rows=[];all_comp=[];records=[]
    for config in ('20_sensors','10_sensors','full_FEM'):
        if config=='full_FEM':proj=make_field
        else:
            chosen=indices[int(config.split('_')[0])]
            proj=lambda obj,ch=chosen:make_virtual(obj,groups,nodes,ch)
        ranking,heat,comp,sv,corr,info=calculate(cases,proj,args.fraction,args.sigma_u,args.sigma_strain,config)
        ranking.to_csv(out/('B9A11N_importance_'+config+'.csv'),index=False)
        heat.to_csv(out/('B9A11N_heatmap_'+config+'.csv'))
        plots(out,ranking,heat,sv,corr,config)
        all_rows.append(ranking);all_comp.append(comp);records.append(info)
        top=ranking.sort_values('importance_RMS_5pct_in_measurement_noise_sigma',ascending=False).head(3)
        print(config,'TOP THREE:',list(zip(top.parameter,top.importance_RMS_5pct_in_measurement_noise_sigma.round(4))))
    pd.concat(all_rows).to_csv(out/'B9A11N_all_importance.csv',index=False)
    pd.concat(all_comp).to_csv(out/'B9A11N_component_sensitivity.csv',index=False)
    summary={'stage':'B9A11N_9param_forward_local_sensitivity','data':'Abaqus 3Point.inp unchanged mesh/loading, healthy + 18 single-property +/- perturbations',
             'nominal':dict(zip(PARS,NOMINAL)), 'fractional_perturbation':args.fraction,
             'measurement_sigma_u_mm':args.sigma_u,'measurement_sigma_strain_ue':args.sigma_strain*1e6,
             'scale':'RMS of (F(+h)-F(-h))/(2 noise_sigma) for h=5%; response to +5% local perturbation (linear approximation)',
             'summary_by_configuration':records,
             'major_limitations':['Forward sensitivity is NOT a fitted inverse parameter importance or Sobol global sensitivity',
               'Virtual near-surface centroid LE22/LE12 are NOT measured surface-gauge strain',
               'One loading state and fixed elastic orientation; some properties may be unidentifiable',
               'Noise-whitened field components treated as independent; correlations not calibrated',
               'Full-FEM response ranking uses inaccessible all-node/all-element fields',
               'Changing the output types or loading changes the ranking',
               'Poisson reciprocity must hold; generated input updates independent nu12,nu13,nu23 only']}
    (out/'B9A11N_summary.json').write_text(json.dumps(summary,indent=2))
    print('FINISHED. All 9 properties + plots in:',str(out))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['selftest','check','run'],required=True)
    p.add_argument('--project',default='.')
    p.add_argument('--exports',default='B9A11J_exports')
    p.add_argument('--tag',default='run01')
    p.add_argument('--fraction',type=float,default=.05)
    p.add_argument('--sigma-u',type=float,default=.0002)
    p.add_argument('--sigma-strain',type=float,default=25e-6)
    args=p.parse_args()
    if args.stage=='selftest':selftest()
    else:run(args)
if __name__=='__main__':main()
