#!/usr/bin/env python3
"""Step 65AM-B5: one command for paper-ready error and field plots.

Reads existing B5 CSVs, original 64J FEM arrays and trained B5 checkpoints.
No retraining or changes to old checkpoints/65V solver.

Scope: two NO subregions, 12 INTERNAL development cases. No unseen validation.
Displacement U2: B5 FE-residual model.
Logarithmic strain LE22/LE12: derived from B5 displacement with existing C3D8R routine.
Stress S22: SEPARATE frozen 65AI LGNO or 65AA model (NOT predicted by B5).
Full seven-region beam profiles require supplied aligned beamline CSV; there is
NO honest way to infer missing FEM-region results from B5's NO-only files.

  python3 plot_all_65AM_B5.py --project . --check-only
  python3 plot_all_65AM_B5.py --project .
  python3 plot_all_65AM_B5.py --project . --case-index 17 --stress 65AA
  python3 plot_all_65AM_B5.py --project . --fullbeam-csv beamline_compare.csv
  python3 plot_all_65AM_B5.py --project . --summary-only
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REGIONS = ('inner_left','inner_right')
BASE = '65AA_frozen'
CANDIDATES = ('centered_grad','standardized_grad','centered_U','standardized_U')
LABELS={'65AA_frozen':'65AA baseline','centered_grad':'B5 centered + gradient',
        'standardized_grad':'B5 standardized + gradient',
        'centered_U':'B5 centered + displacement',
        'standardized_U':'B5 standardized + displacement'}
COLORS={'65AA_frozen':'#6e6e6e','centered_grad':'#0b77bc',
        'standardized_grad':'#df7422','centered_U':'#2f9c72','standardized_U':'#8a59b9'}
TITLES={'U_vec_rel_pct':'Displacement U vector','U2_rel_pct':'Vertical displacement U2',
        'grad_rel_pct':'Displacement gradient ∇U',
        'LE22_rel_pct':'Logarithmic strain LE22 (from U)',
        'LE12_rel_pct':'Logarithmic strain LE12 (from U)',
        'dU2dY_rel_pct':'∂U2 / ∂Y','dU1dY_rel_pct':'∂U1 / ∂Y'}


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=210,bbox_inches='tight')
    plt.close(fig)
    print('  ',path.name,flush=True)


def relerr(a,b):
    return 100*np.linalg.norm((np.asarray(a)-np.asarray(b)).ravel())/max(np.linalg.norm(np.asarray(b).ravel()),1e-14)


def setup_style():
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,
          'figure.titlesize':12,'savefig.facecolor':'white','axes.grid':False})


def summary_figures(root,out):
    src=root/'results'/'65AM_B5_coeff_normalized'/'B5_comparison'
    fp=src/'B5_pooled_metrics.csv';fz=src/'B5_zone_metrics.csv';fc=src/'B5_case_metrics.csv'
    for p in [fp,fz,fc]:
        if not p.is_file():raise FileNotFoundError('Run B5 comparison first: '+str(p))
    pooled=pd.read_csv(fp);zones=pd.read_csv(fz);cases=pd.read_csv(fc)
    needed=set((r,m) for r in REGIONS for m in (BASE,'centered_grad'))
    found=set(zip(pooled.region,pooled.model))
    if not needed.issubset(found):raise RuntimeError('B5 pooled report missing required rows: '+str(needed-found))
    available=[m for m in (BASE,'centered_grad','standardized_grad','centered_U','standardized_U')
               if set(REGIONS).issubset(set(pooled[pooled.model==m].region))]
    metric_cols=['U_vec_rel_pct','U2_rel_pct','grad_rel_pct','LE22_rel_pct','LE12_rel_pct',
                 'dU2dY_rel_pct','dU1dY_rel_pct']
    for metric in metric_cols:
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
        for ax,reg in zip(axes,REGIONS):
            sub=pooled[pooled.region==reg].set_index('model')
            models=[m for m in available if m in sub.index]
            y=[float(sub.loc[m,metric]) for m in models]
            xx=np.arange(len(models))
            ax.bar(xx,y,color=[COLORS[m] for m in models],width=.7)
            ax.set_xticks(xx,[LABELS[m] for m in models],rotation=38,ha='right',fontsize=8)
            ax.set_title(reg.replace('_',' ').title())
            ax.grid(axis='y',alpha=.25)
            ax.set_ylabel('Relative L2 error (%)' if reg==REGIONS[0] else '')
        fig.suptitle(TITLES[metric]+' — internal development (12 cases)')
        save(fig,out/('summary_'+metric+'.png'))

    # Zone charts (geometric left/right + interior; do not relabel as FE-only regions)
    for metric in ['grad_rel_pct','dU2dY_rel_pct']:
        fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
        for ax,reg in zip(axes,REGIONS):
            xs=['left','interior','right'];width=.38
            for i,model in enumerate((BASE,'centered_grad')):
                y=[]
                for zone in xs:
                    sub=zones[(zones.region==reg)&(zones.model==model)&(zones.zone==zone)]
                    y.append(float(sub.iloc[0][metric]) if len(sub) else np.nan)
                ax.bar(np.arange(3)+(i-.5)*width,y,width,label=LABELS[model],color=COLORS[model])
            ax.set_xticks(range(3),['Left edge','Interior','Right edge'])
            ax.set_title(reg.replace('_',' ').title())
            ax.set_ylabel('Relative L2 error (%)' if reg==REGIONS[0] else '')
            ax.grid(axis='y',alpha=.25)
        axes[1].legend(fontsize=8)
        fig.suptitle('NO subregion error zones — '+TITLES[metric])
        save(fig,out/('zones_'+metric+'.png'))

    # per-case differences: case indices are internal, not independent external validation
    for metric in ['U_vec_rel_pct','grad_rel_pct','LE22_rel_pct']:
        fig,axes=plt.subplots(1,2,figsize=(11,4.0))
        for ax,reg in zip(axes,REGIONS):
            for model in (BASE,'centered_grad'):
                sub=cases[(cases.region==reg)&(cases.model==model)].sort_values('case_index')
                ax.plot(sub['case_index'],sub[metric],marker='o',ms=3,lw=1.3,
                        label=LABELS[model],color=COLORS[model])
            ax.set_title(reg.replace('_',' ').title())
            ax.set_xlabel('Original 64J case index (internal tune)')
            ax.set_ylabel('Relative L2 error (%)')
            ax.grid(alpha=.25)
        axes[-1].legend(fontsize=8)
        fig.suptitle(TITLES[metric]+' — by development case')
        save(fig,out/('cases_'+metric+'.png'))
    return pooled


def safe_get_65ai(root,region,case_idx,dev):
    """Optional 65AI stress; never substitute B5 displacement into frozen 65AI features."""
    import importlib
    ai=importlib.import_module('65AI_common')
    st=ai.member_source(root,region,1,dev,True)
    net,_=ai.load_candidate(root,'local_lgno',st,dev)
    feats=ai.base_features(st,np.array([case_idx],dtype=int),dev)
    ls,ss=ai.train_scales(st['ds'])
    _,sp=ai.predict(net,feats,ls,ss,dev)
    return sp[0].astype(np.float64),'65AI LGNO (uses frozen 65AA U)'


def get_65aa_stress(root,region,ds,case_idx,dev):
    import torch
    import importlib
    b=importlib.import_module('65AM_B_common')
    aa=b.AA
    ckfile=root/'results'/'65AA_gradient_learning'/'B_gradient'/region/'member_01'/'best_v4_7region.pt'
    ck=aa.co.load_checkpoint(ckfile)
    model=aa.co.make_model(ck,dev).eval()
    scaled,_,ip=aa.co.scaled_inputs(ck,ds)
    branch=torch.as_tensor(np.array(scaled[[case_idx]],copy=True),dtype=torch.float32,device=dev)
    ips=torch.as_tensor(np.array(ip,copy=True),dtype=torch.float32,device=dev)
    with torch.no_grad():
        norm=model.stress_operator(branch,ips).detach().cpu().numpy()[0]
    pred=norm*np.asarray(ck['S_std'],dtype=float)+np.asarray(ck['S_mean'],dtype=float)
    if pred.shape != ds['S'][case_idx].shape or not np.isfinite(pred).all():
        raise RuntimeError('Frozen 65AA stress output shape or values invalid')
    return pred,'65AA frozen stress (NOT B5)'


def yz_project(xyz,vals):
    """Mean over Y at identical X,Z integration-point locations (avoids hidden overplot)."""
    xyz=np.asarray(xyz,dtype=float);vals=np.asarray(vals,dtype=float)
    if len(xyz)!=len(vals):raise ValueError('Point/field size mismatch')
    key=np.round(xyz[:,[0,2]],6)
    uniq,idx=np.unique(key,axis=0,return_inverse=True)
    counts=np.bincount(idx)
    mean=np.bincount(idx,weights=vals)/counts
    return uniq[:,0],uniq[:,1],mean


def map_3panel(xyz,ref,pred,field,source,reg,out):
    xr,zr,rf=yz_project(xyz,ref);_,_,pf=yz_project(xyz,pred)
    if not np.isfinite(rf).all() or not np.isfinite(pf).all():raise RuntimeError('Invalid field map '+field)
    lim=[min(rf.min(),pf.min()),max(rf.max(),pf.max())]
    if lim[0]==lim[1]:lim[1]=lim[0]+1e-12
    errs=np.abs(pf-rf)
    fig,ax=plt.subplots(1,3,figsize=(12.8,3.9),sharex=True,sharey=True)
    for a,v,title in zip(ax,[rf,pf,errs],['Full FEM',source,'Absolute error']):
        m=a.scatter(xr,zr,c=v,cmap='viridis' if title!='Absolute error' else 'magma',
                    s=65,marker='s',vmin=lim[0] if title!='Absolute error' else 0,
                    vmax=lim[1] if title!='Absolute error' else max(errs.max(),1e-12))
        fig.colorbar(m,ax=a,shrink=.82)
        a.set_title(title,fontsize=9);a.set_xlabel('X (mm)')
    ax[0].set_ylabel('Z (mm)')
    fig.suptitle(f'{reg} | {field} | Y-averaged X–Z field | rel L2 {relerr(pred,ref):.3g}%')
    save(fig,out/f'{reg}_{field}_XZ_fields.png')


def axial_profiles(xyz,ref,pred):
    # Representative x-line: closest Y,Z integration-point coordinates to cross-section center.
    xyz=np.asarray(xyz,float);ref=np.asarray(ref,float);pred=np.asarray(pred,float)
    if len(xyz)!=len(ref) or len(ref)!=len(pred):raise ValueError('Unexpected field shape')
    y0,z0=np.median(xyz[:,1]),np.median(xyz[:,2])
    ux=np.unique(np.round(xyz[:,0],6))
    xlist=[];fr=[];pr=[]
    for x in ux:
        ids=np.flatnonzero(np.abs(xyz[:,0]-x)<2e-5)
        if not len(ids):continue
        distance=(xyz[ids,1]-y0)**2+(xyz[ids,2]-z0)**2
        best=ids[np.argmin(distance)]
        xlist.append(x);fr.append(ref[best]);pr.append(pred[best])
    return np.asarray(xlist),np.asarray(fr),np.asarray(pr)


def make_local_fields(root,out,case_idx,stress_method,variant,dev):
    import importlib
    c=importlib.import_module('65AM_B5_common')
    b=c.b
    profile={field:{} for field in ('U2','LE22','S22')}
    records=[];names={}
    for reg in REGIONS:
        d,mu,sd=c.prepare(root,reg)
        if case_idx not in set(int(v) for v in d['tune']):
            raise RuntimeError(f'Case {case_idx} not in internal 12-case tune split for {reg}. '
                               'Choose one from printed check-only intersection.')
        net,_=c.load_model(root,variant.split('_')[0],variant.split('_')[1],reg,d,mu,sd,dev)
        pred,_=c.predict(net,d,np.asarray([case_idx]),dev)
        up=pred[0];ur=d['ds']['U'][case_idx];ds=d['ds']
        le_pred=b.CO.log_material_strain_eng(up,ds['element_node_indices'],ds['B_center'])
        le_fem=ds['LE'][case_idx]
        if stress_method in ('auto','65AI'):
            try:
                stress,stname=safe_get_65ai(root,reg,case_idx,dev)
            except (ImportError,FileNotFoundError,RuntimeError,KeyError,ValueError) as exc:
                if stress_method=='65AI':raise RuntimeError('Requested 65AI model unavailable: '+str(exc)) from exc
                print('  65AI unavailable for',reg,'— using frozen 65AA stress:',str(exc)[:160])
                stress,stname=get_65aa_stress(root,reg,ds,case_idx,dev)
        elif stress_method=='65AA':
            stress,stname=get_65aa_stress(root,reg,ds,case_idx,dev)
        else:raise ValueError(stress_method)
        names[reg]=stname
        # Match U2 sampling to stress/strain at element centers (mean 8 nodal U per C3D8R element).
        conn=ds['element_node_indices']
        u2_fem=ur[conn,1].mean(axis=1)
        u2_pred=up[conn,1].mean(axis=1)
        fields={
            'U2':(u2_fem,u2_pred,'B5 '+variant+' U2 (mm)'),
            'LE22':(le_fem[:,1],le_pred[:,1],f'B5 {variant} LE22 from U'),
            'S22':(ds['S'][case_idx,:,1],stress[:,1],stname+' S22 (MPa)')}
        coords=ds['ip_coordinates']
        for field,(f,p,label) in fields.items():
            map_3panel(coords,f,p,field,label,reg,out)
            xx,ff,pp=axial_profiles(coords,f,p)
            profile[field][reg]=(xx,ff,pp,label)
            for x,fr,pr in zip(xx,ff,pp):
                records.append(dict(region=reg,x_mm=float(x),field=field,FEM=float(fr),NO=float(pr),
                                    absolute_error=abs(float(pr-fr)),prediction_source=label,
                                    cohort='original64J TRAIN internal 12-case tune',case_index=case_idx))
        print(f'  {reg}: U2 {relerr(u2_pred,u2_fem):.3g}% | LE22 {relerr(le_pred[:,1],le_fem[:,1]):.3g}% '
              f'| S22 {relerr(stress[:,1],ds["S"][case_idx,:,1]):.3g}% [{stname}]')

    pd.DataFrame(records).to_csv(out/f'NO_sections_axial_data_case_{case_idx}.csv',index=False)
    for kind in ('comparison','absolute_error'):
        fig,axes=plt.subplots(1,3,figsize=(14.5,4.4))
        for ax,field in zip(axes,['U2','LE22','S22']):
            for reg in REGIONS:
                xx,ff,pp,_=profile[field][reg]
                if kind=='comparison':
                    ax.plot(xx,ff,'o-',color='#555555',ms=4,lw=1.6,label='FEM' if reg==REGIONS[0] else None)
                    ax.plot(xx,pp,'--',color=COLORS['centered_grad'],lw=1.8,
                            label='NO prediction' if reg==REGIONS[0] else None)
                else:
                    ax.plot(xx,abs(ff-pp),'o-',ms=4,lw=1.5,
                            label=reg.replace('_',' ').title())
                ax.axvspan(xx.min(),xx.max(),alpha=.055,color='#3488bb')
            ax.grid(alpha=.2)
            ax.set_xlabel('Beam X (mm)')
            ax.set_title(field+' (S22 source: '+('/'.join(sorted(set(names.values()))))[:22]+'...)' if field=='S22' else field)
            ax.set_ylabel({'U2':'U2 (mm)','LE22':'LE22 (dimensionless)','S22':'S22 (MPa)'}[field]
                          if kind=='comparison' else 'Absolute error')
            ax.legend(fontsize=8)
        fig.suptitle(('FEM vs NO' if kind=='comparison' else 'FEM–NO absolute error')+
            f' | NO sections only | case {case_idx} | FEM intervals NOT plotted')
        save(fig,out/('NO_sections_3panel_'+kind+'.png'))
    return names


def optional_fullbeam(csv_path,out):
    if csv_path is None:
        print('  No whole-beam CSV supplied: FEM-only regions intentionally left unplotted.')
        return
    df=pd.read_csv(csv_path)
    names=('U2','LE22','S22')
    must={'x_mm'} | {f'{model}_{field}' for field in names for model in ('FEM','NO')}
    if not must.issubset(df.columns):
        raise ValueError('Full-beam CSV must have columns '+', '.join(sorted(must))+
                         '. Optional column: section (FEM/NO).')
    df=df.sort_values('x_mm')
    for field in names:
        if df[[f'FEM_{field}',f'NO_{field}']].isna().all(axis=0).any():
            raise ValueError(f'Fullbeam {field} values missing in FEM or NO columns')
    # Connected lines MUST NOT join across disjoint/omitted x sections.
    section = df['section'].astype(str) if 'section' in df else pd.Series(['unspecified']*len(df),index=df.index)
    xx=df['x_mm'].to_numpy(float)
    maxgap=3*np.median(np.diff(np.unique(xx))) if len(np.unique(xx))>2 else np.inf
    if maxgap<=0: maxgap=np.inf
    gap=np.r_[False,np.diff(xx)>maxgap]
    segment=(section!=section.shift()).to_numpy(dtype=bool)|gap
    groups=np.cumsum(segment)
    for mode in ('comparison','absolute_error'):
        fig,axes=plt.subplots(1,3,figsize=(14.4,4.7))
        for ax,field in zip(axes,names):
            first=True
            for _,piece in df.groupby(groups,sort=True):
                x=piece.x_mm.to_numpy(float)
                f=piece[f'FEM_{field}'].to_numpy(float);p=piece[f'NO_{field}'].to_numpy(float)
                if mode=='comparison':
                    ax.plot(x,f,'o-',ms=3,color='#555555',label='FEM' if first else None)
                    ax.plot(x,p,'--',lw=1.7,color=COLORS['centered_grad'],label='Hybrid/NO' if first else None)
                else: ax.plot(x,np.abs(f-p),'o-',ms=3,lw=1.4,
                              color=COLORS['centered_grad'],label='Absolute error' if first else None)
                if 'section' in piece.columns:
                    col='#aaa' if str(piece.section.iloc[0]).lower().startswith('fem') else '#228bc2'
                    ax.axvspan(min(x),max(x),alpha=.10,color=col)
                first=False
            ax.grid(alpha=.22);ax.set_xlabel('X (mm)');ax.set_title(field)
            ax.set_ylabel('Absolute error' if mode=='absolute_error' else
                          {'U2':'U2 (mm)','LE22':'LE22','S22':'S22 (MPa)'}[field])
            ax.legend(fontsize=8)
        fig.suptitle('Full-beam '+mode.replace('_',' ')+' | supplied aligned data')
        save(fig,out/('whole_beam_3panel_'+mode+'.png'))


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project',default='.')
    ap.add_argument('--case-index',type=int,default=None,help='Original case index in both B5 tune splits')
    ap.add_argument('--variant',choices=CANDIDATES,default='centered_grad')
    ap.add_argument('--stress',choices=['auto','65AI','65AA'],default='auto')
    ap.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    ap.add_argument('--summary-only',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--fullbeam-csv',default=None,help='Optional actual seven-region aligned beamline CSV')
    ap.add_argument('--out',default=None,help='Optional output folder')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    sys.path.insert(0,str(root))
    setup_style()
    out=Path(args.out).expanduser().resolve() if args.out else root/'results'/'65AM_B5_coeff_normalized'/'B5_paper_plots'
    if args.summary_only and args.check_only:
        print('Summary-only: will load B5 CSV reports; no models required.')
    case_idx=args.case_index
    if not args.summary_only:
        import importlib
        b=importlib.import_module('65AM_B_common')
        c=importlib.import_module('65AM_B5_common')
        tuning=[]
        for region in REGIONS:
            d,mu,sd=c.prepare(root,region)
            tuning.append(set(map(int,d['tune'])))
            print(region,'available internal tune indices:',sorted(tuning[-1]))
            dev=b.CO.device_from_option(args.device)
            c.load_model(root,args.variant.split('_')[0],args.variant.split('_')[1],region,d,mu,sd,dev)
        common=sorted(tuning[0] & tuning[1])
        if not common:raise RuntimeError('Left and right internal tune splits do not overlap; cannot combine case plot')
        if case_idx is None:case_idx=common[0]
        if case_idx not in common:raise ValueError('Case index must be in both internal tune sets: '+str(common))
        print('Selected original case index:',case_idx)
    if args.check_only:
        print('CHECK PASS: required B5 checkpoints/bases were found; no plots written.')
        return
    out.mkdir(parents=True,exist_ok=True)
    print('Saving figures to:',out)
    summary_figures(root,out)
    if not args.summary_only:
        make_local_fields(root,out,case_idx,args.stress,args.variant,dev)
    optional_fullbeam(Path(args.fullbeam_csv).expanduser().resolve() if args.fullbeam_csv else None,out)
    (out/'README_PLOTS.txt').write_text(
       'Plots from original 64J TRAIN internal 12-case development only. NOT independent tests.\n'
       'U2 = B5 FE residual displacement; LE22 = FE-derived from same B5 U.\n'
       'S22 = separate frozen 65AI LGNO if available, otherwise frozen 65AA stress head.\n'
       'Do NOT call S22 a B5 stress prediction.\n'
       'NO_sections_* includes only two NO regions; missing FEM-region values NOT filled.\n'
       'X-Z field maps are Y-averaged; they are not full three-dimensional visualizations.\n'
       'Whole beam plots are generated only with a supplied aligned fullbeam CSV.\n')
    print('PLOTS COMPLETE')

if __name__=='__main__':
    main()
