#!/usr/bin/env python3
"""65AM-B: X-Z plots aggregated over Y + axial RMSE profiles.
Reads only 65AM-A results. No models or dataset writes.
python3 65AM_B_plots.py --project .
"""
import argparse, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
am=importlib.import_module('65AM_common')


def plot_case(reg,z,k,out):
    cid=str(z['case_ids'][k]);pts=z['ip_coordinates']
    true=z['grad_ref'][k,:,1,1]
    raw=z['grad_raw'][k,:,1,1]
    hard=z['grad_hard'][k,:,1,1]
    pca=z['grad_pca_only'][k,:,1,1]
    pair,ref,nlayers=am.xz_reduce(pts,true)
    vals=[ref,
          am.xz_reduce(pts,raw)[1],
          am.xz_reduce(pts,hard)[1],
          am.xz_reduce(pts,raw-true,absolute_rms=True)[1],
          am.xz_reduce(pts,hard-true,absolute_rms=True)[1],
          am.xz_reduce(pts,pca-true,absolute_rms=True)[1]]
    if np.any(nlayers<=0):raise RuntimeError('Empty projected XZ group')
    common_max=max(np.max(np.abs(v)) for v in vals[:3]);common_max=max(common_max,1e-10)
    error_max=max(np.max(v) for v in vals[3:]);error_max=max(error_max,1e-10)
    titles=['FEM: mean dU2/dY across Y','Raw NO: mean dU2/dY across Y',
            'Hard-corrected NO: mean dU2/dY across Y',
            'Raw NO error: RMS across Y','Hard-corrected error: RMS across Y',
            'PCA-only FEM error: RMS across Y']
    fig,axs=plt.subplots(2,3,figsize=(16.0,8.0),constrained_layout=True)
    for j,(ax,data,title) in enumerate(zip(axs.flat,vals,titles)):
        if j<3:
            sc=ax.scatter(pair[:,0],pair[:,1],c=data,vmin=-common_max,vmax=common_max,s=48)
        else:
            sc=ax.scatter(pair[:,0],pair[:,1],c=data,vmin=0.,vmax=error_max,s=48)
        ax.set_title(title,fontsize=10);ax.set_xlabel('X (mm)');ax.set_ylabel('Z (mm)')
        ax.set_aspect('auto');fig.colorbar(sc,ax=ax,shrink=.75)
    fig.suptitle(reg+' | '+cid+' | U2/Y (element integration points, Y layers aggregated)',fontsize=14)
    fig.savefig(out/(reg+'_'+cid+'_six_panel_XZ.png'),dpi=165)
    plt.close(fig)


def plot_profile(reg,df,out):
    d=df[df.region==reg]
    s=d.groupby('x_mm',sort=True)[['U2_dY_raw_RMSE','U2_dY_hard_RMSE','U2_dY_pca_only_RMSE']].mean()
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(s.index,s['U2_dY_raw_RMSE'],marker='o',label='Raw NO')
    ax.plot(s.index,s['U2_dY_hard_RMSE'],marker='o',label='After Hermite')
    ax.plot(s.index,s['U2_dY_pca_only_RMSE'],marker='o',label='PCA-only control')
    ax.set(title=reg+': mean dU2/dY RMSE versus axial X (six cases)',xlabel='X (mm)',ylabel='Gradient RMSE (dimensionless)')
    ax.legend();fig.tight_layout()
    fig.savefig(out/(reg+'_U2_dY_axial_RMSE.png'),dpi=165)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    args=parser.parse_args()
    root=Path(args.project).expanduser().resolve()
    prior=root/'results'/am.OUTROOT/'A_attribution'
    for name in ('65AM_A_case_metrics.csv','65AM_A_x_profiles.csv'):
        if not (prior/name).is_file():raise FileNotFoundError('Run 65AM-A before plots: '+str(prior/name))
    frames={}
    for reg in am.REGIONS:
        p=prior/(reg+'_gradient_decomposition.npz')
        if not p.is_file():raise FileNotFoundError(p)
        with np.load(p,allow_pickle=False) as z:
            frames[reg]={k:z[k] for k in z.files}
    cases=pd.read_csv(prior/'65AM_A_case_metrics.csv')
    prof=pd.read_csv(prior/'65AM_A_x_profiles.csv')
    out=am.output(root,'B_plots')
    names=[]
    for reg in am.REGIONS:
        zz=frames[reg]
        local=cases[cases.region==reg].sort_values('grad_hard_error_pct',ascending=False)
        worst=str(local.iloc[0].case_id)
        requested={'AJ_new_02',worst}
        for target in sorted(requested):
            idx=np.flatnonzero(zz['case_ids'].astype(str)==target)
            if len(idx)!=1:
                if target=='AJ_new_02':continue
                raise RuntimeError('Missing cached case '+target)
            plot_case(reg,zz,int(idx[0]),out)
            names.append(reg+'_'+target+'_six_panel_XZ.png')
        plot_profile(reg,prof,out)
        print('65AM-B',reg,'| worst case',worst,'| plots generated',flush=True)
    am.saved_json(out/'65AM_B_summary.json',dict(step='65AM-B',
        aggregation='Mean dU2/dY across repeated Y layers in top row; RMS of signed error across Y in bottom row',
        shared_color_limits=True,source='65AM-A frozen decomposition',files=names))
    print('65AM-B DONE:',out)

if __name__=='__main__':main()
