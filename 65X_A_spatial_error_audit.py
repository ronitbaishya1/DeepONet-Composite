#!/usr/bin/env python3
"""65X-A / Mac: diagnose full seven-region stress and strain errors.

Uses real 65W-H assembled_hybrid_ip.csv and assembled_hybrid_nodes.csv.
Does not run Abaqus, train models, or overwrite 65W results.

From DeepONet_Composite root:
  python3 65X_A_spatial_error_audit.py --project .

Re-running? Manually delete only results/65X_middle_diagnostics/A_spatial_errors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ('NO_OL','FE_L','NO_L','FE_C','NO_R','FE_R','NO_OR')
COMPONENTS = ('11','22','33','12','13','23')
CUTS = np.array([-9.6,-6.6,-1.8,1.8,6.6,9.6])
NO_REGIONS = {'NO_OL','NO_L','NO_R','NO_OR'}


def rel_l2(pred,ref):
    den=float(np.linalg.norm(ref))
    return float(100*np.linalg.norm(pred-ref)/den) if den>1e-12 else None


def metrics(pred,ref):
    err=np.asarray(pred,float)-np.asarray(ref,float)
    ref=np.asarray(ref,float)
    return dict(relative_l2_percent=rel_l2(pred,ref),
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(abs(err))),
                max_abs_error=float(np.max(abs(err))),
                reference_rms=float(np.sqrt(np.mean(ref**2))),
                reference_max_abs=float(np.max(abs(ref))),
                squared_error_sum=float(np.dot(err,err)))


def new_folder(p):
    if p.exists() and any(p.iterdir()):
        raise RuntimeError('Existing output protected. Manually delete only this folder to rerun:\n'+str(p))
    p.mkdir(parents=True,exist_ok=True)


def verify_tables(folder):
    ip_path=folder/'assembled_hybrid_ip.csv'
    node_path=folder/'assembled_hybrid_nodes.csv'
    for p in (ip_path,node_path):
        if not p.is_file():raise FileNotFoundError('Missing validated 65W-H output: '+str(p))
    ip=pd.read_csv(ip_path)
    nodes=pd.read_csv(node_path)
    if len(ip)!=3640 or len(nodes)!=4592:
        raise RuntimeError('Expected 3640 mechanics rows and 4592 nodes; got %d/%d'%(len(ip),len(nodes)))
    req={'X','Y','Z','Owner','ElementLabel'}
    for comp in COMPONENTS:
        for kind in ('LE','S'):
            req.update(('FEM_'+kind+comp,'Hybrid_'+kind+comp))
    if req-set(ip.columns):
        raise RuntimeError('Missing assembled mechanics fields: '+str(sorted(req-set(ip.columns))))
    if set(ip.Owner)!=set(REGIONS):
        raise RuntimeError('Expected all seven regions: '+str(set(ip.Owner)))
    for kind in ('LE','S'):
        cols=[p+kind+c for p in ('FEM_','Hybrid_') for c in COMPONENTS]
        if not np.isfinite(ip[cols].to_numpy(float)).all():
            raise RuntimeError('Nonfinite '+kind+' data')
    return ip,nodes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--near-width',type=float,default=0.6,help='Distance from internal interface (mm)')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    source=root/'results'/'hybrid_online_7region_65W_middle_validation'
    ip,nodes=verify_tables(source)
    out=root/'results'/'65X_middle_diagnostics'/'A_spatial_errors'
    new_folder(out)
    records=[]
    global_sq={}
    for kind in ('LE','S'):
        for c in COMPONENTS:
            key=kind+c
            fem=ip['FEM_'+key].to_numpy(float)
            hy=ip['Hybrid_'+key].to_numpy(float)
            global_sq[key]=float(np.sum((hy-fem)**2))
            for region in ('ALL',)+REGIONS:
                mask=(np.ones(len(ip),bool) if region=='ALL' else (ip.Owner==region).to_numpy())
                met=metrics(hy[mask],fem[mask])
                records.append(dict(region=region,component=key,points=int(mask.sum()),
                                    share_global_error_pct=100*met['squared_error_sum']/max(global_sq[key],1e-30),**met))
    table=pd.DataFrame(records)
    table.to_csv(out/'65X_A_region_component_metrics.csv',index=False)

    # Windowed spatial errors, only from actual held-out/reference fields.
    step=0.6
    xmin,xmax=float(ip.X.min()),float(ip.X.max())
    edges=np.arange(np.floor(xmin/step)*step,np.ceil(xmax/step)*step+step*1.01,step)
    x=ip.X.to_numpy(float)
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(x>=lo-1e-10)&(x<hi-1e-10)
        if not mask.any():continue
        owners=sorted(set(ip.loc[mask,'Owner']))
        for comp in ('S11','S12','S22','S33','LE11','LE12','LE22'):
            m=metrics(ip.loc[mask,'Hybrid_'+comp].to_numpy(float),ip.loc[mask,'FEM_'+comp].to_numpy(float))
            rows.append(dict(x_min_mm=float(lo),x_max_mm=float(hi),x_mid_mm=float((lo+hi)/2),
                             regions=';'.join(owners),points=int(mask.sum()),component=comp,**m))
    pd.DataFrame(rows).to_csv(out/'65X_A_x_window_errors.csv',index=False)

    # Errors versus distance to internal NO/FE boundaries.
    ip2=ip[['ElementLabel','Owner','X','Y','Z']].copy()
    ip2['distance_to_cut_mm']=np.min(np.abs(x[:,None]-CUTS[None,:]),axis=1)
    ip2['cut_zone']=np.where(ip2.distance_to_cut_mm<args.near_width,'near_interface','interior')
    for comp in ('S11','S12','S22','LE11','LE12','LE22'):
        ip2['abs_error_'+comp]=abs(ip['Hybrid_'+comp].to_numpy(float)-ip['FEM_'+comp].to_numpy(float))
    ip2.to_csv(out/'65X_A_spatial_point_errors.csv',index=False)
    cut_rows=[]
    for region in REGIONS:
        for zone in ('near_interface','interior'):
            sel=(ip2.Owner==region)&(ip2.cut_zone==zone)
            if not sel.any():continue
            for comp in ('S11','S12','S22','LE11','LE12','LE22'):
                m=metrics(ip.loc[sel,'Hybrid_'+comp].to_numpy(float),ip.loc[sel,'FEM_'+comp].to_numpy(float))
                cut_rows.append(dict(region=region,zone=zone,component=comp,points=int(sel.sum()),**m))
    pd.DataFrame(cut_rows).to_csv(out/'65X_A_interface_vs_interior.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for comp in ('S11','S12','S22','LE11','LE12','LE22'):
        subset=table[(table.component==comp)&(table.region!='ALL')]
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.bar(subset.region,subset.relative_l2_percent.astype(float))
        ax.set(title=comp+' relative L2 error by region',ylabel='Relative L2 error (%)',xlabel='Region')
        ax.grid(axis='y',alpha=.25);fig.tight_layout()
        fig.savefig(out/('65X_A_region_'+comp+'.png'),dpi=180);plt.close(fig)
    for comp in ('S22','S12','LE22','LE12'):
        err=ip2['abs_error_'+comp].to_numpy(float)
        fig,ax=plt.subplots(figsize=(10,4.5))
        h=ax.scatter(x,ip.Z.to_numpy(float),c=err,cmap='viridis',s=7,linewidths=0)
        fig.colorbar(h,ax=ax,label='|Hybrid - FEM| '+comp)
        for cut in CUTS:ax.axvline(cut,linewidth=.6,linestyle='--',color='grey')
        ax.set(xlabel='Beam x (mm)',ylabel='Z (mm)',title=comp+' absolute error at element/IP locations')
        fig.tight_layout();fig.savefig(out/('65X_A_spatial_'+comp+'.png'),dpi=180);plt.close(fig)
    summary={'step':'65X-A','full_elements':len(ip),'full_nodes':len(nodes),
             'near_interface_width_mm':args.near_width,
             'checks':'All 3640 mechanics points and seven regions provided',
             'main_errors':{key:{region:table.loc[(table.component==key)&(table.region==region),'relative_l2_percent'].iloc[0]
                                 for region in REGIONS}
                            for key in ('S11','S12','S22','LE11','LE12','LE22')},
             'note':'Large relative error in low-magnitude reference stresses needs absolute-error context. Spatial plots contain reference errors and must NOT be used to tune a final model without separate holdout data.'}
    (out/'65X_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('65X-A PASS: 3640 IP rows, 4592 nodes, 7 regions')
    print('Saved spatial errors and figures to:',out)
    print(table[(table.region=='ALL')&table.component.isin(['S11','S12','S22','LE11','LE12','LE22'])][['component','relative_l2_percent','rmse']].to_string(index=False))

if __name__=='__main__':main()
