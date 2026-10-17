#!/usr/bin/env python3
"""65AK-B: location of frozen LGNO and MLP errors in six external cases.

macOS: python3 65AK_B_spatial.py --project .
No model inference, no training; reads 65AK-A cached predictions only.
"""
import argparse, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ak=importlib.import_module('65AK_common')


def region_mask(x,xmin,xmax,width):
    d=np.minimum(np.abs(x-xmin),np.abs(x-xmax))
    return d<=width


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--interface-width-mm',default=.6,type=float)
    ap.add_argument('--bins',type=int,default=16)
    a=ap.parse_args();root=Path(a.project).resolve()
    if a.interface_width_mm<=0 or a.bins<5:raise ValueError('Use positive interface width and at least five bins')
    cache={reg:ak.load_A(root,reg) for reg in ak.REGIONS}
    for reg,z in cache.items():ak.provenance_check(root,reg,z)
    out=ak.new_output(root,'B_spatial')
    binned=[];groups=[]
    for reg,z in cache.items():
        xyz=z['ip_coordinates'].astype(np.float64)
        x=xyz[:,0];xnodes=z['coordinates'][:,0]
        xmin,xmax=float(np.min(xnodes)),float(np.max(xnodes))
        if not xmin<xmax:raise RuntimeError('Invalid X domain in '+reg)
        near=region_mask(x,xmin,xmax,a.interface_width_mm)
        if not np.any(near) or np.all(near):raise RuntimeError('Interface width gives no near or interior elements: '+reg)
        edges=np.linspace(xmin,xmax,a.bins+1)
        bin_ids=np.clip(np.searchsorted(edges,x,side='right')-1,0,a.bins-1)
        for case_i,case in enumerate(z['case_ids'].astype(str)):
            fig,axs=plt.subplots(2,1,figsize=(9.5,7.5),sharex=True)
            for component_name,cidx,idxax in [('S22',1,0),('LE22',1,1)]:
                kind='S' if component_name.startswith('S') else 'LE'
                truth=z[kind+'_ref'][case_i,:,cidx]
                for model in ('65AI_local_lgno','65AJ_matched_mlp'):
                    estimate=z[kind+'_'+model][case_i,:,cidx]
                    for tag,mask in [('near_interface',near),('interior',~near)]:
                        groups.append(dict(region=reg,case_id=case,field=component_name,
                            model=model,zone=tag,element_count=int(mask.sum()),
                            relative_L2_pct=ak.relpct(estimate[mask],truth[mask]),
                            rmse=ak.rmse(estimate[mask],truth[mask])))
                    xs=[];v=[]
                    for b in range(a.bins):
                        mask=bin_ids==b
                        if not np.any(mask):continue
                        xs.append(float(.5*(edges[b]+edges[b+1])))
                        # Absolute RMSE per bin avoids relative-error spikes at ~zero stress/strain.
                        val=ak.rmse(estimate[mask],truth[mask]);v.append(val)
                        binned.append(dict(region=reg,case_id=case,field=component_name,
                            model=model,bin_index=b,bin_x_center_mm=xs[-1],
                            element_count=int(mask.sum()),absolute_RMSE=val,
                            relative_L2_pct=ak.relpct(estimate[mask],truth[mask])))
                    axs[idxax].plot(xs,v,marker='o',markersize=3,linewidth=1.5,
                                    label='Local LGNO' if model=='65AI_local_lgno' else 'Matched MLP')
                axs[idxax].axvline(xmin+a.interface_width_mm,linestyle=':',linewidth=1.2,color='gray')
                axs[idxax].axvline(xmax-a.interface_width_mm,linestyle=':',linewidth=1.2,color='gray')
                axs[idxax].set_ylabel(component_name+' bin RMSE'+(' [MPa]' if kind=='S' else ''))
                axs[idxax].grid(alpha=.2)
                axs[idxax].legend(loc='best',fontsize=9)
            axs[-1].set_xlabel('Axial X coordinate [mm]')
            fig.suptitle(reg+' — '+case+' (boundary zones separated by dotted lines)')
            fig.tight_layout()
            fig.savefig(out/(reg+'_'+case+'_axial_error.png'),dpi=150)
            plt.close(fig)
        # Worst LGNO S22 case: visualize where spatial errors concentrate.
        ranked=[(ak.relpct(z['S_65AI_local_lgno'][j,:,1],z['S_ref'][j,:,1]),j)
                for j in range(len(z['case_ids']))]
        worst_idx=max(ranked)[1]
        case=str(z['case_ids'][worst_idx])
        fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
        for cidx,(field,comp) in enumerate([('S',1),('LE',1)]):
            for midx,model in enumerate(('65AI_local_lgno','65AJ_matched_mlp')):
                ax=axs[cidx,midx]
                abs_error=np.abs(z[field+'_'+model][worst_idx,:,comp]-z[field+'_ref'][worst_idx,:,comp])
                sc=ax.scatter(x,xyz[:,2],c=abs_error,cmap='viridis',s=12,edgecolors='none')
                fig.colorbar(sc,ax=ax,shrink=.8,label='Absolute error'+(' [MPa]' if field=='S' else ''))
                ax.set_title(field+'22: '+('LGNO' if midx==0 else 'matched MLP'))
                ax.set_xlabel('X [mm]');ax.set_ylabel('Z [mm]')
        fig.suptitle(reg+' — highest LGNO S22 error: '+case+' — X-Z integration-point projection')
        fig.tight_layout();fig.savefig(out/(reg+'_worst_case_spatial_map.png'),dpi=155);plt.close(fig)
        print('65AK-B spatial plots:',reg,flush=True)
    pd.DataFrame(groups).to_csv(out/'65AK_B_near_interface_vs_interior.csv',index=False)
    pd.DataFrame(binned).to_csv(out/'65AK_B_axial_binned_errors.csv',index=False)
    ak.atomic_summary(out/'65AK_B_summary.json',dict(step='65AK-B',regions=list(cache),
         cases_per_region=6,interface_width_mm=a.interface_width_mm,bins=a.bins,
         caveat='Spatial diagnostics of the SAME six external cases, not additional generalization evidence. X-Z plots project 3-D integration points into 2-D.'))
    print('65AK-B DONE:',out)

if __name__=='__main__':main()
