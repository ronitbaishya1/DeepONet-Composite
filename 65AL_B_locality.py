#!/usr/bin/env python3
"""65AL-B: local nodal error variation and worst-case 3D derivative plots.

Uses frozen 65AK U_ref/U_pred and 65AL-A gradient arrays; DOES NOT SMOOTH
or change the displacements. Nodal-neighbor fluctuation is a descriptive
mesh-local roughness proxy, NOT a Fourier spectrum or a physical wave number.
"""
import argparse, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
al=importlib.import_module('65AL_common')

def graph_edges(conn):
    # Mesh graph formed by pairs of corners in each hexahedron; no kNN distance.
    edges=[]
    for i in range(8):
        for j in range(i+1,8):
            ii=np.minimum(conn[:,i],conn[:,j]);jj=np.maximum(conn[:,i],conn[:,j])
            edges.append(np.stack((ii,jj),axis=-1))
    return np.unique(np.concatenate(edges,axis=0),axis=0)

def local_fluctuation(u_error,edges,n):
    count=np.zeros(n,dtype=np.float64)
    accum=np.zeros_like(u_error,dtype=np.float64)
    s,t=edges[:,0],edges[:,1]
    np.add.at(accum,s,u_error[t]);np.add.at(accum,t,u_error[s])
    np.add.at(count,s,1);np.add.at(count,t,1)
    if (count==0).any():raise RuntimeError('Some mesh nodes lack any neighbors')
    return u_error-accum/count[:,None]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    a=p.parse_args();root=Path(a.project).resolve()
    ia=root/'results'/al.BASE/'A_gradient'
    if not (ia/'65AL_A_case_gradient_errors.csv').is_file():raise FileNotFoundError('Run 65AL_A_gradient.py first')
    out=al.outdir(root,'B_locality')
    table=[]
    for reg in al.REGIONS:
        z,ds,conn,dn,aff=al.load(root,reg)
        pz=ia/(reg+'_gradient_fields.npz')
        if not pz.is_file():raise FileNotFoundError(pz)
        with np.load(pz,allow_pickle=False) as f:cache={k:f[k] for k in f.files}
        if list(cache['case_ids'].astype(str))!=list(z['case_ids'].astype(str)):raise RuntimeError('Case ordering changed')
        gr=cache['grad_ref'].astype(float); gp=cache['grad_pred'].astype(float)
        edge=graph_edges(conn)
        geo=np.asarray(z['ip_coordinates'],float)
        for i,case in enumerate(z['case_ids'].astype(str)):
            ue=np.asarray(z['U_pred'][i]-z['U_ref'][i],float)
            flap=local_fluctuation(ue,edge,len(z['coordinates']))
            gg=np.sum((gp[i]-gr[i])**2,axis=(1,2))
            f_local=np.sqrt(np.mean(np.sum(flap**2,axis=1)))
            f_global=np.sqrt(np.mean(np.sum(ue**2,axis=1)))
            cell_rough=np.mean(np.linalg.norm(flap[conn],axis=-1),axis=1)
            corr=float(np.corrcoef(cell_rough,np.sqrt(gg))[0,1]) if np.std(cell_rough)>1e-14 and np.std(gg)>1e-14 else float('nan')
            table.append(dict(region=reg,case_id=case,
                RMS_vector_displacement_error_mm=f_global,
                RMS_vector_local_neighbor_fluctuation_mm=f_local,
                neighbor_fluctuation_to_total_error_ratio=f_local/max(f_global,1e-14),
                cell_fluctuation_vs_gradient_error_Pearson=corr,
                graph_nodes=len(z['coordinates']),graph_edges=len(edge)))
        # Make one plot for the worst field-gradient case in each region. No model selection.
        gerrs=np.linalg.norm(gp-gr,axis=(2,3));gref=np.linalg.norm(gr,axis=(2,3));
        scores=[al.relpct(gp[i],gr[i]) for i in range(len(z['case_ids']))]
        j=int(np.argmax(scores));case=str(z['case_ids'][j]);
        fig,ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        charts=[(gr[j,:,1,1],r'FEM $\partial U_2/\partial Y$'),
                (gp[j,:,1,1],r'Predicted $\partial U_2/\partial Y$'),
                (np.abs(gp[j,:,1,1]-gr[j,:,1,1]),r'$|\Delta(\partial U_2/\partial Y)|$'),
                (gerrs[j],r'$||\nabla U_{pred}-\nabla U_{FEM}||_F$')]
        for axis,(vec,title) in zip(ax.reshape(-1),charts):
            h=axis.scatter(geo[:,0],geo[:,2],c=vec,s=15,cmap='viridis')
            fig.colorbar(h,ax=axis,shrink=.74)
            axis.set_title(title);axis.set_xlabel('Axial X [mm]');axis.set_ylabel('Z [mm]')
        fig.suptitle(reg+' | '+case+' | worst relative gradient error | X-Z projected integration points')
        fig.savefig(out/(reg+'_worst_case_gradient_XZ.png'),dpi=160);plt.close(fig)
        fig,axis=plt.subplots(figsize=(8.5,5))
        names=z['case_ids'].astype(str).tolist()
        ar=[r for r in table if r['region']==reg]
        axis.bar(np.arange(len(ar)),[r['neighbor_fluctuation_to_total_error_ratio'] for r in ar])
        axis.set_xticks(np.arange(len(ar)),names,rotation=20)
        axis.set_ylabel('Local neighbor fluctuation / total U error')
        axis.set_title(reg+': fixed mesh graph roughness proxy (no filtering or training)')
        fig.tight_layout();fig.savefig(out/(reg+'_nodal_error_roughness.png'),dpi=155);plt.close(fig)
        print('65AL-B DONE:',reg,flush=True)
    df=pd.DataFrame(table)
    df.to_csv(out/'65AL_B_local_error_roughness.csv',index=False)
    al.summary(out/'65AL_B_summary.json',dict(step='65AL-B',cases_per_region=6,
        graph='nodes connected if within same existing C3D8R element (full eight-corner graph)',
        measure='nodal error minus average neighboring nodal error',
        limit='This is a local fluctuation proxy, not an FFT or directly measurable strain. Values may exceed 1.',
        no_filtering=True,no_retraining=True))

if __name__=='__main__':main()
