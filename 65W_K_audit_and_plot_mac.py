#!/usr/bin/env python3
"""Step 65W-K: audit and plot *measured* 65W-H assembled seven-region results.

Run from the Mac DeepONet-Composite root after 65W_H_reconstruct_on_mac.py.
No Abaqus, retraining, or inference. Only reads real 65W-H artifacts.

python3 65W_K_audit_and_plot_mac.py --project .
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ('NO_OL','FE_L','NO_L','FE_C','NO_R','FE_R','NO_OR')
EXPECTED = 4592


def relative_l2(pred, ref):
    den = np.linalg.norm(ref.ravel())
    return float(100 * np.linalg.norm((pred-ref).ravel()) / max(den,1e-14))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    folder=root/'results'/'hybrid_online_7region_65W_middle_validation'
    for name in ('assembled_hybrid_nodes.csv','region_displacement_metrics.csv',
                 'global_displacement_metrics.csv','ip_matching_audit.json',
                 '65W_H_validation_summary.json'):
        if not (folder/name).is_file():
            raise FileNotFoundError('Missing 65W-H output: '+str(folder/name))
    nodes=pd.read_csv(folder/'assembled_hybrid_nodes.csv')
    required={'NodeLabel','X','Y','Z','Owner','FEM_U1','FEM_U2','FEM_U3',
              'Hybrid_U1','Hybrid_U2','Hybrid_U3'}
    if not required.issubset(nodes.columns):
        raise RuntimeError('Incorrect assembled CSV fields: '+str(sorted(required-set(nodes.columns))))
    if len(nodes)!=EXPECTED or nodes['NodeLabel'].nunique()!=EXPECTED:
        raise RuntimeError('Need exactly 4592 unique full FEM specimen nodes, got %d rows'%len(nodes))
    if set(nodes['Owner'].unique()) != set(REGIONS):
        raise RuntimeError('Expected 7 owner regions, got '+str(sorted(nodes['Owner'].unique())))
    xyz=nodes[['X','Y','Z']].to_numpy(dtype=float)
    FEM=nodes[['FEM_U1','FEM_U2','FEM_U3']].to_numpy(dtype=float)
    HY=nodes[['Hybrid_U1','Hybrid_U2','Hybrid_U3']].to_numpy(dtype=float)
    if not all(np.all(np.isfinite(x)) for x in (xyz,FEM,HY)):
        raise RuntimeError('Non-finite displacement or coordinate data')
    with (folder/'ip_matching_audit.json').open() as f:ip=json.load(f)
    if not ip.get('one_to_one') or float(ip.get('maximum_coordinate_distance_mm',float('inf'))) > 1e-4:
        raise RuntimeError('IP matching audit does not pass strict criteria')
    with (folder/'65W_H_validation_summary.json').open() as f:upstream=json.load(f)
    if not np.allclose(upstream['material_MPa'],[46500.,12500.,4750.],rtol=0,atol=1e-6):
        raise RuntimeError('Wrong material in 65W-H validation summary')
    reports=[]
    for region in REGIONS:
        mask=(nodes['Owner']==region).to_numpy()
        u=HY[mask]; v=FEM[mask]
        reports.append({'region':region,'nodes':int(mask.sum()),
                        'vector_rel_l2_percent':relative_l2(u,v),
                        'U2_rel_l2_percent':relative_l2(u[:,1],v[:,1]),
                        'U2_rmse_mm':float(np.sqrt(np.mean((u[:,1]-v[:,1])**2)))})
    overall={'full_nodes':EXPECTED,'material_MPa':[46500.,12500.,4750.],
             'vector_rel_l2_percent':relative_l2(HY,FEM),
             'U2_rel_l2_percent':relative_l2(HY[:,1],FEM[:,1]),
             'U2_rmse_mm':float(np.sqrt(np.mean((HY[:,1]-FEM[:,1])**2))),
             'regions':reports, 'ip_matching_audit':ip,
             'note':'This audits actual seven-region predictions and node coverage; physical material/loading equivalence and reactions must still be verified separately.'}
    figure_folder=folder/'65W_K_figures'
    if figure_folder.exists() and any(figure_folder.iterdir()):
        raise RuntimeError('Existing figure folder protected: '+str(figure_folder))
    figure_folder.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5))
    # One representative centerline point per x-plane, taken from actual mesh.
    mid_y=(float(xyz[:,1].min())+float(xyz[:,1].max()))/2
    mid_z=(float(xyz[:,2].min())+float(xyz[:,2].max()))/2
    rep=nodes.assign(_plane=np.round(nodes['X'].to_numpy(dtype=float),5),
                     _radius=(nodes['Y'].to_numpy(dtype=float)-mid_y)**2+
                             (nodes['Z'].to_numpy(dtype=float)-mid_z)**2)
    rep=rep.sort_values('_radius').drop_duplicates('_plane').sort_values('_plane')
    ax.plot(rep['X'],rep['FEM_U2'],'.-',label='Full FEM',markersize=4)
    ax.plot(rep['X'],rep['Hybrid_U2'],'o--',label='Seven-region FE–NO',markersize=3)
    ax.set(xlabel='Beam x (mm)',ylabel='U2 (mm)',title='Full FEM vs seven-region hybrid — representative centerline')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(figure_folder/'01_centerline_U2.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    err=rep['Hybrid_U2'].to_numpy()-rep['FEM_U2'].to_numpy()
    ax.plot(rep['X'],err,'.-')
    ax.axhline(0,color='grey',linewidth=.8)
    ax.set(xlabel='Beam x (mm)',ylabel='Hybrid − FEM U2 (mm)',title='Signed centerline vertical displacement error')
    ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(figure_folder/'02_centerline_U2_error.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.bar([r['region'] for r in reports],[r['U2_rel_l2_percent'] for r in reports])
    ax.set(xlabel='Region',ylabel='Relative U2 L2 error (%)',title='Seven-region vertical displacement error')
    ax.grid(axis='y',alpha=.25);fig.tight_layout()
    fig.savefig(figure_folder/'03_region_U2_error.png',dpi=200);plt.close(fig)
    (figure_folder/'65W_K_audit.json').write_text(json.dumps(overall,indent=2))
    pd.DataFrame(reports).to_csv(figure_folder/'65W_K_region_metrics.csv',index=False)
    print('65W-K AUDIT PASSED: 4592 unique nodes, 7 owners, IP mapping valid')
    print('Full vector relative L2 (%%): %.6f'%overall['vector_rel_l2_percent'])
    print('Full U2 relative L2 (%%): %.6f'%overall['U2_rel_l2_percent'])
    print('Plots/report:',figure_folder)


if __name__=='__main__':
    main()
