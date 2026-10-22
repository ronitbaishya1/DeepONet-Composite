#!/usr/bin/env python3
"""B9A11E: read-only spatial error attribution of the ORIGINAL 65W / V4
seven-region hybrid reconstruction (not a coupled B5 or B9A11C evaluation).

Mac M1 Max, from the DeepONet_Composite project directory:
    python3 65AM_B9A11E_spatial_error_audit.py --project .

Reads existing 65W-H CSV/JSON only. Does not retrain or modify anything.
Outputs a NEW results/65AM_B9A11E_spatial_error/ folder (protected).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ('NO_OL', 'FE_L', 'NO_L', 'FE_C', 'NO_R', 'FE_R', 'NO_OR')
INTERFACES = {
    'NO_OL': (-9.6,), 'FE_L': (-9.6, -6.6),
    'NO_L': (-6.6, -1.8), 'FE_C': (-1.8, 1.8),
    'NO_R': (1.8, 6.6), 'FE_R': (6.6, 9.6), 'NO_OR': (9.6,)
}
NODES_METRICS = {'U_vector': ('U1','U2','U3'), 'U1': ('U1',), 'U2': ('U2',), 'U3': ('U3',)}
IP_METRICS = {f: (f,) for f in ('LE22', 'S22', 'S33', 'LE12', 'S12', 'LE11', 'S11', 'LE33', 'S23')}


def err_metrics(ref, pred):
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if ref.shape != pred.shape or not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite or mismatched prediction/reference arrays')
    d = pred - ref
    ss = float(np.sum(d*d))
    den = float(np.sum(ref*ref))
    return {'l2_pct': float(100*np.sqrt(ss/den)) if den > 1e-24 else None,
            'rmse': float(np.sqrt(np.mean(d*d))),
            'mae': float(np.mean(np.abs(d))),
            'squared_error': ss,
            'reference_norm': float(np.sqrt(den))}


def get_values(df, names):
    ref_cols = ['FEM_'+n for n in names]
    pred_cols = ['Hybrid_'+n for n in names]
    absent = [s for s in ref_cols+pred_cols if s not in df.columns]
    if absent:
        raise RuntimeError('Missing data fields '+str(absent))
    return df[ref_cols].to_numpy(dtype=float), df[pred_cols].to_numpy(dtype=float)


def run(project, band_mm, out):
    root = project/'results'/'hybrid_online_7region_65W_middle_validation'
    paths = {'nodes':root/'assembled_hybrid_nodes.csv',
             'ip':root/'assembled_hybrid_ip.csv',
             'summary':root/'65W_H_validation_summary.json',
             'matching':root/'ip_matching_audit.json'}
    for name,path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Prior B9A11E results protected; choose --out for a NEW folder: '+str(out))
    nodal = pd.read_csv(paths['nodes']); ip = pd.read_csv(paths['ip'])
    with open(paths['summary']) as f: summary=json.load(f)
    with open(paths['matching']) as f: matching=json.load(f)
    if len(nodal)!=4592 or len(ip)!=3640:
        raise RuntimeError('Expected 4592 full-FEM nodes and 3640 mechanics points')
    if set(nodal.Owner.unique())!=set(REGIONS) or set(ip.Owner.unique())!=set(REGIONS):
        raise RuntimeError('Seven-region ownership mismatch')
    if len(nodal.NodeLabel.unique())!=len(nodal):
        raise RuntimeError('Duplicate assembled nodes')
    if not np.allclose(summary['material_MPa'],[46500.,12500.,4750.],atol=1e-6,rtol=0):
        raise RuntimeError('Wrong middle-material reference')
    if not matching.get('one_to_one') or matching.get('maximum_coordinate_distance_mm',np.inf) > 1e-4:
        raise RuntimeError('Mechanical integration-point matching is not verified')
    region_rows=[]; zone_rows=[]
    for kind,table,fields in [('nodal',nodal,NODES_METRICS),('mechanics',ip,IP_METRICS)]:
        x=table['X'].to_numpy(dtype=float)
        for name,components in fields.items():
            ref,pred=get_values(table,components)
            # Contribution is measured relative to the SAME global sum-of-squared error,
            # not a sum of regional relative percentages.
            global_squared_error=err_metrics(ref,pred)['squared_error']
            for region in REGIONS:
                mask=(table.Owner==region).to_numpy()
                a=err_metrics(ref[mask],pred[mask])
                row=dict(data=kind,region=region,component=name,count=int(mask.sum()),
                         rel_l2_pct=a['l2_pct'],rmse=a['rmse'],mae=a['mae'],
                         global_squared_error_share_pct=(100*a['squared_error']/global_squared_error
                                                         if global_squared_error>1e-24 else None))
                region_rows.append(row)
                distance=np.min(np.abs(x[mask,None]-np.array(INTERFACES[region])[None,:]),axis=1)
                local_ss=a['squared_error']
                for zone,localmask in [('within_%.3gmm_of_interface'%band_mm,distance<=band_mm),
                                       ('interior',distance>band_mm)]:
                    if not localmask.any():continue
                    b=err_metrics(ref[mask][localmask],pred[mask][localmask])
                    zone_rows.append(dict(data=kind,region=region,component=name,zone=zone,
                                          count=int(localmask.sum()),rel_l2_pct=b['l2_pct'],
                                          rmse=b['rmse'],mae=b['mae'],
                                          fraction_of_region_squared_error_pct=(100*b['squared_error']/local_ss
                                                                                 if local_ss>1e-24 else None)))
    df=pd.DataFrame(region_rows); zones=pd.DataFrame(zone_rows)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[]).to_csv(out/'B9A11E_region_errors.csv',index=False)
    zones.to_csv(out/'B9A11E_interface_vs_interior.csv',index=False)
    focus={}
    for name,kind in [('U_vector','nodal'),('U2','nodal'),('S22','mechanics'),('LE22','mechanics'),('S33','mechanics')]:
        top=df[(df.component==name)&(df.data==kind)].sort_values('global_squared_error_share_pct',ascending=False)
        focus[name]=[{'region':r.region,'error_pct':r.rel_l2_pct,
                       'global_error_share_pct':r.global_squared_error_share_pct} for r in top.itertuples(index=False)]
    report={'step':'B9A11E','based_on':'Original 65W-H seven-region V4 hybrid, NOT coupled B5/B9A11C',
            'nodes':len(nodal),'mechanics_points':len(ip),'interface_band_mm':band_mm,
            'region_rankings':focus,
            'cautions':['Shares are contributions to squared global error, not relative error percentages.',
                        'Near-interface classification uses x distance to fixed region boundaries.',
                        'Physical material/loading equivalence and reaction-force checks remain separate.',
                        'Error localization is an attribution diagnostic; does not identify a unique cause.']}
    (out/'B9A11E_summary.json').write_text(json.dumps(report,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for ax,name in zip(axes,('S22','LE22')):
            p=df[(df.component==name)&(df.data=='mechanics')].set_index('region').loc[list(REGIONS)]
            ax.bar(list(REGIONS),p.rel_l2_pct)
            ax.set_ylabel(name+' relative L2 (%)')
            ax.grid(axis='y',alpha=.2)
        axes[-1].set_xlabel('Original V4 hybrid seven regions')
        fig.tight_layout()
        fig.savefig(out/'B9A11E_S22_LE22_regions.png',dpi=180)
        plt.close(fig)
    except ImportError:
        print('Matplotlib absent; tables generated without optional plot.')
    print('B9A11E COMPLETE:',out)
    for comp in ('U_vector','S22','LE22'):
        ranked=focus[comp]
        print(comp,'largest squared error share:',ranked[0]['region'],
              round(ranked[0]['global_error_share_pct'],2),'%')
    print('Send B9A11E_summary.json and B9A11E_interface_vs_interior.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--band-mm',type=float,default=0.6)
    p.add_argument('--out',default=None)
    a=p.parse_args()
    if not (0<a.band_mm<=2):p.error('Require 0 < --band-mm <= 2 mm')
    project=Path(a.project).expanduser().resolve()
    out=(Path(a.out).expanduser().resolve() if a.out else
         project/'results'/'65AM_B9A11E_spatial_error')
    run(project,a.band_mm,out)

if __name__=='__main__':main()
