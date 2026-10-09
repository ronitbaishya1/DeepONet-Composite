"""65W-G: plots from REAL matched-field CSV and JSON, preferably on M1 Max Mac.
Do not hide missing NO regions; plots are explicitly labeled FE ONLY or seven-region.

python 65W_G_plot_validation.py --comparison-dir PATH_TO_C_field_comparison/fe_only
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def go(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(path).resolve()
    with open(folder/'65W_C_field_comparison.json') as f:summary=json.load(f)
    pairs=[]
    with open(folder/'65W_C_matched_nodes.csv',newline='') as f:
        for row in csv.DictReader(f):
            pairs.append({k:float(v) if k not in ('region',) else v for k,v in row.items()})
    if not pairs:raise RuntimeError('Matched-point CSV has no rows')
    full=summary['full_hybrid_validation_claim_allowed']
    classification='SEVEN-REGION HYBRID' if full else 'PARTIAL FEM-PATCH ONLY'
    # Bin x for clear comparison of beam sections without pretending FE+NO coverage.
    bins=defaultdict(list)
    for p in pairs:bins[round(p['x'],4)].append(p)
    X=sorted(bins)
    fm=[np.mean([r['FEM_U2'] for r in bins[x]]) for x in X]
    hy=[np.mean([r['Hybrid_U2'] for r in bins[x]]) for x in X]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(X,fm,label='Full FEM U2',linewidth=1.6)
    ax.plot(X,hy,'o',label='Supplied FE/NO U2',markersize=3,alpha=.75)
    ax.set_xlabel('x (mm)');ax.set_ylabel('U2 (mm)')
    ax.set_title('Displacement comparison — '+classification)
    ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(folder/'65W_G_U2_comparison.png',dpi=190)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    err=np.asarray(hy)-np.asarray(fm)
    ax.scatter(X,err,s=12)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set_xlabel('x (mm)');ax.set_ylabel('Hybrid minus full FEM U2 (mm)')
    ax.set_title('Signed U2 error — '+classification)
    ax.grid(alpha=.3);fig.tight_layout()
    fig.savefig(folder/'65W_G_U2_signed_error.png',dpi=190)
    plt.close(fig)
    regions=summary['per_region']
    valid=[(n,r['U2']['relative_l2_percent']) for n,r in regions.items()
           if r.get('U2') and r['U2']['relative_l2_percent'] is not None]
    if valid:
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar([v[0] for v in valid],[v[1] for v in valid])
        ax.set_ylabel('Matched-node U2 relative L2 error (%)')
        ax.set_title('Regional displacement error — '+classification)
        ax.tick_params(axis='x',rotation=25)
        ax.grid(axis='y',alpha=.3)
        fig.tight_layout();fig.savefig(folder/'65W_G_U2_region_error.png',dpi=190)
        plt.close(fig)
    print('PLOTS saved to',folder)
    print('DATA COVERAGE:',classification)
    if not full:print('Four neural-operator region fields must be added before full 7-region accuracy claims')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comparison-dir',required=True)
    args=p.parse_args()
    go(args.comparison_dir)

if __name__=='__main__':main()
