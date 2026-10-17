#!/usr/bin/env python3
"""65AM-B0: supervised snapshot POD/FE basis CAPACITY audit (no neural training).

Commands:
 python3 65AM_B0_basis_audit.py --project . --check-only
 python3 65AM_B0_basis_audit.py --project .
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import importlib
b=importlib.import_module('65AM_B_common')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args(); root=Path(a.project).expanduser().resolve()
    for region in b.REGIONS:
        ds,fit,tune,source,aff=b.load_region(root,region)
        b.check_reference(ds,fit)
        print('CHECK PASS',region,'fit',len(fit),'tune',len(tune),'nodes',len(ds['coordinates']),
              'C3D8R affine error',round(aff,8),'source',source.name,flush=True)
    if a.check_only:
        print('65AM-B0 CHECK ONLY complete; no outputs created')
        return
    out=b.output_dir(root,'B0_basis')
    summaries=[]
    for region in b.REGIONS:
        result=b.build_basis(root,region)
        folder=out/region;folder.mkdir(parents=True,exist_ok=True)
        result['metrics'].to_csv(folder/'B0_capacity.csv',index=False)
        selected=result['selected_rank']
        np.savez_compressed(folder/'basis.npz',
            fit_ids=result['fit'],tune_ids=result['tune'],fixed=result['fixed'],
            basis=result['V'].astype(np.float32),sU=result['sU'].astype(np.float32),
            sG=result['sG'].astype(np.float32),sR=result['sR'].astype(np.float32),
            branch_mean=result['branch_mean'].astype(np.float32),
            branch_std=result['branch_std'].astype(np.float32),
            fit_lift=result['lift'][:len(result['fit'])].astype(np.float32),
            selected_rank=np.int32(-1 if selected is None else selected),
            source_sha=np.array(result['source_sha']),affine_error=np.array(result['affine_error']))
        sv=result['singular']
        effective_cond=float(sv[0]/max(sv[-1],1e-15))
        summary={'region':region,'selected_rank':selected,'fit_cases':len(result['fit']),
                 'tune_cases':len(result['tune']),'fitting_only_POD':True,
                 'candidate_ranks':list(b.RANKS),'condition_number_snapshot_matrix':effective_cond,
                 'minimum_singular':float(sv[-1]),'maximum_singular':float(sv[0]),
                 'interface_dofs_zero_by_construction':True,'affine_error':result['affine_error'],
                 'fit_FEM_vs_PCA_interface_rmse_mm':result['interface_fit_rmse_mm'],
                 'selection_gate':'FIT oracle: U vector <=1%, full gradient <=5%, LE22 <=10%',
                 'note':'Oracle uses reference FEM U to project: NOT an inference model. TUNE oracle only diagnoses representational coverage.'}
        (folder/'B0_summary.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print('B0',region,'selected_rank',selected,'snapshot_condition',round(effective_cond,4),flush=True)
        print(result['metrics'].loc[result['metrics']['split']=='tune',
             ['rank','U_vec_rel_pct','grad_rel_pct','LE22_rel_pct']].to_string(index=False),flush=True)
    (out/'B0_overall.json').write_text(json.dumps(summaries,indent=2))
    if any(d['selected_rank'] is None for d in summaries):
        print('65AM-B0 CAPACITY GATE FAILED for at least one region. DO NOT run B1/B2. Share the B0_capacity.csv files.')
    else:
        print('65AM-B0 CAPACITY GATE PASS both regions. You may run B1 and B2.')
    print('Outputs:',out)

if __name__=='__main__':main()
