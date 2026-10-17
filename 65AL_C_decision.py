#!/usr/bin/env python3
"""65AL-C: assemble READ-ONLY derivative audit and pre-register next TRAIN-only path.

Do not tune weights, smoothing parameters, or select models using the six
external ODBs. Future architecture experiments must use original train/tune.
"""
import argparse,importlib
from pathlib import Path
import numpy as np
import pandas as pd
al=importlib.import_module('65AL_common')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    a=p.parse_args();root=Path(a.project).resolve()
    ia=root/'results'/al.BASE/'A_gradient';ib=root/'results'/al.BASE/'B_locality'
    case=pd.read_csv(ia/'65AL_A_case_gradient_errors.csv')
    comp=pd.read_csv(ia/'65AL_A_component_gradient_errors.csv')
    zone=pd.read_csv(ia/'65AL_A_near_vs_interior.csv')
    rough=pd.read_csv(ib/'65AL_B_local_error_roughness.csv')
    out=al.outdir(root,'C_recommendation')
    rows=[]
    for reg in al.REGIONS:
        c=case[case.region==reg];s=comp[comp.region==reg];r=rough[rough.region==reg]
        s=s.groupby('gradient_component',as_index=False)['gradient_RMSE'].mean()
        bad=s.sort_values('gradient_RMSE',ascending=False).head(3)
        amp=float(np.median(c.gradient_error_amplification))
        avg_grad=float(c.grad_full_rel_L2_pct.mean())
        avg_u=float(c.U_vector_rel_L2_pct.mean())
        frac=float(c.grad_energy_top5pct_cells_share_pct.mean())
        roughness=float(r.neighbor_fluctuation_to_total_error_ratio.mean())
        row=dict(region=reg,mean_U_error_pct=avg_u,mean_gradient_error_pct=avg_grad,
                 median_derivative_error_amplification=amp,
                 average_top5pct_elements_share_of_gradient_error_pct=frac,
                 mean_nodal_local_fluctuation_ratio=roughness,
                 dominant_component_RMSE_1=str(bad.iloc[0].gradient_component),
                 dominant_component_RMSE_2=str(bad.iloc[1].gradient_component),
                 dominant_component_RMSE_3=str(bad.iloc[2].gradient_component))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'65AL_C_gradient_diagnosis.csv',index=False)
    plan=dict(step='65AL-C',status='diagnostics_completed_not_a_new_model',frozen_external_cases=6,
      results=rows,
      evidence_based_next_training_experiment={
        'priority_1':'On the ORIGINAL 64J training cases, audit raw vs 65AA vs 65AC local C3D8R gradient vs 65AD compatible U projection. Preserve hard interface U.',
        'priority_2':'If errors concentrate in derivatives, create a new TRAIN-only model with a shared trainable U correction and C3D8R gradient loss; do not impose LE(pred U) as a hard target on the current frozen-U LGNO.',
        'priority_3':'Evaluate a multi-objective loss using training FEM displacement and gradients; begin with component-normalized gradient loss and hard exact interface compatibility. Tune lambda only on preexisting internal train/tune data.',
        'priority_4':'Only after gradient accuracy improves, test an independent mechanics head based on compatible Hencky strain and nonlinear-material diagnostics, with MLP parameter-matched and inference-time comparisons.'},
      paper_motivations=[
        'Hildebrand & Klinge: displacement and derivative accuracy must be evaluated separately; BC transformations and balanced physics loss terms can help but are not guaranteed.',
        'Zhong et al.: compare architectures fairly including normalization, parameter count, accuracy and throughput on the same data; 3D unstructured meshes may favor local/graph representations.',
        'Park & Kang Point-DeepONet: geometry/global condition features can help across many 3D geometries and loads; fixed-geometry glass epoxy does not yet warrant a PointNet/SDF rebuild.'],
      guardrails=['No training or hyperparameter selection on six 65AJ/65AK/65AL external ODB cases.',
                  'No overwrite of 65AI,65AJ,65V checkpoints/solvers.',
                  'Small-strain C@LE is not the exact finite-deformation Abaqus constitutive update.',
                  'Conditional full-FEM-oracle boundary predictions are not a proof of seven-region interface equilibrium.'])
    al.summary(out/'65AL_C_recommendations.json',plan)
    print('65AL-C DONE:',out)
    for r in rows:print(r['region'],'mean grad error %.3f%%, median amplification %.2f, top-5%% cells %.1f%%'%(r['mean_gradient_error_pct'],r['median_derivative_error_amplification'],r['average_top5pct_elements_share_of_gradient_error_pct']))

if __name__=='__main__':main()
