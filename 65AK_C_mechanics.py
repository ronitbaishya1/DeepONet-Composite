#!/usr/bin/env python3
"""65AK-C: diagnose why predicted LE differs from Hencky LE derived from frozen U.

Run: python3 65AK_C_mechanics.py --project .
Comparisons use Abaqus local material component convention already validated by
65AJ-F. C@LE is an approximate small-strain constitutive diagnostic, not a
finite-deformation Abaqus constitutive law or equilibrium check.
NO retraining, no use of new references as model features.
"""
import argparse, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ak=importlib.import_module('65AK_common')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    a=p.parse_args();root=Path(a.project).resolve()
    data={reg:ak.load_A(root,reg) for reg in ak.REGIONS}
    for reg,z in data.items():ak.provenance_check(root,reg,z)
    out=ak.new_output(root,'C_mechanics')
    rows=[]
    for reg,z in data.items():
        for i,case in enumerate(z['case_ids'].astype(str)):
            truthL=z['LE_ref'][i];truthS=z['S_ref'][i];kinF=z['LE_from_FEM_U'][i];kinU=z['LE_from_pred_U'][i]
            rf=ak.relpct(kinF[...,1],truthL[...,1]);ru=ak.relpct(kinU[...,1],truthL[...,1])
            u=ak.relpct(z['U_pred'][i],z['U_ref'][i]);
            for model in ak.ARMS:
                L=z['LE_'+model][i];S=z['S_'+model][i]
                constS=z['S_constitutive_'+model][i]
                rows.append(dict(region=reg,case_id=case,model=model,
                    predicted_U_vector_error_pct=u,
                    reference_kinematic_LE22_error_pct=rf,
                    predicted_U_derived_LE22_error_pct=ru,
                    learned_LE22_error_pct=ak.relpct(L[...,1],truthL[...,1]),
                    learned_LE22_vs_predU_kinematic_gap_pct=ak.relpct(L[...,1],kinU[...,1]),
                    learned_LE_vector_error_pct=ak.relpct(L,truthL),
                    predicted_U_derived_LE_vector_error_pct=ak.relpct(kinU,truthL),
                    S22_prediction_error_pct=ak.relpct(S[...,1],truthS[...,1]),
                    S22_constitutive_error_pct=ak.relpct(constS[...,1],truthS[...,1]),
                    S22_predU_constitutive_error_pct=ak.relpct(z['S_from_pred_U'][i,:,1],truthS[...,1]),
                    predicted_S22_vs_CLE22_gap_pct=ak.relpct(S[...,1],constS[...,1]),
                    FEM_reference_CLE22_vs_S22_pct=ak.relpct(z['S_from_FEM_LE'][i,:,1],truthS[...,1])))
        selected=[r for r in rows if r['region']==reg and r['model'] in ('65AI_local_lgno','65AJ_matched_mlp')]
        names=z['case_ids'].astype(str).tolist();X=np.arange(len(names));width=.35
        fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
        for k,model in enumerate(('65AI_local_lgno','65AJ_matched_mlp')):
            rr=sorted([r for r in selected if r['model']==model],key=lambda r:r['case_id'])
            axs[0].bar(X+(k-.5)*width,[r['learned_LE22_error_pct'] for r in rr],width,
                       label='LGNO' if k==0 else 'matched MLP')
        urow=sorted([r for r in selected if r['model']=='65AI_local_lgno'],key=lambda r:r['case_id'])
        axs[0].plot(X,[r['predicted_U_derived_LE22_error_pct'] for r in urow],color='black',ls='--',marker='x',
                    label='LE22 derived from frozen U')
        axs[0].set_ylabel('LE22 error vs Abaqus [%]');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
        for k,model in enumerate(('65AI_local_lgno','65AJ_matched_mlp')):
            rr=sorted([r for r in selected if r['model']==model],key=lambda r:r['case_id'])
            axs[1].bar(X+(k-.5)*width,[r['learned_LE22_vs_predU_kinematic_gap_pct'] for r in rr],width,
                       label='LGNO' if k==0 else 'matched MLP')
        axs[1].set_ylabel('LE22 vs strain(frozen U) gap [%]');axs[1].grid(alpha=.2)
        axs[1].set_xticks(X);axs[1].set_xticklabels(names,rotation=25,ha='right')
        fig.suptitle(reg+' — displacement/strain inconsistency across six FEM cases')
        fig.tight_layout();fig.savefig(out/(reg+'_kinematic_diagnosis.png'),dpi=155);plt.close(fig)
        print('65AK-C checked mechanics:',reg,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'65AK_C_per_case_mechanics.csv',index=False)
    agg=df.groupby(['region','model'],as_index=False)[
        ['predicted_U_vector_error_pct','reference_kinematic_LE22_error_pct',
         'predicted_U_derived_LE22_error_pct','learned_LE22_error_pct',
         'learned_LE22_vs_predU_kinematic_gap_pct','S22_prediction_error_pct',
         'S22_constitutive_error_pct','S22_predU_constitutive_error_pct',
         'predicted_S22_vs_CLE22_gap_pct']].mean(numeric_only=True)
    agg.to_csv(out/'65AK_C_average_mechanics.csv',index=False)
    ak.atomic_summary(out/'65AK_C_summary.json',dict(step='65AK-C',cases_per_region=6,
        scope='Frozen offline models evaluated with FEM oracle interface PCA coefficients',
        mechanical_diagnostic='LE(U) calculated via original 65AJ-F 3D C3D8R displacement-gradient/Hencky routine',
        constitutive_diagnostic='S vs C@LE uses prior small-deformation approximation; not full Abaqus finite-strain update',
        warning='Large strain mismatch does not prove a unique cause. Compare learned LE error against LE(frozen U) error before designing a constrained architecture.',
        prohibited='Do not retrain or select models using six external FEM cases; do not modify working 65V solver.'))
    print('65AK-C DONE:',out)

if __name__=='__main__':main()
