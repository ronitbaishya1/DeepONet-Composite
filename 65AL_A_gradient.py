#!/usr/bin/env python3
"""65AL-A: True 3D C3D8R derivative errors, LE compatibility audit.

Run:
 python3 65AL_A_gradient.py --project . --check-only
 python3 65AL_A_gradient.py --project .

No training, no modification of any stored model. Uses six previously examined
EXTERNAL FEM cases for diagnostics ONLY; results are not an additional holdout.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import importlib
al=importlib.import_module('65AL_common')

def evaluate(root,reg,z,ds,conn,dn,aff,stage=None):
    u0=np.asarray(z['U_pred'],dtype=np.float64);ur=np.asarray(z['U_ref'],dtype=np.float64)
    gr=al.gradient(ur,conn,dn); gp=al.gradient(u0,conn,dn)
    strain_from_reference=al.ac.logstrain_from_gradient(gr)
    strain_from_pred=al.ac.logstrain_from_gradient(gp)
    refcheck=al.relpct(strain_from_reference,z['LE_ref'])
    predcheck=al.relpct(strain_from_pred,z['LE_from_pred_U'])
    if refcheck>2.0:raise RuntimeError('The reference FEM reconstruction failed: '+reg+' '+str(refcheck))
    if predcheck>0.2:raise RuntimeError('New predicted-U strain differs from original 65AK U->LE: '+reg+' '+str(predcheck))
    near=al.near_mask(z['ip_coordinates'],.6)
    gnames=('U1','U2','U3'); axes=('X','Y','Z')
    cases=[];components=[];zones=[]
    for k,case in enumerate(z['case_ids'].astype(str)):
        r=gr[k];p=gp[k]
        err=p-r; e_nodes=u0[k]-ur[k]
        urel=al.relpct(u0[k],ur[k]);grel=al.relpct(p,r)
        err_energy=np.sum(err**2,axis=(1,2))
        order=np.argsort(err_energy)[::-1]
        top=max(1,int(np.ceil(.05*len(order))))
        frac=float(100*np.sum(err_energy[order[:top]])/max(np.sum(err_energy),1e-30))
        le22=al.relpct(strain_from_pred[k,:,1],z['LE_ref'][k,:,1])
        learned=al.relpct(z['LE_65AI_local_lgno'][k,:,1],z['LE_ref'][k,:,1])
        cases.append(dict(region=reg,case_id=case,U_vector_rel_L2_pct=urel,
            grad_full_rel_L2_pct=grel,grad_full_RMSE=al.rmse(p,r),
            gradient_error_amplification=grel/max(urel,1e-12),
            derived_LE22_rel_L2_pct=le22,LGNO_LE22_rel_L2_pct=learned,
            grad_energy_top5pct_cells_share_pct=frac,
            affine_shape_function_max_abs_error=aff))
        for m,label in enumerate(gnames):
            for d,axis in enumerate(axes):
                components.append(dict(region=reg,case_id=case,displacement_component=label,
                    derivative_direction=axis,gradient_component=label+'_d'+axis,
                    relative_L2_pct=al.relpct(p[:,m,d],r[:,m,d]),
                    gradient_RMSE=al.rmse(p[:,m,d],r[:,m,d]),
                    FEM_gradient_RMS=float(np.sqrt(np.mean(r[:,m,d]**2))),
                    predicted_gradient_RMS=float(np.sqrt(np.mean(p[:,m,d]**2)))))
        for name,mask in (('near_interface',near),('interior',~near)):
            zones.append(dict(region=reg,case_id=case,zone=name,n_elements=int(mask.sum()),
                grad_vector_rel_L2_pct=al.relpct(p[mask],r[mask]),
                grad_vector_RMSE=al.rmse(p[mask],r[mask]),
                grad_U2_Y_RMSE=al.rmse(p[mask,1,1],r[mask,1,1]),
                derived_LE22_rel_L2_pct=al.relpct(strain_from_pred[k,mask,1],z['LE_ref'][k,mask,1]),
                learned_LGNO_LE22_rel_L2_pct=al.relpct(z['LE_65AI_local_lgno'][k,mask,1],z['LE_ref'][k,mask,1])))
    if stage is not None:
        np.savez_compressed(stage/(reg+'_gradient_fields.npz'),case_ids=z['case_ids'].astype(str),
          ip_coordinates=z['ip_coordinates'],node_coordinates=z['coordinates'],
          grad_ref=gr.astype(np.float32),grad_pred=gp.astype(np.float32),
          LE_from_ref_U=strain_from_reference.astype(np.float32),
          LE_from_pred_U=strain_from_pred.astype(np.float32),LE_ref=z['LE_ref'],
          U_error=(u0-ur).astype(np.float32),
          grad_error_energy=np.sum((gp-gr)**2,axis=(2,3)).astype(np.float32))
    return cases,components,zones,refcheck,predcheck

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).resolve()
    inp={}
    for reg in al.REGIONS:
        z,ds,conn,dn,aff=al.load(root,reg)
        inp[reg]=(z,ds,conn,dn,aff)
        print('65AL-A CHECK:',reg,'six cases and C3D8R affine test passed; max affine error',aff,flush=True)
    if a.check_only:return
    out=al.outdir(root,'A_gradient')
    cases=[];components=[];zones=[];checks=[]
    for reg,items in inp.items():
        q,c,m,rc,pc=evaluate(root,reg,*items,stage=out)
        cases+=q;components+=c;zones+=m
        checks.append({'region':reg,'FEM_U_to_LE_all_comp_pct':rc,'prior_65AK_pred_U_to_LE_reproduction_pct':pc})
    pd.DataFrame(cases).to_csv(out/'65AL_A_case_gradient_errors.csv',index=False)
    pd.DataFrame(components).to_csv(out/'65AL_A_component_gradient_errors.csv',index=False)
    pd.DataFrame(zones).to_csv(out/'65AL_A_near_vs_interior.csv',index=False)
    al.summary(out/'65AL_A_summary.json',dict(step='65AL-A',cases_per_region=6,reference_integrity=checks,
        frozen_checkpoint=True,full_FEM_oracle_interface=True,
        derivative='FE C3D8R integration-point dU_i/dX_j from original 64J B_center',
        warning='Six cases were already evaluated in 65AJ and 65AK. Not an additional holdout.'))
    print('65AL-A DONE:',out)

if __name__=='__main__':main()
