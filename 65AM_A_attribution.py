#!/usr/bin/env python3
"""65AM-A: Attribute frozen raw-vs-Hermite-vs-PCA displacement gradient error.

NO training; NO modification to 65V, source models, data, old result folders.
External AJ_new_01..06 already repeatedly studied; NEVER tune using these.

python3 65AM_A_attribution.py --project . --check-only
python3 65AM_A_attribution.py --project .
"""
import argparse, importlib
from pathlib import Path
import numpy as np, pandas as pd, torch
am=importlib.import_module('65AM_common')


def load_frozen(root, reg, device, check_only=False):
    al=importlib.import_module('65AL_common')
    ak=importlib.import_module('65AK_common')
    aj=importlib.import_module('65AJ_common')
    ai=importlib.import_module('65AI_common')
    aa=importlib.import_module('65AA_common')
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region

    # Load the same cached frozen test cases already validated in 65AK/65AL.
    z,ds,conn,dn,aff=al.load(root,reg)
    ext=ak.open_external(root,reg)
    if not np.array_equal(z['case_ids'].astype(str),ext['case_ids'].astype(str)):
        raise RuntimeError('External case ordering differs from 65AK cache for '+reg)
    st=ai.member_source(root,reg,1,device,not check_only)
    if check_only:
        return {'region':reg,'cases':z['case_ids'].astype(str).tolist(),
                'geometry_affine_error':aff,'checkpoint_source_sha':st['source_sha']}

    # The 65AJ attachment sets oracle interface PCA coefficients from full FEM U.
    # Original frozen scalers, source model and material inputs are unchanged.
    ids,branch,pca_rms,mat_distance=aj.attach_external(root,st,ext,device)
    inputs=aa.tensor_inputs(st['ck'],st['ds'],device)
    model=ai.co.make_model(st['ck'],device).eval()
    raw=aa.predict_numpy(model,inputs,ids,batch_size=2,correct=False)
    hard=aa.predict_numpy(model,inputs,ids,batch_size=2,correct=True)
    del model
    cached=np.asarray(z['U_pred'],np.float64)
    difference=np.max(np.abs(np.asarray(hard,np.float64)-cached))
    if difference>1.0e-4:
        raise RuntimeError(f'{reg}: hard U does NOT reproduce frozen 65AK U cache (max diff {difference:.6g} mm). Check source checkpoints/data before proceeding')

    # Synthetic control: start from TRUE FEM interior U, replacing ONLY interface
    # values with original PCA reconstruction (same correction construction).
    # This uses ground-truth U only for offline diagnosis, NEVER as model input.
    fem=np.asarray(z['U_ref'],np.float32)
    physical=inputs['branch'][ids]*inputs['bstd']+inputs['bmean']
    ref_t=torch.as_tensor(fem,device=device)
    with torch.no_grad():
        pca_t=apply_hard_compatibility_7region(ref_t, physical,inputs['context'])
    pca_only=pca_t.detach().cpu().numpy()

    # Independently verify the PCA-only control changes FEM interface values
    # only by residual of PCA projection. Test physical output boundary exactness.
    ctx=inputs['context']
    frame_errors={}
    for side in ('left','right'):
        if ctx[side] is None:continue
        ix=ctx[side+'_local_indices'].detach().cpu().numpy()
        sl=ctx[side+'_slice']
        from src.hard_interface_compatibility_7region import reconstruct_interface_target
        with torch.no_grad(): target=reconstruct_interface_target(physical,ctx[side],sl).detach().cpu().numpy()
        hard_diff=float(np.max(np.abs(hard[:,ix,:]-target)))
        pca_diff=float(np.max(np.abs(pca_only[:,ix,:]-target)))
        if max(hard_diff,pca_diff)>1e-4:
            raise RuntimeError(f'{reg} {side} interface projection check failed, diff={max(hard_diff,pca_diff)}')
        frame_errors[side+'_PCA_reference_RMSE_mm']=am.rmse(target,fem[:,ix,:])
        frame_errors[side+'_hard_target_max_error_mm']=hard_diff
        frame_errors[side+'_PCA_control_target_max_error_mm']=pca_diff

    grad_ref=am.gradient(fem,conn,dn)
    grad_raw=am.gradient(raw,conn,dn)
    grad_hard=am.gradient(hard,conn,dn)
    grad_pca=am.gradient(pca_only,conn,dn)
    # Consistency against original Step 65AL result (same exact FE operator).
    prior=am.gradient(cached,conn,dn)
    if am.relpct(grad_hard,prior)>0.1:
        raise RuntimeError('Gradient of reconstructed hard prediction differs from Step 65AK reference')
    for arr in (grad_ref,grad_raw,grad_hard,grad_pca):
        if not np.isfinite(arr).all():raise RuntimeError('Nonfinite gradient')
    return dict(z=z,raw=raw,hard=hard,pca_only=pca_only,
                grad_ref=grad_ref,grad_raw=grad_raw,grad_hard=grad_hard,grad_pca=grad_pca,
                conn=conn,dn=dn,aff=aff,cached_difference_mm=float(difference),
                pca_rms=pca_rms,frame_errors=frame_errors,source_sha=st['source_sha'])


def evaluate_region(reg,info,out):
    z=info['z']; ip=np.asarray(z['ip_coordinates'],np.float64)
    masks=am.zone_masks(ip)
    geom=am.check_gradient_geometry(z['coordinates'],info['conn'],info['dn'])
    case_rows=[];zone_rows=[];components=[];profile_rows=[]
    gref=info['grad_ref'];graw=info['grad_raw'];ghard=info['grad_hard'];gpca=info['grad_pca']
    for k,cid in enumerate(z['case_ids'].astype(str)):
        base={'region':reg,'case_id':cid}
        row=dict(base,**am.summary_case(gref[k],graw[k],ghard[k],gpca[k],
            z['U_ref'][k],info['raw'][k],info['hard'][k]))
        row['pca_only_displacement_error_pct']=am.relpct(info['pca_only'][k],z['U_ref'][k])
        # Conservation of squared error: ||e_raw+delta||^2 = ||e_raw||^2+||delta||^2+2<e_raw,delta>.
        sq_raw=float(np.mean((graw[k]-gref[k])**2))
        sq_hard=float(np.mean((ghard[k]-gref[k])**2))
        reconstruction=sq_raw+float(np.mean((ghard[k]-graw[k])**2))+2*float(np.mean((graw[k]-gref[k])*(ghard[k]-graw[k])))
        if abs(sq_hard-reconstruction)>1e-10:raise RuntimeError('Error decomposition identity failed')
        row['squared_error_identity_absolute_residual']=abs(sq_hard-reconstruction)
        case_rows.append(row)
        for zone,mask in masks.items():
            zone_rows.append(dict(base,zone=zone,n_elements=int(mask.sum()),
                **am.summary_case(gref[k,mask],graw[k,mask],ghard[k,mask],gpca[k,mask])))
        for m,name in enumerate(am.COMPONENTS):
            for d,axis in enumerate(am.DIRECTIONS):
                a,b,c,v=(field[k,:,m,d] for field in (gref,graw,ghard,gpca))
                components.append(dict(base,component=name+'_d'+axis,
                    grad_ref_RMS=am.rms(a),raw_RMSE=am.rmse(b,a),hard_RMSE=am.rmse(c,a),
                    pca_only_RMSE=am.rmse(v,a),
                    raw_relative_pct=am.relpct(b,a),hard_relative_pct=am.relpct(c,a),
                    hard_effect_pct=100*(1-am.rmse(c,a)/max(am.rmse(b,a),1e-16)),
                    raw_signed_bias=float(np.mean(b-a)),hard_signed_bias=float(np.mean(c-a)),
                    Hermite_correction_signed_mean=float(np.mean(c-b))))
        # Distinct 3D IP X slices; not XZ projections (no hidden Y averaging).
        for xval in np.unique(np.round(ip[:,0],6)):
            mask=np.isclose(ip[:,0],xval,atol=1e-5)
            a,b,c,v=(field[k,mask,1,1] for field in (gref,graw,ghard,gpca))
            profile_rows.append(dict(base,x_mm=float(xval),n_elements=int(mask.sum()),
                U2_dY_FEM_mean=float(np.mean(a)),U2_dY_raw_mean=float(np.mean(b)),U2_dY_hard_mean=float(np.mean(c)),
                U2_dY_raw_RMSE=am.rmse(b,a),U2_dY_hard_RMSE=am.rmse(c,a),U2_dY_pca_only_RMSE=am.rmse(v,a),
                gradient_all_raw_RMSE=am.rmse(graw[k,mask],gref[k,mask]),
                gradient_all_hard_RMSE=am.rmse(ghard[k,mask],gref[k,mask])))
    np.savez_compressed(out/(reg+'_gradient_decomposition.npz'),
        case_ids=z['case_ids'].astype(str),ip_coordinates=ip.astype(np.float32),
        grad_ref=gref.astype(np.float32),grad_raw=graw.astype(np.float32),
        grad_hard=ghard.astype(np.float32),grad_pca_only=gpca.astype(np.float32))
    return case_rows,zone_rows,components,profile_rows,geom


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    root=Path(args.project).expanduser().resolve()
    ai=importlib.import_module('65AI_common')
    device=ai.co.device_from_option(args.device)
    print('65AM-A | project:',root,'| device:',device,'| diagnostics only',flush=True)
    for reg in am.REGIONS:
        info=load_frozen(root,reg,device,check_only=True)
        print('CHECK PASS',reg,'| cases',len(info['cases']),'| trained model SHA:',info['checkpoint_source_sha'][:14],flush=True)
    if args.check_only:
        print('65AM-A CHECK ONLY DONE; no output created')
        return
    # Required prereqs confirmed before opening a new result folder.
    out=am.output(root,'A_attribution')
    all_case=[];all_zone=[];all_comp=[];all_profile=[];summ=[]
    for reg in am.REGIONS:
        print('Processing '+reg+' (recomputing frozen raw and hard displacement)...',flush=True)
        data=load_frozen(root,reg,device,check_only=False)
        ca,zo,co,pr,geom=evaluate_region(reg,data,out)
        all_case+=ca;all_zone+=zo;all_comp+=co;all_profile+=pr
        summ.append(dict(region=reg,case_count=len(ca),
                         cached_hard_prediction_max_abs_difference_mm=data['cached_difference_mm'],
                         geometry=geom,source_sha256=data['source_sha'],
                         pca_reference_residuals=data['frame_errors'],
                         mean_raw_gradient_error_pct=float(np.mean([x['grad_raw_error_pct'] for x in ca])),
                         mean_hard_gradient_error_pct=float(np.mean([x['grad_hard_error_pct'] for x in ca])),
                         mean_PCA_only_gradient_error_pct=float(np.mean([x['grad_pca_only_error_pct'] for x in ca])),
                         mean_effect_of_Hermite_correction_pct=float(np.mean([x['hard_gradient_effect_pct'] for x in ca]))))
        print(reg,'raw-grad %.3f%% | Hermite-grad %.3f%% | PCA only %.5f%%'%(summ[-1]['mean_raw_gradient_error_pct'],summ[-1]['mean_hard_gradient_error_pct'],summ[-1]['mean_PCA_only_gradient_error_pct']),flush=True)
        if device.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(all_case).to_csv(out/'65AM_A_case_metrics.csv',index=False)
    pd.DataFrame(all_zone).to_csv(out/'65AM_A_zone_metrics.csv',index=False)
    pd.DataFrame(all_comp).to_csv(out/'65AM_A_component_metrics.csv',index=False)
    pd.DataFrame(all_profile).to_csv(out/'65AM_A_x_profiles.csv',index=False)
    am.saved_json(out/'65AM_A_summary.json',dict(step='65AM-A',regions=summ,
        provenance='Original frozen 65AA source model, same oracle PCA branch on 65AJ external cases; cached 65AK hard U crosschecked',
        PCA_control='Full FEM nodal U plus original Hermite projection onto truncated PCA interface; NOT a deployable oracle feature',
        zero_training=True,zero_weight_selection=True,full_seven_region_coupling_test=False,
        external_cohort_already_inspected=True,
        warning='This is a reused six-case diagnostic, not new independent validation. If the hard correction increases error, it is evidence of a contribution, not proof of unique cause.'))
    print('65AM-A DONE:',out,flush=True)

if __name__=='__main__':main()
