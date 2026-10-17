#!/usr/bin/env python3
"""65Z-A: Audit nonlinear log(Ustretch) strain from the *predicted, hard-corrected*
V4 displacement, not from FEM-reference displacement. Train/validation only.

python3 65Z_A_predicted_logstrain_audit.py --project . --check-only
python3 65Z_A_predicted_logstrain_audit.py --project . --members 5

Results: results/65Z_nonlinear_mechanics/A_predicted_kinematics/
  65Z_A_summary.json; 65Z_A_member_metrics.csv; internal member_NN_kin.npz.
No Abaqus, no retraining, no use of original test split, no changes to 65V.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from importlib.util import spec_from_file_location,module_from_spec

# Numeric-prefixed filename: explicit import from this script's directory.
sp=spec_from_file_location('s65zcommon',Path(__file__).with_name('65Z_common.py'))
common=module_from_spec(sp);sys.modules[sp.name]=common;sp.loader.exec_module(common)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=5)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--max-kinematic-ratio',type=float,default=1.25,
                   help='Only allow a physics kinematic loss if its predicted-U strain error is at most this multiple of learned-LE error for LE11/22/12')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5 or args.max_kinematic_ratio<=0:raise ValueError('Invalid arguments')
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    device=common.device_from_option(args.device)
    reference_worst=common.reference_audit_gate(root)
    base=root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'
    plans=[]
    for seg in common.SEGMENTS:
        ds=common.segment_data(root,seg)
        for m in range(1,args.members+1):
            path=base/seg/('member_%02d'%m)/'best_v4_7region.pt'
            if not path.is_file():raise FileNotFoundError(str(path))
            plans.append((seg,m,path))
        # Early interface dependency check, as 64N uses it.
        common.interface_context(root,ds,torch.device('cpu'))
    if args.check_only:
        print('65Z-A CHECK PASS: 65Y reference strain audit and %d fine-tuned models'%len(plans))
        print('Reference max logU(LE11/22/12) error %.5f%%'%reference_worst)
        print('Device:',device,'; NO test cases or Abaqus jobs used')
        return
    out=root/'results'/'65Z_nonlinear_mechanics'/'A_predicted_kinematics'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Protected output. Manually delete ONLY:\n'+str(out)+'\nThen rerun 65Z-A.')
    out.mkdir(parents=True,exist_ok=True)
    rows=[];gates={}
    for seg,m,path in plans:
        ds=common.segment_data(root,seg)
        ck=common.load_checkpoint(path)
        if not ck.get('65Y_finetune',{}).get('train_split_only'):
            raise RuntimeError('Checkpoint is not a 65Y-only-train candidate: '+str(path))
        model=common.make_model(ck,device).eval()
        branch,node,ip=common.scaled_inputs(ck,ds)
        ti=ds['splits']['train'];vi=ds['splits']['validation'];idx=np.concatenate((ti,vi))
        U=common.corrected_U(model,ck,ds,root,branch[idx],node,device)
        B=ds['B_center'];connectivity=ds['element_node_indices']
        k=np.stack([common.log_material_strain_eng(u,connectivity,B) for u in U],axis=0)
        if not np.isfinite(k).all():raise RuntimeError('Non-finite predicted kinematic strain')
        # The kinematic target is FIXED: mechanics-only tuning cannot alter U.
        with torch.no_grad():
            coords=torch.tensor(ip,dtype=torch.float32,device=device)
            lepred=[]
            for start in range(0,len(idx),4):
                b=torch.tensor(branch[idx[start:start+4]],dtype=torch.float32,device=device)
                lepred.append(model.strain_operator(b,coords).cpu().numpy())
        lepred=np.concatenate(lepred,axis=0)*np.asarray(ck['LE_std'])+np.asarray(ck['LE_mean'])
        nv=len(vi);truth=ds['LE'][vi].astype(np.float64)
        kinase=k[-nv:].astype(np.float64)
        learnt=lepred[-nv:].astype(np.float64)
        per={}
        for j,c in enumerate(common.COMPONENTS):
            err_kin=common.relative_l2(kinase[:,:,j],truth[:,:,j]);err_le=common.relative_l2(learnt[:,:,j],truth[:,:,j])
            # Normalized MSE is more stable than per-component relative L2 for nearly-zero reference fields.
            sd=max(float(np.asarray(ck['LE_std'])[j]),1e-8)
            mse_k=float(np.mean(((kinase[:,:,j]-truth[:,:,j])/sd)**2))
            mse_l=float(np.mean(((learnt[:,:,j]-truth[:,:,j])/sd)**2))
            ratio=mse_k/max(mse_l,1e-12)
            rows.append({'region':seg,'member':m,'component':'LE'+c,'val_cases':nv,
                         'kin_from_predicted_U_rel_L2_pct':err_kin,
                         'learned_LE_rel_L2_pct':err_le,
                         'kin_MSE_over_LE_MSE':ratio,'reference_logU_max_rel_L2_pct':reference_worst})
            per[c]=ratio
        # Gate on MSE ratios of three priority strains; conservative safety filter.
        permitted=all(np.isfinite(per[k]) and per[k]<=args.max_kinematic_ratio for k in ('11','22','12'))
        gates['%s/member_%02d'%(seg,m)]={'kinematic_loss_permitted':bool(permitted),
              'critical_LE_MSE_ratios':{'LE'+c:float(per[c]) for c in ('11','22','12')},
              'candidate_checkpoint_sha256':common.sha256(path),
              'reference_validation_cases':nv}
        member_folder=out/seg;member_folder.mkdir(parents=True,exist_ok=True)
        cache_file=member_folder/('member_%02d_kin.npz'%m)
        np.savez_compressed(cache_file,indices=idx.astype(np.int64),
            strain_logU_eng=k.astype(np.float32),
            checkpoint_sha256=np.array(common.sha256(path)),
            segment=np.array(seg),member=np.array(m,dtype=np.int64))
        print('65Z-A %-11s member %d: predicted-logU gate %s; LE11/22/12 MSE ratios %s'%(
            seg,m,'PASS' if permitted else 'FAIL',
            ', '.join('%.2f'%per[z] for z in ('11','22','12'))),flush=True)
        del model
        if device.type=='mps':torch.mps.empty_cache()
    df=pd.DataFrame(rows)
    df.to_csv(out/'65Z_A_member_metrics.csv',index=False)
    report={'step':'65Z-A','reference_logU_verified_max_rel_L2_pct':reference_worst,
       'data_cohort':'Original 64J train + validation only; original test excluded',
       'displacement_source':'65Y checkpoint frozen U operator + original 64N hard-compatibility correction',
       'kinematics':'E=.5*matrix_log(F.T F), F=I+grad_X(U), engineering shear',
       'maximum_allowed_kinematic_MSE_ratio':args.max_kinematic_ratio,
       'gates':gates,'number_kinematic_gates_passed':sum(v['kinematic_loss_permitted'] for v in gates.values()),
       'number_members':len(plans),
       'note':'Do NOT force nonlinear kinematic consistency onto LE when predicted-U gradients fail the gate. Reference-U kinematic validity is a different question.'}
    (out/'65Z_A_summary.json').write_text(json.dumps(report,indent=2))
    print('\n65Z-A FINISHED: %d/%d predicted-U nonlinear-kinematic gates passed'%(report['number_kinematic_gates_passed'],len(plans)))
    print('Report:',out/'65Z_A_summary.json')

if __name__=='__main__':main()
