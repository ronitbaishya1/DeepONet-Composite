#!/usr/bin/env python3
"""65AD-B: Compare physics-derived stress from projected strain against Abaqus.

No training. Runs only on the already used 64J development validation cohort.
All S, LE values are taken from the original 64J dataset. Abaqus S is Cauchy,
so C_linear @ material log-strain is only a validated benchmark approximation.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ad=importlib.import_module('65AD_common')
ac=ad.ac


def mech_head_predictions(st,ids,device,batch_size=4):
    ck=st['ck'];ds=st['ds']
    net=ac.aa.co.make_model(ck,device).eval()
    bnorm,_,ipnorm=ac.aa.co.scaled_inputs(ck,ds)
    coords=torch.tensor(np.array(ipnorm,copy=True),dtype=torch.float32,device=device)
    le=[];s=[]
    with torch.no_grad():
        for k in range(0,len(ids),batch_size):
            t=torch.tensor(np.array(bnorm[ids[k:k+batch_size]],copy=True),dtype=torch.float32,device=device)
            le.append(net.strain_operator(t,coords).detach().cpu().numpy())
            s.append(net.stress_operator(t,coords).detach().cpu().numpy())
    L=np.concatenate(le)*np.asarray(ck['LE_std'])+np.asarray(ck['LE_mean'])
    S=np.concatenate(s)*np.asarray(ck['S_std'])+np.asarray(ck['S_mean'])
    return L.astype(np.float64),S.astype(np.float64)


def C_times_LE(LE,C):
    return np.einsum('nec,ndc->ned',LE,C,optimize=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.members<=5:raise ValueError('members must be 1-5')
    root=Path(args.project).expanduser().resolve();dev=ac.aa.co.device_from_option(args.device)
    folder=root/'results'/ad.ROOT_NAME/'A_compatible_projection'
    if not (folder/'65AD_A_summary.json').is_file():raise FileNotFoundError('Run 65AD-A first')
    for seg in ad.SEGMENTS:
        for m in range(1,args.members+1):
            file=folder/seg/('member_%02d'%m)/'projection_validation.npz'
            if not file.is_file():raise FileNotFoundError(str(file))
    if args.check_only:
        print('65AD-B CHECK PASS: compatible projections and mechanics reference available')
        return
    out=ad.safe_output(root,'B_stress_audit')
    allrows=[];integrity=[]
    for seg in ad.SEGMENTS:
        for m in range(1,args.members+1):
            st=ac.prepare(root,seg,m,dev,need_predictions=False)
            ds=st['ds'];ids=ds['splits']['validation']
            path=folder/seg/('member_%02d'%m)/'projection_validation.npz'
            with np.load(path,allow_pickle=False) as dat:
                if str(dat['source_sha256'].item())!=st['source_sha']:
                    raise RuntimeError('Mismatch: original checkpoint changed')
                lgno_file=ac.checkpoint_path(root,'lgno',seg,m)
                if str(dat['lgno_checkpoint_sha256'].item())!=ac.aa.co.sha256(lgno_file):
                    raise RuntimeError('Mismatch: LGNO checkpoint changed')
                if not np.array_equal(dat['indices'],ids):raise RuntimeError('Validation cache indices mismatch')
                stored={field:np.array(dat[field],copy=True) for field in
                        ('baseline_G','lgno_G','compatible_G','baseline_U','compatible_U')}
                bd=np.asarray(dat['fixed_boundary'],int)
            if np.max(np.abs(stored['compatible_U'][:,bd,:]-stored['baseline_U'][:,bd,:]))>1e-8:
                raise RuntimeError('Projection does NOT preserve interface displacement')
            if np.max(np.abs(ac.gradient_from_u(stored['compatible_U'],ds)-stored['compatible_G']))>2e-5:
                raise RuntimeError('Projected gradient NOT equal to derivative of projected U')
            ref_LE=ds['LE'][ids];ref_S=ds['S'][ids]
            learned_LE,learned_S=mech_head_predictions(st,ids,dev)
            C=ac.aa.co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][ids,:3]).astype(float)
            stress={
                'Abaqus_reference_LE_constitutive':C_times_LE(ref_LE,C),
                '65AA_learned_stress_head':learned_S,
                '65AA_constitutive_from_learned_LE':C_times_LE(learned_LE,C),
                '65AA_constitutive_from_U':C_times_LE(ac.logstrain_from_gradient(stored['baseline_G']),C),
                '65AC_constitutive_from_LGNO_gradient':C_times_LE(ac.logstrain_from_gradient(stored['lgno_G']),C),
                '65AD_constitutive_from_compatible_U':C_times_LE(ac.logstrain_from_gradient(stored['compatible_G']),C),
            }
            for model,S in stress.items():
                for k,component in enumerate(ad.COMPONENTS):
                    allrows.append(dict(segment=seg,member=m,model=model,component='S'+component,
                          rel_l2_pct=ad.rel2(S[:,:,k],ref_S[:,:,k]),
                          RMSE_MPa=float(np.sqrt(np.mean((S[:,:,k]-ref_S[:,:,k])**2))),
                          validation_cases=len(ids)))
            ref_const=stress['Abaqus_reference_LE_constitutive']
            ref_const_err=ad.rel2(ref_const[:,:,1],ref_S[:,:,1])
            integrity.append(dict(segment=seg,member=m,reference_CLE_S22_error_pct=ref_const_err,
                interface_displacement_unchanged=True,projected_gradient_compatible=True,
                reference_constitutive_S22_within_2pct=bool(ref_const_err<2.)))
            print('65AD-B',seg,'C(refLE) S22 %.3f%%'%ref_const_err,
                  'learned-head S22 %.2f%%'%ad.rel2(learned_S[:,:,1],ref_S[:,:,1]),
                  'projected-C(LE) S22 %.2f%%'%ad.rel2(stress['65AD_constitutive_from_compatible_U'][:,:,1],ref_S[:,:,1]),flush=True)
    pd.DataFrame(allrows).to_csv(out/'65AD_B_stress_comparison.csv',index=False)
    pd.DataFrame(integrity).to_csv(out/'65AD_B_constitutive_sanity.csv',index=False)
    report={'step':'65AD-B','cases':'64J original development-validation only',
       'mechanics':'Material log strain with engineering shear, mapped through case-dependent orthotropic stiffness',
       'not_exact_finite_deformation_constitutive_update':True,
       'integrity_checks':integrity,
       'warning':'Derived constitutive stress is not the predicted stress head or a coupled Abaqus solution.'}
    (out/'65AD_B_summary.json').write_text(json.dumps(report,indent=2))
    print('65AD-B DONE:',out)

if __name__=='__main__':main()
