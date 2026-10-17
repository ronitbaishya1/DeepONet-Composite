#!/usr/bin/env python3
"""Step 65AE: stress-aware gradient targeting + compatible projection.

Never uses FEM-reference stress at prediction time. The already trained 65AA
stress head supplies the S22 TARGET; original 65AC LGNO supplies gradient,
and original 65AD sparse projection enforces displacement compatibility.
Reference Abaqus stress is used ONLY to select a scalar beta on 64J TRAIN
and to report errors on 64J development VALIDATION. No test data accessed.

This is NOT a nonlinear equilibrium solution or an exact finite-strain law:
S ~ C(E1,E2,G12) @ Hencky LE is an audited benchmark approximation.
"""
import json,importlib
from pathlib import Path
import numpy as np
import torch
ac=importlib.import_module('65AC_common')
ad=importlib.import_module('65AD_common')
adb=importlib.import_module('65AD_B_stress_check')
ROOT='65AE_stress_aware_projection'
REGIONS=('inner_left','inner_right')
# Small predefined candidates. Choose only using ORIGINAL training cases.
BETAS=(0.0,0.25,0.5,1.0,2.0)


def relative(x,ref):return float(ac.rel2(x,ref))


def stiffness(ds,indices):
    return ac.aa.co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][indices,:3]).astype('float64')


def stress_from_grad(G,C):
    LE=ac.logstrain_from_gradient(G)
    return LE,np.einsum('bec,bdc->bed',LE,C,optimize=True)


def stress_target(G,C,head_S,beta,max_delta=.02):
    """Change ONLY 3 diagonal target gradients to aim at predicted head S22.

    First-order local sensitivity (Hencky approx):
       delta(S22) ~= C[1,0] delta(G11) + C[1,1] delta(G22)
                      + C[1,2] delta(G33).
    This is an INFERENCE-time transformation using *predicted* stress,
    never FEM truth. The projection subsequently makes gradients compatible.
    """
    if beta<0:raise ValueError('beta must be >=0')
    _,sc=stress_from_grad(G,C)
    delta=np.asarray(head_S[:,:,1],float)-sc[:,:,1]
    coeff=np.asarray(C[:,1,:3],float)
    square=np.sum(coeff**2,axis=1)
    if np.any(square<=0):raise RuntimeError('Invalid stiffness for S22 coupling')
    d=beta*delta[:,:,None]*coeff[:,None,:]/square[:,None,None]
    d=np.clip(d,-max_delta,max_delta)
    out=np.array(G,dtype='float64',copy=True)
    for j in range(3):out[...,j,j]+=d[:,:,j]
    return out,dict(target_S22_gap_rms_MPa=float(np.sqrt(np.mean(delta**2))),
                    applied_gradient_delta_rms=float(np.sqrt(np.mean(d**2))),
                    clamp_fraction=float(np.mean(abs(d)>=max_delta-1e-12)))


def prepare(root,region,member,device):
    st=ac.prepare(root,region,member,device)
    lgfile=ac.checkpoint_path(root,'lgno',region,member)
    if not lgfile.is_file():raise FileNotFoundError('Missing 65AC LGNO checkpoint '+str(lgfile))
    lgno,lgck=ac.load_adapter(lgfile,'lgno',region,member,st['source_sha'],device)
    ds=st['ds'];boundary=ad.interface_nodes(root,ds)
    solver=ad.CompatibleProjector(ds,boundary,lam=.05)
    return dict(st=st,lgno=lgno,ds=ds,solver=solver,boundary=boundary,
                source_sha=st['source_sha'],lgno_sha=ac.aa.co.sha256(lgfile))


def batch_inputs(data,ids,device):
    st=data['st'];ds=data['ds']
    positions=np.asarray([st['index_map'][int(i)] for i in ids],int)
    U0=np.asarray(st['pred'][positions],float)
    GL=np.asarray(ac.split_predict(st,data['lgno'],ids,device,batch=4),float)
    _,heads=adb.mech_head_predictions(st,ids,device)
    C=stiffness(ds,ids)
    return U0,GL,heads,C


def project(data,U0,G):
    U=[];GG=[];notes=[]
    for u,g in zip(U0,G):
        up,gp,note=data['solver'].reconstruct(u,g)
        U.append(up);GG.append(gp);notes.append(note)
    U=np.asarray(U);GG=np.asarray(GG)
    fixed=data['boundary']
    if np.max(np.abs(U[:,fixed,:]-U0[:,fixed,:]))>1e-10:
        raise RuntimeError('FE-NO interface nodes changed')
    check=ac.gradient_from_u(U,data['ds'])
    if np.max(abs(check-GG))>2e-5:raise RuntimeError('Incompatible reconstructed gradient')
    return U,GG,notes


def evaluate(data,ids,U,G,C,learned_s=None):
    ds=data['ds'];target_U=ds['U'][ids];ref_LE=ds['LE'][ids];ref_S=ds['S'][ids]
    L,S=stress_from_grad(G,C)
    out=dict(U_vector_pct=relative(U,target_U),U2_pct=relative(U[...,1],target_U[...,1]),
             gradient_pct=relative(G,ac.gradient_from_u(target_U,ds)),
             LE11_pct=relative(L[...,0],ref_LE[...,0]),
             LE22_pct=relative(L[...,1],ref_LE[...,1]),
             LE12_pct=relative(L[...,3],ref_LE[...,3]),
             S11_pct=relative(S[...,0],ref_S[...,0]),
             S22_pct=relative(S[...,1],ref_S[...,1]),
             S12_pct=relative(S[...,3],ref_S[...,3]),
             S22_rmse_MPa=float(np.sqrt(np.mean((S[...,1]-ref_S[...,1])**2))))
    if learned_s is not None:
        out['learned_head_S22_pct']=relative(learned_s[...,1],ref_S[...,1])
    if not all(np.isfinite(float(v)) for v in out.values()):
        raise RuntimeError('Nonfinite stress/displacement evaluation')
    return out


def evaluation_safe(candidate,baseline):
    return (candidate['U2_pct']<=baseline['U2_pct']*1.10+.02 and
            candidate['U_vector_pct']<=baseline['U_vector_pct']*1.10+.02 and
            candidate['gradient_pct']<=baseline['gradient_pct']*1.10 and
            candidate['S11_pct']<=max(baseline['S11_pct']*1.10,baseline['S11_pct']+.5) and
            candidate['S12_pct']<=max(baseline['S12_pct']*1.10,baseline['S12_pct']+.5) and
            candidate['S22_pct']<baseline['S22_pct'])


def selection_objective(metrics):
    # All denominators are fixed in the same training cohort for fair choice.
    return float(metrics['S22_pct']+.05*metrics['S11_pct']+.05*metrics['S12_pct']+
                 .05*metrics['gradient_pct']+.20*metrics['U2_pct'])


def output(root,stage):
    path=root/'results'/ROOT/stage
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Output already exists. Manually delete ONLY this folder if rerunning:\n'+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path
