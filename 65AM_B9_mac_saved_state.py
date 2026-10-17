#!/usr/bin/env python3
"""65AM-B9 Mac: SAFE B5 replay at the existing verified 65V 26-mode interface state.

NO Abaqus runs. NO live Broyden step. NO modification to 65V or checkpoints.
Transfer B9_65V_anchor.npz AND B9_65V_anchor_manifest.json from Windows
into <project>/B9_coupled_anchor using Finder.

Run:
  python3 65AM_B9_mac_saved_state.py --selftest
  python3 65AM_B9_mac_saved_state.py --project . --check-only
  python3 65AM_B9_mac_saved_state.py --project .

Output: results/65AM_B9_coupled_feasibility/A_saved_state/
Critical: candidate stress-based forces are NOT the same trained NO-force head
used in 65V. Residual estimates require FE-side re-verification in a NEW solver.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
SIDES={'inner_left':{'left':'left_inner','right':'center_left'},
       'inner_right':{'left':'center_right','right':'right_inner'}}
COUNTS=(4,4,5,5,4,4)
OUT=Path('results/65AM_B9_coupled_feasibility/A_saved_state')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def relative(p,t):
    p=np.asarray(p,dtype=np.float64).ravel();t=np.asarray(t,dtype=np.float64).ravel()
    return float(100*np.linalg.norm(p-t)/max(np.linalg.norm(t),1.e-12))


def load_anchor(root):
    folder=root/'B9_coupled_anchor'
    af=folder/'B9_65V_anchor.npz'; mf=folder/'B9_65V_anchor_manifest.json'
    if not af.is_file() or not mf.is_file():raise FileNotFoundError('Copy the TWO B9 Windows anchor files into '+str(folder))
    with mf.open() as f:md=json.load(f)
    if sha(af)!=md.get('anchor_sha256'):raise RuntimeError('B9 Windows anchor hash mismatch')
    with np.load(af,allow_pickle=False) as z:
        d={k:np.array(z[k],copy=True) for k in z.files}
    needed={'material','q','normalized_residual','force_scale','old_NO_g','FE_g_inferred'}
    needed|={'c_'+n for n in NAMES}
    for n in NAMES:
        needed|={'pca_basis_'+n,'pca_mean_'+n,'pca_coordinates_'+n}
    if needed-set(d):raise RuntimeError('B9 anchor missing '+str(sorted(needed-set(d))))
    for k in ('q','normalized_residual','force_scale','old_NO_g','FE_g_inferred'):
        if d[k].shape!=(26,) or not np.isfinite(d[k]).all():raise RuntimeError('Anchor shape or nonfinite '+k)
    if np.any(d['force_scale']<=0):raise RuntimeError('Invalid force scale')
    if tuple(len(d['c_'+n]) for n in NAMES)!=COUNTS:raise RuntimeError('Saved interface dimensions unexpectedly changed')
    r=np.asarray(d['normalized_residual'],float)
    rrec=(np.asarray(d['FE_g_inferred'],float)+np.asarray(d['old_NO_g'],float))/d['force_scale']
    if not np.allclose(rrec,r,rtol=0,atol=2e-8):raise RuntimeError('Windows FE/NO force anchor does not reproduce saved residual')
    if np.sqrt(np.mean(r**2))>.0200001:raise RuntimeError('Starting 65V state is not converged')
    if not np.allclose(d['material'],md['material_MPa'],rtol=0,atol=1e-4):raise RuntimeError('Material changed since Windows export')
    return d,md


def check_pca(d,ctx,side,name):
    src=ctx[side]
    if src is None or src['name']!=name:raise RuntimeError('Mac interface name does not match 65V: '+name)
    for key in ('basis','mean','coordinates'):
        src0=np.asarray(src[key],dtype=np.float64)
        other=np.asarray(d['pca_'+key+'_'+name],dtype=np.float64)
        if src0.shape!=other.shape:raise RuntimeError('Mac/Windows PCA shape mismatch for '+name+' '+key)
        err=float(np.max(abs(src0-other)))
        tol=1e-4 if key=='coordinates' else 3e-5
        if err>tol:raise RuntimeError('STOP: Mac/Windows PCA values differ for %s %s (%.6g)'%(name,key,err))
    sl=ctx[side+'_slice']
    if sl.stop-sl.start!=len(d['c_'+name]):raise RuntimeError('PCA branch modes mismatch: '+name)


def build_branch_lift(root,ds,d,ctx,device):
    import torch
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    region=ds['_b9_region']
    mat=np.asarray(d['material'],dtype=np.float64)
    p=int(ds['branch_inputs'].shape[1]);branch=np.zeros((1,p),dtype=np.float32)
    branch[0,:3]=mat.astype(np.float32)
    for side,name in SIDES[region].items():
        check_pca(d,ctx,side,name)
        branch[0,ctx[side+'_slice']]=d['c_'+name]
    with torch.no_grad():
        x=torch.as_tensor(branch,dtype=torch.float32,device=device)
        zeros=torch.zeros((1,len(ds['coordinates']),3),device=device,dtype=torch.float32)
        lift=apply_hard_compatibility_7region(zeros,x,ctx).cpu().numpy()
    return branch,lift


def generalized_from_face_stress(x,conn,P,faces,ctx,side):
    """Project C3D8R nominal interface face traction onto SAME nodal PCA basis.

    For constant traction per face, each of its four vertices receives F_face/4.
    This is a diagnostic stress-based generalized force; it is not automatically
    identical to the original independently trained NO generalized-force head.
    """
    info=ctx[side]
    pc=np.asarray(info['coordinates'],dtype=np.float64)
    basis=np.asarray(info['basis'],dtype=np.float64)
    if basis.shape[0]!=3*len(pc):raise RuntimeError('PCA mode basis rows do not match interface node DOFs')
    f=np.zeros((len(pc),3),dtype=np.float64)
    select,normal=faces[side]
    # Strict geometry index mapping; no nearest mismatched nodes accepted.
    def node_id(v):
        dist=np.linalg.norm(pc-v[None,:],axis=1)
        j=int(np.argmin(dist))
        if dist[j]>1.e-4:raise RuntimeError('C3D8R face node missing from PCA boundary')
        return j
    for ei,ids,area in select:
        face_traction=np.asarray(P[ei],float)@normal
        for idx in ids:
            f[node_id(x[idx])] += area*face_traction/4.
    g=np.asarray(basis.T@f.reshape(-1),dtype=np.float64)
    if not np.isfinite(g).all():raise RuntimeError('Nonfinite generalized B5 stress force')
    return g, f.sum(axis=0)


def selftest():
    x=np.array([(xx,yy,zz) for xx in (0.,2.) for yy in (0.,1.) for zz in (0.,1.)],float)
    conn=np.arange(8)[None,:]
    basis=np.eye(12,3)
    # At left face select 4 nodes, each should receive uniform force/4.
    inds=np.flatnonzero(x[:,0]==0.)
    pc=x[inds]
    ref=type('Ctx',(),{})
    ctx={'left':{'coordinates':pc,'basis':basis}}
    P=np.array([np.diag([10.,20.,30.])]);faces={'left':([(0,inds,1.)],np.array([-1.,0.,0.]))}
    g,rs=generalized_from_face_stress(x,conn,P,faces,ctx,'left')
    assert np.allclose(rs,[-10,0,0]) and len(g)==3
    assert np.isclose(g[0],-2.5)
    print('B9 MAC SELFTEST PASS: nominal face traction to nodal PCA generalized force')


def run(root,device,check_only):
    import torch
    b6=importlib.import_module('65AM_B6_mac')
    b7=importlib.import_module('65AM_B7_mac')
    b8=importlib.import_module('65AM_B8_force_traction')
    anchor,manifest=load_anchor(root)
    # Reuses B6 locked hashes; B7 validated small-deformation orthotropic proxy.
    _,models=b6.check(root,device)
    _,_,_,aud,_=b7.load_b6(root,device)
    b7_summary=json.loads((root/'results'/'65AM_B7_stress_consistency'/'A_constitutive_audit'/'B7_summary.json').read_text())
    if not b7_summary.get('law_proxy_acceptable_for_both'):
        raise RuntimeError('B7 material law proxy did not pass; do not generate B9 forces')
    mat=np.asarray(anchor['material'],float).reshape(1,3)
    C=b7.constitutive_stiffness(mat,aud['fixed_orthotropic_constants'])
    result=[];updates={}
    for region,(dsd,model,ck) in models.items():
        ds=dsd['ds'];ds['_b9_region']=region
        ctx=b6.b.CO.interface_context(root,ds,device)
        branch,lift=build_branch_lift(root,ds,anchor,ctx,device)
        U=b6.predict_b5(dsd,model,branch,lift,device)[0]
        fixed=b6.b.boundary_indices(ctx)
        if np.max(abs(U[fixed]-lift[0,fixed]))>1e-5:raise RuntimeError('Interface compatibility lost '+region)
        x,conn,dN,vol,aff=b8.geometry(ds)
        LE=b6.b.CO.log_material_strain_eng(U,conn,ds['B_center'])
        S=b7.stress_from_LE(LE[None,:,:],C)[0]
        F,J=b8.deformations(U,conn,dN)
        P=b8.first_piola(S,F,J)
        faces=b8.faces_at_ends(x,conn)
        out={'region':region,'U_norm_mm':float(np.linalg.norm(U)),
             'S_norm_MPa':float(np.linalg.norm(S)),
             'J_min':float(np.min(J)),
             'interface_U_error_max_mm':float(np.max(abs(U[fixed]-lift[0,fixed])))}
        for side,name in SIDES[region].items():
            g,resultant=generalized_from_face_stress(x,conn,P,faces,ctx,side)
            if len(g)!=len(anchor['c_'+name]):raise RuntimeError('Projected generalized force mode count mismatch '+name)
            updates[name]=g
            out[name+'_resultant_N']=resultant.tolist()
            out[name+'_g_norm']=float(np.linalg.norm(g))
        result.append(out)
        print('B9 CHECK PASS',region,'rank',dsd['rank'],'nodes',len(x),'cells',len(conn))
    old=np.asarray(anchor['old_NO_g'],float)
    new=old.copy();i=0;rows=[]
    for name,count in zip(NAMES,COUNTS):
        sli=slice(i,i+count)
        if name in updates:new[sli]=updates[name]
        for k in range(count):
            rows.append(dict(interface=name,mode_index=k+1,
                old_NO_generalized_force=float(old[i+k]),
                B5_stress_projected_force=float(new[i+k]) if name in updates else np.nan,
                inferred_65V_FE_force=float(anchor['FE_g_inferred'][i+k]),
                old_normalized_residual=float(anchor['normalized_residual'][i+k]),
                candidate_normalized_residual_estimate=float((anchor['FE_g_inferred'][i+k]+new[i+k])/anchor['force_scale'][i+k]),
                replaced_by_B5=bool(name in updates)))
        i+=count
    cand=(anchor['FE_g_inferred']+new)/anchor['force_scale']
    rms0=float(np.sqrt(np.mean(anchor['normalized_residual']**2)))
    rms1=float(np.sqrt(np.mean(cand*cand)))
    if check_only:
        print('B9 CHECK ONLY PASS: B5 replayed at saved 65V q without using new FEM interior results.')
        print('Old VERIFIED 65V RMS = %.7f; B5 fixed-FE-force RMS estimate = %.7f (NOT verified)'%(rms0,rms1))
        print('No results written. Re-run without --check-only to generate reports.')
        return
    out=root/OUT
    if out.exists() and any(out.iterdir()):raise RuntimeError('Existing B9 outputs protected; inspect in Finder: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'B9_interface_force_modes.csv',index=False)
    per=[];k=0
    for name,count in zip(NAMES,COUNTS):
        sl=slice(k,k+count);k+=count
        per.append(dict(interface=name,updated=bool(name in updates),
            old_NO_force_norm=float(np.linalg.norm(old[sl])),
            new_B5_force_norm=float(np.linalg.norm(new[sl])),
            B5_vs_old_force_pct=relative(new[sl],old[sl]) if name in updates else None,
            FE_force_norm=float(np.linalg.norm(anchor['FE_g_inferred'][sl])),
            old_verified_RMS=float(np.sqrt(np.mean(anchor['normalized_residual'][sl]**2))),
            B5_fixed_FE_RMS_estimate=float(np.sqrt(np.mean(cand[sl]**2)))))
    pd.DataFrame(per).to_csv(out/'B9_interface_summary.csv',index=False)
    report=dict(step='65AM-B9-A',phase='saved-65V-interface-state offline B5 replay',
        verified_65V_RMS=rms0,
        B5_substituted_RMS_with_frozen_FE_forces_ESTIMATE_ONLY=rms1,
        inferred_FE_force_definition='Recovered algebraically from old verified 65V normalized residual minus current old NO force head; not separately measured from final patch ODBs',
        old_NO_generalized_force_definition='65V existing neural generalized force heads',
        B5_force_definition='B5 displacement => B7 orthotropic proxy stress => Piola nominal stress => C3D8R face-force/4 nodal distribution => PCA virtual-work projection',
        updated_interfaces=list(updates),
        original_65V_unmodified=True, B5_model_frozen=True,
        no_FEM_recalculation_at_new_B5_response=True,
        not_a_converged_coupled_B5_solution=True,
        B5_replacement_approved=False,
        limitations=['Original NO force head and stress-derived B5 force are separately defined; interface sign/scale and work-conjugacy need independent FE validation',
            'No new Abaqus patch solves or iterative Broyden update performed',
            'External contact, reaction, hourglass residuals not completely assembled'],
        per_region=result,windows_anchor_sha256=manifest['anchor_sha256'])
    (out/'B9_summary.json').write_text(json.dumps(report,indent=2))
    print('B9 SAVED-STATE REPORTS:',out)
    print('Old verified RMS %.7f vs B5 substituted offline ESTIMATE %.7f'%(rms0,rms1))
    print('STOP: No live B5 Broyden coupling has been performed.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--device',default='auto',choices=('auto','cpu','mps'))
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest();return
    import torch
    dev=torch.device(('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device)
    run(Path(a.project).expanduser().resolve(),dev,a.check_only)

if __name__=='__main__':main()
