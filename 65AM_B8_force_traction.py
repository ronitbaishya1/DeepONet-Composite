#!/usr/bin/env python3
"""65AM-B8: frozen B5 vs Abaqus, OFFLINE internal-force / interface-traction audit.

Mac only; does not modify 65V, any checkpoint, or prior result.

This is NOT a proof of assembled FEM-NO equilibrium:
* uses full-FEM-provided interfaces (B6 reference/oracle boundary)
* uses an approximate orthotropic Cauchy stress proxy verified in B7
* no externally applied force, contact, hourglass, or neighboring FEM patch
  nodal traction is available in the existing B6 exports

Reference internal forces are computed using one-point C3D8R virtual work
using P = det(F)*sigma*F^{-T} and the saved reference-configuration shape
function derivatives, with convex-hex reference volume. This is an approximate
INTERNAL FORCE comparison, not Abaqus's complete residual vector.

Run:
 python3 65AM_B8_force_traction.py --selftest
 python3 65AM_B8_force_traction.py --project . --check-only
 python3 65AM_B8_force_traction.py --project .
"""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ('inner_left', 'inner_right')
OUT = '65AM_B8_mechanics_audit'


def s_tensor(a):
    """Abaqus S=(11,22,33,12,13,23), assumed global coordinate system."""
    x = np.asarray(a, dtype=np.float64)
    if x.shape[-1] != 6:
        raise ValueError('Expected six stress components [11,22,33,12,13,23]')
    result = np.zeros(x.shape[:-1]+(3,3), dtype=np.float64)
    result[...,0,0],result[...,1,1],result[...,2,2] = x[...,0],x[...,1],x[...,2]
    result[...,0,1] = result[...,1,0] = x[...,3]
    result[...,0,2] = result[...,2,0] = x[...,4]
    result[...,1,2] = result[...,2,1] = x[...,5]
    return result


def deformations(U, conn, dN):
    """F = I + sum_a u_a tensor grad_X(N_a) at C3D8R integration point."""
    g = np.einsum('eai,eaj->eij', U[conn], dN, optimize=True)
    F = np.eye(3)[None,:,:]+g
    det = np.linalg.det(F)
    if not np.isfinite(F).all() or np.min(det) <= 0:
        raise RuntimeError('Invalid/inverted deformation gradient; cannot transform stress')
    return F,det


def first_piola(stress,F,J):
    return J[:,None,None] * (s_tensor(stress) @ np.linalg.inv(F).swapaxes(-1,-2))


def relative(pred,ref):
    a=np.asarray(pred,dtype=np.float64).ravel()
    b=np.asarray(ref,dtype=np.float64).ravel()
    return float(100.0*np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def rmse(pred,ref):
    return float(np.sqrt(np.mean((np.asarray(pred)-np.asarray(ref))**2)))


def geometry(ds):
    """Reference geometry. Convex hull provides exact volume for convex planar-faced bricks."""
    from scipy.spatial import ConvexHull
    x=np.asarray(ds['coordinates'],dtype=np.float64)
    conn=np.asarray(ds['element_node_indices'],dtype=np.int64)
    dN=np.asarray(importlib.import_module('65AA_common').element_dN_numpy(ds),dtype=np.float64)
    if conn.ndim!=2 or conn.shape[1]!=8 or np.any(conn<0) or np.max(conn)>=len(x):
        raise RuntimeError('Invalid C3D8R connectivity')
    if dN.shape!=(len(conn),8,3):raise RuntimeError('Invalid saved C3D8R B matrix')
    if np.max(abs(dN.sum(axis=1)))>1e-5:raise RuntimeError('Shape function gradient partition-of-unity failure')
    pos=x[conn]
    affine=np.einsum('eai,eaj->eij',pos,dN,optimize=True)
    affine_err=float(np.max(abs(affine-np.eye(3))))
    if affine_err>1e-3:raise RuntimeError('Saved derivative geometry fails affine test')
    volume=np.asarray([ConvexHull(c).volume for c in pos],dtype=np.float64)
    if np.min(volume)<=0 or not np.isfinite(volume).all():raise RuntimeError('Invalid element volume')
    return x,conn,dN,volume,affine_err


def internal_forces(P,conn,dN,vol,n_nodes):
    """Internal nodal forces from 1-pt virtual work, excludes hourglass/contact."""
    f_element=vol[:,None,None]*np.einsum('eij,eaj->eai',P,dN,optimize=True)
    f_nodes=np.zeros((n_nodes,3),dtype=np.float64)
    np.add.at(f_nodes,conn.reshape(-1),f_element.reshape(-1,3))
    return f_nodes


def face_area_yz(face_xyz):
    """Planar quadrilateral reference face area, evaluated in YZ plane."""
    p=np.asarray(face_xyz,dtype=np.float64)[:,1:]
    if p.shape!=(4,2):raise RuntimeError('Interface cross-section is not a four-node face')
    c=p.mean(axis=0)
    ordered=p[np.argsort(np.arctan2(p[:,1]-c[1],p[:,0]-c[0]))]
    area=0.5*abs(np.dot(ordered[:,0],np.roll(ordered[:,1],-1))-
                 np.dot(ordered[:,1],np.roll(ordered[:,0],-1)))
    if area<=0:raise RuntimeError('Zero face area')
    return area


def faces_at_ends(x,conn):
    """Exactly four vertices on an axial end for each boundary element."""
    xmin=float(x[:,0].min());xmax=float(x[:,0].max())
    if xmax<=xmin:raise RuntimeError('Degenerate region axial extent')
    tol=max(1e-5,1e-7*(xmax-xmin))
    result={}
    for side,val,nx in (('left',xmin,-1.),('right',xmax,1.)):
        selection=[]
        for e,nodes in enumerate(conn):
            ids=nodes[abs(x[nodes,0]-val)<tol]
            if len(ids)==4:
                selection.append((e,ids,face_area_yz(x[ids])))
            elif len(ids) not in (0,):
                # Nonmatching end face could be partly cut by selection plane.
                raise RuntimeError(f'{side}: element has partially selected axial end face')
        if not selection:raise RuntimeError('No four-node end faces found: '+side)
        result[side]=(selection,np.array([nx,0.,0.],dtype=np.float64))
    return result


def face_forces(P,faces):
    out={}
    for side,(selection,N) in faces.items():
        elems=np.asarray([v[0] for v in selection],dtype=int)
        areas=np.asarray([v[2] for v in selection],dtype=np.float64)
        vectors=np.einsum('eij,j->ei',P[elems],N)*areas[:,None]
        out[side]=dict(force_by_face=vectors,area=float(areas.sum()),
                       resultant=vectors.sum(axis=0),n_faces=len(elems))
    return out


def selftest():
    """An affine single-brick test of strain, P, nodal force and face resultant."""
    from scipy.spatial import ConvexHull
    x=np.array([[xx,yy,zz] for xx in (0.,2.) for yy in (0.,1.5) for zz in (0.,1.)],dtype=float)
    # Center derivatives for affine brick: (signs)/(4*length)
    side=np.where(x[:,0]>0,1.,-1.)
    sy=np.where(x[:,1]>0,1.,-1.)
    sz=np.where(x[:,2]>0,1.,-1.)
    dN=np.stack((side/(4*2.),sy/(4*1.5),sz/(4*1.)),axis=1)[None]
    conn=np.arange(8).reshape(1,8)
    testgrad=np.array([[0.001,0.002,0.003],[0.0004,-0.002,0.001],[0.001,0.,0.0008]])
    U=x@testgrad.T + np.array([0.02,-0.03,0.01])[None,:]
    F,J=deformations(U,conn,dN)
    assert np.max(abs(F[0]-np.eye(3)-testgrad))<1e-12
    s=np.array([[18.,30.,7.,3.,2.,1.]])
    P=first_piola(s,F,J)
    v=np.array([ConvexHull(x).volume])
    f=internal_forces(P,conn,dN,v,len(x))
    assert np.linalg.norm(f.sum(axis=0))<1e-10
    faces=face_forces(P,faces_at_ends(x,conn))
    assert abs(faces['left']['area']-1.5)<1e-12
    assert abs(faces['right']['area']-1.5)<1e-12
    assert np.linalg.norm(faces['left']['resultant']+faces['right']['resultant'])<1e-10
    assert abs(float(v[0])-3.)<1e-12
    print('B8 SELFTEST PASS | affine 3D C3D8R | Piola transform | nodal force | end-face tractions')


def load_data(root,dev):
    b7=importlib.import_module('65AM_B7_mac')
    b6=b7.b6; b=b7.b
    df,models,packs,audit,_=b7.load_b6(root,dev)
    summary=root/'results'/'65AM_B7_stress_consistency'/'A_constitutive_audit'/'B7_summary.json'
    if not summary.is_file():raise FileNotFoundError('B7 summary missing: '+str(summary))
    j=json.loads(summary.read_text())
    if not j.get('law_proxy_acceptable_for_both'):
        raise RuntimeError('B7 did not verify approximate orthotropic law for both regions')
    if j.get('cases') != df.case_id.astype(str).tolist():
        raise RuntimeError('B7 case IDs changed')
    return b7,b6,b,df,models,packs,audit


def run(root,device,check_only):
    b7,b6,b,df,models,packs,audit=load_data(root,device)
    mats=df[list(b6.COLS)].to_numpy(np.float64)
    Cs=b7.constitutive_stiffness(mats,audit['fixed_orthotropic_constants'])
    cases=df.case_id.astype(str).tolist()
    prepared={}
    for region in REGIONS:
        ds=models[region][0]['ds']
        x,conn,dN,V,affine_err=geometry(ds)
        faces=faces_at_ends(x,conn)
        prepared[region]=(x,conn,dN,V,faces,affine_err)
        print(f'B8 PREFLIGHT {region} | nodes={len(x)} elements={len(conn)} | '
              f'faces={len(faces["left"][0])}/{len(faces["right"][0])} | affine={affine_err:.3g}')
    if check_only:
        print('B8 CHECK ONLY PASS. No result folders or files created.')
        return
    output=root/'results'/OUT/'A_offline_force_traction'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Existing B8 results protected; do not overwrite '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    force_rows=[];traction_rows=[];geometry_rows=[];details=[]
    for region,(d,model,ck) in models.items():
        ds=d['ds'];pack=packs[region]
        x,conn,dN,vol,faces,affine_err=prepared[region]
        geom=dict(region=region,reference_nodes=len(x),elements=len(conn),
                  volume_total_mm3=float(vol.sum()),volume_min_mm3=float(vol.min()),
                  C3D8R_affine_error=affine_err,
                  left_interface_area_mm2=faces['left'][0] and float(sum(t[2] for t in faces['left'][0])),
                  right_interface_area_mm2=faces['right'][0] and float(sum(t[2] for t in faces['right'][0])))
        geometry_rows.append(geom)
        Uref=np.asarray(pack['U'],dtype=np.float64)
        Sref=np.asarray(pack['S'],dtype=np.float64)
        branch,lift,fixed,diag=b6.project_boundary(root,ds,Uref,mats,device)
        Uhat=b6.predict_b5(d,model,branch,lift,device)
        if np.max(abs(Uhat[:,fixed,:]-lift[:,fixed,:]))>1e-5:
            raise RuntimeError('B5 interface displacement mismatch '+region)
        free=np.setdiff1d(np.arange(len(x)),np.asarray(fixed,dtype=int))
        if not len(free):raise RuntimeError('No non-interface nodes for virtual-work audit')
        if np.max(np.abs(np.asarray(pack['coordinates'])-x))>1e-5:
            raise RuntimeError('Reference coordinates not aligned')
        for k,case in enumerate(cases):
            LEhat=b.CO.log_material_strain_eng(Uhat[k],conn,ds['B_center'])
            Shat=b7.stress_from_LE(LEhat[None,:,:],Cs[k:k+1])[0]
            Fref,Jref=deformations(Uref[k],conn,dN)
            Fhat,Jhat=deformations(Uhat[k],conn,dN)
            Pref=first_piola(Sref[k],Fref,Jref)
            Phat=first_piola(Shat,Fhat,Jhat)
            fref=internal_forces(Pref,conn,dN,vol,len(x))
            fhat=internal_forces(Phat,conn,dN,vol,len(x))
            Fnorm=max(np.linalg.norm(fref),1e-12)
            case_force=dict(region=region,case_id=case,
                 internal_nodal_force_diff_rel_pct=relative(fhat,fref),
                 internal_nodal_force_diff_rmse_N=rmse(fhat,fref),
                 internal_free_node_force_diff_rel_global_pct=float(100*np.linalg.norm(fhat[free]-fref[free])/Fnorm),
                 FEM_internal_free_force_rel_global_pct=float(100*np.linalg.norm(fref[free])/Fnorm),
                 B5_internal_free_force_rel_global_pct=float(100*np.linalg.norm(fhat[free])/Fnorm),
                 net_FEM_internal_force_norm_N=float(np.linalg.norm(fref.sum(axis=0))),
                 net_B5_internal_force_norm_N=float(np.linalg.norm(fhat.sum(axis=0))),
                 J_FEM_min=float(Jref.min()),J_B5_min=float(Jhat.min()),
                 U_interface_abs_max_mm=float(np.max(abs(Uhat[k,fixed]-lift[k,fixed]))),
                 S22_rel_pct=relative(Shat[:,1],Sref[k,:,1]))
            force_rows.append(case_force)
            ref_forces=face_forces(Pref,faces)
            pred_forces=face_forces(Phat,faces)
            for side in ('left','right'):
                rf=ref_forces[side];pf=pred_forces[side]
                traction_rows.append(dict(region=region,case_id=case,side=side,
                     face_count=rf['n_faces'],area_mm2=rf['area'],
                     interface_traction_force_rel_pct=relative(pf['force_by_face'],rf['force_by_face']),
                     interface_resultant_diff_N=float(np.linalg.norm(pf['resultant']-rf['resultant'])),
                     interface_resultant_ref_norm_N=float(np.linalg.norm(rf['resultant'])),
                     FEM_Fx_N=float(rf['resultant'][0]),FEM_Fy_N=float(rf['resultant'][1]),
                     FEM_Fz_N=float(rf['resultant'][2]),
                     B5_Fx_N=float(pf['resultant'][0]),B5_Fy_N=float(pf['resultant'][1]),
                     B5_Fz_N=float(pf['resultant'][2])))
    fdf=pd.DataFrame(force_rows);tdf=pd.DataFrame(traction_rows);gdf=pd.DataFrame(geometry_rows)
    fdf.to_csv(output/'B8_internal_force_cases.csv',index=False)
    tdf.to_csv(output/'B8_interface_traction_cases.csv',index=False)
    gdf.to_csv(output/'B8_geometry_audit.csv',index=False)
    pooled=[]
    for region in REGIONS:
        a=fdf[fdf.region==region];t=tdf[tdf.region==region]
        pooled.append(dict(region=region,n_cases=len(a),
             mean_internal_force_diff_pct=float(a.internal_nodal_force_diff_rel_pct.mean()),
             max_internal_force_diff_pct=float(a.internal_nodal_force_diff_rel_pct.max()),
             mean_free_node_force_diff_pct=float(a.internal_free_node_force_diff_rel_global_pct.mean()),
             mean_ref_free_node_residual_indicator_pct=float(a.FEM_internal_free_force_rel_global_pct.mean()),
             mean_B5_free_node_residual_indicator_pct=float(a.B5_internal_free_force_rel_global_pct.mean()),
             mean_interface_traction_error_pct=float(t.interface_traction_force_rel_pct.mean()),
             max_interface_traction_error_pct=float(t.interface_traction_force_rel_pct.max()),
             mean_interface_resultant_diff_N=float(t.interface_resultant_diff_N.mean())))
    pd.DataFrame(pooled).to_csv(output/'B8_pooled_metrics.csv',index=False)
    report=dict(step='65AM-B8-A',cohort=cases,regions=REGIONS,
      source='Frozen B5 centered_grad; six B6 Abaqus cases; B7 C@LE proxy',
      FEM_oracle_interface_conditions=True,full_coupling_tested=False,
      full_force_equilibrium_proven=False,
      total_external_force_contact_hourglass_included=False,
      assumptions=['C3D8R one-point integration at original element centers',
        'Abaqus Cauchy S interpreted in global coordinate system',
        'Reference volume approximated using convex hexahedral volume',
        'Nominal traction computed as P*N in reference configuration',
        'Only NO-region local boundary faces, not FE-side neighboring tractions'],
      blocked_next_step='Do not replace 65V until actual coupled FEM-side interface force and external/contact force residuals are available and audited',
      pooled=pooled)
    (output/'B8_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for reg in REGIONS:
        ff=fdf[fdf.region==reg];tt=tdf[tdf.region==reg]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(ff.case_id,ff.internal_nodal_force_diff_rel_pct,marker='o',label='Internal force mismatch')
        for s in ('left','right'):
            v=tt[tt.side==s]
            ax.plot(v.case_id,v.interface_traction_force_rel_pct,marker='s',label=s+' interface traction mismatch')
        ax.set_title('65AM B8 | '+reg+' | offline stress-derived force error')
        ax.set_ylabel('Relative L2 (%)');ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
        ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(output/(reg+'_B8_force_traction.png'),dpi=180);plt.close(fig)
    print('B8 AUDIT COMPLETE:',output)
    print(pd.DataFrame(pooled).to_string(index=False))
    print('IMPORTANT: No proof of full force equilibrium or complete seven-region coupling.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--device',default='auto',choices=['auto','mps','cpu'])
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.selftest:
        selftest();return
    b7=importlib.import_module('65AM_B7_mac')
    root=Path(args.project).expanduser().resolve()
    device=b7.b.CO.device_from_option(args.device)
    run(root,device,args.check_only)

if __name__=='__main__':main()
