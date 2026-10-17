#!/usr/bin/env python3
"""65Y-A: Audit Abaqus LE shear conventions and finite-strain kinematics.

Mac only; READ ONLY on 65W and training data. No Abaqus job, no retraining.
Run from DeepONet_Composite root:
  python3 65Y_A_audit_reference_physics.py --project . --check-only
  python3 65Y_A_audit_reference_physics.py --project .

Compare actual full-FEM reference U, LE, S with:
  - B @ U small engineering strain,
  - log(V) Eulerian and log(Ustretch) Lagrangian Hencky strain,
  - orthotropic C @ LE with both engineering and tensor-shear hypotheses.
These are diagnostics, NOT exact Abaqus material integration / rotated output frames.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEGMENTS=('outer_left','inner_left','inner_right','outer_right')
REGIONS=('NO_OL','NO_L','NO_R','NO_OR')
CNAME=('11','22','33','12','13','23')


def score(pred,truth):
    p=np.asarray(pred,dtype=np.float64); t=np.asarray(truth,dtype=np.float64)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise RuntimeError('Incompatible/nonfinite prediction and reference arrays')
    delta=p-t; den=np.linalg.norm(t)
    return {'relative_l2_pct':float(100*np.linalg.norm(delta)/den) if den>1e-12 else None,
            'rmse':float(np.sqrt(np.mean(delta**2))),
            'reference_rms':float(np.sqrt(np.mean(t**2))),
            'max_abs':float(np.max(np.abs(delta)))}


def orthotropic_C():
    E1,E2,E3,nu12,nu13,nu23,G12,G13,G23=(46500.,12500.,12000.,.28,.28,.40,4750.,4500.,3500.)
    compliance=np.zeros((6,6),dtype=np.float64)
    compliance[np.diag_indices(6)]=[1/E1,1/E2,1/E3,1/G12,1/G13,1/G23]
    compliance[0,1]=compliance[1,0]=-nu12/E1
    compliance[0,2]=compliance[2,0]=-nu13/E1
    compliance[1,2]=compliance[2,1]=-nu23/E2
    C=np.linalg.inv(compliance)
    if np.min(np.linalg.eigvalsh(C))<=0:raise RuntimeError('Invalid orthotropic stiffness')
    return C


def index_coordinates(reference_xyz, query_xyz, tolerance=2e-4):
    """Map geometry by rounded coordinates, one-to-one and tolerance checked."""
    xyz=np.asarray(reference_xyz,dtype=float); q=np.asarray(query_xyz,dtype=float)
    if xyz.ndim!=2 or q.ndim!=2 or xyz.shape[1]!=3 or q.shape[1]!=3:
        raise RuntimeError('Expected Nx3 coordinate arrays')
    # Domain mesh spacing >> 1e-4 mm, so 1e-4 rounding is unambiguous.
    keys=[tuple(np.rint(c*1e4).astype(np.int64)) for c in xyz]
    if len(set(keys))!=len(keys):raise RuntimeError('Reference coordinates are not unique at tolerance')
    lut={v:i for i,v in enumerate(keys)}
    query_keys=[tuple(np.rint(c*1e4).astype(np.int64)) for c in q]
    missing=[v for v in query_keys if v not in lut]
    if missing:raise RuntimeError('Failed to match %d query points; first %s'%(len(missing),missing[0]))
    ix=np.asarray([lut[k] for k in query_keys],dtype=np.int64)
    if len(set(ix.tolist()))!=len(ix):raise RuntimeError('Duplicate query locations')
    distances=np.linalg.norm(xyz[ix]-q,axis=1)
    if np.max(distances)>tolerance:raise RuntimeError('Reference spatial tolerance exceeded')
    return ix, float(np.max(distances))


def vec6(symmetric,engineering_shear=False):
    v=np.stack([symmetric[:,0,0],symmetric[:,1,1],symmetric[:,2,2],
                symmetric[:,0,1],symmetric[:,0,2],symmetric[:,1,2]],axis=1)
    if engineering_shear:v[:,3:]*=2.0
    return v


def log_spd(x):
    eigvec=np.linalg.eigh(x)
    vals,vectors=eigvec
    if not np.isfinite(vals).all() or np.min(vals)<=0:
        raise RuntimeError('Non-positive Cauchy-Green tensor; check displacement gradients')
    return np.einsum('...ik,...k,...jk->...ij',vectors,np.log(vals),vectors)


def reference_candidates(U,conn,B):
    n=len(conn)
    if conn.shape!=(n,8) or B.shape!=(n,6,24):raise RuntimeError('C3D8R connectivity/B shapes mismatch')
    if np.min(conn)<0 or np.max(conn)>=len(U):raise RuntimeError('Connectivity out of range')
    Ue=U[conn]                                  # (E,8,3)
    gamma=np.einsum('eij,ej->ei',B,Ue.reshape(n,24))  # engineering shear
    # Original 64J center-point B rows 11/22/33 are dN/dX,dN/dY,dN/dZ.
    dN=np.stack([B[:,0,0::3],B[:,1,1::3],B[:,2,2::3]],axis=2)  # (E,8,3)
    G=np.einsum('eai,eaj->eij',Ue,dN)
    F=G+np.eye(3)[None,:,:]
    b=F@F.transpose(0,2,1)
    right=F.transpose(0,2,1)@F
    log_left=.5*log_spd(b)
    log_right=.5*log_spd(right)
    small=.5*(G+G.transpose(0,2,1))
    return {'linear_B_engineering':gamma,
            'linear_B_tensor':np.column_stack([gamma[:,:3],gamma[:,3:]/2]),
            'small_tensor':vec6(small,False),
            'small_engineering':vec6(small,True),
            'logV_spatial_tensor':vec6(log_left,False),
            'logV_spatial_engineering':vec6(log_left,True),
            'logU_material_tensor':vec6(log_right,False),
            'logU_material_engineering':vec6(log_right,True)}, float(np.max(np.linalg.norm(G,axis=(1,2))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    data=root/'data'/'hybrid_online_7region_65W_middle'/'validation_fields'
    nodes=data/'full_reference_nodes.csv'
    ip=data/'full_reference_ip.csv'
    paths=[nodes,ip]
    for seg in SEGMENTS:
        d=root/'data'/'hybrid_bulk_7region'/seg
        paths.extend([d/(x+'.npy') for x in ('coordinates','ip_coordinates','element_node_indices','B_center')])
    absent=[str(p) for p in paths if not p.is_file()]
    if absent:raise FileNotFoundError('Missing Step 65Y input files:\n'+'\n'.join(absent))
    if args.check_only:
        print('65Y-A check PASS: both full-FEM references and all four original segment geometries present')
        return
    full_node=pd.read_csv(nodes);full_ip=pd.read_csv(ip)
    required_nodes={'X','Y','Z','U1','U2','U3'}
    required_ip={'X','Y','Z'}|{p+c for p in ('LE','S') for c in CNAME}
    if required_nodes-set(full_node.columns) or required_ip-set(full_ip.columns):
        raise RuntimeError('65W reference CSV schema unexpected: %s / %s'%(list(full_node.columns),list(full_ip.columns)))
    xyz_nodes=full_node[['X','Y','Z']].to_numpy(float); Uall=full_node[['U1','U2','U3']].to_numpy(float)
    xyz_ip=full_ip[['X','Y','Z']].to_numpy(float)
    if len(full_node)!=4592 or len(full_ip)!=3640:
        raise RuntimeError('Unexpected middle-case full FEM sizes: %d nodes, %d mechanics points'%(len(full_node),len(full_ip)))
    C=orthotropic_C()
    records=[]; matching={}; all_true_LE=[];all_true_S=[]
    for seg,reg in zip(SEGMENTS,REGIONS):
        d=root/'data'/'hybrid_bulk_7region'/seg
        coords=np.load(d/'coordinates.npy'); ipcoords=np.load(d/'ip_coordinates.npy')
        conn=np.load(d/'element_node_indices.npy').astype(int); B=np.load(d/'B_center.npy')
        node_ix,node_distance=index_coordinates(xyz_nodes,coords)
        ip_ix,ip_distance=index_coordinates(xyz_ip,ipcoords)
        if len(ip_ix)!=len(conn):raise RuntimeError(reg+' element/IP count mismatch')
        predicted, max_gradient=reference_candidates(Uall[node_ix],conn,B)
        truthLE=full_ip.iloc[ip_ix][['LE'+c for c in CNAME]].to_numpy(float)
        truthS=full_ip.iloc[ip_ix][['S'+c for c in CNAME]].to_numpy(float)
        all_true_LE.append(truthLE);all_true_S.append(truthS)
        for method,eps in predicted.items():
            for j,c in enumerate(CNAME):
                records.append({'region':reg,'quantity':'LE'+c,'method':method,'points':len(eps),**score(eps[:,j],truthLE[:,j])})
        for hypothesis,multiplier in [('engineering_shear',1.0),('tensor_shear',2.0)]:
            eng=truthLE.copy();eng[:,3:]*=multiplier
            Sapprox=eng@C.T
            for j,c in enumerate(CNAME):
                records.append({'region':reg,'quantity':'S'+c,'method':'C_from_reference_LE_'+hypothesis,'points':len(eng),**score(Sapprox[:,j],truthS[:,j])})
        matching[reg]={'nodes':len(coords),'mechanics_points':len(conn),
                      'maximum_node_match_distance_mm':node_distance,
                      'maximum_ip_match_distance_mm':ip_distance,
                      'maximum_linearized_displacement_gradient_frobenius':max_gradient}
    frame=pd.DataFrame(records)
    # Decide convention using *reference* stress vs *reference* LE, both hypotheses, all shear axes.
    convention=[]
    for c in ('12','13','23'):
        for method in ('C_from_reference_LE_engineering_shear','C_from_reference_LE_tensor_shear'):
            sub=frame[(frame.quantity=='S'+c)&(frame.method==method)]
            convention.append({'shear_component':'S'+c,'method':method,
                               'aggregate_sse_proxy':float(np.sum((sub['rmse'].to_numpy()**2)*sub['points'].to_numpy())),
                               'mean_region_rel_l2_pct':float(sub['relative_l2_pct'].mean())})
    dest=root/'results'/'65Y_middle_mechanics'/'A_reference_physics_audit'
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Output folder exists. Manually delete ONLY this folder to rerun:\n'+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    frame.to_csv(dest/'65Y_A_kinematics_and_stress_comparison.csv',index=False)
    pd.DataFrame(convention).to_csv(dest/'65Y_A_shear_hypotheses.csv',index=False)
    report={'step':'65Y-A','reference_nodes':len(full_node),'reference_elements':len(full_ip),
            'material':[46500.,12500.,12000.,.28,.28,.40,4750.,4500.,3500.],
            'mapping':matching,'convention_comparison':convention,
            'readout':'Compare engineering vs tensor hypotheses against actual Abaqus reference, not external assumptions.',
            'warning':'Geometric nonlinear analysis: log(V) and log(U) computed at original B_center gradients are diagnostic. Abaqus corotated material frames, orientations, incremental material integration, reduced integration/hourglass behavior may differ. Do not adopt a physics penalty from this report alone.',
            'uses_training_test_labels_to_tune_models':False}
    (dest/'65Y_A_summary.json').write_text(json.dumps(report,indent=2))
    print('65Y-A COMPLETE: full FEM reference, C3D8R geometry, finite-log-strain diagnostics')
    print(pd.DataFrame(convention).to_string(index=False))
    print('Output:',dest)

if __name__=='__main__':main()
