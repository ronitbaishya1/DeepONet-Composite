#!/usr/bin/env python3
"""65X-B / Mac: displacement-strain-stress consistency diagnostic.

Uses ORIGINAL 64J B_center matrices and node ordering, and the REAL 65W-H
V4 local_predictions. Compare the learned LE and S heads with:
  (1) linear C3D8R kinematic strain B @ U (tensor shear = gamma/2),
  (2) orthotropic C @ learned LE (tensor shear doubled),
  (3) orthotropic C @ kinematic strain.

These are intentionally SMALL-STRAIN APPROXIMATIONS. Original Abaqus run has
NLGEOM and LE logarithmic strain / Cauchy stress. Do NOT interpret a nonzero
constitutive residual as definitive physical inconsistency, or automatically
replace the learned heads with these approximations.

Run from project root:
 python3 65X_B_physics_consistency.py --project .
Rerun: manually delete only results/65X_middle_diagnostics/B_physics_consistency.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MAP={'outer_left':'NO_OL','inner_left':'NO_L','inner_right':'NO_R','outer_right':'NO_OR'}
COMPS=('11','22','33','12','13','23')


def stiffness():
    E1,E2,E3=46500.,12500.,12000.
    n12,n13,n23=.28,.28,.40
    G12,G13,G23=4750.,4500.,3500.
    # Identical compliance and ordering to the 64N model source.
    S=np.zeros((6,6),float)
    S[0,0],S[1,1],S[2,2]=1/E1,1/E2,1/E3
    S[0,1]=S[1,0]=-n12/E1
    S[0,2]=S[2,0]=-n13/E1
    S[1,2]=S[2,1]=-n23/E2
    S[3,3],S[4,4],S[5,5]=1/G12,1/G13,1/G23
    C=np.linalg.inv(S)
    if np.linalg.eigvalsh(C).min()<=0:raise RuntimeError('Non-positive orthotropic stiffness')
    return C


def c_times_tensor_strain(C,eps):
    eng=np.asarray(eps,dtype=float).copy()
    eng[:,3:]*=2.0
    return eng@C.T


def scores(pred,ref):
    pred=np.asarray(pred,float);ref=np.asarray(ref,float)
    nref=float(np.linalg.norm(ref))
    return dict(rel_l2_percent=float(100*np.linalg.norm(pred-ref)/nref) if nref>1e-12 else None,
                rmse=float(np.sqrt(np.mean((pred-ref)**2))),
                ref_rms=float(np.sqrt(np.mean(ref**2))),
                max_abs_error=float(np.max(abs(pred-ref))))


def nearest_one_to_one(src,query,tol=1e-4):
    """Query coords need one unique source coord each; validated with Euclidean distance."""
    src=np.asarray(src,dtype=float);query=np.asarray(query,dtype=float)
    if src.ndim!=2 or query.ndim!=2 or src.shape[1]!=3 or query.shape[1]!=3:
        raise RuntimeError('Invalid coordinate arrays')
    if len(query)!=len(src):raise RuntimeError('Expected one-to-one source/target elements %d/%d'%(len(src),len(query)))
    # Chunk avoids giant NxN allocation on M1 Max. <= ~784 per NO region.
    ix=[];dist=[]
    for start in range(0,len(query),150):
        Q=query[start:start+150]
        delta=np.sum((Q[:,None,:]-src[None,:,:])**2,axis=2)
        indexes=np.argmin(delta,axis=1)
        ix.extend(indexes.tolist())
        dist.extend(np.sqrt(delta[np.arange(len(indexes)),indexes]).tolist())
    ix=np.asarray(ix,dtype=int)
    dist=np.asarray(dist,float)
    if np.max(dist)>tol or len(np.unique(ix))!=len(query):
        raise RuntimeError('IP mapping failed tolerance/uniqueness (max=%.3g, unique=%d of %d)'%
                           (float(np.max(dist)),len(np.unique(ix)),len(query)))
    return ix,dist


def ensure_empty(path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Output exists. Manually delete only this folder before rerunning:\n'+str(path))
    path.mkdir(parents=True,exist_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    args=parser.parse_args()
    root=Path(args.project).expanduser().resolve()
    base=root/'results'/'hybrid_online_7region_65W_middle_validation'
    ip_path=base/'assembled_hybrid_ip.csv'
    if not ip_path.is_file():raise FileNotFoundError(str(ip_path))
    ip=pd.read_csv(ip_path)
    if len(ip)!=3640:raise RuntimeError('Need 3640 seven-region mechanics points')
    required={'Owner','X','Y','Z','MatchedHybridX','MatchedHybridY','MatchedHybridZ'}
    for kind in ('LE','S'):
        for c in COMPS:
            required.update(('FEM_'+kind+c,'Hybrid_'+kind+c))
    if required-set(ip.columns):raise RuntimeError('Missing mechanics columns: '+str(sorted(required-set(ip.columns))))
    out=root/'results'/'65X_middle_diagnostics'/'B_physics_consistency'
    ensure_empty(out)
    C=stiffness()
    records=[];case_audits={};point_rows=[]
    for segment,region in MAP.items():
        dat=root/'data'/'hybrid_bulk_7region'/segment
        pred_path=base/(segment+'_local_prediction.npz')
        if not pred_path.is_file():raise FileNotFoundError('Missing 65W-H segment prediction: '+str(pred_path))
        files=(dat/'element_node_indices.npy',dat/'B_center.npy',dat/'ip_coordinates.npy')
        for file in files:
            if not file.is_file():raise FileNotFoundError(str(file))
        with np.load(pred_path,allow_pickle=False) as f:
            U=np.asarray(f['U'],float)
            LE=np.asarray(f['LE'],float)
            S=np.asarray(f['S'],float)
            coords=np.asarray(f['ip_coordinates'],float)
            nodes=np.asarray(f['coordinates'],float)
        connectivity=np.load(files[0],allow_pickle=False)
        B=np.load(files[1],allow_pickle=False)
        expected=np.load(files[2],allow_pickle=False)
        nE=len(coords)
        if U.shape!=(len(nodes),3) or LE.shape!=(nE,6) or S.shape!=(nE,6) or B.shape!=(nE,6,24) or connectivity.shape!=(nE,8):
            raise RuntimeError(region+' local prediction or C3D8R tensors shape mismatch')
        if np.max(abs(coords-expected))>1e-5:
            raise RuntimeError(region+' trained IP coordinate order changed')
        if np.min(connectivity)<0 or np.max(connectivity)>=len(U):
            raise RuntimeError(region+' element connectivity points outside node array')
        if not all(np.isfinite(arr).all() for arr in (U,LE,S,coords,B)):
            raise RuntimeError(region+' contains NaN or Inf')
        # Exact 64N kinematic conversion (B matrices generate engineering shear).
        Ue=U[connectivity].reshape(nE,24)
        eps_kin=np.einsum('eij,ej->ei',B,Ue)
        eps_kin[:,3:]*=.5
        s_from_le=c_times_tensor_strain(C,LE)
        s_from_u=c_times_tensor_strain(C,eps_kin)
        target=ip[ip.Owner==region].copy().reset_index(drop=True)
        target_coords=target[['MatchedHybridX','MatchedHybridY','MatchedHybridZ']].to_numpy(float)
        loc,d=nearest_one_to_one(coords,target_coords)
        LE=LE[loc];S=S[loc];eps_kin=eps_kin[loc]
        s_from_le=s_from_le[loc];s_from_u=s_from_u[loc]
        if len(target)!=len(loc):raise RuntimeError('Local match failure')
        truth_le=target[['FEM_LE'+v for v in COMPS]].to_numpy(float)
        truth_s=target[['FEM_S'+v for v in COMPS]].to_numpy(float)
        # Audit reconstructed source agrees with saved hybrid assembly, avoiding wrong data.
        original_le=target[['Hybrid_LE'+v for v in COMPS]].to_numpy(float)
        original_s=target[['Hybrid_S'+v for v in COMPS]].to_numpy(float)
        if not np.allclose(LE,original_le,rtol=5e-5,atol=1e-7) or not np.allclose(S,original_s,rtol=5e-5,atol=1e-5):
            raise RuntimeError(region+' saved NO local results differ from 65W-H assembled hybrid values')
        case_audits[region]={'points':nE,'max_IP_matching_distance_mm':float(np.max(d)),
                             'linearized_strain_warning':'C3D8R B uses initial-configuration small strain; reference LE uses finite logarithmic strain.'}
        for j,c in enumerate(COMPS):
            for field,model,ref,reference in (
                ('LE','learned_LE',LE,truth_le),('LE','kinematic_B_U',eps_kin,truth_le),
                ('S','learned_S',S,truth_s),('S','constitutive_C_LE',s_from_le,truth_s),
                ('S','kinematic_constitutive_C_B_U',s_from_u,truth_s)):
                rec=scores(ref[:,j],reference[:,j])
                records.append(dict(region=region,field=field,component=field+c,method=model,points=nE,**rec))
            for label,values in (('S_learned_minus_C_LE',S-s_from_le),('LE_learned_minus_B_U',LE-eps_kin)):
                arr=values[:,j]
                point_rows.append(dict(region=region,component=label+'_'+c,
                                       abs_consistency_rmse=float(np.sqrt(np.mean(arr**2))),
                                       reference_rms=float(np.sqrt(np.mean((truth_s if label.startswith('S') else truth_le)[:,j]**2))),
                                       normalized_consistency_percent=(float(100*np.linalg.norm(arr)/max(np.linalg.norm((truth_s if label.startswith('S') else truth_le)[:,j]),1e-12)))))
    table=pd.DataFrame(records)
    table.to_csv(out/'65X_B_reference_error_comparison.csv',index=False)
    pd.DataFrame(point_rows).to_csv(out/'65X_B_internal_consistency.csv',index=False)
    (out/'65X_B_audit.json').write_text(json.dumps({
        'step':'65X-B','material':[46500.,12500.,12000.,.28,.28,.40,4750.,4500.,3500.],
        'matching':case_audits,'stiffness_eigenvalues':np.linalg.eigvalsh(C).tolist(),
        'reference_type':'Actual 65W-H full FEM reference fields, used for evaluation only',
        'interpretation':'C @ LE and B @ U are SMALL-STRAIN diagnostic candidates, not guaranteed true NLGEOM stresses/strains. The baseline model is unchanged. Do not select a final model using held-out FEM reference errors.',
        'results_file':'65X_B_reference_error_comparison.csv'},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for comp in ('S11','S12','S22','LE11','LE12','LE22'):
        sub=table[table.component==comp]
        methods=['learned_S','constitutive_C_LE','kinematic_constitutive_C_B_U'] if comp.startswith('S') else ['learned_LE','kinematic_B_U']
        fig,ax=plt.subplots(figsize=(10,4.5))
        x=np.arange(len(MAP))
        width=.8/len(methods)
        for k,method in enumerate(methods):
            s=sub[sub.method==method].set_index('region')
            vals=[s.loc[region,'rel_l2_percent'] for region in MAP.values()]
            ax.bar(x+(k-(len(methods)-1)/2)*width,vals,width,label=method)
        ax.set_xticks(x,[region for region in MAP.values()])
        ax.set(ylabel='Relative L2 error (%)',title=comp+': learned vs approximate physics heads')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25)
        fig.tight_layout();fig.savefig(out/('65X_B_'+comp+'_comparison.png'),dpi=180);plt.close(fig)
    print('65X-B PASS: all four trained NO regions, C3D8R kinematic and orthotropic constitutive checks')
    print('Saved candidate comparisons and plots to:',out)
    print(table[(table.component.isin(['S11','S12','S22'])) & (table.method.isin(['learned_S','constitutive_C_LE']))][['region','component','method','rel_l2_percent']].to_string(index=False))
    print('NOTE: C@LE and B@U are diagnostic approximations for NLGEOM; DO NOT replace your validated model from this test.')

if __name__=='__main__':main()
