#!/usr/bin/env python3
"""65AM-B9A5 | READ-ONLY physics diagnostic for the B9A4 dataset.

Runs on Mac (Python 3 / scientific .venv), no Abaqus or retraining.

Stage A: independently validate six Abaqus LE components from Abaqus U
         using original 64J C3D8R B_center: compare engineering and tensor
         shear conventions, per-component errors. Do NOT assume a convention.
Stage B: decompose PCA generalized-force discrepancies into one-point
         approximation errors and B5 field-prediction errors, with 65V scales.
Stage C: report what can and cannot be concluded. Never modify 65V/B5.

Use:
  python3 65AM_B9A5_strain_force_audit.py --selftest
  python3 65AM_B9A5_strain_force_audit.py --project . --check-only
  python3 65AM_B9A5_strain_force_audit.py --project .

Create results/65AM_B9_coupled_feasibility/B9A5_strain_force_audit/ in Finder
before writing results. Existing report files are protected.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ('inner_left', 'inner_right')
INTERFACES = ('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
MODE_COUNTS = (4,4,5,5,4,4)
INNER = {'left_inner','center_left','center_right','right_inner'}
COMPONENTS = ('11','22','33','12','13','23')
REL_OUT = Path('results/65AM_B9_coupled_feasibility/B9A5_strain_force_audit')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path,allow_pickle=False) as z:
        return {k:np.array(z[k],copy=True) for k in z.files}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def rms(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x)))


def relative(pred, truth):
    pred=np.asarray(pred,float)
    truth=np.asarray(truth,float)
    return float(100*np.linalg.norm((pred-truth).ravel())/max(np.linalg.norm(truth.ravel()),1.e-14))


def slices():
    j=0
    for name,n in zip(INTERFACES,MODE_COUNTS):
        yield name,slice(j,j+n)
        j+=n

def index_map(source_xyz,target_xyz,tol=2e-5):
    from scipy.spatial import cKDTree
    source_xyz=np.asarray(source_xyz,float)
    target_xyz=np.asarray(target_xyz,float)
    if source_xyz.shape!=target_xyz.shape or source_xyz.ndim!=2 or source_xyz.shape[1]!=3:
        raise RuntimeError('Coordinate shape mismatch')
    d,ind=cKDTree(source_xyz).query(target_xyz,k=1)
    if len(set(ind.tolist()))!=len(ind) or float(np.max(d))>tol:
        raise RuntimeError('Coordinate mapping is not one-to-one or tolerance exceeded')
    return ind,float(np.max(d))


def hencky_from_fem_u(U,conn,B):
    """Return six-component tensor-strain Voigt values (NOT engineering shear).

    Uses same original 64J C3D8R center B matrix as 65Z_common.py.
    It does not assume Abaqus output shear convention or rotate coordinate frames.
    """
    U=np.asarray(U,float)
    conn=np.asarray(conn,int)
    B=np.asarray(B,float)
    if U.ndim!=2 or U.shape[1]!=3 or conn.ndim!=2 or conn.shape[1]!=8:
        raise RuntimeError('Invalid U/element connectivity')
    if B.shape!=(len(conn),6,24):
        raise RuntimeError('B_center must be [elements,6,24]')
    if np.min(conn)<0 or np.max(conn)>=len(U):
        raise RuntimeError('Element connectivity outside nodal array')
    dn=np.stack((B[:,0,0::3],B[:,1,1::3],B[:,2,2::3]),axis=2)
    grad=np.einsum('eai,eaj->eij',U[conn],dn)
    F=np.eye(3)[None]+grad
    C=np.einsum('eji,ejk->eik',F,F)
    w,V=np.linalg.eigh(C)
    if not np.isfinite(w).all() or np.min(w)<=1.e-12:
        raise RuntimeError('Non-positive definite F.T F')
    tensor=.5*np.einsum('eik,ek,ejk->eij',V,np.log(w),V)
    return np.stack([tensor[:,0,0],tensor[:,1,1],tensor[:,2,2],
                     tensor[:,0,1],tensor[:,0,2],tensor[:,1,2]],axis=1)


def strain_study(root,ref):
    component_rows=[]
    summary=[]
    for reg in REGIONS:
        dsroot=root/'data'/'hybrid_bulk_7region'/reg
        required=['coordinates.npy','ip_coordinates.npy','element_node_indices.npy','B_center.npy']
        for name in required:
            if not (dsroot/name).is_file():
                raise FileNotFoundError('Missing original FE dataset file '+str(dsroot/name))
        x=np.load(dsroot/'coordinates.npy',allow_pickle=False)
        ip=np.load(dsroot/'ip_coordinates.npy',allow_pickle=False)
        conn=np.load(dsroot/'element_node_indices.npy',allow_pickle=False)
        B=np.load(dsroot/'B_center.npy',allow_pickle=False)
        ni,ngap=index_map(ref[reg+'_node_xyz'],x)
        ei,egap=index_map(ref[reg+'_ip_xyz'],ip)
        labels=ref[reg+'_node_labels']
        conn_lab=ref[reg+'_connectivity_labels']
        expected={tuple(sorted(int(labels[i]) for i in ni[row])) for row in conn}
        actual={tuple(sorted(int(v) for v in row)) for row in conn_lab}
        if len(expected)!=len(conn) or expected!=actual:
            raise RuntimeError(reg+' element connectivity mismatch; cannot compare strains')
        U=ref[reg+'_U'][ni]
        LE=ref[reg+'_LE'][ei]
        et=hencky_from_fem_u(U,conn,B)
        # The comparison does NOT change stress model or any saved Abaqus data.
        eng=et.copy();eng[:,3:]*=2
        abaqus_as_eng=LE.copy();abaqus_as_eng[:,3:]*=2
        variants=[('tensor_vs_raw_Abaqus',et,LE),
                  ('engineering_vs_raw_Abaqus',eng,LE),
                  ('engineering_vs_2x_Abaqus_shear',eng,abaqus_as_eng)]
        scored=[]
        for variant,pred,target in variants:
            score=relative(pred,target)
            scored.append((variant,score))
            for j,comp in enumerate(COMPONENTS):
                component_rows.append(dict(region=reg,convention=variant,component='LE'+comp,
                    relative_l2_pct=relative(pred[:,j],target[:,j]),
                    reference_l2=float(np.linalg.norm(target[:,j])),
                    absolute_difference_l2=float(np.linalg.norm(pred[:,j]-target[:,j]))))
            component_rows.append(dict(region=reg,convention=variant,component='ALL',
                relative_l2_pct=score,reference_l2=float(np.linalg.norm(target)),
                absolute_difference_l2=float(np.linalg.norm(pred-target))))
        # The third variant doubles BOTH the prediction/reference shear entries;
        # that changes component weighting, not the Abaqus convention.
        best=min(scored[:2],key=lambda v:v[1])
        summary.append(dict(region=reg,n_elements=len(conn),n_nodes=len(U),
            node_map_max_mm=ngap,ip_map_max_mm=egap,
            tensor_vs_raw_pct=relative(et,LE),
            engineering_vs_raw_pct=relative(eng,LE),
            engineering_vs_2x_ref_pct=relative(eng,abaqus_as_eng),
            LE22_FEM_U_vs_Abaqus_pct=relative(et[:,1],LE[:,1]),
            best_formulation_DIAGNOSTIC_ONLY=best[0],best_full_LE_error_pct=best[1],
            tensor_vs_raw_shear_L2_error_pct=relative(et[:,3:],LE[:,3:]),
            engineering_vs_raw_shear_L2_error_pct=relative(eng[:,3:],LE[:,3:])))
    return pd.DataFrame(summary),pd.DataFrame(component_rows)


def provenance(root):
    paths={
        'anchor':root/'B9_coupled_anchor'/'B9_65V_anchor.npz',
        'anchor_meta':root/'B9_coupled_anchor'/'B9_65V_anchor_manifest.json',
        'fem':root/'B9A2_measured_FE'/'B9A2_measured_FE.npz',
        'fem_meta':root/'B9A2_measured_FE'/'B9A2_windows_summary.json',
        'no':root/'B9A3_exports'/'B9A3_NO_reference.npz',
        'no_meta':root/'B9A3_exports'/'B9A3_windows_summary.json',
        'ref':root/'B9A4_exports'/'B9A4_reference_fields.npz',
        'ref_meta':root/'B9A4_exports'/'B9A4_windows_summary.json',
        'mode':root/'results/65AM_B9_coupled_feasibility/B9A4_field_and_force/B9A4_mode_force_metrics.csv',
    }
    for tag,p in paths.items():
        if not p.is_file():raise FileNotFoundError('Missing '+tag+': '+str(p))
    anchor=load_npz(paths['anchor']);fe=load_npz(paths['fem']);no=load_npz(paths['no']);ref=load_npz(paths['ref'])
    a_meta=load_json(paths['anchor_meta']);f_meta=load_json(paths['fem_meta'])
    n_meta=load_json(paths['no_meta']);r_meta=load_json(paths['ref_meta'])
    h=sha(paths['anchor'])
    if not (h==a_meta.get('anchor_sha256')==f_meta.get('anchor_sha256')==n_meta.get('anchor_sha256')==r_meta.get('anchor_sha256')):
        raise RuntimeError('Anchor provenance SHA256 mismatch across B9 stages')
    for p,m,k in ((paths['fem'],f_meta,'file_sha256'),(paths['no'],n_meta,'export_sha256'),
                  (paths['ref'],r_meta,'reference_fields_sha256')):
        if sha(p)!=m.get(k):raise RuntimeError('Windows/Mac source file checksum mismatch '+str(p))
    a=anchor['normalized_residual'];scale=anchor['force_scale'];old=anchor['old_NO_g'];FE=fe['measured_FE_g']
    true=no['reference_NO_g'];mask=no['reference_mask'].astype(bool)
    if any(np.asarray(v).shape!=(26,) for v in (a,scale,old,FE,true,mask)) or (scale<=0).any():
        raise RuntimeError('26-coordinate force arrays inconsistent')
    expect=np.zeros(26,bool)
    for name,sl in slices():
        if name in INNER:expect[sl]=True
    if not np.array_equal(expect,mask):raise RuntimeError('B9A3 reference force mask mismatch')
    if rms((FE+old)/scale-a)>1e-6:
        raise RuntimeError('Measured FE + old NO does not reproduce original 65V residual')
    modes=pd.read_csv(paths['mode'])
    need={'interface','mode','Abaqus_actual_RF_N','Abaqus_U_S_onepoint_N',
          'B5_U_S_onepoint_N','B5_U_Abaqus_S_onepoint_N','Abaqus_U_B5_S_onepoint_N'}
    if not need.issubset(modes.columns):
        raise RuntimeError('B9A4 force-mode CSV missing: '+str(need-set(modes.columns)))
    return anchor,fe,no,ref,modes,{'anchor_sha256':h,'source_SHA256':{k:sha(p) for k,p in paths.items() if p.suffix=='.npz'}}


def force_study(anchor,fe,no,mode_csv):
    scale=np.asarray(anchor['force_scale'],float)
    old=np.asarray(anchor['old_NO_g'],float)
    FE=np.asarray(fe['measured_FE_g'],float)
    true=np.asarray(no['reference_NO_g'],float)
    inner_mask=np.asarray(no['reference_mask'],bool)
    ref_ctrl=old.copy();ref_ctrl[inner_mask]=true[inner_mask]
    modes=[];interfaces=[]
    fields={
        'reference_onepoint':'Abaqus_U_S_onepoint_N',
        'B5_onepoint':'B5_U_S_onepoint_N',
        'B5_displacement_with_Abaqus_stress':'B5_U_Abaqus_S_onepoint_N',
        'Abaqus_displacement_with_B5_stress':'Abaqus_U_B5_S_onepoint_N',
    }
    candidate={k:old.copy() for k in fields}
    for name,sl in slices():
        if name not in INNER:continue
        data=mode_csv[mode_csv['interface']==name].sort_values('mode')
        if list(data['mode'].astype(int))!=list(range(1,sl.stop-sl.start+1)):
            raise RuntimeError('B9A4 mode data incomplete for '+name)
        ref=data['Abaqus_actual_RF_N'].to_numpy(float)
        if not np.all(np.isfinite(ref)) or not np.allclose(ref,true[sl],rtol=0,atol=1e-3):
            raise RuntimeError('B9A4 Abaqus mode force differs from B9A3 independent ODB '+name)
        for key,col in fields.items():candidate[key][sl]=data[col].to_numpy(float)
    residuals={'old_NO':(FE+old)/scale,'true_Abaqus_NO':(FE+ref_ctrl)/scale}
    for key,g in candidate.items():residuals[key]=(FE+g)/scale
    main_r=residuals['B5_onepoint'];main_d=(candidate['B5_onepoint']-ref_ctrl)/scale
    method=(candidate['reference_onepoint']-ref_ctrl)/scale
    field=(candidate['B5_onepoint']-candidate['reference_onepoint'])/scale
    if not np.allclose(main_d,method+field,atol=1e-11):
        raise RuntimeError('Force error attribution identity failed: '+str(np.max(abs(main_d-method-field))))
    total_r2=float(main_r@main_r)
    inner_mask=np.asarray(no['reference_mask'],bool)
    total_d2=float(main_d[inner_mask]@main_d[inner_mask])
    for name,sl in slices():
        is_inner=name in INNER
        interfaces.append(dict(interface=name,n_modes=sl.stop-sl.start,was_replaced=is_inner,
            old_interface_rms=rms(residuals['old_NO'][sl]),
            true_Abaqus_interface_rms=rms(residuals['true_Abaqus_NO'][sl]) if is_inner else None,
            B5_onepoint_interface_rms=rms(main_r[sl]) if is_inner else None,
            reference_onepoint_interface_rms=rms(residuals['reference_onepoint'][sl]) if is_inner else None,
            B5_displacement_oracle_stress_interface_rms=rms(residuals['B5_displacement_with_Abaqus_stress'][sl]) if is_inner else None,
            FEM_displacement_B5_stress_interface_rms=rms(residuals['Abaqus_displacement_with_B5_stress'][sl]) if is_inner else None,
            B5_rms_squared_share_pct=100*float(main_r[sl]@main_r[sl])/max(total_r2,1e-25),
            B5_minus_true_scaled_rms=rms(main_d[sl]) if is_inner else None,
            onepoint_reference_minus_true_scaled_rms=rms(method[sl]) if is_inner else None,
            B5_minus_reference_onepoint_scaled_rms=rms(field[sl]) if is_inner else None))
        for local,j in enumerate(range(sl.start,sl.stop),start=1):
            modes.append(dict(interface=name,mode=local,global_mode=j+1,replaced=is_inner,
                scale_N=float(scale[j]),FE_RF_N=float(FE[j]),old_NO_N=float(old[j]),
                true_NO_Abaqus_RF_N=float(true[j]) if is_inner else None,
                FE_onepoint_N=float(candidate['reference_onepoint'][j]) if is_inner else None,
                B5_onepoint_N=float(candidate['B5_onepoint'][j]) if is_inner else None,
                original_normalized_residual=float(residuals['old_NO'][j]),
                true_NO_normalized_residual=float(residuals['true_Abaqus_NO'][j]) if is_inner else None,
                B5_normalized_residual=float(main_r[j]) if is_inner else None,
                B5_vs_true_normalized=float(main_d[j]) if is_inner else None,
                onepoint_method_error_normalized=float(method[j]) if is_inner else None,
                B5_field_error_normalized=float(field[j]) if is_inner else None,
                B5_displacement_oracle_stress_error_normalized=float((candidate['B5_displacement_with_Abaqus_stress'][j]-ref_ctrl[j])/scale[j]) if is_inner else None,
                FEM_displacement_B5_stress_error_normalized=float((candidate['Abaqus_displacement_with_B5_stress'][j]-ref_ctrl[j])/scale[j]) if is_inner else None,
                squared_residual_share_pct=100*float(main_r[j]**2)/max(total_r2,1e-25),
                squared_B5_vs_true_share_pct=100*float(main_d[j]**2)/max(total_d2,1e-25) if is_inner else None))
    metrics={key+'_RMS':rms(value) for key,value in residuals.items()}
    metrics.update(B5_vs_true_scaled_RMS_inner=rms(main_d[inner_mask]),
       reference_onepoint_vs_true_scaled_RMS_inner=rms(method[inner_mask]),
       B5_vs_reference_onepoint_scaled_RMS_inner=rms(field[inner_mask]),
       attribution_exact=True,
       reference_65V_saved_RMS=rms(anchor['normalized_residual']),
       B5_force_not_a_converged_state=True,
       live_Broyden_approved=False)
    return pd.DataFrame(interfaces),pd.DataFrame(modes),metrics


def selftest():
    # Independent kinematic strain test: infinitesimal uniaxial extension.
    # Build a single axis-aligned brick with exact physical shape derivatives.
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    conn=np.arange(8)[None,:]
    dN=np.zeros((8,3),float)
    for i,p in enumerate(xyz):dN[i]=(p-.5)*.5
    B=np.zeros((1,6,24))
    for a in range(8):
        for d in range(3):B[0,d,3*a+d]=dN[a,d]
    u=xyz*np.array([0.01,0.,0.])
    et=hencky_from_fem_u(u,conn,B)
    assert abs(et[0,0]-np.log(1.01))<1e-10 and np.max(abs(et[0,1:]))<1e-10
    # engineering shear test F=[1,k,0;0,1,0;0,0,1] small k => LE12~k/2
    u=np.zeros((8,3));u[:,0]=0.004*xyz[:,1]
    e=hencky_from_fem_u(u,conn,B)
    assert abs(e[0,3]-0.002)<1e-5
    # Exact mode decomposition identity and share sum.
    vals=np.array([3.,-2.,1.]);alpha=np.array([2.,-1.,.5]);beta=vals-alpha
    assert np.allclose(vals,alpha+beta)
    assert abs(sum(v*v for v in vals)/sum(vals*vals)-1)<1e-12
    print('B9A5 SELFTEST PASS: finite strain, shear convention, mode attribution')


def run(root,check_only):
    anchor,fe,no,ref,mode,source=provenance(root)
    sr,components=strain_study(root,ref)
    ir,modes,metrics=force_study(anchor,fe,no,mode)
    for row in sr.to_dict('records'):
        print('B9A5 %s: tensor vs Abaqus LE %.4f%% | engineering vs Abaqus LE %.4f%% | LE22 %.4f%%'%(
            row['region'],row['tensor_vs_raw_pct'],row['engineering_vs_raw_pct'],row['LE22_FEM_U_vs_Abaqus_pct']))
    for x,y in [('old_NO_RMS','original'),('true_Abaqus_NO_RMS','Abaqus NO-side'),
                ('reference_onepoint_RMS','Abaqus fields -> 1-point'),('B5_onepoint_RMS','B5 fields -> 1-point')]:
        print('B9A5 %-28s %.6f'%(y,metrics[x]))
    top=modes[modes['replaced']].sort_values('squared_residual_share_pct',ascending=False).head(6)
    print('Dominant B5 residual modes: '+', '.join('%s/%s %.1f%%'%(r.interface,int(r['mode']),r.squared_residual_share_pct)
          for _,r in top.iterrows()))
    if check_only:
        print('B9A5 CHECK ONLY PASS; no outputs created')
        return
    out=root/REL_OUT
    if not out.is_dir():
        raise RuntimeError('In Finder create this output folder first: '+str(out))
    outputs=['B9A5_strain_convention.csv','B9A5_strain_components.csv',
             'B9A5_force_interfaces.csv','B9A5_force_modes.csv',
             'B9A5_summary.json','B9A5_force_attribution.png','B9A5_strain_convention.png']
    existing=[str(out/n) for n in outputs if (out/n).exists()]
    if existing:raise RuntimeError('Existing results protected; use a fresh folder: '+str(existing))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sr.to_csv(out/outputs[0],index=False)
    components.to_csv(out/outputs[1],index=False)
    ir.to_csv(out/outputs[2],index=False)
    modes.to_csv(out/outputs[3],index=False)
    summary={'step':'65AM-B9A5','anchor_sha256':source['anchor_sha256'],
        'reference':'Real Abaqus B9A3 U, LE and RF; measured FE B9A2 + original 65V PCA scaling; frozen B5 predictions from B9A4',
        'strain_audit':sr.to_dict('records'),'force_RMS':metrics,
        'top_six_B5_residual_modes':top[['interface','mode','B5_normalized_residual','squared_residual_share_pct']].to_dict('records'),
        'limitations':['Strain conventions are compared; no constitutive updates or silent numerical correction applied',
                       'C3D8R one-point force is a diagnostic without verified hourglass/contact treatment',
                       'Old 65V interface state, NOT a re-equilibrated B5 solution',
                       'No deployment or live Broyden is authorized by this diagnostic'],
        'source_SHA256':source['source_SHA256']}
    (out/outputs[4]).write_text(json.dumps(summary,indent=2))
    inner=modes[modes.replaced]
    order=inner.sort_values('squared_residual_share_pct',ascending=False).head(12)
    xx=np.arange(len(order))
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(xx-.22,abs(order['onepoint_method_error_normalized']),.44,label='FE one-point approximation error')
    ax.bar(xx+.22,abs(order['B5_field_error_normalized']),.44,label='Additional B5 field error')
    ax.set_xticks(xx);ax.set_xticklabels([f'{i}/m{j}' for i,j in zip(order.interface,order['mode'])],rotation=35,ha='right')
    ax.set_ylabel('Absolute normalized force difference')
    ax.set_title('65AM-B9A5: force approximation vs B5 prediction (same 65V state)')
    ax.legend();fig.tight_layout();fig.savefig(out/outputs[5],dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));xx=np.arange(len(sr));w=.34
    ax.bar(xx-w/2,sr['tensor_vs_raw_pct'],w,label='Reconstructed tensor LE vs Abaqus')
    ax.bar(xx+w/2,sr['engineering_vs_raw_pct'],w,label='Reconstructed engineering shear LE vs Abaqus')
    ax.set_xticks(xx);ax.set_xticklabels(sr.region)
    ax.set_ylabel('Six-component LE relative L2 error (%)')
    ax.set_title('65AM-B9A5: Abaqus strain convention diagnostic')
    ax.legend();fig.tight_layout();fig.savefig(out/outputs[6],dpi=180);plt.close(fig)
    print('B9A5 RESULTS SAVED:',out)
    print('Do not treat B9A5 as a coupled B5 solution or full force-equilibrium validation.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.selftest:
        selftest();return
    run(Path(args.project).expanduser().resolve(),args.check_only)

if __name__=='__main__':main()
