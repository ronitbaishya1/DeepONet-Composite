#!/usr/bin/env python3
"""B9A4 Mac: actual B9A3 Abaqus FIELD truth and force-integration consistency.

python3 65AM_B9A4_mac_diagnose.py --selftest
python3 65AM_B9A4_mac_diagnose.py --project . --check-only
python3 65AM_B9A4_mac_diagnose.py --project .

Uses existing B9 saved 65V interface, frozen B5, B6/B7/B8 code.
Tests full nodal residuals using *one-point* stress integration, a diagnostic
that does NOT reproduce Abaqus's C3D8R hourglass stabilization automatically.
Never modifies previous results or solver.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_OUT=Path('results/65AM_B9_coupled_feasibility/B9A4_field_and_force')
REGIONS=('inner_left','inner_right')
SIDES={'inner_left':{'left':'left_inner','right':'center_left'},
       'inner_right':{'left':'center_right','right':'right_inner'}}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def npz(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    with np.load(path,allow_pickle=False) as z:
        return {k:np.array(z[k],copy=True) for k in z.files}

def read_json(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    return json.loads(path.read_text())

def rel(a,b):
    x=np.asarray(a,float).ravel();y=np.asarray(b,float).ravel()
    return float(100*np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))

def map_unique(source,target,tol=2e-5):
    """Array of indices into source for every target coordinate."""
    from scipy.spatial import cKDTree
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3:
        raise RuntimeError('Geometry count/dim mismatch: '+str((source.shape,target.shape)))
    dist,ix=cKDTree(source).query(target,k=1)
    if len(set(ix.tolist()))!=len(target) or np.max(dist)>tol:
        raise RuntimeError('Abaqus B9A3 geometry does not align with B5 FE dataset, max gap %.4g'%np.max(dist))
    return ix,float(np.max(dist))

def b9a3_inputs(root):
    folder=root/'B9A4_exports'
    ref_file=folder/'B9A4_reference_fields.npz'
    wm=read_json(folder/'B9A4_windows_summary.json')
    bm=read_json(root/'B9_coupled_anchor'/'B9_65V_anchor_manifest.json')
    ref=read_json(root/'B9A3_exports'/'B9A3_windows_summary.json')
    af=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    if sha(af)!=bm.get('anchor_sha256') or bm['anchor_sha256']!=wm.get('anchor_sha256'):
        raise RuntimeError('B9A4 anchor is not the saved 65V anchor')
    rfile=root/'B9A3_exports'/'B9A3_NO_reference.npz'
    if sha(rfile)!=ref.get('export_sha256') or wm.get('b9a3_reference_sha256')!=sha(rfile):
        raise RuntimeError('B9A3 independent Abaqus reference changed')
    if wm.get('reference_fields_sha256')!=sha(ref_file):
        raise RuntimeError('B9A4 Windows export checksum mismatch')
    arr=npz(ref_file);original=npz(rfile)
    for reg in REGIONS:
        for key in ('node_xyz','node_labels','U','RF','element_labels','connectivity_labels','ip_xyz','S','LE'):
            if reg+'_'+key not in arr:raise RuntimeError('Missing B9A4 '+reg+' '+key)
        if arr[reg+'_U'].shape!=(1008,3) or arr[reg+'_S'].shape!=(728,6):
            raise RuntimeError('Unexpected B9A4 mesh size')
    if str(arr['anchor_sha256'].item())!=bm['anchor_sha256']:
        raise RuntimeError('B9A4 data anchor mismatch')
    return arr,original,wm

def align_export(ds,arrays,region):
    from scipy.spatial import cKDTree
    x=np.asarray(ds['coordinates'],float)
    nodes=np.asarray(arrays[region+'_node_xyz'],float)
    idx,ngap=map_unique(nodes,x)
    ip=np.asarray(ds['ip_coordinates'],float)
    idxe,egap=map_unique(arrays[region+'_ip_xyz'],ip)
    nodal_u=arrays[region+'_U'][idx].astype(float)
    nodal_rf=arrays[region+'_RF'][idx].astype(float)
    s=arrays[region+'_S'][idxe].astype(float)
    le=arrays[region+'_LE'][idxe].astype(float)
    conn=np.asarray(ds['element_node_indices'],int)
    labels=np.asarray(arrays[region+'_node_labels'],int)
    elemconn=np.asarray(arrays[region+'_connectivity_labels'],int)
    expected=set(tuple(sorted(int(labels[k]) for k in idx[c])) for c in conn)
    actual=set(tuple(sorted(map(int,row))) for row in elemconn)
    if len(expected)!=len(conn) or expected!=actual:
        raise RuntimeError('B9A3 and B5 element connectivity do not match; do not compare gradients')
    if not all(np.isfinite(a).all() for a in (nodal_u,nodal_rf,s,le)):
        raise RuntimeError('Nonfinite source fields')
    return nodal_u,nodal_rf,le,s,ngap,egap

def nodal_g(f,x,info):
    from scipy.spatial import cKDTree
    facecoords=np.asarray(info['coordinates'],float)
    dist,idx=cKDTree(np.asarray(x,float)).query(facecoords)
    if len(set(idx.tolist()))!=len(idx) or np.max(dist)>2e-5:
        raise RuntimeError('PCA face mesh does not match target C3D8R geometry')
    forces=np.asarray(f,dtype=float)[idx]
    basis=np.asarray(info['basis'],float)
    if basis.shape[0]!=3*len(idx):raise RuntimeError('Bad face PCA basis')
    return basis.T@forces.ravel(),float(np.max(dist)),idx

def selftest():
    import importlib
    b8=importlib.import_module('65AM_B8_force_traction')
    b8.selftest()
    x=np.array([[0.,0,0],[1.,0,0],[2.,0,0]],float)
    perm=np.array([2,0,1]);idx,d=map_unique(x[perm],x)
    assert np.array_equal(idx,[1,2,0]) and d<1e-12
    idx,d=map_unique(x,x);assert np.array_equal(idx,[0,1,2])
    f=np.array([[1.,2,3.],[4,5,6],[7,8,9]])
    Q=np.eye(9)[:,:2]
    g,_,ids=nodal_g(f,x,{'coordinates':x,'basis':Q})
    assert np.allclose(g,[1.,2.])
    print('B9A4 MAC SELFTEST PASS: C3D8R + geometry permutation + PCA virtual work')

def run(root,device,check_only):
    import torch
    b9=importlib.import_module('65AM_B9_mac_saved_state')
    b8=importlib.import_module('65AM_B8_force_traction')
    b7=importlib.import_module('65AM_B7_mac')
    b6=importlib.import_module('65AM_B6_mac')
    anchor,ameta=b9.load_anchor(root)
    arrays,reference,windows=b9a3_inputs(root)
    _,models=b6.check(root,device)
    _,_,_,aud,_=b7.load_b6(root,device)
    p=root/'results'/'65AM_B7_stress_consistency'/'A_constitutive_audit'/'B7_summary.json'
    if not read_json(p).get('law_proxy_acceptable_for_both'):
        raise RuntimeError('B7 constitutive approximation not validated')
    stiff=b7.constitutive_stiffness(np.asarray(anchor['material'],float).reshape(1,3),
                                   aud['fixed_orthotropic_constants'])
    metrics=[];forces=[];mode_rows=[];mat=np.asarray(anchor['material'],float)
    for reg,(prepared,net,ck) in models.items():
        if reg not in SIDES:
            raise RuntimeError('Unexpected B9A4 region: '+str(reg))
        # The original B9 helper requires an explicit region tag to select
        # the correct interface pairs and PCA coefficients. Keep the original
        # prepared dataset unchanged and pass a tagged, shallow copy instead.
        ds=dict(prepared['ds'])
        ds['_b9_region']=reg
        context=b6.b.CO.interface_context(root,ds,device)
        branch,lift=b9.build_branch_lift(root,ds,anchor,context,device)
        x,conn,dN,vol,aff=b8.geometry(ds)
        Uref,RF,LEref,Sref,ge,gi=align_export(ds,arrays,reg)
        with torch.no_grad():Uhat=b6.predict_b5(prepared,net,branch,lift,device)[0]
        fixed=b6.b.boundary_indices(context)
        if np.max(abs(Uhat[fixed]-lift[0,fixed]))>1e-5:
            raise RuntimeError('B5 interface compatibility failed '+reg)
        # Abaqus LE field uses tensor shear (LE12 etc.), while 65Z returns 2*shear.
        LEreference_eng=LEref.copy();LEreference_eng[:,3:]*=2.
        LEu=b6.b.CO.log_material_strain_eng(Uref,conn,ds['B_center']).astype(float)
        LEb=b6.b.CO.log_material_strain_eng(Uhat,conn,ds['B_center']).astype(float)
        Sb=b7.stress_from_LE(LEb[None,:,:],stiff)[0]
        # 65Z material-frame log strain can differ slightly from Abaqus frame;
        # use Abaqus S directly as primary benchmark, never quietly 'correct' it.
        kin22=rel(LEu[:,1],LEref[:,1]);kin_all=rel(LEu,LEreference_eng)
        if kin22>2.0:raise RuntimeError('FEM U to Abaqus LE22 mismatch %.2f%%: stop'%kin22)
        Fref,Jref=b8.deformations(Uref,conn,dN)
        Fb,Jb=b8.deformations(Uhat,conn,dN)
        Pref=b8.first_piola(Sref,Fref,Jref)
        Pb=b8.first_piola(Sb,Fb,Jb)
        Pin_ref_stress_b_u=b8.first_piola(Sref,Fb,Jb)
        fref=b8.internal_forces(Pref,conn,dN,vol,len(x))
        fb=b8.internal_forces(Pb,conn,dN,vol,len(x))
        f_b_u_ref_s=b8.internal_forces(Pin_ref_stress_b_u,conn,dN,vol,len(x))
        f_ref_u_b_s=b8.internal_forces(b8.first_piola(Sb,Fref,Jref),conn,dN,vol,len(x))
        row=dict(region=reg,n_nodes=len(x),n_elements=len(conn),
                 reference_coordinate_max_mm=ge,IP_coordinate_max_mm=gi,
                 U_rel_pct=rel(Uhat,Uref),U2_rel_pct=rel(Uhat[:,1],Uref[:,1]),
                 grad_rel_pct=rel(b6.b.AA.gradient_numpy(Uhat,ds),b6.b.AA.gradient_numpy(Uref,ds)),
                 FEM_U_to_Abaqus_LE22_pct=kin22,FEM_U_to_Abaqus_LE_all_eng_pct=kin_all,
                 B5_LE22_rel_pct=rel(LEb[:,1],LEref[:,1]),
                 B5_LE_all_eng_pct=rel(LEb,LEreference_eng),
                 B5_S22_rel_pct=rel(Sb[:,1],Sref[:,1]),B5_S_vector_rel_pct=rel(Sb,Sref),
                 min_J_B5=float(np.min(Jb)))
        metrics.append(row)
        for side,name in SIDES[reg].items():
            info=context[side]
            g_abaqus,dref,ids=nodal_g(RF,x,info)
            g_ref,d1,_=nodal_g(fref,x,info)
            g_pred,d2,_=nodal_g(fb,x,info)
            g_oraclestress,d3,_=nodal_g(f_b_u_ref_s,x,info)
            g_oracleU,d4,_=nodal_g(f_ref_u_b_s,x,info)
            stored=np.asarray(reference['reference_NO_g'],float)
            count={'left_inner':4,'center_left':5,'center_right':5,'right_inner':4}[name]
            seq={'left_inner':slice(4,8),'center_left':slice(8,13),
                 'center_right':slice(13,18),'right_inner':slice(18,22)}[name]
            if len(g_abaqus)!=count or rel(g_abaqus,stored[seq])>0.001:
                raise RuntimeError('Actual ODB RF does not match B9A3 exported true RF for '+name)
            # Differentiating FE basis does not by itself include hourglass stabilization.
            ab=np.asarray(RF,float)[ids]
            refnod=fref[ids]
            prednod=fb[ids]
            r=dict(region=reg,interface=name,n_modes=count,n_nodes=len(ids),
                   actual_RF_resultant_N=float(np.linalg.norm(ab.sum(axis=0))),
                   FEM_onepoint_nodal_vs_RF_pct=rel(refnod,ab),
                   B5_onepoint_nodal_vs_RF_pct=rel(prednod,ab),
                   B5_onepoint_vs_FE_onepoint_pct=rel(prednod,refnod),
                   FEM_onepoint_mode_vs_RF_pct=rel(g_ref,g_abaqus),
                   B5_onepoint_mode_vs_RF_pct=rel(g_pred,g_abaqus),
                   FEM_onepoint_negated_force_vs_RF_pct=rel(-g_ref,g_abaqus),
                   B5_onepoint_negated_force_vs_RF_pct=rel(-g_pred,g_abaqus),
                   B5_U_reference_S_mode_vs_RF_pct=rel(g_oraclestress,g_abaqus),
                   FEM_U_B5_S_mode_vs_RF_pct=rel(g_oracleU,g_abaqus),
                   force_gap_reference_onepoint_N=float(np.linalg.norm(g_ref-g_abaqus)),
                   force_gap_B5_onepoint_N=float(np.linalg.norm(g_pred-g_abaqus)),
                   force_gap_B5_vs_FE_onepoint_N=float(np.linalg.norm(g_pred-g_ref)))
            forces.append(r)
            for j in range(count):
                mode_rows.append(dict(region=reg,interface=name,mode=j+1,
                      Abaqus_actual_RF_N=float(g_abaqus[j]),
                      Abaqus_U_S_onepoint_N=float(g_ref[j]),
                      B5_U_S_onepoint_N=float(g_pred[j]),
                      B5_U_Abaqus_S_onepoint_N=float(g_oraclestress[j]),
                      Abaqus_U_B5_S_onepoint_N=float(g_oracleU[j])))
        print('B9A4',reg,'U %.3f%% LE22 %.3f%% S22 %.3f%%'%(row['U_rel_pct'],row['B5_LE22_rel_pct'],row['B5_S22_rel_pct']))
    df=pd.DataFrame(metrics);fd=pd.DataFrame(forces);md=pd.DataFrame(mode_rows)
    ctrl=fd.FEM_onepoint_mode_vs_RF_pct.to_numpy()
    decision=('FE_ONEPOINT_INSUFFICIENT' if np.any(ctrl>2.) else 'FE_ONEPOINT_REFERENCE_CHECK_PASSED')
    summary=dict(step='65AM-B9A4',reference='B9A3 two independently solved NO-side Abaqus ODBs at saved 65V boundary state',
       anchor_sha256=windows['anchor_sha256'],FEM_fields_reused=True,
       model='frozen B5 centered_grad',independent_ORACLE_FEM_interface_BC=True,
       field_errors=metrics,onepoint_force_audit=forces,reference_onepoint_gate_pct=2.0,
       reference_onepoint_status=decision,
       note=('Even a passing one-point audit is NOT proof of full Abaqus C3D8R force reproduction; hourglass/contact omitted. '
             'Do not use this approximate force map in live Broyden without actual Abaqus nodal RF agreement.'),
       original_65V_unmodified=True,B5_unmodified=True,live_Broyden_approved=False)
    print('B9A4 force status:',decision,'| No live Broyden approved.')
    if check_only:
        print('B9A4 CHECK ONLY PASS; no result files written')
        return
    out=root/ROOT_OUT
    if out.exists() and any(out.iterdir()):raise RuntimeError('Existing B9A4 results protected: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'B9A4_field_metrics.csv',index=False)
    fd.to_csv(out/'B9A4_interface_force_metrics.csv',index=False)
    md.to_csv(out/'B9A4_mode_force_metrics.csv',index=False)
    (out/'B9A4_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.8));xx=np.arange(len(fd));width=.24
    ax.bar(xx-width,fd['FEM_onepoint_mode_vs_RF_pct'],width,label='Abaqus U,S -> one-point forces')
    ax.bar(xx,fd['B5_onepoint_mode_vs_RF_pct'],width,label='B5 U,S -> one-point forces')
    ax.bar(xx+width,fd['B5_onepoint_vs_FE_onepoint_pct'],width,label='B5 vs FE one-point')
    ax.set_xticks(xx);ax.set_xticklabels(fd.interface,rotation=15,ha='right')
    ax.set_ylabel('Generalized interface-force relative L2 error (%)')
    ax.set_title('B9A4: separate field-model error from C3D8R force-reconstruction error')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'B9A4_force_comparison.png',dpi=180);plt.close(fig)
    print('SAVED:',out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    p.add_argument('--device',default='auto',choices=('auto','cpu','mps'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--selftest',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest();return
    b6=importlib.import_module('65AM_B6_mac')
    d=b6.b.CO.device_from_option(a.device)
    run(Path(a.project).expanduser().resolve(),d,a.check_only)

if __name__=='__main__':main()
