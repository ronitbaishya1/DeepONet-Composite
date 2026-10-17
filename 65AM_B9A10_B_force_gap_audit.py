#!/usr/bin/env python3
"""65AM B9A10-B: localization and consistency audit of the C3D8R nodal force gap.

MAC ONLY, no Abaqus runs and no model training.
From DeepONet_Composite:
  python3 65AM_B9A10_B_force_gap_audit.py --selftest
  python3 65AM_B9A10_B_force_gap_audit.py --project . --check-only
  python3 65AM_B9A10_B_force_gap_audit.py --project .

Requires untouched B9A10-A Python module and existing B9A4/B9A6 exports.
Signs, PCA modes, and original 65V scaling are copied from B9A10-A.
This script DOES NOT call the force gap hourglass forces: difference also includes
constitutive, stress interpolation, nonlinear discretization, and other effects.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ('inner_left', 'inner_right')
OUTPUT = Path('results/65AM_B9_coupled_feasibility/B9A10B_force_gap_localization')
BASE_MODULE = '65AM_B9A10_weak_form_audit.py'
OUTFILES = (
    'B9A10B_regions.csv', 'B9A10B_axial_zones.csv', 'B9A10B_interface_nodes.csv',
    'B9A10B_modes.csv', 'B9A10B_components.csv', 'B9A10B_summary.json',
    'B9A10B_axial_profiles.png', 'B9A10B_mode_attribution.png',
    'B9A10B_interface_spatial.png',
)

def base_module(root):
    path = root / BASE_MODULE
    if not path.is_file():
        raise FileNotFoundError('Copy '+BASE_MODULE+' into project root; required validated B9A10-A routines')
    spec = importlib.util.spec_from_file_location('B9A10_prior_verified', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def ratio_percent(x, ref):
    return 100.0 * np.linalg.norm(np.asarray(x, float).ravel()) / max(np.linalg.norm(np.asarray(ref, float).ravel()), 1e-12)

def build_data(base, anchor, ref, tr, replay, reg):
    labels=np.asarray(ref[reg+'_node_labels'],np.int64)
    xyz=np.asarray(ref[reg+'_node_xyz'],float)
    U=np.asarray(ref[reg+'_U'],float)
    S=np.asarray(ref[reg+'_S'],float)
    cl=np.asarray(ref[reg+'_connectivity_labels'],np.int64)
    tl=np.asarray(tr[reg+'_node_labels'],np.int64)
    FEA=base.reorder_by_labels(replay[reg+'_FEM_RF_nodal'],tl,labels).astype(float)
    B5A=base.reorder_by_labels(replay[reg+'_B5_RF_nodal'],tl,labels).astype(float)
    UB=base.reorder_by_labels(tr[reg+'_U_B5'],tl,labels).astype(float)
    if np.max(np.abs(base.reorder_by_labels(tr[reg+'_U_FEM'],tl,labels)-U))>2e-5:
        raise RuntimeError('Mismatch between Abaqus FEM displacement and replay, stop')
    if not np.isfinite(xyz).all() or not np.isfinite(U).all():
        raise RuntimeError('Nonfinite source geometry/displacement')
    nodeid={int(i):j for j,i in enumerate(labels)}
    if len(nodeid)!=len(labels):raise RuntimeError('duplicate node label')
    try:
        con=np.asarray([[nodeid[int(k)] for k in e] for e in cl],dtype=int)
    except KeyError as e:
        raise RuntimeError('Wrong element-label map') from e
    weak, vols, jdef=base.assemble_c3d8r_weak(xyz,U,con,S)
    exact_gap=FEA-weak      # not called pure hourglass contribution
    field_gap=B5A-FEA
    bc=np.zeros(len(xyz),bool)
    face={}
    for iface in base.INTERFACES[reg]:
        ids=base.face_indices(xyz,anchor['pca_coordinates_'+iface])
        bc[ids]=True
        basis=np.asarray(anchor['pca_basis_'+iface],float)
        sl=base.SLICES[iface]
        face[iface]=(ids,basis,np.asarray(anchor['force_scale'][sl],float))
    if int(bc.sum())!=224:raise RuntimeError('Constrained nodes did not match expected 224')
    if base.relative_pct(FEA[bc],ref[reg+'_RF'][bc])>.1:raise RuntimeError('B9A6 FEM replay control failed')
    if ratio_percent(FEA[~bc],FEA[bc])>1.:raise RuntimeError('FEM free-node control failed')
    return dict(xyz=xyz,U=U,UB=UB,weak=weak,FEA=FEA,B5A=B5A,
                exact_gap=exact_gap,field_gap=field_gap,bc=bc,face=face,
                total_volume_mm3=float(vols.sum()),min_J=float(jdef.min()),max_J=float(jdef.max()))

def analyze_region(base, anchor, ref, tr, replay, reg):
    d=build_data(base,anchor,ref,tr,replay,reg)
    xyz=d['xyz']; w=d['weak']; fe=d['FEA']; b5=d['B5A']; gp=d['exact_gap']; fg=d['field_gap']; bc=d['bc']
    boundary_ref=fe[bc]
    row={'region':reg, 'nodes':len(xyz),'bc_nodes':int(bc.sum()), 'free_nodes':int((~bc).sum()),
         'reference_volume_mm3':d['total_volume_mm3'],
         'onepoint_vs_abaqus_all_pct':base.relative_pct(w,fe),
         'onepoint_vs_abaqus_interface_pct':base.relative_pct(w[bc],fe[bc]),
         'onepoint_free_node_force_pct_of_boundary_RF':ratio_percent(w[~bc],boundary_ref),
         'abaqus_free_node_force_pct_of_boundary_RF':ratio_percent(fe[~bc],boundary_ref),
         'B5_replay_free_node_force_pct_of_boundary_RF':ratio_percent(b5[~bc],boundary_ref),
         'unmodeled_gap_all_norm_N':base.safe_norm(gp),
         'unmodeled_gap_free_norm_N':base.safe_norm(gp[~bc]),
         'unmodeled_gap_interface_norm_N':base.safe_norm(gp[bc]),
         'B5_field_gap_all_norm_N':base.safe_norm(fg),
         'net_unmodeled_gap_N':np.sum(gp,axis=0).tolist(),
         'total_unmodeled_gap_work_Nmm':float(np.sum(gp*d['U'])),
         'min_J':d['min_J'],'max_J':d['max_J']}
    # Note: f_gap·U is a virtual work diagnostic, NOT Abaqus ALLAE/ALLSE.
    axial_rows=[]; comp_rows=[]; node_rows=[]; mode_rows=[]
    xs=np.unique(np.round(xyz[:,0],6))
    if len(xs)!=9:  # 1008/112=9 longitudinal grid stations
        raise RuntimeError('Unexpected x-station count %d (expected 9)'%len(xs))
    for ix,x in enumerate(xs):
        mask=np.isclose(xyz[:,0],x,atol=1e-5)
        if int(mask.sum())!=112:raise RuntimeError('Expected 112 nodes per axial x plane')
        near= ('interface' if ix in (0,len(xs)-1) else
               'near_interface' if ix in (1,len(xs)-2) else 'interior')
        axial_rows.append({'region':reg,'x_mm':float(x),'station':int(ix+1),'zone':near,
            'nodes':int(mask.sum()), 'weak_to_Abaqus_gap_norm_N':base.safe_norm(gp[mask]),
            'B5_to_FEM_field_gap_norm_N':base.safe_norm(fg[mask]),
            'FEM_RF_norm_N':base.safe_norm(fe[mask]),
            'weak_gap_pct_of_total_boundary_RF':ratio_percent(gp[mask],boundary_ref),
            'B5_gap_pct_of_total_boundary_RF':ratio_percent(fg[mask],boundary_ref)})
    for label,mask in [('all',np.ones(len(xyz),bool)),('free',~bc),('interface',bc)]:
        for idx,component in enumerate(['x','y','z']):
            comp_rows.append({'region':reg,'node_group':label,'component':component,
              'FEM_RF_norm_N':base.safe_norm(fe[mask,idx]),
              'weak_gap_norm_N':base.safe_norm(gp[mask,idx]),
              'B5_gap_norm_N':base.safe_norm(fg[mask,idx])})
    for iface,(ids,Q,scl) in d['face'].items():
        local=gp[ids]; local_B5=fg[ids]; refp=fe[ids]
        pweak=Q.T @ local.ravel(); pb5=Q.T @ local_B5.ravel()
        for j in range(len(scl)):
            q=Q[:,j].reshape(-1,3)
            pernode=np.sum(q*local,axis=1)/scl[j]
            pernode_B5=np.sum(q*local_B5,axis=1)/scl[j]
            # Sum of contributions exactly reconstructs PCA projection.
            if not np.isclose(pernode.sum(),pweak[j]/scl[j],rtol=1e-10,atol=1e-10):
                raise RuntimeError('Mode per-node force contributions do not sum correctly')
            mode_rows.append({'region':reg,'interface':iface,'mode':j+1,
                'original_65V_scale_N':float(scl[j]),
                'Abaqus_RF_projection_N':float((Q[:,j]@refp.ravel())),
                'missing_weak_projection_N':float(pweak[j]),
                'missing_weak_projection_scaled':float(pweak[j]/scl[j]),
                'B5_field_projection_N':float(pb5[j]),
                'B5_field_projection_scaled':float(pb5[j]/scl[j]),
                'max_abs_nodal_weak_mode_contrib_scaled':float(np.max(np.abs(pernode))),
                'top5_node_abs_share_pct':float(100*np.sort(np.abs(pernode))[-5:].sum()/max(np.abs(pernode).sum(),1e-12))})
            for i,node in enumerate(ids):
                node_rows.append({'region':reg,'interface':iface,'mode':j+1,
                    'x_mm':float(xyz[node,0]),'y_mm':float(xyz[node,1]),'z_mm':float(xyz[node,2]),
                    'FE_weak_gap_mode_contribution_scaled':float(pernode[i]),
                    'B5_field_gap_mode_contribution_scaled':float(pernode_B5[i]),
                    'onepoint_minus_Abaqus_force_norm_N':float(np.linalg.norm(local[i])),
                    'B5_minus_FEM_force_norm_N':float(np.linalg.norm(local_B5[i]))})
    return row,axial_rows,comp_rows,node_rows,mode_rows,d

def selftest():
    # PCA nodal contribution, exact decomposition, zero work of constant gap against zero U.
    Q=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])/np.sqrt(2.)
    gap=np.array([3.,-1.,2.,4.]);scales=np.array([3.,2.])
    for i in range(2):
        vals=Q[:,i]*gap/scales[i]
        assert np.allclose(vals.sum(),(Q[:,i]@gap)/scales[i])
    assert abs(ratio_percent(np.array([2.,0]),np.array([4.,0]))-50)<1e-10
    print('B9A10-B SELFTEST PASS | PCA nodal attribution | normalization')

def run(root, check_only=False):
    mod=base_module(root)
    a,ref,tr,replay,sha,meta=mod.load_verified(root)
    regions=[];axial=[]; comps=[];nodes=[];modes=[]; details={}
    for r in REGIONS:
        rr,aa,cc,nn,mm,d=analyze_region(mod,a,ref,tr,replay,r)
        regions.append(rr);axial+=aa;comps+=cc;nodes+=nn;modes+=mm;details[r]=d
        print('%s | FEM gap %.3f%% all | free weak %.3f%% of RF | B5 free %.3f%% of RF'%(
            r,rr['onepoint_vs_abaqus_all_pct'],rr['onepoint_free_node_force_pct_of_boundary_RF'],
            rr['B5_replay_free_node_force_pct_of_boundary_RF']))
    df_r=pd.DataFrame(regions);df_a=pd.DataFrame(axial);df_c=pd.DataFrame(comps)
    df_n=pd.DataFrame(nodes);df_m=pd.DataFrame(modes)
    df_m['gap_abs_scaled']=np.abs(df_m.missing_weak_projection_scaled)
    top=df_m.sort_values('gap_abs_scaled',ascending=False).head(8)
    summary={'step':'65AM-B9A10-B','anchor_sha256':sha,
      'reference_controls_pass':True,'method':'Spatial attribution of one-point force gap and B5 kinematic replay force gap',
      'regions':regions,
      'top_missing_force_modes':top[['interface','mode','missing_weak_projection_scaled','B5_field_projection_scaled']].to_dict('records'),
      'energy_output_not_used_to_fit_or_explain_gap':True,
      'limitations':['One-point versus Abaqus gap includes constitutive/kinematic/discretization/stabilization differences; not identified as hourglass',
         'Force-gap dot displacement is work diagnostic, not Abaqus artificial energy',
         'No C3D8R stabilization formula was inferred from outputs',
         'No physics loss calibrated or deployed; B5/B9A7/65V unchanged'],
      'weak_form_physics_loss_ready':False,'live_Broyden_approved':False}
    if check_only:
        print('B9A10-B CHECK PASS | 2 regions, %d PCA modes, %d per-mode-node rows; NO FILES WRITTEN'%(len(df_m),len(df_n)))
        print('Top mode: %s/%d | %.5f original 65V force scale'%(
            top.iloc[0]['interface'],top.iloc[0]['mode'],top.iloc[0]['missing_weak_projection_scaled']))
        return
    out=root/OUTPUT
    if not out.is_dir():raise RuntimeError('Create this folder with Finder before running: '+str(out))
    for name in OUTFILES:
        if (out/name).exists():raise RuntimeError('Existing B9A10-B result protected, move old files using Finder first: '+str(out/name))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig1,axes=plt.subplots(1,2,figsize=(14,4.7),sharey=True)
    for ax,reg in zip(axes,REGIONS):
        v=df_a[df_a.region==reg].sort_values('x_mm')
        ax.plot(v.x_mm,v.weak_gap_pct_of_total_boundary_RF,'-o',label='One-point force gap')
        ax.plot(v.x_mm,v.B5_gap_pct_of_total_boundary_RF,'-s',label='B5-field force gap')
        ax.set_title(reg);ax.set_xlabel('Axial X (mm)');ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Plane force gap / boundary RF norm (%)')
    fig1.suptitle('B9A10-B | Localize remaining force discrepancy by axial plane')
    fig1.tight_layout()
    fig2,ax=plt.subplots(figsize=(max(12.,.65*len(df_m)),5.4))
    i=np.arange(len(df_m));wid=.38
    ax.bar(i-wid/2,np.abs(df_m.missing_weak_projection_scaled),wid,label='Unmodeled FE weak-form gap')
    ax.bar(i+wid/2,np.abs(df_m.B5_field_projection_scaled),wid,label='B5 displacement-induced gap')
    ax.set_xticks(i,[f'{x}/{k}' for x,k in zip(df_m.interface,df_m['mode'])],rotation=55,ha='right')
    ax.set_ylabel('Absolute generalized-force gap / original 65V scale')
    ax.set_title('B9A10-B | PCA force-mode attribution (not a correction)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig2.tight_layout()
    # Spatial scatter for most affected interface, mode.
    im=top.iloc[0];sel=df_n[(df_n.interface==im['interface'])&(df_n['mode']==im['mode'])]
    fig3,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,col,title in zip(axes,['FE_weak_gap_mode_contribution_scaled','B5_field_gap_mode_contribution_scaled'],
              ['Missing FE weak force: per-node mode contribution','B5 displacement force: per-node mode contribution']):
        maxval=max(np.max(np.abs(sel[col])),1e-10)
        z=ax.scatter(sel.y_mm,sel.z_mm,c=sel[col],cmap='coolwarm',vmin=-maxval,vmax=maxval,s=35)
        fig3.colorbar(z,ax=ax);ax.set_title(title,fontsize=10);ax.set_xlabel('Y (mm)');ax.set_ylabel('Z (mm)')
    fig3.suptitle('B9A10-B | %s, PCA mode %d'%(im['interface'],im['mode']))
    fig3.tight_layout()
    for f,df in zip(OUTFILES[:5],[df_r,df_a,df_n,df_m.drop(columns='gap_abs_scaled'),df_c]):df.to_csv(out/f,index=False)
    (out/OUTFILES[5]).write_text(json.dumps(summary,indent=2))
    for fig,fn in zip([fig1,fig2,fig3],OUTFILES[6:]):fig.savefig(out/fn,dpi=180);plt.close(fig)
    print('B9A10-B COMPLETE | outputs:',out)
    print('No constitutive corrections, mode clipping, force fitting, or retraining performed.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.selftest:selftest();return
    run(Path(args.project).expanduser().resolve(),args.check_only)

if __name__=='__main__':main()
