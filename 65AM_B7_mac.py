#!/usr/bin/env python3
"""65AM-B7: READ-ONLY orthotropic-stress proxy audit on frozen six-case B6 cohort.

Mac project root prerequisites:
- existing 65AM_B6_mac.py, 65AM_B5_common.py, 65AM_B_common.py etc
- frozen B5 centered_grad and 65AA checkpoints
- results/65AM_B6_independent/C_packages/{inner_left,inner_right}.npz
- results/65AM_B6_independent/A_plan/{65AM_B6_cases.csv,65AM_B6_plan.json}
- B7_abaqus_audit/B7_windows_audit.json from Windows Abaqus check

Run:
 python3 65AM_B7_mac.py --project . --check-only
 python3 65AM_B7_mac.py --project .

Uses full FEM U ONLY to derive prescribed interface values and score errors.
FEM LE/S never enter B5/65AA input; FEM strain enters only an oracle
constitutive sanity check. S=C@log(U) is an APPROXIMATION, not exact NLGEOM
Abaqus material integration and not an equilibrium or traction audit.
No model re-training or 65V solver mutation. No external data selection.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import importlib
b6=importlib.import_module('65AM_B6_mac')
b=b6.b
R=('inner_left','inner_right')
METHODS=('B5_constitutive','65AA_stress_head','FEM_LE_constitutive_oracle')
FIELDS=('S11','S22','S33','S12','S13','S23')


def relative(pred,truth):
    p=np.asarray(pred,dtype=np.float64); t=np.asarray(truth,dtype=np.float64)
    return float(100*np.linalg.norm((p-t).ravel())/max(np.linalg.norm(t.ravel()),1e-14))

def rmse(pred,truth):
    p=np.asarray(pred,dtype=np.float64);t=np.asarray(truth,dtype=np.float64)
    return float(np.sqrt(np.mean((p-t)**2)))

def read_audit(root,manifest_hash,names,manifest_mats):
    path=root/'B7_abaqus_audit'/'B7_windows_audit.json'
    if not path.is_file():raise FileNotFoundError('Copy verified Windows B7 audit JSON to '+str(path))
    j=json.loads(path.read_text())
    if not j.get('all_cases_passed') or j.get('case_ids')!=list(names):
        raise RuntimeError('Windows audit not successful for exact B6 case IDs')
    if j.get('manifest_sha256')!=manifest_hash:
        raise RuntimeError('Windows material manifest bytes differ from Mac B6 locked manifest')
    if len(j.get('cases',[]))!=len(names):raise RuntimeError('Windows audit cases missing')
    for k,(item,case) in enumerate(zip(j['cases'],names)):
        if item.get('case_id')!=case or not (item.get('inp_match') and item.get('sta_completed') and item.get('odb_checked')):
            raise RuntimeError('Windows audit incomplete for '+case)
        if not np.allclose(np.asarray(item['engineer_constants'])[[0,1,6]],
                           manifest_mats[k],rtol=1e-7,atol=1e-4):
            raise RuntimeError('Windows ODB input material does not match B6 locked manifest for '+case)
    fixed=j['fixed_orthotropic_constants']
    for k,v in {'E3':12000.,'nu12':.28,'nu13':.28,'nu23':.40,'G13':4500.,'G23':3500.}.items():
        if not np.isclose(float(fixed[k]),v,rtol=1e-8,atol=1e-6):
            raise RuntimeError('Material constants differ from 65Z model assumptions for '+k+
                               ': audit and revise constitutive model before proceeding')
    return j

def constitutive_stiffness(materials,fixed):
    """Orthotropic small-strain stiffness in local principal material axes.

    LE vector = [11,22,33,2*12,2*13,2*23] engineering shear convention.
    This is a proxy for Abaqus Cauchy stress under finite rotations, not an exact law.
    """
    m=np.asarray(materials,dtype=np.float64)
    mats=np.zeros((len(m),6,6),dtype=np.float64)
    for k,(e1,e2,g12) in enumerate(m):
        e3=float(fixed['E3']);v12=float(fixed['nu12']);v13=float(fixed['nu13'])
        v23=float(fixed['nu23']);g13=float(fixed['G13']);g23=float(fixed['G23'])
        S=np.diag([1/e1,1/e2,1/e3,1/g12,1/g13,1/g23])
        S[0,1]=S[1,0]=-v12/e1
        S[0,2]=S[2,0]=-v13/e1
        S[1,2]=S[2,1]=-v23/e2
        if np.min(np.linalg.eigvalsh(S))<=0:raise RuntimeError('Non-SPD orthotropic compliance')
        mats[k]=np.linalg.inv(S)
    return mats

def stress_from_LE(LE,C):
    return np.einsum('bqc,bec->beq',C,np.asarray(LE,dtype=np.float64),optimize=True)

def stress_65aa(root,d,branch,dev):
    """Frozen V4 65AA stress head: predicts S from branch, NOT from B5 U."""
    ds=d['ds'];ck=b.CO.load_checkpoint(d['source'])
    model=b.CO.make_model(ck,dev).eval()
    B=np.asarray(branch,dtype=np.float32)
    if B.shape[1]!=len(ck['branch_mean']):raise RuntimeError('65AA stress branch dimension mismatch')
    B=(B-np.asarray(ck['branch_mean'],np.float32))/np.asarray(ck['branch_std'],np.float32)
    xyz=np.asarray(ds['ip_coordinates'],np.float32)
    low=np.asarray(ck['coordinate_min'],np.float32);high=np.asarray(ck['coordinate_max'],np.float32)
    X=2*(xyz-low)/np.maximum(high-low,1.e-8)-1
    if np.max(np.abs(X))>1.002:raise RuntimeError('65AA stress IP coordinates out of range')
    out=[]
    with torch.no_grad():
        for k in range(0,len(B),4):
            br=torch.as_tensor(B[k:k+4],dtype=torch.float32,device=dev)
            ip=torch.as_tensor(X,dtype=torch.float32,device=dev)
            norm=model.stress_operator(br,ip)
            physical=norm*torch.as_tensor(ck['S_std'],dtype=torch.float32,device=dev)+torch.as_tensor(ck['S_mean'],dtype=torch.float32,device=dev)
            out.append(physical.cpu().numpy())
    a=np.concatenate(out).astype(np.float64)
    if a.shape!=(len(B),len(xyz),6):raise RuntimeError('65AA stress output shape incorrect')
    if not np.all(np.isfinite(a)):raise RuntimeError('Nonfinite 65AA stress')
    return a

def safe_output(path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Existing B7 results protected; do not overwrite '+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path

def load_b6(root,dev):
    df,models=b6.check(root,dev)
    planfile=root/'results'/'65AM_B6_independent'/'A_plan'/'65AM_B6_plan.json'
    plan=json.loads(planfile.read_text())
    mf=b6.manifest_path(root)
    names=df.case_id.astype(str).tolist()
    mats=df[list(b6.COLS)].to_numpy(dtype=np.float64)
    audit=read_audit(root,b6.sha(mf),names,mats)
    packs={}
    for region in R:
        path=b6.refs_path(root)/(region+'.npz')
        if not path.is_file():raise FileNotFoundError('Missing B6 package '+str(path))
        with np.load(path,allow_pickle=False) as z:
            pack={key:z[key] for key in z.files}
        if pack['case_ids'].astype(str).tolist()!=names:raise RuntimeError('B6 case order differs for '+region)
        if not np.allclose(pack['materials'],mats,rtol=1e-6,atol=1e-4):raise RuntimeError('B6 materials changed')
        ds=models[region][0]['ds']; n=len(ds['coordinates']);e=len(ds['ip_coordinates'])
        for key,shape in (('U',(len(names),n,3)),('LE',(len(names),e,6)),('S',(len(names),e,6))):
            if pack[key].shape!=shape:raise RuntimeError(region+' '+key+' unexpected shape '+str(pack[key].shape))
            if not np.isfinite(pack[key]).all():raise RuntimeError(region+' invalid '+key)
        if not np.allclose(pack['coordinates'],ds['coordinates'],rtol=0,atol=1e-5):raise RuntimeError('Wrong geometry')
        packs[region]=pack
    b6summary=root/'results'/'65AM_B6_independent'/'D_independent_evaluation'/'B6_summary.json'
    if not b6summary.is_file():raise FileNotFoundError('B6 independent evaluation summary missing')
    summary=json.loads(b6summary.read_text())
    if summary.get('case_ids')!=names or not all(v.get('pass_gate') for v in summary.get('decisions',[])) or len(summary.get('decisions',[]))!=2:
        raise RuntimeError('B6 independent validation did not pass for both regions')
    package_manifest=root/'results'/'65AM_B6_independent'/'C_packages'/'B6_package_manifest.json'
    if json.loads(package_manifest.read_text())['source_case_manifest_sha256']!=b6.sha(mf):
        raise RuntimeError('B6 source manifest changed after packaging')
    return df,models,packs,audit,summary

def x_profiles(coords,ref,pred):
    xx=np.asarray(coords)[:,0]; keys=np.round(xx,6);unique=np.unique(keys)
    x=[];a=[];q=[]
    for k in unique:
        inds=np.flatnonzero(keys==k)
        x.append(float(k));a.append(float(np.mean(ref[inds])));q.append(float(np.mean(pred[inds])))
    return x,a,q

def plot_results(out,pooled,case_rows,profile_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # S22 case-wise error comparisons (not an inference architecture selection step)
    for region in R:
        sub=case_rows[(case_rows.region==region)&(case_rows.component=='S22')]
        fig,ax=plt.subplots(figsize=(8,4))
        for method in METHODS[:2]:
            rows=sub[sub.method==method].sort_values('case_id')
            ax.plot(rows.case_id,rows.rel_l2_pct,marker='o',label=method)
        ax.set_title('65AM-B7 | '+region+' | S22 error (B6 new cases)')
        ax.set_ylabel('Relative L2 error (%)');ax.set_xlabel('Abaqus case');ax.grid(alpha=.25)
        ax.legend(fontsize=8);plt.xticks(rotation=25);fig.tight_layout()
        fig.savefig(out/(region+'_S22_errors_by_case.png'),dpi=180);plt.close(fig)
        rows=profile_rows[(profile_rows.region==region)&(profile_rows.case_id=='B6_new_01')]
        fig,ax=plt.subplots(figsize=(9,4))
        for method in ('FEM','B5_constitutive','65AA_stress_head'):
            sub=rows[rows.method==method].sort_values('x_mm')
            ax.plot(sub.x_mm,sub.S22_MPa,marker='o',ms=3,label=method)
        ax.set_title(region+' | B6_new_01 | cross-section-average S22')
        ax.set_xlabel('Axial X (mm)');ax.set_ylabel('S22 (MPa)');ax.grid(alpha=.25)
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/(region+'_S22_axial_profile.png'),dpi=180);plt.close(fig)

def evaluate(root,dev,check_only):
    df,models,packs,audit,b6summary=load_b6(root,dev)
    mats=df[list(b6.COLS)].to_numpy(np.float64)
    names=df.case_id.astype(str).tolist()
    C=constitutive_stiffness(mats,audit['fixed_orthotropic_constants'])
    records=[];comp_rows=[];case_rows=[];profiles=[];audits=[]
    for region,(d,model,ck) in models.items():
        ds=d['ds'];pack=packs[region]
        Uref=np.asarray(pack['U'],np.float64);LEref=np.asarray(pack['LE'],np.float64);Sref=np.asarray(pack['S'],np.float64)
        # Independent FE kinematic check (should reproduce LE observed in B6).
        FEMderived=np.stack([b.CO.log_material_strain_eng(u,ds['element_node_indices'],ds['B_center']) for u in Uref])
        kin=relative(FEMderived,LEref)
        if kin>2:raise RuntimeError('FEM U does not reproduce FEM LE in '+region)
        # Offline FEM interface values only. No FEM interior target enters inference.
        branch,lift,fixed,diag=b6.project_boundary(root,ds,Uref,mats,dev)
        Ub5=b6.predict_b5(d,model,branch,lift,dev)
        if np.max(abs(Ub5[:,fixed,:]-lift[:,fixed,:]))>1e-5:
            raise RuntimeError('B5 interface reconstruction mismatch')
        LEb5=np.stack([b.CO.log_material_strain_eng(u,ds['element_node_indices'],ds['B_center']) for u in Ub5])
        Sconstit=stress_from_LE(LEb5,C)
        Soracle=stress_from_LE(LEref,C)
        Saa=stress_65aa(root,d,branch,dev)
        # Oracle sanity check is essential: linear orthotropic proxy may not reproduce NLGEOM Cauchy S.
        oracle_vec=relative(Soracle,Sref)
        oracle22=relative(Soracle[...,1],Sref[...,1])
        per_oracle22=[relative(Soracle[i,:,1],Sref[i,:,1]) for i in range(len(names))]
        oracle_pass=bool(oracle_vec<=2 and oracle22<=2 and max(per_oracle22)<=3)
        audits.append(dict(region=region,FE_U_to_LE_pct=kin,
                           FEM_LE_to_S_linear_proxy_vector_pct=oracle_vec,
                           FEM_LE_to_S_linear_proxy_S22_pct=oracle22,
                           max_case_proxy_S22_pct=max(per_oracle22),
                           proxy_acceptable=oracle_pass,
                           caveat='C@LE is approximation to Abaqus NLGEOM Cauchy stress, not exact constitutive update'))
        print('B7',region,'constitutive reference S22 error %0.4f%% (proxy %s)'%
              (oracle22,'ACCEPTABLE' if oracle_pass else 'NOT VERIFIED'))
        predictions={'B5_constitutive':Sconstit,
                     '65AA_stress_head':Saa,
                     'FEM_LE_constitutive_oracle':Soracle}
        for method,Spred in predictions.items():
            records.append(dict(region=region,method=method,n_cases=len(names),
                                S_vector_rel_pct=relative(Spred,Sref),S22_rel_pct=relative(Spred[...,1],Sref[...,1]),
                                S22_rmse_MPa=rmse(Spred[...,1],Sref[...,1]),
                                constitutive_proxy_verified=oracle_pass))
            for comp,label in enumerate(FIELDS):
                comp_rows.append(dict(region=region,method=method,component=label,
                                      rel_l2_pct=relative(Spred[...,comp],Sref[...,comp]),
                                      RMSE_MPa=rmse(Spred[...,comp],Sref[...,comp])))
            for i,case in enumerate(names):
                case_rows.append(dict(region=region,case_id=case,method=method,component='S22',
                                      rel_l2_pct=relative(Spred[i,:,1],Sref[i,:,1]),
                                      RMSE_MPa=rmse(Spred[i,:,1],Sref[i,:,1])))
            for i,case in enumerate(names[:1]):
                xx,yy,pp=x_profiles(ds['ip_coordinates'],Sref[i,:,1],Spred[i,:,1])
                for x,val in zip(xx,pp):
                    profiles.append(dict(region=region,case_id=case,method=method,x_mm=x,S22_MPa=val))
                if method=='B5_constitutive':
                    for x,val in zip(xx,yy):
                        profiles.append(dict(region=region,case_id=case,method='FEM',x_mm=x,S22_MPa=val))
    all_pass=all(x['proxy_acceptable'] for x in audits)
    if check_only:
        print('B7 INPUT CHECK PASS. Constitutive proxy:',
              'ACCEPTABLE for both regions' if all_pass else 'FAILED in at least one region.')
        print('No output files created.')
        return
    out=safe_output(root/'results'/'65AM_B7_stress_consistency'/'A_constitutive_audit')
    dfp=pd.DataFrame(records);dfc=pd.DataFrame(case_rows);dfcomp=pd.DataFrame(comp_rows);dfpr=pd.DataFrame(profiles)
    dfp.to_csv(out/'B7_stress_pooled_metrics.csv',index=False)
    dfc.to_csv(out/'B7_stress_case_metrics.csv',index=False)
    dfcomp.to_csv(out/'B7_stress_component_metrics.csv',index=False)
    dfpr.to_csv(out/'B7_S22_axial_data.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'B7_reference_constitutive_audit.csv',index=False)
    if all_pass:
        plot_results(out,dfp,dfc,dfpr)
    summary=dict(step='65AM-B7',cases=names,cohort='B6 six new Abaqus material cases',
                 FEM_oracle_interface_inputs=True,not_full_7region_coupling=True,
                 no_B5_or_LGNO_retraining=True,
                 B5_model='centered_grad frozen both regions',
                 stress_method='Approximate C_ortho @ HenckyLE(B5 U); engineering shear convention',
                 direct_65AI_external_inference_included=False,
                 reason_no_65AI='65AI expects 65AA/65AC/65AG frozen features; B6 replay requires a separately verified external feature pipeline',
                 law_proxy_acceptable_for_both=all_pass,validated_against_FEM_S=audits,
                 deployment_allowed=False,
                 note='Reference oracle is diagnostic only, not model prediction. Even an acceptable constitutive stress proxy does not establish force equilibrium or contact/traction accuracy.')
    (out/'B7_summary.json').write_text(json.dumps(summary,indent=2))
    print('B7 AUDIT COMPLETE:',out)
    if not all_pass:print('IMPORTANT: Constitutive proxy was NOT VALIDATED. Do not interpret B5 stress numbers as mechanically reliable.')
    else:print('Constitutive proxy passed reference check; interpret B5 stress as offline approximation, not equilibrium.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();root=Path(args.project).expanduser().resolve()
    device=b.CO.device_from_option(args.device)
    evaluate(root,device,args.check_only)

if __name__=='__main__':main()
