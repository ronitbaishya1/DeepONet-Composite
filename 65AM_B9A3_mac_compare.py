#!/usr/bin/env python3
"""B9A3 Mac: compare two new NO-side Abaqus RF references vs old NO and B5.

python3 65AM_B9A3_mac_compare.py --project . --selftest
python3 65AM_B9A3_mac_compare.py --project . --check-only
python3 65AM_B9A3_mac_compare.py --project .

Input files:
  B9_coupled_anchor/B9_65V_anchor.npz + manifest
  B9A2_measured_FE/B9A2_measured_FE.npz + manifest
  B9A3_exports/B9A3_NO_reference.npz + B9A3_windows_summary.json
  results/65AM_B9_coupled_feasibility/A_saved_state/B9_interface_force_modes.csv
No model weights, training, or solver outputs are modified.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
COUNTS=(4,4,5,5,4,4)
REPLACED=('left_inner','center_left','center_right','right_inner')
OUT=Path('results/65AM_B9_coupled_feasibility/B9A3_direct_NO_reference')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def npz(p):
    if not p.is_file():raise FileNotFoundError('Missing input file: '+str(p))
    with np.load(p,allow_pickle=False) as z:
        return {k:np.asarray(z[k]).copy() for k in z.files}

def readjson(p):
    if not p.is_file():raise FileNotFoundError('Missing input manifest: '+str(p))
    return json.loads(p.read_text())

def rms(a):return float(np.sqrt(np.mean(np.asarray(a,float)**2)))
def norm(a):return float(np.linalg.norm(np.asarray(a,float)))
def rel(a,b):return 100.0*norm(np.asarray(a)-np.asarray(b))/max(norm(b),1.e-12)

def segments():
    k=0
    for name,n in zip(NAMES,COUNTS):
        yield name,slice(k,k+n)
        k+=n

def selftest():
    # Work conjugacy under PCA: delta_u = B dq, force_g = B.T f
    rng=np.random.default_rng(11)
    A=rng.standard_normal((336,5));B,_=np.linalg.qr(A)
    dq=rng.standard_normal(5);f=rng.standard_normal(336)
    assert np.allclose((B@dq)@f,dq@(B.T@f),atol=1e-12)
    # Differential signed equilibrium: a perfect negative reaction cancels FEM.
    fem=np.arange(1.,27.);scale=np.arange(1.,27.)*.7
    assert rms((fem-fem)/scale)==0
    assert abs(rms((fem+fem)/scale)-rms(2*fem/scale))<1e-12
    print('B9A3 Mac selftest PASS: virtual-work projection, scaling and residual signs')

def main(root,check_only=False):
    anchor_file=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    am=root/'B9_coupled_anchor'/'B9_65V_anchor_manifest.json'
    femfile=root/'B9A2_measured_FE'/'B9A2_measured_FE.npz'
    fm=root/'B9A2_measured_FE'/'B9A2_windows_summary.json'
    reffile=root/'B9A3_exports'/'B9A3_NO_reference.npz'
    rm=root/'B9A3_exports'/'B9A3_windows_summary.json'
    b9csv=root/'results/65AM_B9_coupled_feasibility/A_saved_state/B9_interface_force_modes.csv'
    for p in (anchor_file,am,femfile,fm,reffile,rm,b9csv):
        if not p.is_file():raise FileNotFoundError('Missing required file: '+str(p))
    anchor=npz(anchor_file);fem=npz(femfile);no=npz(reffile)
    ameta=readjson(am);fmeta=readjson(fm);nmeta=readjson(rm)
    hash_anchor=sha(anchor_file)
    if hash_anchor!=ameta.get('anchor_sha256') or hash_anchor!=fmeta.get('anchor_sha256') or hash_anchor!=nmeta.get('anchor_sha256'):
        raise RuntimeError('Anchor SHA256 mismatch across Windows/Mac B9 stages')
    if sha(femfile)!=fmeta.get('file_sha256'):raise RuntimeError('B9A2 measured FE file checksum mismatch')
    if sha(reffile)!=nmeta.get('export_sha256'):raise RuntimeError('B9A3 NO reference file checksum mismatch')
    if str(no.get('anchor_sha256',np.array('')).item())!=hash_anchor:raise RuntimeError('Wrong saved 65V anchor in NO reference NPZ')
    if nmeta.get('geometry_and_displacement_checks') is not True:raise RuntimeError('NO ODB geometry/BC verification missing')
    if len(nmeta.get('cases',[]))!=2:raise RuntimeError('Expected two Abaqus reference ODBs')
    mat=np.asarray(anchor['material'],float).reshape(3)
    if not np.allclose(np.asarray(no['material'],float).reshape(3),mat,rtol=0,atol=1e-10):
        raise RuntimeError('Material mismatch between saved state and reference jobs')
    sf=np.asarray(anchor['force_scale'],float).ravel()
    old=np.asarray(anchor['old_NO_g'],float).ravel()
    FE=np.asarray(fem['measured_FE_g'],float).ravel()
    verified=np.asarray(anchor['normalized_residual'],float).ravel()
    mask=np.asarray(no['reference_mask']).ravel().astype(bool)
    ref=np.asarray(no['reference_NO_g'],float).ravel()
    if any(a.shape!=(26,) for a in (sf,old,FE,ref,mask,verified)) or not np.isfinite(sf).all() or np.any(sf<=0):
        raise RuntimeError('26-mode 65V input data invalid')
    expect=np.zeros(26,dtype=bool)
    for name,sl in segments():
        if name in REPLACED:expect[sl]=True
    if not np.array_equal(mask,expect):raise RuntimeError('Incorrect reference mode mask / interface mapping')
    if not np.isfinite(ref[mask]).all():raise RuntimeError('Missing true reference forces at inner interfaces')
    if not np.allclose(np.asarray(fem['old_NO_g'],float),old,atol=1e-9,rtol=0):raise RuntimeError('B9A2 FE and original NO disagreement')
    if not np.allclose(np.asarray(fem['force_scale'],float),sf,atol=1e-9,rtol=0):raise RuntimeError('Force scale mismatch')
    old_resid=(FE+old)/sf
    if rms(old_resid-verified)>1e-5:raise RuntimeError('Measured Abaqus FE forces do not reproduce original 65V RMS')
    df=pd.read_csv(b9csv)
    needed={'interface','mode_index','old_NO_generalized_force','B5_stress_projected_force'}
    if not needed.issubset(df.columns):raise RuntimeError('Prior B9 report missing columns '+str(needed-set(df.columns)))
    b5=old.copy()
    for name,sl in segments():
        rows=df[df.interface==name].sort_values('mode_index')
        if list(rows.mode_index.astype(int))!=list(range(1,sl.stop-sl.start+1)):
            raise RuntimeError('B9 B5 mode report incomplete '+name)
        if not np.allclose(rows.old_NO_generalized_force.to_numpy(float),old[sl],atol=1e-5,rtol=0):
            raise RuntimeError('Old NO force changed from B9 baseline '+name)
        if name in REPLACED:
            b5[sl]=rows.B5_stress_projected_force.to_numpy(float)
    if not np.isfinite(b5).all():raise RuntimeError('Nonfinite B5 mapped force')
    oracle=old.copy();oracle[mask]=ref[mask]
    b5_resid=(FE+b5)/sf
    oracle_resid=(FE+oracle)/sf
    signrev=old.copy();signrev[mask]=-b5[mask]
    # Reference force convention check: Abaqus reaction should cancel opposite FE RF
    interfaces=[];modes=[]
    for name,sl in segments():
        yes=name in REPLACED
        row=dict(interface=name,replaced_by_B5=yes,n_modes=sl.stop-sl.start,
                 FE_plus_old_RMS=rms(old_resid[sl]),
                 FE_plus_B5_RMS_estimate=rms(b5_resid[sl]) if yes else None,
                 FE_plus_true_NO_Abaqus_RMS_estimate=rms(oracle_resid[sl]) if yes else None,
                 true_NO_force_norm_N=norm(ref[sl]) if yes else None,
                 old_NO_vs_true_NO_pct=rel(old[sl],ref[sl]) if yes else None,
                 B5_vs_true_NO_pct=rel(b5[sl],ref[sl]) if yes else None,
                 B5_vs_true_NO_scaled_RMS=rms((b5[sl]-ref[sl])/sf[sl]) if yes else None,
                 B5_force_best_scalar_diagnostic=float(np.dot(b5[sl],ref[sl])/np.dot(b5[sl],b5[sl])) if yes and norm(b5[sl])>1e-10 else None)
        interfaces.append(row)
        for j in range(sl.start,sl.stop):
            modes.append(dict(interface=name,mode_index=j-sl.start+1,measured_FE_RF_N=float(FE[j]),
                original_NO_generalized_N=float(old[j]),B5_stress_derived_N=float(b5[j]) if yes else None,
                true_NO_Abaqus_RF_N=float(ref[j]) if yes else None,force_scale_N=float(sf[j]),
                FE_plus_old_normalized=float(old_resid[j]),
                FE_plus_B5_normalized=float(b5_resid[j]) if yes else None,
                FE_plus_true_NO_normalized=float(oracle_resid[j]) if yes else None,
                B5_minus_true_NO_normalized=float((b5[j]-ref[j])/sf[j]) if yes else None))
    report=dict(step='65AM-B9A3',saved_65V_anchor_sha256=hash_anchor,
        original_65V_measured_FE_plus_old_NO_RMS=rms(old_resid),
        B5_plus_measured_FE_RMS_at_saved_q_ESTIMATE=rms(b5_resid),
        Abaqus_NO_reference_plus_measured_FE_RMS_at_saved_q_ESTIMATE=rms(oracle_resid),
        B5_vs_true_NO_force_scaled_RMS_inner_modes=rms((b5[mask]-ref[mask])/sf[mask]),
        old_NO_vs_true_NO_force_scaled_RMS_inner_modes=rms((old[mask]-ref[mask])/sf[mask]),
        B5_reverse_sign_plus_FE_RMS_DIAGNOSTIC=rms((FE+signrev)/sf),
        B5_direct_force_vs_Abaqus_better_than_old_NO=bool(rms((b5[mask]-ref[mask])/sf[mask]) < rms((old[mask]-ref[mask])/sf[mask])),
        two_NO_side_Abaqus_jobs_confirmed=True,
        full_coupled_B5_solution_verified=False,live_Broyden_approved=False,
        comments=['No sign flip or empirical scaling was applied',
                  'True NO reaction RF gives an independent reference for the four inner interfaces at the same 65V boundary displacement',
                  'The true NO-reference + FE residual at old q is not a new converged coupled solve',
                  'FE/NO matching may still require a consistent weak-form force operator; B5 traction quadrature is approximate'])
    print('B9A3 REFERENCE CHECK PASS | old solver %.6f | B5 estimate %.6f | true NO-side Abaqus replay %.6f'%(
          report['original_65V_measured_FE_plus_old_NO_RMS'],report['B5_plus_measured_FE_RMS_at_saved_q_ESTIMATE'],
          report['Abaqus_NO_reference_plus_measured_FE_RMS_at_saved_q_ESTIMATE']))
    for row in interfaces:
        if row['replaced_by_B5']:
            print('%-13s B5 vs Abaqus RF %.3f%% | old NO vs Abaqus RF %.3f%% | B5 vs true scaled RMS %.5f'%(
                row['interface'],row['B5_vs_true_NO_pct'],row['old_NO_vs_true_NO_pct'],row['B5_vs_true_NO_scaled_RMS']))
    if check_only:
        print('CHECK ONLY: no reports written');return
    out=root/OUT
    if out.exists() and any(out.iterdir()):raise RuntimeError('Existing B9A3 report protected: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(interfaces).to_csv(out/'B9A3_interface_comparison.csv',index=False)
    pd.DataFrame(modes).to_csv(out/'B9A3_mode_comparison.csv',index=False)
    (out/'B9A3_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sel=[x for x in interfaces if x['replaced_by_B5']]
    idx=np.arange(len(sel));w=.25
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar(idx-w,[x['FE_plus_old_RMS'] for x in sel],width=w,label='FE + original NO')
    ax.bar(idx,[x['FE_plus_B5_RMS_estimate'] for x in sel],width=w,label='FE + B5 stress-derived')
    ax.bar(idx+w,[x['FE_plus_true_NO_Abaqus_RMS_estimate'] for x in sel],width=w,label='FE + true NO-side Abaqus')
    ax.axhline(.02,color='black',linestyle='--',label='Old 65V target 0.02')
    ax.set_xticks(idx);ax.set_xticklabels([x['interface'] for x in sel],rotation=18,ha='right')
    ax.set_ylabel('Normalized interface residual RMS at original 65V q')
    ax.set_title('B9A3 | Independent NO-side Abaqus force benchmark')
    ax.legend();fig.tight_layout();fig.savefig(out/'B9A3_interface_RMS.png',dpi=180);plt.close(fig)
    print('SAVED:',out)
    print('Do not replace original 65V or launch Broyden until force definition is reconciled.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest()
    else:main(Path(a.project).expanduser().resolve(),a.check_only)
