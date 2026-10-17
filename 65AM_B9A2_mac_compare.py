#!/usr/bin/env python3
"""65AM-B9A2-B: compare real 65V FEM reaction forces to old NO and B5 stress forces.

Read-only relative to saved 65V and B5. Uses measured Abaqus FE patch reaction
projection, not FE reconstructed from the old residual. B5 is a saved-state
replay; candidate RMS is STILL NOT a new converged hybrid solution.

Run from Mac DeepONet_Composite project root:
 python3 65AM_B9A2_mac_compare.py --project . --check-only
 python3 65AM_B9A2_mac_compare.py --project .
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
COUNTS=(4,4,5,5,4,4)
REPLACED=('left_inner','center_left','center_right','right_inner')
OUT=Path('results/65AM_B9_coupled_feasibility/B9A2_direct_FE_projection')

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rms(v):return float(np.sqrt(np.mean(np.asarray(v,float)**2)))

def rel(a,b):return 100*float(np.linalg.norm(np.asarray(a)-np.asarray(b)))/max(float(np.linalg.norm(b)),1.e-12)

def load_npz(path):
    if not path.is_file():raise FileNotFoundError(str(path))
    with np.load(path,allow_pickle=False) as v:
        return {k:np.asarray(v[k]).copy() for k in v.files}

def run(root,check_only):
    base=root/'B9_coupled_anchor'
    win=root/'B9A2_measured_FE'
    anchor_path=base/'B9_65V_anchor.npz'
    anchor_manifest=base/'B9_65V_anchor_manifest.json'
    measured_path=win/'B9A2_measured_FE.npz'
    measured_manifest=win/'B9A2_windows_summary.json'
    old_b9=root/'results'/'65AM_B9_coupled_feasibility'/'A_saved_state'/'B9_interface_force_modes.csv'
    for p in (anchor_path,anchor_manifest,measured_path,measured_manifest,old_b9):
        if not p.is_file():raise FileNotFoundError('Required file missing: '+str(p))
    a=load_npz(anchor_path);me=load_npz(measured_path)
    with measured_manifest.open() as f:md=json.load(f)
    with anchor_manifest.open() as f:ad=json.load(f)
    if hash_file(measured_path)!=md.get('file_sha256'):raise RuntimeError('Transferred measured-FE NPZ failed SHA256')
    if hash_file(anchor_path)!=md.get('anchor_sha256') or hash_file(anchor_path)!=ad.get('anchor_sha256'):
        raise RuntimeError('Windows B9A2 measured-FE and saved B9 anchor are not the same run')
    needed=('measured_FE_g','old_NO_g','force_scale','old_verified_normalized_residual')
    for n in needed:
        if n not in me or np.asarray(me[n]).shape!=(26,) or not np.isfinite(me[n]).all():
            raise RuntimeError('Measured data missing/wrong shape '+n)
    for n in ('old_NO_g','force_scale','normalized_residual'):
        if n not in a or np.asarray(a[n]).shape!=(26,):raise RuntimeError('Anchor missing '+n)
    for x,y in (('old_NO_g','old_NO_g'),('force_scale','force_scale'),
                ('old_verified_normalized_residual','normalized_residual')):
        if not np.allclose(me[x],a[y],rtol=0,atol=1e-9):raise RuntimeError('Windows/Mac anchor disagreement '+x)
    force_scale=np.asarray(a['force_scale'],float)
    if (force_scale<=0).any():raise RuntimeError('Nonpositive force normalization')
    df=pd.read_csv(old_b9)
    needed_cols={'interface','mode_index','old_NO_generalized_force','B5_stress_projected_force'}
    if not needed_cols.issubset(df.columns):raise RuntimeError('Prior B9 modes CSV missing columns '+str(needed_cols-set(df.columns)))
    old=np.asarray(a['old_NO_g'],float)
    fe=np.asarray(me['measured_FE_g'],float)
    inferred=np.asarray(a['FE_g_inferred'],float)
    b5=old.copy();i=0
    for name,c in zip(NAMES,COUNTS):
        rows=df[df['interface']==name].sort_values('mode_index')
        if len(rows)!=c or list(rows.mode_index.astype(int))!=list(range(1,c+1)):
            raise RuntimeError('B9 modes out of order or missing '+name)
        if not np.allclose(rows.old_NO_generalized_force.astype(float),old[i:i+c],rtol=0,atol=1e-5):
            raise RuntimeError('B9 report disagrees with 65V anchor at '+name)
        if name in REPLACED:
            candidate=rows.B5_stress_projected_force.to_numpy(dtype=float)
            if not np.isfinite(candidate).all():raise RuntimeError('B5 projected forces missing '+name)
            b5[i:i+c]=candidate
        i+=c
    old_r=(fe+old)/force_scale
    new_r=(fe+b5)/force_scale
    saved_r=np.asarray(a['normalized_residual'],float)
    fe_infer_gap=(fe-inferred)/force_scale
    # Test sign flip as a diagnosis only; never auto-apply.
    flip=b5.copy()
    msk=np.zeros(26,dtype=bool);i=0
    for name,c in zip(NAMES,COUNTS):
        if name in REPLACED:flip[i:i+c]*=-1;msk[i:i+c]=True
        i+=c
    flip_r=(fe+flip)/force_scale
    rows=[];modes=[];i=0
    for name,c in zip(NAMES,COUNTS):
        sl=slice(i,i+c);i+=c
        target=-fe[sl];cand=b5[sl];g0=old[sl]
        # alpha fit for curiosity: candidate * alpha approximates -FE at this one point.
        alpha=float((cand@target)/(cand@cand)) if np.linalg.norm(cand)>1e-12 else None
        rows.append(dict(interface=name,replaced_by_B5=name in REPLACED,
                         reference_FE_RF_norm_N=float(np.linalg.norm(fe[sl])),
                         original_NO_norm_N=float(np.linalg.norm(g0)),
                         candidate_B5_norm_N=float(np.linalg.norm(cand)),
                         RF_vs_inferred_FE_rel_pct=rel(fe[sl],inferred[sl]),
                         RF_vs_inferred_FE_scaled_RMS=rms(fe_infer_gap[sl]),
                         original_plus_measured_FE_RMS=rms(old_r[sl]),
                         B5_plus_measured_FE_RMS_ESTIMATE=rms(new_r[sl]),
                         B5_reverse_sign_plus_measured_FE_RMS_DIAGNOSTIC=rms(flip_r[sl]) if name in REPLACED else None,
                         old_NO_vs_negative_FE_rel_pct=rel(g0,target),
                         B5_vs_negative_FE_rel_pct=rel(cand,target) if name in REPLACED else None,
                         B5_vs_old_NO_rel_pct=rel(cand,g0) if name in REPLACED else None,
                         B5_best_fit_scalar_DIAGNOSTIC=alpha if name in REPLACED else None))
        for k in range(c):
            j=sl.start+k
            modes.append(dict(interface=name,mode_index=k+1,
                              measured_FE_RF_N=float(fe[j]),
                              inferred_FE_RF_N=float(inferred[j]),
                              original_NO_force_N=float(old[j]),
                              candidate_B5_force_N=float(b5[j]) if name in REPLACED else None,
                              force_scale_N=float(force_scale[j]),
                              original_verified_saved_residual=float(saved_r[j]),
                              reproduced_old_residual=float(old_r[j]),
                              B5_replacement_residual_ESTIMATE=float(new_r[j]),
                              FE_inference_discrepancy_normalized=float(fe_infer_gap[j])))
    old_RMS=rms(old_r);new_RMS=rms(new_r);saved_RMS=rms(saved_r)
    projection_gate=(rms(old_r-saved_r)<.002 and np.max(abs(old_r-saved_r))<.01)
    report=dict(step='65AM-B9A2',reference='Actual ABAQUS RF from saved 65V FE patch ODBs',
                saved_65V_RMS=saved_RMS,measured_FE_plus_original_NO_RMS=old_RMS,
                previous_inferred_vs_measured_FE_scaled_RMS=rms(fe_infer_gap),
                candidate_B5_plus_measured_FE_RMS_ESTIMATE=new_RMS,
                B5_sign_reversed_RMS_DIAGNOSTIC=rms(flip_r),
                FE_reaction_projection_reproduces_65V=bool(projection_gate),
                candidate_substitution_meets_002=bool(new_RMS<=.02),
                approved_for_live_Broyden=False,
                interpretation=('Projection of real FE patch RF reproduces the prior verified 65V residual' if projection_gate else
                                'FE ODB RF does not reproduce saved 65V residual: STOP; inspect patch selection, PCA ordering and source provenance'),
                limitations=['Candidate RMS is measured FE at old q plus B5 NO at old q: not a re-equilibrated or converged B5 hybrid solution',
                             'No sign flip or numerical rescaling automatically applied',
                             'Face-traction approximation may differ from nodal reactions (hourglass/contact/finite-element effects)',
                             'A true B5 NO-side FEM benchmark at these prescribed 65V interface displacements is still needed if force discrepancy remains'],
                next_step=('B9A3: controlled B5 NO-side patch reference and shared force convention audit' if projection_gate else
                           'Fix ODB/anchor mismatch; do not attempt Broyden'))
    print('B9A2 direct FE RF comparison | saved old RMS %.8f | measured FE+old RMS %.8f | candidate FE+B5 RMS %.8f'%(saved_RMS,old_RMS,new_RMS))
    print('FE ODB projection reproduces 65V:',projection_gate)
    for row in rows:
        print('%-13s original %.5f B5 %.5f%s'%(row['interface'],row['original_plus_measured_FE_RMS'],row['B5_plus_measured_FE_RMS_ESTIMATE'],
            ' | B5/negative-FE %.2f%%'%row['B5_vs_negative_FE_rel_pct'] if row['replaced_by_B5'] else ''))
    if check_only:
        print('CHECK ONLY: no outputs written. Run without --check-only for reports.')
        return
    out=root/OUT
    if out.exists() and any(out.iterdir()):raise RuntimeError('Existing B9A2 result protected: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'B9A2_interface_comparison.csv',index=False)
    pd.DataFrame(modes).to_csv(out/'B9A2_mode_comparison.csv',index=False)
    (out/'B9A2_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[r['interface'] for r in rows]
    orig=[r['original_plus_measured_FE_RMS'] for r in rows]
    cand=[r['B5_plus_measured_FE_RMS_ESTIMATE'] for r in rows]
    fig,ax=plt.subplots(figsize=(9,4.8))
    z=np.arange(len(names))
    ax.bar(z-.18,orig,.36,label='Old NO + measured FEM RF')
    ax.bar(z+.18,cand,.36,label='B5 + measured FEM RF (not converged)')
    ax.axhline(.02,color='black',linestyle='--',linewidth=1,label='Target .02')
    ax.set_xticks(z);ax.set_xticklabels(names,rotation=20,ha='right');ax.set_ylabel('Normalized interface residual RMS')
    ax.set_title('B9-A2 | Real Abaqus RF at saved 65V state')
    ax.legend();fig.tight_layout();fig.savefig(out/'B9A2_interface_residuals.png',dpi=180);plt.close(fig)
    print('SAVED:',out)
    print('Do NOT proceed to online coupling solely from these results.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    run(Path(args.project).expanduser().resolve(),args.check_only)

if __name__=='__main__':main()
