#!/usr/bin/env python3
"""B9A6 Mac: produce frozen-B5 whole-field displacement for Abaqus kinematic replay;
compare true Abaqus C3D8R nodal reactions to B5 one-point force predictions.

Never retrains B5 or runs Broyden. Outputs protected. All transfer folders must be
created by user via Finder; scripts never create/clear project directories.

Prepare:
 python3 65AM_B9A6_mac.py --project . --stage prepare --check-only
 python3 65AM_B9A6_mac.py --project . --stage prepare
Compare (after four Windows Abaqus jobs and transfer):
 python3 65AM_B9A6_mac.py --project . --stage compare --check-only
 python3 65AM_B9A6_mac.py --project . --stage compare
Selftest: python3 65AM_B9A6_mac.py --selftest
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ('inner_left', 'inner_right')
IFACES = ('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
COUNTS = (4,4,5,5,4,4)
INTERFACES = {
  'inner_left': ('left_inner','center_left'),
  'inner_right': ('center_right','right_inner'),
}
TRANSFER = Path('B9A6_transfer_to_windows')
EXPORTS = Path('B9A6_exports')
OUT = Path('results/65AM_B9_coupled_feasibility/B9A6_kinematic_replay')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    if not Path(path).is_file():raise FileNotFoundError(str(path))
    return json.loads(Path(path).read_text())


def npz(path):
    if not Path(path).is_file():raise FileNotFoundError(str(path))
    with np.load(path,allow_pickle=False) as a:
        return {k:np.asarray(a[k]).copy() for k in a.files}


def rms(x):
    x=np.asarray(x,float).reshape(-1)
    return float(np.sqrt(np.mean(x*x)))


def relative(a,b):
    a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1)
    return float(100*np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))


def mode_slices():
    start=0
    for n,k in zip(IFACES,COUNTS):
        yield n,slice(start,start+k)
        start+=k


def anchor_sources(root):
    ap=root/'B9_coupled_anchor/B9_65V_anchor.npz'
    am=root/'B9_coupled_anchor/B9_65V_anchor_manifest.json'
    d=npz(ap);meta=read_json(am)
    h=sha(ap)
    if h!=meta.get('anchor_sha256'):
        raise RuntimeError('B9 anchor does not match its manifest SHA256')
    for name in ('force_scale','old_NO_g','normalized_residual','material'):
        if name not in d:raise RuntimeError('B9 anchor missing '+name)
    for name in ('force_scale','old_NO_g','normalized_residual'):
        if d[name].shape!=(26,):raise RuntimeError('Incorrect shape for '+name)
    if np.any(d['force_scale']<=0):raise RuntimeError('Nonpositive 65V force scaling')
    if rms(d['normalized_residual'])>.02001:
        raise RuntimeError('Saved original 65V state is not converged')
    return d,h


def field_source(root,anchhash):
    p=root/'B9A4_exports/B9A4_reference_fields.npz'
    m=read_json(root/'B9A4_exports/B9A4_windows_summary.json')
    if m.get('anchor_sha256')!=anchhash or m.get('reference_fields_sha256')!=sha(p):
        raise RuntimeError('B9A4 reference does not match the 65V anchor or hash')
    return npz(p),sha(p)


def prepare(root,check):
    import torch
    b6=importlib.import_module('65AM_B6_mac')
    b9=importlib.import_module('65AM_B9_mac_saved_state')
    # B9A4 named original or fixed (user may have kept either).
    try: b9a4=importlib.import_module('65AM_B9A4_mac_diagnose_FIXED')
    except ModuleNotFoundError: b9a4=importlib.import_module('65AM_B9A4_mac_diagnose')
    anchor,ah=anchor_sources(root)
    ref,rh=field_source(root,ah)
    dev=b6.b.CO.device_from_option('auto')
    _,models=b6.check(root,dev)
    payload={'anchor_sha256':np.asarray(ah),'source_reference_sha256':np.asarray(rh)}
    manifest={'step':'65AM-B9A6','anchor_sha256':ah,'reference_fields_sha256':rh,
              'regions':{},'status':'offline kinematic replay, no live coupling'}
    for reg in REGIONS:
        prepared,net,ck=models[reg]
        ds=dict(prepared['ds']);ds['_b9_region']=reg
        context=b6.b.CO.interface_context(root,ds,dev)
        branch,lift=b9.build_branch_lift(root,ds,anchor,context,dev)
        # Optional checkpoint digest from known model path is not inferred.
        with torch.no_grad():uhat=np.asarray(b6.predict_b5(prepared,net,branch,lift,dev)[0],float)
        x=np.asarray(ds['coordinates'],float)
        labels=np.asarray(ref[reg+'_node_labels'],int)
        refxyz=np.asarray(ref[reg+'_node_xyz'],float)
        idx,gap=b9a4.map_unique(x,refxyz)
        uh=uhat[idx]
        if len(labels)!=1008 or x.shape!=(1008,3) or refxyz.shape!=(1008,3):
            raise RuntimeError('Unexpected B9A3 1008-node geometry for '+reg)
        if np.unique(labels).size!=len(labels) or not np.isfinite(uh).all():
            raise RuntimeError('Bad node labels / B5 displacement '+reg)
        uref=np.asarray(ref[reg+'_U'],float)
        if uref.shape!=uh.shape or np.max(abs(uh))>10:
            raise RuntimeError('Invalid B5/reference displacement shapes or values')
        # Hard interface values MUST agree with reference prescribed 65V displacements.
        xmax=np.max(refxyz[:,0]);xmin=np.min(refxyz[:,0]);ends=(abs(refxyz[:,0]-xmin)<1.e-5)|(abs(refxyz[:,0]-xmax)<1.e-5)
        hard_gap=float(np.max(abs(uh[ends]-uref[ends])))
        if hard_gap>5.e-5:raise RuntimeError('B5/reference prescribed interface disagreement: '+reg+' gap '+str(hard_gap))
        # The MAC field result MUST be in the original B9A4 ODB node-label order.
        for k,v in {'node_labels':labels,'node_xyz':refxyz,'U_B5':uh,'U_FEM':uref}.items():
            payload[reg+'_'+k]=v
        manifest['regions'][reg]={'nodes':len(labels),'coordinate_max_mm':gap,
                'max_interface_U_difference_mm':hard_gap,
                'field_displacement_rel_pct':relative(uh,uref)}
    print('B9A6 MAC PREPARE PASS',json.dumps(manifest['regions'],indent=2))
    if check:
        print('B9A6 PREPARE CHECK ONLY; no files created')
        return
    folder=root/TRANSFER
    if not folder.is_dir():raise RuntimeError('In Finder create '+str(folder)+' before running')
    p=folder/'B9A6_kinematic_fields.npz';mp=folder/'B9A6_mac_manifest.json'
    if p.exists() or mp.exists():raise RuntimeError('B9A6 transfer files already exist; leave prior results untouched')
    np.savez(p,**payload)
    manifest['transfer_sha256']=sha(p)
    mp.write_text(json.dumps(manifest,indent=2))
    print('B9A6 Mac files ready:',p,mp)


def compare(root,check):
    anchor,ah=anchor_sources(root)
    transfers=root/TRANSFER
    mf=read_json(transfers/'B9A6_mac_manifest.json')
    tf=transfers/'B9A6_kinematic_fields.npz'
    if mf.get('anchor_sha256')!=ah or mf.get('transfer_sha256')!=sha(tf):
        raise RuntimeError('Mac kinematic fields changed since export')
    xf=root/EXPORTS/'B9A6_kinematic_RF.npz'
    meta=read_json(root/EXPORTS/'B9A6_windows_summary.json')
    if meta.get('anchor_sha256')!=ah or meta.get('transfer_sha256')!=sha(tf) or meta.get('output_sha256')!=sha(xf):
        raise RuntimeError('Windows kinematic reference mismatches Mac files/anchor')
    exp=npz(xf)
    f=npz(root/'B9A2_measured_FE/B9A2_measured_FE.npz')
    a3=npz(root/'B9A3_exports/B9A3_NO_reference.npz')
    modes_csv=root/'results/65AM_B9_coupled_feasibility/B9A4_field_and_force/B9A4_mode_force_metrics.csv'
    if not modes_csv.is_file():raise FileNotFoundError('Missing B9A4 mode comparison '+str(modes_csv))
    mdf=pd.read_csv(modes_csv)
    need={'interface','mode','B5_U_S_onepoint_N','Abaqus_U_S_onepoint_N'}
    if not need.issubset(mdf.columns):raise RuntimeError('Missing force-mode CSV columns: '+str(need-set(mdf.columns)))
    FE=f['measured_FE_g'];old=anchor['old_NO_g'];scale=anchor['force_scale']
    true=a3['reference_NO_g'];mask=a3['reference_mask'].astype(bool)
    if any(a.shape!=(26,) for a in (FE,old,scale,true,mask)) or np.any(scale<=0):
        raise RuntimeError('Invalid 26 force components')
    if rms((FE+old)/scale-anchor['normalized_residual'])>1e-6:
        raise RuntimeError('Measured FEM residual no longer reproduces 65V')
    expected=np.zeros(26,bool)
    for iface,s in mode_slices():
        if iface in sum([list(v) for v in INTERFACES.values()],[]):expected[s]=True
    if not np.array_equal(mask,expected):raise RuntimeError('Incorrect independent NO reference mask')
    variants={v:old.copy() for v in ('replay_FEM','replay_B5','true_NO','B5_onepoint','FEM_onepoint')}
    variants['true_NO'][mask]=true[mask]
    for reg,ifaces in INTERFACES.items():
        for tag in ('FEM','B5'):
            key=reg+'_'+tag+'_g'
            if key not in exp:raise RuntimeError('Windows export missing '+key)
            v=np.asarray(exp[key],float)
            width=sum(COUNTS[IFACES.index(x)] for x in ifaces)
            if v.shape!=(width,) or not np.isfinite(v).all():raise RuntimeError('B9A6 malformed '+key)
            k=0
            for n,sl in mode_slices():
                if n in ifaces:
                    variants['replay_'+tag][sl]=v[k:k+sl.stop-sl.start]
                    k+=sl.stop-sl.start
    for name,sl in mode_slices():
        if name not in [n for pair in INTERFACES.values() for n in pair]:continue
        data=mdf[mdf.interface==name].sort_values('mode')
        if data['mode'].astype(int).tolist()!=list(range(1,sl.stop-sl.start+1)):
            raise RuntimeError('B9A4 onepoint mode rows missing for '+name)
        variants['B5_onepoint'][sl]=data.B5_U_S_onepoint_N.to_numpy(float)
        variants['FEM_onepoint'][sl]=data.Abaqus_U_S_onepoint_N.to_numpy(float)
    controls=[];rows=[];summary={}
    for reg,ifaces in INTERFACES.items():
        mask_reg=np.zeros(26,bool)
        for name,sl in mode_slices():
            if name in ifaces:mask_reg[sl]=True
        gap=(variants['replay_FEM']-variants['true_NO'])/scale
        # This is a control: if prescribed true U does not reproduce RF, cannot interpret B5.
        control_rms=rms(gap[mask_reg]);control_rel=relative(variants['replay_FEM'][mask_reg],variants['true_NO'][mask_reg])
        controls.append({'region':reg,'FEM_kinematic_replay_vs_original_Abaqus_scaled_RMS':control_rms,
                          'FEM_kinematic_replay_vs_original_Abaqus_rel_pct':control_rel,
                          'control_within_0p01_scaled_RMS':bool(control_rms<=.01)})
    for name,sl in mode_slices():
        if name not in [n for pair in INTERFACES.values() for n in pair]:continue
        for j in range(sl.start,sl.stop):
            def val(key):return float(variants[key][j])
            rows.append({'interface':name,'mode':j-sl.start+1,'global_mode':j+1,
                'scale_N':float(scale[j]),'FE_RF_N':float(FE[j]),'original_NO_N':float(old[j]),
                'true_Abaqus_NO_RF_N':val('true_NO'),
                'FEM_kinematic_replay_RF_N':val('replay_FEM'),
                'B5_kinematic_replay_RF_N':val('replay_B5'),
                'B5_onepoint_N':val('B5_onepoint'),
                'replay_FEM_minus_true_scaled':(val('replay_FEM')-val('true_NO'))/float(scale[j]),
                'B5_kinematic_minus_true_scaled':(val('replay_B5')-val('true_NO'))/float(scale[j]),
                'B5_kinematic_minus_B5_onepoint_scaled':(val('replay_B5')-val('B5_onepoint'))/float(scale[j]),
                'true_NO_normalized_residual':(float(FE[j])+val('true_NO'))/float(scale[j]),
                'B5_replay_normalized_residual':(float(FE[j])+val('replay_B5'))/float(scale[j])})
    series={label:(FE+g)/scale for label,g in variants.items()}
    main=series['replay_B5']
    denom=float(np.sum(main**2))
    for row in rows:
        j=row['global_mode']-1
        row['B5_replay_squared_residual_share_pct']=float(100*main[j]**2/max(denom,1e-25))
    for k,v in series.items():summary[k+'_saved_q_RMS']=rms(v)
    m=np.asarray(mask,bool)
    summary.update({'B5_kinematic_vs_true_scaled_RMS_inner':rms((variants['replay_B5']-true)[m]/scale[m]),
                    'FEM_kinematic_vs_true_scaled_RMS_inner':rms((variants['replay_FEM']-true)[m]/scale[m]),
                    'B5_onepoint_vs_B5_kinematic_scaled_RMS_inner':rms((variants['B5_onepoint']-variants['replay_B5'])[m]/scale[m]),
                    'reference_control_pass_all':bool(all(v['control_within_0p01_scaled_RMS'] for v in controls)),
                    'no_live_Broyden_authorization':True})
    print('B9A6 MODE CONTROL:',json.dumps(controls,indent=2))
    print('B9A6 COUPLED RMS (all saved old q, NOT new solves):',json.dumps(summary,indent=2))
    if check:
        print('B9A6 COMPARE CHECK ONLY; no result files created')
        return
    output=root/OUT
    if not output.is_dir():raise RuntimeError('Create folder using Finder: '+str(output))
    expected_paths=[output/x for x in ('B9A6_reference_controls.csv','B9A6_force_modes.csv','B9A6_summary.json','B9A6_force_comparison.png')]
    if any(x.exists() for x in expected_paths):raise RuntimeError('B9A6 results already exist; not overwriting')
    pd.DataFrame(controls).to_csv(expected_paths[0],index=False)
    df=pd.DataFrame(rows).sort_values('B5_replay_squared_residual_share_pct',ascending=False)
    df.to_csv(expected_paths[1],index=False)
    result={'step':'65AM-B9A6','anchor_sha256':ah,'method':'ABAQUS all-node kinematic replay of frozen B5 and actual FEM U on identical C3D8R geometry',
            'RMS':summary,'reference_controls':controls,
            'limits':['Abaqus kinematic replay is an offline force oracle requiring full-field prescribed displacements',
                      'Not deployable as a NO force head without further surrogate modeling',
                      'Must match FEM replay control before interpreting B5 replay reactions',
                      'No new coupled Broyden solve; 65V remains unchanged',
                      'Exact-minus-onepoint includes all discretization/constitutive/stabilization differences; not exclusively hourglass']}
    expected_paths[2].write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5))
    q=df.head(12).copy();xx=np.arange(len(q));width=.27
    for shift,key,label in ((-width,'B5_onepoint_N','B5 one-point'),(0,'B5_kinematic_replay_RF_N','B5 Abaqus replay'),(width,'true_Abaqus_NO_RF_N','True NO Abaqus')):
        dif=np.abs((q[key].to_numpy(float)-q['true_Abaqus_NO_RF_N'].to_numpy(float))/q['scale_N'].to_numpy(float))
        ax.bar(xx+shift,dif,width,label=label)
    ax.set_xticks(xx);ax.set_xticklabels([f'{n}/m{i}' for n,i in zip(q.interface,q['mode'])],rotation=35,ha='right')
    ax.set_title('B9A6 | PCA force error relative to same-BC Abaqus NO benchmark')
    ax.set_ylabel('Absolute difference / 65V force scale')
    ax.legend();fig.tight_layout();fig.savefig(expected_paths[3],dpi=190);plt.close(fig)
    print('B9A6 SAVED:',output)
    if not summary['reference_control_pass_all']:
        print('CONTROL NOT PASSED: do not interpret B5 as accurate force replay without investigation')


def selftest():
    seq=list(mode_slices())
    assert len(seq)==6 and seq[0][1]==slice(0,4) and seq[-1][1]==slice(22,26)
    a=np.array([3.,4.]);assert abs(rms(a)-np.sqrt(12.5))<1e-12
    assert relative(np.array([1.,2.]),np.array([1.,2.]))==0
    assert np.array_equal(np.r_[np.ones(4,bool),np.zeros(22,bool)],np.array([i<4 for i in range(26)]))
    print('B9A6 MAC SELFTEST PASS: mode alignment / RMS / relative errors')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('prepare','compare'))
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--selftest',action='store_true')
    args=p.parse_args()
    if args.selftest:selftest();return
    if args.stage is None:p.error('--stage required unless --selftest')
    root=Path(args.project).expanduser().resolve()
    if args.stage=='prepare':prepare(root,args.check_only)
    else:compare(root,args.check_only)

if __name__=='__main__':main()
