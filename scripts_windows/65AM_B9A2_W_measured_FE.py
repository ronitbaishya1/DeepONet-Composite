#!/usr/bin/env python3
"""65AM-B9A2-A: measure *real Abaqus patch RF* at the frozen 65V state.

Place in ODBS/scripts. Run from ODBS using Abaqus Python:
  abaqus python scripts\65AM_B9A2_W_measured_FE.py --stage check
  abaqus python scripts\65AM_B9A2_W_measured_FE.py --stage export

Before export, manually create ODBS/B9A2_measured_FE in File Explorer.
No new Abaqus jobs, no source modifications. Uses 65V saved final/{left,center,right}.odb,
65V anchor, original hybrid_runtime_7region PATCH_METADATA and Windows PCA bases.
"""
from __future__ import print_function
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NAMES=('left_outer','left_inner','center_left','center_right','right_inner','right_outer')
PATCHES=('left','center','right')
EXPECTED=(4,4,5,5,4,4)
BASE=ROOT/'online_results_7region_65V_middle'/'65V_D_lm_convergence'
DEFAULT_ANCHOR=ROOT/'B9_anchor'/'B9_65V_anchor.npz'
OUT=ROOT/'B9A2_measured_FE'


def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as fp:
        while True:
            block=fp.read(1<<20)
            if not block:break
            h.update(block)
    return h.hexdigest()


def vec(a,k):
    out=np.asarray(a[k],dtype=np.float64).reshape(-1)
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite '+k)
    return out


def measure_patch(odb_path,patch_name,metadata,ifaces):
    from odbAccess import openOdb
    if not odb_path.is_file():raise FileNotFoundError('Saved 65V patch missing: '+str(odb_path))
    odb=openOdb(path=str(odb_path),readOnly=True)
    try:
        if 'Loading' not in odb.steps:raise RuntimeError(str(odb_path)+': Loading step missing')
        frame=odb.steps['Loading'].frames[-1]
        if 'RF' not in frame.fieldOutputs:raise RuntimeError(str(odb_path)+': RF not saved')
        from hybrid_runtime_7region import find_odb_instance
        inst=find_odb_instance(odb,'Composite-1')
        rf_field=frame.fieldOutputs['RF'].getSubset(region=inst)
        rf={}
        for v in rf_field.values:
            label=int(v.nodeLabel)
            if label in rf:raise RuntimeError('Duplicate RF node label %d in %s'%(label,odb_path))
            rf[label]=np.asarray(v.data[:3],dtype=np.float64)
        node_xyz={int(v.label):np.asarray(v.coordinates[:3],dtype=np.float64) for v in inst.nodes}
        if not rf:raise RuntimeError('No RF nodes in '+str(odb_path))
        observed={}
        for name,info in metadata[patch_name]['interfaces'].items():
            labels=[int(z) for z in info['node_labels']]
            if len(labels)!=len(set(labels)):raise RuntimeError('Duplicate metadata labels '+name)
            missing=[z for z in labels if z not in rf or z not in node_xyz]
            if missing:raise RuntimeError('%s: %d missing RF/coordinate nodes, first=%s'%(name,len(missing),missing[:3]))
            nodal=np.asarray([rf[z] for z in labels],dtype=np.float64)
            coords=np.asarray([node_xyz[z] for z in labels],dtype=np.float64)
            interface=ifaces[name]
            pca_coord=np.asarray(interface['coordinates'],dtype=np.float64)
            pca_basis=np.asarray(interface['basis'],dtype=np.float64)
            if coords.shape!=pca_coord.shape or pca_basis.shape[0]!=nodal.size:
                raise RuntimeError(name+' PCA shape mismatch')
            delta=float(np.max(abs(coords-pca_coord)))
            if delta>2.e-4:raise RuntimeError(name+' ODB-to-PCA coordinate/order mismatch %.8g mm'%delta)
            g=pca_basis.T@nodal.reshape(-1)
            observed[name]=dict(g=g,nodal=nodal,coordinates=coords,resultant=nodal.sum(axis=0),
                max_coord_error_mm=delta,n_nodes=len(labels),n_modes=len(g))
        return observed,dict(odb=str(odb_path),odb_sha256=sha(odb_path),
            frame_value=float(frame.frameValue),field_RF_records=len(rf),
            interfaces={name:dict(node_count=int(data['n_nodes']),modes=int(data['n_modes']),
                max_coordinate_error_mm=float(data['max_coord_error_mm']),
                RF_resultant_N=data['resultant'].tolist()) for name,data in observed.items()})
    finally:odb.close()


def run(args):
    anchor=Path(args.anchor).resolve()
    if not anchor.is_file():raise FileNotFoundError('Run earlier B9 Windows export; missing '+str(anchor))
    with np.load(str(anchor),allow_pickle=False) as z:
        old=vec(z,'old_NO_g'); inferred=vec(z,'FE_g_inferred')
        scale=vec(z,'force_scale'); saved=vec(z,'normalized_residual')
        if any(len(vec(z,'c_'+n))!=k for n,k in zip(NAMES,EXPECTED)):
            raise RuntimeError('Unexpected interface mode counts in B9 anchor')
        # Save exact Windows anchor basis integrity in the report.
        bases={n:np.asarray(z['pca_basis_'+n],float) for n in NAMES}
    if any(len(a)!=26 for a in (old,inferred,scale,saved)) or np.any(scale<=0):
        raise RuntimeError('Invalid 65V anchor forces or scaling')
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    from hybrid_runtime_7region import PATCH_METADATA,load_interface
    interfaces={n:load_interface(n) for n in NAMES}
    for n in NAMES:
        if not np.allclose(bases[n],interfaces[n]['basis'],atol=1.e-7,rtol=0):
            raise RuntimeError(n+' Windows PCA basis differs from earlier B9 anchor')
    all_obs={};audit=[]
    patch_dir=Path(args.patch_dir).resolve()
    for patch in PATCHES:
        odb=patch_dir/(patch+'.odb')
        observed,report=measure_patch(odb,patch,PATCH_METADATA,interfaces)
        overlap=set(all_obs).intersection(observed)
        if overlap:raise RuntimeError('Duplicated patch interface '+str(overlap))
        all_obs.update(observed);audit.append(report)
    if set(all_obs)!=set(NAMES):raise RuntimeError('Missing interface '+str(set(NAMES)-set(all_obs)))
    measured=np.concatenate([all_obs[n]['g'] for n in NAMES])
    diff=(measured-inferred)/scale
    recon=(measured+old)/scale
    inferred_rms=float(np.sqrt(np.mean(saved**2)))
    measured_rms=float(np.sqrt(np.mean(recon**2)))
    gap=float(np.sqrt(np.mean(diff**2)))
    print('B9A2 WINDOWS DIRECT Abaqus RF PASS | old saved RMS %.8f | measured FE+old NO RMS %.8f | FE normalized gap %.8f'%(inferred_rms,measured_rms,gap))
    rows=[];start=0
    for n,num in zip(NAMES,EXPECTED):
        sl=slice(start,start+num);start+=num
        rows.append(dict(interface=n,mode_count=num,measured_FE_norm_N=float(np.linalg.norm(measured[sl])),
            inferred_FE_norm_N=float(np.linalg.norm(inferred[sl])),
            difference_N=float(np.linalg.norm(measured[sl]-inferred[sl])),
            diff_normalized_RMS=float(np.sqrt(np.mean(diff[sl]**2))),
            old_verified_residual_RMS=float(np.sqrt(np.mean(saved[sl]**2))),
            measured_FE_plus_old_NO_RMS=float(np.sqrt(np.mean(recon[sl]**2)))))
    for row in rows:print(row['interface'],'FE gap normalized RMS %.7g'%row['diff_normalized_RMS'])
    if args.stage=='check':
        print('CHECK ONLY: no files created. Export next if no errors.')
        return
    if not OUT.is_dir():raise RuntimeError('In Windows File Explorer, create ODBS\\B9A2_measured_FE first')
    result_path=OUT/'B9A2_measured_FE.npz'
    meta_path=OUT/'B9A2_windows_summary.json'
    if result_path.exists() or meta_path.exists():raise RuntimeError('Prior B9A2 result protected; do not overwrite')
    data=dict(measured_FE_g=measured,inferred_FE_g=inferred,old_NO_g=old,
        force_scale=scale,old_verified_normalized_residual=saved)
    for n in NAMES:
        for key in ('nodal','coordinates','resultant'):
            data[key+'_'+n]=np.asarray(all_obs[n][key])
        data['g_'+n]=np.asarray(all_obs[n]['g'])
    np.savez_compressed(str(result_path),**data)
    report=dict(step='65AM-B9A2',stage='direct patch ODB RF extraction',
        anchor_sha256=sha(anchor),file_sha256=sha(result_path),
        measured_65V_old_RMS=measured_rms,saved_65V_RMS=inferred_rms,
        normalized_FE_gap_RMS=gap,absolute_sign_and_scale_NOT_modified=True,
        notes=['Real RF extracted from final saved 65V patch ODBs; not reconstructed from old residual',
               'RF projected onto the same displacement PCA basis using original patch metadata ordering',
               'No new Abaqus jobs, no corrected B5 force solution'],
        patch_audits=audit,interface_diagnostics=rows)
    with open(str(meta_path),'w') as fp:json.dump(report,fp,indent=2)
    print('SAVED:',result_path);print('SAVED:',meta_path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',required=True,choices=('check','export'))
    ap.add_argument('--patch-dir',default=str(BASE/'final'))
    ap.add_argument('--anchor',default=str(DEFAULT_ANCHOR))
    args=ap.parse_args()
    run(args)

if __name__=='__main__':main()
