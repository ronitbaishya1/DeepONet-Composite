#!/usr/bin/env python3
"""B9A11G: frozen B5 at 65V interface and actual Abaqus-RF force diagnostic.

Use only trusted project checkpoints/exports. NO training, Abaqus runs,
checkpoint updates or edits to existing 65V/65W results. Stage --check,
--preview or --rf. Preview is an UNCOUPLED fixed-65V-q diagnostic.

Mac: python3 65AM_B9A11G_fast_track.py --project . --stage check
     python3 65AM_B9A11G_fast_track.py --project . --stage preview
     python3 65AM_B9A11G_fast_track.py --project . --stage rf
"""
import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REGIONS = ('inner_left', 'inner_right')
OWNERS = {'inner_left':'NO_L', 'inner_right':'NO_R'}
FACES = {'inner_left':('left_inner','center_left'),
         'inner_right':('center_right','right_inner')}
COMP = ('11','22','33','12','13','23')
ANCHOR = Path('B9_coupled_anchor/B9_65V_anchor.npz')
FEMDIR = Path('results/hybrid_online_7region_65W_middle_validation')
OUT = Path('results/65AM_B9A11G_fast_track')
MAT = np.array([46500.,12500.,4750.],dtype=np.float64)


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def save_csv(path,rows):
    if not rows:raise RuntimeError('No rows for '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def relative(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise RuntimeError('Invalid comparison arrays')
    return 100*float(np.linalg.norm((a-b).ravel()))/max(float(np.linalg.norm(b.ravel())),1e-12)


def metrics(p,ref):
    return {'rel_l2_pct':relative(p,ref),
            'rmse':float(np.sqrt(np.mean((np.asarray(p)-np.asarray(ref))**2)))}


def match(ref_xyz,source_xyz,limit=2e-5):
    """Match ALL reference query points uniquely to source local mesh positions."""
    from scipy.spatial import cKDTree
    q=np.asarray(ref_xyz,dtype=np.float64)
    x=np.asarray(source_xyz,dtype=np.float64)
    if q.ndim!=2 or x.ndim!=2 or q.shape[1]!=3 or x.shape[1]!=3:
        raise RuntimeError('Coordinate shape mismatch')
    if not len(q):raise RuntimeError('Empty reference points')
    dist,ids=cKDTree(x).query(q,k=1)
    if np.max(dist)>limit or len(set(ids.tolist()))!=len(ids):
        raise RuntimeError('Failed unique coordinate match: max distance %.9g mm; %d/%d unique'%(float(np.max(dist)),len(set(ids.tolist())),len(ids)))
    return ids


def source(root):
    """Exact frozen B6 candidate: centered_grad in BOTH inner regions."""
    for path in [ANCHOR, FEMDIR/'assembled_hybrid_nodes.csv',
                 FEMDIR/'assembled_hybrid_ip.csv',
                 Path('results/65AM_B6_independent/D_independent_evaluation/B6_summary.json'),
                 Path('results/65AM_B6_independent/A_plan/65AM_B6_plan.json')]:
        if not (root/path).is_file():raise FileNotFoundError(str(root/path))
    summary=json.loads((root/'results/65AM_B6_independent/D_independent_evaluation/B6_summary.json').read_text())
    if summary.get('not_full_seven_region_online_coupling') is not True or summary.get('frozen_candidate')!='B5 centered_grad both regions':
        raise RuntimeError('Unexpected B6 provenance: this is offline B5 only')
    gates=summary.get('decisions',[])
    if len(gates)!=2 or {r['region'] for r in gates}!=set(REGIONS) or not all(r['pass_gate'] for r in gates):
        raise RuntimeError('B6 pre-specified independent test gate failed')
    plan=json.loads((root/'results/65AM_B6_independent/A_plan/65AM_B6_plan.json').read_text())
    if plan.get('candidate')!='centered_grad fixed both regions before B6':raise RuntimeError('Candidate selection changed')
    for reg in REGIONS:
        path=root/'results/65AM_B5_coeff_normalized/centered_grad'/reg/'best_model.pt'
        if not path.is_file() or hash_file(path)!=plan['frozen_checkpoint_sha256'][reg]:
            raise RuntimeError('Frozen B5 model SHA mismatch: '+reg)
    import torch
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    B6=importlib.import_module('65AM_B6_mac')
    C=importlib.import_module('65AM_B5_common')
    L=importlib.import_module('65AM_B9A10L_torch_force')
    H=importlib.import_module('65AM_B9A10H_mac_weakform_validation')
    with np.load(root/ANCHOR,allow_pickle=False) as z:
        must=['material','force_scale','old_NO_g','FE_g_inferred','normalized_residual']
        must+=['c_'+face for faces in FACES.values() for face in faces]
        must+=['pca_'+name+'_'+face for name in ('mean','basis','coordinates') for faces in FACES.values() for face in faces]
        missing=set(must)-set(z.files)
        if missing:raise RuntimeError('B9 anchor missing '+str(missing))
        anc={k:np.array(z[k],copy=True) for k in z.files}
    if not np.allclose(anc['material'].reshape(-1),MAT,rtol=0,atol=1e-7):
        raise RuntimeError('65V anchor material not middle case')
    r=np.asarray(anc['normalized_residual'],float).ravel()
    if r.shape!=(26,) or not np.all(np.isfinite(r)) or abs(np.sqrt(np.mean(r*r))-.019522576349400432)>1e-6:
        raise RuntimeError('Wrong original 65V anchor state')
    models={}
    dev=torch.device('cpu')
    for reg in REGIONS:
        d,mu,sd=C.prepare(root,reg)
        mdl,ck=C.load_model(root,'centered','grad',reg,d,mu,sd,dev)
        ds=d['ds']
        if np.asarray(ds['coordinates']).shape!=(1008,3) or np.asarray(ds['element_node_indices']).shape!=(728,8):
            raise RuntimeError('B5 geometry mismatch '+reg)
        models[reg]=(d,mdl,ck)
    return C,B6,L,H,anc,models


def force_op(root,d,L):
    p=root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_calibrated_coefficient.json'
    if not p.is_file():raise FileNotFoundError(p)
    beta=float(json.loads(p.read_text())['beta'])
    if not 0<beta<0.05:raise RuntimeError('Invalid frozen beta')
    ds=d['ds'];xyz=np.asarray(ds['coordinates'],float)
    con=np.asarray(ds['element_node_indices'],np.int64)
    if np.min(con)<0 or np.max(con)>=len(xyz):raise RuntimeError('Invalid connectivity')
    return L.ConstitutiveC3D8R(np.arange(1,len(xyz)+1),xyz,con+1,beta,
                               root/'65AM_B9A10K_hourglass_operator.py'),beta


def anchored_prediction(root,reg,d,net,anc,B6):
    import torch
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    dev=torch.device('cpu')
    ds=d['ds'];ctx=B6.b.CO.interface_context(root,ds,dev)
    if ctx['branch_dim']!=12:raise RuntimeError('Expected 3 material + 9 interface coefficients')
    branch=np.zeros((1,12),dtype=np.float32);branch[0,:3]=MAT
    for side,face in zip(('left','right'),FACES[reg]):
        datum=ctx[side]
        sl=ctx[side+'_slice']
        xx=np.asarray(datum['coordinates'],float)
        aa=np.asarray(anc['pca_coordinates_'+face],float)
        if xx.shape!=aa.shape or np.max(abs(xx-aa))>2e-5:raise RuntimeError('Windows/Mac interface node mismatch: '+face)
        for nm in ('mean','basis'):
            m=np.asarray(datum[nm],float)
            w=np.asarray(anc['pca_'+nm+'_'+face],float)
            if m.shape!=w.shape or np.max(abs(m-w))>1e-4:
                raise RuntimeError('Windows/Mac interface PCA '+nm+' changed for '+face)
        coeff=np.asarray(anc['c_'+face],float).ravel()
        if len(coeff)!=sl.stop-sl.start:raise RuntimeError('Physical PCA coefficient shape mismatch '+face)
        branch[0,sl]=coeff.astype(np.float32)
    n=len(ds['coordinates'])
    with torch.no_grad():
        lift=apply_hard_compatibility_7region(torch.zeros((1,n,3),dtype=torch.float32),
                 torch.as_tensor(branch,dtype=torch.float32),ctx).cpu().numpy()
    for side,face in zip(('left','right'),FACES[reg]):
        ids=ctx[side+'_local_indices'].cpu().numpy()
        pca=np.asarray(anc['pca_mean_'+face],float).reshape(-1)+np.asarray(anc['pca_basis_'+face],float)@np.asarray(anc['c_'+face],float).ravel()
        gap=np.max(abs(lift[0,ids,:].reshape(-1)-pca))
        if gap>1e-4:raise RuntimeError('65V physical interface lift mismatch: '+face+' gap='+str(gap))
    prediction=B6.predict_b5(d,net,branch,lift,dev)[0]
    fixed=np.unique(np.r_[ctx['left_local_indices'].cpu().numpy(),ctx['right_local_indices'].cpu().numpy()])
    if np.max(abs(prediction[fixed]-lift[0,fixed]))>1e-5:raise RuntimeError('Predicted U differs from prescribed interface')
    return prediction,branch,lift


def score_preview(root,anc,mods,C,B6,L,tag):
    import pandas as pd
    import torch
    out=root/OUT/tag/'preview'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Protected preview output: '+str(out))
    nodes=pd.read_csv(root/FEMDIR/'assembled_hybrid_nodes.csv')
    ip=pd.read_csv(root/FEMDIR/'assembled_hybrid_ip.csv')
    if len(nodes)!=4592 or len(ip)!=3640:raise RuntimeError('65W mesh size changed')
    needs_node={'Owner','X','Y','Z'}|{'FEM_U'+str(k) for k in (1,2,3)}|{'Hybrid_U'+str(k) for k in (1,2,3)}
    needs_ip={'Owner','X','Y','Z'}|{'FEM_'+typ+c for typ in ('LE','S') for c in COMP}|{'Hybrid_'+typ+c for typ in ('LE','S') for c in COMP}
    if not needs_node.issubset(nodes.columns) or not needs_ip.issubset(ip.columns):raise RuntimeError('65W CSV lacks expected columns')
    if set(nodes.Owner)!=set(('NO_OL','FE_L','NO_L','FE_C','NO_R','FE_R','NO_OR')):
        raise RuntimeError('Wrong seven-region ownership')
    xyz=nodes[['X','Y','Z']].to_numpy(float)
    xyz_ip=ip[['X','Y','Z']].to_numpy(float)
    nref=nodes[['FEM_U'+str(k) for k in (1,2,3)]].to_numpy(float)
    nbase=nodes[['Hybrid_U'+str(k) for k in (1,2,3)]].to_numpy(float)
    nnew=nbase.copy()
    ref_ip={typ:ip[['FEM_'+typ+c for c in COMP]].to_numpy(float) for typ in ('LE','S')}
    base_ip={typ:ip[['Hybrid_'+typ+c for c in COMP]].to_numpy(float) for typ in ('LE','S')}
    new_ip={typ:base_ip[typ].copy() for typ in ('LE','S')}
    output=[];files=[]
    pending=[]
    for reg,(d,net,ck) in mods.items():
        own=OWNERS[reg];ds=d['ds']
        U,branch,lift=anchored_prediction(root,reg,d,net,anc,B6)
        proxy,beta=force_op(root,d,L)
        with torch.no_grad():
            ff=proxy.fields(torch.as_tensor(U,dtype=torch.float64),torch.as_tensor(MAT,dtype=torch.float64))
        stress=ff['S_global'].cpu().numpy()
        S=np.stack((stress[:,0,0],stress[:,1,1],stress[:,2,2],stress[:,0,1],stress[:,0,2],stress[:,1,2]),axis=-1)
        LE=C.b.CO.log_material_strain_eng(U,ds['element_node_indices'],ds['B_center'])
        nids=np.flatnonzero(nodes.Owner.to_numpy()==own)
        pids=np.flatnonzero(ip.Owner.to_numpy()==own)
        ni=match(xyz[nids],np.asarray(ds['coordinates'],float))
        pi=match(xyz_ip[pids],np.asarray(ds['ip_coordinates'],float))
        nnew[nids]=U[ni]
        new_ip['LE'][pids]=LE[pi]
        new_ip['S'][pids]=S[pi]
        pending.append((reg,U,LE,S,branch))
    for own in ('NO_L','NO_R'):
        for typ,base,pred,ref in [('U',nbase,nnew,nref),('LE',base_ip['LE'],new_ip['LE'],ref_ip['LE']),('S',base_ip['S'],new_ip['S'],ref_ip['S'])]:
            sel=(nodes.Owner.to_numpy()==own) if typ=='U' else (ip.Owner.to_numpy()==own)
            cc=('1','2','3') if typ=='U' else COMP
            for j,c in enumerate(cc):
                base_m=metrics(base[sel,j],ref[sel,j]);new_m=metrics(pred[sel,j],ref[sel,j])
                output.append(dict(region=own,component=typ+c,baseline_rel_l2_pct=base_m['rel_l2_pct'],
                    B5_fixedq_rel_l2_pct=new_m['rel_l2_pct'],baseline_rmse=base_m['rmse'],B5_fixedq_rmse=new_m['rmse'],
                    improvement_in_percentage_points=base_m['rel_l2_pct']-new_m['rel_l2_pct']))
    global_metrics={}
    for typ,base,pred,ref in [('U',nbase,nnew,nref),('LE',base_ip['LE'],new_ip['LE'],ref_ip['LE']),('S',base_ip['S'],new_ip['S'],ref_ip['S'])]:
        global_metrics[typ]={'original_v4':metrics(base,ref),'B5_frozen_q':metrics(pred,ref)}
        for j,c in enumerate(('1','2','3') if typ=='U' else COMP):
            global_metrics[typ+c]={'original_v4':metrics(base[:,j],ref[:,j]),'B5_frozen_q':metrics(pred[:,j],ref[:,j])}
    out.mkdir(parents=True)
    save_csv(out/'B9A11G_fixedq_regional_errors.csv',output)
    for reg,U,LE,S,branch in pending:
        np.savez_compressed(out/('B9A11G_'+reg+'_candidate_fields.npz'),U=U,LE=LE,S=S,branch=branch,material=MAT)
    review=dict(step='B9A11G-preview',uses_real_B5_centered_grad=True,source='frozen 65V q plus original 65W/V4 baseline',
        no_abaqus_jobs=True,not_re_equilibrated=True,not_full_B5_coupled=True,
        proxy_stress_is_approximate=True,original_B5_untouched=True,
        original_65V_residual_rms=float(np.sqrt(np.mean(np.asarray(anc['normalized_residual'])**2))),
        model='B5 original frozen centered_grad both inner segments; 65Y not mixed',
        global_metrics=global_metrics,output_files=['B9A11G_fixedq_regional_errors.csv']+[f'B9A11G_{r}_candidate_fields.npz' for r in REGIONS],
        interpretation='Diagnostic preview only. B5 stress uses candidate L+K C3D8R proxy, NOT Abaqus stresses; 65V force equilibrium does not transfer.')
    (out/'B9A11G_fixedq_summary.json').write_text(json.dumps(review,indent=2))
    print('PREVIEW complete. Relative L2 %, original -> frozen-q B5:')
    for component in ('U','U2','LE22','S22','S33'):
        xx=global_metrics[component]
        print('%-5s %.5f -> %.5f'%(component,xx['original_v4']['rel_l2_pct'],xx['B5_frozen_q']['rel_l2_pct']))
    print('Saved',out)


def score_rf(root,anc,mods,C,B6,L,H,tag):
    """B9A9 recorded Abaqus REACTION FORCES, not surrogate labels."""
    import torch
    src=root/'B9A10H_exports/B9A10H_independent_fields.npz'
    manifest=root/'B9A10H_exports/B9A10H_windows_summary.json'
    if not src.is_file() or not manifest.is_file():raise FileNotFoundError('B9A10H saved 24-case Abaqus RF export missing')
    raw=json.loads(manifest.read_text())
    if hash_file(src)!=raw.get('output_sha256'):raise RuntimeError('B9A10H RF export SHA mismatch')
    out=root/OUT/tag/'rf'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Protected RF output '+str(out))
    per=[];mode=[]
    with np.load(src,allow_pickle=False) as z:
        for reg,(d,net,ck) in mods.items():
            ds=d['ds']; force,beta=force_op(root,d,L)
            for k in range(1,13):
                casename='B9A9_'+('L' if reg=='inner_left' else 'R')+('%02d'%k)
                case=H.make_case(z,casename)
                perm=match(np.asarray(ds['coordinates'],float),np.asarray(case['X'],float))
                refu=np.asarray(case['U'],np.float64)[perm][None]
                mats=case_material_from_plan(root,reg,k)
                branch,lift,fixed,checks=B6.project_boundary(root,ds,refu,np.asarray(mats,dtype=np.float32)[None],torch.device('cpu'))
                maxpca=max((float(v) for key,v in checks.items() if key.endswith('_pca_max_mm')),default=0.)
                if maxpca>0.001:raise RuntimeError(casename+': PCA projection exceeds .001 mm; no valid oracle-boundary comparison')
                u=B6.predict_b5(d,net,branch,lift,torch.device('cpu'))[0]
                original=np.asarray(case['RF'],np.float64)[perm]
                mask=np.zeros(len(u),bool);mask[fixed]=True
                mat=torch.as_tensor(branch[0,:3].astype(np.float64),dtype=torch.float64)
                with torch.no_grad():
                    f_pred=force(torch.as_tensor(u,dtype=torch.float64),mat).cpu().numpy()
                    f_trueu=force(torch.as_tensor(refu[0],dtype=torch.float64),mat).cpu().numpy()
                den=max(np.linalg.norm(original[mask]),1e-12)
                face_comps=[];pca_comps=[]
                # H.pca_force expects node ordering from the Abaqus H export.
                f_pred_case=np.zeros_like(f_pred);f_pred_case[perm]=f_pred
                f_trueu_case=np.zeros_like(f_trueu);f_trueu_case[perm]=f_trueu
                for face in FACES[reg]:
                    mp,scale=H.pca_force(case,f_pred_case,anc,face)
                    mf,_=H.pca_force(case,f_trueu_case,anc,face)
                    mr,_=H.pca_force(case,np.asarray(case['RF'],float),anc,face)
                    for i in range(len(scale)):
                        mode.append(dict(case_id=casename,region=reg,interface=face,mode=i+1,
                            abaqus_RF_mode_N=float(mr[i]),B5_proxy_mode_N=float(mp[i]),
                            FEM_U_proxy_mode_N=float(mf[i]),
                            normalized_B5_minus_RF=float((mp[i]-mr[i])/scale[i]),
                            normalized_proxy_floor_minus_RF=float((mf[i]-mr[i])/scale[i])))
                        face_comps.append((float(mp[i]-mr[i])/scale[i])**2)
                        pca_comps.append((float(mf[i]-mr[i])/scale[i])**2)
                per.append(dict(case_id=casename,region=reg,material_E1=float(mat[0]),material_E2=float(mat[1]),material_G12=float(mat[2]),
                    oracle_pca_projection_max_mm=maxpca,
                    B5_U_relative_pct=relative(u,refu[0]),
                    B5_proxy_RF_boundary_error_pct=100*np.linalg.norm((f_pred-original)[mask])/den,
                    proxy_floor_RF_boundary_error_pct=100*np.linalg.norm((f_trueu-original)[mask])/den,
                    B5_generalized_force_RMS=float(np.sqrt(np.mean(face_comps))),
                    proxy_reference_generalized_force_RMS=float(np.sqrt(np.mean(pca_comps)))))
                print('RF',casename,'normalized modes',round(per[-1]['B5_generalized_force_RMS'],6),
                      'proxy floor',round(per[-1]['proxy_reference_generalized_force_RMS'],6),flush=True)
    out.mkdir(parents=True)
    save_csv(out/'B9A11G_true_Abaqus_RF_cases.csv',per)
    save_csv(out/'B9A11G_true_Abaqus_RF_modes.csv',mode)
    summary={reg:{'cases':sum(t['region']==reg for t in per),
        'B5_force_RMS_mean':float(np.mean([t['B5_generalized_force_RMS'] for t in per if t['region']==reg])),
        'B5_force_RMS_worst':float(np.max([t['B5_generalized_force_RMS'] for t in per if t['region']==reg])),
        'FEM_U_proxy_floor_mean':float(np.mean([t['proxy_reference_generalized_force_RMS'] for t in per if t['region']==reg])),
        'Abaqus_boundary_force_error_pct_mean':float(np.mean([t['B5_proxy_RF_boundary_error_pct'] for t in per if t['region']==reg]))}
        for reg in REGIONS}
    review=dict(step='B9A11G-RF',method='frozen B5-centered_grad + frozen B9A10L/K force approximation',
         physical_truth='24 previously generated Abaqus reaction-force cases from B9A10H (B9A9)',
         forces_are_actual_Abaqus_RF=True,calibrated_hourglass_proxy_still_approximate=True,
         training_performed=False,not_online_coupled=True,
         expected_reaction_sign='proxy internal force versus Abaqus RF: inspect whether their sign convention matches',
         result=summary,
         use_as_model_selection_holdout=False,
         gate_suggestion='Do not run new expensive coupled B5 solve unless predicted normalized force errors are compatible with the 0.02 65V target AND proxy floor and sign/frame audit are acceptable.')
    (out/'B9A11G_true_Abaqus_RF_summary.json').write_text(json.dumps(review,indent=2))
    print('Saved',out,'\nResult:',json.dumps(summary,indent=2))


def case_material_from_plan(root,region,case_number):
    """B9A9 plan stores x_inner_left/right: 12 x [E1,E2,G12,9 PCA modes]."""
    p=root/'B9A9_transfer_to_windows/B9A9_case_plan.npz'
    if not p.is_file():raise FileNotFoundError('B9A9 material plan missing '+str(p))
    with np.load(p,allow_pickle=False) as z:
        key='x_'+region
        if key not in z or 'case_ids' not in z:
            raise RuntimeError('Unexpected B9A9 plan keys: '+str(z.files))
        i=case_number-1
        ids=np.asarray(z['case_ids']).astype(str).tolist()
        x=np.asarray(z[key],float)
        if x.shape!=(12,12) or ids[i]!='B9A9_%02d'%case_number:
            raise RuntimeError('Invalid B9A9 case plan data for '+region)
        return x[i,:3]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--stage',choices=('selftest','check','preview','rf','all'),required=True)
    ap.add_argument('--tag',default='run01')
    args=ap.parse_args()
    if not args.tag.replace('_','').replace('-','').isalnum():ap.error('Use alphanumeric run tag')
    if args.stage=='selftest':
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        assert np.array_equal(match(x,x),np.arange(3))
        assert relative(x,x)==0
        print('B9A11G SELFTEST PASSED')
        return
    root=Path(args.project).expanduser().resolve()
    C,B6,L,H,anc,mods=source(root)
    if args.stage=='check':
        print('B9A11G CHECK passed: B6 gate, B5 checkpoint hashes, 65V anchor, and original scripts.')
    elif args.stage=='preview':
        score_preview(root,anc,mods,C,B6,L,args.tag)
    elif args.stage=='rf':
        score_rf(root,anc,mods,C,B6,L,H,args.tag)
    else:
        score_preview(root,anc,mods,C,B6,L,args.tag)
        score_rf(root,anc,mods,C,B6,L,H,args.tag)

if __name__=='__main__':main()
