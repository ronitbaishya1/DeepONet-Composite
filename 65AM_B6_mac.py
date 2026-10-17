#!/usr/bin/env python3
"""65AM-B6: locked new-material plan, Abaqus export packaging, independent frozen-B5 evaluation.

Mac only. On Windows use B6_W scripts with Abaqus 2025. Never fit any models
from B6 inputs or choose between models based on the final B6 test. B6 B5
candidate was fixed to centered_grad for both regions BEFORE any new ODB.

FEM oracle boundary inputs are explicitly allowed in this *offline* test.
FEM interior displacement is only the reference label, never a network input.
"""
import argparse, csv, hashlib, json, os, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

c = importlib.import_module('65AM_B5_common')
b = c.b
ROOT = '65AM_B6_independent'
N_CASES = 6
SEED = 65012027
CANDIDATE = ('centered', 'grad')
R = ('inner_left','inner_right')
COLS = ('E1_MPa','E2_MPa','G12_MPa')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for z in iter(lambda:f.read(1024*1024),b''): h.update(z)
    return h.hexdigest()


def safe_dir(path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Existing results protected; do not overwrite:\n'+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path


def base(root):return root/'results'/ROOT

def manifest_path(root):return base(root)/'A_plan'/'65AM_B6_cases.csv'

def refs_path(root):return base(root)/'C_packages'


def json_save(path,value):
    path.write_text(json.dumps(value,indent=2))


def read_manifest(root):
    p=manifest_path(root)
    if not p.is_file():raise FileNotFoundError('Run --stage plan first: '+str(p))
    df=pd.read_csv(p)
    if len(df)!=N_CASES or not all(x in df for x in ('case_id',)+COLS):
        raise RuntimeError('B6 manifest missing fields or unexpected number of cases')
    names=[f'B6_new_{i:02d}' for i in range(1,N_CASES+1)]
    if df.case_id.astype(str).tolist()!=names:raise RuntimeError('Invalid B6 case list/order')
    mats=df[list(COLS)].to_numpy(np.float64)
    if np.min(mats)<=0 or not np.isfinite(mats).all():raise RuntimeError('Invalid B6 material parameters')
    return df


def get_previously_seen(root,ds):
    old=np.asarray(ds['branch_inputs'][:,:3],np.float64)
    prior=root/'results'/'65AJ_architecture_verification'/'D_new_material_manifest'/'65AJ_D_new_cases.csv'
    ext=root/'results'/'65AJ_architecture_verification'/'E_external_npz'/'inner_left.npz'
    if ext.is_file() and not prior.is_file():
        raise RuntimeError('65AJ external results exist but 65AJ material manifest is missing. Restore it to exclude prior cases.')
    if prior.is_file():
        z=pd.read_csv(prior)
        if not all(k in z for k in COLS):raise RuntimeError('Invalid 65AJ manifest')
        old=np.concatenate([old,z[list(COLS)].to_numpy(np.float64)])
    return old


def load_models(root,dev):
    result={}
    for region in R:
        d,mu,sd=c.prepare(root,region)
        mdl,ck=c.load_model(root,*CANDIDATE,region,d,mu,sd,dev)
        result[region]=(d,mdl,ck)
    return result


def plan(root,check,dev):
    # Freeze selected checkpoint hashes BEFORE any new material response is seen.
    checkpoint_hashes={}
    for region in R:
        d,mu,sd=c.prepare(root,region)
        model,ck=c.load_model(root,*CANDIDATE,region,d,mu,sd,dev)
        checkpoint_hashes[region]=sha(c.ck_path(root,*CANDIDATE,region))
        del model
    old={region:b.AA.load_data(root,region) for region in R}
    for region,d in old.items():
        if d['branch_inputs'].shape[1]<5:raise RuntimeError('Expected 3 material+interface PCA entries')
    fit=np.concatenate([d['branch_inputs'][d['splits']['train'],:3] for d in old.values()])
    lo=fit.min(axis=0); hi=fit.max(axis=0)
    if np.any(hi<=lo):raise RuntimeError('No material variation')
    prev=np.concatenate([get_previously_seen(root,d) for d in old.values()])
    span=hi-lo
    rng=np.random.default_rng(SEED)
    rows=[];chosen=[]
    for attempt in range(100000):
        if len(rows)==N_CASES:break
        x=lo+rng.uniform(0.12,0.88,3)*span
        nearest=np.min(np.linalg.norm((prev-x)/span,axis=1))
        if nearest<0.055:continue
        if chosen and min(np.linalg.norm((np.asarray(chosen)-x)/span,axis=1))<0.15:continue
        chosen.append(x)
        rows.append(dict(case_id=f'B6_new_{len(rows)+1:02d}', **dict(zip(COLS,x.tolist())),
                    nearest_previous_case_scaled=float(nearest)))
    if len(rows)!=N_CASES:raise RuntimeError('Could not choose 6 distinct unseen in-range materials')
    result=base(root)/'A_plan'
    if check:
        print('B6 PLAN CHECK PASS; six materials can be sampled; no files created')
        for row in rows:print(row)
        return
    out=safe_dir(result)
    pd.DataFrame(rows).to_csv(out/'65AM_B6_cases.csv',index=False)
    json_save(out/'65AM_B6_plan.json', dict(step='65AM-B6',seed=SEED,
         case_count=N_CASES,candidate='centered_grad fixed both regions before B6',
         selected_from='prior B5 internal development',
         seen_cases_excluded='original 64J all cases, plus 65AJ manifest where available',
         training_bounds={'low':lo.tolist(),'high':hi.tolist()},
         evaluation_only=True,not_yet_solved_in_abaqus=True,
         frozen_checkpoint_sha256=checkpoint_hashes,
         predefined_gate={'full_gradient_gain_percent_min':10,
           'U_worsening_max_percent':5,'LE22_must_improve':True,
           'minimum_case_wins_per_region':5},
         note='Copy manifest to Windows. Do not select or retrain models using B6.'))
    print('B6 PLAN CREATED:',out/'65AM_B6_cases.csv')


def check(root,dev,needs_export=False):
    df=read_manifest(root)
    planfile=base(root)/'A_plan'/'65AM_B6_plan.json'
    if not planfile.is_file():raise FileNotFoundError('Missing B6 plan lock JSON')
    planrecord=json.loads(planfile.read_text())
    if planrecord.get('candidate')!='centered_grad fixed both regions before B6':
        raise RuntimeError('B6 candidate no longer matches frozen plan')
    models=load_models(root,dev)
    for region,(d,_,ck) in models.items():
        if planrecord.get('frozen_checkpoint_sha256',{}).get(region)!=sha(c.ck_path(root,*CANDIDATE,region)):
            raise RuntimeError('Frozen B5 checkpoint changed since B6 plan: '+region)
        print('B6 CHECK PASS',region,'rank',d['rank'],'model epoch',ck['selected_epoch'])
        if needs_export:
            folder=root/'B6_FEM_exports'/region
            for name in ('U_stack.npy','LE_stack.npy','S_stack.npy',
                         'coordinates.npy','ip_coordinates.npy','source_odb_paths.txt'):
                if not (folder/name).is_file(): raise FileNotFoundError(str(folder/name))
    print('B6 CHECK COMPLETE; candidate centered_grad fixed and loaded; no outputs created')
    return df,models


def package(root,dev,checkonly=False):
    df,models=check(root,dev,needs_export=True)
    names=np.asarray(df.case_id.astype(str).tolist(),dtype=str)
    mats=df[list(COLS)].to_numpy(np.float32)
    packs={}
    allpaths={}
    for region,(d,_,_) in models.items():
        ds=d['ds'];folder=root/'B6_FEM_exports'/region
        arrays={field:np.load(folder/(field+'_stack.npy'),allow_pickle=False) for field in ('U','LE','S')}
        arrays['coordinates']=np.load(folder/'coordinates.npy',allow_pickle=False)
        arrays['ip_coordinates']=np.load(folder/'ip_coordinates.npy',allow_pickle=False)
        n=len(ds['coordinates']);e=len(ds['ip_coordinates'])
        expected={'U':(len(names),n,3),'LE':(len(names),e,6),'S':(len(names),e,6),
                  'coordinates':(n,3),'ip_coordinates':(e,3)}
        for key,shape in expected.items():
            if arrays[key].shape!=shape:raise RuntimeError(f'{region} {key} shape: {arrays[key].shape}, expected {shape}')
            if not np.isfinite(arrays[key]).all():raise RuntimeError(region+' '+key+' contains invalid numbers')
        for field in ('coordinates','ip_coordinates'):
            if not np.allclose(arrays[field],ds[field],atol=1.e-5,rtol=0):
                raise RuntimeError(region+' '+field+' ordering/geometry differs from original 64J')
        paths=[x.strip() for x in (folder/'source_odb_paths.txt').read_text().splitlines() if x.strip()]
        if len(paths)!=len(names) or len(set(paths))!=len(names):
            raise RuntimeError(region+' missing distinct source ODB paths')
        for name,path in zip(names,paths):
            # Windows absolute paths: Python on Mac does not natively parse Windows slashes.
            tail=path.replace('\\','/').split('/')[-1]
            if tail.lower()!=name.lower()+'.odb':
                raise RuntimeError(f'{region}: ODB list order mismatch: expected {name}.odb; got {path}')
        allpaths[region]=paths
        arrays.update(case_ids=names,materials=mats,source_odb_paths=np.asarray(paths,dtype=str))
        packs[region]=arrays
    if allpaths['inner_left']!=allpaths['inner_right']:
        raise RuntimeError('The two region exports do not point to the same six ODB files')
    if checkonly:
        print('B6 PACKAGE CHECK PASS; both Abaqus exports align. No files created.')
        return
    out=safe_dir(refs_path(root))
    for region,arrays in packs.items():
        np.savez_compressed(out/(region+'.npz'),**arrays)
    json_save(out/'B6_package_manifest.json',dict(step='65AM-B6',phase='packaging',
          case_ids=names.tolist(),source_case_manifest_sha256=sha(manifest_path(root)),
          source_full_FEM_files_must_be_audited_manually=True,
          caveat='ODB filenames alone cannot prove correct physical material assignments or convergence.',
          note='No reference solutions fabricated.'))
    print('B6 PACKAGED verified-shape Abaqus arrays:',out)


def project_boundary(root,ds,Uref,mats,dev):
    """Only FEM nodes on FE-NO interfaces enter PCA boundary branch inputs.

    Interior FEM displacement cannot enter prediction; it remains evaluation label.
    """
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    ctx=b.CO.interface_context(root,ds,dev)
    k=len(mats);p=int(ds['branch_inputs'].shape[1])
    branch=np.zeros((k,p),dtype=np.float32)
    branch[:,:3]=mats
    diag={}
    for side_name in ('left','right'):
        s=ctx[side_name]
        if s is None:continue
        ids=ctx[side_name+'_local_indices']
        ids=ids.cpu().numpy().astype(int) if isinstance(ids,torch.Tensor) else np.asarray(ids,dtype=int)
        y=Uref[:,ids,:].reshape(k,-1).astype(np.float64)
        mean_raw=s['mean'] if 'mean' in s else s['mean_tensor'].detach().cpu().numpy()
        basis_raw=s['basis'] if 'basis' in s else s['basis_tensor'].detach().cpu().numpy()
        mean=np.asarray(mean_raw,dtype=np.float64).reshape(-1)
        basis=np.asarray(basis_raw,dtype=np.float64)
        coeff=np.linalg.lstsq(basis,(y-mean).T,rcond=None)[0].T
        sl=ctx[side_name+'_slice']
        if coeff.shape[1]!=sl.stop-sl.start:raise RuntimeError('Interface PCA coefficients mismatch')
        branch[:,sl]=coeff.astype(np.float32)
        reconstructed=mean[None,:]+coeff@basis.T
        diag[side_name+'_pca_rmse_mm']=float(np.sqrt(np.mean((reconstructed-y)**2)))
        diag[side_name+'_pca_max_mm']=float(np.max(abs(reconstructed-y)))
    # Same interface lifting function as B0. No reference interior U used.
    zeros=torch.zeros((k,len(ds['coordinates']),3),dtype=torch.float32,device=dev)
    with torch.no_grad():
        lifted=apply_hard_compatibility_7region(zeros,b.np_tens(branch,dev),ctx).cpu().numpy()
    fixed=b.boundary_indices(ctx)
    if np.any(~np.isfinite(lifted)):raise RuntimeError('Nonfinite interface lift')
    return branch,lifted.astype(np.float64),fixed,diag


def predict_b5(d,model,branch,lift,dev):
    inp=(branch.astype(np.float64)-d['branch_mean'])/d['branch_std']
    basis=b.np_tens(d['basis'],dev)
    results=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(branch),8):
            coef=model(b.np_tens(inp[i:i+8],dev))
            pred=b.np_tens(lift[i:i+8],dev)+torch.einsum('br,rnc->bnc',coef,basis)
            results.append(pred.detach().cpu().numpy())
    return np.concatenate(results).astype(np.float64)


def predict_65aa(root,d,branch,dev):
    ds=d['ds'];ck=b.CO.load_checkpoint(d['source'])
    model=b.CO.make_model(ck,dev)
    aug=dict(ds)
    aug['branch_inputs']=np.concatenate((ds['branch_inputs'],branch.astype(np.float32)),axis=0)
    aug['_root']=root
    inputs=b.AA.tensor_inputs(ck,aug,dev)
    first=len(ds['branch_inputs'])
    ids=np.arange(first,first+len(branch),dtype=int)
    pred=b.AA.predict_numpy(model,inputs,ids,correct=True)
    return pred.astype(np.float64)


def region_masks(ds):
    ip=np.asarray(ds['ip_coordinates'],dtype=np.float64)
    x=ip[:,0];d=0.60
    l=(x<=x.min()+d);r=(x>=x.max()-d)
    return dict(left=l,right=r,interior=~(l|r))


def evaluate(root,dev,checkonly=False):
    df,models=check(root,dev)
    names=np.asarray(df.case_id.astype(str).tolist(),dtype=str)
    mats=df[list(COLS)].to_numpy(np.float32)
    pack={}
    for region in R:
        path=refs_path(root)/(region+'.npz')
        if not path.is_file():raise FileNotFoundError('Run --stage package first: '+str(path))
        with np.load(path,allow_pickle=False) as z: ext={key:z[key] for key in z.files}
        for name in ('case_ids','materials','U','LE','S','coordinates','ip_coordinates','source_odb_paths'):
            if name not in ext:raise RuntimeError(region+' missing '+name)
        if not np.array_equal(names,ext['case_ids'].astype(str)) or not np.allclose(mats,ext['materials'],atol=1e-3,rtol=1e-5):
            raise RuntimeError(region+' manifest/pack mismatch')
        ds=models[region][0]['ds']
        if ext['U'].shape!=(len(names),len(ds['coordinates']),3) or ext['LE'].shape!=(len(names),len(ds['ip_coordinates']),6):
            raise RuntimeError('B6 reference shape changed')
        if not np.allclose(ext['coordinates'],ds['coordinates'],atol=1e-5,rtol=0):raise RuntimeError('B6 geometry changed')
        pack[region]=ext
    p=base(root)/'C_packages'/'B6_package_manifest.json'
    info=json.loads(p.read_text())
    if info['source_case_manifest_sha256']!=sha(manifest_path(root)):
        raise RuntimeError('B6 manifest was changed AFTER packaging')
    if checkonly:
        print('B6 EVALUATION CHECK PASS; six new FEM cases and frozen B5 models ready; no outputs created')
        return
    # Compute in memory first; protect reports before any writing.
    report=[];cases=[];zones=[];refchecks=[];gates=[]
    for region,(d,model,ck) in models.items():
        ds=d['ds'];ext=pack[region];Uref=np.asarray(ext['U'],np.float64);LEref=np.asarray(ext['LE'],np.float64)
        # Mechanical convention gate: original FEM U must reproduce exported FEM LE.
        Ltruth=np.stack([b.CO.log_material_strain_eng(U,ds['element_node_indices'],ds['B_center']) for U in Uref])
        refLE=b.CO.relative_l2(Ltruth,LEref)
        refLE22=b.CO.relative_l2(Ltruth[...,1],LEref[...,1])
        if refLE>2 or refLE22>2:
            raise RuntimeError(f'{region}: Abaqus U/LE convention mismatch: LE {refLE:.4f}%, LE22 {refLE22:.4f}%. STOP.')
        branch,lift,fixed,diagnostics=project_boundary(root,ds,Uref,mats,dev)
        if any(diag>0.001 for n,diag in diagnostics.items() if n.endswith('_pca_max_mm')):
            raise RuntimeError(region+' PCA boundary max >0.001mm; not safely represented')
        Ub5=predict_b5(d,model,branch,lift,dev)
        Uaa=predict_65aa(root,d,branch,dev)
        for name,U in [('65AA_frozen',Uaa),('B5_centered_grad',Ub5)]:
            if np.max(abs(U[:,fixed,:]-lift[:,fixed,:]))>1e-5:raise RuntimeError('Interface mismatch '+region+' '+name)
            m=b.metrics(U,Uref,ds,LEref)
            report.append(dict(region=region,model=name,n_cases=len(names),**m))
            G=b.AA.gradient_numpy(U,ds);G0=b.AA.gradient_numpy(Uref,ds)
            for i,case in enumerate(names):
                mm=b.metrics(U[i:i+1],Uref[i:i+1],ds,LEref[i:i+1])
                cases.append(dict(region=region,case_id=str(case),model=name,**mm))
            for z,mask in region_masks(ds).items():
                if not mask.any():continue
                zones.append(dict(region=region,model=name,zone=z,
                    gradient_rel_pct=b.CO.relative_l2(G[:,mask],G0[:,mask]),
                    dU2dY_rel_pct=b.CO.relative_l2(G[:,mask,1,1],G0[:,mask,1,1]),
                    elements=int(mask.sum())))
        refchecks.append(dict(region=region,truth_U_to_LE_pct=refLE,truth_U_to_LE22_pct=refLE22,
                              **diagnostics))
        ref0=next(x for x in report if x['region']==region and x['model']=='65AA_frozen')
        ref1=next(x for x in report if x['region']==region and x['model']=='B5_centered_grad')
        gain=100*(ref0['grad_rel_pct']-ref1['grad_rel_pct'])/max(ref0['grad_rel_pct'],1e-14)
        percase=pd.DataFrame([x for x in cases if x['region']==region]).pivot(index='case_id',columns='model',values='grad_rel_pct')
        wins=int(np.sum(percase['B5_centered_grad']<percase['65AA_frozen']))
        passed=(gain>=10 and ref1['U_vec_rel_pct']<=1.05*ref0['U_vec_rel_pct']
                and ref1['LE22_rel_pct']<ref0['LE22_rel_pct'] and wins>=5)
        gates.append(dict(region=region,baseline_gradient_pct=ref0['grad_rel_pct'],
                B5_gradient_pct=ref1['grad_rel_pct'],gradient_gain_pct=gain,
                baseline_LE22_pct=ref0['LE22_rel_pct'],B5_LE22_pct=ref1['LE22_rel_pct'],
                B5_case_gradient_wins=wins,total_cases=len(names),pass_gate=bool(passed)))
        print('B6',region,'gradient',round(ref0['grad_rel_pct'],3),'->',round(ref1['grad_rel_pct'],3),
              '| LE22',round(ref0['LE22_rel_pct'],3),'->',round(ref1['LE22_rel_pct'],3),
              '| pass',passed,flush=True)
    out=safe_dir(base(root)/'D_independent_evaluation')
    pd.DataFrame(report).to_csv(out/'B6_pooled_metrics.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'B6_case_metrics.csv',index=False)
    pd.DataFrame(zones).to_csv(out/'B6_zone_metrics.csv',index=False)
    pd.DataFrame(refchecks).to_csv(out/'B6_reference_checks.csv',index=False)
    pd.DataFrame(gates).to_csv(out/'B6_decisions.csv',index=False)
    json_save(out/'B6_summary.json', dict(step='65AM-B6',case_ids=names.tolist(),n_cases=len(names),
             frozen_candidate='B5 centered_grad both regions',
             source_manifest_sha256=sha(manifest_path(root)),
             original_models_unchanged=True,new_material_cases_not_retrained_on=True,
             oracle_FEM_interface_conditions=True,
             not_full_seven_region_online_coupling=True,
             independently_audit_ODB_material_and_contact_state_before_publication=True,
             decisions=gates))
    print('B6 DONE:',out)
    print('UPLOAD: B6_pooled_metrics.csv, B6_case_metrics.csv, B6_zone_metrics.csv, B6_reference_checks.csv, B6_decisions.csv, B6_summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    parser.add_argument('--stage',required=True,choices=('plan','check','package','evaluate'))
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--device',default='auto',choices=('auto','cpu','mps'))
    args=parser.parse_args()
    root=Path(args.project).expanduser().resolve()
    dev=b.CO.device_from_option(args.device)
    if args.stage=='plan':plan(root,args.check_only,dev)
    elif args.stage=='check':check(root,dev)
    elif args.stage=='package':package(root,dev,args.check_only)
    else:evaluate(root,dev,args.check_only)

if __name__=='__main__':main()
