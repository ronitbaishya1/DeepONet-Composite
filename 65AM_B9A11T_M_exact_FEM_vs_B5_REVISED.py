#!/usr/bin/env python3
"""B9A11T-REVISED Mac: fixed-q FEM vs B5, with noise-aware inverse Jacobian diagnostics.

From DeepONet_Composite root:
  python3 65AM_B9A11T_M_exact_FEM_vs_B5.py --stage selftest
  python3 65AM_B9A11T_M_exact_FEM_vs_B5.py --stage check --project .
  python3 65AM_B9A11T_M_exact_FEM_vs_B5.py --stage run --project . --tag run01

Requires B9A11T_exports/B9A11T_U_reference.npz and windows_summary.json,
B9_coupled_anchor, B9A11R frozen sensor design, original B5 centered_grad checkpoints.
The SAME q is prescribed for all FEM and B5 perturbations. NO full 7-region equilibrium.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REGIONS=('inner_left','inner_right')
PROPS=('E1','E2','G12')
CASES=('base','E1_M','E1_P','E2_M','E2_P','G12_M','G12_P')
OUT_BASE='65AM_B9A11T_exact_material_gradients'
DEFAULT_DESIGN='pilot01'
EPS=0.03


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def compare_vectors(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    den=max(np.linalg.norm(a),1e-14)
    ra=float(100*np.linalg.norm(b-a)/den)
    cos=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-24))
    return ra,cos


def norm_shift(a,sigma):
    return float(np.sqrt(np.mean((np.asarray(a,dtype=np.float64)/sigma)**2)))


def derive_mats(base,fraction):
    mats=[np.asarray(base,dtype=np.float64).copy()]
    for j in range(3):
        for sign in (-1,1):
            p=np.asarray(base,dtype=np.float64).copy()
            p[j]*=(1+sign*fraction)
            mats.append(p)
    return np.stack(mats)


def selftest():
    p=np.array([46500.,12500.,4750.])
    mats=derive_mats(p,.03)
    assert mats.shape==(7,3)
    assert np.isclose(mats[6,2],4892.5)
    assert np.isclose((mats[4,1]-mats[3,1])/(.06*p[1]),1)
    a=np.array([1.,2.,3.]);b=1.25*a
    r,c=compare_vectors(a,b)
    assert abs(r-25)<1e-10 and abs(c-1)<1e-10
    assert abs(norm_shift(a,1)-np.sqrt(np.mean(a*a)))<1e-10
    enhanced_selftest()
    print('B9A11T MAC SELFTEST PASS | central differences and vector metrics')


def load_export(root):
    folder=root/'B9A11T_exports'
    path=folder/'B9A11T_U_reference.npz'
    meta=folder/'B9A11T_windows_summary.json'
    if not path.is_file() or not meta.is_file():
        raise FileNotFoundError('Transfer BOTH files from Windows B9A11T_exports to '+str(folder))
    md=json.loads(meta.read_text())
    if not md.get('all_jobs_pass') or md.get('cases')!=14:
        raise RuntimeError('Windows summary did not verify all 14 ODBs (including 2 baseline)')
    if md.get('npz_sha256')!=sha(path):raise RuntimeError('B9A11T Windows NPZ SHA256 mismatch')
    with np.load(path,allow_pickle=False) as z:
        data={k:np.asarray(z[k]).copy() for k in z.files}
    if tuple(data['case_ids'].astype(str))!=CASES:raise RuntimeError('Expected exact 7-case ordering')
    if abs(float(data['fraction'].item())-EPS)>1e-12:raise RuntimeError('Perturbation fraction mismatch')
    if data['material_MPa'].shape!=(7,3) or not np.allclose(
        data['material_MPa'],derive_mats(data['material_MPa'][0],EPS),rtol=0,atol=1e-6
    ):raise RuntimeError('Case material grid mismatch')
    for r in REGIONS:
        xyz=data.get('xyz_'+r);labels=data.get('labels_'+r);U=data.get('U_'+r)
        if xyz is None or labels is None or U is None:raise RuntimeError('Missing '+r)
        if xyz.shape!=(1008,3) or U.shape!=(7,1008,3) or labels.shape!=(1008,):
            raise RuntimeError('Unexpected FEM arrays '+r+': '+str((xyz.shape,U.shape,labels.shape)))
        if not np.isfinite(U).all() or not np.array_equal(labels,np.arange(1,1009)):
            raise RuntimeError('Invalid U fields or node labels '+r)
    return data,md


def map_coords(a,b):
    from scipy.spatial import cKDTree
    dist,idx=cKDTree(np.asarray(b,float)).query(np.asarray(a,float),k=1)
    if len(np.unique(idx))!=len(a) or float(dist.max())>2e-5:
        raise RuntimeError('FEM/B5 mesh does not match: max distance %.8g mm'%float(dist.max()))
    return idx, float(dist.max())


def context(args):
    import torch
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    b6=importlib.import_module('65AM_B6_mac')
    b9=importlib.import_module('65AM_B9_mac_saved_state')
    data,md=load_export(root)
    qc_path=root/'B9A11T_exports'/'B9A11T_FEM_quality.json'
    if not qc_path.is_file():raise FileNotFoundError('Run updated Windows --stage qc and transfer B9A11T_FEM_quality.json into B9A11T_exports')
    qc=json.loads(qc_path.read_text())
    if qc.get('npz_sha256')!=md.get('npz_sha256') or not qc.get('all_12_material_inputs_audited'):
        raise RuntimeError('Windows FEM material/finite-difference QC does not match the export')
    if qc.get('warnings'):print('REVIEW WARNING: finite-difference nonlinearity: ',qc['warnings'])
    anchor,anchorman=b9.load_anchor(root)
    ahash=str(data['anchor_sha256'].item())
    if ahash!=md.get('anchor_sha256') or ahash!=anchorman.get('anchor_sha256'):
        raise RuntimeError('Mac/Windows 65V anchor SHA256 mismatch')
    if not np.allclose(anchor['material'],data['material_MPa'][0],rtol=0,atol=1e-5):
        raise RuntimeError('The B5 and Abaqus anchor materials differ')
    design=root/'results'/'65AM_B9A11R_sensor_inverse'/args.design_tag/'B9A11R_locked_design.json'
    if not design.is_file():raise FileNotFoundError('Missing B9A11R locked sensor design '+str(design))
    locked=json.loads(design.read_text())
    if locked['variant']!='centered' or locked['loss']!='grad' or locked['sensors']!=20:
        raise RuntimeError('B9A11R design must use centered_grad, 20 sensors')
    dev=torch.device('cpu')
    models=b6.load_models(root,dev)
    info={}
    for region in REGIONS:
        d,model,ck=models[region]
        ds=d['ds'];ds['_b9_region']=region
        coords=np.asarray(ds['coordinates'],float)
        ii,dist=map_coords(coords,data['xyz_'+region])
        for j in range(3):
            m=data['material_MPa'][:,j]
            lo=float(np.asarray(ds['branch_inputs'][d['fit'],j]).min())
            hi=float(np.asarray(ds['branch_inputs'][d['fit'],j]).max())
            if m.min()<lo-1e-4 or m.max()>hi+1e-4:
                raise RuntimeError('%s %s (+/- 3%%) outside B5 FIT bounds %.4g..%.4g: %.4g..%.4g'%(
                    region,PROPS[j],lo,hi,m.min(),m.max()))
        hash_actual=hashlib.sha256(np.ascontiguousarray(coords,dtype=np.float64).tobytes()).hexdigest()
        if locked['regions'][region]['mesh_sha256']!=hash_actual:
            raise RuntimeError('B9A11R locked sensor nodes refer to a different mesh for '+region)
        indexes={}
        for mode in ('baseline','optimized'):
            ids=np.asarray(locked['regions'][region][mode],dtype=int)
            if ids.shape!=(20,) or len(np.unique(ids))!=20 or np.any(ids<0) or np.any(ids>=len(coords)):
                raise RuntimeError('Invalid 20-sensor layout '+region+' '+mode)
            if not np.isin(ids,np.setdiff1d(np.arange(len(coords)),d['fixed'])).all():
                raise RuntimeError('Locked sensors include prescribed interface node')
            indexes[mode]=ids
        info[region]=dict(d=d,model=model,coordinates=coords,indices=ii,layouts=indexes,distance=dist)
        print('CHECK PASS',region,'B5 nodes',len(coords),'max FEM match distance',dist,
              'B5 bounds include all perturbations')
    return root,b6,b9,anchor,data,md,info


def predict_all(root,b6,b9,anchor,data,region,packed):
    import torch
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    d=packed['d'];ds=d['ds'];model=packed['model']
    dev=torch.device('cpu')
    ctx=b6.b.CO.interface_context(root,ds,dev)
    branch0,lift0=b9.build_branch_lift(root,ds,anchor,ctx,dev)
    n=len(ds['coordinates']);out=[];err=0.
    fixed=np.asarray(d['fixed'],dtype=int)
    for mat in data['material_MPa']:
        branch=branch0.copy()
        branch[0,:3]=np.asarray(mat,np.float32)
        with torch.no_grad():
            lifted=apply_hard_compatibility_7region(
                torch.zeros((1,n,3),device=dev,dtype=torch.float32),
                torch.as_tensor(branch,device=dev,dtype=torch.float32),ctx).cpu().numpy()
        err=max(err,float(np.max(abs(lifted-lift0))))
        pred=b6.predict_b5(d,model,branch,lifted,dev)[0]
        if np.max(abs(pred[fixed]-lifted[0,fixed]))>2e-5:
            raise RuntimeError('B5 hard boundary compatibility failed '+region)
        out.append(pred)
    if err>3e-6:
        raise RuntimeError('Material changes unexpectedly altered the fixed-q interface lifting '+region+' '+str(err))
    return np.stack(out),err



def normalized_jacobian(U, sigma):
    """Columns = response to +3% relative property changes, in units of sigma.

    Actual central +/-3% FEM or B5 predictions are evaluated at the same q.
    This is a noise-normalized MATERIAL Jacobian, not the network NTK.
    """
    U=np.asarray(U,dtype=np.float64)
    if U.shape[0]!=7 or not np.isfinite(U).all() or sigma<=0:
        raise ValueError('7 finite material states and sigma>0 required')
    return np.stack([((U[2+2*j]-U[1+2*j])/(2.*sigma)).ravel()
                    for j in range(3)],axis=1)


def jacobian_report(Jf,Jb,region,layout):
    import numpy.linalg as LA
    if Jf.shape!=Jb.shape or Jf.shape[1]!=3:raise ValueError('Expected matched 3-column material Jacobians')
    sf=LA.svd(Jf,compute_uv=False);sb=LA.svd(Jb,compute_uv=False)
    floor=1.e-12
    gf=Jf.T@Jf;gb=Jb.T@Jb
    def correlations(J):
        norm=np.linalg.norm(J,axis=0)
        C=(J.T@J)/(norm[:,None]*norm[None,:]+floor)
        return np.clip(C,-1,1)
    def uncertainty(J):
        # Noise-only local linearized standard deviation on physical % perturbations.
        # J columns correspond to +3% parameter perturbation, so multiply by 3.
        return 3.*np.sqrt(np.maximum(np.diag(LA.pinv(J.T@J,rcond=1e-10)),0))
    relF=100*LA.norm(Jb-Jf,'fro')/max(LA.norm(Jf,'fro'),floor)
    rel2=100*LA.norm(Jb-Jf,2)/max(LA.norm(Jf,2),floor)
    return dict(region=region,layout=layout,
       FEM_singular_values=sf.tolist(),B5_singular_values=sb.tolist(),
       FEM_condition=float(sf[0]/max(sf[-1],floor)),
       B5_condition=float(sb[0]/max(sb[-1],floor)),
       FEM_collinearity=correlations(Jf).tolist(),
       B5_collinearity=correlations(Jb).tolist(),
       FEM_noise_only_linearized_std_pct=uncertainty(Jf).tolist(),
       B5_noise_only_linearized_std_pct=uncertainty(Jb).tolist(),
       relative_Jacobian_error_Fro_pct=float(relF),
       relative_Jacobian_error_spectral_pct=float(rel2),
       min_singular_ratio_B5_over_FEM=float(sb[-1]/max(sf[-1],floor)))


def enhanced_selftest():
    rng=np.random.default_rng(65)
    J=rng.normal(size=(14,3));data=np.zeros((7,14,1))
    for j in range(3):
        data[1+2*j,:,0]=-J[:,j]
        data[2+2*j,:,0]=J[:,j]
    found=normalized_jacobian(data,1.0)
    assert np.allclose(found,J)
    a=jacobian_report(J,J,'synthetic','test')
    assert abs(a['relative_Jacobian_error_Fro_pct'])<1e-10
    assert np.isclose(a['min_singular_ratio_B5_over_FEM'],1)
    assert np.isfinite(a['FEM_noise_only_linearized_std_pct']).all()
    print('B9A11T-REVISED SELFTEST PASS: normalized sensitivities, SVD, conditioning')


def diagnostics_figures(out,rows,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    for layout in ('baseline','optimized'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,region in zip(axes,REGIONS):
            q=next(x for x in diagnostics if x['layout']==layout and x['region']==region)
            ax.semilogy([1,2,3],q['FEM_singular_values'],'o-',label='Exact FEM')
            ax.semilogy([1,2,3],q['B5_singular_values'],'s--',label='Frozen B5')
            ax.set_xticks([1,2,3]);ax.set_xlabel('Singular-value index')
            ax.set_ylabel('Noise-whitened 3%-shift singular value')
            ax.set_title(region+' | cond(FEM) %.1f, cond(B5) %.1f'%(
                q['FEM_condition'],q['B5_condition']))
            ax.grid(alpha=.25);ax.legend()
        fig.suptitle('B9A11T: local inverse identifiability | '+layout)
        fig.tight_layout();fig.savefig(out/('B9A11T_noise_whitened_spectrum_'+layout+'.png'),dpi=190);plt.close(fig)
    df=pd.DataFrame(rows)
    df.to_csv(out/'B9A11T2_jacobian_identifiability.csv',index=False)
    for region in REGIONS:
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for ax,layout in zip(axes,('baseline','optimized')):
            r=next(x for x in diagnostics if x['layout']==layout and x['region']==region)
            C=np.asarray(r['FEM_collinearity'])
            D=np.asarray(r['B5_collinearity'])
            dif=D-C
            image=ax.imshow(dif,vmin=-.5,vmax=.5,cmap='coolwarm')
            ax.set_xticks(range(3));ax.set_xticklabels(PROPS)
            ax.set_yticks(range(3));ax.set_yticklabels(PROPS)
            ax.set_title(layout+' | B5 - FEM')
            for i in range(3):
                for j in range(3):ax.text(j,i,'%.2f'%dif[i,j],ha='center',va='center',fontsize=9)
        fig.colorbar(image,ax=list(axes),fraction=.05,pad=.04,label='Sensitivity correlation difference')
        fig.suptitle('B9A11T: property confounding difference | '+region)
        fig.subplots_adjust(left=.08,right=.86,top=.85,bottom=.12,wspace=.3)
        fig.savefig(out/('B9A11T_confounding_difference_'+region+'.png'),dpi=190);plt.close(fig)


def run(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,b6,b9,anchor,data,md,info=context(args)
    out=root/'results'/OUT_BASE/args.tag
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Use another tag: '+str(out))
    rows=[];forward=[];per_region={};jreports=[];curvature=[]
    for region,packed in info.items():
        Ub5,lift_err=predict_all(root,b6,b9,anchor,data,region,packed)
        Ufem=data['U_'+region][:,packed['indices'],:]
        if Ub5.shape!=Ufem.shape:
            raise RuntimeError('Predicted/FEM field shape mismatch '+region)
        material=data['material_MPa'][0]
        for layout,ids in packed['layouts'].items():
            a=Ufem[:,ids,:];b=Ub5[:,ids,:]
            jac_f=normalized_jacobian(a,args.sigma_u)
            jac_b=normalized_jacobian(b,args.sigma_u)
            jreports.append(jacobian_report(jac_f,jac_b,region,layout))
            for j,prop in enumerate(PROPS):
                def curvature_stat(x):
                    um=x[1+2*j]; up=x[2+2*j]; u0=x[0]
                    odd=(up-um)/2.; even=(up+um)/2.-u0
                    return float(np.sqrt(np.mean(even*even))/max(np.sqrt(np.mean(odd*odd)),1e-15))
                curvature.append(dict(region=region,layout=layout,property=prop,
                    FEM_even_over_odd=curvature_stat(a),B5_even_over_odd=curvature_stat(b)))
            for cid,af,bf in zip(CASES,a,b):
                forward.append(dict(region=region,layout=layout,case=cid,
                    FEM_B5_rmse_mm=float(np.sqrt(np.mean((af-bf)**2))),
                    FEM_B5_relative_pct=float(100*np.linalg.norm(af-bf)/max(np.linalg.norm(af),1e-14))))
            for j,prop in enumerate(PROPS):
                lo,hi=1+2*j,2+2*j
                # Actual finite differences are taken from independently converged Abaqus ODBs.
                delta_f=(a[hi]-a[lo])/2.
                delta_b=(b[hi]-b[lo])/2.
                err,cos=compare_vectors(delta_f,delta_b)
                ip_f=norm_shift(delta_f,args.sigma_u)
                ip_b=norm_shift(delta_b,args.sigma_u)
                rows.append(dict(region=region,layout=layout,property=prop,baseline_MPa=float(material[j]),
                                 perturbation_fraction=EPS,
                                 FEM_rms_shift_over_sigma=ip_f,
                                 B5_rms_shift_over_sigma=ip_b,
                                 B5_to_FEM_importance_ratio=float(ip_b/max(ip_f,1e-12)),
                                 derivative_relative_error_pct=err,
                                 derivative_cosine_similarity=cos,
                                 n_nodes=len(ids),n_measurement_components=3*len(ids)))
        # Additional full-subdomain diagnostics, NOT scored as physical surface sensors.
        for j,prop in enumerate(PROPS):
            lo,hi=1+2*j,2+2*j
            f=(Ufem[hi]-Ufem[lo])/2.;b=(Ub5[hi]-Ub5[lo])/2.
            er,cs=compare_vectors(f,b)
            rows.append(dict(region=region,layout='full_subdomain_diagnostic',property=prop,
                  baseline_MPa=float(material[j]),perturbation_fraction=EPS,
                  FEM_rms_shift_over_sigma=norm_shift(f,args.sigma_u),
                  B5_rms_shift_over_sigma=norm_shift(b,args.sigma_u),
                  B5_to_FEM_importance_ratio=norm_shift(b,args.sigma_u)/max(norm_shift(f,args.sigma_u),1e-12),
                  derivative_relative_error_pct=er,derivative_cosine_similarity=cs,
                  n_nodes=len(packed['coordinates']),n_measurement_components=f.size))
        per_region[region]=dict(node_matching_max_distance_mm=packed['distance'],
                                max_fixed_q_lifting_change_mm=lift_err)
        print('PREDICTED',region,'7 conditions using B5, FEM labels used only for evaluation')
    df=pd.DataFrame(rows);fwd=pd.DataFrame(forward)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'B9A11T_exact_gradients.csv',index=False)
    fwd.to_csv(out/'B9A11T_forward_cases.csv',index=False)
    import pandas as pd
    pd.DataFrame(curvature).to_csv(out/'B9A11T2_finite_difference_curvature.csv',index=False)
    diagnostics_figures(out,jreports,jreports)
    for layout in ('baseline','optimized'):
        sub=df[df.layout==layout]
        agg=sub.groupby('property',sort=False)[['FEM_rms_shift_over_sigma','B5_rms_shift_over_sigma',
              'derivative_relative_error_pct','derivative_cosine_similarity']].mean().reindex(PROPS)
        x=np.arange(3);w=.38
        fig,ax=plt.subplots(figsize=(8.6,5.1))
        ax.bar(x-w/2,agg['FEM_rms_shift_over_sigma'],w,label='Exact Abaqus finite difference')
        ax.bar(x+w/2,agg['B5_rms_shift_over_sigma'],w,label='Frozen B5 at same prescribed q')
        ax.set_xticks(x);ax.set_xticklabels(PROPS)
        ax.set_ylabel('RMS displacement shift / sigma for a 3% material change')
        ax.set_title('B9A11T | Matched FEM vs B5 | '+layout+' 20 sensors')
        ax.legend();fig.tight_layout()
        fig.savefig(out/('B9A11T_matched_importance_'+layout+'.png'),dpi=190)
        plt.close(fig)
    for metric,ylim,filename,ylabel in [
      ('derivative_relative_error_pct',None,'B9A11T_gradient_relative_error.png','Relative FEM-B5 gradient error (%)'),
      ('derivative_cosine_similarity',(-1.02,1.02),'B9A11T_gradient_cosine.png','FEM-B5 gradient cosine similarity')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
        for ax,region in zip(axes,REGIONS):
            sub=df[(df.region==region)&(df.layout.isin(['baseline','optimized']))]
            piv=sub.pivot(index='property',columns='layout',values=metric).reindex(PROPS)
            xx=np.arange(3)
            ax.bar(xx-.19,piv['baseline'],.38,label='Original 20')
            ax.bar(xx+.19,piv['optimized'],.38,label='Optimized 20')
            ax.set_xticks(xx);ax.set_xticklabels(PROPS)
            if ylim:ax.set_ylim(*ylim)
            ax.set_title(region);ax.set_ylabel(ylabel);ax.legend()
        fig.suptitle('B9A11T: exact same-q FEM derivative against frozen B5')
        fig.tight_layout();fig.savefig(out/filename,dpi=190);plt.close(fig)
    summary=dict(step='B9A11T',stage='matched_exact_FEM_B5_jacobian',
         source_anchor_sha256=md['anchor_sha256'],fem_export_sha256=md['npz_sha256'],
         fraction=EPS,noise_sigma_U_mm=args.sigma_u,
         cases_per_region=len(CASES),
         same_interface_q=True,original_B5_centered_grad_unchanged=True,
         is_true_Abaqus_finite_difference=True,not_full_coupled_model=True,
         is_local_sensitivity_not_inverse_validation=True,
         per_region_checks=per_region,
         rankings={},metrics={},
         source_paper='Sanjalawe et al., Archives of Computational Methods in Engineering (2026), DOI 10.1007/s11831-026-10747-9',
         gradient_diagnostic_is_material_Jacobian_not_PINN_NTK=True,
         local_linear_noise_only_uncertainty_is_NOT_calibrated_Bayesian_posterior=True,
         quality_control_requires_exact_original_12_jobs=True,
         enhanced_diagnostics=jreports,
         finite_difference_curvature=curvature)
    for layout in ('baseline','optimized'):
        g=df[df.layout==layout].groupby('property').agg(
           FEM=('FEM_rms_shift_over_sigma','mean'),
           B5=('B5_rms_shift_over_sigma','mean'),
           relative_error=('derivative_relative_error_pct','mean'),
           cosine=('derivative_cosine_similarity','mean'))
        summary['rankings'][layout]={k:g[k].sort_values(ascending=False).index.tolist() for k in ('FEM','B5')}
        summary['metrics'][layout]={prop:{key:float(g.loc[prop,key]) for key in ('FEM','B5','relative_error','cosine')} for prop in PROPS}
    (out/'B9A11T_summary.json').write_text(json.dumps(summary,indent=2))
    print('B9A11T DONE:',out)
    print('No optimizer retrained. Not a fully coupled seven-region inverse test.')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--project',default='.')
    a.add_argument('--stage',choices=('selftest','check','run'),required=True)
    a.add_argument('--tag',default='run01')
    a.add_argument('--design-tag',default=DEFAULT_DESIGN)
    a.add_argument('--sigma-u',type=float,default=0.0002)
    args=a.parse_args()
    if args.stage=='selftest':return selftest()
    if args.sigma_u<=0:raise ValueError('sigma-u must be positive')
    if args.stage=='check':context(args);print('B9A11T MAC INPUT CHECK PASS');return
    run(args)


if __name__=='__main__':main()