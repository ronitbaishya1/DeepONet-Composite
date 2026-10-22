#!/usr/bin/env python3
"""B9A11H-C: full-field three-component U + RF parameter sensitivity and greedy sensor design.

Read-only 50-case full-FEM data, same quadratic response surface as B9A11H-B.
Use ONLY train data for fitting, validation for regularization/noise floors,
and untouched test for ONE final forward-surrogate assessment. Not FE–NO/B5.
Finite parameter prior: z=(theta-CENTER)/HALF in [-1,1]^3.

Examples:
  python3 65AM_B9A11H2_C_sensitivity.py --stage check --project .
  python3 65AM_B9A11H2_C_sensitivity.py --stage run --project . --tag run01
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

COMP=('U1','U2','U3')
MATS=('E1_MPa','E2_MPa','G12_MPa')

def rel(a,b):
    return float(100*np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-14))


def fit_full(d,B):
    # A 10-term quadratic surface and ordinary ridge training. Avoid any TEST influence on fitting.
    z=(d['X']-B.CENTER)/B.HALF
    tr,va,te=(d['groups'][k] for k in ('train','validation','test'))
    Y=np.concatenate([d['u1'],d['u2'],d['u3'],d['rf'][:,None]],axis=1)
    sd=np.maximum(Y[tr].std(axis=0),1e-7)
    fits=[]
    for alpha in B.LAMBDAS:
        m=B.fit_model(z,Y,alpha,tr)
        err=float(np.sqrt(np.mean(((B.predict(m,z[va])-Y[va])/sd)**2)))
        fits.append(dict(alpha=alpha,validation_standardized_RMSE=err))
    chosen=min(fits,key=lambda x:(x['validation_standardized_RMSE'],x['alpha']))
    model=B.fit_model(z,Y,chosen['alpha'],np.r_[tr,va])
    pred=B.predict(model,z[te])
    hold=[dict(component=comp,relative_l2_percent=rel(pred[:,i*len(d['xyz']):(i+1)*len(d['xyz'])],
                       Y[te,i*len(d['xyz']):(i+1)*len(d['xyz'])]),
                       rmse=float(np.sqrt(np.mean((pred[:,i*len(d['xyz']):(i+1)*len(d['xyz'])]-
                                                Y[te,i*len(d['xyz']):(i+1)*len(d['xyz'])])**2))))
          for i,comp in enumerate(COMP)]
    hold.append(dict(component='RF2_N',relative_l2_percent=rel(pred[:,-1],Y[te,-1]),
         rmse=float(np.sqrt(np.mean((pred[:,-1]-Y[te,-1])**2)))))
    # Source of model-error floors is an UNREFITTED train-only model vs validation responses.
    prem=B.fit_model(z,Y,chosen['alpha'],tr)
    valres=B.predict(prem,z[va])-Y[va]
    rmse=np.sqrt(np.mean(valres*valres,axis=0))
    floors=[float(np.sqrt(np.mean(rmse[i*len(d['xyz']):(i+1)*len(d['xyz'])]**2))) for i in range(3)]
    floors.append(float(rmse[-1]))
    return model,Y,rmse,floors,chosen,fits,hold


def greedy_sensor_selection(xyz,J,sigma,min_dist_mm=0.85,nmax=50):
    """Greedy D-optimal node selection. One node provides all THREE displacement components.
    Prior precision=3I for independent uniform z_j ~ U[-1,1].
    sigma is a 3-vector with conservative validation-model-error floors.
    """
    n=len(xyz)
    j=J.reshape(3,n,3).transpose(1,0,2)/np.asarray(sigma)[None,:,None]
    node_fisher=np.einsum('nki,nkj->nij',j,j,optimize=True)
    precision=3*np.eye(3,dtype=float)
    taken=[];rows=[]
    possible=np.ones(n,dtype=bool)
    for k in range(nmax):
        candidates=np.flatnonzero(possible)
        if not len(candidates):raise RuntimeError('Insufficient geometry-separated nodes to place 50 sensors')
        signs,ld=np.linalg.slogdet(precision[None,:,:]+node_fisher[candidates])
        scores=np.where(signs>0,ld,-np.inf)
        ix=int(candidates[int(np.argmax(scores))])
        if not np.isfinite(scores).any():raise RuntimeError('Nonpositive candidate Fisher matrices')
        precision+=node_fisher[ix];taken.append(ix)
        dist=np.linalg.norm(xyz-xyz[ix],axis=1)
        possible &= (dist>=min_dist_mm)
        possible[ix]=False
        s=np.linalg.svd(precision-3*np.eye(3),compute_uv=False)
        record=dict(Rank=k+1,NodeIndex=ix,X=float(xyz[ix,0]),Y=float(xyz[ix,1]),Z=float(xyz[ix,2]),
              d_opt_logdet=float(np.linalg.slogdet(precision)[1]),
              whitened_data_singular_min=float(np.sqrt(max(s[-1],0))))
        rows.append(record)
    return rows


def compute(root,tag,chosen_data,B,only_check=False,noise_u=.0002,noise_rf=5.):
    data_folder=B.dataset_dir(root,chosen_data)
    d=B.read_data(data_folder)
    if not (d['u3'].shape==(50,4592) and len(d['groups']['test'])>=3):raise RuntimeError('Missing independent full-field U3 or test cohort')
    print('C input: 50 full FEM cases; displacements 3 x 4592; RF2')
    print('Original train/validation/test:',*[len(d['groups'][g]) for g in ('train','validation','test')])
    if only_check:return
    model,Y,err,fl,chosen,cv,hold=fit_full(d,B)
    D=B.jacdesign(np.zeros(3)).T # shape [3,10]
    jac=(D@model['coef']*model['scale']).T # [3*nodes+RF, 3] per unit z_j
    n=len(d['xyz']); jfield=jac[:-1,:]
    sig_u=np.sqrt(np.array(fl[:3])**2+noise_u**2)
    sensors=greedy_sensor_selection(d['xyz'],jfield,sig_u)
    sens=[]
    for i,name in enumerate(COMP):
        J=jac[i*n:(i+1)*n]
        reference_rms=float(np.sqrt(np.mean(Y[:,i*n:(i+1)*n]**2)))
        for k,mat in enumerate(MATS):
            sens.append(dict(component=name,parameter=mat,
                parameter_unit='normalized +/-10% range',
                field_jacobian_rms=float(np.sqrt(np.mean(J[:,k]**2))),
                fractional_reference_rms_change=float(np.sqrt(np.mean(J[:,k]**2))/max(reference_rms,1e-12)),
                reference_field_rms=reference_rms))
    for k,mat in enumerate(MATS):
        sens.append(dict(component='RF2_N',parameter=mat,parameter_unit='normalized +/-10% range',
           field_jacobian_rms=float(abs(jac[-1,k])),
           fractional_reference_rms_change=float(abs(jac[-1,k])/max(np.sqrt(np.mean(Y[:,-1]**2)),1e-12)),
           reference_field_rms=float(np.sqrt(np.mean(Y[:,-1]**2)))))
    # Use the same (conservative) validation-derived component noise floors for all sensors.
    sets={};maxn=50
    for ns in (5,10,20,50):
        ids=[v['NodeIndex'] for v in sensors[:ns]]
        J=jac[:-1].reshape(3,n,3)[:,ids,:].reshape(-1,3)
        var=np.repeat(sig_u**2,ns)
        jr=jac[-1:,:]/np.sqrt(fl[3]**2+noise_rf**2)
        w=np.concatenate([J/np.sqrt(var)[:,None],jr])
        s=np.linalg.svd(w,compute_uv=False)
        prec=3*np.eye(3)+w.T@w
        cov=np.linalg.inv(prec)
        sets[str(ns)]=dict(whitened_jacobian_singular_values=s.tolist(),
           jacobian_condition_number=float(s[0]/max(s[-1],1e-15)),
           posterior_linearized_1sigma_MPa=(np.sqrt(np.diag(cov))*B.HALF).tolist(),
           node_count=ns,displacement_components_per_node=3,plus_reaction_force=True)
    out=root/'results'/'65AM_B9A11H_v2'/tag/'C_sensitivity'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output protected: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(sensors).to_csv(out/'B9A11H_C_sensor_rankings.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'B9A11H_C_component_sensitivity.csv',index=False)
    pd.DataFrame(hold).to_csv(out/'B9A11H_C_heldout_forward_errors.csv',index=False)
    pd.DataFrame(cv).to_csv(out/'B9A11H_C_ridge_validation.csv',index=False)
    summary=dict(step='B9A11H-C',cohort='50-case FULL-FEM dataset, not B5 or 65V parametric solver',
      source=str(data_folder),material_MPa=B.CENTER.tolist(),normalized_parameter_half_ranges_MPa=B.HALF.tolist(),
      best_alpha=chosen,holdout_errors=hold,validation_component_model_error_floors={'U1_mm':fl[0],'U2_mm':fl[1],'U3_mm':fl[2],'RF2_N':fl[3]},
      fisher_sensor_summaries=sets,
      measurement_noise_assumed={'U_mm_1sigma':noise_u,'RF_N_1sigma':noise_rf},
      cautions=['Sensitivity is derivative of fitted quadratic emulator, not exact Abaqus derivative',
       'Identifiability is local at nominal material; model error and measurement correlations can make posterior wider',
       'Use independent experimental/held-out damage data for damage inference',
       'Do not interpret parameters whose posterior is prior-boundary-limited as well identified',
       'Ranked sensor positions optimized using train/validation only; tests were not used to select them',
       'A poorly conditioned/singular whitened measurement Jacobian means all three properties are not separably identifiable'])
    (out/'B9A11H_C_summary.json').write_text(json.dumps(summary,indent=2))
    print('C COMPLETE:',out)
    for x in hold:print('  held-out %-6s relative L2 %8.3f%%'%(x['component'],x['relative_l2_percent']))
    for ns in ('5','10','20','50'):
        x=sets[ns]
        print('  %2s sensors: singular values %s'%(ns,np.round(x['whitened_jacobian_singular_values'],2).tolist()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='.')
    ap.add_argument('--data-dir')
    ap.add_argument('--tag',default='run01')
    ap.add_argument('--stage',choices=('check','run','selftest'),required=True)
    ap.add_argument('--noise-u-mm',type=float,default=.0002)
    ap.add_argument('--noise-rf-N',type=float,default=5.)
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import importlib
    B=importlib.import_module('65AM_B9A11H2_B_inverse_uq')
    if args.stage=='selftest':
        rng=np.random.default_rng(8);x=rng.normal(size=(30,3));j=rng.normal(size=(3,30,3));
        rows=greedy_sensor_selection(x,j.reshape(90,3),np.array([.001,.001,.001]),.001,10)
        assert len(set(r['NodeIndex'] for r in rows))==10
        print('C SELFTEST PASS');return
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid tag')
    if args.noise_u_mm<=0 or args.noise_rf_N<=0:raise ValueError('Positive noise assumptions needed')
    compute(root,args.tag,args.data_dir,B,args.stage=='check',args.noise_u_mm,args.noise_rf_N)

if __name__=='__main__':main()
