#!/usr/bin/env python3
"""B9A10-P: finite shear, stress-measure and material-frame audit.

This is a read-only constitutive diagnostic using 12 NEW C3D8 ODBs.
No parameter tuning or changes to B5 / Broyden / 65V.

python3 65AM_B9A10P_mac_shear_audit.py --project . --stage selftest
python3 65AM_B9A10P_mac_shear_audit.py --project . --stage check
python3 65AM_B9A10P_mac_shear_audit.py --project . --stage diagnose
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
DT=torch.float64
ROOT_OUTPUT=Path('results/65AM_B9_coupled_feasibility/B9A10P_finite_shear')
CASES=[('B9A10P_%s_%s'%(mat,code),mat,code,amp)
       for mat in ('O00','O17','ISO') for code,amp in (('Q25',.25),('Q50',.5),('Q100',1.),('Q200',2.))]
P=dict(E1=46500.,E2=12500.,E3=12000.,nu12=.28,nu13=.28,nu23=.40,G12=4750.,G13=4500.,G23=3500.)
COMP=('S11','S22','S33','S12','S13','S23')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def report_csv(path,records):
    if not records:raise RuntimeError('Empty report '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def pct(error,reference):
    return 100.*float(np.linalg.norm(error))/max(1e-12,float(np.linalg.norm(reference)))

def Qref(code):
    a=math.radians(17. if code=='O17' else 0.)
    return torch.tensor([[math.cos(a),-math.sin(a),0.],
                         [math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=DT)

def Cmat(material):
    C=torch.zeros((6,6),dtype=DT)
    if material=='ISO':
        E,nu=20000.,.25
        la=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
        for i in range(3):
            for j in range(3):C[i,j]=la
            C[i,i]+=2*mu
        for i in range(3,6):C[i,i]=mu
        return C
    c=torch.zeros((6,6),dtype=DT)
    c[0,0]=1/P['E1'];c[1,1]=1/P['E2'];c[2,2]=1/P['E3']
    c[0,1]=c[1,0]=-P['nu12']/P['E1']
    c[0,2]=c[2,0]=-P['nu13']/P['E1']
    c[1,2]=c[2,1]=-P['nu23']/P['E2']
    c[3,3]=1/P['G12'];c[4,4]=1/P['G13'];c[5,5]=1/P['G23']
    if not bool((torch.linalg.eigvalsh(c)>0).all()):raise RuntimeError('Invalid elastic stiffness')
    return torch.linalg.inv(c)

def eng(e):
    return torch.stack([e[0,0],e[1,1],e[2,2],2*e[0,1],2*e[0,2],2*e[1,2]])

def t6_torch(s):
    a,b,c,d,e,f=s.unbind(-1)
    return torch.stack([torch.stack([a,d,e]),torch.stack([d,b,f]),torch.stack([e,f,c])])

def t6_np(s):
    a,b,c,d,e,f=np.asarray(s)
    return np.array([[a,d,e],[d,b,f],[e,f,c]],float)

def vec6(t):
    return np.array([t[0,0],t[1,1],t[2,2],t[0,1],t[0,2],t[1,2]],float)

def polar(F):
    R=F
    for _ in range(9):R=.5*(R+torch.linalg.inv(R).T)
    return R

def log_strain(F):
    # Differentiable matrix log near identity via atanh expansion, as in B9A10-O.
    ident=torch.eye(3,dtype=DT)
    CC=F.T@F
    Y=torch.linalg.solve(CC+ident,CC-ident)
    Y=.5*(Y+Y.T)
    if torch.linalg.matrix_norm(Y).detach().item()>.45:
        raise RuntimeError('Large-deformation range exceeds series validation')
    z=Y;Y2=Y@Y;result=Y
    for k in range(1,22):z=z@Y2;result=result+z/(2*k+1)
    return .5*(result+result.T)

def predictions(F,material):
    # Reference orientation, not measured Abaqus orientation.
    R=polar(F);Q=Qref(material);e=Q.T@log_strain(F)@Q
    sloc=Cmat(material)@eng(e)
    srot=R@Q@t6_torch(sloc)@Q.T@R.T
    # Two PREDECLARED possibilities (not fitted to output):
    sigma_cauchy=srot
    sigma_from_kirchhoff=srot/torch.linalg.det(F)
    return sigma_cauchy,sigma_from_kirchhoff,sloc,eng(e),R

def F_from_U(X,U):
    XC=X-X.mean(axis=0)
    grad=np.linalg.lstsq(XC,U,rcond=None)[0].T
    fit=XC@grad.T + U.mean(axis=0)
    return np.eye(3)+grad,float(np.max(np.abs(U-fit)))

def validate_source(project):
    directory=project/'B9A10P_exports'
    npz=directory/'B9A10P_shear_fields.npz'; summary=directory/'B9A10P_windows_summary.json'
    if not npz.is_file() or not summary.is_file():
        raise FileNotFoundError('Copy complete B9A10P_exports folder from Windows into project root')
    metadata=json.loads(summary.read_text())
    if metadata.get('count')!=12 or sha(npz)!=metadata.get('npz_sha256'):
        raise RuntimeError('Corrupted or mismatched Windows export (checksum)')
    original={r['job']:r for r in metadata['cases']}
    if set(original)!={r[0] for r in CASES}:raise RuntimeError('Windows case list mismatch')
    with np.load(npz,allow_pickle=False) as z:
        for name,mat,code,alpha in CASES:
            arrays={k:z[name+'__'+k] for k in ('U','RF','S','LE','Q','t','X')}
            nf=len(arrays['t'])
            if arrays['U'].shape!=(nf,8,3) or arrays['S'].shape!=(nf,8,6) or arrays['LE'].shape!=(nf,8,6) or arrays['Q'].shape!=(nf,8,3,3):
                raise RuntimeError('Unexpected array shapes: '+name)
            if nf<3 or not all(np.isfinite(v).all() for v in arrays.values()):
                raise RuntimeError('Invalid values: '+name)
            f,fit=F_from_U(arrays['X'],arrays['U'][-1])
            expect=np.eye(3);expect[0,1]=.016*alpha;expect[1,2]=-.004*alpha
            if fit>2e-5 or np.linalg.norm(f-expect)>5e-5:
                raise RuntimeError('Unexpected final deformation: '+name)
    print('B9A10P MAC CHECK PASS | 12 C3D8 cases | SHA verified | all deformations match')
    return npz,metadata

def selftest():
    I=torch.eye(3,dtype=DT)
    for m in ('O00','O17','ISO'):
        sa,sb,sl,eps,rr=predictions(I,m)
        assert sa.abs().max().item()<1e-10
    theta=.25;c=math.cos(theta);s=math.sin(theta)
    Rot=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=DT)
    assert predictions(Rot,'O17')[0].abs().max().item()<1e-8
    G=torch.eye(3,dtype=DT,requires_grad=True)
    trial=G+torch.tensor([[.006,.015,0.],[0.,-.004,0.],[0.,0.,.003]],dtype=DT)
    loss=predictions(trial,'O17')[0].square().sum()
    grad=torch.autograd.grad(loss,G)[0]
    assert torch.isfinite(grad).all() and grad.norm().item()>0.
    x=np.array([[0,0,0],[.6,0,0],[.6,.5,0],[0,.5,0],[0,0,.62],[.6,0,.62],[.6,.5,.62],[0,.5,.62]])
    ff=np.eye(3);ff[0,1]=.016;ff[1,2]=-.004
    f,fit=F_from_U(x,x@(ff-np.eye(3)).T)
    np.testing.assert_allclose(f,ff,atol=1e-12); assert fit<1e-12
    print('B9A10P MAC SELFTEST PASS | objectivity | affine recovery | orthotropic stability | autograd')

def analyze(project):
    npz,metadata=validate_source(project)
    dest=project/ROOT_OUTPUT
    if not dest.is_dir():raise RuntimeError('Create output folder using Finder: '+str(dest))
    names=['B9A10P_cases.csv','B9A10P_frames.csv','B9A10P_components.csv',
           'B9A10P_summary.json','B9A10P_amplitude_sweep.png','B9A10P_shear_components.png',
           'B9A10P_stress_measure.png','B9A10P_gradients.json']
    if any((dest/n).exists() for n in names):raise RuntimeError('Existing P results protected; move them before rerun')
    cases=[];frames=[];components=[]
    with np.load(npz,allow_pickle=False) as z:
        for name,mat,code,alpha in CASES:
            U=z[name+'__U'];Sa=z[name+'__S'];LE=z[name+'__LE'];Q=z[name+'__Q'];tt=z[name+'__t'];X=z[name+'__X']
            # Normalize all saved-time quantities by this case's final stress norm.
            refnorm=np.linalg.norm(Sa[-1]);
            if refnorm<1e-7:raise RuntimeError('Reference stress nearly zero: '+name)
            final=None
            for k in range(len(tt)):
                F,affine=F_from_U(X,U[k]);J=np.linalg.det(F)
                if J<.7 or affine>2e-5:raise RuntimeError('Non-affine or inverted shear patch '+name)
                with torch.no_grad():
                    ca,ki,sl,ev,rr=predictions(torch.tensor(F,dtype=DT),mat)
                    ca=ca.numpy();ki=ki.numpy();sl=sl.numpy();ev=ev.numpy();rr=rr.numpy()
                    cc=Cmat(mat).numpy()
                # Convention from the verified O diagnostic: Abaqus tensor
                # reported in local coordinates, with row-basis localCoordSystem.
                refg=np.stack([Q[k,i].T@t6_np(Sa[k,i])@Q[k,i] for i in range(8)])
                pg=np.broadcast_to(ca,(8,3,3));kg=np.broadcast_to(ki,(8,3,3))
                eta=np.mean(LE[k],axis=0)
                # Report BOTH shear conventions; don't quietly select one.
                hooke_direct=eta@cc.T
                etadouble=eta.copy();etadouble[3:]*=2.
                hooke_double=etadouble@cc.T
                norm_final=refnorm
                frames.append(dict(job=name,material=mat,amplitude=alpha,frame=k,time=float(tt[k]),J=float(J),
                    cauchy_pct_final_scale=100*np.linalg.norm(pg-refg)/norm_final,
                    kirchhoff_pct_final_scale=100*np.linalg.norm(kg-refg)/norm_final,
                    strain_direct_pct_final_scale=100*np.linalg.norm(ev-eta)/max(np.linalg.norm(LE[-1]),1e-12),
                    hooke_direct_pct_final_scale=100*np.linalg.norm(hooke_direct-Sa[k])/norm_final,
                    hooke_doubled_pct_final_scale=100*np.linalg.norm(hooke_double-Sa[k])/norm_final,
                    affine_fit_mm=affine))
                if k==len(tt)-1:
                    # The actual reference local stress is S. Predicted local sigma is
                    # brought into the *current* objective material frame.
                    rQ=(np.eye(3) if mat=='ISO' else rr@Qref(mat).numpy())
                    loc_pred=rQ.T@ca@rQ
                    obs_loc=np.mean(Sa[-1],axis=0)
                    pred_loc=vec6(loc_pred)
                    # Component errors normalized by total S norm, so comparable
                    # across different components without tiny component denominators.
                    for j,lab in enumerate(COMP):
                        normcomp=100.*np.linalg.norm(np.full(8,pred_loc[j])-Sa[-1,:,j])/norm_final
                        components.append(dict(job=name,material=mat,amplitude=alpha,component=lab,
                                observed_local_MPa=float(obs_loc[j]),predicted_local_MPa=float(pred_loc[j]),
                                component_error_pct_totalnorm=normcomp))
                    final=dict(job=name,material=mat,amplitude=alpha,frames=len(tt),J=float(J),
                        cauchy_error_pct=pct(pg-refg,refg),kirchhoff_error_pct=pct(kg-refg,refg),
                        strain_engineering_error_pct=pct(ev-eta,LE[-1]),
                        hooke_abaqus_LE_direct_pct=pct(hooke_direct-Sa[-1],Sa[-1]),
                        hooke_abaqus_LE_doubled_pct=pct(hooke_double-Sa[-1],Sa[-1]),
                        observed_local_S12_MPa=float(obs_loc[3]),predicted_local_S12_MPa=float(pred_loc[3]),
                        local_S12_abs_error_MPa=float(abs(pred_loc[3]-obs_loc[3])))
            cases.append(final)
            print('CASE',name,'amplitude',alpha,'Cauchy %.6f%%'%final['cauchy_error_pct'],
                  'Kirchhoff %.6f%%'%final['kirchhoff_error_pct'])
    # Use existing B9A10O reference shear test as regression indicator only.
    target=next(c for c in cases if c['job']=='B9A10P_O00_Q100')
    baseline_diff=abs(target['cauchy_error_pct']-.12404839820956058)
    summary=dict(step='65AM-B9A10-P',cases=12,material_constants=P,
           reference_B9A10O_O00_SH_final_error_pct=.12404839820956058,
           O00_Q100_minus_O_SH_abs_percentage_points=float(baseline_diff),
           O_regression_warning=bool(baseline_diff>.015),
           methods=['L corotational Hencky-Cauchy','Kirchhoff interpretation divided by J'],
           group_by_material={m:{'mean_cauchy_error_pct':float(np.mean([x['cauchy_error_pct'] for x in cases if x['material']==m])),
                        'mean_kirchhoff_error_pct':float(np.mean([x['kirchhoff_error_pct'] for x in cases if x['material']==m]))}
                        for m in ('O00','O17','ISO')},
           physical_exactness='candidate only, cannot infer exact Abaqus constitutive law',
           predeclared_no_fitting=True,selection_requires_fresh_holdout=True,
           no_changes_to_B5_65V=True,physics_loss_ready=False)
    report_csv(dest/names[0],cases);report_csv(dest/names[1],frames);report_csv(dest/names[2],components)
    (dest/names[3]).write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    for m in ('O00','O17','ISO'):
        rr=sorted((r for r in cases if r['material']==m),key=lambda r:r['amplitude'])
        ax.plot([r['amplitude'] for r in rr],[r['cauchy_error_pct'] for r in rr],'-o',label=m+' Cauchy')
    ax.set(xlabel='Shear amplitude / B9A10-O baseline',ylabel='Final stress error (%)',title='B9A10-P | finite-shear stress error')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(dest/names[4],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    for m in ('O00','O17','ISO'):
        rr=sorted((r for r in cases if r['material']==m),key=lambda r:r['amplitude'])
        ax.plot([r['amplitude'] for r in rr],[r['local_S12_abs_error_MPa'] for r in rr],'-o',label=m)
    ax.set(xlabel='Shear amplitude / B9A10-O baseline',ylabel='Local S12 absolute error (MPa)',title='B9A10-P | constitutive shear-component gap')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(dest/names[5],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    rr=sorted((r for r in cases if r['material']=='O00'),key=lambda r:r['amplitude'])
    ax.plot([r['amplitude'] for r in rr],[r['cauchy_error_pct'] for r in rr],'-o',label='Hencky-Cauchy L')
    ax.plot([r['amplitude'] for r in rr],[r['kirchhoff_error_pct'] for r in rr],'-s',label='Kirchhoff / J')
    ax.set(xlabel='Shear amplitude / B9A10-O baseline',ylabel='Final stress error (%)',title='B9A10-P | stress measure audit (O00)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(dest/names[6],dpi=160);plt.close(fig)
    seed=torch.eye(3,dtype=DT,requires_grad=True)
    trial=seed+torch.tensor([[0.,.013,0.],[0.,0.,-.003],[0.,0.,0.]],dtype=DT)
    stress=predictions(trial,'O17')[0];gradient=torch.autograd.grad(stress.square().sum(),seed)[0]
    gd=dict(autograd_pass=bool(torch.isfinite(gradient).all() and gradient.norm().item()>0),
            gradient_norm_F=float(gradient.norm().item()),dtype='float64',
            scope='single-element constitutive prediction, not B5')
    (dest/names[7]).write_text(json.dumps(gd,indent=2,allow_nan=False))
    print('B9A10P COMPLETE | 12 new cases | O baseline difference %.7f percentage points'%baseline_diff,
          '| reports',dest)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.')
    ap.add_argument('--stage',required=True,choices=('selftest','check','diagnose'))
    args=ap.parse_args();root=Path(args.project).resolve()
    if args.stage=='selftest':return selftest()
    if args.stage=='check':return validate_source(root)
    return analyze(root)
if __name__=='__main__':main()
