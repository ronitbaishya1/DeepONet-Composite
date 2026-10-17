#!/usr/bin/env python3
"""B9A10-O Mac read-only nonlinear orthotropic C3D8 constitutive-path audit.

This is a stress-path experiment, not a new B5 training or full FE force operator.
Does not fit/modify orthotropic elastic constants or existing K beta.
Uses predicted stresses from U + input material constants; ODB S/LE/Q only for comparison.
  python3 65AM_B9A10O_mac_path_audit.py --project . --stage selftest
  python3 65AM_B9A10O_mac_path_audit.py --project . --stage check
  python3 65AM_B9A10O_mac_path_audit.py --project . --stage diagnose
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
T=torch.float64
OUT=Path('results/65AM_B9_coupled_feasibility/B9A10O_constitutive_path_probe')
CODES=[('O00_ST','O00','ST'),('O00_SH','O00','SH'),('O00_SR','O00','SR'),
       ('O00_RS','O00','RS'),('O00_HR','O00','HR'),('O00_RH','O00','RH'),
       ('O17_SR','O17','SR'),('O17_RS','O17','RS'),('ISO_SR','ISO','SR'),('ISO_RS','ISO','RS')]
BASE=dict(E1=46500.,E2=12500.,E3=12000.,nu12=.28,nu13=.28,nu23=.40,G12=4750.,G13=4500.,G23=3500.)
C_NAMES=['S11','S22','S33','S12','S13','S23']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def csv_out(p,rows):
    if not rows:raise ValueError('No data for '+str(p))
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)

def pct(a,b):return 100.*float(np.linalg.norm(a))/max(float(np.linalg.norm(b)),1e-12)

def mat_parameters(code):
    if code=='ISO':return None
    return BASE

def Q0(code):
    angle=np.deg2rad(17. if code=='O17' else 0.)
    a,b=np.cos(angle),np.sin(angle)
    return np.array([[a,-b,0],[b,a,0],[0,0,1]],float)

def stiffness(mat, dtype=T,device='cpu'):
    if mat is None:
        E=20000.; nu=.25; G=E/(2*(1+nu))
        c=torch.zeros((6,6),dtype=dtype,device=device)
        lmb=E*nu/((1+nu)*(1-2*nu))
        mu=G
        for i in range(3):
            for j in range(3):c[i,j]=lmb
            c[i,i]+=2*mu
        for i in range(3,6):c[i,i]=mu
        return c
    p=mat;E1,E2,E3=p['E1'],p['E2'],p['E3']
    c=torch.zeros((6,6),dtype=dtype,device=device)
    # Compliance defined by elastic reciprocity, consistent with B9A10-L/N.
    c[0,0]=1/E1;c[1,1]=1/E2;c[2,2]=1/E3
    c[0,1]=c[1,0]=-p['nu12']/E1
    c[0,2]=c[2,0]=-p['nu13']/E1
    c[1,2]=c[2,1]=-p['nu23']/E2
    c[3,3]=1/p['G12'];c[4,4]=1/p['G13'];c[5,5]=1/p['G23']
    if not bool((torch.linalg.eigvalsh(c)>0).all()):raise ValueError('Invalid orthotropic constants')
    return torch.linalg.inv(c)

def eng(t):
    return torch.stack((t[...,0,0],t[...,1,1],t[...,2,2],
                       2*t[...,0,1],2*t[...,0,2],2*t[...,1,2]),dim=-1)

def stress_tensor(s):
    a,b,c,d,e,f=s.unbind(-1)
    return torch.stack((torch.stack((a,d,e),dim=-1),
                        torch.stack((d,b,f),dim=-1),
                        torch.stack((e,f,c),dim=-1)),dim=-2)

def strain_hencky(F):
    # analytic matrix logarithm near identity; differentiable at repeated eigenvalues.
    I=torch.eye(3,dtype=F.dtype,device=F.device)
    C=F.transpose(-1,-2)@F
    Y=torch.linalg.solve(C+I,C-I)
    Y=.5*(Y+Y.transpose(-1,-2))
    if torch.linalg.matrix_norm(Y).detach().item()>.45:
        raise ValueError('Outside small-deformation validation domain')
    Y2=Y@Y; z=Y; result=Y
    for k in range(1,22):
        z=z@Y2;result=result+z/(2*k+1)
    return .5*(result+result.transpose(-1,-2))

def polar(F):
    R=F
    for _ in range(9):R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
    return R

def predicted(F,code):
    """Uses only deformation F and the *known* reference material orientation.
    Returns predicted global Cauchy stress, predicted engineering Hencky strain
    in rotating material system, and rotation matrix.
    """
    r=polar(F);q=torch.tensor(Q0(code),dtype=F.dtype,device=F.device)
    e= q.T@strain_hencky(F)@q
    stiff=stiffness(mat_parameters(code),F.dtype,F.device)
    sloc=stiff@eng(e)
    sigma=r@q@stress_tensor(sloc)@q.T@r.T
    return sigma,eng(e),r,sloc

def F_from_U(X,U):
    Xc=X-X.mean(axis=0)
    # For a single affine element, least-squares gradient must be exact.
    grad=np.linalg.lstsq(Xc,U,rcond=None)[0].T
    fit=Xc@grad.T + U.mean(axis=0)
    resid=float(np.max(np.abs(U-fit)))
    return np.eye(3)+grad,resid

def t6_np(s):
    a,b,c,d,e,f=s
    return np.array([[a,d,e],[d,b,f],[e,f,c]],float)

def valid(root):
    files=root/'B9A10O_exports'
    p=files/'B9A10O_stress_paths.npz';m=files/'B9A10O_windows_summary.json'
    if not p.is_file() or not m.is_file():
        raise FileNotFoundError('Copy the entire B9A10O_exports folder from Windows into project root')
    manifest=json.loads(m.read_text())
    if int(manifest['count'])!=10 or sha(p)!=manifest['npz_sha256']:
        raise RuntimeError('B9A10O Windows export/manifest mismatch')
    cases={r['job']:r for r in manifest['cases']}
    if set(cases)!=set('B9A10O_'+s[0] for s in CODES):
        raise RuntimeError('Unexpected case matrix')
    with np.load(p,allow_pickle=False) as z:
        for label,material,code in CODES:
            key='B9A10O_'+label
            arrays={n:z[key+'__'+n] for n in ('U','RF','S','LE','Q','segment','t','X','finalF')}
            nt=len(arrays['t'])
            if arrays['U'].shape!=(nt,8,3) or arrays['S'].shape!=(nt,8,6) or arrays['LE'].shape!=(nt,8,6) or arrays['Q'].shape!=(nt,8,3,3):
                raise RuntimeError('Invalid history array dimensions for '+key)
            if not np.isfinite(arrays['S']).all() or not np.isfinite(arrays['Q']).all():
                raise RuntimeError('Nonfinite Abaqus field '+key)
            if nt<3 or len(np.unique(arrays['segment'])) != (1 if code in ('ST','SH') else 2):
                raise RuntimeError('Insufficient saved loading history for '+key)
            F,end_err=F_from_U(arrays['X'],arrays['U'][-1])
            if end_err>2e-5 or np.linalg.norm(F-arrays['finalF'])>5e-5:
                raise RuntimeError('Final deformation mismatch '+key)
    print('B9A10O MAC CHECK PASS | 10 original Abaqus C3D8 stress histories | provenance SHA verified')
    return p,manifest

def selftest():
    z=torch.zeros((3,3),dtype=T,requires_grad=True)
    I=torch.eye(3,dtype=T)
    c=math.cos(.3);ss=math.sin(.3);R=torch.tensor([[c,-ss,0],[ss,c,0],[0,0,1]],dtype=T)
    for code in ('O00','O17','ISO'):
        s,e,_,_=predicted(I,code)
        if s.abs().max().item()>1e-10:raise AssertionError('Identity must create no stress')
        s,e,_,_=predicted(R,code)
        if s.abs().max().item()>1e-8:raise AssertionError('Rigid rotation must create no stress')
    F=I+torch.tensor([[.008,.006,0],[0,-.003,.001],[0,0,.002]],dtype=T)
    sigma,_,_,_=predicted(F,'O00')
    if not torch.isfinite(sigma).all():raise AssertionError('Stress failed')
    trial=(I+z@torch.tensor([[1.,.2,0],[0,1.,0],[0,0,1.]],dtype=T))@F
    loss=predicted(trial,'O17')[0].square().sum()
    g=torch.autograd.grad(loss,z)[0]
    if not torch.isfinite(g).all() or g.norm().item()<1e-10:raise AssertionError('autograd failure')
    F0=np.eye(3)
    X=np.array([[0,0,0],[.6,0,0],[.6,.5,0],[0,.5,0],[0,0,.62],[.6,0,.62],[.6,.5,.62],[0,.5,.62]])
    U=X@(F.detach().numpy()-F0).T
    FF,err=F_from_U(X,U)
    np.testing.assert_allclose(FF,F.detach().numpy(),atol=1e-13)
    print('B9A10O MAC SELFTEST PASS | objectivity | elastic 9 constants | affine recovery | torch gradients')

def analyze(root):
    data,meta=valid(root)
    dest=root/OUT
    if not dest.is_dir():raise RuntimeError('Create the empty result folder in Finder first: '+str(dest))
    names=['B9A10O_cases.csv','B9A10O_frames.csv','B9A10O_same_final_F.csv',
           'B9A10O_summary.json','B9A10O_stress_paths.png','B9A10O_same_final_F.png',
           'B9A10O_gradients.json']
    if any((dest/n).exists() for n in names):
        raise RuntimeError('Existing B9A10O reports protected; move previous reports before rerun')
    frames=[];cases=[];pred_final={};actual_final={};results=[]
    with np.load(data,allow_pickle=False) as z:
        for key,mat,route in CODES:
            job='B9A10O_'+key
            X=z[job+'__X'];U=z[job+'__U'];Sa=z[job+'__S'];LE=z[job+'__LE'];Q=z[job+'__Q'];seg=z[job+'__segment'];tt=z[job+'__t']
            originalS=np.linalg.norm(Sa[-1])
            if originalS<=1e-6:raise RuntimeError('Zero final reference stress '+job)
            sref_last=[];pred_last=[];strain_last=[];oracle_last=[]
            for f in range(len(tt)):
                F,affine_gap=F_from_U(X,U[f])
                if affine_gap>2e-5:raise RuntimeError('Non-affine displacement in single element '+job)
                if np.linalg.det(F)<0.5:raise RuntimeError('Inverted element '+job)
                with torch.no_grad():
                    pr,eps,r,local = predicted(torch.tensor(F,dtype=T),mat)
                    pr=pr.cpu().numpy();eps=eps.cpu().numpy();r=r.cpu().numpy()
                    stiff=stiffness(mat_parameters(mat)).cpu().numpy()
                observed_global=np.stack([Q[f,i].T@t6_np(Sa[f,i])@Q[f,i] for i in range(8)])
                pred_global=np.broadcast_to(pr,(8,3,3))
                err_final_norm=100*np.linalg.norm(pred_global-observed_global)/originalS
                le_obs=np.mean(LE[f],axis=0)
                le_gap=pct(eps-le_obs,LE[-1])
                oracle_pred=LE[f]@stiff.T
                constitutive=pct(oracle_pred-Sa[f],Sa[-1])
                predicted_Q=(r@Q0(mat)).T
                qdist=(None if mat=='ISO' else min(np.linalg.norm(Q[f,0]-predicted_Q),np.linalg.norm(Q[f,0].T-predicted_Q)))
                frames.append(dict(job=job,material=mat,path=route,frame=f,segment=int(seg[f]),time=float(tt[f]),
                    error_stress_pct_of_final=err_final_norm,
                    error_strain_pct_of_final=le_gap,
                    error_hooke_using_abaqus_LE_pct_of_final=constitutive,
                    orientation_difference_fro=qdist,
                    affine_gradient_fit_mm=affine_gap,
                    deformation_J=float(np.linalg.det(F))))
                if f==len(tt)-1:
                    sref_last=observed_global;pred_last=pred_global
                    strain_last=le_gap;oracle_last=constitutive
            final_stress=pct(pred_last-sref_last,sref_last)
            cases.append(dict(job=job,material=mat,path=route,nframes=len(tt),
                final_stress_error_pct=final_stress,
                final_strain_error_pct=strain_last,
                hooke_on_abaqus_LE_error_pct=oracle_last,
                frame_max_abs_stress_error_pct_final_scale=max(row['error_stress_pct_of_final'] for row in frames if row['job']==job)))
            pred_final[key]=pred_last.copy();actual_final[key]=sref_last.copy()
            print('CASE',job,'final stress %.5f%%'%final_stress,'hooke-from-LE %.5f%%'%oracle_last)
    for left,right in [('O00_SR','O00_RS'),('O00_HR','O00_RH'),('O17_SR','O17_RS'),('ISO_SR','ISO_RS')]:
        gap=pct(actual_final[left]-actual_final[right],actual_final[left]);
        predgap=pct(pred_final[left]-pred_final[right],pred_final[left])
        results.append(dict(pair=left+'__vs__'+right,reference_stress_path_difference_pct=gap,
                 predicted_total_strain_path_difference_pct=predgap,final_F_equal_by_design=True))
    # Diagnostics are on known controls; no constitutive selection or calibration here.
    groups={}
    for mat in ('O00','O17','ISO'):
        cc=[c for c in cases if c['material']==mat]
        groups[mat]=dict(n=len(cc),mean_final_stress_error_pct=float(np.mean([c['final_stress_error_pct'] for c in cc])),
            max_final_stress_error_pct=float(max(c['final_stress_error_pct'] for c in cc)))
    summary={'step':'65AM-B9A10-O','experiment':'Fully prescribed affine single-element C3D8 nonlinear loading path; ten new controls',
        'cases':len(cases),'material_constants':BASE,'isotropic_control_E_MPa':20000.,'isotropic_nu':.25,
        'predicted_total_strain_Hencky_without_observed_stress':True,
        'stored_S_LE_Q_only_used_as_reference':True,
        'group_metrics':groups,'same_final_F_pairs':results,
        'decision':'Diagnostic only: evaluate stress-frame/path dependence before any new constitutive law',
        'B5_Broyden_65V_unchanged':True,'physics_loss_ready':False}
    csv_out(dest/names[0],cases);csv_out(dest/names[1],frames);csv_out(dest/names[2],results)
    (dest/names[3]).write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(9,4.8))
    for key in ('O00_ST','O00_SH','O00_SR','O00_RS','O17_SR','O17_RS'):
        rr=[r for r in frames if r['job']=='B9A10O_'+key]
        ax.plot(np.arange(len(rr)),[r['error_stress_pct_of_final'] for r in rr],marker='o',ms=2,label=key)
    ax.set(xlabel='Saved frame index (successive steps)',ylabel='Stress error / final Abaqus stress norm (%)',
           title='B9A10-O: constitutive stress error during loading');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(dest/names[4],dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    xx=np.arange(len(results))
    ax.bar(xx-.18,[r['reference_stress_path_difference_pct'] for r in results],.36,label='Abaqus: difference between paths')
    ax.bar(xx+.18,[r['predicted_total_strain_path_difference_pct'] for r in results],.36,label='Total-strain model: difference')
    ax.set_xticks(xx,[r['pair'].replace('__vs__',' vs ') for r in results],rotation=20,ha='right')
    ax.set(ylabel='Final stress path difference (%)',title='Same final F, different loading paths')
    ax.legend(fontsize=8);ax.grid(alpha=.25,axis='y');fig.tight_layout();fig.savefig(dest/names[5],dpi=165);plt.close(fig)
    zz=torch.eye(3,dtype=T,requires_grad=True)
    sigma,_,_,_=predicted(zz + torch.tensor([[.005,.01,0],[0,-.003,0],[0,0,.002]],dtype=T),'O17')
    test=sigma.square().sum();gradient=torch.autograd.grad(test,zz)[0]
    gd={'autograd_pass':bool(torch.isfinite(gradient).all()),'gradient_norm_F':float(gradient.norm().item()),
        'torch_dtype':'float64','scope':'single-element constitutive stress, not full B5 training'}
    (dest/names[6]).write_text(json.dumps(gd,indent=2,allow_nan=False))
    print('B9A10O COMPLETE | 10 independent runs | output',dest)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='.')
    parser.add_argument('--stage',choices=('selftest','check','diagnose'),required=True)
    args=parser.parse_args();root=Path(args.project).resolve()
    if args.stage=='selftest':return selftest()
    if args.stage=='check':return valid(root)
    return analyze(root)
if __name__=='__main__':main()
