#!/usr/bin/env python3
"""B9A10-L: candidate differentiable orthotropic C3D8R force operator.

mm, MPa, N. Uses U + undeformed XYZ + constitutive constants, never ODB S/LE/Q
for predictions. C3D8R stress integration plus the FROZEN B9A10-K hourglass proxy.

WARNING: finite-deformation Abaqus/Standard orthotropic elastic constitutive
update is incremental. Hencky-total-strain / corotational Hooke is a candidate
approximation, NOT an exact Abaqus law. Validate before training B5.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn

SIGNS = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)


def read_hourglass_module(path):
    spec=importlib.util.spec_from_file_location('b9a10k_existing',str(path))
    if spec is None or spec.loader is None: raise FileNotFoundError(str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def compliance_stiffness(params):
    """Abaqus TYPE=ENGINEERING CONSTANTS, engineering shear order 12,13,23."""
    e1,e2,g12=params.unbind()
    base=torch.stack([e1,e2,g12]);
    if not bool(torch.all(torch.isfinite(base) & (base>0))):raise ValueError('Bad material constants')
    e3=e1.new_tensor(12000.); g13=e1.new_tensor(4500.);g23=e1.new_tensor(3500.)
    nu12=.28;nu13=.28;nu23=.40
    z=e1.new_zeros(())
    compliance=torch.stack([
        torch.stack((1/e1,-nu12/e1,-nu13/e1,z,z,z)),
        torch.stack((-nu12/e1,1/e2,-nu23/e2,z,z,z)),
        torch.stack((-nu13/e1,-nu23/e2,1/e3,z,z,z)),
        torch.stack((z,z,z,1/g12,z,z)),
        torch.stack((z,z,z,z,1/g13,z)),
        torch.stack((z,z,z,z,z,1/g23)),
    ])
    # Positive definite is necessary but not sufficient for an accurate constitutive model.
    if not bool((torch.linalg.eigvalsh(compliance)>0).all()):
        raise ValueError('Engineering-constant compliance is not positive definite')
    return torch.linalg.inv(compliance)


def eng_from_tensor(e):
    return torch.stack((e[...,0,0],e[...,1,1],e[...,2,2],
                        2*e[...,0,1],2*e[...,0,2],2*e[...,1,2]),dim=-1)


def tensor_from_stress(s):
    a,b,c,d,e,f=s.unbind(-1)
    return torch.stack((torch.stack((a,d,e),dim=-1),
                        torch.stack((d,b,f),dim=-1),
                        torch.stack((e,f,c),dim=-1)),dim=-2)


def tensor6_strain(e):
    return torch.stack((e[...,0,0],e[...,1,1],e[...,2,2],
                        e[...,0,1],e[...,0,2],e[...,1,2]),dim=-1)


def polar_newton(F,steps=8):
    """Differentiable polar factor, robust near repeated principal stretches."""
    R=F
    for _ in range(steps):
        R=.5*(R+torch.linalg.inv(R).transpose(-1,-2))
    return R


def hencky_ref_from_F(F,order=20):
    """0.5*log(F.T F) using convergent matrix-atanh series; autograd stable at I.

    Y=(C-I)(C+I)^-1, 0.5 log(C)=sum_{k>=0}Y^(2k+1)/(2k+1).
    Restrict ||Y||_2 <=0.45, a conservative small-strain domain.
    """
    C=F.transpose(-1,-2)@F
    I=torch.eye(3,dtype=F.dtype,device=F.device).expand_as(C)
    Y=torch.linalg.solve(C+I,(C-I)).transpose(-1,-2) # C and I commute; symmetric
    Y=.5*(Y+Y.transpose(-1,-2))
    with torch.no_grad():
        if not bool((torch.linalg.eigvalsh(Y).abs().amax(dim=-1)<=.45).all()):
            raise ValueError('Strain exceeds tested logarithm-series domain; STOP')
    Y2=Y@Y; term=Y; E=term
    for k in range(1,order):
        term=term@Y2
        E=E+term/(2*k+1)
    return .5*(E+E.transpose(-1,-2))


class ConstitutiveC3D8R(nn.Module):
    """Independent U->S and U->forces candidate; J/Houglass depends only on X.

    torch inputs: u[n,3], material[E1,E2,G12]. All constants in MPa.
    Reference material orientation is identity (B9A9 INPUT MAT_ORI).
    No Abaqus Q/S/LE arrays in this class.
    """
    def __init__(self,node_labels,xyz,connectivity,beta,hg_path,device='cpu',dtype=torch.float64):
        super().__init__()
        hk=read_hourglass_module(hg_path)
        self.hg=hk.HourglassProxy(node_labels,xyz,connectivity,beta,device=device,dtype=dtype)
        self.n_nodes=len(node_labels)
        conn=self.hg.connectivity
        X=torch.as_tensor(xyz,dtype=dtype,device=device)[conn]
        dnat=torch.as_tensor(SIGNS/8.,dtype=dtype,device=device)
        jac=torch.einsum('eai,aj->eij',X,dnat)
        det=torch.linalg.det(jac)
        if not bool((det>1e-12).all()):raise ValueError('Invalid reference C3D8R element')
        gradients=torch.einsum('aj,ejk->eak',dnat,torch.linalg.inv(jac))
        self.register_buffer('grad',gradients)
        self.register_buffer('volume',8*det)

    def fields(self,u,material,stress_kind='cauchy'):
        if u.shape!=(self.n_nodes,3):raise ValueError('U must be nodes x 3')
        if stress_kind not in ('cauchy','kirchhoff'):raise ValueError('Invalid stress candidate')
        F=torch.eye(3,dtype=u.dtype,device=u.device)[None,:,:]+torch.einsum(
           'eai,eaj->eij',u[self.hg.connectivity],self.grad)
        detF=torch.linalg.det(F)
        if not bool((detF>0).all()):raise ValueError('Inverted element')
        R=polar_newton(F)
        E=hencky_ref_from_F(F)
        Cmat=compliance_stiffness(material)
        sloc=torch.einsum('ij,ej->ei',Cmat,eng_from_tensor(E))
        T=tensor_from_stress(sloc)
        Sglob=R@T@R.transpose(-1,-2)
        if stress_kind=='kirchhoff': Sglob=Sglob/detF[:,None,None]
        P=detF[:,None,None]*(Sglob@torch.linalg.inv(F).transpose(-1,-2))
        return {'F':F,'J':detF,'R':R,'hencky_local':E,
                'S_local':sloc,'S_global':Sglob,'P':P}

    def parts(self,u,material,stress_kind='cauchy'):
        flds=self.fields(u,material,stress_kind)
        local=(flds['P']@self.grad.transpose(-1,-2)).transpose(-1,-2)*self.volume[:,None,None]
        fs=torch.zeros_like(u).index_add(0,self.hg.connectivity.reshape(-1),local.reshape(-1,3))
        fh=self.hg(u,material[2]);return fs,fh,flds

    def forward(self,u,material,stress_kind='cauchy'):
        a,b,_=self.parts(u,material,stress_kind)
        return a+b


def selftest():
    root=Path(__file__).resolve().parent
    h=root/'65AM_B9A10K_hourglass_operator.py'
    if not h.is_file():raise RuntimeError('Keep existing B9A10K operator beside L code')
    X=np.asarray([[0,0,0],[2,0,0],[2,1,0],[0,1,0],
                  [0,0,3],[2,0,3],[2,1,3],[0,1,3]],float)
    labels=np.arange(1,9)
    model=ConstitutiveC3D8R(labels,X,labels.reshape(1,8),.0043817,h)
    m=torch.tensor([46500.,12500.,4750.],dtype=torch.float64,requires_grad=True)
    z=torch.zeros((8,3),dtype=torch.float64,requires_grad=True)
    ff=model(z,m)
    if not torch.allclose(ff,torch.zeros_like(ff),atol=1e-10):raise RuntimeError('Zero-U force test failed')
    u=torch.as_tensor(X,dtype=torch.float64)@torch.tensor([[.001,.0002,0],[.0003,-.002,0],[0,0,.001]],dtype=torch.float64)
    fld=model.fields(u,m)
    if not bool((fld['J']>0).all()):raise RuntimeError('Invalid polar strain')
    if torch.max(torch.abs(fld['R'].transpose(-1,-2)@fld['R']-torch.eye(3)))>1e-9:
        raise RuntimeError('Polar rotation not orthogonal')
    if torch.linalg.norm(model.hg(u,m[2]))>1e-9:raise RuntimeError('Affine hourglass null failed')
    # Rigid rotation must not create elastic stress, incl repeated eigenvalue cases.
    angle=.2;c=np.cos(angle);s=np.sin(angle)
    rot=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=torch.float64)
    ur=torch.as_tensor(X,dtype=torch.float64)@(rot.T-torch.eye(3))
    if torch.linalg.norm(model.fields(ur,m)['S_local'])>1e-8:
        raise RuntimeError('Rigid rotation creates stress')
    rng=np.random.default_rng(3);ua=u+torch.tensor(rng.normal(0,.004,(8,3)),dtype=torch.float64)
    ua=ua.detach().requires_grad_(True)
    force=model(ua,m);loss=force.square().sum()
    gU,gM=torch.autograd.grad(loss,(ua,m))
    if not torch.isfinite(gU).all() or not torch.isfinite(gM).all():
        raise RuntimeError('Nonfinite differentiable gradient')
    # directional derivative check of full displacement-dependent force loss
    du=torch.tensor(rng.normal(size=(8,3)),dtype=torch.float64);du=du/torch.linalg.norm(du)
    step=1e-6
    fd=(model(ua+step*du,m).square().sum()-model(ua-step*du,m).square().sum())/(2*step)
    ad=(gU*du).sum()
    if abs(float((fd-ad).detach()))>5e-3 and abs(float((fd-ad).detach()))/max(abs(float(ad.detach())),1e-14)>1e-4:
        raise RuntimeError('Finite difference gradient does not match autograd')
    if torch.linalg.norm(force.sum(dim=0))>1e-8:
        raise RuntimeError('Element internal forces have nonzero resultant')
    print('B9A10L SELFTEST PASS | material stiffness | objective rotation | affine-null HG | differentiability | virtual work')

if __name__=='__main__':selftest()
