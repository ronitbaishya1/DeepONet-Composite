#!/usr/bin/env python3
"""B9A10-M objective incremental orthotropic C3D8R candidate.

No Abaqus S/LE/orientation enters prediction. Inputs are displacement HISTORY,
reference mesh, engineering constants, and frozen B9A10-K hourglass proxy.

The update is a transparent research approximation; it is NOT claimed to be
Abaqus/Standard's exact internal corotational constitutive integration.
Units mm, MPa, N. Model's material axes initially align with reference axes.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch


def load_l_module(path):
    spec=importlib.util.spec_from_file_location('b9a10m_l_dependency',str(path))
    if spec is None or spec.loader is None:raise FileNotFoundError(str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class IncrementalC3D8R(torch.nn.Module):
    """Incremental corotational small-step log-strain update in material frame.

    Scheme START: dE_mat = R_(n-1)^T logU(F_n F_(n-1)^-1) R_(n-1).
    Scheme END: use current relative polar-rotated strain and R_n.
    S_local(n) = S_local(n-1) + C : dE_mat.
    sigma_global(n) = R_n S_local(n) R_n^T (Cauchy candidate).
    No step from ODB S/LE, and K beta does not change with history.
    """
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.L=load_l_module(Path(__file__).resolve().parent/'65AM_B9A10L_torch_force.py')

    def fields(self,Uhist,material,scheme='start'):
        if scheme not in ('start','end'):raise ValueError('scheme must be start or end')
        if Uhist.ndim!=3 or Uhist.shape[1:]!=(self.base.n_nodes,3) or len(Uhist)<3:
            raise ValueError('U history must be [at least 3 frames,n_nodes,3]')
        if not torch.isfinite(Uhist).all():raise ValueError('nonfinite history')
        if torch.linalg.norm(Uhist[0])>1e-9:raise ValueError('History must start at zero reference state')
        if Uhist.device!=self.base.grad.device:raise ValueError('Mismatched device')
        C=self.L.compliance_stiffness(material)
        I=torch.eye(3,dtype=Uhist.dtype,device=Uhist.device)[None,:,:]
        G=self.base.grad
        conn=self.base.hg.connectivity
        Fhist=I+torch.einsum('teai,eaj->teij',Uhist[:,conn],G)
        if not bool((torch.linalg.det(Fhist)>1e-9).all()):raise RuntimeError('Element inversion in history')
        R=self.L.polar_newton(Fhist)
        accumulated=torch.zeros((G.shape[0],6),dtype=Uhist.dtype,device=Uhist.device)
        Sh=[]
        for n in range(1,len(Uhist)):
            relative=Fhist[n]@torch.linalg.inv(Fhist[n-1])
            incr=self.L.hencky_ref_from_F(relative)
            if scheme=='start':
                transformed=R[n-1].transpose(-1,-2)@incr@R[n-1]
            else:
                Rrel=self.L.polar_newton(relative)
                rotated=Rrel@incr@Rrel.transpose(-1,-2)
                transformed=R[n].transpose(-1,-2)@rotated@R[n]
            eps=self.L.eng_from_tensor(transformed)
            accumulated=accumulated+torch.einsum('ij,ej->ei',C,eps)
            Sh.append(accumulated)
        S_history=torch.stack(Sh,dim=0)
        # For final constitutive response, use current corotating material axes.
        S_global=R[-1]@self.L.tensor_from_stress(accumulated)@R[-1].transpose(-1,-2)
        det=torch.linalg.det(Fhist[-1])
        P=det[:,None,None]*(S_global@torch.linalg.inv(Fhist[-1]).transpose(-1,-2))
        return dict(S_local=accumulated,S_history=S_history,S_global=S_global,P=P,
                    F=Fhist[-1],R=R[-1],J=det)

    def parts(self,Uhist,material,scheme='start'):
        flds=self.fields(Uhist,material,scheme)
        local=torch.einsum('eij,eaj->eai',flds['P'],self.base.grad)*self.base.volume[:,None,None]
        fs=torch.zeros_like(Uhist[-1]).index_add(0,self.base.hg.connectivity.reshape(-1),local.reshape(-1,3))
        fh=self.base.hg(Uhist[-1],material[2])
        return fs,fh,flds

    def forward(self,Uhist,material,scheme='start'):
        a,b,_=self.parts(Uhist,material,scheme)
        return a+b


def selftest():
    L=load_l_module(Path(__file__).resolve().parent/'65AM_B9A10L_torch_force.py')
    X=np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0],
                [0,0,3],[2,0,3],[2,1,3],[0,1,3]],float)
    base=L.ConstitutiveC3D8R(np.arange(1,9),X,np.arange(1,9).reshape(1,8),
          0.004381715594969032,Path(__file__).resolve().parent/'65AM_B9A10K_hourglass_operator.py')
    mod=IncrementalC3D8R(base)
    mat=torch.tensor([46000.,12500.,4750.],dtype=torch.float64,requires_grad=True)
    X_t=torch.tensor(X,dtype=torch.float64)
    tiny=torch.zeros((4,8,3),dtype=torch.float64)
    # Rigid rotations in 3 successive increments create no stress/force.
    for j,theta in enumerate((0.,.08,.15,.23)):
        c=np.cos(theta);s=np.sin(theta)
        Q=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=torch.float64)
        tiny[j]=X_t@(Q.T-torch.eye(3))
    for scheme in ('start','end'):
        f=mod.fields(tiny,mat,scheme)
        if torch.linalg.norm(f['S_local'])>1e-7:
            raise RuntimeError('Rigid rotation creates incremental stress '+scheme)
    # Proportional small affine deformation should approximate total-strain law.
    affine=torch.stack([X_t@torch.tensor([[.002*j,0,0],[0,-.001*j,0],[0,0,.0003*j]],dtype=torch.float64)
                        for j in (0,.25,.5,1.)])
    for scheme in ('start','end'):
        got=mod.fields(affine,mat,scheme)['S_local']
        expected=base.fields(affine[-1],mat)['S_local']
        if float((torch.linalg.norm(got-expected)/torch.linalg.norm(expected)).detach())>0.005:
            raise RuntimeError('Small-deformation proportional test failed '+scheme)
    # Autograd through history and material; finite differences of scalar loss.
    rng=np.random.default_rng(123)
    uh=affine+torch.as_tensor(rng.normal(0,.001,(4,8,3)),dtype=torch.float64)
    uh[0]=0.
    uh=uh.clone().requires_grad_(True)
    forces=mod(uh,mat,'start');loss=(forces**2).sum()
    gu,gm=torch.autograd.grad(loss,(uh,mat))
    if not bool(torch.isfinite(gu).all() and torch.isfinite(gm).all() and torch.linalg.norm(gu)>0):
        raise RuntimeError('Autograd invalid')
    perturb=torch.zeros_like(uh);perturb[-1]=torch.as_tensor(rng.normal(size=(8,3)),dtype=torch.float64)
    eps=1e-6
    fd=((mod(uh+eps*perturb,mat,'start')**2).sum()-(mod(uh-eps*perturb,mat,'start')**2).sum())/(2*eps)
    exact=(gu*perturb).sum()
    if not torch.allclose(fd,exact,rtol=5e-4,atol=5e-3):
        raise RuntimeError('Finite-difference gradient mismatch')
    if torch.linalg.norm(forces.sum(dim=0))>1e-8:raise RuntimeError('Internal force resultant nonzero')
    print('B9A10M TORCH SELFTEST PASS | objective rigid rotations | proportional affine path | autograd')

if __name__=='__main__':selftest()
