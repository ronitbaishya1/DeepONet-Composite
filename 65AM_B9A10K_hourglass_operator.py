#!/usr/bin/env python3
"""65AM-B9A10-K: differentiable, explicitly APPROXIMATE C3D8R hourglass force.

Conventions: mm, MPa=N/mm^2, N. H_e is the orthogonal projector to
{1, X, Y, Z} on the 8 nodes of each element. E_hg = beta/2 * sum_e
G12*V_e^(1/3)*||H_e U_e||_F^2, f_hg=dE/du.

This is a mechanically motivated, energy-positive, affine-free PROXY, not
Abaqus's proprietary C3D8R hourglass law. beta is fitted ONLY using the two
B9A10-J controlled stiffness-sensitivity cases. Do not use in B5 training
or production coupling before held-out verification.
"""
import numpy as np
import torch
from torch import nn

SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float64)


def geometry(node_labels, xyz, connectivity):
    labels=np.asarray(node_labels,dtype=np.int64)
    xyz=np.asarray(xyz,dtype=np.float64)
    conn=np.asarray(connectivity,dtype=np.int64)
    if xyz.shape!=(len(labels),3) or conn.ndim!=2 or conn.shape[1]!=8:
        raise ValueError('Incompatible nodes/coordinates/C3D8R connectivity')
    if len(np.unique(labels))!=len(labels):raise ValueError('Duplicate nodes')
    mapping={int(label):i for i,label in enumerate(labels)}
    try: idx=np.array([[mapping[int(node)] for node in row] for row in conn],dtype=np.int64)
    except KeyError as exc:raise ValueError('Element points to unknown node '+str(exc))
    X=xyz[idx]
    jac=np.einsum('eai,aj->eij',X,SIGNS/8)
    detj=np.linalg.det(jac)
    if np.any(detj<=1e-14):raise ValueError('Nonpositive/degenerate element Jacobian')
    volume=8*detj
    # QR on dimensionless centered geometry, avoiding ill-conditioning caused
    # by translating a mesh far from the origin. Q columns are the affine basis.
    length=volume**(1/3)
    A=np.concatenate([np.ones((len(idx),8,1)),
       (X-X.mean(axis=1,keepdims=True))/length[:,None,None]],axis=2)
    Q,R=np.linalg.qr(A,mode='reduced')
    if np.any(np.abs(np.diagonal(R,axis1=-2,axis2=-1))<1e-9):
        raise ValueError('Degenerate affine basis')
    P=np.eye(8)[None,:,:]-Q@Q.transpose(0,2,1)
    P=.5*(P+P.transpose(0,2,1))
    if np.max(np.abs(P@P-P))>1e-9 or np.max(np.abs(P@A))>1e-9:
        raise RuntimeError('Hourglass projector is not affine-free')
    return idx,P,length


def numpy_template(U,conn,P,length,G12):
    """Unscaled hourglass-force proxy; N for G12 in MPa and U in mm."""
    U=np.asarray(U,dtype=np.float64)
    if U.ndim!=2 or U.shape[1]!=3:raise ValueError('U must be nodes x 3')
    if float(G12)<=0:raise ValueError('G12 must be positive')
    u_elem=U[conn]
    local=np.einsum('eab,ebc->eac',P,u_elem)
    f_elem=G12*length[:,None,None]*local
    force=np.zeros_like(U)
    np.add.at(force,conn,f_elem)
    return force


class HourglassProxy(nn.Module):
    """Torch differentiable f_hg and E_hg with frozen geometry and beta.

    Torch u is [n_nodes,3]; G12 can be a scalar torch tensor requiring grad.
    `forward` uses index_add so autograd propagates from force to displacement.
    """
    def __init__(self,node_labels,X,connectivity,beta,device='cpu',dtype=None):
        super().__init__()
        if dtype is None:dtype=torch.float32 if device=='mps' else torch.float64
        idx,P,length=geometry(node_labels,X,connectivity)
        if not (float(beta)>0):raise ValueError('Energy-positive beta required')
        self.n_nodes=len(node_labels)
        self.register_buffer('connectivity',torch.as_tensor(idx,dtype=torch.long,device=device))
        self.register_buffer('projector',torch.as_tensor(P,dtype=dtype,device=device))
        self.register_buffer('length',torch.as_tensor(length,dtype=dtype,device=device))
        self.register_buffer('beta',torch.as_tensor(float(beta),dtype=dtype,device=device))

    def _local(self,u,G12):
        if u.shape!=(self.n_nodes,3):raise ValueError('Unexpected displacement dimensions')
        G=torch.as_tensor(G12,dtype=u.dtype,device=u.device)
        if torch.any(G<=0):raise ValueError('G12 must be positive')
        pu=torch.matmul(self.projector,u[self.connectivity])
        return pu,self.beta*G*self.length

    def energy(self,u,G12):
        pu,k=self._local(u,G12)
        return .5*torch.sum(k[:,None,None]*pu.square())

    def forward(self,u,G12):
        pu,k=self._local(u,G12)
        local=k[:,None,None]*pu
        return torch.zeros_like(u).index_add(0,self.connectivity.reshape(-1),local.reshape(-1,3))


def selftest():
    # exact affine null on rectangular brick; rigid motion and rotation are
    # included. Compare torch force with energy derivative and finite difference.
    X=np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0],
                [0,0,3],[2,0,3],[2,1,3],[0,1,3]],float)
    lab=np.arange(1,9)
    idx,P,h=geometry(lab,X,lab.reshape(1,8))
    uaff=X@np.array([[.002,.001,0],[.003,0,-.001],[.004,.005,0]])+np.array([.02,-.01,.03])
    if np.linalg.norm(numpy_template(uaff,idx,P,h,1000))>1e-10:
        raise RuntimeError('Affine nullspace failure')
    model=HourglassProxy(lab,X,lab.reshape(1,8),.8,dtype=torch.float64)
    u=torch.tensor(uaff,dtype=torch.float64,requires_grad=True)
    u=u+torch.randn_like(u)*.008
    G=torch.tensor(1120.,dtype=torch.float64,requires_grad=True)
    force=model(u,G)
    energy=model.energy(u,G)
    grad,gg=torch.autograd.grad(energy,(u,G),retain_graph=True)
    if not torch.allclose(force,grad,rtol=1e-10,atol=1e-10):
        raise RuntimeError('Force must equal energy displacement gradient')
    q=torch.randn_like(u);eps=1e-5
    central=(model.energy(u+eps*q,G)-model.energy(u-eps*q,G))/(2*eps)
    virtual=(force*q).sum()
    if not torch.allclose(central,virtual,rtol=2e-7,atol=2e-7):
        raise RuntimeError('Virtual-work finite-difference failure')
    if energy.item()<0 or gg.item()<=0:raise RuntimeError('Nonpositive hourglass energy')
    if torch.linalg.norm(force.sum(dim=0)).item()>1e-9:raise RuntimeError('Unbalanced internal resultant')
    print('B9A10K OPERATOR SELFTEST PASS | affine-free | energy-positive | torch autograd | virtual work')

if __name__=='__main__':selftest()
