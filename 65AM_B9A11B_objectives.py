#!/usr/bin/env python3
"""B9A11-B: small POD correction with floor-aware equilibrium loss.
All calculations in mm / MPa / N. No oracle interface force in training loss.
"""
import torch
from torch import nn

class PODResidualAdapter(nn.Module):
    def __init__(self, n_features, rank, hidden=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features,hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,rank))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, features, coefficient_std):
        # Bounded, dimensionally consistent POD coefficient correction.
        return 0.05 * coefficient_std * torch.tanh(self.net(features))

def normalized_data_loss(u,uref,free):
    den=torch.linalg.vector_norm(uref[free].flatten()).clamp_min(1e-12)
    return ((u[free]-uref[free]).square().sum()) / den.square()

def floor_aware_equilibrium(free_force,interface_norm_N,floor_frac,margin=1.25):
    ratio = torch.linalg.vector_norm(free_force.flatten())/interface_norm_N
    return torch.relu(ratio - margin*floor_frac).square()

def gradient_scale(data_grad, physics_grad, fraction=0.20):
    gd=torch.linalg.vector_norm(data_grad.flatten()).item()
    gp=torch.linalg.vector_norm(physics_grad.flatten()).item()
    return gd,gp,fraction*gd/max(gp,1e-30)

def selftest():
    torch.set_default_dtype(torch.float64)
    a=PODResidualAdapter(7,3)
    feat=torch.randn(7)
    s=torch.ones(3)
    assert torch.allclose(a(feat,s),torch.zeros(3),atol=1e-13)
    x=torch.randn((8,3),requires_grad=True)
    truth=x.detach()+0.2
    free=torch.tensor([True,True,True,False,False,False,True,True])
    dl=normalized_data_loss(x,truth,free)
    el=floor_aware_equilibrium(x[free].flatten(),torch.tensor(40.),torch.tensor(.05))
    g=torch.autograd.grad(dl+el,x)[0]
    assert torch.isfinite(g).all()
    # Floor remains inactive when predicted reference residual is at the floor.
    assert float(floor_aware_equilibrium(torch.tensor([.1]),torch.tensor(10.),torch.tensor(.01)))==0
    print('B9A11B OBJECTIVES SELFTEST PASS')
if __name__=='__main__':selftest()
