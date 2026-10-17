#!/usr/bin/env python3
"""Read-only B9A11-A loss utilities.

All forces in N, displacements in mm; normalized metrics are dimensionless.
The interface term below is a DIAGNOSTIC using oracle FEM force, NOT a
production boundary/coupling loss. Never use FEM labels as physics inputs.
"""
import torch


def norm(x):
    return torch.linalg.vector_norm(x.reshape(-1))


def force_metrics(f_pred, f_ref, mask, *, floor=1e-10):
    """Return normalized magnitudes, separating FE-operator floor from NO excess."""
    if f_pred.shape != f_ref.shape or f_pred.ndim != 2 or f_pred.shape[1] != 3:
        raise ValueError('force arrays must have matching [nodes,3] shape')
    if mask.ndim != 1 or len(mask) != len(f_ref):
        raise ValueError('bad interface mask')
    if mask.dtype != torch.bool or not mask.any() or mask.all():
        raise ValueError('mask must contain prescribed and free nodes')
    denom=norm(f_ref[mask])
    if not bool(denom > floor):
        raise RuntimeError('Reference interface-force norm is zero; cannot normalize')
    result = {
        'ref_interface_norm_N': denom,
        'ref_free_pct': 100*norm(f_ref[~mask])/denom,
        'pred_free_pct': 100*norm(f_pred[~mask])/denom,
        'pred_minus_ref_free_pct': 100*norm(f_pred[~mask]-f_ref[~mask])/denom,
        'pred_minus_ref_interface_pct': 100*norm(f_pred[mask]-f_ref[mask])/denom,
        'pred_minus_ref_all_pct': 100*norm(f_pred-f_ref)/denom,
    }
    return result


def normalized_objectives(u_pred,u_ref,f_pred,f_ref,mask):
    fm=force_metrics(f_pred,f_ref,mask)
    den=fm['ref_interface_norm_N']
    # L2 relative norms squared. Unlike raw force in N, these are dimensionless.
    displacement_scale=torch.clamp(norm(u_ref[~mask]),min=1e-12)
    data=((u_pred[~mask]-u_ref[~mask]).square().sum()/displacement_scale.square())
    eq=f_pred[~mask].square().sum()/den.square()
    traction=(f_pred[mask]-f_ref[mask]).square().sum()/den.square()
    return {'data':data,'eq':eq,'interface_oracle_diagnostic':traction}


def gradient_characteristics(grad_data,grad_eq,grad_if):
    gd,ge,gi=[norm(x).detach() for x in (grad_data,grad_eq,grad_if)]
    cos=(grad_data.detach()*grad_eq.detach()).sum()/torch.clamp(gd*ge,min=1e-30)
    return {
        'grad_data_norm':float(gd),
        'grad_eq_norm':float(ge),
        'grad_interface_diagnostic_norm':float(gi),
        'grad_eq_over_data':float(ge/torch.clamp(gd,min=1e-30)),
        'grad_cos_data_eq':float(cos),
        'all_gradients_finite':bool(all(torch.isfinite(x).all().item() for x in (grad_data,grad_eq,grad_if)))
    }


def selftest():
    dtype=torch.float64
    ref=torch.tensor([[1.,2.,3.],[0.,0.,0.],[1.,0.,0.]],dtype=dtype)
    pred=ref+0.01
    mask=torch.tensor([True,False,True])
    f=force_metrics(pred,ref,mask)
    assert abs(float(f['pred_minus_ref_free_pct'])-100*(3**.5*.01)/(15**.5))<1e-10
    assert abs(float(f['ref_free_pct']))<1e-12
    c=torch.tensor([.04,-.06],dtype=dtype,requires_grad=True)
    p=ref+c[0]*torch.ones_like(ref)
    ff=pred+c[1]*torch.ones_like(pred)
    loss=normalized_objectives(p,ref,ff,ref,mask)
    gs=[torch.autograd.grad(v,c,retain_graph=True)[0] for v in loss.values()]
    assert gradient_characteristics(*gs)['all_gradients_finite']
    print('B9A11A LOSSES SELFTEST PASS: dimensions, scaling, differentiability')

if __name__=='__main__':selftest()
