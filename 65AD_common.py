#!/usr/bin/env python3
"""65AD: compatible displacement projection for 65AC LGNO gradients.

No training. Retains 65AA displacement at both artificial FE-NO interfaces.
Uses old 64J VALIDATION examples for diagnostics only; NEVER uses the old
64J test or 65W middle full reference. This is not a coupled FEM solve.
"""
from pathlib import Path
import importlib, json
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import torch

ac=importlib.import_module('65AC_common')
ROOT_NAME='65AD_compatible_projection'
SEGMENTS=('inner_left','inner_right')
COMPONENTS=('11','22','33','12','13','23')


def interface_nodes(root,ds):
    """Trust original hard-interface node mapping, verify that both full faces match."""
    ctx=ac.aa.co.interface_context(root,ds,torch.device('cpu'))
    xyz=np.asarray(ds['coordinates'],float)
    x=xyz[:,0]
    result=[]
    for side,face_x in [('left',float(x.min())),('right',float(x.max()))]:
        if ctx.get(side) is None:raise RuntimeError('Required %s interface not configured'%side)
        ix=ctx.get(side+'_local_indices')
        if ix is None:raise RuntimeError('Missing original interface indices for '+side)
        if torch.is_tensor(ix):ix=ix.detach().cpu().numpy()
        ix=np.asarray(ix,dtype=int)
        face=np.flatnonzero(np.isclose(x,face_x,rtol=0,atol=2e-5))
        if len(face)!=len(ix) or set(face)!=set(ix.tolist()):
            raise RuntimeError(side+' hard-interface nodes DO NOT match the full x-face: stop rather than impose wrong boundaries')
        result.extend(ix.tolist())
    boundary=np.unique(np.array(result,dtype=int))
    if len(boundary)==len(xyz):raise RuntimeError('Mesh has no free interior nodes')
    return boundary


def build_D(ds):
    """Sparse C3D8R center-point derivative matrix [element*xyz-axis,node]."""
    nodes=len(ds['coordinates'])
    conn=np.asarray(ds['element_node_indices'],int)
    dN=ac.aa.element_dN_numpy(ds).astype(np.float64)
    ne=conn.shape[0]
    rows=np.broadcast_to(np.arange(ne*3).reshape(ne,3,1),(ne,3,8)).ravel()
    cols=np.broadcast_to(conn.reshape(ne,1,8),(ne,3,8)).ravel()
    data=np.broadcast_to(dN.transpose(0,2,1),(ne,3,8)).ravel()
    D=sparse.coo_matrix((data,(rows,cols)),shape=(ne*3,nodes)).tocsr()
    test=np.asarray(ds['coordinates'],float)
    affine=D@test
    expected=np.broadcast_to(np.eye(3)[None,:,:],(ne,3,3)).reshape(ne*3,3)
    err=np.max(np.abs(affine-expected))
    if err>5e-3:raise RuntimeError('C3D8R sparse matrix fails affine reproduction %.4g'%err)
    return D,float(err)


class CompatibleProjector:
    """Solve min ||D(U0+dU)-G_LGNO||² + lambda*scale||dU||²;
    enforce dU=0 at both FE-NO interfaces. Exact sparse direct solution.
    """
    def __init__(self,ds,boundary,lam=0.05):
        if lam<=0:raise ValueError('lambda must be strictly positive')
        self.D,self.affine_error=build_D(ds)
        n=self.D.shape[1]
        self.boundary=np.asarray(boundary,int)
        self.free=np.setdiff1d(np.arange(n),self.boundary)
        self.Dfree=self.D[:,self.free].tocsc()
        h=self.Dfree.T@self.Dfree
        diag=np.asarray(h.diagonal())
        self.scale=float(np.maximum(np.mean(diag),1e-12))
        self.lam=float(lam)
        H=h + sparse.eye(h.shape[0],format='csc')*(self.lam*self.scale)
        self.factor=splu(H.tocsc())

    def reconstruct(self,base_u,gradient):
        base=np.asarray(base_u,float)
        target=np.asarray(gradient,float)
        ne=self.D.shape[0]//3
        if base.ndim!=2 or base.shape[1]!=3 or target.shape!=(ne,3,3):
            raise RuntimeError('Bad input array shapes to projection')
        dbase=(self.D@base).reshape(ne,3,3).transpose(0,2,1)
        gap=target-dbase
        correction=np.zeros_like(base)
        for comp in range(3):
            rhs=self.Dfree.T@gap[:,comp,:].reshape(ne*3)
            correction[self.free,comp]=self.factor.solve(rhs)
        # The reference boundary U remains exactly equal, not just within tolerance.
        correction[self.boundary,:]=0.
        reconstructed=base+correction
        reconstructed[self.boundary,:]=base[self.boundary,:]
        grad_proj=(self.D@reconstructed).reshape(ne,3,3).transpose(0,2,1)
        return reconstructed,grad_proj,{
            'interface_max_abs_change_mm':float(np.max(np.abs(reconstructed[self.boundary]-base[self.boundary]))),
            'deltaU_rms_mm':float(np.sqrt(np.mean(correction**2))),
            'unintegrable_gradient_residual_rms':float(np.sqrt(np.mean((grad_proj-target)**2))),
            'LGNO_correction_abs_rms':float(np.sqrt(np.mean(gap**2)))
        }


def rel2(a,b):return ac.rel2(np.asarray(a),np.asarray(b))


def errors_from_values(U,g,ds,case_idx):
    trueU=ds['U'][case_idx];refg=ac.gradient_from_u(trueU,ds)
    strains=ac.logstrain_from_gradient(g)
    trueLE=ds['LE'][case_idx]
    results={'U_vector_rel_l2_pct':rel2(U,trueU),
             'U2_rel_l2_pct':rel2(U[:,:,1],trueU[:,:,1]),
             'gradient_rel_l2_pct':rel2(g,refg)}
    for j,name in [(0,'LE11'),(1,'LE22'),(3,'LE12')]:
        results[name+'_rel_l2_pct']=rel2(strains[:,:,j],trueLE[:,:,j])
    return results


def with_row_errors(row,errors):
    for k,v in errors.items():row[k]=float(v)
    return row


def safe_output(root,step):
    out=root/'results'/ROOT_NAME/step
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output already exists. Manually delete ONLY this folder to rerun:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out
