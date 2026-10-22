#!/usr/bin/env python3
"""B9A11-S1/S2/S3: read-only FEM/B5/POD symmetry audit.
Run from DeepONet_Composite. Existing source data and checkpoints unchanged.
Symmetry results valid only for eligible mirrored boundary/material conditions.
"""
import argparse, csv, importlib, json, math, sys
from pathlib import Path
import numpy as np
import torch

REG=('inner_left','inner_right')
OUT=Path('results/65AM_B9_coupled_feasibility/B9A11S_symmetry')
PAIRS=(('left_inner','right_inner'),('center_left','center_right'))

def save_csv(p, rows, fields=None):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def num(x):
    if isinstance(x,(np.integer,np.floating)):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64).reshape(-1)))
def rel(a,b):return norm(np.asarray(a)-np.asarray(b))/max(norm(a),norm(b),1e-12)

def mirror_mapping(left,right,x0,tol=2e-5):
    """Index map: map_left_to_right[i] = corresponding right index."""
    L=np.asarray(left,dtype=np.float64);R=np.asarray(right,dtype=np.float64)
    if L.ndim!=2 or R.ndim!=2 or L.shape!=R.shape or L.shape[1]!=3:
        raise ValueError('Mirror requires matching [N,3] coordinates')
    Q=L.copy();Q[:,0]=2*x0-Q[:,0]
    from scipy.spatial import cKDTree
    dist,ix=cKDTree(R).query(Q,k=1)
    if np.max(dist)>tol or len(set(ix.tolist()))!=len(ix):
        raise RuntimeError('Mirror node/IP mapping failed: max distance %.7g mm, unique %d/%d'
                           %(np.max(dist),len(set(ix.tolist())),len(ix)))
    return ix,float(np.max(dist))

def reflected_vectors(L,idx,nright):
    L=np.asarray(L,dtype=np.float64)
    output=np.empty((nright,3),dtype=np.float64)
    output[idx]=L*np.array([-1.,1.,1.])[None,:]
    return output

def reflected_tensor_voigt(L,idx,nright):
    # Voigt order: 11,22,33,12,13,23. Valid ONLY in common global frame.
    L=np.asarray(L,dtype=np.float64)
    out=np.empty((nright,6),dtype=np.float64)
    out[idx]=L*np.array([1.,1.,1.,-1.,-1.,1.])[None,:]
    return out

def base_data(root):
    sys.path.insert(0,str(root)) if str(root) not in sys.path else None
    torch.set_default_dtype(torch.float32)
    c=importlib.import_module('65AM_B5_common')
    dd={}
    for r in REG:
        d,mu,sd=c.prepare(root,r)
        ds=d['ds'];need=('coordinates','ip_coordinates','U','S','element_node_indices','branch_inputs')
        for k in need:
            if k not in ds:raise KeyError('Missing '+r+' dataset key '+k)
        dd[r]=dict(d=d,mu=mu,sd=sd)
    return c,dd

def common_cohort(dd):
    all_ids=[]
    for r in REG:
        d=dd[r]['d']; ids=np.r_[d['fit'],d['tune']].astype(int)
        all_ids.append(ids)
    return sorted(set(all_ids[0]) & set(all_ids[1]))

def prepare_geometry(dd,atol):
    dL=dd[REG[0]]['d'];dR=dd[REG[1]]['d']
    XL=np.asarray(dL['ds']['coordinates'],float);XR=np.asarray(dR['ds']['coordinates'],float)
    x0=float((XL[:,0].mean()+XR[:,0].mean())/2)
    node_map,ngap=mirror_mapping(XL,XR,x0,atol)
    ip_map,igap=mirror_mapping(dL['ds']['ip_coordinates'],dR['ds']['ip_coordinates'],x0,atol)
    fixedL=np.asarray(dL['fixed'],dtype=int);fixedR=np.asarray(dR['fixed'],dtype=int)
    if not np.array_equal(np.sort(node_map[fixedL]),np.sort(fixedR)):
        raise RuntimeError('Prescribed-node masks NOT mirror-compatible; stop')
    # Element topology, with unordered node sets (mirror changes handedness).
    cL=np.asarray(dL['ds']['element_node_indices'],int)
    cR=np.asarray(dR['ds']['element_node_indices'],int)
    setR={tuple(sorted(e)) for e in cR}
    mirrored={tuple(sorted(node_map[e])) for e in cL}
    top_ok=(setR==mirrored and len(setR)==len(cR))
    return dict(x0=x0,node_map=node_map,ip_map=ip_map,node_gap_mm=ngap,
                ip_gap_mm=igap,topology_mirror=top_ok)

def candidate_pairs(dd,g,a):
    """BC/material matching is NECESSARY, not sufficient for physical mirror eligibility.

    Default compares SAME case IDs; optional cross-case scan searches for a compatible
    partner with identical material and mirrored prescribed U. No FEM solution
    similarity is used to select pairs, so there is no symmetry-result leakage.
    """
    dl=dd['inner_left']['d'];dr=dd['inner_right']['d']
    xL=np.asarray(dl['ds']['branch_inputs'],float);xR=np.asarray(dr['ds']['branch_inputs'],float)
    uL=np.asarray(dl['ds']['U'],float);uR=np.asarray(dr['ds']['U'],float)
    fixedR=np.asarray(dr['fixed'],int)
    l_ids=sorted(set(np.r_[dl['fit'],dl['tune']].astype(int)))
    r_ids=sorted(set(np.r_[dr['fit'],dr['tune']].astype(int)))
    both=sorted(set(l_ids)&set(r_ids))
    rows=[];used_right=set()
    for kl in (l_ids if a.cross_case else both):
        ml=reflected_vectors(uL[kl],g['node_map'],len(g['node_map']))
        choices=[]
        for kr in (r_ids if a.cross_case else [kl]):
            material=rel(xL[kl,:3],xR[kr,:3])
            if a.cross_case and material>a.material_tol:continue
            bc=rel(ml[fixedR],uR[kr,fixedR])
            choices.append((bc,material,int(kr)))
        if not choices:continue
        choices.sort(key=lambda z:(z[0],z[1],z[2]))
        # One-to-one matching in cross-case mode.
        pick=next((v for v in choices if v[2] not in used_right),None)
        if pick is None:continue
        bc,material,kr=pick
        preliminary=(material<=a.material_tol and bc<=a.bc_tol)
        if a.cross_case and not preliminary:continue
        if a.cross_case:used_right.add(kr)
        rows.append(dict(left_case_id=int(kl),right_case_id=int(kr),
                   material_relative_gap=material,mirrored_prescribed_U_relative_gap=bc,
                   eligible=bool(preliminary),
                   cohort_left='fit' if kl in set(dl['fit']) else 'tune',
                   cohort_right='fit' if kr in set(dr['fit']) else 'tune'))
    return rows

def field_tests(dd,g,pairs,model_fields=None,allow_stress=False):
    L=dd['inner_left']['d']['ds'];R=dd['inner_right']['d']['ds'];n=len(g['node_map'])
    ipn=len(g['ip_map']);rows=[]
    for p in pairs:
        kl=int(p['left_case_id']);kr=int(p['right_case_id']);out=dict(left_case_id=kl,right_case_id=kr,eligible=p['eligible'],cohort_left=p['cohort_left'],cohort_right=p['cohort_right'])
        uL=reflected_vectors(L['U'][kl],g['node_map'],n);uR=np.asarray(R['U'][kr],float)
        out['fem_U_relative_mirror_gap']=rel(uL,uR)
        if allow_stress:
            # This is conditional: coordinate convention of exported Abaqus S MUST be confirmed global.
            sL=reflected_tensor_voigt(L['S'][kl],g['ip_map'],ipn)
            out['fem_S_global_assumption_mirror_gap']=rel(sL,R['S'][kr])
        if model_fields is not None:
            left=reflected_vectors(model_fields['inner_left'][kl],g['node_map'],n)
            right=model_fields['inner_right'][kr]
            out['B5_U_relative_mirror_gap']=rel(left,right)
            out['left_B5_vs_FEM_U_relative_gap']=rel(model_fields['inner_left'][kl],L['U'][kl])
            out['right_B5_vs_FEM_U_relative_gap']=rel(right,R['U'][kr])
        rows.append(out)
    return rows

def b5_predict(c,dd,a):
    res={}
    models=dict(inner_left=a.left_model,inner_right=a.right_model)
    for r in REG:
        v,loss=models[r].split('_',1)
        d=dd[r]['d'];mu=dd[r]['mu'];sd=dd[r]['sd']
        net,_=c.load_model(a.project,v,loss,r,d,mu,sd,torch.device('cpu'))
        ids=np.asarray(np.r_[d['fit'],d['tune']],int)
        u,co=c.predict(net,d,ids,torch.device('cpu'))
        res[r]={int(k):u[i] for i,k in enumerate(ids)}
    return res

def principal_cosine(A,B):
    A=np.asarray(A,float);B=np.asarray(B,float)
    # span vectors as columns, robust to PCA sign, permutation, rotations
    QA,sa,_=np.linalg.svd(A,full_matrices=False)
    QB,sb,_=np.linalg.svd(B,full_matrices=False)
    ra=int(np.sum(sa>max(sa[0]*1e-9,1e-12)))
    rb=int(np.sum(sb>max(sb[0]*1e-9,1e-12)))
    cos=np.linalg.svd(QA[:,:ra].T@QB[:,:rb],compute_uv=False)
    return float(np.min(cos)),float(np.mean(cos)),ra,rb

def paired_basis(A,point_map,nright):
    A=np.asarray(A,float)
    if A.ndim!=2 or len(A)!=3*len(point_map):
        raise ValueError('Force PCA basis expected shape [3*number_of_nodes,rank], got '+str(A.shape))
    vals=A.reshape(len(point_map),3,-1)*np.array([-1.,1.,1.])[None,:,None]
    out=np.empty((nright,3,A.shape[1]));out[point_map]=vals
    return out.reshape(3*nright,-1)

def s3_audit(root,dd,g):
    rows=[]
    left=dd['inner_left']['d']['basis'];right=dd['inner_right']['d']['basis'];idx=g['node_map'];n=len(idx)
    # B5 POD residual displacement bases [rank, node, 3]
    X=np.empty((n,3,left.shape[0]),float)
    X[idx]=np.transpose(left,(1,2,0))*np.array([-1.,1.,1.])[None,:,None]
    Q=np.transpose(right,(1,2,0))
    small,avg,rankL,rankR=principal_cosine(X.reshape(-1,left.shape[0]),Q.reshape(-1,right.shape[0]))
    rows.append(dict(subspace='B5 displacement residual POD',left='inner_left',right='inner_right',
                     rank_left=rankL,rank_right=rankR,min_principal_cosine=small,
                     mean_principal_cosine=avg,coordinate_gap_mm=g['node_gap_mm'],status='measured'))
    fd=root/'data/hybrid_force_pca_7region'
    for leftname,rightname in PAIRS:
        pl=fd/('force_pca_'+leftname+'.npz');pr=fd/('force_pca_'+rightname+'.npz')
        if not pl.is_file() or not pr.is_file():
            rows.append(dict(subspace='force PCA',left=leftname,right=rightname,status='MISSING source, skipped'))
            continue
        with np.load(pl,allow_pickle=False) as lz,np.load(pr,allow_pickle=False) as rz:
            coordsL=np.asarray(lz['coordinates'],float);coordsR=np.asarray(rz['coordinates'],float)
            try:
                m,dist=mirror_mapping(coordsL,coordsR,g['x0'],2e-5)
            except (ValueError,RuntimeError) as e:
                rows.append(dict(subspace='force PCA',left=leftname,right=rightname,status='geometry mismatch: '+str(e)))
                continue
            B=paired_basis(lz['basis'],m,len(m));C=np.asarray(rz['basis'],float)
            pc,avg,rl,rr=principal_cosine(B,C)
            rows.append(dict(subspace='force PCA',left=leftname,right=rightname,rank_left=rl,
                             rank_right=rr,min_principal_cosine=pc,mean_principal_cosine=avg,
                             coordinate_gap_mm=dist,status='measured; compare subspaces not individual modes'))
    return rows

def selftest():
    X=np.array([[-2.,0,0],[-2,1,0],[-3,0,0],[-3,1,0.]])
    Y=X.copy();Y[:,0]=-X[:,0];Y=Y[[2,0,3,1]]
    m,d=mirror_mapping(X,Y,0,1e-10)
    q=np.array([[1.,2.,3.]]*4)
    expected=reflected_vectors(q,m,4)
    assert np.all(expected==np.array([-1.,2.,3.]))
    a=np.eye(4); b=a[:,[2,1,0,3]];c=principal_cosine(a,b)
    assert c[0]>0.999999
    print('B9A11S SELFTEST PASS: mirror permutation, reflection, PCA sign/order invariance')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--stage',choices=['selftest','check','s1','s2','s3','all'],required=True)
    p.add_argument('--material-tol',type=float,default=1e-4)
    p.add_argument('--bc-tol',type=float,default=0.01,help='relative prescribed displacement tolerance')
    p.add_argument('--coordinate-tol',type=float,default=2e-5)
    p.add_argument('--left-model',default='standardized_grad')
    p.add_argument('--right-model',default='centered_grad')
    p.add_argument('--cross-case',action='store_true',help='search material/BC-compatible pairs across case IDs (one-to-one)')
    p.add_argument('--stress-global-confirmed',action='store_true',help='only after confirming exported S in global frame')
    a=p.parse_args();a.project=a.project.resolve();root=a.project
    if a.stage=='selftest':selftest();return
    c,dd=base_data(root);g=prepare_geometry(dd,a.coordinate_tol)
    pairs=candidate_pairs(dd,g,a)
    out=root/OUT
    common=len(pairs);eligible=sum(p['eligible'] for p in pairs)
    info=dict(step='B9A11S',candidate_mirrored_pairs=eligible,common_case_ids=common,
              paired_case_rule='cross-case scan' if a.cross_case else 'same case ID; check E1/E2/G12 and mirrored FEM prescribed U',
              mirror_center_x_mm=g['x0'],node_geometry_error_mm=g['node_gap_mm'],
              ip_geometry_error_mm=g['ip_gap_mm'],element_topology_mirror=g['topology_mirror'],
              stress_comparison_requires_global_frame=a.stress_global_confirmed,
              caution='Matching BC and three material constants is necessary but not sufficient: verify ply axes, material frame and loading. No independent validation.')
    print('S CHECK',json.dumps(info,indent=2,default=num))
    if a.stage=='check':return
    if out.exists() and not out.is_dir():raise RuntimeError('Output path is a file')
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'B9A11S_pair_eligibility.csv',pairs)
    if a.stage in ('s1','all'):
        save_csv(out/'B9A11S_fem_mirror.csv',field_tests(dd,g,pairs,allow_stress=a.stress_global_confirmed))
    if a.stage in ('s2','all'):
        predict=b5_predict(c,dd,a)
        save_csv(out/'B9A11S_b5_mirror.csv',field_tests(dd,g,pairs,predict,allow_stress=False))
    if a.stage in ('s3','all'):
        save_csv(out/'B9A11S_pod_subspaces.csv',s3_audit(root,dd,g))
    (out/'B9A11S_summary.json').write_text(json.dumps(info,indent=2,default=num))
    if eligible==0:print('WARNING: no matching mirrored BC/material pairs. S1/S2 cannot prove symmetry or asymmetry.')
    print('B9A11S',a.stage.upper(),'COMPLETE',out)

if __name__=='__main__':main()
