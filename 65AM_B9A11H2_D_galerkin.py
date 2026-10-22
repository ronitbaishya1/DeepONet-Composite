#!/usr/bin/env python3
"""B9A11H-D: audit FE–NO Galerkin virtual-work interface projection on 24 Abaqus RF cases.

No training, no Abaqus, no solving the coupled 7-region system.
Using original frozen B5 and B9A10L/K force proxy, evaluate:
  1) interface trace test: Galerkin virtual work == projected assembled nodal force;
  2) Mesh-harmonic extension test: total virtual work and its free-node residual,
     and corrected boundary-only traction;
  3) independent Abaqus-RF modal force agreement.

A different choice of test function may change a raw virtual-work number but
DOES NOT repair a bad constitutive force or inaccurate free-node equilibrium.
The raw extended test is NOT a valid replacement for interface reactions unless
its interior residual is removed or the interior truly satisfies equilibrium.

Example:
  python3 65AM_B9A11H2_D_galerkin.py --project . --stage check
  python3 65AM_B9A11H2_D_galerkin.py --project . --stage run --tag run01
"""
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def virtual(P,vol,grad,conn,fh,V):
    gradV=np.einsum('eai,eaj->eij',V[conn],grad,optimize=True)
    continuum=float(np.einsum('eij,eij,e->',P,gradV,vol,optimize=True))
    hg=float(np.sum(fh*V))
    return continuum+hg,continuum,hg


# 12 physical edges of a standard C3D8R eight-node element. Each edge
# connects nodes of the SAME continuous displacement FE field.
HEX_EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7))


def make_mode_templates(xyz, conn, anc, faces):
    """Build a conforming nodal virtual-displacement extension on the REAL mesh.

    Previously the code assumed all interior nodes had y,z coordinates exactly
    on the boundary PCA grid, which is FALSE for the 65W inner meshes. Instead,
    extend prescribed boundary nodal PCA modes by solving a positive-weighted
    mesh-edge graph Laplace equation on the free nodes. This creates a smooth
    nodal FE test function for ANY internal mesh refinement/warping. It does not
    change the boundary trace or the physical constitutive/force model.

    The graph-harmonic extension is a choice of TEST FUNCTION, not a better
    constitutive model, an Abaqus solution, or a Galerkin coupling solver.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    xyz=np.asarray(xyz,dtype=np.float64)
    conn=np.asarray(conn,dtype=np.int64)
    n=len(xyz)
    if xyz.shape!=(n,3) or conn.ndim!=2 or conn.shape[1]!=8:
        raise RuntimeError('Expected n x 3 coordinates and C3D8R nElem x 8 connectivity')
    if conn.min()<0 or conn.max()>=n or not np.isfinite(xyz).all():
        raise RuntimeError('Invalid C3D8R nodal geometry')
    xmin=float(xyz[:,0].min());xmax=float(xyz[:,0].max())
    if xmax-xmin<=1.e-5:raise RuntimeError('Degenerate axial region')
    left=np.isclose(xyz[:,0],xmin,atol=2.e-5,rtol=0)
    right=np.isclose(xyz[:,0],xmax,atol=2.e-5,rtol=0)
    boundary=left|right
    free=~boundary
    if not left.any() or not right.any() or not free.any():
        raise RuntimeError('Both interfaces and interior nodes are required')

    # Mesh graph Laplacian L: sum_edges |v_i-v_j|^2 / edge_length.
    # Duplicate edges across adjacent hexes contribute a positive stiffness.
    ia=np.concatenate([conn[:,i] for i,j in HEX_EDGES])
    ib=np.concatenate([conn[:,j] for i,j in HEX_EDGES])
    lengths=np.linalg.norm(xyz[ia]-xyz[ib],axis=1)
    if np.any(lengths<=1.e-9) or not np.isfinite(lengths).all():
        raise RuntimeError('Zero-length C3D8R graph edge')
    w=1./lengths
    ii=np.r_[ia,ib,ia,ib]
    jj=np.r_[ia,ib,ib,ia]
    vv=np.r_[w,w,-w,-w]
    L=coo_matrix((vv,(ii,jj)),shape=(n,n)).tocsr()
    ids_f=np.flatnonzero(free);ids_b=np.flatnonzero(boundary)
    Lff=L[ids_f,:][:,ids_f].tocsc()
    Lfb=L[ids_f,:][:,ids_b].tocsr()
    try:lu=splu(Lff)
    except RuntimeError as exc:
        raise RuntimeError('Interior mesh component not connected to an interface: '+str(exc)) from exc

    modes=[]
    for side,face in zip(('left','right'),faces):
        face_xyz=np.asarray(anc['pca_coordinates_'+face],float)
        mat=np.asarray(anc['pca_basis_'+face],float)
        if face_xyz.ndim!=2 or face_xyz.shape[1]!=3 or mat.ndim!=2 or mat.shape[0]!=len(face_xyz)*3:
            raise RuntimeError('PCA coordinates/basis mismatch '+face)
        # Preserve the original exact anchor-to-interface correspondence. Never
        # replace the Dirichlet trace with a nearest-neighbour approximation.
        near,match=cKDTree(xyz).query(face_xyz)
        if float(np.max(near))>2.e-5 or len(set(match.tolist()))!=len(match):
            raise RuntimeError('Interface PCA nodes cannot be matched one-to-one '+face)
        own=left if side=='left' else right
        if not own[match].all():raise RuntimeError('PCA nodes mapped to wrong axial face '+face)
        for k in range(mat.shape[1]):
            vtrace=np.zeros((n,3),dtype=np.float64)
            vtrace[match]=mat[:,k].reshape(-1,3)
            vext=vtrace.copy()
            # Solve all three vector components against a single factorization.
            rhs=-(Lfb@vtrace[ids_b])
            vext[ids_f]=lu.solve(rhs)
            # Check H1-like graph harmonic extension and exact trace.
            residual=Lff@vext[ids_f]+Lfb@vext[ids_b]
            rel=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1.e-12))
            if not np.isfinite(vext).all() or rel>1.e-8:
                raise RuntimeError('Graph harmonic extension failed '+face+' mode '+str(k+1))
            if not np.array_equal(vext[ids_b],vtrace[ids_b]):
                raise RuntimeError('Boundary virtual-work trace was changed')
            modes.append((face,k+1,vtrace,vext))
    return modes,boundary


def selftest():
    rng=np.random.default_rng(23)
    P=rng.normal(size=(2,3,3));grad=rng.normal(size=(2,8,3));vol=np.array([2.1,1.4])
    conn=np.array([list(range(8)),list(range(3,11))],int)
    v=rng.normal(size=(11,3));fh=rng.normal(size=(11,3))
    # independent discrete force assembly: f_a = V_e P gradN_a + fh_a
    fint=np.zeros_like(v)
    local=np.einsum('eij,eaj,e->eai',P,grad,vol)
    np.add.at(fint,conn.ravel(),local.reshape(-1,3))
    direct=float(np.sum((fint+fh)*v))
    weak,*_=virtual(P,vol,grad,conn,fh,v)
    assert abs(direct-weak)<1e-10,(direct,weak)
    # Regression: interior y,z coordinates deliberately differ from either
    # interface y,z grid (the previous script raised KeyError here).
    plane=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    xyz=np.array([[xx,*yz] for xx in (0.,1.,2.) for yz in plane],dtype=float)
    xyz[4:8,1] += np.array([.10,.12,-.09,-.08])
    con=np.array([[0,4,5,1,3,7,6,2],[4,8,9,5,7,11,10,6]],dtype=int)
    anchor={}
    for name,indices in [('first',[0,1,2,3]),('last',[8,9,10,11])]:
        anchor['pca_coordinates_'+name]=xyz[indices]
        anchor['pca_basis_'+name]=rng.normal(size=(12,2))
    template,bound=make_mode_templates(xyz,con,anchor,('first','last'))
    assert len(template)==4 and int(bound.sum())==8
    assert any(np.max(abs(vext[~bound]))>1.e-6 for _,_,_,vext in template)
    for name,k,tr,ext in template:
        assert np.array_equal(tr[bound],ext[bound])
        local=rng.normal(size=(2,8,3))
        f=np.zeros_like(xyz)
        np.add.at(f,con.ravel(),local.reshape(-1,3))
        direct=float(np.sum(f*tr))
        corrected=float(np.sum(f*ext)-np.sum(f[~bound]*ext[~bound]))
        assert abs(corrected-direct)<1.e-10
    print('D SELFTEST PASS: virtual work identity + irregular interior yz mesh harmonic extension')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='.')
    ap.add_argument('--stage',choices=('selftest','check','run'),required=True)
    ap.add_argument('--tag',default='run01')
    args=ap.parse_args()
    if args.stage=='selftest':return selftest()
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe output tag')
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    G=importlib.import_module('65AM_B9A11G_fast_track')
    C,B6,L,H,anc,models=G.source(root)
    source=root/'B9A10H_exports'/'B9A10H_independent_fields.npz'
    manifest=root/'B9A10H_exports'/'B9A10H_windows_summary.json'
    if not source.is_file() or not manifest.is_file():raise FileNotFoundError('Missing authenticated B9A10H independent RF cases')
    rec=json.loads(manifest.read_text())
    if G.hash_file(source)!=rec['output_sha256']:raise RuntimeError('B9A10H export checksum mismatch')
    if args.stage=='check':
        print('D CHECK PASS: exact B6 B5 checkpoints/65V anchor and Abaqus 24-case export authenticated')
        return
    import torch
    out=root/'results'/'65AM_B9A11H_v2'/args.tag/'D_galerkin'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output protected '+str(out))
    records=[];cases=[];max_identity=0.;max_corrected=0.
    with np.load(source,allow_pickle=False) as z:
        for region,(d,net,ck) in models.items():
            ds=d['ds'];force,beta=G.force_op(root,d,L)
            xyz=np.asarray(ds['coordinates'],np.float64);n=len(xyz)
            faces=G.FACES[region]
            boundary=np.isclose(xyz[:,0],xyz[:,0].min(),atol=2e-5)|np.isclose(xyz[:,0],xyz[:,0].max(),atol=2e-5)
            # Geometry-only; compute once per region, not once per Abaqus case.
            conn=force.hg.connectivity.detach().cpu().numpy()
            mode_templates, boundary_h=make_mode_templates(xyz,conn,anc,faces)
            if not np.array_equal(boundary,boundary_h):
                raise RuntimeError('Interface boundary masks do not agree')
            print('D interface modes:',region,len(mode_templates),
                  '| mesh-harmonic extensions initialized',flush=True)
            for i in range(1,13):
                case_id='B9A9_'+('L' if region=='inner_left' else 'R')+('%02d'%i)
                case=H.make_case(z,case_id)
                perm=G.match(xyz,np.asarray(case['X'],float))
                refu=case['U'][perm][None]
                mat0=G.case_material_from_plan(root,region,i)
                branch,lift,fixed,diagnostics=B6.project_boundary(root,ds,refu,np.asarray(mat0,np.float32)[None,:],torch.device('cpu'))
                if any(x>0.001 for k,x in diagnostics.items() if k.endswith('_pca_max_mm')):
                    raise RuntimeError('PCA projection failed '+case_id)
                U=B6.predict_b5(d,net,branch,lift,torch.device('cpu'))[0]
                m=torch.as_tensor(mat0,dtype=torch.float64)
                with torch.no_grad():
                    fs,fh,fl=force.parts(torch.as_tensor(U,dtype=torch.float64),m)
                    f=(fs+fh).cpu().numpy()
                    Fh=fh.cpu().numpy();P=fl['P'].cpu().numpy()
                vol=force.volume.detach().cpu().numpy()
                grad=force.grad.detach().cpu().numpy()
                conn=force.hg.connectivity.detach().cpu().numpy()
                RF=np.asarray(case['RF'],np.float64)[perm]
                for face,mode,vtrace,vext in mode_templates:
                    num=np.asarray(anc['pca_basis_'+face]).shape[1]
                    scale=np.asarray(anc['force_scale'],float)[H.SLICES[face]]
                    if len(scale)!=num:raise RuntimeError('Incorrect modal normalization '+face)
                    wt,_,_=virtual(P,vol,grad,conn,Fh,vtrace)
                    direct=float(np.sum(f*vtrace))
                    ext,continuum,hg=virtual(P,vol,grad,conn,Fh,vext)
                    free=float(np.sum(f[~boundary]*vext[~boundary]))
                    corr=ext-free
                    rf=float(np.sum(RF*vtrace))
                    dt=abs(direct-wt); dc=abs(direct-corr)
                    max_identity=max(max_identity,dt);max_corrected=max(max_corrected,dc)
                    if dt>2e-6 or dc>2e-6:
                        raise RuntimeError('Virtual work identity failed %s %s mode %d | %.4g %.4g'%(case_id,face,mode,dt,dc))
                    records.append(dict(case_id=case_id,region=region,interface=face,mode=mode,
                      abaqus_RF_N=rf,projected_B5_N=direct,galerkin_trace_N=wt,
                      galerkin_extended_raw_N=ext,galerkin_extended_minus_free_residual_N=corr,
                      Galerkin_free_residual_N=free,continuum_work_N=continuum,hourglass_work_N=hg,
                      normalized_direct_error=(direct-rf)/scale[mode-1],
                      normalized_extended_raw_error=(ext-rf)/scale[mode-1],
                      normalized_corrected_error=(corr-rf)/scale[mode-1]))
                print('D checked',case_id,flush=True)
    table=pd.DataFrame(records)
    # Verify this independent implementation recovers saved B9A11G modal projections.
    old=root/'results'/'65AM_B9A11G_fast_track'/'run01'/'rf'/'B9A11G_true_Abaqus_RF_modes.csv'
    if old.is_file():
        existing=pd.read_csv(old)
        joined=table.merge(existing,on=['case_id','region','interface','mode'],how='outer',indicator=True,validate='one_to_one')
        if len(joined)!=len(table) or not (joined['_merge']=='both').all():
            raise RuntimeError('B9A11G modal sample/provenance mismatch')
        delta=max(float(np.max(abs(joined.abaqus_RF_N-joined.abaqus_RF_mode_N))),
                  float(np.max(abs(joined.projected_B5_N-joined.B5_proxy_mode_N))))
        if delta>2e-4:raise RuntimeError('Independent Galerkin force does not reproduce saved B9A11G mode audit: '+str(delta))
        print('D historical G modal cross-check PASS: max difference N',delta)
    for (reg,cid),group in table.groupby(['region','case_id'],sort=True):
        cases.append(dict(region=reg,case_id=cid,
           B5_nodal_projected_normalized_RMS=float(np.sqrt(np.mean(group.normalized_direct_error**2))),
           raw_mesh_harmonic_virtual_work_normalized_RMS=float(np.sqrt(np.mean(group.normalized_extended_raw_error**2))),
           physically_corrected_normalized_RMS=float(np.sqrt(np.mean(group.normalized_corrected_error**2))),
           free_virtual_work_RMS_N=float(np.sqrt(np.mean(group.Galerkin_free_residual_N**2)))))
    out.mkdir(parents=True)
    table.to_csv(out/'B9A11H_D_virtual_work_modes.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'B9A11H_D_case_comparison.csv',index=False)
    mean={}
    for reg,sub in pd.DataFrame(cases).groupby('region'):
        mean[str(reg)]={k:float(sub[k].mean()) for k in ('B5_nodal_projected_normalized_RMS',
                 'raw_mesh_harmonic_virtual_work_normalized_RMS','physically_corrected_normalized_RMS','free_virtual_work_RMS_N')}
    report=dict(step='B9A11H-D',cases=len(cases),modes=len(records),
       test='independent Galerkin volume work + frozen hourglass proxy vs nodal PCA force projection and real Abaqus RF',
       mean_case_metrics=mean,max_trace_virtual_work_identity_N=max_identity,
       max_corrected_extension_identity_N=max_corrected,
       extension_method='mesh-edge graph-harmonic nodal FE test function',
       galerkin_trace_changes_modal_force=False,
       raw_extended_virtual_work_is_valid_reaction_estimate=False,
       conclusion='Consistent Galerkin testing and nodal projected forces coincide with the same underlying FE force law. Uncorrected extended-work values include free-node residual; they cannot be claimed as improved interface traction. A real force improvement needs better constitutive/stabilization model or equilibrated B5 field.',
       limitations=['No new seven-region coupled solve','Same approximate B9A10L/K law, not exact Abaqus material update',
       'Existing 24 RF cases are diagnostics and not a newly held-out selection cohort','Opposing FE-patch force balance has not been verified here'])
    (out/'B9A11H_D_galerkin_summary.json').write_text(json.dumps(report,indent=2))
    print('D COMPLETE',out)
    for reg,v in mean.items():print(' ',reg,'nodal RMS',v['B5_nodal_projected_normalized_RMS'],
        'raw-work RMS',v['raw_mesh_harmonic_virtual_work_normalized_RMS'])
    print('IMPORTANT: projection-method changes alone cannot fix the B5 force residual')

if __name__=='__main__':main()
