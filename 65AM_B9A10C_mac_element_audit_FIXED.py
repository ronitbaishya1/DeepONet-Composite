#!/usr/bin/env python3
'''B9A10-C Mac: FE-consistent stress integration and controlled C3D8R/C3D8 comparison.
Requires Windows export from 8 independent Abaqus jobs. Read-only to old models.
Selftest uses an affine brick; no model training or coupled solve.
'''
from __future__ import annotations
import argparse,csv,hashlib,json,itertools
from pathlib import Path
import numpy as np

ROOTMODES={'inner_left':(('left_inner',4),('center_left',5)),
           'inner_right':(('center_right',5),('right_inner',4))}
LOC={'left_inner':slice(4,8),'center_left':slice(8,13),
     'center_right':slice(13,18),'right_inner':slice(18,22)}
JOBS={(r,t,bc):'B9A10C_%s_%s_%s'%(r,t,bc)
      for r in ('L','R') for t in ('R','F') for bc in ('FREE','REPLAY')}
SIGNS=np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
A=1/np.sqrt(3.0)
QUADS=np.asarray(list(itertools.product((-A,A),repeat=3)),float)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=float)))
def pct(a,b,scale):return 100.*norm(np.asarray(a)-np.asarray(b))/max(float(scale),1e-14)
def shp(xi):
    q=np.asarray(xi,dtype=float).reshape(3)
    return np.prod(1+SIGNS*q[None,:],axis=1)/8.0
def grad_nat(xi):
    a,b,c=np.asarray(xi,dtype=float)
    sg=SIGNS
    return np.column_stack((sg[:,0]*(1+sg[:,1]*b)*(1+sg[:,2]*c),
                            sg[:,1]*(1+sg[:,0]*a)*(1+sg[:,2]*c),
                            sg[:,2]*(1+sg[:,0]*a)*(1+sg[:,1]*b)))/8
def sig_tensor(s):
    a,b,c,d,e,f=np.asarray(s,float)[:6]
    return np.asarray([[a,d,e],[d,b,f],[e,f,c]])
def match_ip(xd,coords,nip):
    if nip==1:
        expected=shp([0,0,0])@xd
        diff=norm(expected-coords[0])
        if diff>5e-4:raise RuntimeError('Reduced-IP COORD not at deformed element center: %.5g'%diff)
        return np.asarray([[0,0,0]],float),[0],diff
    pred=np.asarray([shp(q)@xd for q in QUADS],float)
    dist=np.linalg.norm(pred[:,None,:]-coords[None,:,:],axis=2)
    # For a regular brick each Gauss point has a unique nearest observed coordinate.
    js=np.argmin(dist,axis=1)
    if len(set(js.tolist()))!=8:
        raise RuntimeError('Nonunique Gauss point match (COORD / FE node ordering mismatch)')
    diff=float(np.max(dist[np.arange(8),js]))
    if diff>5e-4:raise RuntimeError('FE quadrature/COORD mismatch %.5g mm'%diff)
    return QUADS,js.tolist(),diff

def integrate(data,typ):
    X=np.asarray(data['X'],float);U=np.asarray(data['U'],float)
    labels=np.asarray(data['node_labels'],int)
    elabels=np.asarray(data['element_labels'],int)
    conn=np.asarray(data['connectivity'],int)
    S=np.asarray(data['S'],float);C=np.asarray(data['IP_COORD'],float)
    nip=1 if typ=='R' else 8
    if X.shape!=(1008,3) or U.shape!=(1008,3) or conn.shape!=(728,8) or S.shape!=(728,nip,6) or C.shape!=(728,nip,3):
        raise RuntimeError('Incorrect Abaqus field dimensions')
    if len(set(labels.tolist()))!=1008:raise RuntimeError('Duplicate node labels')
    ix={int(l):i for i,l in enumerate(labels)}
    forces=np.zeros_like(X)
    worst_ip=0.;jmin=1e100;jmax=-1e100
    for i in range(len(conn)):
        ids=[ix[int(k)] for k in conn[i]]
        x=X[ids];u=U[ids];xd=x+u
        quad,permutation,iperr=match_ip(xd,C[i],nip)
        worst_ip=max(worst_ip,iperr)
        for j,q in enumerate(quad):
            dn=grad_nat(q)
            jac=x.T@dn;det0=float(np.linalg.det(jac))
            if det0<=1e-12:raise RuntimeError('Nonpositive reference element Jacobian at '+str(elabels[i]))
            dn_X=dn@np.linalg.inv(jac)
            F=np.eye(3)+u.T@dn_X
            J=float(np.linalg.det(F));jmin=min(jmin,J);jmax=max(jmax,J)
            if J<=0:raise RuntimeError('Negative deformation Jacobian at element '+str(elabels[i]))
            stress=sig_tensor(S[i,permutation[j]])
            P=J*stress@np.linalg.inv(F).T
            weight=8.0 if nip==1 else 1.0
            fe=(P@dn_X.T).T*(det0*weight)
            forces[ids]+=fe
    return forces,{'max_IP_coordinate_error_mm':worst_ip,'min_detF':jmin,'max_detF':jmax,
                   'net_internal_force_N':forces.sum(axis=0).tolist()}

def case(z,name):
    cols=('node_labels','X','U','RF','element_labels','connectivity','S','IP_COORD')
    missing=[c for c in cols if name+'_'+c not in z]
    if missing:raise RuntimeError('Missing '+name+' arrays '+str(missing))
    data={c:np.asarray(z[name+'_'+c]).copy() for c in cols}
    for c,ar in data.items():
        if not np.isfinite(ar).all():raise RuntimeError('Nonfinite '+name+' '+c)
    return data

def mask_interface(X):
    xmin=float(np.min(X[:,0]));xmax=float(np.max(X[:,0]))
    boundary=np.isclose(X[:,0],xmin,atol=2e-5)|np.isclose(X[:,0],xmax,atol=2e-5)
    if boundary.sum()!=224:raise RuntimeError('Boundary should contain 224 nodes')
    return boundary

def prepare(root):
    source=root/'B9A10C_exports'/'B9A10C_fields.npz'
    meta=root/'B9A10C_exports'/'B9A10C_windows_summary.json'
    anchor=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    required=[source,meta,anchor]
    for p in required:
        if not p.is_file():raise FileNotFoundError('Missing '+str(p))
    report=json.loads(meta.read_text())
    if sha(source)!=report.get('output_sha256'):raise RuntimeError('Windows export checksum mismatch')
    if len(report.get('jobs',[]))!=8:raise RuntimeError('Expected 8 Windows job records')
    with np.load(source,allow_pickle=False) as z:
        for nm in JOBS.values():
            if nm+'_X' not in z:raise RuntimeError('Missing job '+nm)
    return source,meta,anchor,report

def projected_rf(case_data,anchor_z,interface):
    X=case_data['X'];labels=case_data['node_labels'];rf=case_data['RF']
    pcoords=np.asarray(anchor_z['pca_coordinates_'+interface],float)
    basis=np.asarray(anchor_z['pca_basis_'+interface],float)
    mapped=[];used=set()
    for c in pcoords:
        dist=np.linalg.norm(X-c[None,:],axis=1)
        k=int(np.argmin(dist))
        if k in used or dist[k]>2e-5:raise RuntimeError('Interface PCA node match failed '+interface)
        used.add(k);mapped.append(rf[k])
    return basis.T@np.asarray(mapped,float).reshape(-1)

def selftest():
    x=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    u=np.column_stack((.0008*x[:,0],-.0015*x[:,1],.0002*x[:,2]))
    s=np.array([2.5,3.3,4.1,0.2,0.15,-0.1])
    for typ,nip in [('R',1),('F',8)]:
        q=np.zeros((1,nip,3))
        if nip==1:q[0,0]=shp([0,0,0])@(x+u)
        else:q[0]=np.asarray([shp(t)@(x+u) for t in QUADS])
        # Repeat an affine hex 8 times? For selftest only integrate element-level directly.
        forces=np.zeros((8,3))
        for nat in ([np.zeros(3)] if nip==1 else QUADS):
            grad=grad_nat(nat);Jac=x.T@grad;dndx=grad@np.linalg.inv(Jac)
            F=np.eye(3)+u.T@dndx;J=np.linalg.det(F)
            P=J*sig_tensor(s)@np.linalg.inv(F).T
            forces+=(P@dndx.T).T*(np.linalg.det(Jac)*(8 if nip==1 else 1))
        if norm(forces.sum(axis=0))>1e-10:raise RuntimeError('Nodal forces not self equilibrated')
        if typ=='R':baseline=forces.copy()
        elif norm(forces-baseline)>1e-10:raise RuntimeError('Affine C3D8/C3D8R mismatch')
        _,inds,e=match_ip(x+u,q[0],nip)
        if e>1e-12:raise RuntimeError('IP coordinate matching failed')
    print('B9A10C SELFTEST PASS | affine C3D8R/C3D8 nodal forces, virtual work, IP coordinate matching')

def run(root,check_only=False):
    export,meta,anchor,report=prepare(root)
    print('B9A10C INPUT CHECK PASS | 8 exported jobs | checksum verified')
    if check_only:return
    folder=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10C_element_controls'
    if not folder.is_dir():raise RuntimeError('Using Finder, create output folder first: '+str(folder))
    targets=[folder/x for x in ('B9A10C_case_metrics.csv','B9A10C_mode_metrics.csv',
                               'B9A10C_summary.json','B9A10C_comparison.png')]
    if any(p.exists() for p in targets):raise RuntimeError('Existing output protected; use new folder or move old results manually')
    with np.load(export,allow_pickle=False) as z, np.load(anchor,allow_pickle=False) as az:
        dat={key:case(z,nm) for key,nm in JOBS.items()}
        per_case=[];per_mode=[];ip_checks={}
        all_reports={q['job']:q for q in report['jobs']}
        for reg in ('L','R'):
            reg_long='inner_left' if reg=='L' else 'inner_right'
            boundary=mask_interface(dat[(reg,'R','FREE')]['X'])
            ref_free=dat[(reg,'R','FREE')]
            base_scale=max(norm(ref_free['RF'][boundary]),1e-12)
            for t in ('R','F'):
                for bc in ('FREE','REPLAY'):
                    d=dat[(reg,t,bc)]
                    if not np.array_equal(d['node_labels'],ref_free['node_labels']) or not np.allclose(d['X'],ref_free['X'],atol=2e-5,rtol=0):
                        raise RuntimeError('Mesh mismatch across element/BC controls')
                    f,diag=integrate(d,t)
                    ip_checks[JOBS[(reg,t,bc)]]=diag
                    r=d['RF'];free=~boundary
                    resid=f-r
                    per_case.append(dict(region=reg_long,element='C3D8R' if t=='R' else 'C3D8',BC=bc,
                      reference_boundary_RF_norm_N=base_scale,
                      relative_all_nodes_error_pct=pct(f,r,base_scale),
                      relative_interface_error_pct=pct(f[boundary],r[boundary],base_scale),
                      relative_free_node_error_pct=100*norm(resid[free])/base_scale,
                      actual_free_node_RF_pct=100*norm(r[free])/base_scale,
                      max_integration_point_position_error_mm=diag['max_IP_coordinate_error_mm'],
                      min_detF=diag['min_detF'],max_detF=diag['max_detF'],
                      ALLAE_over_ALLIE_pct=all_reports[JOBS[(reg,t,bc)]]['ALLAE_over_ALLIE_pct']))
            # Verify identical applied interface BC on both free solutions, and
            # identical full prescribed fields on the two replay experiments.
            rfree=dat[(reg,'R','FREE')]['U']; ffree=dat[(reg,'F','FREE')]['U']
            rrep=dat[(reg,'R','REPLAY')]['U']; frep=dat[(reg,'F','REPLAY')]['U']
            if np.max(np.abs(rfree[boundary]-ffree[boundary]))>2e-5:
                raise RuntimeError('C3D8 and C3D8R FREE runs have different applied boundary displacements')
            if np.max(np.abs(rrep-frep))>2e-5:
                raise RuntimeError('C3D8 and C3D8R REPLAY runs have different prescribed displacement fields')
            # Compare original boundary-only C3D8R to C3D8 reactions at identical BC.
            for iface,n_modes in ROOTMODES[reg_long]:
                old=projected_rf(dat[(reg,'R','FREE')],az,iface)
                new=projected_rf(dat[(reg,'F','FREE')],az,iface)
                force_scale=np.asarray(az['force_scale'][LOC[iface]],float)
                if (force_scale<=0).any() or old.size!=n_modes or new.size!=n_modes:
                    raise RuntimeError('Original 65V PCA mode dimension mismatch')
                for i in range(n_modes):
                    per_mode.append(dict(region=reg_long,interface=iface,mode=i+1,
                       C3D8R_force_N=float(old[i]),C3D8_force_N=float(new[i]),
                       difference_N=float(new[i]-old[i]),
                       absolute_scaled_difference=float(abs((new[i]-old[i])/force_scale[i]))))
        # Original reduced integration replay must be exactly the old Abaqus replay.
        replay=root/'B9A6_exports'/'B9A6_kinematic_RF.npz'
        controls={}
        if replay.is_file():
            with np.load(replay,allow_pickle=False) as b:
                for reg in ('L','R'):
                    long_name='inner_left' if reg=='L' else 'inner_right'
                    old=np.asarray(b[long_name+'_FEM_RF_nodal'],float)
                    current=dat[(reg,'R','REPLAY')]['RF']
                    ctrl_pct=pct(current,old,max(norm(old),1e-14))
                    controls[long_name+'_R_REPLAY_vs_B9A6_pct']=ctrl_pct
                    if ctrl_pct>0.5:
                        raise RuntimeError('C3D8R replay failed original B9A6 control (>0.5%) for '+long_name)
        rows={x['region']+'_'+x['element']+'_'+x['BC']:x for x in per_case}
        gap_r=np.mean([x['relative_all_nodes_error_pct'] for x in per_case if x['element']=='C3D8R' and x['BC']=='REPLAY'])
        gap_f=np.mean([x['relative_all_nodes_error_pct'] for x in per_case if x['element']=='C3D8' and x['BC']=='REPLAY'])
        summary=dict(step='65AM-B9A10-C',case_count=len(per_case),free_case_count=4,replay_case_count=4,
           average_R_REPLAY_force_gap_pct=gap_r,average_F_REPLAY_force_gap_pct=gap_f,
           R_replay_vs_prior_B9A6=controls,ip_geometry_checks=ip_checks,
           test='C3D8 and C3D8R have identical reference mesh, material, and applied BC; integration/formulation differs',
           conclusions_forbidden=['ALLAE/ALLIE does not directly measure missing nodal hourglass force',
              'C3D8 full-integration is not necessarily a better physical reference without mesh-convergence checks',
              'REPLAY constrains every node and is NOT a free equilibrium solution'],
           physics_loss_ready=False,live_Broyden_approved=False,old_models_untouched=True)
    def write_csv(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    write_csv(targets[0],per_case);write_csv(targets[1],per_mode)
    targets[2].write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=np.asarray([[next(row['relative_all_nodes_error_pct'] for row in per_case
                       if row['region']==reg and row['element']==typ and row['BC']=='REPLAY')
                   for typ in ('C3D8R','C3D8')] for reg in ('inner_left','inner_right')])
    fig,ax=plt.subplots(figsize=(8,4.8))
    x=np.arange(2);width=.33
    ax.bar(x-width/2,vals[:,0],width,label='C3D8R replay')
    ax.bar(x+width/2,vals[:,1],width,label='C3D8 replay')
    ax.set_xticks(x,('inner_left','inner_right'));ax.set_ylabel('Force gap / boundary RF norm (%)')
    ax.set_title('B9A10-C | Same FEM displacement: FE integration vs Abaqus reaction')
    ax.legend();fig.tight_layout();fig.savefig(targets[3],dpi=180);plt.close(fig)
    print('B9A10C DONE | R replay average gap %.4f%% ; F replay average gap %.4f%%'%(gap_r,gap_f))
    print('OUTPUT:',folder)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(Path(a.project).resolve(),a.check_only)
