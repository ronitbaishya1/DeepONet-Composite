#!/usr/bin/env python3
"""B9A10-D: independent Mac force-assembly audit with rotating Abaqus material axes.

Reads only: B9A10C_exports (fields + summary), B9A10D_exports (local orientations),
B9_coupled_anchor (PCA modes). Does not train, change old files, or solve FEM.

S in Abaqus is in local material directions when *ORIENTATION is used.
FieldValue.localCoordSystem: Q (global -> local). Thus sigma_global = Q.T @ sigma_local @ Q.
Every error uses the *same job's* Abaqus RF norm; BC=REPLAY constrains every node.
"""
import argparse, hashlib, json, csv, itertools
from pathlib import Path
import numpy as np

CASE_KEYS=[(reg,typ,bc) for reg in ('L','R') for typ in ('R','F') for bc in ('FREE','REPLAY')]
FACES={'L':('left_inner','center_left'),'R':('center_right','right_inner')}
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
GAUSS=1/np.sqrt(3)
PTS=np.array(list(itertools.product((-GAUSS,GAUSS),repeat=3)))

def jobkey(reg,typ,bc):return 'B9A10C_%s_%s_%s'%(reg,typ,bc)
def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def norm(a):return float(np.linalg.norm(np.asarray(a,float)))
def shp(q):return np.prod(1+SIGNS*np.asarray(q)[None,:],axis=1)/8.
def grad_nat(q):
    a,b,c=np.asarray(q,float)
    t=SIGNS
    return np.column_stack((t[:,0]*(1+t[:,1]*b)*(1+t[:,2]*c),
                            t[:,1]*(1+t[:,0]*a)*(1+t[:,2]*c),
                            t[:,2]*(1+t[:,0]*a)*(1+t[:,1]*b)))/8.
def stress6(s):
    s=np.asarray(s,float).reshape(6)
    return np.array([[s[0],s[3],s[4]],[s[3],s[1],s[5]],[s[4],s[5],s[2]]])
def integration_mapping(xd,ip,nip):
    pts=np.zeros((1,3)) if nip==1 else PTS
    xyz=np.asarray([shp(p)@xd for p in pts],float)
    dist=np.linalg.norm(xyz[:,None,:]-ip[None,:,:],axis=2)
    perm=np.argmin(dist,axis=1)
    gap=float(np.max(dist[np.arange(nip),perm]))
    if len(set(perm.tolist()))!=nip or gap>1e-4:
        raise RuntimeError('Integration point geometry/order mismatch: %.7g mm'%gap)
    return pts,perm,gap

def case(z,j):
    cols=('node_labels','X','U','RF','element_labels','connectivity','S','IP_COORD')
    a={k:np.asarray(z[j+'_'+k]).copy() for k in cols}
    if (a['X'].shape!=(1008,3) or a['U'].shape!=(1008,3)
            or a['RF'].shape!=(1008,3) or a['connectivity'].shape!=(728,8)):
        raise RuntimeError('Wrong mesh dimensions: '+j)
    if not all(np.isfinite(x).all() for x in a.values()):raise RuntimeError('Nonfinite input: '+j)
    return a

def boundary(X):
    x=X[:,0];b=np.isclose(x,np.min(x),atol=2e-5)|np.isclose(x,np.max(x),atol=2e-5)
    if int(b.sum())!=224:raise RuntimeError('Expected 224 constrained interface nodes')
    return b

def assemble(data,typ,mode,Q,avail):
    """Compute reference weak-form nodal forces from Cauchy stress and nodal kinematics.

    mode: raw (old incorrect assumption); local_to_global (documented correction)
    For non-identity rotation data, caller must have recorded every local frame.
    """
    X,U=data['X'],data['U'];nodeids=data['node_labels'].astype(int)
    conn=data['connectivity'].astype(int);elabels=data['element_labels'].astype(int)
    S=data['S'];ip=data['IP_COORD'];nip=1 if typ=='R' else 8
    if S.shape!=(728,nip,6) or ip.shape!=(728,nip,3):raise RuntimeError('IP shape mismatch')
    if Q.shape!=(728,nip,3,3) or avail.shape!=(728,nip):raise RuntimeError('Q shape mismatch')
    if mode=='local_to_global' and np.any(avail==0):
        raise RuntimeError('Cannot rotate local stress when some Abaqus orientations are absent')
    idx={int(lab):i for i,lab in enumerate(nodeids)}
    if len(idx)!=1008:raise RuntimeError('Duplicate Abaqus node labels')
    result=np.zeros_like(X);min_j=1e4;max_j=-1e4;worst_ip=0.
    for e in range(728):
        ids=[idx[int(n)] for n in conn[e]]
        x=X[ids];u=U[ids];xd=x+u
        xyz,perm,ip_error=integration_mapping(xd,ip[e],nip)
        worst_ip=max(worst_ip,ip_error)
        for k,pt in enumerate(xyz):
            dnat=grad_nat(pt);jac=x.T@dnat
            d0=float(np.linalg.det(jac))
            if d0<=1e-13:raise RuntimeError('Bad Jacobian for element '+str(elabels[e]))
            dNx=dnat@np.linalg.inv(jac)
            F=np.eye(3)+u.T@dNx
            J=float(np.linalg.det(F));min_j=min(min_j,J);max_j=max(max_j,J)
            if J<=0:raise RuntimeError('Inverted F at '+str(elabels[e]))
            ipidx=int(perm[k]);sig=stress6(S[e,ipidx])
            if mode=='local_to_global':
                rot=Q[e,ipidx]
                if np.max(np.abs(rot@rot.T-np.eye(3)))>1e-3 or abs(np.linalg.det(rot)-1)>1e-3:
                    raise RuntimeError('Orientation is not a proper orthogonal rotation')
                sig=rot.T@sig@rot
            P=J*sig@np.linalg.inv(F).T
            weight=8. if nip==1 else 1.
            result[ids]+=(P@dNx.T).T*(d0*weight)
    return result,dict(min_detF=min_j,max_detF=max_j,worst_IP_error_mm=worst_ip,
                       net_force_N=result.sum(axis=0).tolist())

def match_pca(x,force,az,iface):
    coords=np.asarray(az['pca_coordinates_'+iface],float)
    basis=np.asarray(az['pca_basis_'+iface],float)
    sc=np.asarray(az['force_scale'],float)
    segs={'left_inner':slice(4,8),'center_left':slice(8,13),
          'center_right':slice(13,18),'right_inner':slice(18,22)}
    idx=[];used=set()
    for c in coords:
        dd=np.linalg.norm(x-c[None,:],axis=1);i=int(np.argmin(dd))
        if dd[i]>2e-5 or i in used:raise RuntimeError('Invalid interface node mapping: '+iface)
        used.add(i);idx.append(i)
    return basis.T@force[idx].reshape(-1),sc[segs[iface]]

def inputs(root):
    pa=root/'B9A10C_exports'/'B9A10C_fields.npz'
    pm=root/'B9A10C_exports'/'B9A10C_windows_summary.json'
    qa=root/'B9A10D_exports'/'B9A10D_orientations.npz'
    qm=root/'B9A10D_exports'/'B9A10D_windows_summary.json'
    anchor=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    for p in (pa,pm,qa,qm,anchor):
        if not p.is_file():raise FileNotFoundError('Missing '+str(p))
    m=json.loads(pm.read_text());n=json.loads(qm.read_text())
    if hash_file(pa)!=m['output_sha256'] or hash_file(qa)!=n['export_sha256']:
        raise RuntimeError('Input file checksum failure')
    if hash_file(pm)!=n['prior_reference_sha256']:
        raise RuntimeError('Orientation export is from a different B9A10C reference')
    if len(m['jobs'])!=8 or len(n['cases'])!=8:raise RuntimeError('Eight job records required')
    lookup={v['job']:v for v in n['cases']};prior={v['job']:v for v in m['jobs']}
    if set(lookup)!=set(prior):raise RuntimeError('Case identity mismatch')
    for key in lookup:
        if lookup[key]['odb_sha256']!=prior[key]['odb_sha256']:
            raise RuntimeError('ODB identity mismatch for '+key)
    return pa,qa,anchor,m,n

def selftest():
    # Geometry, stress-rotation and virtual-work identities at single element level.
    x=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    u=np.einsum('ij,nj->ni',np.array([[.002,.005,.003],[.0,-.001,.004],[0,0,.003]]),x)
    ang=np.deg2rad(14);c=np.cos(ang);s=np.sin(ang)
    Q=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    local=np.array([[8.,1.,2.],[1.,5.,.7],[2.,.7,3.]])
    global_s=Q.T@local@Q
    if abs(np.trace(global_s)-np.trace(local))>1e-12:raise RuntimeError('Rotation trace not conserved')
    if np.linalg.norm(Q@global_s@Q.T-local)>1e-12:raise RuntimeError('Global->local frame transform incorrect')
    rhs=[]
    for pts in ([np.zeros(3)],PTS):
        fs=np.zeros((8,3));energy=0.
        for q in pts:
            d=grad_nat(q);D=x.T@d
            gx=d@np.linalg.inv(D);F=np.eye(3)+u.T@gx
            P=np.linalg.det(F)*global_s@np.linalg.inv(F).T
            w=8 if len(pts)==1 else 1
            fe=(P@gx.T).T*np.linalg.det(D)*w
            fs+=fe
        rhs.append(fs)
    if np.linalg.norm(rhs[0]-rhs[1])>1e-10:raise RuntimeError('Affine patch test failed')
    if np.linalg.norm(rhs[0].sum(axis=0))>1e-10:raise RuntimeError('Weak nodal forces not balanced')
    rnd=np.random.default_rng(9).normal(size=(8,3))
    q=np.zeros(3);d=grad_nat(q);D=x.T@d;gx=d@np.linalg.inv(D)
    F=np.eye(3)+u.T@gx;P=np.linalg.det(F)*global_s@np.linalg.inv(F).T
    work=float(np.sum(rhs[0]*rnd));ref=float(np.linalg.det(D)*8*np.sum(P*(rnd.T@gx)))
    if abs(work-ref)>1e-9:raise RuntimeError('Virtual work check failed')
    print('B9A10D SELFTEST PASS | local stress rotation | affine 1/8-point | virtual work')

def run(root,check=False):
    pa,qa,anchor,meta,mq=inputs(root)
    with np.load(qa,allow_pickle=False) as qz:
        for reg,typ,bc in CASE_KEYS:
            key=jobkey(reg,typ,bc)
            available=np.asarray(qz[key+'_Q_available'],np.uint8)
            print('CHECK',key,'orientation matrices %d / %d'%(int(available.sum()),available.size))
    print('B9A10D INPUT CHECK PASS | matching original ODB SHA256 and exports')
    if check:return
    out=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10D_orientation_assembly'
    if not out.is_dir():raise RuntimeError('Using Finder create output folder: '+str(out))
    targets=[out/x for x in ('B9A10D_case_errors.csv','B9A10D_mode_errors.csv',
                             'B9A10D_summary.json','B9A10D_force_comparison.png')]
    if any(p.exists() for p in targets):raise RuntimeError('Result files exist: move previous run manually before continuing')
    rows=[];modes=[];details={}
    with np.load(pa,allow_pickle=False) as pz,np.load(qa,allow_pickle=False) as qz,np.load(anchor,allow_pickle=False) as az:
        for reg,typ,bc in CASE_KEYS:
            key=jobkey(reg,typ,bc);d=case(pz,key)
            elabel=np.asarray(qz[key+'_element_labels'],int)
            if not np.array_equal(elabel,d['element_labels']):raise RuntimeError('ODB element-order mismatch '+key)
            Q=np.asarray(qz[key+'_Q_global_to_local'],float)
            has=np.asarray(qz[key+'_Q_available'],np.uint8)
            complete=bool(np.all(has==1))
            raw,raw_diag=assemble(d,typ,'raw',Q,has)
            converted=None
            if complete:
                converted,corr_diag=assemble(d,typ,'local_to_global',Q,has)
            bc_nodes=boundary(d['X']);r=d['RF']
            ref_norm=norm(r[bc_nodes])
            if ref_norm<1e-12:raise RuntimeError('Zero interface reaction norm '+key)
            for label,f in [('old_raw_assumption',raw),('rotated_local_to_global',converted)]:
                if f is None:continue
                allgap=100*norm(f-r)/ref_norm
                facegap=100*norm(f[bc_nodes]-r[bc_nodes])/ref_norm
                freegap=100*norm(f[~bc_nodes]-r[~bc_nodes])/ref_norm
                rows.append(dict(job=key,region='inner_left' if reg=='L' else 'inner_right',
                  element_type='C3D8R' if typ=='R' else 'C3D8',BC=bc,assembly=label,
                  own_boundary_reaction_scale_N=ref_norm,
                  all_nodes_gap_pct=allgap,boundary_gap_pct=facegap,interior_gap_pct=freegap,
                  actual_interior_RF_pct=100*norm(r[~bc_nodes])/ref_norm,
                  known_rotation_matrices=int(has.sum()),expected_rotation_matrices=has.size,
                  worst_IP_match_mm=raw_diag['worst_IP_error_mm'],
                  net_internal_force_norm_N=norm(f.sum(axis=0))))
                if bc=='FREE':
                    for iface in FACES[reg]:
                        fp,scale=match_pca(d['X'],f,az,iface)
                        rp,_=match_pca(d['X'],r,az,iface)
                        for k in range(len(scale)):
                            modes.append(dict(job=key,interface=iface,mode=k+1,
                              element_type='C3D8R' if typ=='R' else 'C3D8',assembly=label,
                              signed_error_normalized_65V=float((fp[k]-rp[k])/scale[k]),
                              absolute_error_normalized_65V=float(abs((fp[k]-rp[k])/scale[k])),
                              force_calculated_N=float(fp[k]),reference_Abaqus_RF_N=float(rp[k])))
            details[key]={'rotation_available':complete,'rotation_fraction':float(np.mean(has)),
                          'min_detF':raw_diag['min_detF'],'max_detF':raw_diag['max_detF'],
                          'max_IP_geometry_error_mm':raw_diag['worst_IP_error_mm']}
    if not rows:raise RuntimeError('No results')
    summary={'step':'65AM-B9A10-D','purpose':'Audit material-local stress rotation and force assembly',
      'source_rotation_convention':'Q=global_to_local; global_sigma=Q.T@local_sigma@Q',
      'cases':details,'comparisons':{},'orientation_missing_is_not_identity':True,
      'old_outputs_modified':False,'B5_and_65V_unchanged':True,
      'physics_loss_ready':False,'live_Broyden_approved':False,
      'warnings':['C3D8 full-displacement replay is not an equilibrated C3D8 solution',
                  'Abaqus C3D8R hourglass stabilization is not reconstructed by this stress integral',
                  'Small free-node forces of corrected C3D8 FREE cases are needed before use as a loss']}
    for reg in ('inner_left','inner_right'):
        for typ in ('C3D8R','C3D8'):
            for bc in ('FREE','REPLAY'):
                rs=[r for r in rows if r['region']==reg and r['element_type']==typ and r['BC']==bc]
                summary['comparisons'][reg+'/'+typ+'/'+bc]={r['assembly']:r['all_nodes_gap_pct'] for r in rs}
    def write_csv(p,records):
        if not records:return
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0].keys()));w.writeheader();w.writerows(records)
    write_csv(targets[0],rows);write_csv(targets[1],modes)
    targets[2].write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[];rvals=[];cvals=[]
    for region in ('inner_left','inner_right'):
        for typ in ('C3D8R','C3D8'):
            for bc in ('FREE','REPLAY'):
                q=[r for r in rows if r['region']==region and r['element_type']==typ and r['BC']==bc]
                a=next(x['all_nodes_gap_pct'] for x in q if x['assembly']=='old_raw_assumption')
                b=next((x['all_nodes_gap_pct'] for x in q if x['assembly']=='rotated_local_to_global'),np.nan)
                labels.append(('L' if region=='inner_left' else 'R')+'/'+typ+'/'+bc[0]);rvals.append(a);cvals.append(b)
    fig,ax=plt.subplots(figsize=(12,5.8))
    x=np.arange(len(labels));w=.37
    ax.bar(x-w/2,rvals,width=w,label='Old local-stress-as-global')
    ax.bar(x+w/2,cvals,width=w,label='Corrected local-to-global stress')
    ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel('Force gap / same-job boundary RF norm (%)')
    ax.set_title('B9A10-D | Material orientation test (FREE vs fully prescribed REPLAY)')
    ax.legend();fig.tight_layout();fig.savefig(targets[3],dpi=170);plt.close(fig)
    print('B9A10D COMPLETE | summary, interface modes and comparison plot:',out)
    for k,v in summary['comparisons'].items():print(k,v)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='.')
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.selftest:selftest()
    else:run(Path(args.project).resolve(),args.check_only)
