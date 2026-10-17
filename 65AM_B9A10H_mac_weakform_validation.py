#!/usr/bin/env python3
"""65AM-B9A10-H Mac: nonlinear C3D8R weak-form validation across 24 independent Abaqus cases.

READ-ONLY: B9A9 case plan, B9A10C/D originals, B9A10H independent export,
           B9_coupled_anchor/B9_65V_anchor.npz.
NO training, fitted stabilization correction, Broyden, or modification to older outputs.

Use --selftest; --project . --check-only; --project .
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
R=('L','R')
SLICES={'left_inner':slice(4,8),'center_left':slice(8,13),
        'center_right':slice(13,18),'right_inner':slice(18,22)}
FACES={'L':('left_inner','center_left'),'R':('center_right','right_inner')}
NAMES=['B9A9_%s%02d'%(r,k) for r in R for k in range(1,13)]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def L2(a):return float(np.linalg.norm(np.asarray(a,float)))

def grad0():
    # C3D8 natural derivatives at element center
    return SIGNS/8.

def sigma(s):
    a=np.asarray(s,float).reshape(6)
    return np.array([[a[0],a[3],a[4]],[a[3],a[1],a[5]],[a[4],a[5],a[2]]])

def check_arrays(a,name):
    spec={'node_labels':(1008,),'X':(1008,3),'U':(1008,3),'RF':(1008,3),
          'element_labels':(728,),'connectivity':(728,8),'S':(728,6),'Q':(728,3,3)}
    for key,shape in spec.items():
        if key not in a or a[key].shape!=shape:raise RuntimeError('%s wrong shape %s'%(name,key))
        if not np.isfinite(a[key]).all():raise RuntimeError('%s nonfinite %s'%(name,key))
    if len(set(a['node_labels'].astype(int)))!=1008:raise RuntimeError('Duplicate node labels '+name)
    if len(set(a['element_labels'].astype(int)))!=728:raise RuntimeError('Duplicate element labels '+name)
    q=a['Q']
    if np.max(np.abs(q@q.transpose(0,2,1)-np.eye(3)))>1e-3:
        raise RuntimeError('Local rotation not orthogonal '+name)
    if np.max(np.abs(np.linalg.det(q)-1.0))>1e-3:
        raise RuntimeError('Invalid local coordinate rotation '+name)

def assemble(a):
    """Vectorized one-point C3D8R stress-only FE force assembly.

    This omits Abaqus's proprietary stabilization force; it is NOT a physics loss.
    """
    lab={int(k):i for i,k in enumerate(a['node_labels'])}
    ix=np.asarray([[lab[int(v)] for v in row] for row in a['connectivity']],dtype=int)
    X=a['X'][ix];U=a['U'][ix]
    dnat=grad0()
    jac=np.einsum('eai,aj->eij',X,dnat)
    detj=np.linalg.det(jac)
    if np.any(detj<=1e-14):raise RuntimeError('Invalid element reference Jacobian')
    grad=np.einsum('aj,ejk->eak',dnat,np.linalg.inv(jac))
    F=np.eye(3)[None,:,:]+np.einsum('eai,eaj->eij',U,grad)
    detF=np.linalg.det(F)
    if np.any(detF<=0):raise RuntimeError('Non-positive deformation-gradient determinant')
    v=a['S']
    sig=np.zeros((len(v),3,3))
    sig[:,0,0]=v[:,0];sig[:,1,1]=v[:,1];sig[:,2,2]=v[:,2]
    sig[:,0,1]=sig[:,1,0]=v[:,3]
    sig[:,0,2]=sig[:,2,0]=v[:,4]
    sig[:,1,2]=sig[:,2,1]=v[:,5]
    Q=a['Q']
    global_sig=Q.transpose(0,2,1)@sig@Q
    Piola=detF[:,None,None]*(global_sig@np.linalg.inv(F).transpose(0,2,1))
    f_elem=(Piola@grad.transpose(0,2,1)).transpose(0,2,1)*(8*detj[:,None,None])
    result=np.zeros((len(a['X']),3))
    np.add.at(result,ix,f_elem)
    # Best affine displacement fit on each element. No fitted physics terms.
    design=np.concatenate((np.ones((len(ix),8,1)),X),axis=2)
    At=design.transpose(0,2,1)
    coeff=np.linalg.solve(At@design,At@U)
    fitted=design@coeff
    numer=np.linalg.norm((U-fitted).reshape(len(ix),-1),axis=1)
    denom=np.linalg.norm((U-U.mean(axis=1,keepdims=True)).reshape(len(ix),-1),axis=1)
    ratio=numer/np.maximum(denom,1e-12)
    return result,dict(min_detF=float(detF.min()),max_detF=float(detF.max()),
                       mean_nonaffine_ratio=float(np.mean(ratio)),
                       max_nonaffine_ratio=float(np.max(ratio)),
                       net_internal_force_N=result.sum(axis=0).tolist())

def boundary(X):
    ax=X[:,0]
    b=np.isclose(ax,ax.min(),atol=2e-5)|np.isclose(ax,ax.max(),atol=2e-5)
    if np.sum(b)!=224:raise RuntimeError('Expected 224 interface nodes; got '+str(int(np.sum(b))))
    return b

def pca_force(a,force,anchor,iface):
    coords=np.asarray(anchor['pca_coordinates_'+iface],float)
    basis=np.asarray(anchor['pca_basis_'+iface],float)
    scale=np.asarray(anchor['force_scale'],float)[SLICES[iface]]
    idx=[];used=set()
    for c in coords:
        ds=np.linalg.norm(a['X']-c[None,:],axis=1);i=int(np.argmin(ds))
        if ds[i]>2e-5 or i in used:raise RuntimeError('Cannot match PCA interface node '+iface)
        used.add(i);idx.append(i)
    flattened=force[idx].reshape(-1)
    if basis.shape[0]!=len(flattened) or basis.shape[1]!=len(scale):
        raise RuntimeError('PCA basis/scale incompatibility '+iface)
    if np.any(scale<=0):raise RuntimeError('Nonpositive PCA force scale')
    return basis.T@flattened,scale

def make_case(z,name):
    parts=('node_labels','X','U','RF','element_labels','connectivity','S','Q_global_to_local')
    d={}
    for k in parts:
        key=name+'_'+k
        if key not in z:raise RuntimeError('Missing case array '+key)
        d['Q' if k=='Q_global_to_local' else k]=np.asarray(z[key]).copy()
    check_arrays(d,name)
    return d

def baseline_case(z,q,reg,bc):
    j='B9A10C_%s_R_%s'%(reg,bc)
    d={k:np.asarray(z[j+'_'+k]).copy() for k in
        ('node_labels','X','U','RF','element_labels','connectivity')}
    d['S']=np.asarray(z[j+'_S'])[:,0,:].copy()
    d['Q']=np.asarray(q[j+'_Q_global_to_local'])[:,0,:,:].copy()
    if not np.all(np.asarray(q[j+'_Q_available'])==1):raise RuntimeError('Missing B9A10D Q '+j)
    if not np.array_equal(z[j+'_element_labels'],q[j+'_element_labels']):
        raise RuntimeError('B9A10D element label mismatch '+j)
    check_arrays(d,j)
    return d

def evaluate(d,j,kind,group,reg,anchor):
    f,info=assemble(d)
    RF=d['RF'];b=boundary(d['X']);ref=L2(RF[b]);
    if ref<1e-8:raise RuntimeError('Unusable RF norm '+j)
    delta=f-RF
    record=dict(job=j,kind=kind,condition=group,region='inner_left' if reg=='L' else 'inner_right',
       n_elements=728,n_nodes=1008,reference_boundary_RF_N=ref,
       all_nodes_gap_pct=100*L2(delta)/ref,
       interface_gap_pct=100*L2(delta[b])/ref,
       free_node_gap_pct=100*L2(delta[~b])/ref,
       reference_free_node_RF_pct=100*L2(RF[~b])/ref,
       mean_element_nonaffinity=info['mean_nonaffine_ratio'],
       max_element_nonaffinity=info['max_nonaffine_ratio'],
       min_detF=info['min_detF'],max_detF=info['max_detF'],
       net_force_gap_N=L2(delta.sum(axis=0)),
       residual_virtual_work_Nmm=float(np.sum(delta*d['U'])))
    modes=[]
    for iface in FACES[reg]:
        a,scale=pca_force(d,f,anchor,iface)
        r,_=pca_force(d,RF,anchor,iface)
        for k in range(len(scale)):
            modes.append(dict(job=j,kind=kind,condition=group,interface=iface,mode=k+1,
                fe_force_N=float(a[k]),abaqus_force_N=float(r[k]),
                signed_difference_on_65V_scale=float((a[k]-r[k])/scale[k]),
                abs_difference_on_65V_scale=float(abs((a[k]-r[k])/scale[k]))))
    record['pca_65V_scaled_RMS']=float(np.sqrt(np.mean([m['signed_difference_on_65V_scale']**2 for m in modes])))
    axial=[];X=d['X'][:,0]
    for v in np.sort(np.unique(np.round(X,6))):
        mask=np.abs(X-v)<3e-5
        axial.append(dict(job=j,kind=kind,region=record['region'],axial_x_mm=float(v),
                          node_count=int(mask.sum()),normalized_plane_gap_pct=100*L2(delta[mask])/ref))
    return record,modes,axial

def prepare(root):
    c=root/'B9A10C_exports'/'B9A10C_fields.npz'
    cm=root/'B9A10C_exports'/'B9A10C_windows_summary.json'
    d=root/'B9A10D_exports'/'B9A10D_orientations.npz'
    dm=root/'B9A10D_exports'/'B9A10D_windows_summary.json'
    h=root/'B9A10H_exports'/'B9A10H_independent_fields.npz'
    hm=root/'B9A10H_exports'/'B9A10H_windows_summary.json'
    a=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    p=root/'B9A9_transfer_to_windows'/'B9A9_case_plan.npz'
    for f in (c,cm,d,dm,h,hm,a,p):
        if not f.is_file():raise FileNotFoundError('Missing '+str(f))
    m=json.loads(cm.read_text());md=json.loads(dm.read_text());mh=json.loads(hm.read_text())
    if sha(c)!=m['output_sha256'] or sha(d)!=md['export_sha256'] or sha(cm)!=md['prior_reference_sha256']:
        raise RuntimeError('B9A10C/D baseline file SHA256 mismatch')
    if sha(h)!=mh['output_sha256']:raise RuntimeError('B9A10H exported data checksum mismatch')
    old_manifest=root/'B9A9_exports'/'B9A9_windows_summary.json'
    if not old_manifest.is_file() or sha(old_manifest)!=mh['original_B9A9_manifest_sha256']:
        raise RuntimeError('Independent B9A9 manifest mismatch')
    orig=json.loads(old_manifest.read_text())
    prev={x['job']:x['odb_sha256'] for x in orig['jobs']}
    if set(prev)!=set(NAMES) or len(mh['cases'])!=24:
        raise RuntimeError('Expected 24 independent B9A9 cases')
    for case in mh['cases']:
        if case['odb_sha256']!=prev.get(case['job']):raise RuntimeError('ODB mismatch '+case['job'])
    return c,d,h,a,p,mh

def selftest():
    # One HEX8 affine patch: local sigma rotation + nominal Piola + work balance.
    nodes=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    c=np.cos(.2);t=np.sin(.2)
    q=np.array([[c,-t,0],[t,c,0],[0,0,1.]])
    u=nodes@np.array([[.001,.004,0],[0,-.002,.002],[0,0,.003]])
    jac=nodes.T@grad0();dN=grad0()@np.linalg.inv(jac)
    F=np.eye(3)+u.T@dN;Sig=q.T@sigma([8,4,2,1,.5,.3])@q
    P=np.linalg.det(F)*Sig@np.linalg.inv(F).T
    fe=(P@dN.T).T*(8*np.linalg.det(jac))
    if L2(fe.sum(axis=0))>1e-11:raise RuntimeError('Affine nodal force does not balance')
    test=np.arange(24,dtype=float).reshape(8,3)/13
    work=float(np.sum(fe*test))
    reference=float(8*np.linalg.det(jac)*np.sum(P*(test.T@dN)))
    if abs(work-reference)>1e-9:raise RuntimeError('Virtual work mismatch')
    design=np.column_stack([np.ones(8),nodes])
    a=design@np.linalg.lstsq(design,u,rcond=None)[0]
    if L2(a-u)>1e-13:raise RuntimeError('Affine nonaffinity test failed')
    uu=u.copy();uu[7,0]+=.02
    b=design@np.linalg.lstsq(design,uu,rcond=None)[0]
    if L2(uu-b)<1e-5:raise RuntimeError('Non-affinity test failed')
    print('B9A10H MAC SELFTEST PASS | C3D8R Piola force | local/global rotation | virtual work | non-affinity')

def write_csv(p,rows):
    if not rows:raise RuntimeError('Empty report '+str(p))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(root,check_only):
    c,d,h,anc,plan,mh=prepare(root)
    with np.load(plan,allow_pickle=False) as pz:
        classes=np.asarray(pz['class_names']).astype(str)
        if len(classes)!=12 or set(classes)!={'material','interface','combined'}:
            raise RuntimeError('B9A9 condition classes have unexpected order')
    with np.load(h,allow_pickle=False) as hz:
        for j in NAMES:
            for k in ('node_labels','X','U','RF','element_labels','connectivity','S','Q_global_to_local'):
                if j+'_'+k not in hz:raise RuntimeError('Missing '+j+'_'+k)
    print('B9A10H INPUT CHECK PASS | 24 independent C3D8R fields, true B9A9 provenance, B9A10C/D, saved 65V PCA')
    if check_only:return
    output=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10H_nonlinear_C3D8R'
    if not output.is_dir():raise RuntimeError('Create in Finder first: '+str(output))
    files=[output/f for f in ('B9A10H_case_metrics.csv','B9A10H_modes.csv','B9A10H_axial_zones.csv',
            'B9A10H_summary.json','B9A10H_force_validation.png','B9A10H_nonaffinity.png')]
    if any(p.exists() for p in files):raise RuntimeError('Results already exist; no overwrite. Move them manually to rerun.')
    rows=[];modes=[];zones=[];base={}
    with np.load(c,allow_pickle=False) as cz,np.load(d,allow_pickle=False) as dz,\
         np.load(h,allow_pickle=False) as hz,np.load(anc,allow_pickle=False) as anchor:
        # Baseline reproduces already-established B9A10D results before considering new cases.
        expected={'L':1.1770800278693914,'R':1.173082591821882}
        for r in R:
            for bc in ('FREE','REPLAY'):
                data=baseline_case(cz,dz,r,bc)
                key='B9A10C_%s_R_%s'%(r,bc)
                record,m,z=evaluate(data,key,'baseline',bc,r,anchor)
                exp=expected[r] if bc=='FREE' else {'L':1.1770788954991,'R':1.1730814646256718}[r]
                if abs(record['all_nodes_gap_pct']-exp)>0.005:
                    raise RuntimeError('B9A10D regression mismatch %s %.5f vs %.5f'%(key,record['all_nodes_gap_pct'],exp))
                rows.append(record);modes.extend(m);zones.extend(z);base[key]=record['all_nodes_gap_pct']
                print('B9A10D REGRESSION',key,'%.5f%%'%record['all_nodes_gap_pct'])
        for r in R:
            for num in range(1,13):
                j='B9A9_%s%02d'%(r,num);data=make_case(hz,j)
                record,m,z=evaluate(data,j,'independent',str(classes[num-1]),r,anchor)
                rows.append(record);modes.extend(m);zones.extend(z)
                print('INDEPENDENT',j,record['condition'],'all %.4f%%'%record['all_nodes_gap_pct'],
                      'interface %.4f%%'%record['interface_gap_pct'],
                      'PCA RMS %.4f'%record['pca_65V_scaled_RMS'])
    new=[r for r in rows if r['kind']=='independent']
    agg={}
    for reg in ('inner_left','inner_right'):
        for group in ('material','interface','combined','all'):
            ss=[z for z in new if z['region']==reg and (group=='all' or z['condition']==group)]
            agg[reg+'/'+group]={
              'n':len(ss),'rms_all_pct':float(np.sqrt(np.mean([v['all_nodes_gap_pct']**2 for v in ss]))),
              'rms_interface_pct':float(np.sqrt(np.mean([v['interface_gap_pct']**2 for v in ss]))),
              'rms_free_pct':float(np.sqrt(np.mean([v['free_node_gap_pct']**2 for v in ss]))),
              'rms_PCA_on_65V_scale':float(np.sqrt(np.mean([v['pca_65V_scaled_RMS']**2 for v in ss]))),
              'worst_case_pct':float(max(v['all_nodes_gap_pct'] for v in ss))}
    summary=dict(step='65AM-B9A10-H',source_cases=24,model='C3D8R NLGEOM=YES',
        force_formula='one-point reference weak force, P=det(F)*Q.T*sigma_local*Q*inv(F).T',
        B9A10D_baseline_reproduced=base,independent_aggregates=agg,
        no_stabilization_formula_inferred=True,
        no_ground_truth_hourglass_nodal_forces=True,
        energy_ALLAE_not_treated_as_force=True,
        physics_loss_ready=False,live_Broyden_approved=False,
        models_B5_B9A7_65V_unchanged=True,
        limitations=['Exact local Abaqus stresses come from ODB, not an NO differentiable stress constitutive prediction',
          'Difference from Abaqus RF may include reduced-integration stabilization and other force terms',
          'Free-node Abaqus RF=0 does not itself encode element internal forces',
          'No fitted correction, no zeroing of force errors, no new coupled solution'])
    for p,r in zip(files[:3],(rows,modes,zones)):write_csv(p,r)
    files[3].write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5))
    names=[];vals=[]
    for reg in ('inner_left','inner_right'):
        for group in ('material','interface','combined'):
            names.append(reg.replace('inner_','')+'/'+group)
            vals.append(agg[reg+'/'+group]['rms_interface_pct'])
    ax.bar(np.arange(len(names)),vals);ax.set_xticks(np.arange(len(names)),names,rotation=25,ha='right')
    ax.set_ylabel('Interface nodal-force discrepancy / boundary RF norm (%)')
    ax.set_title('B9A10-H | 24 independent C3D8R nonlinear Abaqus cases')
    fig.tight_layout();fig.savefig(files[4],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for reg in ('inner_left','inner_right'):
        ss=[q for q in new if q['region']==reg]
        ax.scatter([s['mean_element_nonaffinity'] for s in ss],[s['all_nodes_gap_pct'] for s in ss],label=reg)
    ax.set_xlabel('Mean element non-affinity (dimensionless proxy)')
    ax.set_ylabel('All-node force discrepancy / boundary RF norm (%)')
    ax.set_title('B9A10-H | Correlation is diagnostic, not proof of stabilization force')
    ax.legend();fig.tight_layout();fig.savefig(files[5],dpi=160);plt.close(fig)
    print('B9A10H COMPLETE | reports in',output)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(Path(a.project).resolve(),a.check_only)
