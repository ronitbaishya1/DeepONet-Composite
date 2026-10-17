#!/usr/bin/env python3
"""65AM-B9A10-I: read-only, hourglass-COMPATIBLE residual projection.

This is NOT reconstruction or identification of Abaqus hourglass forces.
Uses the original B9A10H data, same P / stress rotation, true RF, PCA scales.
Does not train or alter B5/B9A7/65V. Python 3 / NumPy / matplotlib.

Commands:
  python3 65AM_B9A10I_mac_hourglass_audit.py --selftest
  python3 65AM_B9A10I_mac_hourglass_audit.py --project . --check-only
  python3 65AM_B9A10I_mac_hourglass_audit.py --project .
"""
import argparse, csv, json, hashlib, importlib.util
from pathlib import Path
import numpy as np

NAMES = ['B9A9_%s%02d' % (r, k) for r in 'LR' for k in range(1,13)]
OUT_REL = Path('results/65AM_B9_coupled_feasibility/B9A10I_hourglass_patterns')
OLD_REL = Path('results/65AM_B9_coupled_feasibility/B9A10H_nonlinear_C3D8R')
FILE_NAMES = ['B9A10I_case_metrics.csv','B9A10I_element_modes.csv',
              'B9A10I_pca_modes.csv','B9A10I_summary.json',
              'B9A10I_hourglass_fraction.png','B9A10I_spatial_fraction.png']


def load_h(root):
    filename = root / '65AM_B9A10H_mac_weakform_validation.py'
    if not filename.is_file():
        raise FileNotFoundError('Place the ORIGINAL B9A10H Mac script in project root: '+str(filename))
    spec=importlib.util.spec_from_file_location('b9a10h_reference',filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patches(x):
    """Eight node HEX8 local affine and compatible-hourglass projectors.

    For a one-point brick, the 4 scalar null modes are orthogonal to
    [1,X,Y,Z] at the nodes. This is a DIAGNOSTIC reference-geometry basis.
    Not the proprietary Abaqus stabilization matrix.
    """
    x=np.asarray(x,float)
    if x.ndim!=3 or x.shape[1:]!=(8,3):raise ValueError('Need (ne,8,3) reference nodes')
    d=x-x.mean(axis=1,keepdims=True)
    a=np.concatenate((np.ones((len(x),8,1)),d),axis=2)
    u,s,_=np.linalg.svd(a,full_matrices=True)
    if np.any(s[:,-1]<1e-9):raise RuntimeError('Degenerate affine basis for one or more elements')
    hg=u[:,:,4:]
    if np.max(np.abs(np.einsum('eai,eaj->eij',hg,hg)-np.eye(4)))>1e-8:
        raise RuntimeError('Hourglass frame is not orthonormal')
    if np.max(np.abs(np.einsum('eai,eaj->eij',a,hg)))>1e-7:
        raise RuntimeError('Hourglass vectors do not annihilate affine fields')
    return hg


def mode_scores(X,U,RF,f,node_labels,conn,iface_mask):
    labels={int(v):i for i,v in enumerate(node_labels)}
    idx=np.asarray([[labels[int(v)] for v in row] for row in conn],dtype=int)
    xel=X[idx]; rel=(f-RF)[idx]
    hg=patches(xel)
    coeff=np.einsum('eak,eac->ekc',hg,rel)
    proj=np.einsum('eak,ekc->eac',hg,coeff)
    err=rel-proj
    den=np.einsum('eac,eac->e',rel,rel)
    hn=np.einsum('eac,eac->e',proj,proj)
    an=np.einsum('eac,eac->e',err,err)
    good=np.maximum(den,1e-26)
    if not np.allclose(den,hn+an,rtol=1e-8,atol=1e-8):
        raise RuntimeError('Pythagorean projector work test failed')
    # Measure nonaffinity of the **displacement** on each element as a separate predictor.
    u=U[idx]
    du=np.einsum('eak,eac->ekc',hg,u)
    u0=u-u.mean(axis=1,keepdims=True)
    nonaff=np.linalg.norm(du,axis=(1,2))/np.maximum(np.linalg.norm(u0,axis=(1,2)),1e-12)
    near=iface_mask[idx].any(axis=1)
    # Apparent HG fraction is a LOCAL projection of the ASSEMBLED residual.
    # The same global nodal residual enters multiple adjacent element patches.
    return idx,dict(near=near,den=den,hg=hn,aff=an,
        fractions=hn/good,nonaff=nonaff,
        work_hg_norm_N=np.linalg.norm(coeff,axis=(1,2)))


def evaluate(h,case,z,anchor,group):
    d=h.make_case(z,case)
    ff,info=h.assemble(d)
    x=d['X'];b=h.boundary(x);rf=d['RF']; res=ff-rf
    ref=h.L2(rf[b])
    if ref<=1e-10:raise RuntimeError('Zero Abaqus reaction norm for '+case)
    region='inner_left' if '_L' in case else 'inner_right'
    area='L' if '_L' in case else 'R'
    erows=[]
    ix,sc=mode_scores(x,d['U'],rf,ff,d['node_labels'],d['connectivity'],b)
    xmid=x[ix].mean(axis=1)[:,0]
    for i in range(len(ix)):
        erows.append(dict(job=case,region=region,condition=group,
            element_label=int(d['element_labels'][i]),axial_center_mm=float(xmid[i]),
            touches_interface=bool(sc['near'][i]),
            patch_residual_N2=float(sc['den'][i]),
            hourglass_compatible_N2=float(sc['hg'][i]),
            affine_compatible_N2=float(sc['aff'][i]),
            local_hourglass_fraction=float(sc['fractions'][i]),
            local_nonaffine_displacement=float(sc['nonaff'][i]),
            hg_virtual_work_coefficient_norm_N=float(sc['work_hg_norm_N'][i])))
    def fraction(mask):
        den=float(sc['den'][mask].sum())
        return float(sc['hg'][mask].sum()/den) if den>1e-25 else None
    mrows=[]
    for face in h.FACES[area]:
        p,scale=h.pca_force(d,ff,anchor,face)
        r,_=h.pca_force(d,rf,anchor,face)
        for k in range(len(scale)):
            mrows.append(dict(job=case,region=region,condition=group,interface=face,mode=k+1,
                signed_difference_on_65V_scale=float((p[k]-r[k])/scale[k]),
                abs_difference_on_65V_scale=float(abs(p[k]-r[k])/scale[k])))
    row=dict(job=case,region=region,condition=group,
        gap_all_pct=100*h.L2(res)/ref,
        gap_interface_pct=100*h.L2(res[b])/ref,
        gap_free_pct=100*h.L2(res[~b])/ref,
        pca_mode_gap_rms=float(np.sqrt(np.mean([m['signed_difference_on_65V_scale']**2 for m in mrows]))),
        local_hourglass_fraction_all=fraction(np.ones(len(sc['near']),dtype=bool)),
        local_hourglass_fraction_bulk=fraction(~sc['near']),
        local_hourglass_fraction_boundary=fraction(sc['near']),
        n_bulk_elements=int((~sc['near']).sum()),
        n_boundary_adjacent_elements=int(sc['near'].sum()),
        mean_displacement_nonaffinity=float(sc['nonaff'].mean()),
        mean_element_hourglass_fraction=float(sc['fractions'].mean()),
        max_element_hourglass_fraction=float(sc['fractions'].max()),
        boundary_RF_N=float(ref))
    return row,erows,mrows


def sha(p):
    hash_=hashlib.sha256()
    with open(p,'rb') as fd:
        for chunk in iter(lambda:fd.read(1<<20),b''):hash_.update(chunk)
    return hash_.hexdigest()


def write_csv(path,rows):
    if not rows:raise RuntimeError('No output rows: '+str(path))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def old_metrics(path):
    if not path.is_file():raise FileNotFoundError('Missing original B9A10H case metrics: '+str(path))
    with open(path,newline='') as f:
        records=list(csv.DictReader(f))
    old={r['job']:r for r in records if r['kind']=='independent'}
    if len(old)!=24:raise RuntimeError('B9A10H case metrics must contain exactly 24 independent cases')
    return old


def optional_energy(root,expected):
    target=root/'B9A10I_energy_exports'/'B9A10I_energy_history.csv'
    manifest=target.parent/'B9A10I_energy_summary.json'
    if not target.exists() and not manifest.exists():return None
    if not target.exists() or not manifest.exists():raise RuntimeError('Incomplete optional Windows energy export')
    meta=json.loads(manifest.read_text())
    if sha(target)!=meta['csv_sha256']:raise RuntimeError('Energy CSV checksum mismatch')
    if meta['source_manifest_sha256']!=sha(root/'B9A9_exports'/'B9A9_windows_summary.json'):
        raise RuntimeError('Optional energy export does not match B9A9 manifest')
    with open(target,newline='') as f:rows=list(csv.DictReader(f))
    if {r['job'] for r in rows}!=set(expected):raise RuntimeError('Energy CSV jobs mismatch')
    result={}
    for r in rows:
        def num(k):
            try:return float(r[k]) if r[k]!='' else None
            except (TypeError,ValueError):return None
        ae=num('ALLAE');ie=num('ALLIE')
        result[r['job']]={'ALLAE':ae,'ALLIE':ie,
          'ratio_pct':(100*abs(ae)/abs(ie) if ae is not None and ie is not None and abs(ie)>1e-14 else None)}
    return result


def selftest():
    # One regular reference brick, arbitrary affine and pure hourglass nodal signals.
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],
                  [1,-1,1],[1,1,1],[-1,1,1]],float)
    X=0.5*(signs+1)
    hb=patches(X[None,:,:])[0]
    A=np.concatenate((np.ones((8,1)),X-X.mean(axis=0)),axis=1)
    assert np.linalg.norm(hb.T@A)<1e-10
    hgforce=hb@np.array([[1,2,3],[4,3,2],[1,0,1],[2,4,0]],float)
    affforce=A@np.array([[1,0,3],[2,3,0],[0,1,2],[4,0,1]],float)
    for force,want in [(hgforce,1.),(affforce,0.)]:
        a=hb@(hb.T@force)
        score=np.sum(a*a)/np.sum(force*force)
        assert abs(score-want)<1e-10
    # Rotated/sheared distorted brick should still annihilate affine displacement exactly.
    Y=X@np.array([[1.,0.2,0.],[0.,1.1,.12],[0.05,0.,.9]])
    hb2=patches(Y[None,:,:])[0]
    assert np.linalg.norm(hb2.T@(Y@np.array([[2,.1,.3],[.1,.4,.2],[.5,.2,.1]])+1.7))<1e-10
    print('B9A10I SELFTEST PASS | 12 compatible HG DOFs | affine null | orthogonal work | distorted brick')


def go(root,check_only=False):
    h=load_h(root)
    c,d,p,a,plan,m=h.prepare(root) # original H's provenance checks; preserve them
    hsummary=root/OLD_REL/'B9A10H_summary.json'
    if not hsummary.is_file():raise FileNotFoundError('Missing original H summary '+str(hsummary))
    hs=json.loads(hsummary.read_text())
    if hs.get('source_cases')!=24 or hs.get('model')!='C3D8R NLGEOM=YES':
        raise RuntimeError('Wrong original B9A10H summary')
    records=old_metrics(root/OLD_REL/'B9A10H_case_metrics.csv')
    with np.load(plan,allow_pickle=False) as z:
        classes=np.asarray(z['class_names']).astype(str)
        if len(classes)!=12:raise RuntimeError('Invalid B9A9 case classifications')
    with np.load(p,allow_pickle=False) as z:
        for j in NAMES:
            for suffix in ('node_labels','X','U','RF','element_labels','connectivity','S','Q_global_to_local'):
                if j+'_'+suffix not in z:raise RuntimeError('Missing B9A10H field '+j+'_'+suffix)
    e=optional_energy(root,NAMES)
    print('B9A10I CHECK PASS | 24 independent H cases | checksums & saved 65V scales verified')
    print('Optional energy audit:', 'AVAILABLE' if e is not None else 'NOT PROVIDED (allowed)')
    if check_only:return
    out=root/OUT_REL
    if not out.is_dir():raise RuntimeError('Create empty output folder in Finder first: '+str(out))
    for name in FILE_NAMES:
        if (out/name).exists():raise RuntimeError('Existing report protected; move it in Finder before rerunning: '+name)
    cases=[];elements=[];pmodes=[]
    with np.load(p,allow_pickle=False) as z,np.load(a,allow_pickle=False) as anc:
        for job in NAMES:
            k=int(job[-2:])-1
            d=h.make_case(z,job)
            row,els,ms=evaluate(h,job,z,anc,str(classes[k]))
            orig=records[job]
            # Numerical regression: use exact H force reconstruction, same normalizer.
            if abs(row['gap_all_pct']-float(orig['all_nodes_gap_pct']))>1e-6:
                raise RuntimeError('B9A10H regression failed '+job+' '+str(row['gap_all_pct']))
            if e is not None:
                row['ALLAE_over_ALLIE_pct']=e[job]['ratio_pct']
                row['ALLAE_Nmm']=e[job]['ALLAE']
                row['ALLIE_Nmm']=e[job]['ALLIE']
            else:
                row['ALLAE_over_ALLIE_pct']=None
                row['ALLAE_Nmm']=None
                row['ALLIE_Nmm']=None
            cases.append(row);elements.extend(els);pmodes.extend(ms)
            print('AUDIT',job,'gap %.4f%%'%row['gap_all_pct'],
                  'local HG share %.1f%%'%(100*row['local_hourglass_fraction_all']),
                  'bulk HG share', ('%.1f%%'%(100*row['local_hourglass_fraction_bulk'])
                     if row['local_hourglass_fraction_bulk'] is not None else 'n/a'))
    group_stats={}
    for r in ('inner_left','inner_right'):
        for category in ('material','interface','combined','all'):
            group=[x for x in cases if x['region']==r and (category=='all' or x['condition']==category)]
            group_stats[r+'/'+category]={
                'n':len(group),
                'mean_patch_hourglass_compatible_fraction':float(np.mean([x['local_hourglass_fraction_all'] for x in group])),
                'mean_bulk_fraction':float(np.mean([x['local_hourglass_fraction_bulk'] for x in group
                                                  if x['local_hourglass_fraction_bulk'] is not None])),
                'mean_boundary_fraction':float(np.mean([x['local_hourglass_fraction_boundary'] for x in group
                                                  if x['local_hourglass_fraction_boundary'] is not None])),
                'mean_nodal_force_gap_pct':float(np.mean([x['gap_all_pct'] for x in group])),
                'mean_displacement_nonaffinity':float(np.mean([x['mean_displacement_nonaffinity'] for x in group]))}
    top=sorted(pmodes,key=lambda x:x['abs_difference_on_65V_scale'],reverse=True)[:12]
    summary={'step':'65AM-B9A10-I','source':'24 B9A9 independent cases, B9A10H unchanged',
        'method':'SVD local 8-node affine-null projections of ASSEMBLED nodal residual',
        'not_an_hourglass_force_recovery':True,
        'critical_caveat':'Patch-local projections of assembled forces include neighboring elements; percentages are NON-ADDITIVE and do not identify Abaqus stabilization.',
        'group_statistics':group_stats,'largest_PCA_mode_errors':top,
        'optional_Abaqus_energy_provided':e is not None,
        'ALLAE_not_a_nodal_force':True,
        'physics_loss_ready':False,'B5_B9A7_65V_unchanged':True,
        'next_gate':'Only a controlled independent stabilization switch/scaling experiment or independently accessible element internal force can identify stabilization causally.'}
    # Create plots before writing any reports; no modifying other project files.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[];vals=[];vals2=[]
    for region in ('inner_left','inner_right'):
        for cat in ('material','interface','combined'):
            keys.append(('left' if region=='inner_left' else 'right')+'/'+cat)
            g=group_stats[region+'/'+cat]
            vals.append(100*g['mean_bulk_fraction']);vals2.append(100*g['mean_boundary_fraction'])
    fig,ax=plt.subplots(figsize=(11,5));pos=np.arange(len(keys));w=.38
    ax.bar(pos-w/2,vals,w,label='Interior-element patches')
    ax.bar(pos+w/2,vals2,w,label='Interface-adjacent patches')
    ax.set_xticks(pos,keys,rotation=20,ha='right')
    ax.set_ylabel('Local hourglass-compatible share of squared assembled residual (%)')
    ax.set_title('B9A10-I | Patch projection only; not Abaqus hourglass force')
    ax.legend();fig.tight_layout();fig.savefig(out/FILE_NAMES[4],dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,region in zip(axes,('inner_left','inner_right')):
        ex=[x for x in elements if x['region']==region]
        xx=sorted(set(round(x['axial_center_mm'],5) for x in ex))
        ys=[]
        for xv in xx:
            sub=[x for x in ex if abs(x['axial_center_mm']-xv)<1e-4]
            hn=sum(x['hourglass_compatible_N2'] for x in sub)
            den=sum(x['patch_residual_N2'] for x in sub)
            ys.append(100*hn/max(den,1e-25))
        ax.plot(xx,ys,marker='o');ax.set_xlabel('Axial element center X (mm)')
        ax.set_title(region)
    axes[0].set_ylabel('Local HG-compatible share (%)')
    fig.suptitle('B9A10-I | Axial pattern of ASSEMBLED force difference across 12 conditions')
    fig.tight_layout();fig.savefig(out/FILE_NAMES[5],dpi=170);plt.close(fig)
    write_csv(out/FILE_NAMES[0],cases)
    write_csv(out/FILE_NAMES[1],elements)
    write_csv(out/FILE_NAMES[2],pmodes)
    (out/FILE_NAMES[3]).write_text(json.dumps(summary,indent=2))
    print('B9A10I DONE | saved 6 reports to',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',default='.')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:selftest()
    else:go(Path(args.project).expanduser().resolve(),args.check_only)
