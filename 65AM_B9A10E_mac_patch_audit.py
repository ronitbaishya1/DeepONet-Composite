#!/usr/bin/env python3
"""B9A10-E Mac: independent C3D8/C3D8R single-element force patch audit.

Inputs:
 B9A10E_exports/B9A10E_patch_fields.npz
 B9A10E_exports/B9A10E_windows_summary.json
Output folder must be created manually in Finder:
 results/65AM_B9_coupled_feasibility/B9A10E_element_patch_audit

Does not alter B5, B9A7, 65V, or old Abaqus jobs. All 8 element nodes
are prescribed: RF is a nodal internal force check, not a free equilibrium test.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

KINDS=[(t,m,k) for t in ('R','F') for m in ('ISO','ORTHO') for k in ('AFF','NONAFF')]
TYPES={'R':'C3D8R','F':'C3D8'}
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
G=1/np.sqrt(3.)
GAUSS=np.array(list(itertools.product((-G,G),repeat=3)),float)
CASE_FIELDS=('X','U','RF','element_connectivity','S','IP_COORD','Q','Q_available')

def name(t,m,k):return 'B9A10E_%s_%s_%s'%(t,m,k)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def grad_nat(q):
    a,b,c=np.asarray(q,float)
    t=SIGNS
    return np.stack((t[:,0]*(1+t[:,1]*b)*(1+t[:,2]*c),
                     t[:,1]*(1+t[:,0]*a)*(1+t[:,2]*c),
                     t[:,2]*(1+t[:,0]*a)*(1+t[:,1]*b)),axis=1)/8.

def shape(q):return np.prod(1+SIGNS*np.asarray(q)[None,:],axis=1)/8.

def tensor(s):
    a=np.asarray(s,float).reshape(6)
    return np.array([[a[0],a[3],a[4]], [a[3],a[1],a[5]], [a[4],a[5],a[2]]])

def mag(a):return float(np.linalg.norm(a))

def load_case(z,j,t):
    d={field:np.asarray(z[j+'_'+field]).copy() for field in CASE_FIELDS}
    nip=1 if t=='R' else 8
    if d['X'].shape!=(8,3) or d['U'].shape!=(8,3) or d['RF'].shape!=(8,3):
        raise RuntimeError('Invalid 8-node array shape for '+j)
    if d['element_connectivity'].shape!=(8,) or not np.array_equal(d['element_connectivity'],np.arange(1,9)):
        raise RuntimeError('Unexpected Abaqus node connectivity '+j)
    if (d['S'].shape!=(nip,6) or d['IP_COORD'].shape!=(nip,3) or
        d['Q'].shape!=(nip,3,3) or d['Q_available'].shape!=(nip,)):
        raise RuntimeError('Unexpected Abaqus integration point arrays '+j)
    for v in d.values():
        if not np.all(np.isfinite(v)):
            raise RuntimeError('Nonfinite entries in '+j)
    return d

def assemble(d,t,rotation=True):
    """Direct Piola and updated-coordinate Cauchy element internal forces.

    Reorders S and material rotation by matched physical integration-point
    COORD, not by Abaqus's integration point number.
    """
    x=d['X'];u=d['U'];xd=x+u
    ip=d['IP_COORD'];n=1 if t=='R' else 8
    pts=np.zeros((1,3),float) if n==1 else GAUSS
    exact=np.array([shape(pt).dot(xd) for pt in pts])
    distances=np.linalg.norm(exact[:,None,:]-ip[None,:,:],axis=2)
    idx=np.argmin(distances,axis=1)
    mis=float(np.max(distances[np.arange(n),idx]))
    if mis>2e-5 or len(set(idx.tolist()))!=n:
        raise RuntimeError('IP location or ordering is invalid: max mismatch %.6g mm'%mis)
    fpiola=np.zeros((8,3));fcauchy=np.zeros((8,3))
    minF=100.;maxF=-100.;rotangles=[]
    for nq,pt in enumerate(pts):
        a=int(idx[nq]);Dnat=grad_nat(pt)
        Jac0=x.T@Dnat
        det0=float(np.linalg.det(Jac0))
        if det0<=1e-14:raise RuntimeError('Bad reference element Jacobian')
        gradX=Dnat@np.linalg.inv(Jac0)
        F=np.eye(3)+u.T@gradX
        detF=float(np.linalg.det(F))
        if detF<=0:raise RuntimeError('Inverted deformation')
        minF=min(minF,detF);maxF=max(maxF,detF)
        sig=tensor(d['S'][a])
        if rotation:
            if d['Q_available'][a]:
                Q=d['Q'][a]
                if (np.max(abs(Q@Q.T-np.eye(3)))>1e-3 or
                    abs(np.linalg.det(Q)-1)>1e-3):
                    raise RuntimeError('Nonorthogonal Abaqus material orientation')
                sig=Q.T@sig@Q
                angle=np.arccos(np.clip((np.trace(Q)-1)/2,-1.,1.))
                rotangles.append(float(np.rad2deg(angle)))
        P=detF*sig@np.linalg.inv(F).T
        w=8. if n==1 else 1.
        fpiola+=(P@gradX.T).T*(det0*w)
        JacCur=xd.T@Dnat
        detcur=float(np.linalg.det(JacCur))
        gradx=Dnat@np.linalg.inv(JacCur)
        fcauchy+=(sig@gradx.T).T*(detcur*w)
    identity_gap=mag(fpiola-fcauchy)
    if identity_gap>1.e-8*max(1.,mag(fpiola)):
        raise RuntimeError('Reference/spatial virtual work transformations disagree')
    return fpiola,dict(max_IP_mismatch_mm=mis,detF_min=minF,detF_max=maxF,
                       max_material_rotation_deg=max(rotangles) if rotangles else None,
                       spatial_vs_piola_gap_N=identity_gap,
                       resultant_force_N=fpiola.sum(axis=0).tolist())

def inputs(root):
    data=root/'B9A10E_exports'/'B9A10E_patch_fields.npz'
    summ=root/'B9A10E_exports'/'B9A10E_windows_summary.json'
    if not data.is_file() or not summ.is_file():
        raise FileNotFoundError('Copy complete B9A10E_exports from Windows first')
    s=json.loads(summ.read_text())
    if digest(data)!=s.get('exports_sha256'):
        raise RuntimeError('Export NPZ SHA256 mismatch')
    jobs={row['job']:row for row in s['cases']}
    expected={name(*x) for x in KINDS}
    if set(jobs)!=expected or s.get('all_nodes_prescribed') is not True:
        raise RuntimeError('Invalid eight-case manifest')
    return data,summ,jobs

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)

def run(root,check_only=False):
    inp,meta,jobs=inputs(root)
    with np.load(inp,allow_pickle=False) as z:
        for t,m,k in KINDS:
            j=name(t,m,k)
            d=load_case(z,j,t)
            if m=='ORTHO' and not d['Q_available'].all():
                raise RuntimeError('Missing local-to-global orientation for '+j)
            if jobs[j]['n_ip']!=d['S'].shape[0]:raise RuntimeError('IP count manifest mismatch '+j)
            print('CHECK',j,'IP',d['S'].shape[0],'rotations',int(d['Q_available'].sum()))
    print('B9A10E INPUT CHECK PASS')
    if check_only:return
    out=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10E_element_patch_audit'
    if not out.is_dir():
        raise RuntimeError('Using Finder create output folder '+str(out))
    targets=[out/a for a in ('B9A10E_cases.csv','B9A10E_nodes.csv','B9A10E_summary.json','B9A10E_force_comparison.png')]
    if any(a.exists() for a in targets):
        raise RuntimeError('Outputs already exist; move previous outputs manually if rerunning')
    cases=[];nodes=[]
    with np.load(inp,allow_pickle=False) as z:
        for t,m,k in KINDS:
            j=name(t,m,k)
            d=load_case(z,j,t)
            F,diag=assemble(d,t,rotation=True)
            Fraw,_=assemble(d,t,rotation=False)
            R=d['RF']
            norm=max(mag(R),1e-12)
            gap=100*mag(F-R)/norm
            oldgap=100*mag(Fraw-R)/norm
            maxnode=max(mag(F[i]-R[i]) for i in range(8))
            energy=jobs[j].get('energy',{})
            ae=energy.get('ALLAE',{}).get('value')
            ie=energy.get('ALLIE',{}).get('value')
            ratio=None if ae is None or ie is None or abs(ie)<1.e-15 else 100*abs(ae)/abs(ie)
            row=dict(job=j,element=TYPES[t],material=m,pattern=k,
                force_gap_corrected_pct=gap,force_gap_raw_pct=oldgap,
                abaqus_RF_norm_N=mag(R),max_node_force_gap_N=maxnode,
                net_Abaqus_RF_N=mag(R.sum(axis=0)),
                max_IP_match_mm=diag['max_IP_mismatch_mm'],
                min_detF=diag['detF_min'],max_detF=diag['detF_max'],
                max_rotation_deg=diag['max_material_rotation_deg'],
                spatial_vs_piola_gap_N=diag['spatial_vs_piola_gap_N'],
                ALLAE_to_ALLIE_pct=ratio,
                patch_consistency_0p5pct=(gap<=0.5))
            cases.append(row)
            for i in range(8):
                nodes.append(dict(job=j,node=i+1,X=float(d['X'][i,0]),
                    Y=float(d['X'][i,1]),Z=float(d['X'][i,2]),
                    Abaqus_RX=float(R[i,0]),Abaqus_RY=float(R[i,1]),
                    Abaqus_RZ=float(R[i,2]),
                    weak_RX=float(F[i,0]),weak_RY=float(F[i,1]),weak_RZ=float(F[i,2]),
                    node_error_norm_N=mag(F[i]-R[i])))
            print('%-26s | rotated gap %8.4f%% | raw %8.4f%% | IP %.2g mm'%
                  (j,gap,oldgap,row['max_IP_match_mm']))
    write_csv(targets[0],cases)
    write_csv(targets[1],nodes)
    report={'step':'65AM-B9A10-E', 'purpose':'Isolated element stress/orientation/quadrature/assembly patch audit',
            'all_nodes_prescribed':True,'comparison_scale':'each one-element job Abaqus RF L2 norm',
            'patch_0p5pct_threshold_diagnostic_only':True,
            'cases':{r['job']:{k:v for k,v in r.items() if k!='job'} for r in cases},
            'affine_patch_pass_all':all(r['patch_consistency_0p5pct'] for r in cases if r['pattern']=='AFF'),
            'nonaffine_pass_all':all(r['patch_consistency_0p5pct'] for r in cases if r['pattern']=='NONAFF'),
            'next':'If affine patch fails, fix basic assembly; if affine passes but nonlinear C3D8 fails, inspect local-to-global and IP stress output conventions on whole mesh.',
            'not_a_coupled_solution':True,'physics_loss_ready':False,'B5_and_65V_unchanged':True,
            'warnings':['A single-element patch does not establish whole-mesh force consistency',
                        'C3D8R nonaffine tests may include hourglass stabilization forces not in stress integral',
                        'C3D8 nonaffine tests may differ due to Abaqus discrete formulation',
                        'Energy ALLAE is not a direct measure of nodal hourglass force']}
    targets[2].write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5.5))
    x=np.arange(len(cases));width=.37
    ax.bar(x-width/2,[r['force_gap_raw_pct'] for r in cases],width,label='Raw material-local stress')
    ax.bar(x+width/2,[r['force_gap_corrected_pct'] for r in cases],width,label='Corrected global stress')
    ax.axhline(.5,ls='--',color='black',lw=1,label='Patch diagnostic 0.5%')
    ax.set_xticks(x,[r['element'].replace('C3D8','8')+'/'+r['material']+'/'+r['pattern'] for r in cases],rotation=35,ha='right')
    ax.set_ylabel('Absolute nodal force gap / Abaqus RF norm (%)')
    ax.set_title('B9A10-E | Single-element C3D8/C3D8R force patch tests')
    ax.legend();fig.tight_layout();fig.savefig(targets[3],dpi=180);plt.close(fig)
    print('B9A10E REPORT WRITTEN:',out)
    print('AFFINE PATCH PASS (diagnostic):',report['affine_patch_pass_all'])


def selftest():
    x=(SIGNS+1)*np.array([.6,.5,.62])[None,:]/2
    u=x@np.array([[.004,.002,.001],[-.001,-.002,.001],[.0005,-.0008,.003]]).T
    qang=np.deg2rad(17);c=np.cos(qang);s=np.sin(qang)
    Q=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
    S_local=np.array([12.,8.,6.,2.,1.,.5])
    # Independent analytic constant spatial stress test for an affine element.
    for t in ('R','F'):
        points=[np.zeros(3)] if t=='R' else GAUSS
        ip=np.array([shape(p).dot(x+u) for p in points])
        d=dict(X=x,U=u,RF=np.zeros((8,3)),S=np.tile(S_local,(len(points),1)),
            IP_COORD=ip,Q=np.tile(Q,(len(points),1,1)),Q_available=np.ones(len(points),np.uint8))
        f,checks=assemble(d,t)
        if mag(f.sum(axis=0))>1.e-10:raise RuntimeError('Element nodal balance failed')
        if checks['spatial_vs_piola_gap_N']>1.e-10:raise RuntimeError('Spatial form error')
        if t=='R':first=f
        elif mag(first-f)>1.e-9:raise RuntimeError('Affine one- and eight-point quadrature differ')
        # Virtual work independent of energy: nodal forces · virtual displacement
        rnd=np.random.default_rng(19).normal(size=(8,3))
        virtual=0.
        sig=Q.T@tensor(S_local)@Q
        for p in points:
            grad=grad_nat(p);J0=x.T@grad
            GX=grad@np.linalg.inv(J0)
            F=np.eye(3)+u.T@GX
            P=np.linalg.det(F)*sig@np.linalg.inv(F).T
            w=8. if t=='R' else 1.
            virtual+=np.sum(P*(rnd.T@GX))*np.linalg.det(J0)*w
        if abs(float(np.sum(f*rnd))-virtual)>1e-8:raise RuntimeError('Virtual work mismatch')
    print('B9A10E MAC SELFTEST PASS | local frames | spatial/Piola | 1/8 point affine | virtual work')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='.')
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest()
    else:run(Path(a.project).expanduser().resolve(),a.check_only)
