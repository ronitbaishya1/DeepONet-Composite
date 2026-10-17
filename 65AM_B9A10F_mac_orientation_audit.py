#!/usr/bin/env python3
"""65AM-B9A10-F Mac force audit. Independent one-brick diagnostics only.

Requires B9A10F_exports/B9A10F_patch_fields.npz and ...windows_summary.json.
Create results/65AM_B9_coupled_feasibility/B9A10F_orientation_amplitude/ manually.
B5, B9A7 and 65V are not loaded, trained or modified.
"""
import argparse, csv, hashlib, itertools, json, math
from pathlib import Path
import numpy as np

ANGLES=('Z00','P17','M17','ISO')
AMPS={'A00':0.0,'A10':0.1,'A50':0.5,'A100':1.0}
CASES=[('F',a,v) for a in ANGLES for v in AMPS]
CASES += [('F',a,'LIN') for a in ANGLES]
CASES += [('R','P17','A100')]
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
G=1.0/np.sqrt(3.0)
GAUSS=np.array(list(itertools.product((-G,G),repeat=3)),float)
FIELDS=('X','U','RF','S','Q','Q_available','IP_COORD','connectivity')


def case_name(t,a,v):return 'B9A10F_%s_%s_%s'%(t,a,v)

def config(t,a,v):
    return dict(nlgeom=v!='LIN',material='ISO' if a=='ISO' else 'ORTHO',
                alpha=1.0 if v=='LIN' else AMPS[v],angle={'Z00':0.,'P17':17.,'M17':-17.,'ISO':None}[a])

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def shape(q):return np.prod(1.+SIGNS*np.asarray(q)[None,:],axis=1)/8.

def natural_grad(q):
    a,b,c=np.asarray(q,float)
    t=SIGNS
    return np.column_stack((t[:,0]*(1+t[:,1]*b)*(1+t[:,2]*c),
        t[:,1]*(1+t[:,0]*a)*(1+t[:,2]*c),
        t[:,2]*(1+t[:,0]*a)*(1+t[:,1]*b)))/8.

def voigt_to_tensor(s):
    a=np.asarray(s,dtype=float).reshape(6)
    # Abaqus 3D stress component order S11,S22,S33,S12,S13,S23
    return np.array([[a[0],a[3],a[4]], [a[3],a[1],a[5]], [a[4],a[5],a[2]]])

def norm(a):return float(np.linalg.norm(a))

def load(z,j,t):
    x={field:np.asarray(z[j+'_'+field]).copy() for field in FIELDS}
    nip=1 if t=='R' else 8
    if any(x[f].shape!=(8,3) for f in ('X','U','RF')):
        raise RuntimeError('Incorrect node vectors '+j)
    if x['S'].shape!=(nip,6) or x['Q'].shape!=(nip,3,3) or x['IP_COORD'].shape!=(nip,3):
        raise RuntimeError('Incorrect integration point data '+j)
    if x['Q_available'].shape!=(nip,) or not np.array_equal(x['connectivity'],np.arange(1,9)):
        raise RuntimeError('Wrong local frames or connectivity '+j)
    if any(not np.isfinite(v).all() for v in x.values()):
        raise RuntimeError('Nonfinite data in '+j)
    return x

def ip_match(data,t,nlgeom):
    pts=np.zeros((1,3)) if t=='R' else GAUSS
    reference=np.array([shape(p)@data['X'] for p in pts])
    current=np.array([shape(p)@(data['X']+data['U']) for p in pts])
    coords=data['IP_COORD']
    candidates=[]
    for tag,exp in (('reference',reference),('current',current)):
        dis=np.linalg.norm(exp[:,None,:]-coords[None,:,:],axis=2)
        ix=np.argmin(dis,axis=1)
        maximum=float(np.max(dis[np.arange(len(pts)),ix]))
        if len(set(ix.tolist()))==len(pts):candidates.append((maximum,tag,ix))
    if not candidates:raise RuntimeError('IP positions are not one-to-one')
    mis,frame,idx=min(candidates,key=lambda v:v[0])
    if mis>2.e-5:
        raise RuntimeError('Abaqus IP geometry mismatch (%.3g mm)'%mis)
    return pts,idx,mis,frame


def calculate(data,t,nlgeom,rotate=True,individual=False):
    """Assemble f_a=∫_Ω0 P·GradN dV (NLGEOM=YES),
       or f_a=∫_Ω0 sigma·GradN dV (NLGEOM=NO small strain).
       Q is global-to-local, per output FieldValue.localCoordSystem.
    """
    X=data['X'];U=data['U'];pts,ids,mis,frame=ip_match(data,t,nlgeom)
    forces=np.zeros((8,3)); all_ips=[];min_detF=np.inf; max_detF=-np.inf
    sigma_rotation_change=[]
    for q,pt in enumerate(pts):
        ind=int(ids[q]); g0=natural_grad(pt)
        J0=X.T@g0;det0=float(np.linalg.det(J0))
        if det0<=0:raise RuntimeError('Bad Jacobian')
        gradX=g0@np.linalg.inv(J0)
        F=np.eye(3)+U.T@gradX
        J=float(np.linalg.det(F))
        if J<=0:raise RuntimeError('Inverted deformation')
        min_detF=min(min_detF,J); max_detF=max(max_detF,J)
        sloc=voigt_to_tensor(data['S'][ind]);sglob=sloc
        Q=data['Q'][ind]
        has=bool(data['Q_available'][ind])
        if has:
            if abs(np.linalg.det(Q)-1.)>1e-3 or np.max(np.abs(Q@Q.T-np.eye(3)))>1e-3:
                raise RuntimeError('Invalid rotation matrix')
            sglob=Q.T@sloc@Q
            sigma_rotation_change.append(norm(sglob-sloc))
        if not rotate:
            sglob=sloc
        if nlgeom:
            P=J*sglob@np.linalg.inv(F).T
        else:
            P=sglob   # small-strain Abaqus: reference shape-gradient times Cauchy stress
        weight=8. if t=='R' else 1.
        f=(P@gradX.T).T*(det0*weight)
        forces+=f
        all_ips.append(dict(ip=q+1,odb_ip=ind+1,detF=J,
                            detJ0=det0,rotation_deg=(None if not has else float(np.degrees(
                                np.arccos(np.clip((np.trace(Q)-1.)/2.,-1.,1.))))),
                            stress_local_norm=norm(sloc),
                            stress_rotation_difference_norm=norm(Q.T@sloc@Q-sloc) if has else 0.,
                            force_contribution_norm=norm(f),
                            max_nodal_contribution= float(np.max(np.linalg.norm(f,axis=1)))))
    return forces,dict(ip_geometry_error_mm=mis,ip_coordinates_frame=frame,
        min_detF=float(min_detF),max_detF=float(max_detF),
        net_internal_force_N=norm(forces.sum(axis=0)),
        mean_stress_rotation_change= float(np.mean(sigma_rotation_change)) if sigma_rotation_change else 0.),all_ips


def required_inputs(root):
    folder=root/'B9A10F_exports'
    data=folder/'B9A10F_patch_fields.npz';summary=folder/'B9A10F_windows_summary.json'
    if not data.is_file() or not summary.is_file():
        raise FileNotFoundError('Transfer B9A10F_exports folder from Windows to project root')
    m=json.loads(summary.read_text())
    if digest(data)!=m.get('exports_sha256'):
        raise RuntimeError('B9A10F SHA256 mismatch: export incomplete/modified')
    if not m.get('all_nodes_prescribed') or m.get('case_count')!=21:
        raise RuntimeError('Expected B9A10F 21-case fully prescribed reference')
    metadata={r['job']:r for r in m['cases']}
    if set(metadata)!=set(case_name(*c) for c in CASES):
        raise RuntimeError('B9A10F manifest case names mismatch')
    return data,metadata


def csv_write(path, rows):
    if not rows:raise RuntimeError('Cannot write empty CSV')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


def run(root,check_only=False):
    inputpath,meta=required_inputs(root)
    with np.load(inputpath,allow_pickle=False) as z:
        for t,a,v in CASES:
            j=case_name(t,a,v);d=load(z,j,t)
            if a!='ISO' and not np.all(d['Q_available']):
                raise RuntimeError('Missing orthotropic local stress orientation in '+j)
            if int(meta[j]['n_ip'])!=(1 if t=='R' else 8):
                raise RuntimeError('Manifest IP count invalid for '+j)
            _=ip_match(d,t,v!='LIN')
            print('CHECK OK',j)
    print('B9A10F 21-CASE CHECK PASS')
    if check_only:return
    out=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10F_orientation_amplitude'
    if not out.is_dir():
        raise RuntimeError('Using Finder create output folder first: '+str(out))
    names=('B9A10F_case_metrics.csv','B9A10F_ip_diagnostics.csv',
           'B9A10F_nodes.csv','B9A10F_summary.json','B9A10F_force_sweep.png',
           'B9A10F_nonaffine_nodes.png')
    paths=[out/s for s in names]
    if any(f.exists() for f in paths):
        raise RuntimeError('Previous outputs protected; do not overwrite. In Finder move only old B9A10F result files before rerunning.')
    rows=[];iprows=[];noderows=[]
    with np.load(inputpath,allow_pickle=False) as z:
        for t,a,v in CASES:
            j=case_name(t,a,v);c=config(t,a,v);d=load(z,j,t)
            model,diag,ips=calculate(d,t,c['nlgeom'],True)
            wrong,_,_=calculate(d,t,c['nlgeom'],False)
            R=d['RF']; ref=max(norm(R),1.e-12)
            err=100*norm(model-R)/ref
            wrongerr=100*norm(wrong-R)/ref
            ae=meta[j]['energy'].get('ALLAE',{}).get('value')
            ie=meta[j]['energy'].get('ALLIE',{}).get('value')
            aepercent=None if ae is None or ie is None or abs(ie)<1.e-15 else 100*abs(ae)/abs(ie)
            one=dict(job=j,element='C3D8R' if t=='R' else 'C3D8',
                material=c['material'],angle_code=a,angle_deg=c['angle'],
                mode='finite' if c['nlgeom'] else 'small_strain',
                warping_alpha=c['alpha'],force_error_global_pct=err,
                force_error_raw_pct=wrongerr,
                RF_norm_N=norm(R),peak_nodal_error_N=float(np.max(np.linalg.norm(model-R,axis=1))),
                ALLAE_over_ALLIE_pct=aepercent,ip_coord_frame=diag['ip_coordinates_frame'],
                ip_mismatch_mm=diag['ip_geometry_error_mm'],
                min_detF=diag['min_detF'],max_detF=diag['max_detF'],
                net_internal_force_N=diag['net_internal_force_N'],
                mean_rotation_effect=diag['mean_stress_rotation_change'])
            rows.append(one)
            for ip in ips:
                iprows.append(dict(job=j,**ip))
            for i in range(8):
                noderows.append(dict(job=j,node=i+1,X_mm=float(d['X'][i,0]),
                    Y_mm=float(d['X'][i,1]),Z_mm=float(d['X'][i,2]),
                    RFx_N=float(R[i,0]),RFy_N=float(R[i,1]),RFz_N=float(R[i,2]),
                    assembled_x_N=float(model[i,0]),assembled_y_N=float(model[i,1]),
                    assembled_z_N=float(model[i,2]),nodal_error_N=norm(model[i]-R[i])))
            print('%-27s corrected=%9.5f%% raw=%9.5f%% mode=%s'%(j,err,wrongerr,one['mode']))
    csv_write(paths[0],rows);csv_write(paths[1],iprows);csv_write(paths[2],noderows)
    byname={r['job']:r for r in rows}
    # The exact B9A10E test condition is F/P17/A100 at NLGEOM YES.
    comparison={'reference_B9A10E_F_ORTHO_NONAFF_error_pct':5.761989488498557,
        'current_error_pct':byname['B9A10F_F_P17_A100']['force_error_global_pct']}
    comparison['absolute_error_difference_pp']=abs(comparison['reference_B9A10E_F_ORTHO_NONAFF_error_pct']-comparison['current_error_pct'])
    comparison['passes_0p15pp_reproducibility_check']=comparison['absolute_error_difference_pp']<0.15
    if not comparison['passes_0p15pp_reproducibility_check']:
        print('WARNING: B9A10F +17/A100 does not reproduce previous B9A10E 5.762% result; investigate before drawing conclusions')
    report={'step':'65AM-B9A10-F','summary':'One-brick C3D8 orthotropic orientation/warping and small-strain tests',
        'cases_total':len(rows),'baseline_reproduction':comparison,
        'cases':{r['job']:{k:v for k,v in r.items() if k!='job'} for r in rows},
        'all_nodes_prescribed':True,
        'interpretation_rules':[
            'A dependence on orientation or non-affinity does not alone identify the force discrepancy cause',
            'Small-strain NLGEOM=NO uses sigma times reference GradN, not finite-deformation Piola',
            'C3D8R hourglass stabilization is not included in reconstructed stress-only nodal force',
            'A single element cannot verify force consistency on the full multi-element mesh',
            'This does not provide an approved physics-informed B5 loss or a converged coupled solver'],
        'B5_B9A7_65V_unchanged':True,'physics_loss_ready':False,
        'live_Broyden_approved':False}
    paths[3].write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5.7))
    for ang in ANGLES:
        subset=[r for r in rows if r['element']=='C3D8' and r['angle_code']==ang and r['mode']=='finite']
        subset.sort(key=lambda r:r['warping_alpha'])
        ax.plot([r['warping_alpha'] for r in subset],[r['force_error_global_pct'] for r in subset],
                marker='o',lw=2,label={'ISO':'Isotropic','Z00':'Ortho 0 deg','P17':'Ortho +17 deg','M17':'Ortho -17 deg'}[ang])
    ax.axhline(.5,color='gray',linestyle='--',label='Diagnostic 0.5% line')
    ax.set_xlabel('Non-affine displacement amplitude (relative to B9A10E)')
    ax.set_ylabel('Assembled vs Abaqus RF error (%)')
    ax.set_title('B9A10F | C3D8 orientation and non-affine amplitude')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(paths[4],dpi=160);plt.close(fig)
    target='B9A10F_F_P17_A100'
    selected=[r for r in noderows if r['job']==target]
    fig,ax=plt.subplots(figsize=(9.5,4.6))
    ax.bar(np.arange(8)+1,[r['nodal_error_N'] for r in selected])
    ax.set_xlabel('Single-element node label')
    ax.set_ylabel('Force error norm (N)')
    ax.set_title('B9A10F | Ortho +17°, nonaffine amplitude 1.0')
    fig.tight_layout();fig.savefig(paths[5],dpi=160);plt.close(fig)
    print('B9A10F REPORTS:',out)
    print('B9A10E SAME CASE REPRODUCED:',comparison['passes_0p15pp_reproducibility_check'])


def selftest():
    X=(SIGNS+1.)*np.array([.6,.5,.62])[None,:]/2.
    U=X@np.array([[.004,.002,.001],[-.001,-.002,.001],[.0005,-.0008,.003]]).T
    th=np.deg2rad(17);Q=np.array([[math.cos(th),math.sin(th),0],[-math.sin(th),math.cos(th),0],[0,0,1.]])
    sg=np.array([[15.,1.,0],[1.,8.,0],[0,0,6.]])
    sl=Q@sg@Q.T
    for t in ('F','R'):
        PTS=GAUSS if t=='F' else np.zeros((1,3))
        dat=dict(X=X,U=U,RF=np.zeros((8,3)),IP_COORD=np.array([shape(p)@(X+U) for p in PTS]),
            S=np.tile([sl[0,0],sl[1,1],sl[2,2],sl[0,1],sl[0,2],sl[1,2]],(len(PTS),1)),
            Q=np.tile(Q,(len(PTS),1,1)),Q_available=np.ones(len(PTS),dtype=np.uint8),
            connectivity=np.arange(1,9))
        for nonlinear in (True,False):
            good,diag,_=calculate(dat,t,nonlinear,True)
            bad,_,_=calculate(dat,t,nonlinear,False)
            if norm(good.sum(axis=0))>1.e-9:raise RuntimeError('Net force is not zero')
            if norm(good-bad)<1.e-6:raise RuntimeError('Incorrect stress rotation not detected')
            if nonlinear:
                F=np.eye(3)+U.T@natural_grad(np.zeros(3))@np.linalg.inv(X.T@natural_grad(np.zeros(3)))
                if np.linalg.det(F)<=0:raise RuntimeError('Bad F')
    # Local to global identity independent of FE assembly
    assert norm(Q.T@sl@Q-sg)<1.e-10
    print('B9A10F MAC SELFTEST PASS | C3D8/C3D8R, Piola, small strain, local orientation')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='.')
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.selftest:selftest()
    else:run(Path(args.project).expanduser().resolve(),args.check_only)
