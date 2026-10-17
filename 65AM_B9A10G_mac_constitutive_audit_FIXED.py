#!/usr/bin/env python3
"""B9A10-G Mac: offline Abaqus orthotropic strain / B-bar diagnostic.

Uses verified B9A10F stress/U/RF exports and B9A10G E-or-LE ODB export.
For NLGEOM=NO, E (total small strain) is used; other jobs use LE.
Does not train, modify, or couple any neural-operator model.

Commands:
 python3 65AM_B9A10G_mac_constitutive_audit.py --selftest
 python3 65AM_B9A10G_mac_constitutive_audit.py --project . --check-only
 python3 65AM_B9A10G_mac_constitutive_audit.py --project .
"""
import argparse, csv, hashlib, json, itertools, math
from pathlib import Path
import numpy as np

ANGS=('Z00','P17','M17','ISO')
AMPS=('A00','A10','A50','A100','LIN')
CASES=[('F',a,b) for a in ANGS for b in AMPS]+[('R','P17','A100')]
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
G=1/np.sqrt(3)
GAUSS=np.array(list(itertools.product((-G,G),repeat=3)),float)

def name(c):return 'B9A10F_%s_%s_%s'%c

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def shape(q):return np.prod(1+SIGNS*np.asarray(q)[None,:],axis=1)/8

def grad_nat(q):
    a,b,c=q;t=SIGNS
    return np.column_stack((t[:,0]*(1+b*t[:,1])*(1+c*t[:,2]),
                            t[:,1]*(1+a*t[:,0])*(1+c*t[:,2]),
                            t[:,2]*(1+a*t[:,0])*(1+b*t[:,1])))/8

def tensor(x):
    a=np.asarray(x,float).reshape(6)
    return np.array([[a[0],a[3],a[4]],[a[3],a[1],a[5]],[a[4],a[5],a[2]]])

def eng(e):
    return np.array([e[0,0],e[1,1],e[2,2],2*e[0,1],2*e[0,2],2*e[1,2]])

def stiffness(material):
    if material=='ISO':
        E,nu=20000.,.25
        lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
        C=np.zeros((6,6));C[:3,:3]=lam
        for i in range(3):C[i,i]+=2*mu
        C[3,3]=C[4,4]=C[5,5]=mu
    else:
        E1,E2,E3=46500.,12500.,12000.
        nu12,nu13,nu23=.28,.28,.40
        S=np.zeros((6,6))
        S[0,0],S[1,1],S[2,2]=1/E1,1/E2,1/E3
        S[0,1]=S[1,0]=-nu12/E1
        S[0,2]=S[2,0]=-nu13/E1
        S[1,2]=S[2,1]=-nu23/E2
        S[3,3],S[4,4],S[5,5]=1/4750.,1/4500.,1/3500.
        C=np.linalg.inv(S)
    if np.min(np.linalg.eigvalsh(C))<=0:raise RuntimeError('Elastic matrix not positive definite')
    return C

def nominal_q(angle):
    if angle=='ISO':return np.eye(3)
    rad=math.radians({'Z00':0.,'P17':17.,'M17':-17.}[angle]);c,s=np.cos(rad),np.sin(rad)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def load_case(z,le,c):
    j=name(c);nip=1 if c[0]=='R' else 8
    specs={'X':(8,3),'U':(8,3),'RF':(8,3),'S':(nip,6),
           'Q':(nip,3,3),'Q_available':(nip,), 'IP_COORD':(nip,3),'connectivity':(8,)}
    d={}
    for field,dim in specs.items():
        key=j+'_'+field
        if key not in z:raise RuntimeError('Missing B9A10F '+key)
        d[field]=np.asarray(z[key])
        if d[field].shape!=dim or not np.isfinite(d[field]).all():raise RuntimeError('Invalid '+key)
    for k,s in {'LE':(nip,6),'LE_Q':(nip,3,3),'LE_Q_available':(nip,)}.items():
        key=j+'_'+k
        if key not in le:raise RuntimeError('Missing B9A10G '+key)
        d[k]=np.asarray(le[key])
        if d[k].shape!=s or not np.isfinite(d[k]).all():raise RuntimeError('Invalid '+key)
    k=j+'_STRAIN_KIND'
    if k not in le:raise RuntimeError('Missing strain-field metadata for '+j+'; use FIXED Windows exporter')
    d['strain_kind']=str(np.asarray(le[k]).item())
    expected='E' if c[2]=='LIN' else 'LE'
    if d['strain_kind']!=expected:
        raise RuntimeError('Wrong Abaqus strain output for %s: expected %s, got %s'%(j,expected,d['strain_kind']))
    if not np.array_equal(d['connectivity'],np.arange(1,9)):raise RuntimeError('Unexpected node order '+j)
    if c[1]!='ISO' and not np.all(d['Q_available']):raise RuntimeError('Missing S local coordinates '+j)
    if c[1]!='ISO' and not np.all(d['LE_Q_available']):
        raise RuntimeError('Missing local frame for Abaqus LE in '+j+'; do not assume global strain')
    for k in ('Q','LE_Q'):
        for v in d[k]:
            if abs(np.linalg.det(v)-1)>2e-3 or np.max(abs(v@v.T-np.eye(3)))>2e-3:
                raise RuntimeError('Bad frame '+j+' '+k)
    return d

def q_match(d,c):
    pts=np.array([[0.,0.,0.]]) if c[0]=='R' else GAUSS
    current=d['X']+d['U']
    frames=[]
    for x in (d['X'],current):
        positions=np.array([shape(q)@x for q in pts])
        dist=np.linalg.norm(positions[:,None,:]-d['IP_COORD'][None,:,:],axis=2)
        index=np.argmin(dist,axis=1)
        if len(set(index.tolist()))==len(index):
            frames.append((float(np.max(dist[np.arange(len(index)),index])),index))
    if not frames:raise RuntimeError('Integration points not one-to-one: '+name(c))
    mis,ix=min(frames,key=lambda x:x[0])
    if mis>2e-5:raise RuntimeError('Bad IP coordinate matching %s %.7g'%(name(c),mis))
    return pts,ix,mis

def pct(delta,ref):
    return float(100*np.linalg.norm(delta)/max(float(np.linalg.norm(ref)),1e-12))

def analyze_case(d,c):
    pts,ix,mis=q_match(d,c)
    C=stiffness('ISO' if c[1]=='ISO' else 'ORTHO')
    Q0=nominal_q(c[1]); U=d['U'];X=d['X'];RF=d['RF']
    f_std=np.zeros((8,3));f_bbar=np.zeros((8,3));grad=[];weights=[];sigma=[]; eps=[]
    iprecords=[]; ipangle=[]
    for q_id,pt in enumerate(pts):
        ip=int(ix[q_id]); nat=grad_nat(pt);jac=X.T@nat; v=float(np.linalg.det(jac))
        if v<=0:raise RuntimeError('Nonpositive detJ0')
        g=nat@np.linalg.inv(jac); F=np.eye(3)+U.T@g
        epsglob=(U.T@g+g.T@U)/2
        if c[2]=='LIN':
            strain=epsglob
        else:
            strain=.5*(F.T@F-np.eye(3)) # Green strain: diagnostic ONLY for NLGEOM=YES
        Qs=d['Q'][ip];Ql=d['LE_Q'][ip]
        sig_local=tensor(d['S'][ip]);sig_global=Qs.T@sig_local@Qs
        volume=v*(8 if c[0]=='R' else 1)
        P=sig_global if c[2]=='LIN' else np.linalg.det(F)*sig_global@np.linalg.inv(F).T
        f_std+=volume*g@P.T
        grad.append(g);weights.append(volume);sigma.append(sig_global);eps.append(strain)
        LE=d['LE'][ip]
        # Abaqus E (linear) or LE (nonlinear) has six components; test shear conventions.
        model_eng=eng(Ql@strain@Ql.T)
        model_tensor=model_eng.copy();model_tensor[3:]*=.5
        # Hookean stress checks use Abaqus E (linear) or LE (nonlinear) field.
        stress_from_LE_eng=C@LE
        LE_twice=LE.copy();LE_twice[3:]*=2
        stress_from_LE_twice=C@LE_twice
        stress_from_U_nominal=C@eng(Q0@strain@Q0.T)
        iprecords.append({'ip':q_id+1,'odb_ip':ip+1,'detJ0':v,'strain_field':d['strain_kind'],'strain_norm':np.linalg.norm(LE),
            'LE_vs_gradU_engineering_pct':pct(model_eng-LE,LE),
            'LE_vs_gradU_tensor_pct':pct(model_tensor-LE,LE),
            'S_local_vs_C_LE_engineering_pct':pct(stress_from_LE_eng-d['S'][ip],d['S'][ip]),
            'S_local_vs_C_LE_twice_shear_pct':pct(stress_from_LE_twice-d['S'][ip],d['S'][ip]),
            'S_local_vs_C_gradU_nominal_pct':pct(stress_from_U_nominal-d['S'][ip],d['S'][ip]),
            'S11_abaqus':d['S'][ip,0], 'S11_from_LE':stress_from_LE_eng[0],
            'S12_abaqus':d['S'][ip,3], 'S12_from_LE':stress_from_LE_eng[3],
            'Q_stress_LE_difference_norm':np.linalg.norm(Qs-Ql),
            'Q_stress_nominal_difference_norm':np.linalg.norm(Qs-Q0)})
    # B-bar is a small-strain, first-order fully integrated element hypothesis ONLY.
    if c[0]=='F' and c[2]=='LIN':
        weight=np.asarray(weights);g=np.asarray(grad);s=np.asarray(sigma)
        gavg=np.einsum('q,qai->ai',weight,g)/weight.sum()
        for qi in range(len(pts)):
            pressure=float(np.trace(s[qi]))/3
            f_bbar+=weight[qi]*pressure*(gavg-g[qi])
        f_bbar+=f_std
    else:f_bbar[:]=np.nan
    def aggr(k):
        return float(np.sqrt(np.mean(np.square([r[k] for r in iprecords]))))
    return dict(RF_pct_standard=pct(f_std-RF,RF),
                RF_pct_Bbar=None if not np.isfinite(f_bbar).all() else pct(f_bbar-RF,RF),
                RF_norm_N=float(np.linalg.norm(RF)),
                LE_gradU_eng_RMS_pct=aggr('LE_vs_gradU_engineering_pct'),
                LE_gradU_tensor_RMS_pct=aggr('LE_vs_gradU_tensor_pct'),
                S_from_Abaqus_LE_RMS_pct=aggr('S_local_vs_C_LE_engineering_pct'),
                S_from_Abaqus_LE_double_shear_RMS_pct=aggr('S_local_vs_C_LE_twice_shear_pct'),
                S_from_gradU_nominal_RMS_pct=aggr('S_local_vs_C_gradU_nominal_pct'),
                max_Q_S_LE_difference=float(max(r['Q_stress_LE_difference_norm'] for r in iprecords)),
                max_Q_S_nominal_difference=float(max(r['Q_stress_nominal_difference_norm'] for r in iprecords)),
                max_IP_geometry_gap_mm=mis),iprecords,f_std,f_bbar

def inputs(root):
    a=root/'B9A10F_exports'/'B9A10F_patch_fields.npz'
    am=root/'B9A10F_exports'/'B9A10F_windows_summary.json'
    b=root/'B9A10G_exports'/'B9A10G_LE_fields.npz'
    bm=root/'B9A10G_exports'/'B9A10G_windows_summary.json'
    for p in (a,am,b,bm):
        if not p.is_file():raise FileNotFoundError('Missing expected input: '+str(p))
    m=json.loads(am.read_text());n=json.loads(bm.read_text())
    if m['exports_sha256']!=sha(a) or n['LE_sha256']!=sha(b):raise RuntimeError('Export SHA256 mismatch')
    if n.get('source_B9A10F_npz_sha256')!=m['exports_sha256']:
        raise RuntimeError('B9A10F/B9A10G based on different originals')
    if m.get('case_count')!=21 or n.get('case_count')!=21:
        raise RuntimeError('Wrong number of cases')
    return a,b

def write_csv(path,records):
    if not records:raise RuntimeError('Empty report')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def run(root,check_only=False):
    fa,gb=inputs(root);rows=[];ips=[];nodes=[]
    with np.load(fa,allow_pickle=False) as z, np.load(gb,allow_pickle=False) as extra:
        for c in CASES:
            d=load_case(z,extra,c)
            m,ip,ff,fb=analyze_case(d,c)
            print('%-25s FE=%8.5f%% Bbar=%s LE_U=%8.4f%% Hooke_LE=%8.4f%%'%(
                name(c),m['RF_pct_standard'], 'n/a' if m['RF_pct_Bbar'] is None else ('%8.5f%%'%m['RF_pct_Bbar']),
                m['LE_gradU_eng_RMS_pct'],m['S_from_Abaqus_LE_RMS_pct']))
            rows.append(dict(job=name(c),angle=c[1],deformation=c[2],strain_field=d['strain_kind'],element='C3D8R' if c[0]=='R' else 'C3D8',
                             **m))
            for k in ip:ips.append(dict(job=name(c),**k))
            if c[2]=='LIN':
                for i in range(8):
                    nodes.append(dict(job=name(c),node=i+1,RF_x=d['RF'][i,0],RF_y=d['RF'][i,1],RF_z=d['RF'][i,2],
                                      ordinary_error_N=np.linalg.norm(ff[i]-d['RF'][i]),
                                      Bbar_error_N=np.linalg.norm(fb[i]-d['RF'][i])))
    print('B9A10G INPUT VALIDATION PASS | 21 cases')
    if check_only:return
    out=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10G_constitutive_audit'
    if not out.is_dir():raise RuntimeError('Create result folder manually in Finder: '+str(out))
    names=('B9A10G_cases.csv','B9A10G_integration_points.csv','B9A10G_nodes.csv',
           'B9A10G_summary.json','B9A10G_force_comparison.png','B9A10G_strain_stress.png')
    if any((out/n).exists() for n in names):
        raise RuntimeError('Existing B9A10G outputs protected. Do not overwrite; move only B9A10G files in Finder.')
    write_csv(out/names[0],rows);write_csv(out/names[1],ips);write_csv(out/names[2],nodes)
    lin={x['angle']:x for x in rows if x['deformation']=='LIN'}
    checks={'B9A10F_zero_angle_linear_original_error_7p4477pp':abs(lin['Z00']['RF_pct_standard']-7.447716022439524)<.15,
            'B9A10F_plus17_linear_original_error_5p6183pp':abs(lin['P17']['RF_pct_standard']-5.618317834237389)<.15,
            'B9A10F_iso_linear_original_error_0pct':lin['ISO']['RF_pct_standard']<.01}
    report={'step':'65AM-B9A10-G','hypothesis':'Abaqus C3D8 modified mean-dilatation (B-bar) operation',
      'saved_cases':21,'strain_fields':{'NLGEOM=NO':'E','NLGEOM=YES':'LE'},'linear_comparisons':lin,'baseline_checks':checks,
      'all_baselines_reproduced':all(checks.values()),
      'no_training_no_new_Abaqus_jobs':True,'physics_loss_ready':False,
      'limits':['Bbar correction tested only for C3D8 NLGEOM=NO, not for nonlinear C3D8R model',
                'Abaqus E for linear and LE for nonlinear are output fields, not direct access to element internal force algorithm',
                'Successful Bbar comparison is evidence, not a proof of exact Abaqus implementation',
                'No model or 65V solver changed']}
    (out/names[3]).write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['ISO','Z00','P17','M17'];names_map={'ISO':'Isotropic','Z00':'Ortho 0°','P17':'Ortho +17°','M17':'Ortho −17°'}
    xx=np.arange(4);fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(xx-.18,[lin[a]['RF_pct_standard'] for a in order],.36,label='Ordinary integration')
    ax.bar(xx+.18,[lin[a]['RF_pct_Bbar'] for a in order],.36,label='B-bar diagnostic')
    ax.axhline(.5,ls='--',color='gray',label='0.5% diagnostic')
    ax.set_xticks(xx,[names_map[a] for a in order]);ax.set_ylabel('Reaction-force discrepancy (%)')
    ax.set_title('B9A10G | C3D8 small-strain force reconstruction');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/names[4],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));wid=.23
    for offset,k,label in ((-wid,'LE_gradU_eng_RMS_pct','Abaqus strain vs displacement gradient'),
                           (0,'S_from_Abaqus_LE_RMS_pct','Stress vs Hooke(Abaqus strain)'),
                           (wid,'S_from_gradU_nominal_RMS_pct','Stress vs Hooke(GradU)')):
        ax.bar(xx+offset,[lin[a][k] for a in order],wid,label=label)
    ax.set_xticks(xx,[names_map[a] for a in order]);ax.set_ylabel('Integration-point RMS relative error (%)')
    ax.set_title('B9A10G | strain and constitutive consistency');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/names[5],dpi=160);plt.close(fig)
    print('B9A10G OUTPUTS:',out)
    print('BASELINE REPRODUCTION:', 'PASS' if all(checks.values()) else 'WARNING — investigate before interpretation')

def selftest():
    C=stiffness('ORTHO');assert np.min(np.linalg.eigvalsh(C))>0
    assert np.max(abs(nominal_q('P17')@nominal_q('P17').T-np.eye(3)))<1e-12
    X=(SIGNS+1)*np.array([.6,.5,.62])/2
    U=X@np.diag([.006,-.003,.002])
    pt=GAUSS; vols=[];fs=[];grad=[]
    sigma=np.diag([1.5,2,3])
    for q in pt:
        n=grad_nat(q);A=X.T@n;v=np.linalg.det(A);g=n@np.linalg.inv(A)
        vols.append(v);grad.append(g);fs.append(v*g@sigma.T)
    avg=sum(v*g for v,g in zip(vols,grad))/sum(vols)
    std=sum(fs);mod=std+sum(v*(np.trace(sigma)/3)*(avg-g) for v,g in zip(vols,grad))
    assert np.linalg.norm(std-mod)<1e-10
    assert np.linalg.norm(std.sum(axis=0))<1e-10
    strain=np.array([[.01,.002,.003],[.002,.02,.001],[.003,.001,-.01]])
    q=nominal_q('P17');s=C@eng(q@strain@q.T)
    assert np.linalg.norm(s)>0
    # A purely nonaffine gradient has nontrivial differences among Gauss points.
    assert len(CASES)==21
    print('B9A10G SELFTEST PASS | elasticity, frames, mean-grad, constant-stress B-bar identity, 21 cases')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.');ap.add_argument('--selftest',action='store_true');ap.add_argument('--check-only',action='store_true');arg=ap.parse_args()
    if arg.selftest:selftest()
    else:run(Path(arg.project).expanduser().resolve(),arg.check_only)
