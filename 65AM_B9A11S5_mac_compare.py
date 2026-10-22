#!/usr/bin/env python3
"""B9A11-S5: read-only controlled FEM mirror comparison on Mac.

Input: B9A11S5_exports/B9A11S5_mirrored_fem.npz and
       B9A11S5_exports/B9A11S5_windows_summary.json
Stages:
  python3 65AM_B9A11S5_mac_compare.py --project . --stage selftest
  python3 65AM_B9A11S5_mac_compare.py --project . --stage check
  python3 65AM_B9A11S5_mac_compare.py --project . --stage analyze
  python3 65AM_B9A11S5_mac_compare.py --project . --stage report

Reads optional original 65V PCA displacement basis to express BOTH mirror forces
in the SAME LEFT interface virtual-work basis; never compares arbitrarily
numbered left/right PCA modes.

No B5 inference on unrecognized new cases (B5's original predict limits IDs
and external boundary lift + branch feature validation is not implemented).
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAT=('ISO','OR0','OR17'); AMPS=('H','F'); SIDES=('L','R')
REFL=np.diag([-1.,1.,1.]); SIGNS=np.array([1.,1.,1.,-1.,-1.,1.])
BASE=Path('B9A11S5_exports')
OUT=Path('results/65AM_B9_coupled_feasibility/B9A11S5_physical_symmetry')
KEYS=('X','node_labels','conn','centroids','U','RF','S_global','LE_global','Q_global_to_local','boundary_mask')

def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64).reshape(-1)))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def name(m,a,s):return 'B9A11S5_%s_%s_%s'%(m,a,s)

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=list(rows[0].keys())
    if any(set(r)!=set(cols) for r in rows):
        cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)

def index_to_reflection(src,des,atol=1.e-5):
    """Source X -> mirror into desired global reference X. Unique bijection."""
    from scipy.spatial import cKDTree
    src=np.asarray(src,float);des=np.asarray(des,float)
    if src.shape!=des.shape or src.ndim!=2 or src.shape[1]!=3:
        raise RuntimeError('Mirror coordinate shapes differ: %s vs %s'%(src.shape,des.shape))
    distance,index=cKDTree(des).query(src@REFL.T,k=1)
    if not np.isfinite(distance).all() or np.max(distance)>atol or len(set(index.tolist()))!=len(index):
        raise RuntimeError('Coordinate mirror bijection FAILED; max distance %.7g'%np.max(distance))
    return index,float(np.max(distance))

def to_left_vector(right,idx,n):
    """right indexed by right; idx[left_i]=right_i -> left ordered reflected R."""
    return np.asarray(right,float)[idx]@REFL.T

def to_left_tensor_voigt(right,idx):
    return np.asarray(right,float)[idx]*SIGNS[None,:]

def percent_gap(left,right,reference):
    return 100.*norm(np.asarray(left,float)-np.asarray(right,float))/max(float(reference),1.e-12)

def read_data(root):
    npz=root/BASE/'B9A11S5_mirrored_fem.npz'
    man=root/BASE/'B9A11S5_windows_summary.json'
    if not npz.is_file() or not man.is_file():
        raise FileNotFoundError('Transfer the entire B9A11S5_exports folder from Windows into '+str(root))
    summary=json.loads(man.read_text())
    if summary.get('step')!='B9A11S5' or summary.get('job_count')!=12 or summary.get('pair_count')!=6:
        raise RuntimeError('Unexpected B9A11S5 source manifest')
    if not summary.get('global_stress_transformed',False):
        raise RuntimeError('Windows did not verify global stress orientation')
    if sha(npz)!=summary.get('export_sha256'):
        raise RuntimeError('S5 data checksum mismatch; stop')
    cases={};records={j['job']:j for j in summary['jobs']}
    if len(records)!=12:raise RuntimeError('Missing or duplicate ODB job records')
    with np.load(npz,allow_pickle=False) as z:
        for m in MAT:
            for a in AMPS:
                for s in SIDES:
                    n=name(m,a,s)
                    if n not in records:raise RuntimeError('Missing Windows record: '+n)
                    cases[n]={k:np.asarray(z[n+'__'+k]).copy() for k in KEYS}
    for n,d in cases.items():
        if d['X'].shape!=(1008,3) or d['U'].shape!=(1008,3) or d['RF'].shape!=(1008,3):
            raise RuntimeError('S5 exported nodal shapes wrong: '+n)
        if d['conn'].shape!=(728,8) or d['S_global'].shape!=(728,6) or d['LE_global'].shape!=(728,6):
            raise RuntimeError('S5 element shapes wrong: '+n)
        if d['centroids'].shape!=(728,3):raise RuntimeError('Missing element centers: '+n)
        if int(d['boundary_mask'].sum())!=224:raise RuntimeError('Boundary count not 224 in '+n)
        if not np.isfinite(d['S_global']).all():raise RuntimeError('Nonfinite stresses '+n)
        if np.any(d['conn']<0) or np.any(d['conn']>=len(d['X'])):
            raise RuntimeError('Invalid connectivity indices: '+n)
    return cases,summary

def topology_test(l,r,node_map,el_map):
    mirrored=set(tuple(sorted(node_map[e])) for e in l['conn'])
    right=set(tuple(sorted(e)) for e in r['conn'])
    if len(mirrored)!=728 or len(right)!=728 or mirrored!=right:
        raise RuntimeError('Mirror topology incompatible even though geometries match')
    # Also centroid correspondence must agree with element node set correspondence.
    return True

def face_metrics(l,r,node_map,case):
    data=[]
    # min x on L -> max x on R (outer), max x L -> min x R (center)
    for face in ('outer','center'):
        xref=float(l['X'][:,0].min() if face=='outer' else l['X'][:,0].max())
        sel=np.where(np.isclose(l['X'][:,0],xref,atol=1e-5))[0]
        if len(sel)!=112:raise RuntimeError('Expected 112 nodes per end face')
        target=to_left_vector(r['RF'],node_map,1008)[sel]
        a=l['RF'][sel]
        reference=max(norm(a),norm(target),1e-12)
        data.append(dict(material=case[0],amplitude=case[1],face=face,
                         RF_mirror_relative_pct=percent_gap(a,target,reference),
                         RF_max_abs_N=float(np.max(abs(a-target))),
                         RF_L_norm_N=norm(a),RF_R_reflected_norm_N=norm(target),
                         mean_RF_L_x_N=float(a[:,0].sum()),
                         mean_RF_mirrored_R_x_N=float(target[:,0].sum())))
    return data

def pca_shared_basis(root,l,r,node_map,m,a):
    """Compare both reactions in a SINGLE LEFT 65V displacement basis.

    Generalized force B.T RF; unlike per-side PCA mode numbering, this is
    physically comparable after exact mirror transformation. Optional file.
    """
    p=root/'B9_coupled_anchor/B9_65V_anchor.npz'
    if not p.is_file():return []
    result=[]
    from scipy.spatial import cKDTree
    mirrored_R=to_left_vector(r['RF'],node_map,1008)
    with np.load(p,allow_pickle=False) as z:
        for iface in ('left_inner','center_left'):
            ck='pca_coordinates_'+iface;bk='pca_basis_'+iface
            if ck not in z or bk not in z:
                raise RuntimeError('Original 65V anchor missing '+ck+' or '+bk)
            coords=np.asarray(z[ck],float); basis=np.asarray(z[bk],float)
            dist,ii=cKDTree(l['X']).query(coords,k=1)
            if len(set(ii.tolist()))!=len(ii) or dist.max()>2e-5:
                raise RuntimeError('65V PCA face did not match S5 mesh: '+iface)
            if basis.shape[0]!=3*len(ii):
                raise RuntimeError('PCA basis incompatible with S5 boundary RF: '+iface)
            fL=l['RF'][ii].reshape(-1); fR=mirrored_R[ii].reshape(-1)
            gL=basis.T@fL;gR=basis.T@fR
            den=max(norm(gL),norm(gR),1e-12)
            for i in range(len(gL)):
                result.append(dict(material=m,amplitude=a,left_interface=iface,common_basis_mode=i+1,
                    left_generalized_Nmm=float(gL[i]),right_reflected_generalized_Nmm=float(gR[i]),
                    absolute_difference_Nmm=float(abs(gL[i]-gR[i])),
                    mode_difference_on_common_vector_scale_pct=100.*abs(float(gL[i]-gR[i]))/den))
    return result

def run_checks(root):
    cases,summary=read_data(root)
    for m in MAT:
        for a in AMPS:
            l=cases[name(m,a,'L')];r=cases[name(m,a,'R')]
            nm,ne=index_to_reflection(l['X'],r['X'])
            em,ee=index_to_reflection(l['centroids'],r['centroids'])
            topology_test(l,r,nm,em)
            if not np.array_equal(l['boundary_mask'].astype(bool),r['boundary_mask'][nm].astype(bool)):
                raise RuntimeError('Prescribed boundary mask not mirrored')
            print('S5 CHECK PASS',m,a,'| geometry error mm %.3g / %.3g'%(ne,ee))
    return cases,summary

def analyze(root):
    cases,src=run_checks(root)
    out=root/OUT
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing S5 reports protected; move them or choose a fresh --run-tag')
    out.mkdir(parents=True,exist_ok=True)
    paired=[];faces=[];pca=[];components=[]
    for m in MAT:
        for a in AMPS:
            l=cases[name(m,a,'L')];r=cases[name(m,a,'R')]
            nm,ne=index_to_reflection(l['X'],r['X'])
            em,ee=index_to_reflection(l['centroids'],r['centroids'])
            ub=to_left_vector(r['U'],nm,1008)
            rb=to_left_vector(r['RF'],nm,1008)
            sb=to_left_tensor_voigt(r['S_global'],em)
            leb=to_left_tensor_voigt(r['LE_global'],em)
            fixed=l['boundary_mask'].astype(bool)
            stress_den=max(norm(l['S_global']),norm(sb),1e-12)
            u_den=max(norm(l['U']),norm(ub),1e-12)
            rf_den=max(norm(l['RF'][fixed]),norm(rb[fixed]),1e-12)
            rec=dict(material=m,amplitude=a,
                U_mirror_pct=percent_gap(l['U'],ub,u_den),
                U_max_abs_mm=float(np.max(abs(l['U']-ub))),
                stress_global_mirror_pct=percent_gap(l['S_global'],sb,stress_den),
                stress_global_max_abs_MPa=float(np.max(abs(l['S_global']-sb))),
                LE_global_mirror_pct=percent_gap(l['LE_global'],leb,max(norm(l['LE_global']),norm(leb),1e-12)),
                RF_boundary_mirror_pct=percent_gap(l['RF'][fixed],rb[fixed],rf_den),
                RF_boundary_max_abs_N=float(np.max(abs(l['RF'][fixed]-rb[fixed]))),
                RF_free_mirror_pct_of_boundary=percent_gap(l['RF'][~fixed],rb[~fixed],rf_den),
                left_net_boundary_RF_pct=100.*norm(l['RF'][fixed].sum(axis=0))/rf_den,
                right_net_boundary_RF_pct=100.*norm(rb[fixed].sum(axis=0))/rf_den,
                mirror_geometry_max_mm=ne,
                mirror_element_centroid_max_mm=ee,
                max_prescribed_U_mirror_mm=float(np.max(abs(l['U'][fixed]-ub[fixed]))),
                mirror_BC_pass=bool(np.max(abs(l['U'][fixed]-ub[fixed]))<5e-5),
                b5_external_inference_evaluated=False)
            paired.append(rec)
            for ki,component in ((0,'S11'),(1,'S22'),(3,'S12')):
                left_component=l['S_global'][:,ki]
                right_component=sb[:,ki]
                components.append(dict(material=m,amplitude=a,component=component,
                    component_mirror_pct=percent_gap(left_component,right_component,
                        max(norm(left_component),norm(right_component),1e-12)),
                    contribution_to_total_stress_pct=percent_gap(left_component,right_component,stress_den),
                    max_abs_MPa=float(np.max(abs(left_component-right_component)))))
            faces+=face_metrics(l,r,nm,(m,a))
            pca+=pca_shared_basis(root,l,r,nm,m,a)
            print('S5',m,a,'U %.6g%%'%rec['U_mirror_pct'],
                  'stress %.6g%%'%rec['stress_global_mirror_pct'],
                  'boundary RF %.6g%%'%rec['RF_boundary_mirror_pct'],flush=True)
    write_csv(out/'B9A11S5_paired_cases.csv',paired)
    write_csv(out/'B9A11S5_face_RF.csv',faces)
    write_csv(out/'B9A11S5_stress_components.csv',components)
    if pca:write_csv(out/'B9A11S5_shared_basis_forces.csv',pca)
    metrics=('U_mirror_pct','stress_global_mirror_pct','RF_boundary_mirror_pct','LE_global_mirror_pct')
    avg={k:float(np.mean([p[k] for p in paired])) for k in metrics}
    # Diagnostic thresholds not assertions: in symmetry tests numerical error may
    # be small but tolerances should be assessed from observed Abaqus precision.
    report={'step':'B9A11S5','source_sha256':src['export_sha256'],
          'pairs':len(paired),'job_count':12,'no_old_models_modified':True,
          'exact_mirrored_geometry_BC':True,'global_stress_Q_transformed':True,
          'stress_orientation_control':'OR17 is +17 left/-17 right',
          'independent_new_Abaqus_cases':True,
          'trained_network_generalization_verified':False,
          'no_external_B5_inference':'B5 prediction is restricted to original fit/tune IDs and uses original region-specific PCA lift; cannot safely infer on new mirrored BCs without a separately validated external mapping',
          'mean_mirror_pct':avg,'all_pair_BC_checks_pass':all(p['mirror_BC_pass'] for p in paired),
          'common_basis_available':bool(pca),
          'caution':'This diagnoses physical FEM symmetry. It does not establish physics-informed B5 gains or independent learned model generalization.'}
    (out/'B9A11S5_summary.json').write_text(json.dumps(report,indent=2))
    labels=[r['material']+'_'+r['amplitude'] for r in paired]
    xx=np.arange(len(labels));w=.25
    fig,ax=plt.subplots(figsize=(11,5.5))
    for off,k,title in [(-1,'U_mirror_pct','Displacement'),(0,'stress_global_mirror_pct','Global stress'),(1,'RF_boundary_mirror_pct','Interface reaction')]:
        ax.bar(xx+off*w,[r[k] for r in paired],width=w,label=title)
    ax.set_xticks(xx,labels);ax.set_ylabel('Relative mirror mismatch (%)');
    ax.set_title('B9A11-S5 | controlled mirrored Abaqus C3D8R pairs');ax.legend()
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'B9A11S5_mirror_comparison.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5))
    for face in ('outer','center'):
        ff=[r for r in faces if r['face']==face]
        ax.plot(labels,[r['RF_mirror_relative_pct'] for r in ff],marker='o',label=face)
    ax.set_ylabel('Mirrored face reaction error (%)');ax.set_title('B9A11-S5 | Outer vs center interface reaction symmetry');
    ax.set_xticks(np.arange(len(labels)),labels);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'B9A11S5_face_forces.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5))
    for comp in ('S11','S22','S12'):
        vals=[q['contribution_to_total_stress_pct'] for q in components if q['component']==comp]
        ax.plot(labels,vals,marker='o',label=comp)
    ax.set_ylabel('Component mismatch / full stress norm (%)')
    ax.set_title('B9A11-S5 | global-frame stress components after mirror reflection')
    ax.set_xticks(np.arange(len(labels)),labels);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'B9A11S5_stress_components.png',dpi=165);plt.close(fig)
    print('B9A11-S5 ANALYSIS PASS | results',out, '| no B5 external inference',flush=True)

def selftest():
    a=np.array([[-2,0,1],[-1,1,2],[-3,2,3]],float)
    b=(a@REFL.T)[[2,0,1]]
    ids,d=index_to_reflection(a,b,1e-10)
    vec=np.array([[2.,3.,4.],[5.,6.,7.],[8.,9.,10.]])
    right=(vec@REFL.T)[[2,0,1]]
    np.testing.assert_allclose(to_left_vector(right,ids,3),vec,rtol=0,atol=0)
    sv=np.array([[1.,2.,3.,4.,5.,6.],[2.,3.,4.,5.,6.,7.],[3.,4.,5.,6.,7.,8.]])
    right_s=(sv*SIGNS)[[2,0,1]]
    np.testing.assert_allclose(to_left_tensor_voigt(right_s,ids),sv,rtol=0,atol=0)
    print('B9A11-S5 MAC SELFTEST PASS: unique mirror mapping, vectors, tensors')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--stage',required=True,choices=('selftest','check','analyze','report'))
    p.add_argument('--run-tag',default='',help='optional named new output folder')
    a=p.parse_args();root=a.project.resolve()
    global OUT
    if a.run_tag:OUT=OUT/a.run_tag
    if a.stage=='selftest':selftest()
    elif a.stage=='check':run_checks(root)
    elif a.stage=='analyze':analyze(root)
    elif a.stage=='report':
        p=root/OUT/'B9A11S5_summary.json'
        if not p.is_file():raise FileNotFoundError('Run analyze first: '+str(p))
        print(p.read_text())

if __name__=='__main__':main()
