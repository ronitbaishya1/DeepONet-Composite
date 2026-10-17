#!/usr/bin/env python3
"""B9A10-L Mac: full U->S->force screening, no retraining or production change.

Inputs: existing H/J/K exports, original B9A9 12-feature plan, NEW read-only
LE export. Tests two pre-specified finite-strain constitutive approximations;
Abaqus S, LE, Q and RF are used ONLY to diagnose, never to construct the
candidate's predicted stress or fit any parameter.
"""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Batched 3x3 algebra is often faster with fewer CPU threads on an M1 Max.
torch.set_num_threads(2)

REG=('L','R'); OUT=Path('results/65AM_B9_coupled_feasibility/B9A10L_constitutive_validation')
MODELS=('cauchy','kirchhoff')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def import_module(p,key):
    if not p.is_file():raise FileNotFoundError(str(p))
    spec=importlib.util.spec_from_file_location(key,str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def paths(root):
    d={
      'h':root/'B9A10H_exports/B9A10H_independent_fields.npz',
      'hm':root/'B9A10H_exports/B9A10H_windows_summary.json',
      'l':root/'B9A10L_exports/B9A10L_LE_fields.npz',
      'lm':root/'B9A10L_exports/B9A10L_windows_summary.json',
      'k':root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_summary.json',
      'kc':root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_calibrated_coefficient.json',
      'km':root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_independent_cases.csv',
      'plan':root/'B9A9_transfer_to_windows/B9A9_case_plan.npz',
      'anchor':root/'B9_coupled_anchor/B9_65V_anchor.npz',
      'hcode':root/'65AM_B9A10H_mac_weakform_validation.py',
      'kcode':root/'65AM_B9A10K_hourglass_operator.py',
      'code':root/'65AM_B9A10L_torch_force.py'}
    for key,p in d.items():
        if not p.is_file():raise FileNotFoundError('Missing '+key+': '+str(p))
    hm=json.loads(d['hm'].read_text());lm=json.loads(d['lm'].read_text());k=json.loads(d['k'].read_text());kc=json.loads(d['kc'].read_text())
    if sha(d['h'])!=hm['output_sha256'] or sha(d['l'])!=lm['file_sha256']:
        raise RuntimeError('Windows export checksum mismatch')
    if sha(d['hm'])!=lm['source_H_manifest_sha256']:
        raise RuntimeError('LE export from different H source')
    if sha(d['h'])!=k['source_files_sha256']['h'] or sha(d['plan'])!=k['source_files_sha256']['plan']:
        raise RuntimeError('K baseline does not refer to these H/plan files')
    if not k['numerical_screening_pass'] or abs(float(k['beta_global'])-float(kc['beta']))>1e-12:
        raise RuntimeError('K baseline/coefficient mismatch')
    if len(hm['cases'])!=24 or len(lm['cases'])!=24 or lm['case_count']!=24:
        raise RuntimeError('Expected 24 cases in H and L')
    for j in [f'B9A9_{r}{i:02d}' for r in REG for i in range(1,13)]:
        a=next((x for x in hm['cases'] if x['job']==j),None)
        b=next((x for x in lm['cases'] if x['job']==j),None)
        if not a or not b or a['odb_sha256']!=b['odb_sha256']:
            raise RuntimeError('Wrong ODB provenance for '+j)
    return d,k,float(kc['beta'])


def writecsv(p,records):
    if not records:raise RuntimeError('Empty CSV data '+str(p))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def pct(delta,ref):
    return 100.*float(np.linalg.norm(delta))/max(float(np.linalg.norm(ref)),1e-12)


def make_mask(H,d):
    return H.boundary(d['X'])


def metrics(f,d,b):
    base=float(np.linalg.norm(d['RF'][b]));delta=f-d['RF']
    return (100*np.linalg.norm(delta)/base,
            100*np.linalg.norm(delta[b])/base,
            100*np.linalg.norm(delta[~b])/base)


def calculate(root,check_only=False):
    d,k,beta=paths(root)
    H=import_module(d['hcode'],'b9a10h_existing')
    L=import_module(d['code'],'b9a10l_candidate')
    with np.load(d['plan'],allow_pickle=False) as z:
        for r in ('inner_left','inner_right'):
            if z['x_'+r].shape!=(12,12):raise RuntimeError('Wrong original material/PCA plan')
    with np.load(d['h'],allow_pickle=False) as hz,np.load(d['l'],allow_pickle=False) as lz:
        for r in REG:
            for i in range(1,13):
                j=f'B9A9_{r}{i:02d}';h=H.make_case(hz,j)
                lab=j+'_element_labels';le=j+'_LE';q=j+'_LE_Q_global_to_local'
                if not all(key in lz for key in (lab,le,q)):
                    raise RuntimeError('Missing LE arrays: '+j)
                if not np.array_equal(h['element_labels'],lz[lab]):raise RuntimeError('LE element order mismatch '+j)
                if lz[le].shape!=(728,6) or lz[q].shape!=(728,3,3):raise RuntimeError('Invalid LE shapes '+j)
                if np.max(np.abs(h['Q']-lz[q]))>1e-3:
                    raise RuntimeError('Abaqus S and LE orientations differ; inspect '+j)
    print('B9A10L CHECK PASS | H+L ODB provenance | 24 cases | frozen K beta %.9g | original materials'%beta)
    if check_only:return
    out=root/OUT
    if not out.is_dir():raise RuntimeError('Create this output folder in Finder before running: '+str(out))
    names=('B9A10L_cases.csv','B9A10L_modes.csv','B9A10L_summary.json',
           'B9A10L_force_comparison.png','B9A10L_strain_stress.png','B9A10L_gradients.json')
    target=[out/n for n in names]
    if any(p.exists() for p in target):raise RuntimeError('Previous L results protected. Move results manually if rerunning.')
    Kreg={};rows=[];mode_rows=[]
    baseline={}
    with d['km'].open(newline='') as f:
        for row in csv.DictReader(f):baseline[row['job']]=float(row['proxy_total_gap_pct'])
    if len(baseline)!=24:raise RuntimeError('Expected 24 K independent case metrics')
    with np.load(d['h'],allow_pickle=False) as hz, np.load(d['l'],allow_pickle=False) as lz,\
         np.load(d['plan'],allow_pickle=False) as pz, np.load(d['anchor'],allow_pickle=False) as anchor:
        for r in REG:
            matplan=pz['x_inner_left' if r=='L' else 'x_inner_right']
            d0=H.make_case(hz,f'B9A9_{r}01')
            Kreg[r]=L.ConstitutiveC3D8R(d0['node_labels'],d0['X'],d0['connectivity'],beta,d['kcode'])
            for i in range(12):
                j=f'B9A9_{r}{i+1:02d}';data=H.make_case(hz,j)
                model=Kreg[r]
                U=torch.as_tensor(data['U'],dtype=torch.float64)
                params=torch.as_tensor(matplan[i,:3],dtype=torch.float64)
                b=make_mask(H,data)
                fs_oracle,_=H.assemble(data)
                hg=model.hg(U,params[2]).detach().numpy()
                oracle=metrics(fs_oracle+hg,data,b)
                if abs(oracle[0]-baseline[j])>.002:
                    raise RuntimeError('K positive-control regression failed '+j+': '+str(oracle[0])+' versus '+str(baseline[j]))
                if not np.allclose(data['X'],d0['X'],atol=1e-8):raise RuntimeError('Mesh changed '+j)
                LE= lz[j+'_LE']
                Qobs=data['Q']
                # C3D8R material axes in B9A9 start along global XYZ; use predicted polar R only.
                fields=model.fields(U,params,'cauchy')
                qpred=fields['R'].transpose(-1,-2).detach().numpy()
                orient_diff=float(np.max(np.linalg.norm(qpred-Qobs,axis=(1,2))))
                E=fields['hencky_local'].detach().numpy()
                eps_tensor=L.tensor6_strain(torch.as_tensor(E)).detach().numpy()
                eps_eng=L.eng_from_tensor(torch.as_tensor(E)).detach().numpy()
                local_le_error_tensor=pct(eps_tensor-LE,LE)
                local_le_error_engineering=pct(eps_eng-LE,LE)
                C=L.compliance_stiffness(params).detach().numpy()
                # Diagnostics ONLY: oracle Abaqus LE as input to Hooke law; never used for prediction.
                sig_from_LE_plain=np.einsum('ij,ej->ei',C,LE)
                shear_twice=LE.copy();shear_twice[:,3:]*=2
                sig_from_LE_2x=np.einsum('ij,ej->ei',C,shear_twice)
                oracle_hooke_plain=pct(sig_from_LE_plain-data['S'],data['S'])
                oracle_hooke_2x=pct(sig_from_LE_2x-data['S'],data['S'])
                base_row=dict(job=j,region='inner_left' if r=='L' else 'inner_right',
                     condition=['material']*4+['interface']*4+['combined']*4,
                     E1_MPa=float(params[0]),E2_MPa=float(params[1]),G12_MPa=float(params[2]),
                     Q_pred_vs_Abaqus_max_F_norm=orient_diff,
                     strain_LE_tensor_pct=local_le_error_tensor,
                     strain_LE_engineering_pct=local_le_error_engineering,
                     oracle_S_from_LE_direct_shear_pct=oracle_hooke_plain,
                     oracle_S_from_LE_doubled_shear_pct=oracle_hooke_2x,
                     K_oracle_stress_plus_HG_force_pct=oracle[0])
                base_row['condition']=base_row['condition'][i]
                for method in MODELS:
                    fs,fh,fld=model.parts(U,params,stress_kind=method)
                    force=fs+fh
                    m=metrics(force.detach().numpy(),data,b)
                    prefix=method+'_'
                    base_row[prefix+'S_local_error_pct']=pct(fld['S_local'].detach().numpy()-data['S'],data['S'])
                    base_row[prefix+'force_all_pct']=m[0]
                    base_row[prefix+'force_interface_pct']=m[1]
                    base_row[prefix+'force_free_pct']=m[2]
                    for iface in H.FACES[r]:
                        p,scale=H.pca_force(data,force.detach().numpy(),anchor,iface)
                        true,_=H.pca_force(data,data['RF'],anchor,iface)
                        for mode,v in enumerate((p-true)/scale,1):
                            mode_rows.append(dict(job=j,region=base_row['region'],method=method,
                                                  interface=iface,mode=mode,scaled_signed_error=float(v),scaled_abs_error=abs(float(v))))
                rows.append(base_row)
                print('CASE',j,'strainTensor %.3f%%'%local_le_error_tensor,'cauchy force %.3f%%'%base_row['cauchy_force_all_pct'],
                      'kirchhoff force %.3f%%'%base_row['kirchhoff_force_all_pct'],
                      'oracle %.4f%%'%oracle[0])
    # Differentiable trial at one independent case. Neither Abaqus S nor RF is used as a force input.
    with np.load(d['h'],allow_pickle=False) as hz,np.load(d['plan'],allow_pickle=False) as pz:
        a=H.make_case(hz,'B9A9_L01');mod=Kreg['L']
        u=torch.tensor(a['U'],dtype=torch.float64,requires_grad=True)
        m=torch.tensor(pz['x_inner_left'][0,:3],dtype=torch.float64,requires_grad=True)
        b=make_mask(H,a)
        f=mod(u,m)
        loss=torch.mean((f[~torch.as_tensor(b)])**2)
        gradU,gradM=torch.autograd.grad(loss,(u,m))
        grad_good=bool(torch.isfinite(gradU).all() and torch.isfinite(gradM).all() and
                       torch.linalg.norm(gradU)>0 and torch.linalg.norm(gradM)>0)
        if not grad_good:raise RuntimeError('Differentiable free-node loss failed')
        gradients=dict(step='B9A10L',device='cpu',dtype='torch.float64',
            free_node_equilibrium_loss_float=float(loss.detach()),
            gradient_norm_displacement=float(torch.linalg.norm(gradU)),
            gradient_norm_material=float(torch.linalg.norm(gradM)),
            autograd_pass=grad_good,not_a_B5_training_run=True)
    agg={}
    for region in ('inner_left','inner_right'):
        rr=[x for x in rows if x['region']==region]
        agg[region]={}
        for method in MODELS:
            agg[region][method]={
               'pooled_force_pct':float(np.sqrt(np.mean([x[method+'_force_all_pct']**2 for x in rr]))),
               'worst_force_pct':float(max(x[method+'_force_all_pct'] for x in rr)),
               'pooled_stress_pct':float(np.sqrt(np.mean([x[method+'_S_local_error_pct']**2 for x in rr])))}
        agg[region]['oracle_K_force_pct']=float(np.sqrt(np.mean([x['K_oracle_stress_plus_HG_force_pct']**2 for x in rr])))
        agg[region]['LE_tensor_pct']=float(np.sqrt(np.mean([x['strain_LE_tensor_pct']**2 for x in rr])))
        agg[region]['LE_engineering_pct']=float(np.sqrt(np.mean([x['strain_LE_engineering_pct']**2 for x in rr])))
        agg[region]['oracle_hooke_direct_pct']=float(np.sqrt(np.mean([x['oracle_S_from_LE_direct_shear_pct']**2 for x in rr])))
        agg[region]['oracle_hooke_doubled_pct']=float(np.sqrt(np.mean([x['oracle_S_from_LE_doubled_shear_pct']**2 for x in rr])))
    gate=all(agg[r]['cauchy']['pooled_force_pct']<.25 and agg[r]['cauchy']['worst_force_pct']<.5 and
             agg[r]['cauchy']['pooled_stress_pct']<1 and
             min(agg[r]['LE_tensor_pct'],agg[r]['LE_engineering_pct'])<2 for r in agg)
    summary={'step':'65AM-B9A10-L','basis':'Hencky log(U) with polar rotation from displacement; fixed B9A10K hourglass proxy',
       'candidate_not_exact_Abaqus':True,'independent_24_cases':True,'no_fitting_on_validation_data':True,
       'linear_constitutive_constants':{'E3_MPa':12000,'nu12':.28,'nu13':.28,'nu23':.40,'G13_MPa':4500,'G23_MPa':3500},
       'material_orientation_reference':'identity (B9A9 MAT_ORI)',
       'methods':['Cauchy = C:logU','Kirchhoff = C:logU, sigma=tau/J'],
       'LE_shear_conventions_both_reported_not_fitted':True,
       'independent_regions':agg,'autograd':gradients,'screening_gate_cauchy':bool(gate),
       'screening_thresholds':{'pooled_force_pct':.25,'worst_force_pct':.5,'pooled_stress_pct':1.0,'LE_comparison_pct':2.0},
       'physics_loss_ready':False,'b5_training_done':False,'broyden_changed':False,
       'limitations':['Predicted S is a total-strain corotational approximation, whereas Abaqus may integrate an objective stress rate.',
                      'Abaqus LE/S used ONLY for diagnosis; predictions use U, material and reference mesh.',
                      'No matching calibrated stress to 24 case labels; no test tuning.',
                      'K beta remains a fitted mesh-specific candidate. Passing screening does not establish exact Abaqus law.'],
       'source_hashes':{key:sha(d[key]) for key in ('h','l','plan','kc','k')}}
    writecsv(target[0],rows);writecsv(target[1],mode_rows)
    target[2].write_text(json.dumps(summary,indent=2));target[5].write_text(json.dumps(gradients,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,r in zip(axes,('inner_left','inner_right')):
        rr=[x for x in rows if x['region']==r]
        ax.plot(range(1,13),[x['K_oracle_stress_plus_HG_force_pct'] for x in rr],label='Oracle Abaqus S + K (control)')
        ax.plot(range(1,13),[x['cauchy_force_all_pct'] for x in rr],label='U-derived Cauchy+K')
        ax.plot(range(1,13),[x['kirchhoff_force_all_pct'] for x in rr],label='U-derived Kirchhoff+K')
        ax.set_title(r);ax.set_xlabel('Independent B9A9 case');ax.set_ylabel('Nodal-force error / boundary RF (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(target[3],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    tags=['L tensor LE','L eng LE','L direct Hooke','L double Hooke',
          'R tensor LE','R eng LE','R direct Hooke','R double Hooke']
    vals=[]
    for r in ('inner_left','inner_right'):
        vals.extend([agg[r]['LE_tensor_pct'],agg[r]['LE_engineering_pct'],
                     agg[r]['oracle_hooke_direct_pct'],agg[r]['oracle_hooke_doubled_pct']])
    ax.bar(range(8),vals);ax.set_xticks(range(8),tags,rotation=35,ha='right')
    ax.set_ylabel('Pooled strain/stress diagnostic error (%)');fig.tight_layout();fig.savefig(target[4],dpi=160);plt.close(fig)
    for r,v in agg.items():
        print('B9A10L',r,'oracle-K %.4f%%'%v['oracle_K_force_pct'],
              'cauchy %.4f%%'%v['cauchy']['pooled_force_pct'],
              'kirchhoff %.4f%%'%v['kirchhoff']['pooled_force_pct'])
    print('B9A10L DONE | cauchy screening:',gate,'| physics loss ready: NO | results:',out)


def selftest(root):
    mod=import_module(root/'65AM_B9A10L_torch_force.py','b9a10l_new')
    mod.selftest()
    print('B9A10L VALIDATION SELFTEST PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('selftest','check','validate'),required=True)
    a=p.parse_args();root=Path(a.project).resolve()
    if a.stage=='selftest':selftest(root)
    else:calculate(root,check_only=a.stage=='check')
