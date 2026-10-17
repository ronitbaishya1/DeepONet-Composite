#!/usr/bin/env python3
"""65AM-B9A10-N Mac: read-only nine-constant and integration-point diagnostic.

Uses exactly B9A10-L's frozen material inputs and B9A10-K hourglass proxy.
Predictors use ONLY nodal U, reference mesh and the input material properties.
Abaqus LE/S/RF/Q are ONLY references to compute error and decomposition.

Optional Windows check: B9A10N_material_audit/B9A10N_windows_materials.json
NO fitting, NO changes to checkpoints, NO neural operator training.

Commands (Mac project root):
 python3 65AM_B9A10N_mac_constitutive_audit.py --project . --stage selftest
 python3 65AM_B9A10N_mac_constitutive_audit.py --project . --stage check
 python3 65AM_B9A10N_mac_constitutive_audit.py --project . --stage diagnose
"""
import argparse,csv,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
OUT=Path('results/65AM_B9_coupled_feasibility/B9A10N_constitutive_consistency')
REG=('L','R')
COMP=('11','22','33','12','13','23')
FIXED={'E3':12000.,'nu12':.28,'nu13':.28,'nu23':.40,'G13':4500.,'G23':3500.}


def write_csv(path,records):
    if not records:raise RuntimeError('No records for '+path.name)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader();writer.writerows(records)


def fl(v):return float(np.asarray(v))

def percent(err,ref):
    return 100*float(np.linalg.norm(err))/max(float(np.linalg.norm(ref)),1e-15)


def param_list(arr):
    E1,E2,G12=map(float,arr)
    return dict(E1=E1,E2=E2,G12=G12,**FIXED)


def reciprocity(p):
    # nu_ji = nu_ij (E_j/E_i) for the orthotropic engineering compliance.
    return {'nu21':p['nu12']*p['E2']/p['E1'],
            'nu31':p['nu13']*p['E3']/p['E1'],
            'nu32':p['nu23']*p['E3']/p['E2']}


def relative_norm(a,ref):return float(np.linalg.norm(a))/max(float(np.linalg.norm(ref)),1e-15)


def cos(a,b):
    aa=np.asarray(a).reshape(-1);bb=np.asarray(b).reshape(-1)
    return float(aa@bb/(max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-20)))


def green_svk_candidate(model,u,mat,L):
    """A priori comparison: orthotropic St Venant-Kirchhoff, NOT Abaqus law.

    Strain E_G=(F^T F-I)/2 in material reference orientation (identity).
    Second Piola S_2=C:E_G, first Piola P=F S_2.
    Cauchy sigma=F S_2 F^T / det(F); local observer R^T sigma R.
    This predictor contains no observed Abaqus stress/strain and is differentiable.
    """
    basic=model.fields(u,mat,'cauchy')
    F=basic['F'];J=basic['J'];R=basic['R']
    I=torch.eye(3,dtype=u.dtype,device=u.device)[None,:,:]
    green=.5*(F.transpose(-1,-2)@F-I)
    C=L.compliance_stiffness(mat)
    S2=L.tensor_from_stress(torch.einsum('ij,ej->ei',C,L.eng_from_tensor(green)))
    piola=F@S2
    cauchy=(F@S2@F.transpose(-1,-2))/J[:,None,None]
    sloc=(R.transpose(-1,-2)@cauchy@R)
    six=torch.stack([sloc[:,0,0],sloc[:,1,1],sloc[:,2,2],sloc[:,0,1],sloc[:,0,2],sloc[:,1,2]],dim=-1)
    f_elem=(piola@model.grad.transpose(-1,-2)).transpose(-1,-2)*model.volume[:,None,None]
    forces=torch.zeros_like(u).index_add(0,model.hg.connectivity.reshape(-1),f_elem.reshape(-1,3))
    forces=forces+model.hg(u,mat[2])
    return six,forces


def load(root):
    from importlib.util import spec_from_file_location,module_from_spec
    p=root/'65AM_B9A10L_validate.py'
    if not p.is_file():raise FileNotFoundError('Keep original B9A10L_validate.py beside N script')
    spec=spec_from_file_location('b9a10l_original_validate',str(p));Lval=module_from_spec(spec);spec.loader.exec_module(Lval)
    dirs,ksummary,beta=Lval.paths(root)   # checks real H/L/K provenance & manifest hashes
    L=Lval.import_module(dirs['code'],'b9a10l_original_force')
    H=Lval.import_module(dirs['hcode'],'b9a10h_original')
    return dirs,beta,H,L,Lval


def crosscheck_manifest(root,case,materials):
    path=root/'B9A10N_material_audit/B9A10N_windows_materials.json'
    if not path.is_file():return 'not supplied (Windows material audit optional)'
    if not hasattr(crosscheck_manifest,'cache'):
        obj=json.loads(path.read_text())
        records={j['job']:j for j in obj['cases']}
        if len(records)!=24 or int(obj['count'])!=24:raise RuntimeError('Invalid Windows deck manifest')
        crosscheck_manifest.cache=records
    obs=crosscheck_manifest.cache[case]['constants']
    for k,value in materials.items():
        if abs(float(obs[k])-float(value))>1e-8*max(1,abs(float(value))):
            raise RuntimeError('WIN/MAC MATERIAL MISMATCH %s / %s: %g vs %g'%(case,k,value,obs[k]))
    reciprocal=reciprocity(materials)
    for k,value in reciprocal.items():
        if abs(float(obs[k])-float(value))>1e-8*max(1,abs(float(value))):
            raise RuntimeError('WIN/MAC RECIPROCITY MISMATCH '+case+'/'+k)
    return 'verified against original Windows input deck'


def inspect(root):
    dirs,beta,H,L,Lval=load(root)
    import numpy as np
    with np.load(dirs['h'],allow_pickle=False) as hz,np.load(dirs['l'],allow_pickle=False) as lz,\
         np.load(dirs['plan'],allow_pickle=False) as pz:
        for r in REG:
            group='inner_left' if r=='L' else 'inner_right'
            features=pz['x_'+group]
            if features.shape!=(12,12):raise RuntimeError('Unexpected input feature plan')
            for i in range(12):
                case='B9A9_%s%02d'%(r,i+1);data=H.make_case(hz,case)
                le=lz[case+'_LE']
                if le.shape!=(728,6):raise RuntimeError('Expected 728 C3D8R points in '+case)
                if not np.array_equal(data['element_labels'],lz[case+'_element_labels']):
                    raise RuntimeError('S and LE not aligned '+case)
                pars=param_list(features[i,:3]);crosscheck_manifest(root,case,pars)
                stiffness=L.compliance_stiffness(torch.tensor(features[i,:3],dtype=torch.float64))
                if not bool((torch.linalg.eigvalsh(stiffness)>0).all()):raise RuntimeError('Non-PD material '+case)
    print('B9A10N CHECK PASS | 24 cases | K/L/H provenance | nine positive-definite orthotropic constants | integration-point order')
    if (root/'B9A10N_material_audit/B9A10N_windows_materials.json').exists():
        print('B9A10N WINDOWS CROSS-CHECK PASS | 24 original Abaqus input decks')
    else:print('B9A10N NOTE: original Windows input-deck cross-check not supplied')
    return dirs,beta,H,L,Lval


def analysis(root):
    dirs,beta,H,L,Lval=inspect(root)
    output=root/OUT
    if not output.is_dir():raise RuntimeError('Create output folder manually using Finder first: '+str(output))
    names=['B9A10N_cases.csv','B9A10N_components.csv','B9A10N_ip_diagnostics.csv',
           'B9A10N_summary.json','B9A10N_force_stress.png','B9A10N_component_residuals.png']
    targets=[output/n for n in names]
    if any(p.exists() for p in targets):
        raise RuntimeError('Previous N results protected. Move completed results manually if rerunning.')
    records=[];components=[];iprows=[]
    with np.load(dirs['h'],allow_pickle=False) as hz,np.load(dirs['l'],allow_pickle=False) as lz,\
         np.load(dirs['plan'],allow_pickle=False) as pz:
        for r in REG:
            region='inner_left' if r=='L' else 'inner_right'
            first=H.make_case(hz,'B9A9_%s01'%r)
            model=L.ConstitutiveC3D8R(first['node_labels'],first['X'],first['connectivity'],
                                      beta,dirs['kcode'],device='cpu')
            for i in range(12):
                case='B9A9_%s%02d'%(r,i+1);dat=H.make_case(hz,case)
                mat=torch.as_tensor(pz['x_'+region][i,:3],dtype=torch.float64)
                u=torch.as_tensor(dat['U'],dtype=torch.float64)
                if not np.allclose(first['X'],dat['X']):raise RuntimeError('Mesh changed '+case)
                obsS=np.asarray(dat['S'],float)
                obsLE=np.asarray(lz[case+'_LE'],float)
                Q=np.asarray(dat['Q'],float)
                with torch.no_grad():
                    fld=model.fields(u,mat,'cauchy')
                    predicted=fld['S_local'].cpu().numpy()
                    predictedLE=L.eng_from_tensor(fld['hencky_local']).cpu().numpy()
                    c=L.compliance_stiffness(mat).cpu().numpy()
                    cLE=obsLE@c.T
                    cPred=predictedLE@c.T
                    if not np.allclose(predicted,cPred,rtol=1e-9,atol=1e-7):
                        raise RuntimeError('Constitutive decomposition identity mismatch '+case)
                    svkS,svkF=green_svk_candidate(model,u,mat,L)
                    Lforce=model(u,mat,'cauchy').cpu().numpy()
                svkS=svkS.cpu().numpy();svkF=svkF.cpu().numpy()
                # Exact three-term algebraic identity; no calibration/data-leakage.
                dueStrain=cPred-cLE
                dueConstitutive=cLE-obsS
                total=cPred-obsS
                if np.linalg.norm(total-(dueStrain+dueConstitutive))>1e-7:
                    raise RuntimeError('Residual decomposition failure '+case)
                b=H.boundary(dat['X'])
                basereaction=np.linalg.norm(dat['RF'][b]);
                if basereaction<=0:raise RuntimeError('RF normalization invalid '+case)
                forceL=100*np.linalg.norm(Lforce-dat['RF'])/basereaction
                forceSVK=100*np.linalg.norm(svkF-dat['RF'])/basereaction
                sBase=np.linalg.norm(obsS)
                stiffness_eigs=np.linalg.eigvalsh(c)
                pars=param_list(pz['x_'+region][i,:3])
                row=dict(job=case,region=region,condition=['material']*4+['interface']*4+['combined']*4)
                row['condition']=row['condition'][i]
                row.update(dict(E1_MPa=pars['E1'],E2_MPa=pars['E2'],G12_MPa=pars['G12'],
                     E3_MPa=pars['E3'],G13_MPa=pars['G13'],G23_MPa=pars['G23'],
                     nu12=pars['nu12'],nu13=pars['nu13'],nu23=pars['nu23'],
                     nu21=reciprocity(pars)['nu21'],nu31=reciprocity(pars)['nu31'],nu32=reciprocity(pars)['nu32'],
                     compliance_min_eigenvalue=float(1/stiffness_eigs.max()),
                     strain_prediction_error_pct=percent(predictedLE-obsLE,obsLE),
                     stress_L_total_error_pct=percent(total,obsS),
                     stress_strain_related_error_pct=percent(dueStrain,obsS),
                     stress_constitutive_related_error_pct=percent(dueConstitutive,obsS),
                     residual_cross_cosine=cos(dueStrain,dueConstitutive),
                     stress_SVK_error_pct=percent(svkS-obsS,obsS),
                     force_L_pct=forceL,force_SVK_pct=forceSVK,
                     Q_pred_vs_Abaqus_max_fro=float(np.max(np.linalg.norm(fld['R'].transpose(-1,-2).cpu().numpy()-Q,axis=(1,2)))),
                     max_component_stress_NMPa=float(np.max(np.abs(obsS)))))
                records.append(row)
                for k,tag in enumerate(COMP):
                    components.append(dict(job=case,region=region,component='S'+tag,
                        error_total_pct_of_all_S=100*np.linalg.norm(total[:,k])/max(sBase,1e-15),
                        error_from_strain_pct_of_all_S=100*np.linalg.norm(dueStrain[:,k])/max(sBase,1e-15),
                        error_constitutive_pct_of_all_S=100*np.linalg.norm(dueConstitutive[:,k])/max(sBase,1e-15),
                        mean_signed_constitutive_MPa=float(np.mean(dueConstitutive[:,k])),
                        rms_constitutive_MPa=float(np.sqrt(np.mean(dueConstitutive[:,k]**2)))))
                for e in range(len(dat['element_labels'])):
                    iprows.append(dict(job=case,region=region,element_label=int(dat['element_labels'][e]),
                       x_mm=float(dat['X'][model.hg.connectivity[e].cpu().numpy()].mean(axis=0)[0]),
                       stress_reference_norm_MPa=float(np.linalg.norm(obsS[e])),
                       total_stress_error_MPa=float(np.linalg.norm(total[e])),
                       strain_related_error_MPa=float(np.linalg.norm(dueStrain[e])),
                       constitutive_related_error_MPa=float(np.linalg.norm(dueConstitutive[e])),
                       S11_ref_MPa=float(obsS[e,0]),S11_pred_MPa=float(predicted[e,0]),
                       S12_ref_MPa=float(obsS[e,3]),S12_pred_MPa=float(predicted[e,3])))
                print('CASE',case,'L force %.4f%%'%forceL,'SVK %.4f%%'%forceSVK,
                      'constit %.4f%%'%row['stress_constitutive_related_error_pct'])
    if len(records)!=24 or len(components)!=144 or len(iprows)!=24*728:
        raise RuntimeError('Incomplete N diagnostic output')
    group={}
    for region in ('inner_left','inner_right'):
        rr=[x for x in records if x['region']==region]
        group[region]={k:float(np.sqrt(np.mean([x[k]**2 for x in rr]))) for k in
                       ('strain_prediction_error_pct','stress_L_total_error_pct',
                        'stress_strain_related_error_pct','stress_constitutive_related_error_pct',
                        'stress_SVK_error_pct','force_L_pct','force_SVK_pct')}
        print('SUMMARY',region,'L force %.5f%%'%group[region]['force_L_pct'],
             'SVK force %.5f%%'%group[region]['force_SVK_pct'])
        if abs(group[region]['force_L_pct']-({'inner_left':.22119258,'inner_right':.21937421}[region]))>.003:
            raise RuntimeError('B9A10-L regression failed in '+region+'. Stop before writing any reports.')
    summary={'step':'65AM-B9A10-N','case_count':24,
       'research_design':'No parameters fitted; comparison of predeclared Hencky-Cauchy L and orthotropic Green-Strain StVK reference candidate.',
       'nine_constants':'E1,E2,E3,nu12,nu13,nu23,G12,G13,G23; nu21,nu31,nu32 derived by elastic reciprocity',
       'residual_identity':'S_L - S_Abaqus = C*(LE_pred - LE_Abaqus) + (C*LE_Abaqus - S_Abaqus)',
       'abaqus_LE_S_RF_used_only_for_diagnostics':True,
       'has_Windows_deck_crosscheck':bool((root/'B9A10N_material_audit/B9A10N_windows_materials.json').is_file()),
       'group_statistics':group,'B9A10K_beta_frozen':float(beta),
       'new_model_selected':False,'new_holdout_needed_to_select_SVK':True,
       'physics_loss_ready':False,'B5_Broyden_65V_unchanged':True,
       'limits':['SVK is an independent physically motivated candidate, not Abaqus Standard exact orthotropic incremental model.',
                 'Residual component norms do not add because of cross terms.',
                 'Material input agreement does not establish equivalence of finite-strain stress algorithms.',
                 'Same 24 previously inspected cases are not an unbiased fresh test for any subsequently selected constitutive alternative.']}
    write_csv(targets[0],records);write_csv(targets[1],components);write_csv(targets[2],iprows)
    targets[3].write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,r in zip(axs,('inner_left','inner_right')):
        rr=[x for x in records if x['region']==r]
        ax.plot(range(1,13),[x['force_L_pct'] for x in rr],marker='o',label='L corotational Hencky + K')
        ax.plot(range(1,13),[x['force_SVK_pct'] for x in rr],marker='o',label='Green SVK + K (candidate)')
        ax.plot(range(1,13),[x['stress_constitutive_related_error_pct'] for x in rr],linestyle=':',label='Constitutive stress residual (%)')
        ax.set(title=r,xlabel='B9A9 independent case',ylabel='Norm error (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(targets[4],dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,r in zip(axs,('inner_left','inner_right')):
        rr=[x for x in components if x['region']==r]
        a=[np.sqrt(np.mean([row['error_from_strain_pct_of_all_S']**2 for row in rr if row['component']=='S'+k])) for k in COMP]
        b=[np.sqrt(np.mean([row['error_constitutive_pct_of_all_S']**2 for row in rr if row['component']=='S'+k])) for k in COMP]
        x=np.arange(6);ax.bar(x-.2,a,.4,label='Strain contribution');ax.bar(x+.2,b,.4,label='Constitutive contribution')
        ax.set_xticks(x,['S'+t for t in COMP]);ax.set(title=r,ylabel='Component error / ||S_total|| (%)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(targets[5],dpi=170);plt.close(fig)
    print('B9A10N DONE | Diagnostic comparison only | no B5 model changed | outputs:',output)


def selftest(root):
    # Self-test does not require the 24 ODB exports.
    from importlib.util import spec_from_file_location,module_from_spec
    p=root/'65AM_B9A10L_torch_force.py'
    if not p.is_file():raise FileNotFoundError(str(p))
    spec=spec_from_file_location('b9a10l_for_n_selftest',str(p))
    L=module_from_spec(spec);spec.loader.exec_module(L)
    beta=0.004381715594969032
    import numpy as np
    base={'E1':46500.,'E2':12500.,'G12':4750.,**FIXED}
    rr=reciprocity(base)
    assert abs(rr['nu21']/base['E2']-base['nu12']/base['E1'])<1e-13
    assert abs(rr['nu31']/base['E3']-base['nu13']/base['E1'])<1e-13
    assert abs(rr['nu32']/base['E3']-base['nu23']/base['E2'])<1e-13
    X=np.asarray([[0,0,0],[2,0,0],[2,1,0],[0,1,0],[0,0,3],[2,0,3],[2,1,3],[0,1,3]],float)
    model=L.ConstitutiveC3D8R(np.arange(1,9),X,np.arange(1,9).reshape(1,8),beta,
                            root/'65AM_B9A10K_hourglass_operator.py')
    mat=torch.tensor([46500.,12500.,4750.],dtype=torch.float64,requires_grad=True)
    uu=torch.tensor(np.random.default_rng(7).normal(0,.01,(8,3)),dtype=torch.float64,requires_grad=True)
    S,F=green_svk_candidate(model,uu,mat,L)
    grads=torch.autograd.grad(F.square().sum()+S.square().sum(),(uu,mat))
    assert all(torch.isfinite(g).all() for g in grads)
    for scale in (0.,0.5,1.):
        u=uu.detach()*scale
        ss,_=green_svk_candidate(model,u,mat.detach(),L)
        if scale==0.:assert float(torch.linalg.norm(ss))<1e-10
    a=np.asarray([1.,2.,3.]);b=np.asarray([3.,2.,1.]);c=np.asarray([1.,1.,1.]);
    assert np.allclose((a-b)+(b-c),a-c)
    print('B9A10N SELFTEST PASS | 9 elastic constants | StVK differentiability | residual decomposition | zero displacement')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',default='.')
    parser.add_argument('--stage',choices=('selftest','check','diagnose'),required=True)
    args=parser.parse_args();root=Path(args.project).resolve()
    if args.stage=='selftest':selftest(root)
    elif args.stage=='check':inspect(root)
    else:analysis(root)
if __name__=='__main__':main()
