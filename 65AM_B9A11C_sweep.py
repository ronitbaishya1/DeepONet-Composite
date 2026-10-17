#!/usr/bin/env python3
"""B9A11-C: leakage-aware POD adapter physics-loss sweep, Mac CPU float64.

Requires B9A11-B NATIVE_FIXED and its objectives and prior validated project data.
Never updates B5 / Broyden / 65V, never uses 12 internal-tune cases for training
or hyperparameter selection. This is NOT independent generalization.

Stage selftest: mathematical and selection safeguards.
Stage check: exercise original B5 forward pass and audited physical mesh.
Stage sweep: fit-only train/validation, frozen K beta; evaluate tune ONCE at end.
Stage report: reopen an existing summary, no retraining.
"""
import argparse, csv, copy, hashlib, importlib.util, json, math, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASENAME='65AM_B9A11B_train_NATIVE_FIXED.py'
OUTPUT=Path('results/65AM_B9_coupled_feasibility/B9A11C_weight_convergence')
REGIONS=('inner_left','inner_right')
WEIGHTS_FULL=(0.,.25,.5,1.,2.,4.)
WEIGHTS_PILOT=(0.,.5,1.)
SEEDS_FULL=(11,27,43)
SEEDS_PILOT=(27,)


def import_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:raise ImportError(str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write_csv(p,rows):
    if not rows:raise RuntimeError('No records for '+str(p))
    names=list(rows[0].keys())
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');w.writeheader();w.writerows(rows)


def val(x):return float(x.detach().cpu().item() if torch.is_tensor(x) else x)

def norm(x):return torch.linalg.vector_norm(x.reshape(-1))

def jser(x):
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x)
    if isinstance(x,(np.ndarray,)):return x.tolist()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(str(type(x)))


def load_b(root):
    path=root/BASENAME
    if not path.is_file():raise FileNotFoundError('Missing ORIGINAL working B9A11-B NATIVE_FIXED: '+str(path))
    return import_module(path,'b9a11b_native_for_c')


def selection_score(data_ratio,force_ratio):
    if not math.isfinite(data_ratio+force_ratio):return float('inf')
    return 0.5*data_ratio+0.5*force_ratio


def force_ratio_linear(s,delta,mask):
    f=s['f0'][~mask].reshape(-1)+s['jac']@delta
    return norm(f-s['fr'][~mask].reshape(-1))/s['den']


def objective(s,adapter,features,cs,basis,mask,B,Obj,kind,lam,trust,margin):
    u,dc=B.get_u(s,adapter,features,cs,basis,mask,trust)
    ldata=Obj.normalized_data_loss(u,s['ref'],~mask)
    if kind=='supervised':
        lphysics=ldata.new_zeros(())
    else:
        f=s['f0'][~mask].reshape(-1)+s['jac']@dc
        if kind=='floor':
            lphysics=Obj.floor_aware_equilibrium(f,s['den'],s['floor'],margin)
        elif kind=='force_match':
            # SUPERVISED force regularizer. FEM reference force used as label
            # on TRAIN cases, not a traction-free or unsupervised physics loss.
            lphysics=(norm(f-s['fr'][~mask].reshape(-1))/s['den'])**2
        else:raise ValueError('Unknown objective: '+kind)
    return ldata+lam*lphysics,ldata,lphysics


def grad_weight(train_samples,basis,mask,Obj,kind,margin,target):
    ratios=[]
    for s in train_samples:
        p=s['coeff0'].clone().detach().requires_grad_(True)
        u=s['lift']+torch.einsum('r,rnc->nc',p,basis)
        ld=Obj.normalized_data_loss(u,s['ref'],~mask)
        ff=s['f0'][~mask].reshape(-1)+s['jac']@(p-s['coeff0'])
        if kind=='floor':lp=Obj.floor_aware_equilibrium(ff,s['den'],s['floor'],margin)
        elif kind=='force_match':lp=(norm(ff-s['fr'][~mask].reshape(-1))/s['den'])**2
        else:raise ValueError(kind)
        gd=torch.autograd.grad(ld,p,retain_graph=True)[0]
        gp=torch.autograd.grad(lp,p)[0]
        d=norm(gd).item();f=norm(gp).item()
        if d>1e-20 and f>1e-20:ratios.append(f/d)
    if not ratios:
        print('B9A11C WARNING: no active '+kind+' gradient; setting its calibrated weight to zero',flush=True)
        return 0.,0.
    ratio=float(np.median(ratios))
    lam=float(np.clip(target/ratio,1.e-8,1.e-2))
    return lam,ratio


def metrics(model,samples,indices,features,cs,basis,mask,B,Obj,trust):
    """Fast validation metrics. Data and force use the SAME denominators across arms."""
    dd=[];fr=[];eq=[]
    with torch.no_grad():
        for i in indices:
            s=samples[int(i)]
            u,delta=B.get_u(s,model,features[int(i)],cs,basis,mask,trust)
            dd.append(val(Obj.normalized_data_loss(u,s['ref'],~mask)))
            f=s['f0'][~mask].reshape(-1)+s['jac']@delta
            fr.append(val((norm(f-s['fr'][~mask].reshape(-1))/s['den'])**2))
            eq.append(val((norm(f)/s['den'])**2))
    return {'displacement_sq':float(np.mean(dd)),
            'force_difference_sq':float(np.mean(fr)),
            'equilibrium_sq':float(np.mean(eq))}


def fit_adapter(B,Obj,samples,train_ids,val_ids,mean,std,cs,basis,mask,lam,kind,seed,
                epochs,lr,trust,margin,patience,history,region,multiplier):
    torch.manual_seed(seed)
    model=Obj.PODResidualAdapter(len(mean),basis.shape[0],hidden=24).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    features=[(s['raw_features']-mean)/std for s in samples]
    # Fixed objective used for early stopping across ALL arms: validation
    # displacement + normalized validation force consistency, not the
    # variable weighted training loss. Validation force targets are used only
    # for selection, not training in the floor formulation.
    ref_model=Obj.PODResidualAdapter(len(mean),basis.shape[0],hidden=24).double()
    baseline=metrics(ref_model,samples,val_ids,features,cs,basis,mask,B,Obj,trust)
    den_d=max(baseline['displacement_sq'],1e-22)
    den_f=max(baseline['force_difference_sq'],1e-22)
    best_score=float('inf');best_epoch=0;best_sd=None;best_metrics=None;stale=0
    for epoch in range(1,epochs+1):
        model.train()
        rng=np.random.default_rng(seed*100003+epoch)
        for i in rng.permutation(train_ids):
            optimizer.zero_grad(set_to_none=True)
            total,_,_=objective(samples[int(i)],model,features[int(i)],cs,basis,mask,B,Obj,
                                kind,lam,trust,margin)
            if not torch.isfinite(total):raise RuntimeError('Nonfinite objective for '+region)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step()
        model.eval()
        m=metrics(model,samples,val_ids,features,cs,basis,mask,B,Obj,trust)
        score=selection_score(m['displacement_sq']/den_d,m['force_difference_sq']/den_f)
        history.append({'region':region,'kind':kind,'weight_multiplier':multiplier,
            'seed':seed,'epoch':epoch,'lambda_eq':lam,'val_displacement_sq':m['displacement_sq'],
            'val_force_difference_sq':m['force_difference_sq'],
            'val_equilibrium_sq':m['equilibrium_sq'],'val_common_score':score})
        if score<best_score-1e-9:
            best_score=score;best_epoch=epoch;best_sd=copy.deepcopy(model.state_dict());best_metrics=m;stale=0
        else:stale+=1
        if stale>=patience:break
    if best_sd is None:raise RuntimeError('Optimizer failed without a valid epoch')
    model.load_state_dict(best_sd)
    return model,best_epoch,best_score,best_metrics,baseline


def summary_metrics(B,samples,model,mean,std,cs,basis,mask,force,Obj,trust):
    rr=B.evaluate_exact(samples,model,mean,std,cs,basis,mask,force,Obj,trust,1.25,'','', '')
    keys=['displacement_pct','free_residual_pct','additional_free_force_pct',
          'reference_interface_force_difference_pct']
    return {key:float(np.mean([r[key] for r in rr])) for key in keys}


def modes_on_tune(B,c,H,root,d,tune_samples,models,mean,std,cs,basis,mask,force,trust,region):
    modes=[]
    anchor=root/'B9_coupled_anchor/B9_65V_anchor.npz'
    coords={'X':np.asarray(d['ds']['coordinates'],dtype=float),
            'node_labels':np.arange(1,len(d['ds']['coordinates'])+1,dtype=np.int64)}
    with np.load(anchor,allow_pickle=False) as anc:
        for name,mod in models.items():
            for s in tune_samples:
                ft=(s['raw_features']-mean)/std
                with torch.no_grad():
                    u,_=B.get_u(s,mod,ft,cs,basis,mask,trust)
                    ff=force(u,s['mat']).detach().numpy()
                for face in H.FACES['L' if region=='inner_left' else 'R']:
                    p,sc=H.pca_force(coords,ff,anc,face)
                    ref,_=H.pca_force(coords,s['fr'].detach().numpy(),anc,face)
                    for k,(a,b,scale) in enumerate(zip(p,ref,sc),start=1):
                        modes.append(dict(region=region,arm=name,case_id=s['id'],interface=face,mode=k,
                              normalized_pca_force_difference=abs(float(a)-float(b))/float(scale)))
    return modes


def run(root,a):
    start=time.time()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(a.threads)
    B=load_b(root)
    A,Obj,c,H,checked,beta,source=B.verify(root,a)
    out=root/OUTPUT/a.run_tag
    if out.exists():raise RuntimeError('Output folder already exists. Choose a NEW --run-tag: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    matrix=[];curves=[];final=[];fitval=[];modes=[];summary={
        'step':'B9A11C','run_tag':a.run_tag,'preset':a.preset,'training_type':'frozen B5 + POD adapter; not backbone fine-tuning',
        'independent_validation':False,'model_selection':'fit 35 training / 9 validation; 12 internal tune only after selection',
        'reference_label_policy':'force_match is explicitly SUPERVISED force consistency; floor is reference-floor-aware residual',
        'beta_frozen':beta,'source_sha256':source,'base_checkpoint_modified':False,
        'approved_for_coupled_use':False,'results':{},'threshold_relative_gain_vs_supervised':a.min_force_gain,
        'displacement_max_degradation':a.max_disp_degradation}
    kinds=('supervised','floor','force_match')
    weights=WEIGHTS_PILOT if a.preset=='pilot' else WEIGHTS_FULL
    seeds=SEEDS_PILOT if a.preset=='pilot' else SEEDS_FULL
    for region in ((a.region,) if a.region!='both' else REGIONS):
        st=checked[region];d=st['d'];force=st['force'];fit=np.asarray(d['fit'],dtype=int)
        splitseed=27+(1 if region=='inner_left' else 2)
        rng=np.random.default_rng(splitseed);perm=rng.permutation(len(fit))
        n_val=max(2,int(math.ceil(len(fit)*.20)))
        vi=perm[:n_val];ti=perm[n_val:]
        samples,basis,mask,baseline_name=B.assemble_samples(c,st,region,fit)
        mean,std,cs=B.feature_statistics(samples,ti)
        B.make_linearized_tangents(samples,basis,mask,force)
        scaled={}
        for kind in ('floor','force_match'):
            scaled[kind]=grad_weight([samples[int(i)] for i in ti],basis,mask,Obj,kind,
                                       a.floor_margin,a.target_grad_fraction)
        best_by_kind={}; bestmodels={}
        for kind in kinds:
            these=(0.,) if kind=='supervised' else weights[1:]
            for mult in these:
                lam=0. if kind=='supervised' else scaled[kind][0]*mult
                for seed in seeds:
                    model,ep,score,mm,baseline=fit_adapter(B,Obj,samples,ti,vi,mean,std,cs,basis,mask,
                        lam,kind,seed,a.epochs,a.lr,a.max_correction_pct,a.floor_margin,
                        a.patience,curves,region,mult)
                    row={'region':region,'kind':kind,'multiplier':mult,'seed':seed,
                         'lambda':lam,'epoch':ep,'val_score':score,
                         'val_disp_pct':100*math.sqrt(mm['displacement_sq']),
                         'val_force_diff_pct':100*math.sqrt(mm['force_difference_sq']),
                         'val_equilibrium_pct':100*math.sqrt(mm['equilibrium_sq'])}
                    matrix.append(row)
                    if kind not in best_by_kind or score<best_by_kind[kind]['val_score']:
                        best_by_kind[kind]=row.copy();bestmodels[kind]=copy.deepcopy(model)
                    print('B9A11C',region,kind,'x',mult,'seed',seed,'best epoch',ep,
                          'validation score %.6g'%score,flush=True)
        # VALIDATION-only selection. Choose hyperparameters/seed by common
        # linearized validation score, then check the selected models using
        # EXACT NONLINEAR forces on FIT VALIDATION (never on internal tune).
        exact_validation={}; max_linearization={}
        for kind in kinds:
            rr=B.evaluate_exact([samples[int(i)] for i in vi],bestmodels[kind],
                     mean,std,cs,basis,mask,force,Obj,a.max_correction_pct,
                     a.floor_margin,region,kind,'fit_validation')
            fitval.extend(rr)
            keys=('displacement_pct','free_residual_pct','additional_free_force_pct',
                  'reference_interface_force_difference_pct')
            exact_validation[kind]={k:float(np.mean([r[k] for r in rr])) for k in keys}
            max_linearization[kind]=float(max(r['linearization_rel_error'] for r in rr))
        sup=exact_validation['supervised']
        candidates=[]
        for kind in ('floor','force_match'):
            v=exact_validation[kind]
            passed=(v['free_residual_pct']<=(1-a.min_force_gain)*sup['free_residual_pct'] and
                    v['displacement_pct']<=(1+a.max_disp_degradation)*sup['displacement_pct'] and
                    max_linearization[kind]<=.05)
            candidates.append((kind,passed))
        feasible=[kind for kind,yes in candidates if yes]
        selected=min(feasible,key=lambda k:best_by_kind[k]['val_score']) if feasible else 'supervised'
        # Pilot is a TRAIN/VALIDATION smoke test only: do not open the
        # 12-case internal tuning cohort until the full experiment.
        evaluate={}
        if a.preset=='full':
            tune,basis2,mask2,name2=B.assemble_samples(c,st,region,d['tune'])
            if name2!=baseline_name or not torch.allclose(basis,basis2) or not torch.equal(mask,mask2):
                raise RuntimeError('Tune basis/mask mismatch')
            allmodels={'B5_baseline':B.blank_adapter(mean,basis,Obj),
                       'supervised':bestmodels['supervised'],
                       'floor':bestmodels['floor'],
                       'force_match':bestmodels['force_match']}
            for arm,mod in allmodels.items():
                rows=B.evaluate_exact(tune,mod,mean,std,cs,basis,mask,force,Obj,a.max_correction_pct,
                                      a.floor_margin,region,arm,'internal_tune')
                final.extend(rows)
                evaluate[arm]={k:float(np.mean([r[k] for r in rows])) for k in
                   ('displacement_pct','free_residual_pct','additional_free_force_pct',
                    'reference_interface_force_difference_pct')}
            modes+=modes_on_tune(B,c,H,root,d,tune,allmodels,mean,std,cs,basis,mask,force,a.max_correction_pct,region)
        # Save only selected best-of-each-kind adapter weights, never B5.
        for kind in ('supervised','floor','force_match'):
            rec=best_by_kind[kind]
            ck={'kind':'B9A11C POD adapter (NOT original B5)','region':region,
                'model':baseline_name,'state_dict':{k:v.detach().clone().cpu() for k,v in bestmodels[kind].state_dict().items()},
                'feature_mean':mean.detach().cpu(),'feature_std':std.detach().cpu(),
                'coefficient_std':cs.detach().cpu(),'beta':beta,'source_sha256':source,
                'selection_from':'fit_validation_only','validation_case_ids':fit[vi].tolist(),
                'training_case_ids':fit[ti].tolist(),'selection_record':rec,
                'max_correction_pct':a.max_correction_pct}
            torch.save(ck,out/(region+'_'+kind+'_adapter.pt'))
        summary['results'][region]={
           'baseline_model':baseline_name,'training_cases':fit[ti].tolist(),
           'validation_cases':fit[vi].tolist(),'internal_tune_cases':np.asarray(d['tune'],dtype=int).tolist(),
           'grad_calibration':{k:{'lambda_base':x[0],'median_physics_over_data_gradient':x[1]}
                               for k,x in scaled.items()},
           'selected_from_validation':selected,
           'validation_candidate_gates':{kind:yes for kind,yes in candidates},
           'best_by_kind':best_by_kind,'exact_fit_validation_metrics':exact_validation,
           'max_fit_validation_linearization_relative_error':max_linearization,
           'internal_tune_evaluated':a.preset=='full',
           'internal_tune_metrics_diagnostic_only':evaluate,
           'internal_tune_selected_winner_metrics':evaluate.get(selected),
           'screening_pass_on_fit_validation':bool(selected!='supervised'),
           'approved_for_independent_generalization':False}
        print('B9A11C REGION',region,'exact validation',exact_validation,
              'chosen VAL',selected,'internal tune evaluated?',a.preset=='full',flush=True)
    write_csv(out/'B9A11C_sweep.csv',matrix)
    write_csv(out/'B9A11C_training.csv',curves)
    write_csv(out/'B9A11C_fit_validation.csv',fitval)
    if final:
        write_csv(out/'B9A11C_internal_tune_cases.csv',final)
        write_csv(out/'B9A11C_modes.csv',modes)
    summary['elapsed_seconds']=time.time()-start
    summary['internal_screening_pass']=bool(all(x['screening_pass_on_fit_validation'] for x in summary['results'].values()))
    (out/'B9A11C_summary.json').write_text(json.dumps(summary,indent=2,default=jser))
    make_plots(out,matrix,final,curves,a.region)
    print('B9A11C FINISHED',out,'| original B5 unchanged | no independent validation',flush=True)


def make_plots(out,matrix,final,curves,regionarg):
    regions=sorted(set(r['region'] for r in matrix))
    fig,axes=plt.subplots(1,len(regions),figsize=(6*len(regions),4.6),squeeze=False)
    for j,region in enumerate(regions):
        ax=axes[0,j]
        for kind in ('supervised','floor','force_match'):
            rr=[r for r in matrix if r['region']==region and r['kind']==kind]
            if not rr:continue
            zz=sorted(set(r['multiplier'] for r in rr))
            y=[np.mean([r['val_equilibrium_pct'] for r in rr if r['multiplier']==x]) for x in zz]
            ax.plot(zz,y,'-o',label=kind)
        ax.set_title(region);ax.set_xlabel('Physics weight / calibrated base weight')
        ax.set_ylabel('Fit-validation equilibrium residual (%)');ax.legend(fontsize=8)
    fig.suptitle('B9A11C | Validation sweep (NOT independent)')
    fig.tight_layout();fig.savefig(out/'B9A11C_weight_sweep.png',dpi=140);plt.close(fig)
    if final:
        fig,axes=plt.subplots(1,len(regions),figsize=(6*len(regions),4.6),squeeze=False)
        for j,region in enumerate(regions):
            ax=axes[0,j]
            rr=[r for r in final if r['region']==region]
            labels=('B5_baseline','supervised','floor','force_match')
            y=[np.mean([r['free_residual_pct'] for r in rr if r['arm']==m]) for m in labels]
            ax.bar(np.arange(len(labels)),y)
            ax.set_xticks(range(len(labels)),labels,rotation=22)
            ax.set_title(region);ax.set_ylabel('Internal tune equilibrium residual (%)')
        fig.suptitle('B9A11C | Internal tuning ONLY, no independent generalization')
        fig.tight_layout();fig.savefig(out/'B9A11C_tune_comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,len(regions),figsize=(6*len(regions),4.6),squeeze=False)
    for j,region in enumerate(regions):
        ax=axes[0,j]
        for kind in ('supervised','floor','force_match'):
            rr=[r for r in curves if r['region']==region and r['kind']==kind]
            if not rr:continue
            first=min(rr,key=lambda x:x['val_common_score'])
            group=[r for r in rr if r['seed']==first['seed'] and r['weight_multiplier']==first['weight_multiplier']]
            ax.plot([r['epoch'] for r in group],[r['val_common_score'] for r in group],label=kind)
        ax.set_title(region);ax.set_xlabel('Epoch');ax.set_ylabel('Common fit-validation score');ax.legend(fontsize=8)
    fig.suptitle('B9A11C | Training convergence (best sweep per objective)')
    fig.tight_layout();fig.savefig(out/'B9A11C_convergence.png',dpi=140);plt.close(fig)


def selftest(root):
    torch.set_default_dtype(torch.float32)
    B=load_b(root)
    B.selftest(root)
    A,Obj=B.get_dep(root)
    # synthetic differentiable linearized free force
    torch.manual_seed(12)
    basis=torch.randn(3,10,3,dtype=torch.float64)*0.01
    mask=torch.tensor([True,True]+[False]*8)
    cp=torch.randn(3,dtype=torch.float64,requires_grad=False)
    lift=torch.zeros(10,3,dtype=torch.float64)
    target=lift+torch.einsum('r,rnc->nc',cp+.1,basis)
    J=torch.randn((24,3),dtype=torch.float64)
    f0=torch.randn(24,dtype=torch.float64)
    fr=f0+J@torch.tensor([.03,.02,-.01],dtype=torch.float64)
    s={'coeff0':cp,'lift':lift,'u0':lift+torch.einsum('r,rnc->nc',cp,basis),'ref':target,
       'fr':torch.cat([torch.zeros(2,3,dtype=torch.float64),fr.reshape(8,3)]),
       'f0':torch.cat([torch.zeros(2,3,dtype=torch.float64),f0.reshape(8,3)]),
       'jac':J,'den':torch.tensor(50.,dtype=torch.float64),
       'floor':torch.tensor(.03,dtype=torch.float64)}
    ft=torch.randn(10,dtype=torch.float64)
    net=Obj.PODResidualAdapter(10,3).double()
    scale=torch.ones(3,dtype=torch.float64)
    for kind in ('supervised','floor','force_match'):
        out=objective(s,net,ft,scale,basis,mask,B,Obj,kind,.001,.3,1.25)
        assert all(torch.isfinite(x) for x in out)
        out[0].backward()
        assert all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)
        net.zero_grad(set_to_none=True)
    assert math.isclose(selection_score(.2,.4),.3,rel_tol=1e-12)
    assert selection_score(float('nan'),.1)==float('inf')
    assert not (1.01 <= .90*1.00)
    print('B9A11C SELFTEST PASS | B5 native | two losses | gradients | safe selection')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--stage',choices=('selftest','check','sweep','report'),required=True)
    p.add_argument('--run-tag',default='pilot01')
    p.add_argument('--preset',choices=('pilot','full'),default='pilot')
    p.add_argument('--region',choices=('both','inner_left','inner_right'),default='both')
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--floor-margin',type=float,default=1.25)
    p.add_argument('--target-grad-fraction',type=float,default=.20)
    p.add_argument('--max-correction-pct',type=float,default=.10)
    p.add_argument('--min-force-gain',type=float,default=.10)
    p.add_argument('--max-disp-degradation',type=float,default=.05)
    p.add_argument('--left',default='standardized_grad')
    p.add_argument('--right',default='centered_grad')
    p.add_argument('--all-variants',action='store_false',dest='all_variants',default=False)
    a=p.parse_args()
    if a.epochs is None:a.epochs=15 if a.preset=='pilot' else 75
    if not (1<=a.epochs<=200 and 1<=a.patience<=100):raise ValueError('Invalid epochs/patience')
    if not (0<a.max_correction_pct<=.5 and 1<=a.floor_margin<=3):raise ValueError('Invalid trust/floor')
    if not (0<=a.min_force_gain<1 and 0<=a.max_disp_degradation<=.5):raise ValueError('Invalid gates')
    if not a.run_tag.replace('_','').replace('-','').isalnum():raise ValueError('Bad run tag')
    root=Path(a.project).expanduser().resolve()
    # Required: original B5 builds FP32 before our explicitly FP64 FE/POD tensors.
    torch.set_default_dtype(torch.float32)
    if a.stage=='selftest':selftest(root)
    elif a.stage=='check':
        B=load_b(root);B.verify(root,a);print('B9A11C CHECK PASS')
    elif a.stage=='sweep':run(root,a)
    else:
        target=root/OUTPUT/a.run_tag/'B9A11C_summary.json'
        if not target.exists():raise FileNotFoundError(str(target))
        print(target.read_text())

if __name__=='__main__':main()
