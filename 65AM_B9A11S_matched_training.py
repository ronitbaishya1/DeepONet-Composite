#!/usr/bin/env python3
"""B9A11-S4: matched train/val case IDs and adapter seeds for left/right.
The two ORIGINAL B5 checkpoints remain different, frozen, and unmodified.
This study is a training-control experiment, NOT a mirror-symmetry proof.
"""
import argparse,csv,importlib.util,json,math,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/65AM_B9_coupled_feasibility/B9A11S_symmetry/S4_matched_training')
REG=('inner_left','inner_right')

def imp(p,name):
    sp=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def write_csv(p,rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def load(root):
    cfile=root/'65AM_B9A11C_sweep.py'
    if not cfile.is_file():raise FileNotFoundError('B9A11-C source missing '+str(cfile))
    C=imp(cfile,'b9a11s_C_existing')
    B=C.load_b(root)
    O=imp(root/'65AM_B9A11B_objectives.py','b9a11s_O_existing')
    return C,B,O

def provenance(root,a):
    torch.set_default_dtype(torch.float32)
    C,B,O=load(root)
    verified=B.verify(root,a)
    _,_,c,H,checked,beta,source=verified
    left=set(np.asarray(checked[REG[0]]['d']['fit'],int).tolist())
    right=set(np.asarray(checked[REG[1]]['d']['fit'],int).tolist())
    shared=sorted(left & right)
    if left != right:
        raise RuntimeError('Fit case-ID sets are different between regions! STOP. S4 needs matched IDs, left-only=%s, right-only=%s'%(sorted(left-right),sorted(right-left)))
    if len(shared)<15:raise RuntimeError('Insufficient common fit cases')
    return C,B,O,c,H,checked,beta,source,shared

def rows_summary(rows):
    return {r: {kind: [q for q in rows if q['region']==r and q['kind']==kind] for kind in ('supervised','floor')}
            for r in REG}

def run(root,a):
    t0=time.time();torch.set_num_threads(a.threads)
    C,B,O,c,H,checked,beta,source,shared=provenance(root,a)
    out=root/OUT/a.run_tag
    if out.exists():raise RuntimeError('Existing output protected: '+str(out)+'; choose a new --run-tag')
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(a.split_seed)
    perm=rng.permutation(np.asarray(shared,int))
    val_ids=sorted(perm[:a.n_val].tolist());train_ids=sorted(perm[a.n_val:].tolist())
    if len(set(val_ids)&set(train_ids)):raise AssertionError('Overlapping train/validation cases')
    rows=[];hist=[];summary={'step':'B9A11S4','scope':'original 44 B5 fit cases; NOT independent generalization',
        'identical_case_ids_both_regions':True,'shared_training_case_ids':train_ids,'shared_validation_case_ids':val_ids,
        'baseline_B5_models_still_region_specific':{'inner_left':a.left,'inner_right':a.right},
        'seed_list':a.seeds,'epochs':a.epochs,'physics_weight_multiplier':a.physics_mult,
        'data_policy':'validation selects no hyperparameters in this fixed S4; internal tune never touched',
        'original_B5_modified':False,'coupled_use_approved':False,'beta_frozen':beta,'source':source}
    for r in REG:
        st=checked[r];d=st['d'];force=st['force'];fit=np.asarray(d['fit'],int)
        sample,basis,mask,baseline=B.assemble_samples(c,st,r,fit)
        idx={int(case):i for i,case in enumerate(fit)}
        ti=np.asarray([idx[k] for k in train_ids],int)
        vi=np.asarray([idx[k] for k in val_ids],int)
        mean,std,cs=B.feature_statistics(sample,ti)
        B.make_linearized_tangents(sample,basis,mask,force)
        base_w,grad_ratio=C.grad_weight([sample[i] for i in ti],basis,mask,O,'floor',a.floor_margin,a.target_grad_fraction)
        for seed in a.seeds:
            for kind in ('supervised','floor'):
                lam=0.0 if kind=='supervised' else base_w*a.physics_mult
                curve=[]
                model,ep,score,metrics,baseline_metric=C.fit_adapter(B,O,sample,ti,vi,mean,std,cs,basis,mask,
                            lam,kind,int(seed),a.epochs,a.lr,a.max_correction_pct,a.floor_margin,
                            a.patience,curve,r,0.0 if kind=='supervised' else a.physics_mult)
                hist.extend(curve)
                exact=B.evaluate_exact([sample[int(i)] for i in vi],model,mean,std,cs,basis,mask,force,O,
                              a.max_correction_pct,a.floor_margin,r,kind,'matched_fit_validation')
                valdisp=float(np.mean([v['displacement_pct'] for v in exact]))
                valeq=float(np.mean([v['free_residual_pct'] for v in exact]))
                valforce=float(np.mean([v['additional_free_force_pct'] for v in exact]))
                row=dict(region=r,baseline_model=baseline,kind=kind,seed=int(seed),
                   training_case_ids=';'.join(map(str,train_ids)),validation_case_ids=';'.join(map(str,val_ids)),
                   best_epoch=ep,validation_score=score,lambda_equilibrium=lam,
                   base_gradient_ratio=grad_ratio,exact_val_displacement_pct=valdisp,
                   exact_val_equilibrium_pct=valeq,exact_val_additional_force_pct=valforce)
                rows.append(row)
                ck={'step':'B9A11S4','region':r,'kind':kind,'seed':int(seed),'original_B5_frozen':True,
                    'shared_fit_case_ids':train_ids,'shared_validation_case_ids':val_ids,
                    'state_dict':{k:v.detach().clone().cpu() for k,v in model.state_dict().items()},
                    'feature_mean':mean.cpu(),'feature_std':std.cpu(),'coefficient_std':cs.cpu(),
                    'lambda_equilibrium':lam,'model':baseline,'beta':beta}
                torch.save(ck,out/('S4_%s_%s_seed%d_adapter.pt'%(r,kind,seed)))
                print('S4',r,kind,'seed',seed,'epoch',ep,'exact val equilibrium %.5f%%'%valeq,flush=True)
    write_csv(out/'B9A11S4_matched_validation.csv',rows)
    write_csv(out/'B9A11S4_training.csv',hist)
    gains={}
    for r in REG:
        sub=[x for x in rows if x['region']==r]
        d={}
        for seed in a.seeds:
            s=next(x for x in sub if x['seed']==seed and x['kind']=='supervised')
            p=next(x for x in sub if x['seed']==seed and x['kind']=='floor')
            d[str(seed)]={'equilibrium_relative_change_pct':100*(p['exact_val_equilibrium_pct']/max(s['exact_val_equilibrium_pct'],1e-12)-1),
                          'displacement_relative_change_pct':100*(p['exact_val_displacement_pct']/max(s['exact_val_displacement_pct'],1e-12)-1)}
        gains[r]=d
    summary['paired_seed_changes']=gains
    summary['time_seconds']=time.time()-t0
    (out/'B9A11S4_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    plot(out,rows)
    print('S4 COMPLETE',out,'| all original B5 weights unchanged')

def plot(out,rows):
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for j,r in enumerate(REG):
        x=np.arange(len(sorted(set(z['seed'] for z in rows))))
        seeds=sorted(set(z['seed'] for z in rows))
        for q,kind in enumerate(('supervised','floor')):
            v=[next(z['exact_val_equilibrium_pct'] for z in rows if z['region']==r and z['seed']==s and z['kind']==kind) for s in seeds]
            ax[j].bar(x+(q-.5)*.35,v,width=.34,label=kind)
        ax[j].set_xticks(x,seeds);ax[j].set_xlabel('Shared adapter seed')
        ax[j].set_title(r);ax[j].set_ylabel('Exact nonlinear fit-val equilibrium (%)')
        ax[j].legend()
    fig.suptitle('B9A11S4 | Matched fit/val IDs and seeds; not independent')
    fig.tight_layout();fig.savefig(out/'B9A11S4_matched_validation.png',dpi=180);plt.close(fig)

def selftest():
    ids=list(range(44));rng=np.random.default_rng(57);perm=rng.permutation(ids)
    train=set(perm[9:]);val=set(perm[:9]);assert not train&val and len(train)==35
    assert (np.asarray(ids,dtype=np.int64)[0]==0)
    print('B9A11S4 SELFTEST PASS: shared split, no overlap, serialization scaffold')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--stage',required=True,choices=['selftest','check','run','report'])
    p.add_argument('--run-tag',default='matched01')
    p.add_argument('--split-seed',type=int,default=57)
    p.add_argument('--n-val',type=int,default=9)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=12)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,27])
    p.add_argument('--physics-mult',type=float,default=4.)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--floor-margin',type=float,default=1.25)
    p.add_argument('--target-grad-fraction',type=float,default=.20)
    p.add_argument('--max-correction-pct',type=float,default=.10)
    p.add_argument('--left',default='standardized_grad')
    p.add_argument('--right',default='centered_grad')
    p.add_argument('--all-variants',action='store_false',dest='all_variants',default=False)
    a=p.parse_args();root=a.project.resolve()
    if a.stage=='selftest':selftest();return
    if a.stage=='report':
        q=root/OUT/a.run_tag/'B9A11S4_summary.json'
        if not q.is_file():raise FileNotFoundError(str(q))
        print(q.read_text());return
    C,B,O,c,H,s,beta,source,shared=provenance(root,a)
    print('B9A11S4 CHECK PASS: common 44 fit IDs?',len(shared),'B5 native model forward checks passed')
    if a.stage=='check':return
    if a.n_val<2 or a.n_val>=len(shared)-5:raise ValueError('Bad validation count')
    if a.epochs<1 or a.physics_mult<0:raise ValueError('Bad epochs/physics multiplier')
    run(root,a)

if __name__=='__main__':main()
