#!/usr/bin/env python3
"""65AM-B4: READ-ONLY diagnostics for existing B0/B1/B2 models.

Answers: Which POD coefficients fail to generalize, and which modes most affect
FE displacement gradients? Uses original 44 fit and 12 tune ONLY. No retraining.

Run:
  python3 65AM_B4_coefficient_audit.py --project . --check-only
  python3 65AM_B4_coefficient_audit.py --project .

Writes new diagnostic files ONLY in results/65AM_B_FE_residual_operator/
B4_coefficient_audit; refuses to overwrite existing outputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

b = importlib.import_module('65AM_B_common')
ARMS = ('B1','B2')


def coefficient_oracle(data, ids, basis_path):
    """Least-squares oracle projection using reference U, for DIAGNOSIS only."""
    with np.load(basis_path, allow_pickle=False) as saved:
        sR = np.asarray(saved['sR'], np.float64)
    basis = np.asarray(data['basis'], np.float64)
    free = np.setdiff1d(np.arange(basis.shape[1]), data['fixed'])
    # Fit-case orthogonality held in a weighted displacement norm.
    modes = (basis[:,free,:]/sR[None,None,:]).reshape(data['rank'],-1)
    gram = modes @ modes.T
    gram_err = float(np.max(np.abs(gram - np.eye(data['rank']))))
    cond = float(np.linalg.cond(gram))
    if not np.all(np.isfinite(gram)) or cond > 1e6:
        raise RuntimeError('Ill-conditioned retained POD basis; stop and inspect B0')
    lookup = {int(v):k for k,v in enumerate(np.r_[data['fit'],data['tune']])}
    indices = np.array([lookup[int(v)] for v in ids], np.int64)
    L = np.asarray(data['lift'][indices], np.float64)
    gt = np.asarray(data['ds']['U'][ids], np.float64)
    residual = ((gt-L)[:,free,:]/sR).reshape(len(ids),-1)
    oracle = np.linalg.solve(gram,(residual@modes.T).T).T
    reconstruction = L+np.einsum('kr,rnc->knc',oracle,basis,optimize=True)
    return oracle, reconstruction, L, gt, gram_err, cond


def predict_coefficients(net,data,ids,dev):
    x=(np.asarray(data['ds']['branch_inputs'][ids],np.float64)-data['branch_mean'])/data['branch_std']
    result=[]
    net.eval()
    with torch.no_grad():
        for start in range(0,len(ids),16):
            result.append(net(b.np_tens(x[start:start+16],dev)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def case_error_rows(region,arm,split,ids,gt,baseline,pred,ds):
    rows=[]
    for k,c in enumerate(ids):
        ref = ds['LE'][int(c):int(c)+1]
        result=b.metrics(pred[k:k+1],gt[k:k+1],ds,ref)
        result_oracle=b.metrics(baseline[k:k+1],gt[k:k+1],ds,ref)
        rows.append(dict(region=region,arm=arm,split=split,case_index=int(c),
                         **result,oracle_grad_rel_pct=result_oracle['grad_rel_pct'],
                         oracle_U_vec_rel_pct=result_oracle['U_vec_rel_pct']))
    return rows


def get_history(root,region,arm):
    hist=root/'results'/b.ROOT_NAME/(arm+'_train')/region/'training_history.csv'
    if not hist.is_file():raise FileNotFoundError('Missing history: '+str(hist))
    t=pd.read_csv(hist)
    required=['epoch','fit_loss','tune_objective','tune_LU','tune_LG','tune_L22','tune_L12']
    if any(c not in t for c in required): raise RuntimeError('Missing training history columns')
    best=t.loc[t.tune_objective.idxmin()]
    last=t.iloc[-1]
    return dict(region=region,arm=arm,min_tune_epoch=int(best.epoch),last_epoch=int(last.epoch),
       best_fit_loss=float(best.fit_loss),last_fit_loss=float(last.fit_loss),
       best_tune_loss=float(best.tune_objective),last_tune_loss=float(last.tune_objective),
       best_tune_LU=float(best.tune_LU),last_tune_LU=float(last.tune_LU),
       best_tune_LG=float(best.tune_LG),last_tune_LG=float(last.tune_LG),
       best_tune_L22=float(best.tune_L22),last_tune_L22=float(last.tune_L22))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='.')
    parser.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    root=Path(args.project).expanduser().resolve()
    device=b.CO.device_from_option(args.device)
    loaded={}
    for region in b.REGIONS:
        data=b.load_saved(root,region)
        basis_file=b.saved_basis_path(root,region)
        if not basis_file.is_file():raise FileNotFoundError(str(basis_file))
        for arm in ARMS:
            if not b.candidate_path(root,arm,region).is_file():
                raise FileNotFoundError(str(b.candidate_path(root,arm,region)))
            get_history(root,region,arm)
        loaded[region]=data
        print('CHECK PASS:',region,'rank',data['rank'],'fit',len(data['fit']),'tune',len(data['tune']))
    if args.check_only:
        print('65AM-B4 CHECK ONLY PASS; no files created')
        return
    out=b.output_dir(root,'B4_coefficient_audit')
    mode_rows, case_rows, split_rows, hist_rows = [], [], [], []
    all_summary={}
    for region,data in loaded.items():
        ds=data['ds']; rank=data['rank'];basis=np.asarray(data['basis'],np.float64)
        Gm=b.AA.gradient_numpy(basis,ds).astype(np.float64)  # [rank, E, 3, 3]
        grad_sensitivity=np.sqrt(np.mean(Gm**2,axis=(1,2,3)))
        d22_sensitivity=np.sqrt(np.mean(Gm[:,:,1,1]**2,axis=1))
        fit_oracle, fit_rec, fit_lift, fit_ref,fit_orth,fit_gramcond=coefficient_oracle(
                 data,data['fit'],b.saved_basis_path(root,region))
        oracle_std=np.maximum(np.std(fit_oracle,axis=0),1e-8)
        models={arm:b.load_candidate(root,arm,region,data,device)[0] for arm in ARMS}
        region_summary={}
        plot_by_arm={arm:[] for arm in ARMS}
        for split,ids in [('fit',data['fit']),('tune',data['tune'])]:
            truth,oracleU,lift,Uref,_,_=coefficient_oracle(data,ids,b.saved_basis_path(root,region))
            fit_gt_gradient=b.AA.gradient_numpy(Uref,ds)
            oracle_metrics=b.metrics(oracleU,Uref,ds,ds['LE'][ids])
            for arm in ARMS:
                predicted=predict_coefficients(models[arm],data,ids,device)
                if predicted.shape!=truth.shape: raise RuntimeError('Coefficient prediction shape mismatch')
                Uout=lift+np.einsum('kr,rnc->knc',predicted,basis,optimize=True)
                if np.max(abs(Uout[:,data['fixed'],:]-lift[:,data['fixed'],:]))>1e-7:
                    raise RuntimeError('Interface displacement changed')
                met=b.metrics(Uout,Uref,ds,ds['LE'][ids])
                case_rows.extend(case_error_rows(region,arm,split,ids,Uref,oracleU, Uout, ds))
                split_rows.append(dict(region=region,arm=arm,split=split,rank=rank,
                                       oracle_grad_rel_pct=oracle_metrics['grad_rel_pct'],
                                       oracle_U_vec_rel_pct=oracle_metrics['U_vec_rel_pct'],**met))
                # Compare against existing B3 tune report, if available.
                if split=='tune':
                    p=root/'results'/b.ROOT_NAME/'B3_comparison'/'B3_pooled_metrics.csv'
                    if p.is_file():
                        previous=pd.read_csv(p)
                        expected=previous[(previous.region==region)&(previous.model==(arm+'_FE_residual'))]
                        if len(expected)!=1:raise RuntimeError('B3 candidate row missing '+region+arm)
                        if abs(float(expected.iloc[0].grad_rel_pct)-met['grad_rel_pct'])>0.05:
                            raise RuntimeError('B4 gradient does not reproduce B3: '+region+arm)
                Gpred=b.AA.gradient_numpy(Uout,ds).astype(np.float64)
                E=Gpred-fit_gt_gradient
                ss=float(np.sum(E**2))
                if ss<=0:raise RuntimeError('Unexpected zero gradient error')
                for m in range(rank):
                    residual=predicted[:,m]-truth[:,m]
                    pred_sd=float(np.std(predicted[:,m]))
                    truth_sd=float(np.std(truth[:,m]))
                    rmse=float(np.sqrt(np.mean(residual**2)))
                    corr=float(np.corrcoef(truth[:,m],predicted[:,m])[0,1]) if truth_sd>1e-9 and pred_sd>1e-9 else np.nan
                    r2=1-float(np.sum(residual**2))/max(float(np.sum((truth[:,m]-np.mean(truth[:,m]))**2)),1e-12)
                    # Oracle exchange of exactly this mode, holding every other coefficient frozen.
                    delta=truth[:,m]-predicted[:,m]
                    swape=E+delta[:,None,None,None]*Gm[m][None,:,:,:]
                    improved=100*(ss-float(np.sum(swape**2)))/ss
                    mode_rows.append(dict(region=region,arm=arm,split=split,mode=m+1,
                         oracle_fit_std=oracle_std[m],oracle_mean=float(np.mean(truth[:,m])),
                         pred_mean=float(np.mean(predicted[:,m])),coeff_RMSE=rmse,
                         coeff_RMSE_over_fit_std=rmse/oracle_std[m],coeff_R2=r2,
                         coeff_correlation=corr,gradient_basis_RMS=grad_sensitivity[m],
                         dU2dY_basis_RMS=d22_sensitivity[m],
                         gradient_error_reduction_if_mode_oracle_pct=improved))
                    if split=='tune': plot_by_arm[arm].append(rmse/oracle_std[m])
                region_summary[arm+'_'+split]=dict(grad_pct=met['grad_rel_pct'],
                    U_pct=met['U_vec_rel_pct'],LE22_pct=met['LE22_rel_pct'])
        for arm in ARMS: hist_rows.append(get_history(root,region,arm))
        plt.figure(figsize=(8,4.7))
        for arm in ARMS:
            plt.plot(np.arange(1,rank+1),plot_by_arm[arm],marker='o',label=arm)
        plt.xlabel('FE/POD coefficient mode')
        plt.ylabel('TUNE coefficient RMSE / FIT coefficient standard deviation')
        plt.title(region+' | internal-development coefficient error')
        plt.legend();plt.tight_layout()
        plt.savefig(out/(region+'_coeff_error_by_mode.png'),dpi=175)
        plt.close()
        region_summary.update(basis_rank=rank,basis_gram_error=fit_orth,basis_gram_condition=fit_gramcond,
                             fit_cases=len(data['fit']),tune_cases=len(data['tune']))
        all_summary[region]=region_summary
        print(region, 'B1/B2 fit-tune analyzed; basis Gram condition',round(fit_gramcond,3))
    pd.DataFrame(mode_rows).to_csv(out/'B4_mode_errors.csv',index=False)
    pd.DataFrame(case_rows).to_csv(out/'B4_case_errors.csv',index=False)
    pd.DataFrame(split_rows).to_csv(out/'B4_split_metrics.csv',index=False)
    pd.DataFrame(hist_rows).to_csv(out/'B4_training_gaps.csv',index=False)
    (out/'B4_summary.json').write_text(json.dumps(dict(step='65AM-B4',
       type='diagnostics only; no retraining or deployment',
       data='original 64J train only (44 fit + 12 internal development); no AJ external',
       source_65V_and_existing_checkpoints_unchanged=True,
       note='Oracle POD coefficients use reference FEM displacement only for diagnostics. Per-mode oracle exchange is not an inference method. Negative mode-exchange gain may reflect cross-mode cancellation.',
       regions=all_summary),indent=2))
    print('65AM-B4 COMPLETE:',out)
    print('Send B4_mode_errors.csv, B4_split_metrics.csv, B4_case_errors.csv, B4_training_gaps.csv, B4_summary.json and both coefficient plots.')

if __name__=='__main__':main()
