#!/usr/bin/env python3
"""Train one B5 arm (both regions) using 44 original-TRAIN fitting cases.

    python3 65AM_B5_train.py --project . --variant centered --loss U --check-only
    python3 65AM_B5_train.py --project . --variant centered --loss U

Repeat with (centered,grad), (standardized,U), (standardized,grad).
The neural network and supervised loss match original B1/B2; ONLY coefficient
output centering/standardization is different. No other physics losses added.
"""
import argparse,random,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import importlib
c=importlib.import_module('65AM_B5_common')
b=c.b
oldtrain=importlib.import_module('65AM_B_train')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--variant',required=True,choices=c.VARIANTS)
    p.add_argument('--loss',required=True,choices=c.LOSSES)
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--patience',type=int,default=100)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--device',default='auto',choices=('auto','cpu','mps'))
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.epochs<=4000 or not 1<=a.patience<=a.epochs or not 16<=a.hidden<=256 or not 0<a.lr<=0.02:
        raise ValueError('Unsupported training hyperparameter')
    dev=b.CO.device_from_option(a.device)
    loaded={}
    for region in b.REGIONS:
        d,mu,sd=c.prepare(root,region)
        b.check_reference(d['ds'],np.r_[d['fit'],d['tune']])
        loaded[region]=(d,mu,sd)
        print('CHECK PASS',region,'rank',d['rank'],'fit',len(d['fit']),
              'tune',len(d['tune']),'mean0',round(float(mu[0]),4),
              'SD range',(round(float(sd.min()),4),round(float(sd.max()),4)),flush=True)
    if a.check_only:
        print('65AM-B5 CHECK ONLY: no results created')
        return
    out=c.guarded_directory(c.paths(root,a.variant,a.loss))
    torch.set_num_threads(min(4,torch.get_num_threads()))
    arm='B1' if a.loss=='U' else 'B2'
    weights=(0.20,0.10,0.10)  # identical to previous B2 experiment
    summaries=[]
    for ri,region in enumerate(b.REGIONS):
        seed=b.SEED+100*ri
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        d,mu,sd=loaded[region];ds=d['ds']
        fitpack=oldtrain.array_pack(d,d['fit'],dev)
        tunepack=oldtrain.array_pack(d,d['tune'],dev)
        basis=b.np_tens(d['basis'],dev)
        conn=torch.as_tensor(np.array(ds['element_node_indices'],copy=True),dtype=torch.long,device=dev)
        dN=b.np_tens(b.AA.element_dN_numpy(ds),dev)
        sU=b.np_tens(d['sU'],dev).reshape(1,1,3)
        sG=b.np_tens(d['sG'],dev).reshape(1,1,3,3)
        model=c.build_model(d,mu,sd,a.variant,a.hidden,dev)
        initial=model(fitpack['x']).detach().cpu().numpy()
        if not np.allclose(initial,mu[None,:],rtol=0,atol=2e-5):
            raise RuntimeError('B5 zero-init must predict coefficient FIT mean')
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
        best=float('inf');best_epoch=None;best_weights=None;stale=0;history=[]
        for epoch in range(1,a.epochs+1):
            model.train();opt.zero_grad(set_to_none=True)
            loss,_=oldtrain.objective(model,fitpack,basis,conn,dN,sU,sG,arm,weights)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            model.eval()
            with torch.no_grad():
                tuneloss,parts=oldtrain.objective(model,tunepack,basis,conn,dN,sU,sG,arm,weights)
                scalar=float(tuneloss.item())
                vals=[float(v.item()) for v in parts]
            if not np.isfinite(scalar):raise RuntimeError('Nonfinite tune loss')
            history.append(dict(epoch=epoch,fit_loss=float(loss.item()),tune_objective=scalar,
                                tune_LU=vals[0],tune_LG=vals[1],tune_L22=vals[2],tune_L12=vals[3]))
            if scalar<best-1e-6:
                best=scalar;best_epoch=epoch;stale=0
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if epoch==1 or epoch%25==0 or epoch==a.epochs:
                print('B5',a.variant,a.loss,region,'epoch',epoch,'fit',round(float(loss.item()),6),
                      'tune',round(scalar,6),'best',best_epoch,flush=True)
            if stale>=a.patience:
                print('B5 early stopping',region,epoch,flush=True)
                break
        if best_weights is None: raise RuntimeError('No best checkpoint')
        folder=out/region;folder.mkdir(parents=True,exist_ok=True)
        payload=dict(step='65AM-B5',variant=a.variant,loss=a.loss,region=region,rank=d['rank'],
                     hidden=a.hidden,baseline_sha=d['source_sha'],
                     basis_sha=b.sha(b.saved_basis_path(root,region)),
                     fit_ids=d['fit'].tolist(),tune_ids=d['tune'].tolist(),
                     coeff_mean=mu,coeff_std=sd,selected_epoch=best_epoch,
                     best_tune_objective=best,training_seed=seed,
                     weights=weights,optimizer='AdamW',lr=a.lr,
                     note='B5 original TRAIN 44 fit/12 tune only; 65V and frozen 65AI unchanged',
                     state_dict=best_weights)
        torch.save(payload,folder/'best_model.pt')
        pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
        summaries.append(dict(region=region,variant=a.variant,loss=a.loss,rank=d['rank'],
                              selected_epoch=best_epoch,best_tune_objective=best,
                              parameter_count=sum(v.numel() for v in model.parameters() if v.requires_grad),
                              fit_cases=len(d['fit']),tune_cases=len(d['tune']),
                              checkpoint_sha256=b.sha(folder/'best_model.pt')))
    pd.DataFrame(summaries).to_csv(out/'training_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(step='65AM-B5',variant=a.variant,loss=a.loss,
        original_train_only=True,rank=8,fit_cases=44,tune_cases=12,
        source_65V_unchanged=True,old_64J_validation_test_not_used=True,
        external_AJ_cases_not_used=True,
        selection_is_internal_development_not_independent=True),indent=2))
    print('65AM-B5 TRAIN COMPLETE',out,flush=True)

if __name__=='__main__':main()
