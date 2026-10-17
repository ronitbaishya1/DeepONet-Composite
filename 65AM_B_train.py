#!/usr/bin/env python3
"""65AM-B1/B2: matched FE-basis coefficient operator.

B1: supervised normalized nodal U only.
B2: same network/data + supervised C3D8R gradient, dU2/dY, shear-12.
These are mechanics-informed SUPERVISED objectives; not PDE equilibrium losses.

 python3 65AM_B_train.py --project . --arm B1 --check-only
 python3 65AM_B_train.py --project . --arm B1
 python3 65AM_B_train.py --project . --arm B2 --check-only
 python3 65AM_B_train.py --project . --arm B2
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import importlib
b=importlib.import_module('65AM_B_common')


def array_pack(data,ids,dev):
    concat=np.r_[data['fit'],data['tune']]
    lookup={int(v):i for i,v in enumerate(concat)}
    rows=np.array([lookup[int(v)] for v in ids])
    x=(data['ds']['branch_inputs'][ids]-data['branch_mean'])/data['branch_std']
    return dict(x=b.np_tens(x,dev),lift=b.np_tens(data['lift'][rows],dev),
                true=b.np_tens(data['ds']['U'][ids],dev))


def objective(model,pack,basis,con,dN,sU,sG,arm,lambdas):
    coeff=model(pack['x'])
    U=pack['lift']+torch.einsum('br,rnc->bnc',coeff,basis)
    # Channel normalization and elements all use only fit-derived scales.
    LU=torch.mean(((U-pack['true'])/sU)**2)
    GP=b.AA.gradient_torch(U,con,dN)
    GT=b.AA.gradient_torch(pack['true'],con,dN)
    normalized=(GP-GT)/sG
    LG=torch.mean(normalized**2)
    L22=torch.mean(normalized[...,1,1]**2)
    shear_scale=torch.sqrt(sG[0,0,0,1]**2+sG[0,0,1,0]**2)
    L12=torch.mean(((GP[...,0,1]+GP[...,1,0]-GT[...,0,1]-GT[...,1,0])/shear_scale)**2)
    full=LU if arm=='B1' else LU+lambdas[0]*LG+lambdas[1]*L22+lambdas[2]*L12
    return full, (LU,LG,L22,L12)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--arm',required=True,choices=['B1','B2'])
    ap.add_argument('--epochs',type=int,default=500)
    ap.add_argument('--patience',type=int,default=100)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args(); root=Path(a.project).expanduser().resolve()
    if not 1<=a.epochs<=4000 or not 1<=a.patience<=a.epochs or not 16<=a.hidden<=256 or not 0<a.lr<=0.02:
        raise ValueError('Invalid hyperparameter')
    device=b.CO.device_from_option(a.device)
    loaded={}
    for region in b.REGIONS:
        item=b.load_saved(root,region)
        # Required reference quality audit matches Abaqus LE component conventions.
        quality=b.check_reference(item['ds'],np.r_[item['fit'],item['tune']])
        loaded[region]=item
        print('CHECK',a.arm,region,'rank',item['rank'],'fit',len(item['fit']),
              'tune',len(item['tune']),'LE reference',quality,'device',device,flush=True)
    if a.check_only:
        print('65AM-'+a.arm+' CHECK ONLY PASS; no results created')
        return
    out=b.output_dir(root,a.arm+'_train')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    weights=(0.20,0.10,0.10) # predefined, fixed before any tuning; compare B1 against B2
    rows=[]
    for ri,region in enumerate(b.REGIONS):
        np.random.seed(b.SEED+100*ri);random.seed(b.SEED+100*ri);torch.manual_seed(b.SEED+100*ri)
        data=loaded[region];ds=data['ds']
        fitted=array_pack(data,data['fit'],device)
        tuned=array_pack(data,data['tune'],device)
        basis=b.np_tens(data['basis'],device)
        con=torch.as_tensor(np.array(ds['element_node_indices'],copy=True),dtype=torch.long,device=device)
        dN=b.np_tens(b.AA.element_dN_numpy(ds),device)
        sU=b.np_tens(data['sU'],device).reshape(1,1,3)
        sG=b.np_tens(data['sG'],device).reshape(1,1,3,3)
        model=b.ResidualCoeffNet(int(ds['branch_inputs'].shape[1]),data['rank'],a.hidden).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
        best=float('inf');best_state=None;best_epoch=None;stale=0;history=[]
        for epoch in range(1,a.epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            L,_=objective(model,fitted,basis,con,dN,sU,sG,a.arm,weights)
            if not torch.isfinite(L):raise RuntimeError('Nonfinite fit loss')
            L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            model.eval()
            with torch.no_grad():
                devloss,ls=objective(model,tuned,basis,con,dN,sU,sG,a.arm,weights)
                dv=float(devloss.item())
                record={'epoch':epoch,'fit_loss':float(L.item()),'tune_objective':dv,
                        'tune_LU':float(ls[0].item()),'tune_LG':float(ls[1].item()),
                        'tune_L22':float(ls[2].item()),'tune_L12':float(ls[3].item())}
            history.append(record)
            if dv<best-1e-6:
                best=dv;best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if epoch==1 or epoch%25==0 or epoch==a.epochs:
                print(a.arm,region,'epoch',epoch,'fit %.5g tune %.5g best %s'%(L.item(),dv,best_epoch),flush=True)
            if stale>=a.patience:
                print(a.arm,region,'early stopping at epoch',epoch,flush=True);break
        if best_state is None:raise RuntimeError('No best checkpoint')
        folder=out/region;folder.mkdir(parents=True,exist_ok=True)
        payload=dict(step='65AM-'+a.arm,arm=a.arm,region=region,basis_rank=data['rank'],
                     hidden=a.hidden,source_sha=data['source_sha'],seed=b.SEED+100*ri,
                     fit_ids=data['fit'].tolist(),tune_ids=data['tune'].tolist(),
                     selected_epoch=best_epoch,weights=weights,model_selection='internal 12 train-subset cases',
                     optimizer='AdamW',lr=a.lr,normalization='FIT only',
                     scope='frozen 65V/65AI, no external or original 64J validation/test',
                     best_tune_objective=best,state_dict=best_state)
        torch.save(payload,folder/'best_model.pt')
        pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
        model.load_state_dict(best_state)
        rows.append(dict(region=region,arm=a.arm,basis_rank=data['rank'],selected_epoch=best_epoch,
                         best_tune_objective=best,parameters=sum(p.numel() for p in model.parameters()),
                         fit_cases=len(data['fit']),tune_cases=len(data['tune']),
                         checkpoint_sha256=b.sha(folder/'best_model.pt')))
    pd.DataFrame(rows).to_csv(out/'training_summary.csv',index=False)
    print('65AM-'+a.arm+' COMPLETE:',out)

if __name__=='__main__':main()
