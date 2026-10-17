#!/usr/bin/env python3
"""65AI-B: Train displacement-conditioned joint LE/S correction (one member each).

Control:      python3 65AI_B_train_local.py --project . --arm pointwise --members 1 --epochs 200
Local stencil: python3 65AI_B_train_local.py --project . --arm local_lgno --members 1 --epochs 200

No original 64J validation or test cases used for NEW weight selection;
internal holdout drawn from 56 original 64J train cases. *Upstream frozen*
65AG/65AF networks have already seen training cases and historical validation
indirectly; internal tune is NOT an independent generalization test.
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ai=importlib.import_module('65AI_common')


def subdata(data,idx):
    n=len(data['ids'])
    return {k:(v[idx] if isinstance(v,np.ndarray) and len(v)==n else v)
            for k,v in data.items()}


def train_one(st,region,member,device,a):
    seed=651010+100*ai.REGIONS.index(region)+member
    np.random.seed(seed);torch.manual_seed(seed);random.seed(seed)
    ds=st['ds'];train=np.asarray(ds['splits']['train'],np.int64)
    fit,tune=ai.internal_split(ds,region,member,a.tune_cases)
    data=ai.base_features(st,train,device,a.batch_size)
    mapper={int(n):i for i,n in enumerate(train)}
    fit_rows=np.array([mapper[int(i)] for i in fit],dtype=int)
    tune_rows=np.array([mapper[int(i)] for i in tune],dtype=int)
    tuned=subdata(data,tune_rows)
    ls,ss=ai.train_scales(ds)
    ls_t=ai.tensor(ls,device).view(1,1,6)
    ss_t=ai.tensor(ss,device).view(1,1,6)
    wL=ai.tensor([1.,4.,1.,1.2,.8,.8],device).view(1,1,6)
    wS=ai.tensor([1.,5.,1.,1.2,.8,.8],device).view(1,1,6)
    net=ai.DisplacementToLocalMechanics(data['features'].shape[-1],a.arm,a.hidden,a.cap).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
    base=ai.error_metrics(tuned['L0'],tuned['S0'],tuned)
    best_metrics=dict(base);best_score=ai.objective(base);best_epoch=0
    best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    stale=0;history=[]
    print('65AI-B',a.arm,region,'member',member,'fit',len(fit),'tune',len(tune),
          'baseline S22 %.3f%% LE22 %.3f%%'%(base['S22_pct'],base['LE22_pct']),flush=True)
    for epoch in range(1,a.epochs+1):
        net.train();p=np.random.permutation(fit_rows);epoch_loss=0.;seen=0
        for k in range(0,len(p),a.batch_size):
            ii=p[k:k+a.batch_size]
            x=ai.tensor(data['features'][ii],device)
            du=ai.tensor(data['du'][ii],device)
            le0=ai.tensor(data['L0'][ii],device);s0=ai.tensor(data['S0'][ii],device)
            le=ai.tensor(data['Lref'][ii],device);s=ai.tensor(data['Sref'][ii],device)
            L,S=net(x,du,le0,s0,ls_t.view(-1),ss_t.view(-1))
            lossL=(((L-le)/ls_t)**2*wL).mean()
            lossS=(((S-s)/ss_t)**2*wS).mean()
            anchor=(((L-le0)/ls_t)**2).mean()+(((S-s0)/ss_t)**2).mean()
            loss=lossL+a.stress_weight*lossS+a.anchor_weight*anchor
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss at epoch '+str(epoch))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
            epoch_loss+=loss.item()*len(ii);seen+=len(ii)
        pl,ps=ai.predict(net,tuned,ls,ss,device,a.batch_size)
        met=ai.error_metrics(pl,ps,tuned)
        score=ai.objective(met)
        safe=ai.safety(met,base)
        if safe and score<best_score-1e-6:
            best_score=score;best_metrics=dict(met);best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=epoch_loss/max(1,seen),
                            internal_tune_objective=score,internal_tune_S22_pct=met['S22_pct'],
                            internal_tune_LE22_pct=met['LE22_pct'],
                            internal_tune_S11_pct=met['S11_pct'],internal_tune_S12_pct=met['S12_pct'],
                            safety=bool(safe),best_epoch=best_epoch))
        if epoch==1 or epoch%10==0 or epoch==a.epochs:
            print('   epoch',epoch,'S22 %.3f%% LE22 %.3f%% best %d'%(met['S22_pct'],met['LE22_pct'],best_epoch),flush=True)
        if stale>=a.patience:
            print('   stopped by internal-tune patience',a.patience,flush=True);break
    pt=ai.ck_path(a.root,a.arm,region,member)
    pt.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(architecture=ai.ARCH,arm=a.arm,region=region,member=member,
                 input_dim=data['features'].shape[-1],hidden=a.hidden,cap=a.cap,
                 source_sha=st['source_sha'],prior_sha=st['prior_sha'],
                 pretrained_65AG_from_repeated_historical_validation=True,
                 checkpoint_selection='internal split of original 64J TRAIN only',
                 fitted_case_ids=fit.tolist(),tune_case_ids=tune.tolist(),
                 old_validation_test_65W_not_used=True,
                 frozen_displacement_and_force_head=True,
                 output_is_not_kinematically_compatible=True,
                 selected_epoch=best_epoch,selected_tune_metrics=best_metrics,
                 weights_differ_only_in_new_local_mechanics_adapter=True,
                 state_dict=best_state)
    torch.save(payload,pt)
    pd.DataFrame(history).to_csv(pt.parent/'training_history.csv',index=False)
    del net
    if device.type=='mps':torch.mps.empty_cache()
    return dict(region=region,member=member,arm=a.arm,fit_cases=len(fit),
                internal_tune_cases=len(tune),selected_epoch=best_epoch,
                reference_65AG_tune_S22_pct=base['S22_pct'],
                selected_tune_S22_pct=best_metrics['S22_pct'],
                reference_65AG_tune_LE22_pct=base['LE22_pct'],
                selected_tune_LE22_pct=best_metrics['LE22_pct'],
                reference_65AG_tune_S11_pct=base['S11_pct'],
                selected_tune_S11_pct=best_metrics['S11_pct'],
                adapter_parameters=sum(int(p.numel()) for p in ai.DisplacementToLocalMechanics(
                    data['features'].shape[-1],a.arm,a.hidden,a.cap).parameters()),
                prior_sha256=st['prior_sha'],model_checkpoint=str(pt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',choices=ai.ARMS,required=True)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--tune-cases',type=int,default=12)
    p.add_argument('--lr',type=float,default=4e-4)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--cap',type=float,default=.8)
    p.add_argument('--stress-weight',type=float,default=1.0)
    p.add_argument('--anchor-weight',type=float,default=.04)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();a.root=Path(a.project).expanduser().resolve()
    if not (1<=a.members<=5 and 1<=a.epochs<=3000 and 1<=a.patience<=a.epochs+1
            and 1<=a.batch_size<=16 and 6<=a.tune_cases<=18 and 16<=a.hidden<=256
            and 0<a.lr<=.02 and 0<a.cap<=3 and 0<a.stress_weight<=4
            and 0<=a.anchor_weight<=2):raise ValueError('Invalid training hyperparameter')
    req=a.root/'results'/ai.ROOT/'A_audit'/'65AI_A_summary.json'
    if not req.is_file():raise FileNotFoundError('Run Step 65AI-A first: '+str(req))
    if int(json.loads(req.read_text())['members'])<a.members:
        raise RuntimeError('Insufficient audited members')
    dev=ai.co.device_from_option(a.device)
    for reg in ai.REGIONS:
        for m in range(1,a.members+1):
            ai.member_source(a.root,reg,m,dev,False)
    if a.check_only:
        print('65AI-B CHECK PASS: arm',a.arm,'device',dev,'prerequisite models present');return
    out=ai.no_overwrite(a.root,'B_'+a.arm)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rows=[]
    for reg in ai.REGIONS:
        for m in range(1,a.members+1):
            st=ai.member_source(a.root,reg,m,dev,True)
            rows.append(train_one(st,reg,m,dev,a))
    pd.DataFrame(rows).to_csv(out/'65AI_B_training_summary.csv',index=False)
    (out/'65AI_B_manifest.json').write_text(json.dumps(dict(step='65AI-B',arm=a.arm,members=a.members,
        target='local material log strain LE(6) + stress S(6)',
        features='65AA predicted hard-compatible U and C3D8R neighbors, material & interface PCA, frozen LGNO derivatives',
        pretrain='65AG uncoupled learned strain/stress',
        old_test_and_65W_excluded=True,
        internal_train_tune_not_independent_due_to_pretraining=True,
        original_source_U_and_interface_force_unchanged=True,
        local_arm_has_more_parameters=True),indent=2))
    print('65AI-B DONE:',out)

if __name__=='__main__':main()
