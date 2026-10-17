#!/usr/bin/env python3
"""65AG-B: train shared local strain/stress correction (physics ON vs OFF).

Both arms use IDENTICAL architecture and random seed, differing ONLY in
constitutive-consistency loss. New network weights fit only 44 of original
56 train cases and select epoch on the other 12 *original-train* cases.
Previously-used original validation/test remain excluded from NEW 65AG tuning.

 python3 65AG_B_train_joint.py --project . --arm uncoupled --members 1 --epochs 150
 python3 65AG_B_train_joint.py --project . --arm coupled --members 1 --epochs 150
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ag=importlib.import_module('65AG_common')


def objective(m):
    return float(m['S22_pct']+.30*m['LE22_pct']+.04*m['S11_pct']+.04*m['S12_pct'])


def fit_segment(st,reg,member,dev,args,out):
    seed=650712+100*ag.REGIONS.index(reg)+member
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    ds=st['ds'];fit,tune=ag.train_tune_split(ds,reg,member,args.tune_cases)
    fdata=ag.make_dataset(st,fit,dev,args.batch_size)
    tdata=ag.make_dataset(st,tune,dev,args.batch_size)
    ls=ag.f.strain_scales(ds);ss=ag.f.stress_scales(ds)
    lsc=ag.to_tensor(ls,dev);ssc=ag.to_tensor(ss,dev)
    wLE=ag.to_tensor([1.,4.,1.,1.2,.7,.7],dev).view(1,1,6)
    wS=ag.to_tensor([1.,4.,1.,1.,.7,.7],dev).view(1,1,6)
    wC=ag.to_tensor([1.,2.,1.,1.,.7,.7],dev).view(1,1,6)
    net=ag.JointLocalMechanics(args.hidden,args.cap).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-5)
    physics_w=args.physics_weight if args.arm=='coupled' else 0.0
    reference={'tune':tdata,'fit':fdata}
    tbase=ag.report(tdata['LE_pointwise'],tdata['S_pointwise'],tdata['LEref'],tdata['Sref'],tdata['C'])
    torig=ag.report(tdata['HL'],tdata['HS'],tdata['LEref'],tdata['Sref'],tdata['C'])
    best_epoch=0;best_score=objective(tbase);best_metrics=tbase
    best=copy.deepcopy({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    history=[];stale=0
    print('65AG-B',reg,'member',member,'arm',args.arm,'fit/tune',len(fit),len(tune),
          'initial tune S22 %.2f%% LE22 %.2f%%'%(tbase['S22_pct'],tbase['LE22_pct']),flush=True)
    for epoch in range(1,args.epochs+1):
        net.train();perm=np.random.permutation(len(fit));runloss=0.0
        for start in range(0,len(perm),args.batch_size):
            ij=perm[start:start+args.batch_size]
            def tensor(key):return ag.to_tensor(fdata[key][ij],dev)
            F=tensor('F');l0=tensor('LE_pointwise');s0=tensor('S_pointwise')
            lr=tensor('LEref');sr=tensor('Sref');C=tensor('C')
            L,S=net(F,l0,s0,lsc,ssc)
            phys=torch.einsum('bci,bei->bec',C,L)
            lerr=((L-lr)/lsc.view(1,1,6)).square()
            serr=((S-sr)/ssc.view(1,1,6)).square()
            cerr=((S-phys)/ssc.view(1,1,6)).square()
            anchL=((L-l0)/lsc.view(1,1,6)).square().mean()
            anchS=((S-s0)/ssc.view(1,1,6)).square().mean()
            loss=(lerr*wLE).mean()+args.stress_weight*(serr*wS).mean()+physics_w*(cerr*wC).mean()+args.anchor_weight*(anchL+anchS)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step();runloss+=float(loss.item())*len(ij)
        L,S=ag.forward_numpy(net,tdata,ls,ss,dev,args.batch_size)
        met=ag.report(L,S,tdata['LEref'],tdata['Sref'],tdata['C'])
        score=objective(met);safe=ag.safe_gates(met,tbase,torig)
        if safe and score < best_score-1e-7:
            best_score=score;best_metrics=dict(met);best_epoch=epoch
            best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=runloss/len(fit),tune_objective=score,
                            tune_S22_pct=met['S22_pct'],tune_LE22_pct=met['LE22_pct'],
                            tune_S11_pct=met['S11_pct'],tune_S12_pct=met['S12_pct'],
                            tune_safety=bool(safe)))
        if epoch in (1,args.epochs) or epoch%10==0:
            print('    %d S22 %.2f%% LE22 %.2f%% best %d'%(epoch,met['S22_pct'],met['LE22_pct'],best_epoch),flush=True)
        if stale>=args.patience:break
    dest=ag.model_path(args.project_root,args.arm,reg,member)
    dest.parent.mkdir(parents=True,exist_ok=True)
    ck=dict(arch=ag.ARCH,arm=args.arm,region=reg,member=member,hidden=args.hidden,cap=args.cap,
            physics_weight=float(physics_w),stress_weight=float(args.stress_weight),
            anchor_weight=float(args.anchor_weight),
            source_sha=st['source_sha'],af_pointwise_sha=st['af_p_sha'],
            af_lgno_sha=st['af_l_sha'],af_policy_sha=st['af_summary_sha'],
            fitted_cases=list(map(int,fit)),internal_tune_cases=list(map(int,tune)),
            selected_epoch=best_epoch,checkpoint_selected_on='internal subset of ORIGINAL 64J TRAIN only',
            baseline_when_selected_epoch_zero='exact 65AF pointwise strain + train-selected S blend',
            original_64J_validation_not_used_in_new_65AG_training_or_selection=True,
            warning='Inherited 65AF model already used old validation for selection; old validation NOT independent',
            original_64J_test_and_65W_untouched=True,
            physical_strain_integrability_not_enforced=True,
            mechanics_heads_and_force_head_unchanged=True,
            state_dict=best)
    torch.save(ck,dest)
    pd.DataFrame(history).to_csv(dest.parent/'training_history.csv',index=False)
    del net
    if dev.type=='mps':torch.mps.empty_cache()
    return dict(region=reg,member=member,arm=args.arm,fit_cases=len(fit),internal_tune_cases=len(tune),
                selected_epoch=best_epoch,physics_weight=physics_w,
                old_baseline_tune_S22_pct=tbase['S22_pct'],selected_tune_S22_pct=best_metrics['S22_pct'],
                old_baseline_tune_LE22_pct=tbase['LE22_pct'],selected_tune_LE22_pct=best_metrics['LE22_pct'],
                checkpoint=str(dest),source_sha=st['source_sha'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',choices=ag.ARMS,required=True)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--tune-cases',type=int,default=12)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--cap',type=float,default=1.0)
    p.add_argument('--physics-weight',type=float,default=.15)
    p.add_argument('--stress-weight',type=float,default=1.0)
    p.add_argument('--anchor-weight',type=float,default=.02)
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    a.project_root=Path(a.project).expanduser().resolve()
    if not (1<=a.members<=5 and 1<=a.epochs<=4000 and 1<=a.patience<=a.epochs+1 and
            1<=a.batch_size<=56 and 4<=a.tune_cases<=20 and 8<=a.hidden<=512 and
            0<a.lr<.1 and 0<a.cap<=5 and 0<=a.physics_weight<=3 and
            0<a.stress_weight<=5 and 0<=a.anchor_weight<=2):raise ValueError('Invalid training settings')
    outaudit=a.project_root/'results'/ag.ROOT/'A_baseline_audit'/'65AG_A_summary.json'
    if not outaudit.is_file():raise FileNotFoundError('Run 65AG-A first: '+str(outaudit))
    audit=json.loads(outaudit.read_text());assert audit['members']>=a.members
    dev=ag.ac.aa.co.device_from_option(a.device)
    for reg in ag.REGIONS:
        for member in range(1,a.members+1):
            ag.member_sources(a.project_root,reg,member,dev,need_predictions=False)
    if a.check_only:
        print('65AG-B CHECK PASS',a.arm,'device',dev,'source checkpoints + train splits ready');return
    out=ag.safe_output(a.project_root,'B_'+a.arm)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    rows=[]
    for reg in ag.REGIONS:
        for member in range(1,a.members+1):
            st=ag.attach_st(ag.member_sources(a.project_root,reg,member,dev),reg,member)
            rows.append(fit_segment(st,reg,member,dev,a,out))
    pd.DataFrame(rows).to_csv(out/'65AG_B_training_summary.csv',index=False)
    (out/'65AG_B_manifest.json').write_text(json.dumps(dict(step='65AG-B',arm=a.arm,members=a.members,
        constitutive_loss_weight=a.physics_weight if a.arm=='coupled' else 0.,
        new_65AG_train_selection='original 64J TRAIN only, deterministic fit/tune split',
        original_validation_inherited_from_upstream=True, original_test_and_65W_unused=True,
        models_jointly_predict='6 material log strains + 6 Cauchy stress approximations',
        note='Both arms use exactly identical architecture and zero-initialized residual output heads'),indent=2))
    print('65AG-B DONE:',out)

if __name__=='__main__':main()
