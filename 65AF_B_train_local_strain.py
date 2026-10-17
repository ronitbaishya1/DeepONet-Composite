#!/usr/bin/env python3
"""65AF-B: train pointwise and LGNO-inspired local strain refiners.

Train on the original 64J TRAIN cases only. Choose checkpoint using original
VALIDATION cases; do not use original TEST or 65W middle full FEM.
The V4 displacement, strain, stress and force heads remain frozen.

python3 65AF_B_train_local_strain.py --project . --arm pointwise --members 1 --epochs 100
python3 65AF_B_train_local_strain.py --project . --arm lgno --members 1 --epochs 100
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
f=importlib.import_module('65AF_common')


def new_metric(L,S,ds,ids):
    le=ds['LE'][ids];s=ds['S'][ids]
    return dict(LE11=f.relative(L[...,0],le[...,0]),
        LE22=f.relative(L[...,1],le[...,1]),
        LE12=f.relative(L[...,3],le[...,3]),
        S11=f.relative(S[...,0],s[...,0]),
        S22=f.relative(S[...,1],s[...,1]),
        S12=f.relative(S[...,3],s[...,3]))


def safe(x,base):
    return (x['LE11'] <= base['LE11']*1.10+.30 and
            x['LE12'] <= base['LE12']*1.10+.30 and
            x['S11'] <= base['S11']*1.12+.50 and
            x['S12'] <= base['S12']*1.12+.50 and
            x['LE22'] <= base['LE22']*1.02+.30)


def validate(net,features,du,lehead,le_sc,C,ds,ids,dev,batch):
    L=f.predict(net,features,du,lehead,le_sc,dev,batch)
    S=f.C_LE(L,C)
    metrics=new_metric(L,S,ds,ids)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',choices=f.ARMS,required=True)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--cap',type=float,default=1.0)
    p.add_argument('--physics-weight',type=float,default=0.25)
    p.add_argument('--anchor-weight',type=float,default=0.02)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not (1<=a.members<=5 and a.epochs>=1 and a.batch_size>=1 and a.lr>0 and
            a.hidden>=8 and 0<a.cap<=5 and 0<=a.physics_weight<=2 and
            0<=a.anchor_weight<=2 and a.patience>=1):raise ValueError('Invalid configuration')
    audit=root/'results'/f.ROOT/'A_baselines'/'65AF_A_summary.json'
    if not audit.is_file():raise FileNotFoundError('Run 65AF-A first')
    if json.loads(audit.read_text()).get('members',0)<a.members:
        raise RuntimeError('65AF-A needs at least this many members')
    dev=f.ac.aa.co.device_from_option(a.device)
    for reg in f.REGIONS:
        for m in range(1,a.members+1):f.cases(root,reg,m,dev,need_predictions=False)
    if a.check_only:
        print('65AF-B CHECK PASS: '+a.arm+'; both regions available; 64J train+validation only')
        return
    out=f.safe_output(root,'B_'+a.arm)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    records=[]
    for reg in f.REGIONS:
        for member in range(1,a.members+1):
            seed=650000+100*f.REGIONS.index(reg)+member
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            st=f.cases(root,reg,member,dev)
            ds=st['ds'];tr=ds['splits']['train'];val=ds['splits']['validation']
            if np.intersect1d(tr,val).size:raise RuntimeError('Split leakage')
            trF,trdu,trL,trS,trG=f.build_features(st,tr,dev)
            vF,vdu,vL,vS,vG=f.build_features(st,val,dev)
            trainC=f.stiffness(ds,tr);valC=f.stiffness(ds,val)
            le_sc=f.strain_scales(ds);s_sc=f.stress_scales(ds)
            srcbase=new_metric(vL,f.C_LE(vL,valC),ds,val)
            source_head=new_metric(vL,vS,ds,val)
            net=f.LocalStrainRefiner(a.arm,a.hidden,a.cap).to(dev)
            opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
            baseline_score=srcbase['S22']+.15*srcbase['LE22']+.05*srcbase['S11']+.05*srcbase['S12']
            best_score=baseline_score;best_epoch=0
            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            best_metrics=dict(srcbase);stale=0
            history=[]
            wLE=torch.tensor([1.,5.,1.,2.,.4,.4],dtype=torch.float32,device=dev)
            wS=torch.tensor([.3,3.,.2,.8,.1,.1],dtype=torch.float32,device=dev)
            let=torch.tensor(np.array(le_sc,copy=True),dtype=torch.float32,device=dev)
            sct=torch.tensor(np.array(s_sc,copy=True),dtype=torch.float32,device=dev)
            print('65AF-B',reg,a.arm,'baseline LE22=%.2f%% C(LE) S22=%.2f%%; learned-head S22=%.2f%%'%(
                srcbase['LE22'],srcbase['S22'],source_head['S22']),flush=True)
            for epoch in range(1,a.epochs+1):
                net.train();perm=np.random.permutation(len(tr));loss_acc=0.;count=0
                for start in range(0,len(perm),a.batch_size):
                    ix=perm[start:start+a.batch_size]
                    def t(x):return torch.tensor(np.array(x[ix],copy=True),dtype=torch.float32,device=dev)
                    F=t(trF);du=t(trdu);Lh=t(trL)
                    target=t(ds['LE'][tr]); stress=t(ds['S'][tr]);C=t(trainC)
                    L=net(F,du,Lh,let)
                    S=torch.einsum('bci,bei->bec',C,L)
                    lerr=((L-target)/let.reshape(1,1,6)).square()
                    serr=((S-stress)/sct.reshape(1,1,6)).square()
                    anchor=((L-Lh)/let.reshape(1,1,6)).square()
                    loss=(lerr*wLE.reshape(1,1,6)).mean()+a.physics_weight*(serr*wS.reshape(1,1,6)).mean()+a.anchor_weight*anchor.mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite 65AF training loss')
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
                    loss_acc+=float(loss.item())*len(ix);count+=len(ix)
                metrics=validate(net,vF,vdu,vL,le_sc,valC,ds,val,dev,a.batch_size)
                score=metrics['S22']+.15*metrics['LE22']+.05*metrics['S11']+.05*metrics['S12']
                allowed=safe(metrics,srcbase)
                if allowed and score<best_score-1e-8:
                    best_score=score;best_epoch=epoch;best_metrics=dict(metrics)
                    best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    stale=0
                else:stale+=1
                history.append(dict(epoch=epoch,train_loss=loss_acc/count,validation_score=score,
                    validation_LE22_pct=metrics['LE22'],validation_C_S22_pct=metrics['S22'],
                    validation_LE11_pct=metrics['LE11'],validation_LE12_pct=metrics['LE12'],
                    selected_safe=bool(allowed)))
                if epoch==1 or epoch%10==0:
                    print('  epoch %3d val LE22 %.2f%% C(LE) S22 %.2f%% best %d'%(
                        epoch,metrics['LE22'],metrics['S22'],best_epoch),flush=True)
                if stale>=a.patience:break
            path=f.checkpoint_path(root,a.arm,reg,member)
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(arch=f.ARCH,arm=a.arm,region=reg,member=member,
                source_sha=st['source_sha'],lgno_sha=st['lgno_sha'],
                hidden=a.hidden,cap=a.cap,selected_epoch=best_epoch,
                model_state_dict=best_state,
                training_data='Original 64J train',model_selection_data='Original 64J validation',
                original_test_and_65W_not_used=True,
                original_U_S_LE_force_heads_frozen=True,
                not_physical_strain_compatible=True),path)
            pd.DataFrame(history).to_csv(path.parent/'training_history.csv',index=False)
            row=dict(region=reg,member=member,arm=a.arm,selected_epoch=best_epoch,
                train_cases=len(tr),validation_cases=len(val),source_LE22_pct=srcbase['LE22'],
                source_C_S22_pct=srcbase['S22'],source_learned_head_S22_pct=source_head['S22'],
                candidate_LE22_pct=best_metrics['LE22'],candidate_C_S22_pct=best_metrics['S22'],
                candidate_LE11_pct=best_metrics['LE11'],candidate_LE12_pct=best_metrics['LE12'],
                model_changed=bool(best_epoch>0),weights_frozen=True,
                source_sha=st['source_sha'],lgno_sha=st['lgno_sha'],checkpoint=str(path))
            records.append(row)
            del net
            if dev.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(records).to_csv(out/'65AF_B_training_summary.csv',index=False)
    (out/'65AF_B_manifest.json').write_text(json.dumps(dict(step='65AF-B',arm=a.arm,members=a.members,
        original_data_test_unused=True,source_heads_frozen=True,training='64J train',
        selection='64J validation',WARNING='Model estimates material strain independently; not guaranteed compatible with U'),indent=2))
    print('65AF-B DONE',out)

if __name__=='__main__':main()
