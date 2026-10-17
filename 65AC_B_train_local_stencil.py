#!/usr/bin/env python3
"""65AC-B: train LGNO-inspired local C3D8R stencil / ordinary MLP control.

Both use 65AA-gradient *frozen predicted* displacement inputs. FEM gradients are
training targets ONLY for 64J TRAIN cases, validation used solely for selection.
Their output is an element-center GRADIENT ESTIMATE, not an integrated U field.
Train separate arms, same data/split/initial (zero correction) and learning rate:
 python3 65AC_B_train_local_stencil.py --project . --arm pointwise --members 1 --epochs 100
 python3 65AC_B_train_local_stencil.py --project . --arm lgno --members 1 --epochs 100
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ac=importlib.import_module('65AC_common')


def loss_weights(device):
    return torch.tensor([[1.,2.,1.],[2.,3.,1.],[1.,1.,1.]],device=device).reshape(1,1,3,3)


def validation_score(pred,target,gs):
    weights=np.array([[1.,2.,1.],[2.,3.,1.],[1.,1.,1.]],dtype=np.float64)
    return float(np.mean(((pred-target)/gs[None,None,:,:])**2*weights[None,None,:,:]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',required=True,choices=ac.ARMS)
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=4e-4)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--coeff-limit',type=float,default=.8)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--lambda-anchor',type=float,default=.01)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not(1<=args.members<=5 and args.epochs>0 and args.batch_size>0 and 0<args.lr<.02 and
           8<=args.hidden<=256 and 0<args.coeff_limit<=3 and args.patience>0 and 0<=args.lambda_anchor<=2):
        raise ValueError('Invalid training parameters')
    root=Path(args.project).expanduser().resolve()
    audit=root/'results'/ac.ROOT_NAME/'A_stencil_audit'/'65AC_A_summary.json'
    if not audit.is_file():raise FileNotFoundError('Run 65AC-A first: '+str(audit))
    a=json.loads(audit.read_text())
    if a.get('num_members',0)<args.members:raise RuntimeError('Run 65AC-A with at least this many members')
    dev=ac.aa.co.device_from_option(args.device)
    for seg in ac.SEGMENTS:
        for m in range(1,args.members+1):
            ac.prepare(root,seg,m,dev,need_predictions=False)
    if args.check_only:
        print('65AC-B CHECK PASS: arm=%s, members=%d per region, device=%s'%(args.arm,args.members,dev))
        return
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=root/'results'/ac.ROOT_NAME/('B_'+args.arm)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing output protected. Manually delete ONLY:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    allrows=[];weights=loss_weights(dev)
    for seg in ac.SEGMENTS:
        for m in range(1,args.members+1):
            seed=65000+100*ac.SEGMENTS.index(seg)+m
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            st=ac.prepare(root,seg,m,dev)
            ds=st['ds'];train=ds['splits']['train'];val=ds['splits']['validation']
            if set(train).intersection(set(val)):raise RuntimeError('Training/validation leakage')
            source_val=ac.split_predict(st,None,val,dev,args.batch_size)
            gval=ac.gradient_from_u(ds['U'][val],ds)
            base=ac.score(source_val,gval,ds['LE'][val]);base_score=validation_score(source_val,gval,st['gs'])
            net=ac.LocalStencilGradientOperator(args.arm,args.hidden,args.coeff_limit).to(dev)
            optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-5)
            best=copy.deepcopy(net.state_dict());best_epoch=0;best_score=base_score;best_metrics=dict(base)
            history=[];stale=0
            print('65AC-B %s %s member %d baseline grad %.3f%% logLE22 %.2f%%'%(
                    args.arm,seg,m,base['gradient_rel_l2_pct'],base['logU_LE22_rel_l2_pct']),flush=True)
            for epoch in range(1,args.epochs+1):
                net.train();permutation=np.random.permutation(train);total=0.;n=0
                for start in range(0,len(permutation),args.batch_size):
                    ids=permutation[start:start+args.batch_size]
                    ii=np.array([st['index_map'][int(j)] for j in ids],dtype=np.int64)
                    inputs=torch.tensor(np.array(st['pred'][ii],copy=True),device=dev,dtype=torch.float32)
                    refs=torch.tensor(np.array(ac.gradient_from_u(ds['U'][ids],ds),copy=True),device=dev,dtype=torch.float32)
                    baseline=torch.einsum('beic,eij->becj',inputs[:,st['ctx']['conn'],:],st['ctx']['dN'])
                    pred=net(inputs,st['mats'][ids],st['ctx'])
                    scale=st['ctx']['gstd'].view(1,1,3,3)
                    supervised=((((pred-refs)/scale)**2)*weights).mean()
                    anchor=(((pred-baseline)/scale)**2).mean()
                    loss=supervised+args.lambda_anchor*anchor
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
                    optimizer.step()
                    total+=float(loss.item())*len(ids);n+=len(ids)
                net.eval()
                val_pred=ac.split_predict(st,net,val,dev,args.batch_size)
                vals=ac.score(val_pred,gval,ds['LE'][val]);obj=validation_score(val_pred,gval,st['gs'])
                ok=ac.safety(vals,base)
                history.append(dict(epoch=epoch,train_loss=total/max(n,1),validation_score=obj,
                                    gradient_rel_l2_pct=vals['gradient_rel_l2_pct'],
                                    logU_LE22_rel_l2_pct=vals['logU_LE22_rel_l2_pct'],
                                    safety_gate=bool(ok)))
                if ok and obj<best_score-1e-10:
                    best=copy.deepcopy({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
                    best_epoch=epoch;best_score=obj;best_metrics=dict(vals);stale=0
                else:stale+=1
                if epoch in (1,args.epochs) or epoch%10==0:
                    print('  epoch %d val grad %.3f%% logLE22 %.2f%% best=%d'%(
                          epoch,vals['gradient_rel_l2_pct'],vals['logU_LE22_rel_l2_pct'],best_epoch),flush=True)
                if stale>=args.patience:break
            if not ac.safety(best_metrics,base):raise RuntimeError('Stored candidate violates safety gate')
            folder=out/seg/('member_%02d'%m);folder.mkdir(parents=True,exist_ok=True)
            pt=ac.checkpoint_path(root,args.arm,seg,m)
            payload=dict(architecture=ac.ARCH,arm=args.arm,segment=seg,member=m,
                         hidden=args.hidden,coeff_limit=args.coeff_limit,
                         source_sha256=st['source_sha'],model_state_dict=best,
                         selected_epoch=best_epoch,train_only=True,validation_only=True,
                         test_and_65W_not_used=True,source_displacement_and_interface_forces_unchanged=True,
                         data_note='Non-integrable gradient-estimate adapter; source displacement unchanged')
            torch.save(payload,pt)
            pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            record=dict(arm=args.arm,segment=seg,member=m,selected_epoch=best_epoch,
                        train_cases=len(train),validation_cases=len(val),
                        baseline_gradient_pct=base['gradient_rel_l2_pct'],
                        candidate_gradient_pct=best_metrics['gradient_rel_l2_pct'],
                        baseline_logLE22_pct=base['logU_LE22_rel_l2_pct'],
                        candidate_logLE22_pct=best_metrics['logU_LE22_rel_l2_pct'],
                        source_sha256=st['source_sha'],checkpoint=str(pt),
                        parameter_count=sum(t.numel() for t in net.parameters()),
                        selected_validation_score=best_score)
            allrows.append(record)
            del net
            if dev.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(allrows).to_csv(out/'65AC_B_training_summary.csv',index=False)
    (out/'65AC_B_manifest.json').write_text(json.dumps(dict(step='65AC-B',arm=args.arm,
        source='Frozen 65AA-gradient V4 displacement checkpoints',train_cases_only=True,
        validation_for_model_selection=True,original_test_and_65W_untouched=True,
        claim='Gradient estimate only. Does not modify actual U or FE-NO interface force heads.',
        members=args.members),indent=2))
    print('65AC-B DONE:',out)

if __name__=='__main__':main()
