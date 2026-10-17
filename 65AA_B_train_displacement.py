#!/usr/bin/env python3
"""65AA-B: controlled displacement-gradient training (Mac MPS/CPU).

Compare TWO arms with IDENTICAL starting 65Z constitutive checkpoints:
    --arm displacement : more nodal U supervision (control)
    --arm gradient     : same U supervision + reference gradient loss

Only displacement_operator is trainable. LE/S/force heads remain identical.
Uses only 64J train and validation cases. Saves in a NEW folder.

Recommended small pilot (one member each region):
 python3 65AA_B_train_displacement.py --project . --arm displacement --members 1 --epochs 60
 python3 65AA_B_train_displacement.py --project . --arm gradient --members 1 --epochs 60
"""
import argparse,copy,json,random,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import pandas as pd
import torch

spec=spec_from_file_location('step65aa',Path(__file__).with_name('65AA_common.py'))
aa=module_from_spec(spec);sys.modules[spec.name]=aa;spec.loader.exec_module(aa)


def get_val(model, ctx, ds, index, Utruth, Uoriginal, gscale, lam, anchor_w, batch):
    """Return immutable validation diagnostics; NO test cases used."""
    U=aa.predict_numpy(model,ctx,index,batch_size=batch)
    grad=aa.gradient_numpy(U,ds)
    truth_grad=aa.gradient_numpy(Utruth,ds)
    s=np.asarray(ctx['ustd'].detach().cpu().numpy(),dtype=np.float64)
    u_fit=float(np.mean(((U-Utruth)/s)**2))
    grad_fit=float(np.mean(((grad-truth_grad)/gscale)**2))
    anchor_fit=float(np.mean(((U-Uoriginal)/s)**2))
    d=aa.errors(U,Utruth,ds)
    d['validation_U_normalized_MSE']=u_fit
    d['validation_gradient_normalized_MSE']=grad_fit
    d['selection_objective']=u_fit+lam*grad_fit+anchor_w*anchor_fit
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',required=True,choices=['displacement','gradient'])
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--lambda-grad',type=float,default=.25)
    p.add_argument('--lambda-anchor',type=float,default=.02)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if not (1<=args.members<=5 and args.epochs>=1 and args.batch_size>=1 and args.lr>0
            and 0<=args.lambda_grad<=3 and 0<=args.lambda_anchor<=2 and args.patience>=1):
        raise ValueError('Invalid training arguments')
    root=Path(args.project).expanduser().resolve()
    device=aa.co.device_from_option(args.device)
    aroot=root/'results'/aa.ROOT_NAME/'A_baseline_gradient_audit'
    if not (aroot/'65AA_A_summary.json').is_file():
        raise FileNotFoundError('Run 65AA-A first: '+str(aroot/'65AA_A_summary.json'))
    audit=json.loads((aroot/'65AA_A_summary.json').read_text())
    if audit.get('model_count',0)<2*args.members:raise RuntimeError('65AA-A must audit at least this many members')
    for seg in aa.SEGMENTS:
        aa.load_data(root,seg)
        for m in range(1,args.members+1):
            f=aa.source_checkpoint(root,seg,m)
            if not f.is_file():raise FileNotFoundError(str(f))
    if args.check_only:
        print('65AA-B CHECK PASS: arm=%s, members=%d per region, device=%s'%(args.arm,args.members,device))
        print('Original test/65W case excluded; freeze LE, S and force heads; checkpoint output separate')
        return
    torch.set_num_threads(min(4,torch.get_num_threads()))
    out=root/'results'/aa.ROOT_NAME/('B_'+args.arm)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY:\n'+str(out)+'\nthen rerun this arm.')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for seg in aa.SEGMENTS:
        for m in range(1,args.members+1):
            seed=65100+100*aa.SEGMENTS.index(seg)+m
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            ds,ck,model,ctx,source=aa.init_case(root,seg,m,device)
            original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for param in model.parameters():param.requires_grad_(False)
            for param in model.displacement_operator.parameters():param.requires_grad_(True)
            train=ds['splits']['train'];val=ds['splits']['validation']
            trainval=np.concatenate([train,val]);loc={int(j):i for i,j in enumerate(trainval)}
            with torch.no_grad():
                Ubase=aa.predict_numpy(model,ctx,trainval,batch_size=args.batch_size)
            Utrue=ds['U'][trainval]
            grad_true=aa.gradient_numpy(Utrue,ds)
            grad_scale=max(float(np.sqrt(np.mean(grad_true[:len(train)]**2))),.001)
            val_ii=np.array([loc[int(i)] for i in val],dtype=int)
            Ubase_val=Ubase[val_ii]
            ref_val=ds['U'][val]
            active_lambda=(args.lambda_grad if args.arm=='gradient' else 0.0)
            baseline=get_val(model,ctx,ds,val,ref_val,Ubase_val,grad_scale,
                             active_lambda,args.lambda_anchor,args.batch_size)
            best=copy.deepcopy(original);best_epoch=0
            best_score=baseline['selection_objective'];best_stats=dict(baseline)
            opt=torch.optim.Adam(model.displacement_operator.parameters(),lr=args.lr,weight_decay=1e-6)
            history=[];stale=0
            print('65AA-B %s %-11s member %d baseline U2 %.4f%% grad %.3f%%'%(
                args.arm,seg,m,baseline['U2_rel_l2_pct'],baseline['gradient_rel_l2_pct']),flush=True)
            for epoch in range(1,args.epochs+1):
                model.train(); shuffled=np.random.permutation(train)
                running=0.;nb=0
                for start in range(0,len(shuffled),args.batch_size):
                    ids=shuffled[start:start+args.batch_size]
                    srcids=np.array([loc[int(i)] for i in ids],dtype=np.int64)
                    uref=torch.as_tensor(np.array(ds['U'][ids],copy=True),dtype=torch.float32,device=device)
                    uanchor=torch.as_tensor(np.array(Ubase[srcids],copy=True),dtype=torch.float32,device=device)
                    corrected=aa.predict_torch(model,ctx,ids,correct=True)
                    ufit=((corrected-uref)/ctx['ustd']).square().mean()
                    anchor=((corrected-uanchor)/ctx['ustd']).square().mean()
                    loss=ufit+args.lambda_anchor*anchor
                    if args.arm=='gradient':
                        gg=aa.gradient_torch(corrected,ctx['connectivity'],ctx['dN'])
                        gt=aa.gradient_torch(uref,ctx['connectivity'],ctx['dN'])
                        gl=((gg-gt)/grad_scale).square().mean()
                        loss=loss+active_lambda*gl
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite displacement training loss')
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.displacement_operator.parameters(),1.0)
                    opt.step()
                    running+=float(loss.item())*len(ids);nb+=len(ids)
                model.eval()
                stats=get_val(model,ctx,ds,val,ref_val,Ubase_val,grad_scale,
                              active_lambda,args.lambda_anchor,args.batch_size)
                safe_u2=stats['U2_rel_l2_pct']<=max(1.08*baseline['U2_rel_l2_pct'],baseline['U2_rel_l2_pct']+.02)
                safe_vec=stats['U_vector_rel_l2_pct']<=1.08*baseline['U_vector_rel_l2_pct']+.02
                safe_grad=(args.arm!='gradient' or
                           stats['gradient_rel_l2_pct']<=baseline['gradient_rel_l2_pct'])
                acceptable=safe_u2 and safe_vec and safe_grad
                history.append(dict(epoch=epoch,train_loss=running/max(nb,1),
                    validation_score=stats['selection_objective'],
                    validation_U2_rel_l2_pct=stats['U2_rel_l2_pct'],
                    validation_gradient_rel_l2_pct=stats['gradient_rel_l2_pct'],
                    accepted_by_safety_gate=acceptable))
                if acceptable and stats['selection_objective']<best_score-1e-9:
                    best_score=stats['selection_objective'];best_epoch=epoch
                    best_stats=dict(stats);stale=0
                    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:stale+=1
                if epoch==1 or epoch%10==0:
                    print(' ep %3d val U2 %.4f%% grad %.3f%% best=%d'%(epoch,stats['U2_rel_l2_pct'],
                          stats['gradient_rel_l2_pct'],best_epoch),flush=True)
                if stale>=args.patience:break
            modified={k:v for k,v in best.items() if not k.startswith('displacement_operator.')}
            if not all(torch.equal(v,original[k]) for k,v in modified.items()):
                raise RuntimeError('FROZEN stress/strain/force weights changed')
            savefolder=out/seg/('member_%02d'%m);savefolder.mkdir(parents=True,exist_ok=True)
            candidate=dict(ck);candidate['model_state_dict']=best
            candidate['65AA_gradient_experiment']=dict(arm=args.arm,member=m,segment=seg,
                initialization_sha256=aa.co.sha256(source),selected_epoch=best_epoch,
                lambda_grad=active_lambda,lambda_anchor=args.lambda_anchor,
                train_split_only=True,validation_selection_only=True,
                original_test_used=False,middle_65W_case_used=False,
                original_source_preserved=True,
                displacement_head_modified=best_epoch>0,
                frozen_LE_S_force_equal=True,baseline_validation=baseline,
                chosen_validation=best_stats)
            torch.save(candidate,savefolder/'best_v4_7region.pt')
            pd.DataFrame(history).to_csv(savefolder/'training_history.csv',index=False)
            row=dict(segment=seg,member=m,arm=args.arm,best_epoch=best_epoch,
                train_cases=len(train),validation_cases=len(val),
                baseline_U2_pct=baseline['U2_rel_l2_pct'],candidate_U2_pct=best_stats['U2_rel_l2_pct'],
                baseline_gradient_pct=baseline['gradient_rel_l2_pct'],
                candidate_gradient_pct=best_stats['gradient_rel_l2_pct'],
                gradient_improvement_pct=100*(baseline['gradient_rel_l2_pct']-best_stats['gradient_rel_l2_pct'])/
                                         max(baseline['gradient_rel_l2_pct'],1e-12),
                displacement_and_mechanics_frozen_ok=True,
                chosen_score=best_score,source_sha256=aa.co.sha256(source))
            records.append(row)
            print('  CHOSEN epoch %d: U2 %.4f%% gradient %.3f%%'%(
                best_epoch,row['candidate_U2_pct'],row['candidate_gradient_pct']),flush=True)
            del model
            if device.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(records).to_csv(out/'65AA_B_training_summary.csv',index=False)
    (out/'65AA_B_manifest.json').write_text(json.dumps(dict(
        step='65AA-B',arm=args.arm,number_models=len(records),
        only_train_validation=True,original_test_and_65W_not_used=True,
        source='Step 65Z constitutive checkpoints',
        frozen_heads=['strain_operator','stress_operator','force_coefficient_head'],
        trainable_heads=['displacement_operator'],
        validation_safety_gates=['U2 degradation at most 8%','full-vector degradation at most 8%',
                                 'gradient may not deteriorate for gradient arm'],
        note='Offline candidates. Changing displacement requires full coupling verification before deployment.'),indent=2))
    print('65AA-B DONE:',out)

if __name__=='__main__':main()
