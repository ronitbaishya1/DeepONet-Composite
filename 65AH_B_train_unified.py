#!/usr/bin/env python3
"""B: train original U/LE/S DeepONet heads END-TO-END with a shared pathway.

Arm shared: shared branch/trunk + direct FEM element gradient descriptors.
Arm local : same + LGNO-style material/state-dependent local gradient stencil.
Neither changes force-PCA head. Original 56 TRAIN -> 44 fit, 12 internal tune.

python3 65AH_B_train_unified.py --project . --arm shared --epochs 150
python3 65AH_B_train_unified.py --project . --arm local --epochs 150
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
import importlib
ah=importlib.import_module('65AH_common')


def loss_terms(pred,t,ids):
 U,L,S=pred
 pu=(U-t['U'][ids])/t['ustd']
 pl=(L-t['LE'][ids])/t['lestd']
 ps=(S-t['S'][ids])/t['sstd']
 wu=pu.new_tensor([1.,3.,1.]).reshape(1,1,3)
 wl=pu.new_tensor([1.,5.,1.,2.,1.,1.]).reshape(1,1,6)
 ws=pu.new_tensor([1.,5.,1.,2.,1.,1.]).reshape(1,1,6)
 lu=(wu*pu.square()).mean()
 ll=(wl*pl.square()).mean()
 ls=(ws*ps.square()).mean()
 return lu,ll,ls


def fit_one(a,root,region,member,dev):
 random.seed(650882+100*ah.SEGMENTS.index(region)+member)
 np.random.seed(650882+100*ah.SEGMENTS.index(region)+member)
 torch.manual_seed(650882+100*ah.SEGMENTS.index(region)+member)
 st=ah.load_source(root,region,member,dev)
 fit,tune=ah.split_in_train(st['ds'],region,member,a.tune_cases)
 net=ah.get_model(st,dev,a.arm,a.hidden,a.cap)
 frozen_before={k:v.clone() for k,v in net.backbone.force_coefficient_head.state_dict().items()}
 # Fine tune already-trained physics heads conservatively; initialize new shared/local heads with higher LR.
 groups=[{'params': [p for name,p in net.backbone.named_parameters() if p.requires_grad], 'lr':a.base_lr},
         {'params': list(net.shared_branch.parameters())+list(net.shared_trunk.parameters())+
                    list(net.node_head.parameters())+list(net.stencil.parameters())+
                    list(net.ip_mixer.parameters())+list(net.strain_head.parameters())+
                    list(net.stress_head.parameters()),'lr':a.lr}]
 opt=torch.optim.AdamW(groups,weight_decay=1e-6)
 # The shared arm's stencil has no forward effect; harmless to carry parameters,
 # but ensure no optimizer/gradient spill into force head.
 assert all(not p.requires_grad for p in net.backbone.force_coefficient_head.parameters())
 net.eval()
 base=ah.baseline_numpy(st,tune,dev)
 base_met=ah.metrics(base,ah.reference(st,tune))
 best=copy.deepcopy({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
 best_epoch=0;best_obj=ah.objective(base_met);best_met=base_met
 stale=0;history=[]
 print('65AH-B %s %s member %d: initial U2 %.3f LE22 %.3f S22 %.3f'%(a.arm,region,member,base_met['U2_pct'],base_met['LE22_pct'],base_met['S22_pct']),flush=True)
 for epoch in range(1,a.epochs+1):
  net.train();ordered=np.random.permutation(fit)
  tot=0.;steps=0
  for start in range(0,len(ordered),a.batch_size):
   ids=np.array(ordered[start:start+a.batch_size],np.int64)
   pred=net(ids,st['t'])
   lu,ll,ls=loss_terms(pred,st['t'],ids)
   loss=a.u_weight*lu+a.le_weight*ll+a.s_weight*ls
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss (try lower --lr and --base-lr)')
   opt.zero_grad(set_to_none=True);loss.backward()
   torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
   opt.step();tot+=float(loss.detach())*len(ids);steps+=len(ids)
  net.eval();predict=ah.predict_numpy(net,st['t'],tune,a.batch_size)
  met=ah.metrics(predict,ah.reference(st,tune))
  ob=ah.objective(met)
  safe=(met['U2_pct']<=base_met['U2_pct']*1.10+.1 and
        met['S11_pct']<=base_met['S11_pct']*1.25+.3 and
        met['LE11_pct']<=base_met['LE11_pct']*1.25+.3)
  history.append(dict(epoch=epoch,train_loss=tot/steps,internal_tune_obj=ob,
      internal_tune_S22_pct=met['S22_pct'],internal_tune_LE22_pct=met['LE22_pct'],
      internal_tune_U2_pct=met['U2_pct'],safe=bool(safe)))
  if safe and ob<best_obj-1e-6:
   best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
   best_epoch=epoch;best_met=dict(met);best_obj=ob;stale=0
  else:stale+=1
  if epoch in (1,a.epochs) or epoch%10==0:
   print('  epoch %d: U2 %.3f LE22 %.3f S22 %.3f best %d'%(epoch,met['U2_pct'],met['LE22_pct'],met['S22_pct'],best_epoch),flush=True)
  if stale>=a.patience:break
 net.load_state_dict(best,strict=True)
 for k,v in frozen_before.items():
  if not torch.equal(net.backbone.force_coefficient_head.state_dict()[k].detach().cpu(),v.cpu()):
   raise RuntimeError('Force PCA head changed -- ABORT')
 # Some fine-tuning may select epoch 0: safe rejection (do not force a false gain).
 dest=ah.ck_path(root,a.arm,region,member);dest.parent.mkdir(parents=True,exist_ok=True)
 ck=dict(arch=ah.ARCH,arm=a.arm,seg=region,member=member,
         source_sha=st['source_sha'],hidden=a.hidden,cap=a.cap,
         hyperparameters=dict(epochs=a.epochs,base_lr=a.base_lr,lr=a.lr,
                              u_weight=a.u_weight,le_weight=a.le_weight,s_weight=a.s_weight),
         fit_cases=list(map(int,fit)),internal_tune_cases=list(map(int,tune)),
         selected_epoch=best_epoch,selected_objective=best_obj,
         original_validation_unused_for_65AH_selection=True,
         original_test_and_65W_unused=True,source_force_head_frozen=True,
         direct_joint_U_LE_S_training=True,physical_consistency_not_enforced=True,
         state_dict=best)
 torch.save(ck,dest)
 pd.DataFrame(history).to_csv(dest.parent/'training_history.csv',index=False)
 print('  saved',dest,flush=True)
 return dict(segment=region,member=member,arm=a.arm,selected_epoch=best_epoch,
             internal_fit_cases=len(fit),internal_tune_cases=len(tune),
             baseline_tune_U2_pct=base_met['U2_pct'],candidate_tune_U2_pct=best_met['U2_pct'],
             baseline_tune_LE22_pct=base_met['LE22_pct'],candidate_tune_LE22_pct=best_met['LE22_pct'],
             baseline_tune_S22_pct=base_met['S22_pct'],candidate_tune_S22_pct=best_met['S22_pct'],
             source_sha=st['source_sha'],file=str(dest))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.')
 p.add_argument('--arm',choices=ah.ARMS,required=True)
 p.add_argument('--members',type=int,default=1)
 p.add_argument('--epochs',type=int,default=150)
 p.add_argument('--patience',type=int,default=35)
 p.add_argument('--batch-size',type=int,default=2)
 p.add_argument('--tune-cases',type=int,default=12)
 p.add_argument('--lr',type=float,default=2e-4,help='new shared/stencil LR')
 p.add_argument('--base-lr',type=float,default=1e-5,help='pretrained V4 U/LE/S LR')
 p.add_argument('--hidden',type=int,default=64)
 p.add_argument('--cap',type=float,default=.5)
 p.add_argument('--u-weight',type=float,default=.3)
 p.add_argument('--le-weight',type=float,default=1.)
 p.add_argument('--s-weight',type=float,default=1.)
 p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
 p.add_argument('--check-only',action='store_true')
 a=p.parse_args();root=Path(a.project).expanduser().resolve()
 if not (1<=a.members<=5 and a.epochs>=1 and a.batch_size>=1 and a.patience>=1 and
         4<=a.tune_cases<=20 and 8<=a.hidden<=256 and 0<a.lr<=.01 and 0<a.base_lr<=.01 and
         0<a.cap<=5 and a.u_weight>0 and a.le_weight>0 and a.s_weight>0):raise ValueError('Invalid training configuration')
 audit=root/'results'/ah.ROOT_NAME/'A_audit'/'65AH_A_summary.json'
 if not audit.is_file():raise FileNotFoundError('Run 65AH-A first: '+str(audit))
 assert json.loads(audit.read_text())['members']>=a.members
 dev=ah.co.device_from_option(a.device)
 for reg in ah.SEGMENTS:
  for m in range(1,a.members+1):ah.load_source(root,reg,m,dev)
 if a.check_only:
  print('65AH-B check PASS:',a.arm,dev);return
 out=ah.output_dir(root,'B_'+a.arm)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 rows=[]
 for reg in ah.SEGMENTS:
  for m in range(1,a.members+1):rows.append(fit_one(a,root,reg,m,dev))
 pd.DataFrame(rows).to_csv(out/'65AH_B_training_summary.csv',index=False)
 (out/'65AH_B_manifest.json').write_text(json.dumps(dict(step='65AH-B',arm=a.arm,regions=ah.SEGMENTS,
         training='44 fit / 12 internal tuning from original 56 train cases, deterministic split',
         original_validation_was_reused_by_upstream=True,original_test_and_65W_untouched=True,
         safety='Preserves original generalized-force PCA head EXACTLY'),indent=2))
 print('65AH-B DONE',out)

if __name__=='__main__':main()
