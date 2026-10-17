#!/usr/bin/env python3
"""65AB-B: train ONLY the zero-at-interface displacement residual network.

Both arms share the same 65Z base weights and same zero-initialized architecture.
  --arm nodal     : supervised nodal displacement (control)
  --arm gradient  : same plus FEM displacement-gradient supervision
The old NO force, strain, stress and displacement weights are *never updated*.
Uses original 64J train cases; selects epochs on original validation cases.

Start with one member per region, both arms, to limit Mac MPS compute.
"""
import argparse,copy,json,random,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np,pandas as pd,torch
spec=spec_from_file_location('step65ab',Path(__file__).with_name('65AB_common.py'))
ab=module_from_spec(spec);sys.modules[spec.name]=ab;spec.loader.exec_module(ab)


def evaluate(op,ctx,ds,idx,baseU,phi,ustd_np,gradscale,batch):
    """Reads original validation cases only, no test or 65W reference."""
    op.eval();out=[]
    with torch.no_grad():
        for k in range(0,len(idx),batch):
            ids=idx[k:k+batch]
            u0=torch.as_tensor(np.array(baseU[k:k+len(ids)],copy=True),dtype=torch.float32,device=phi.device)
            uu=ab.corrected_from_residual(u0,op,ctx,ids,phi)
            out.append(uu.detach().cpu().numpy())
    pred=np.concatenate(out)
    truth=ds['U'][idx]
    gpred=ab.aa.gradient_numpy(pred,ds)
    gtrue=ab.aa.gradient_numpy(truth,ds)
    d=ab.aa.errors(pred,truth,ds)
    d['objective_U_MSE']=float(np.mean(((pred-truth)/ustd_np)**2))
    d['objective_gradient_MSE']=float(np.mean(((gpred-gtrue)/gradscale)**2))
    d['objective_delta_MSE']=float(np.mean(((pred-baseU)/ustd_np)**2))
    d['selection_objective']=d['objective_U_MSE']+.25*d['objective_gradient_MSE']+.01*d['objective_delta_MSE']
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--arm',required=True,choices=['nodal','gradient'])
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lambda-gradient',type=float,default=.25)
    p.add_argument('--lambda-residual',type=float,default=.01)
    p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--latent-dim',type=int,default=48)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not(1<=a.members<=5 and a.epochs>=1 and a.batch_size>0 and 0<a.lr<=.01 and
           0<=a.lambda_gradient<=3 and 0<=a.lambda_residual<=2 and
           a.hidden_dim>=16 and a.latent_dim>=8 and a.depth>=2 and a.patience>=1):
        raise ValueError('Invalid arguments')
    device=ab.aa.co.device_from_option(a.device)
    audit=root/'results'/ab.ROOT_NAME/'A_interface_lift_audit'/'65AB_A_summary.json'
    if not audit.is_file():raise FileNotFoundError('Run 65AB-A first: '+str(audit))
    d=json.loads(audit.read_text())
    if not d.get('all_zero_initialization_passed') or not d.get('all_prescribed_interfaces_unchanged') or d.get('members',0)<a.members:
        raise RuntimeError('65AB-A audit did not pass for all requested members')
    for seg in ab.SEGMENTS:
        ab.aa.load_data(root,seg)
        for member in range(1,a.members+1):
            if not ab.aa.source_checkpoint(root,seg,member).is_file():
                raise FileNotFoundError(str(ab.aa.source_checkpoint(root,seg,member)))
    if a.check_only:
        print('65AB-B CHECK PASS:',a.arm,'members',a.members,'device',device)
        print('Original 65Z models frozen; original 64J train/validation only; original test unused')
        return
    torch.set_num_threads(min(4,torch.get_num_threads()))
    output=root/'results'/ab.ROOT_NAME/('B_'+a.arm)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY:\n'+str(output)+'\nthen rerun')
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for seg in ab.SEGMENTS:
        for member in range(1,a.members+1):
            seed=650200+100*ab.SEGMENTS.index(seg)+member
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            ds,ck,base_model,ctx,source=ab.aa.init_case(root,seg,member,device)
            base_model.eval()
            for param in base_model.parameters():param.requires_grad_(False)
            tr=ds['splits']['train'];va=ds['splits']['validation']
            trainval=np.concatenate((tr,va)); pos={int(j):i for i,j in enumerate(trainval)}
            ub_all=ab.aa.predict_numpy(base_model,ctx,trainval,batch_size=a.batch_size)
            ub_val=ub_all[len(tr):]
            grad_ref=ab.aa.gradient_numpy(ds['U'][tr],ds)
            gradscale=max(float(np.sqrt(np.mean(grad_ref**2))),1e-3)
            env,ends=ab.envelope_numpy(ds)
            boundary=ab.boundary_local_indices(ctx,ds)
            phi=torch.tensor(np.array(env,copy=True),dtype=torch.float32,device=device)
            op=ab.make_operator(ck['branch_dim'],device,a.hidden_dim,a.latent_dim,a.depth,seed=seed)
            init={k:v.detach().cpu().clone() for k,v in op.state_dict().items()}
            ustd=ctx['ustd'].detach().cpu().numpy()
            baseline=evaluate(op,ctx,ds,va,ub_val,phi,ustd,gradscale,a.batch_size)
            if baseline['objective_delta_MSE']>1e-11:
                raise RuntimeError('Initial residual MUST be zero; got nonzero modification')
            best=copy.deepcopy(init);best_epoch=0;best_metrics=dict(baseline);best_score=baseline['selection_objective']
            optimizer=torch.optim.Adam(op.parameters(),lr=a.lr,weight_decay=1e-6)
            stale=0;history=[]
            print('65AB-B %-8s %-11s member%02d baseline U2 %.4f%% gradient %.4f%%'%(
                 a.arm,seg,member,baseline['U2_rel_l2_pct'],baseline['gradient_rel_l2_pct']),flush=True)
            for epoch in range(1,a.epochs+1):
                op.train(); perm=np.random.permutation(tr); loss_sum=0.;seen=0
                for start in range(0,len(perm),a.batch_size):
                    ids=perm[start:start+a.batch_size]
                    basepos=np.asarray([pos[int(j)] for j in ids],dtype=np.int64)
                    u0=torch.tensor(np.array(ub_all[basepos],copy=True),dtype=torch.float32,device=device)
                    uref=torch.tensor(np.array(ds['U'][ids],copy=True),dtype=torch.float32,device=device)
                    pred=ab.corrected_from_residual(u0,op,ctx,ids,phi)
                    uloss=((pred-uref)/ctx['ustd']).square().mean()
                    correction=((pred-u0)/ctx['ustd']).square().mean()
                    loss=uloss+a.lambda_residual*correction
                    if a.arm=='gradient':
                        gp=ab.aa.gradient_torch(pred,ctx['connectivity'],ctx['dN'])
                        gt=ab.aa.gradient_torch(uref,ctx['connectivity'],ctx['dN'])
                        loss=loss+a.lambda_gradient*((gp-gt)/gradscale).square().mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite bubble displacement loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(op.parameters(),max_norm=1.0)
                    optimizer.step();loss_sum+=float(loss.item())*len(ids);seen+=len(ids)
                stats=evaluate(op,ctx,ds,va,ub_val,phi,ustd,gradscale,a.batch_size)
                safe=(stats['U2_rel_l2_pct']<=1.05*baseline['U2_rel_l2_pct']+.02 and
                      stats['U_vector_rel_l2_pct']<=1.05*baseline['U_vector_rel_l2_pct']+.02 and
                      stats['gradient_rel_l2_pct']<=1.02*baseline['gradient_rel_l2_pct'])
                valid=bool(safe and np.isfinite(stats['selection_objective']))
                history.append(dict(epoch=epoch,train_loss=loss_sum/max(1,seen),
                    val_U2_pct=stats['U2_rel_l2_pct'],val_gradient_pct=stats['gradient_rel_l2_pct'],
                    val_selection_score=stats['selection_objective'],safe=valid))
                if valid and stats['selection_objective']<best_score-1e-10:
                    best_score=stats['selection_objective'];best_epoch=epoch
                    best_metrics=dict(stats);best={k:v.detach().cpu().clone() for k,v in op.state_dict().items()};stale=0
                else:stale+=1
                if epoch==1 or epoch%10==0:
                    print(' epoch %3d val U2 %.4f%% gradient %.4f%% best %d'%(
                         epoch,stats['U2_rel_l2_pct'],stats['gradient_rel_l2_pct'],best_epoch),flush=True)
                if stale>=a.patience:break
            # Evaluate selected state and STRICTLY check zero interface correction.
            op.load_state_dict(best);op.eval()
            pred=[]
            with torch.no_grad():
                for start in range(0,len(va),a.batch_size):
                    ids=va[start:start+a.batch_size]
                    u0=torch.tensor(np.array(ub_val[start:start+len(ids)],copy=True),dtype=torch.float32,device=device)
                    u1=ab.corrected_from_residual(u0,op,ctx,ids,phi)
                    pred.append(u1.cpu().numpy())
            pred=np.concatenate(pred)
            interface_max=float(np.max(np.abs(pred[:,boundary,:]-ub_val[:,boundary,:])))
            if interface_max>2e-6:raise RuntimeError('Boundary correction is NOT exactly zero')
            f=ab.candidate_dir(root,a.arm,seg,member);f.mkdir(parents=True,exist_ok=True)
            candidate=dict(architecture=ab.ARCHITECTURE,arm=a.arm,segment=seg,member=member,
                source_sha256=ab.aa.co.sha256(source),source_checkpoint_path=str(source),
                hidden_dim=a.hidden_dim,latent_dim=a.latent_dim,depth=a.depth,
                residual_model_state_dict=best,baseline_validation=baseline,
                selected_validation=best_metrics,selected_epoch=best_epoch,
                interface_max_displacement_change_mm=interface_max,
                only_original_64J_train_validation_used=True,
                original_64J_test_and_65W_reference_unused=True,
                frozen_original_heads=True,
                note='Requires 65AB adapter to infer; NOT drop-in V4 model')
            torch.save(candidate,f/'bubble_residual.pt')
            pd.DataFrame(history).to_csv(f/'training_history.csv',index=False)
            row=dict(segment=seg,arm=a.arm,member=member,best_epoch=best_epoch,
                original_U2_pct=baseline['U2_rel_l2_pct'],new_U2_pct=best_metrics['U2_rel_l2_pct'],
                original_gradient_pct=baseline['gradient_rel_l2_pct'],
                new_gradient_pct=best_metrics['gradient_rel_l2_pct'],
                gradient_gain_pct=100*(baseline['gradient_rel_l2_pct']-best_metrics['gradient_rel_l2_pct'])/
                                  max(baseline['gradient_rel_l2_pct'],1e-10),
                interface_max_change_mm=interface_max,reference_model_sha256=ab.aa.co.sha256(source),
                checkpoint=str(f/'bubble_residual.pt'))
            records.append(row)
            print(' CHOSEN epoch %d: U2 %.4f%%, gradient %.4f%%, interface change %.1e mm'%(
                best_epoch,row['new_U2_pct'],row['new_gradient_pct'],interface_max),flush=True)
            del base_model,op
            if device.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(records).to_csv(output/'65AB_B_training_summary.csv',index=False)
    (output/'65AB_B_manifest.json').write_text(json.dumps(dict(
        step='65AB-B',arm=a.arm,architecture=ab.ARCHITECTURE,members=a.members,
        cases='original 64J train and development-validation only',
        independent_test_used=False,old_65Z_checkpoint_unchanged=True,
        entire_source_V4_network_frozen=True,
        trainable_model='new interior bubble residual only',
        common_model_selection='normalized U MSE + 0.25 normalized gradient MSE + 0.01 residual MSE',
        manual_coupled_verification_required_before_deployment=True),indent=2))
    print('65AB-B FINISHED:',output)
if __name__=='__main__':main()
