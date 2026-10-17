#!/usr/bin/env python3
"""65Z-B: protected mechanics-only V4 physics-ablation from 65Y checkpoints.

Two arms (train separately; do not mix conclusions):
   constitutive: normalized supervised + anchored + soft orthotropic stress/strain
   combined:     above + nonlinear logarithmic-strain consistency, ONLY if every
                 requested model passed the predicted-U gate in 65Z-A.

The U and generalized interface-force heads stay byte-for-byte identical.
Select best epoch on original 64J validation supervised loss (NOT test/65W).

  python3 65Z_B_physics_finetune.py --project . --arm constitutive --check-only
  python3 65Z_B_physics_finetune.py --project . --arm constitutive --members 5 --epochs 100
  python3 65Z_B_physics_finetune.py --project . --arm combined --check-only

If combined fails its audit gate: SKIP it. Never force --arm combined.
"""
import argparse,copy,json,random,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset

sp=spec_from_file_location('s65zcommon',Path(__file__).with_name('65Z_common.py'))
co=module_from_spec(sp);sys.modules[sp.name]=co;sp.loader.exec_module(co)


def supervised(pred_LE,pred_S,LE,S,weights):
    w=weights.reshape(1,1,6)
    return (((pred_LE-LE)**2)*w).mean()+(((pred_S-S)**2)*w).mean()


def smooth_l1_zero(residual, beta=1.0):
    """Smooth-L1 against zero, without MPS-incompatible expanded/view tensors.

    Equivalent to torch.nn.functional.smooth_l1_loss(residual, zeros_like(residual),
    beta=beta, reduction='mean'), including autograd, but uses elementwise ops.
    """
    residual = residual.contiguous()
    absolute = residual.abs()
    return torch.where(absolute < beta,
                       0.5 * residual.square() / beta,
                       absolute - 0.5 * beta).mean()


def physics_loss(pred_LE,pred_S,kin,ck_t,stiffness,arm):
    """Physical heads are stored normalized; convert to physical using 64N stats.

    Constitutive residual here is a SOFT *empirically Abaqus-audited* approximation.
    Engineering shear mapping is LE[:,3:] * 1, not *2.
    Use robust SmoothL1 for potentially large initial residuals.
    """
    lemu,lestd,smu,sstd=ck_t
    physicalLE=pred_LE*lestd+lemu
    physicalS=pred_S*sstd+smu
    approximateS=torch.einsum('bec,bdc->bed',physicalLE,stiffness)
    residue=(physicalS-approximateS)/sstd
    const=smooth_l1_zero(residue, beta=1.0)
    if arm=='combined':
        if kin is None:raise RuntimeError('Missing validated predicted-U logarithmic strain targets')
        kin_norm=(kin-lemu)/lestd
        kresidue=pred_LE-kin_norm
        kinematic=smooth_l1_zero(kresidue, beta=1.0)
    else:kinematic=const.new_zeros(())
    return const,kinematic


def verify_cache(audit,root,seg,m,ck_path,ds):
    name='%s/member_%02d'%(seg,m)
    state=audit['gates'].get(name)
    if not state:raise RuntimeError('65Z-A missing gate for '+name)
    path=root/'results'/'65Z_nonlinear_mechanics'/'A_predicted_kinematics'/seg/('member_%02d_kin.npz'%m)
    if not path.is_file():raise FileNotFoundError(str(path))
    with np.load(path,allow_pickle=False) as cache:
        orig_sha=str(cache['checkpoint_sha256'].item())
        idx=np.asarray(cache['indices'],dtype=int)
        target=np.asarray(cache['strain_logU_eng'],dtype=np.float32)
        seg_cached=str(cache['segment'].item());m_cached=int(cache['member'].item())
    expected_ix=np.concatenate((ds['splits']['train'],ds['splits']['validation']))
    if orig_sha!=co.sha256(ck_path) or orig_sha!=state['candidate_checkpoint_sha256']:
        raise RuntimeError('65Z-A cache checkpoint differs from 65Y model: '+name)
    if seg_cached!=seg or m_cached!=m or not np.array_equal(idx,expected_ix):
        raise RuntimeError('65Z-A cache index/segment mismatch: '+name)
    if target.shape!=(len(idx),len(ds['ip_coordinates']),6) or not np.isfinite(target).all():
        raise RuntimeError('65Z-A cache tensor shape mismatch')
    return state,target,idx


def validation_loss(model,branch,ip,LE,S,ix,w,dev,batch=8):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for start in range(0,len(ix),batch):
            ids=ix[start:start+batch]
            predle=model.strain_operator(branch[ids].to(dev),ip)
            preds=model.stress_operator(branch[ids].to(dev),ip)
            score=supervised(predle,preds,LE[ids].to(dev),S[ids].to(dev),w)
            total+=float(score.item())*len(ids);count+=len(ids)
    return total/max(count,1)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--project',default='.')
    a.add_argument('--arm',required=True,choices=['constitutive','combined'])
    a.add_argument('--members',type=int,default=5)
    a.add_argument('--epochs',type=int,default=100)
    a.add_argument('--patience',type=int,default=30)
    a.add_argument('--lr',type=float,default=8e-6)
    a.add_argument('--batch-size',type=int,default=4)
    a.add_argument('--anchor',type=float,default=0.05)
    a.add_argument('--lambda-const',type=float,default=0.005)
    a.add_argument('--lambda-kin',type=float,default=0.002)
    a.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    a.add_argument('--check-only',action='store_true')
    args=a.parse_args()
    if not(1<=args.members<=5 and args.epochs>=1 and args.patience>=1 and args.batch_size>=1
           and 0<args.lr<=1e-3 and args.anchor>=0 and 0<=args.lambda_const<=1
           and 0<=args.lambda_kin<=1):raise ValueError('Invalid hyperparameters')
    if args.arm=='constitutive' and args.lambda_const==0:raise ValueError('constitutive arm must have positive lambda-const')
    if args.arm=='combined' and (args.lambda_kin==0 or args.lambda_const==0):raise ValueError('combined arm requires both physics losses')
    root=Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    worst=co.reference_audit_gate(root)
    adir=root/'results'/'65Z_nonlinear_mechanics'/'A_predicted_kinematics'
    if not (adir/'65Z_A_summary.json').is_file():raise FileNotFoundError('Run 65Z-A first')
    audit=json.loads((adir/'65Z_A_summary.json').read_text())
    device=co.device_from_option(args.device)
    plans=[]
    basedir=root/'results'/'65Y_middle_mechanics'/'B_mechanics_finetune'
    for seg in co.SEGMENTS:
        ds=co.segment_data(root,seg)
        for m in range(1,args.members+1):
            path=basedir/seg/('member_%02d'%m)/'best_v4_7region.pt'
            if not path.is_file():raise FileNotFoundError(str(path))
            gate,target,ix=verify_cache(audit,root,seg,m,path,ds)
            if args.arm=='combined' and not gate['kinematic_loss_permitted']:
                raise RuntimeError('Combined kinematic training BLOCKED: '+seg+' member '+str(m)+
                    ' failed predicted-U gradient gate. Run constitutive arm only.')
            plans.append((seg,m,path,gate))
    if args.check_only:
        print('65Z-B CHECK PASS: arm=%s models=%d validated reference logU %.5f%%, device=%s'%(
            args.arm,len(plans),worst,device))
        print('65V and original 65Y model checkpoints will remain unchanged')
        return
    out=root/'results'/'65Z_nonlinear_mechanics'/('B_'+args.arm)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output protected. Manually delete ONLY:\n'+str(out)+'\nThen rerun.')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    weights=torch.tensor(co.WEIGHTS,dtype=torch.float32,device=device)
    records=[]
    for seg,m,path,gate in plans:
        random.seed(65000+101*m+1000*co.SEGMENTS.index(seg))
        np.random.seed(65000+101*m+1000*co.SEGMENTS.index(seg))
        torch.manual_seed(65000+101*m+1000*co.SEGMENTS.index(seg))
        ds=co.segment_data(root,seg)
        material_stiffness=torch.tensor(
            co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][:,:3]),
            dtype=torch.float32)
        ck=co.load_checkpoint(path)
        model=co.make_model(ck,device)
        for p in model.parameters():p.requires_grad_(False)
        for p in list(model.strain_operator.parameters())+list(model.stress_operator.parameters()):p.requires_grad_(True)
        baseline_model=copy.deepcopy(model).eval()
        for p in baseline_model.parameters():p.requires_grad_(False)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        bnorm,_,ipnorm=co.scaled_inputs(ck,ds)
        bt=torch.tensor(bnorm,dtype=torch.float32)
        ip=torch.tensor(ipnorm,dtype=torch.float32,device=device)
        lem=np.asarray(ck['LE_mean'],dtype=np.float32);les=np.asarray(ck['LE_std'],dtype=np.float32)
        sm=np.asarray(ck['S_mean'],dtype=np.float32);ss=np.asarray(ck['S_std'],dtype=np.float32)
        LET=torch.tensor((ds['LE']-lem)/les,dtype=torch.float32)
        ST=torch.tensor((ds['S']-sm)/ss,dtype=torch.float32)
        ctx=tuple(torch.tensor(x,dtype=torch.float32,device=device).reshape(1,1,6)
                  for x in (lem,les,sm,ss))
        _,K,ix=verify_cache(audit,root,seg,m,path,ds)
        # Cache is train+validation only, never test; index map here makes this explicit.
        mapping={int(j):i for i,j in enumerate(ix)}
        kt=torch.tensor(np.array(K,copy=True),dtype=torch.float32)
        train=ds['splits']['train'];validation=ds['splits']['validation']
        loader=DataLoader(TensorDataset(torch.tensor(train,dtype=torch.int64)),
                          batch_size=args.batch_size,shuffle=True,drop_last=False)
        opt=torch.optim.Adam(list(model.strain_operator.parameters())+list(model.stress_operator.parameters()),
                             lr=args.lr,weight_decay=1e-6)
        baseline_val=validation_loss(model,bt,ip,LET,ST,validation,weights,device)
        best_val=baseline_val;best_epoch=0;best=copy.deepcopy(initial);stale=0;history=[]
        print('65Z-B %s %-11s member %d baseline val %.7f'%(args.arm,seg,m,baseline_val),flush=True)
        for ep in range(1,args.epochs+1):
            model.train();running=0.;count=0
            for (case_indices,) in loader:
                ids=case_indices.numpy().astype(int)
                branch=bt[ids].to(device)
                le=LET[ids].to(device);s=ST[ids].to(device)
                with torch.no_grad():
                    anchor_le=baseline_model.strain_operator(branch,ip)
                    anchor_s=baseline_model.stress_operator(branch,ip)
                pred_le=model.strain_operator(branch,ip)
                pred_s=model.stress_operator(branch,ip)
                fit=supervised(pred_le,pred_s,le,s,weights)
                anchor=(pred_le-anchor_le).square().mean()+(pred_s-anchor_s).square().mean()
                if args.arm=='combined':
                    kk=kt[[mapping[int(i)] for i in ids]].to(device)
                else:kk=None
                stiff_batch=material_stiffness[ids].to(device)
                cpen,kpen=physics_loss(pred_le,pred_s,kk,ctx,stiff_batch,args.arm)
                objective=fit+args.anchor*anchor+args.lambda_const*cpen
                if args.arm=='combined':objective=objective+args.lambda_kin*kpen
                if not torch.isfinite(objective):raise RuntimeError('Non-finite optimization loss')
                opt.zero_grad(set_to_none=True);objective.backward()
                torch.nn.utils.clip_grad_norm_(list(model.strain_operator.parameters())+
                                               list(model.stress_operator.parameters()),1.0)
                opt.step()
                running+=float(objective.item())*len(ids);count+=len(ids)
            val=validation_loss(model,bt,ip,LET,ST,validation,weights,device)
            history.append({'epoch':ep,'train_total_loss':running/max(count,1),
                            'validation_supervised_loss':val})
            if val<best_val-1e-8:
                best_val=val;best_epoch=ep;stale=0
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if ep==1 or ep%20==0:
                print(' epoch %3d train %.6f val %.6f best %.6f'%(ep,running/max(count,1),val,best_val),flush=True)
            if stale>=args.patience:break
        for name,tensor in initial.items():
            if name.startswith(co.FREEZE) and not torch.equal(tensor,best[name]):
                raise RuntimeError('Frozen head changed unexpectedly: '+name)
        member_folder=out/seg/('member_%02d'%m)
        member_folder.mkdir(parents=True,exist_ok=True)
        newck=dict(ck);newck['model_state_dict']=best
        newck['65Z_physics_finetune']={'arm':args.arm,'source_65Y_sha256':co.sha256(path),
            'source_path':str(path),'segment':seg,'member':m,
            'trained_heads':['strain_operator','stress_operator'],
            'frozen_heads':['displacement_operator','force_coefficient_head'],
            'train_only':True,'validation_selected_epoch':best_epoch,
            'test_data_used':False,'reference_65W_data_used':False,
            'kinematic_loss_used':args.arm=='combined',
            'constitutive_loss_used':True,'logarithmic_strain_convention':'material logU engineering shear',
            'physics_weights':{'constitutive':args.lambda_const,
                               'kinematic':args.lambda_kin if args.arm=='combined' else 0.},
            'baseline_validation_loss':baseline_val,'chosen_validation_loss':best_val}
        torch.save(newck,member_folder/'best_v4_7region.pt')
        pd.DataFrame(history).to_csv(member_folder/'training_history.csv',index=False)
        row={'region':seg,'member':m,'arm':args.arm,'baseline_val_supervised':baseline_val,
             'selected_val_supervised':best_val,'improvement_pct':
             100*(baseline_val-best_val)/max(baseline_val,1e-12),
             'selected_epoch':best_epoch,'mechanics_changed':best_epoch>0,
             'frozen_U_force_verified':True,'kinematic_gate_pass':gate['kinematic_loss_permitted']}
        records.append(row)
        print('  selected epoch %d: validation improvement %.3f%%'%(best_epoch,row['improvement_pct']),flush=True)
        del model,baseline_model
        if device.type=='mps':torch.mps.empty_cache()
    pd.DataFrame(records).to_csv(out/'65Z_B_validation_training_summary.csv',index=False)
    manifest={'step':'65Z-B','arm':args.arm,'models':len(plans),
      'source':'65Y-B mechanics-only checkpoints','train_and_validation_only':True,
      'test_split_unused_in_this_experiment':True,'prior_65Y_test_set_already_seen':True,
      'reference_65W_middle_not_used_for_training':True,
      'source_65Y_checkpoints_preserved':True,
      'displacement_and_force_heads_frozen_and_equal':True,
      'kinematic_gate_enforced':args.arm=='combined',
      'reference_logU_max_error_pct':worst,
      'physics_shear':'Engineering shear; no factor-of-2 applied to Abaqus LE12/13/23',
      'constitutive_relation':'per-training-case C(E1,E2,G12) from original 64J branch inputs; soft engineering-shear constitutive penalty, not exact finite-deformation constitutive update',
      'epochs_requested':args.epochs,'learning_rate':args.lr,
      'lambda_const':args.lambda_const,'lambda_kin':args.lambda_kin if args.arm=='combined' else 0.,
      'warning':'OFFLINE candidates only; no validated changes to 65V coupling or 65W reference.'}
    (out/'65Z_B_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('65Z-B FINISHED:',out)

if __name__=='__main__':main()
