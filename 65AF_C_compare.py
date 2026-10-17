#!/usr/bin/env python3
"""65AF-C: compare local strain operators, C@LE and train-selected stress blend.

Training cases select alpha; previously-used 64J validation reports errors.
No 64J held-out TEST and no 65W reference used. No coupled model deployment.

python3 65AF_C_compare.py --project . --members 1
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import importlib
f=importlib.import_module('65AF_common')
ALPHAS=(0.,.25,.5,.75,1.)


def choose_alpha(head,derived,ref,baseS):
    choices=[]
    for alpha in ALPHAS:
        s=(1-alpha)*head+alpha*derived
        score=dict(S11_pct=f.relative(s[...,0],ref[...,0]),
                   S22_pct=f.relative(s[...,1],ref[...,1]),
                   S12_pct=f.relative(s[...,3],ref[...,3]))
        safe=(score['S11_pct']<=baseS['S11_pct']*1.10+.5 and
              score['S12_pct']<=baseS['S12_pct']*1.10+.5)
        objective=score['S22_pct']+.1*score['S11_pct']+.1*score['S12_pct']
        choices.append((alpha,safe,objective,score))
    allowed=[x for x in choices if x[1]]
    if not allowed:raise RuntimeError('alpha=0 baseline should always be safe')
    chosen=min(allowed,key=lambda z:z[2])[0]
    return chosen,choices


def metrics(L,S,ds,ids,C):
    result=f.derived_metrics(L,S,C,ds,ids)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5:raise ValueError('Members 1..5')
    dev=f.ac.aa.co.device_from_option(a.device)
    for region in f.REGIONS:
        for member in range(1,a.members+1):
            f.cases(root,region,member,dev,need_predictions=False)
            for arm in f.ARMS:
                path=f.checkpoint_path(root,arm,region,member)
                if not path.is_file():raise FileNotFoundError('Run 65AF-B '+arm+' first: '+str(path))
    if a.check_only:
        print('65AF-C CHECK PASS: LGNO + pointwise + frozen V4 checkpoints ready')
        return
    out=f.safe_output(root,'C_validation_comparison')
    allmetrics=[];tuning=[];decisions=[]
    for region in f.REGIONS:
        for member in range(1,a.members+1):
            st=f.cases(root,region,member,dev)
            ds=st['ds'];tr=ds['splits']['train'];val=ds['splits']['validation']
            vF,vdu,vL,vS,vg=f.build_features(st,val,dev)
            tF,tdu,tL,tS,tg=f.build_features(st,tr,dev)
            Ct=f.stiffness(ds,tr);Cv=f.stiffness(ds,val)
            sc=f.strain_scales(ds)
            headval=metrics(vL,vS,ds,val,Cv)
            headtrain=metrics(tL,tS,ds,tr,Ct)
            # Original baselines, no new training.
            variants={'original_learned_head':(vL,vS,None),
                      'C_from_original_learned_LE':(vL,f.C_LE(vL,Cv),None),
                      'C_from_65AC_LGNO_gradient':(f.ac.logstrain_from_gradient(vg),
                         f.C_LE(f.ac.logstrain_from_gradient(vg),Cv),None)}
            armdata={}
            for arm in f.ARMS:
                checkpoint=f.checkpoint_path(root,arm,region,member)
                net,ck=f.load_model(checkpoint,arm,region,member,st['source_sha'],st['lgno_sha'],dev)
                Lt=f.predict(net,tF,tdu,tL,sc,dev)
                Lv=f.predict(net,vF,vdu,vL,sc,dev)
                St=f.C_LE(Lt,Ct);Sv=f.C_LE(Lv,Cv)
                variants['65AF_'+arm+'_C_LE']=(Lv,Sv,None)
                alpha,grid=choose_alpha(tS,St,ds['S'][tr],headtrain)
                for candidate,safe,objective,m in grid:
                    tuning.append(dict(region=region,member=member,arm=arm,
                        alpha=candidate,train_cases=len(tr),safe=bool(safe),train_objective=objective,**m))
                blend=(1-alpha)*vS+alpha*Sv
                variants['65AF_'+arm+'_train_selected_blend']=(Lv,blend,alpha)
                armdata[arm]={'alpha':alpha,'Lv':Lv,'Sv':Sv,'blend':blend,
                              'Ltrain':Lt,'Strain':St,'epoch':ck['selected_epoch']}
                del net
            for name,(LE,S,alpha) in variants.items():
                row=dict(region=region,member=member,model=name,validation_cases=len(val),
                         selected_alpha=alpha,**metrics(LE,S,ds,val,Cv))
                allmetrics.append(row)
                print('65AF-C %-11s %-35s S22 %6.2f%% LE22 %6.2f%%'%(
                    region,name,row['S22_pct'],row['LE22_pct']),flush=True)
            baseline_s=headval['S22_pct']
            lm=metrics(armdata['lgno']['Lv'],armdata['lgno']['blend'],ds,val,Cv)
            pm=metrics(armdata['pointwise']['Lv'],armdata['pointwise']['blend'],ds,val,Cv)
            improve=100*(baseline_s-lm['S22_pct'])/baseline_s
            decisions.append(dict(region=region,member=member,lgno_alpha=armdata['lgno']['alpha'],
                pointwise_alpha=armdata['pointwise']['alpha'],
                head_S22_pct=baseline_s,lgno_S22_pct=lm['S22_pct'],
                pointwise_S22_pct=pm['S22_pct'],
                lgno_LE22_pct=lm['LE22_pct'],original_LE22_pct=headval['LE22_pct'],
                lgno_S22_gain_pct=improve,
                lgno_beats_original_S22_by_5pct=bool(lm['S22_pct']<=.95*baseline_s),
                lgno_beats_pointwise_S22_by_5pct=bool(lm['S22_pct']<=.95*pm['S22_pct']),
                LE22_improved=bool(lm['LE22_pct']<headval['LE22_pct']),
                LE11_safe=bool(lm['LE11_pct']<=headval['LE11_pct']*1.1+.3),
                LE12_safe=bool(lm['LE12_pct']<=headval['LE12_pct']*1.1+.3),
                S11_safe=bool(lm['S11_pct']<=headval['S11_pct']*1.1+.5),
                S12_safe=bool(lm['S12_pct']<=headval['S12_pct']*1.1+.5),
                U_and_interface_force_unchanged=True))
    for row in decisions:
        row['preliminary_gate_passed']=bool(all(row[k] for k in (
            'lgno_beats_original_S22_by_5pct','lgno_beats_pointwise_S22_by_5pct',
            'LE22_improved','LE11_safe','LE12_safe','S11_safe','S12_safe')))
    df=pd.DataFrame(allmetrics)
    df.to_csv(out/'65AF_C_all_validation_metrics.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'65AF_C_train_blend_selection.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'65AF_C_decisions.csv',index=False)
    summary=dict(step='65AF-C',validation_cohort='Original 64J validation, reused repeatedly; NOT independent',
        members=a.members,decisions=decisions,
        all_preliminary_gates_passed=bool(all(x['preliminary_gate_passed'] for x in decisions)),
        source_U_LE_S_force_heads_unchanged=True,
        target_reference_at_inference='None: only predicted features, original learned LE/S, material properties',
        trained='Original 64J train only; checkpoint selection on 64J validation',
        alpha_selected='Original 64J train only; validation used for evaluation',
        original_test_and_65W_untouched=True,
        caution='Local strain independent of displacement and lacks compatibility guarantee. Stress computed by approximate C@Hencky; not a verified FE-NO solution.',
        next='Fresh independent Abaqus materials and physical compatibility/interface force validation before deployment.')
    (out/'65AF_C_summary.json').write_text(json.dumps(summary,indent=2))
    print('65AF-C DONE',out)

if __name__=='__main__':main()
