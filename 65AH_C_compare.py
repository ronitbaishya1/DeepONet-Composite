#!/usr/bin/env python3
"""C: fair DEV diagnostics: original 65AA, prior 65AG, joint shared, joint LGNO.

Only the previously examined 64J VALIDATION set is used here; NOT independent.
Do NOT copy any 65AH weights into live 65V seven-region coupling.

python3 65AH_C_compare.py --project . --members 1
"""
import argparse,json,importlib
from pathlib import Path
import numpy as np,pandas as pd,torch
ah=importlib.import_module('65AH_common')


def add_diagnostics(values,st):
 U,LE,S=values
 ds=st['ds']
 grad=ah.aa.gradient_numpy(U,ds)
 ln=np.stack([ah.co.log_material_strain_eng(u,ds['element_node_indices'],ds['B_center']) for u in U])
 C=ah.co.orthotropic_stiffness_batch_numpy(ds['branch_inputs'][st['_ids'],:3])
 derived=np.einsum('bij,bej->bei',C,LE,optimize=True)
 ans=dict(kinematic_LE22_disagreement_pct=ah.co.relative_l2(LE[...,1],ln[...,1]),
          kinematic_LE_all_disagreement_pct=ah.co.relative_l2(LE,ln),
          approx_constitutive_S22_disagreement_pct=ah.co.relative_l2(S[...,1],derived[...,1]),
          approx_constitutive_all_disagreement_pct=ah.co.relative_l2(S,derived),
          raw_gradient_rms=float(np.sqrt(np.mean(grad**2))))
 return ans


def prior_65ag(root,reg,m,ids,device,baselineU):
 """Prior 65AG is an optional baseline. It must never affect 65AH training."""
 try:
  ag=importlib.import_module('65AG_common')
  st=ag.attach_st(ag.member_sources(root,reg,m,device),reg,m)
  d=ag.make_dataset(st,ids,device)
  mod,_=ag.load_model(root,'uncoupled',st,device)
  L,S=ag.forward_numpy(mod,d,ag.f.strain_scales(st['ds']),ag.f.stress_scales(st['ds']),device)
  return (baselineU,L,S),None
 except (ImportError,FileNotFoundError) as e:
  return None, '65AG unavailable: '+str(e)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.')
 p.add_argument('--members',type=int,default=1)
 p.add_argument('--batch-size',type=int,default=2)
 p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
 p.add_argument('--skip-65ag',action='store_true')
 p.add_argument('--check-only',action='store_true')
 a=p.parse_args();root=Path(a.project).expanduser().resolve();device=ah.co.device_from_option(a.device)
 if not 1<=a.members<=5:raise ValueError('Members must be 1 through 5')
 for region in ah.SEGMENTS:
  for m in range(1,a.members+1):
   st=ah.load_source(root,region,m,device)
   for arm in ah.ARMS:
    if not ah.ck_path(root,arm,region,m).is_file():
     raise FileNotFoundError('Train arm '+arm+' before comparison: '+str(ah.ck_path(root,arm,region,m)))
 if a.check_only:
  print('65AH-C CHECK PASS: source and both arms available');return
 out=ah.output_dir(root,'C_development_comparison')
 allrows=[];decisions=[];warnings=[]
 for reg in ah.SEGMENTS:
  for m in range(1,a.members+1):
   st=ah.load_source(root,reg,m,device);ds=st['ds'];ids=ds['splits']['validation'];st['_ids']=ids
   b=ah.baseline_numpy(st,ids,device,a.batch_size)
   results={'65AA_frozen_source':b}
   if not a.skip_65ag:
    prev,warning=prior_65ag(root,reg,m,ids,device,b[0])
    if prev is not None:results['65AG_uncoupled']=prev
    elif warning:warnings.append(reg+' '+warning)
   for arm in ah.ARMS:
    net,info=ah.load_candidate(root,st,device,arm)
    U,L,S=ah.predict_numpy(net,st['t'],ids,a.batch_size)
    if not ah.source_force_equal(net,st):raise RuntimeError('Force head changed in '+arm)
    berr=ah.boundary_error(U,st,ids,device)
    if berr>3e-5:raise RuntimeError('Hard interfaces changed in '+arm+': '+str(berr))
    results['65AH_'+arm]=(U,L,S)
    print('65AH-C',reg,arm,'U2/LE22/S22',*(round(v,3) for v in (ah.metrics((U,L,S),ah.reference(st,ids))[n] for n in ('U2_pct','LE22_pct','S22_pct'))),flush=True)
   vals={}
   for name,pred in results.items():
    v=ah.metrics(pred,ah.reference(st,ids))
    v.update(add_diagnostics(pred,st))
    vals[name]=v
    allrows.append(dict(segment=reg,member=m,model=name,cases=len(ids),**v))
   base=vals['65AA_frozen_source'];shared=vals['65AH_shared'];local=vals['65AH_local']
   prior=vals.get('65AG_uncoupled')
   s22best=min(base['S22_pct'],shared['S22_pct'],prior['S22_pct'] if prior else float('inf'))
   le22best=min(base['LE22_pct'],shared['LE22_pct'],prior['LE22_pct'] if prior else float('inf'))
   # Require genuine gain against all comparable data-driven baselines; not just
   # LE and S, but U and interface safety too.
   safe=bool(local['U2_pct']<=base['U2_pct']*1.05+.03 and
        local['LE11_pct']<=base['LE11_pct']*1.15+.25 and
        local['S11_pct']<=base['S11_pct']*1.15+.25 and
        local['S12_pct']<=base['S12_pct']*1.15+.25)
   gain=bool(local['S22_pct']<=s22best*.95 and local['LE22_pct']<=le22best*.95)
   wins_shared=bool(local['S22_pct']<=shared['S22_pct']*.95)
   decision=dict(segment=reg,member=m,source_U2_pct=base['U2_pct'],
       shared_U2_pct=shared['U2_pct'],local_U2_pct=local['U2_pct'],
       source_LE22_pct=base['LE22_pct'],shared_LE22_pct=shared['LE22_pct'],
       local_LE22_pct=local['LE22_pct'],
       prior_65AG_S22_pct=prior['S22_pct'] if prior else None,
       source_S22_pct=base['S22_pct'],shared_S22_pct=shared['S22_pct'],local_S22_pct=local['S22_pct'],
       local_kinematic_LE22_gap_pct=local['kinematic_LE22_disagreement_pct'],
       local_constitutive_S22_gap_pct=local['approx_constitutive_S22_disagreement_pct'],
       stress_and_strain_gain_vs_all_5pct=gain,local_beats_shared_S22_by_5pct=wins_shared,
       displacement_and_other_mechanics_safe=safe,
       hard_interface_and_force_head_preserved=True,
       preliminary_gate_passed=bool(gain and wins_shared and safe and prior is not None))
   decisions.append(decision)
 pd.DataFrame(allrows).to_csv(out/'65AH_C_all_validation_metrics.csv',index=False)
 pd.DataFrame(decisions).to_csv(out/'65AH_C_decisions.csv',index=False)
 summary=dict(step='65AH-C',architecture=ah.ARCH,regions=ah.SEGMENTS,members=a.members,
   validation='REPEATED ORIGINAL 64J development-validation, NOT independent',
   training='original 64J train, internal fitting/tuning split',
   tested='Direct joint U(3 nodes), LE(6 IP), S(6 IP) vs shared-only ablation and baseline',
   preserved='Original force PCA checkpoint frozen and prescribed interface values unchanged',
   limitations=['No enforced kinematic compatibility: LE may differ from log(F.T F)/2',
                'No exact finite-strain constitutive stress update','No equilibrium/coupled solver evaluation',
                'Original validation has been seen many times; fresh FEM cases necessary'],
   original_old_test_and_65W_unused=True,
   decisions=decisions,all_preliminary_gates_passed=all(d['preliminary_gate_passed'] for d in decisions),warnings=warnings,
   action='DO NOT deploy on seven-region solver; collect independent Abaqus cases')
 (out/'65AH_C_summary.json').write_text(json.dumps(summary,indent=2))
 print('65AH-C DONE',out)

if __name__=='__main__':main()
