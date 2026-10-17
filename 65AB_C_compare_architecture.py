#!/usr/bin/env python3
"""65AB-C: fair original 65Z vs bubble-nodal vs bubble-gradient comparison.

Optional comparator: previously trained 65AA gradient model, if present.
Uses original 64J VALIDATION cases ONLY (not test or 65W middle).
Every candidate is reconstructed from its saved source 65Z checkpoint.
"""
import argparse,json,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np,pandas as pd,torch
spec=spec_from_file_location('step65ab',Path(__file__).with_name('65AB_common.py'))
ab=module_from_spec(spec);sys.modules[spec.name]=ab;spec.loader.exec_module(ab)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--members',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    if not 1<=a.members<=5:raise ValueError('--members must be 1..5')
    device=ab.aa.co.device_from_option(a.device)
    for arm in ('nodal','gradient'):
        manifest=root/'results'/ab.ROOT_NAME/('B_'+arm)/'65AB_B_manifest.json'
        if not manifest.is_file():raise FileNotFoundError('Complete 65AB-B '+arm+' first: '+str(manifest))
        for seg in ab.SEGMENTS:
            for m in range(1,a.members+1):
                fp=ab.candidate_dir(root,arm,seg,m)/'bubble_residual.pt'
                if not fp.is_file():raise FileNotFoundError(str(fp))
    if a.check_only:
        print('65AB-C CHECK PASS: two training arms, candidates, original 65Z models available')
        return
    out=root/'results'/ab.ROOT_NAME/'C_architecture_comparison'
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing comparison folder protected. Manually delete ONLY:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    rows=[];groups=[]
    for seg in ab.SEGMENTS:
        for m in range(1,a.members+1):
            ds,ck,base,ctx,source=ab.aa.init_case(root,seg,m,device)
            va=ds['splits']['validation'];truth=ds['U'][va];LE_ref=ds['LE'][va]
            reference=ab.aa.predict_numpy(base,ctx,va)
            pred={'65Z_baseline':reference}
            interface_ids=ab.boundary_local_indices(ctx,ds)
            for arm in ('nodal','gradient'):
                op,meta=ab.load_candidate(root,arm,seg,m,ck,device)
                ubase,u1=ab.predictions_from_candidate(root,seg,m,ds,ck,ctx,op,va)
                if np.max(np.abs(ubase-reference))>2e-6:
                    raise RuntimeError('65Z source prediction unexpectedly changed')
                dmax=float(np.max(np.abs((u1-reference)[:,interface_ids,:])))
                if dmax>2e-6:raise RuntimeError('Prescribed-interface displacement altered: %.4g'%dmax)
                pred['65AB_'+arm]=u1
            aa_file=root/'results'/ab.aa.ROOT_NAME/'B_gradient'/seg/('member_%02d'%m)/'best_v4_7region.pt'
            if aa_file.is_file():
                try:
                    aa_ck=ab.aa.co.load_checkpoint(aa_file)
                    if aa_ck.get('65AA_gradient_experiment',{}).get('arm')=='gradient':
                        old=ab.aa.co.make_model(aa_ck,device).eval()
                        pred['65AA_gradient']=ab.aa.predict_numpy(old,ctx,va)
                except (RuntimeError,KeyError,ValueError) as exc:
                    print('NOTE: optional 65AA comparison skipped:',exc,flush=True)
            model_stats={}
            for label,u in pred.items():
                metrics=ab.aa.errors(u,truth,ds,LE_ref=LE_ref)
                metrics.update(dict(segment=seg,member=m,model=label,validation_cases=len(va),
                                    known_interface_max_change_mm=float(np.max(np.abs((u-reference)[:,interface_ids,:])))))
                rows.append(metrics);model_stats[label]=metrics
            base_g=model_stats['65Z_baseline']['gradient_rel_l2_pct']
            cn=model_stats['65AB_nodal']['gradient_rel_l2_pct']
            cg=model_stats['65AB_gradient']['gradient_rel_l2_pct']
            u2_old=model_stats['65Z_baseline']['U2_rel_l2_pct']
            u2_new=model_stats['65AB_gradient']['U2_rel_l2_pct']
            le22_old=model_stats['65Z_baseline']['logU_LE22_rel_l2_pct']
            le22_new=model_stats['65AB_gradient']['logU_LE22_rel_l2_pct']
            g65aa=model_stats.get('65AA_gradient',{}).get('gradient_rel_l2_pct')
            gain_b=100*(base_g-cg)/max(base_g,1e-12)
            gain_n=100*(cn-cg)/max(cn,1e-12)
            gain_old=None if g65aa is None else 100*(g65aa-cg)/max(g65aa,1e-12)
            group=dict(segment=seg,member=m,
                bubble_gradient_gain_over_65Z_pct=gain_b,
                bubble_gradient_gain_over_bubble_nodal_pct=gain_n,
                bubble_gradient_gain_over_65AA_pct=gain_old,
                U2_baseline_pct=u2_old,U2_bubble_gradient_pct=u2_new,
                logU_LE22_baseline_pct=le22_old,logU_LE22_bubble_gradient_pct=le22_new,
                same_interface_displacements=True,
                U2_safe=bool(u2_new<=1.05*u2_old+.02),
                gradient_better_than_control_5pct=bool(gain_n>=5),
                gradient_better_than_original_5pct=bool(gain_b>=5),
                nonlinear_LE22_improved=bool(le22_new<le22_old),
                preliminary_gate_passed=bool(u2_new<=1.05*u2_old+.02 and gain_n>=5 and gain_b>=5 and le22_new<le22_old))
            groups.append(group)
            print('65AB-C %s member%d: grad baseline %.3f%% nodal %.3f%% bubble %.3f%%; U2 %.3f%%'%(
                seg,m,base_g,cn,cg,u2_new),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'65AB_C_all_validation_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'65AB_C_decisions.csv',index=False)
    summary=dict(step='65AB-C',architecture=ab.ARCHITECTURE,
        cohort='Original 64J development-validation ONLY; not fresh holdout',
        members=a.members,all_preliminary_gates_passed=bool(all(x['preliminary_gate_passed'] for x in groups)),
        decisions=groups,models_compared=sorted(df['model'].unique().tolist()),
        important='65AB residual candidates are NOT drop-in replacements for the original V4 model. They require the 65AB inference adapter.',
        next='If candidate succeeds, repeat full ensemble with validation-only selection; then new independent Abaqus material and coupled force verification before any deployment.',
        original_65V_65W_solution_unchanged=True,
        original_64J_test_and_middle_FEM_untouched=True)
    (out/'65AB_C_summary.json').write_text(json.dumps(summary,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for seg in ab.SEGMENTS:
            dd=df[df.segment==seg]
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.bar(np.arange(len(dd)),dd.gradient_rel_l2_pct.to_numpy())
            ax.set_xticks(np.arange(len(dd)));ax.set_xticklabels(dd.model.to_list(),rotation=25,ha='right')
            ax.set_ylabel('Displacement gradient rel L2 error (%)')
            ax.set_title('65AB gradient comparison — '+seg)
            fig.tight_layout();fig.savefig(out/('65AB_C_'+seg+'_gradients.png'),dpi=160);plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; wrote all numeric outputs')
    print('65AB-C FINISHED:',out)
if __name__=='__main__':main()
