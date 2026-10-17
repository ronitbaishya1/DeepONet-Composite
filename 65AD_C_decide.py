#!/usr/bin/env python3
"""65AD-C: jointly compare compatible gradients, displacement, and S22.

This is an OFFLINE gate, not a final or independent test.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import importlib
ad=importlib.import_module('65AD_common')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=Path(a.project).expanduser().resolve()
    base=root/'results'/ad.ROOT_NAME
    af=base/'A_compatible_projection'/'65AD_A_projection_metrics.csv'
    bf=base/'B_stress_audit'/'65AD_B_stress_comparison.csv'
    sf=base/'B_stress_audit'/'65AD_B_constitutive_sanity.csv'
    for f in (af,bf,sf):
        if not f.is_file():raise FileNotFoundError(str(f))
    if a.check_only:
        print('65AD-C CHECK PASS: compatible displacement and stress audits present')
        return
    A=pd.read_csv(af);B=pd.read_csv(bf);S=pd.read_csv(sf)
    rows=[]
    for seg in ad.SEGMENTS:
        # One member per region is current pilot.
        for m in sorted(A[A.segment==seg].member.unique()):
            aa=A[(A.segment==seg)&(A.member==m)].set_index('model')
            bs=B[(B.segment==seg)&(B.member==m)&(B.component=='S22')].set_index('model')
            sanity=S[(S.segment==seg)&(S.member==m)].iloc[0]
            src=aa.loc['65AA_source']; lg=aa.loc['65AC_LGNO_gradient_only']; proj=aa.loc['65AD_compatible_projection']
            original_s=float(bs.loc['65AA_learned_stress_head','rel_l2_pct'])
            projected_s=float(bs.loc['65AD_constitutive_from_compatible_U','rel_l2_pct'])
            g_gain=100*(1-proj.gradient_rel_l2_pct/max(src.gradient_rel_l2_pct,1e-12))
            s_gain=100*(1-projected_s/max(original_s,1e-12))
            u2_safe=bool(proj.U2_rel_l2_pct<=src.U2_rel_l2_pct*1.05+.02)
            g_safe=bool(g_gain>=5)
            constitutive_safe=bool(sanity.reference_constitutive_S22_within_2pct)
            stress_better=bool(s_gain>=5)
            interface_safe=bool(proj.max_interface_change_mm<=1e-8)
            passed=bool(u2_safe and g_safe and stress_better and constitutive_safe and interface_safe)
            rows.append(dict(segment=seg,member=int(m),
                original_gradient_pct=float(src.gradient_rel_l2_pct),
                LGNO_gradient_pct=float(lg.gradient_rel_l2_pct),
                projected_gradient_pct=float(proj.gradient_rel_l2_pct),
                projected_gradient_gain_vs_original_pct=g_gain,
                original_U2_pct=float(src.U2_rel_l2_pct),projected_U2_pct=float(proj.U2_rel_l2_pct),
                original_learned_S22_pct=original_s,projected_constitutive_S22_pct=projected_s,
                S22_gain_vs_learned_head_pct=s_gain,
                original_LE22_pct=float(src.LE22_rel_l2_pct),
                LGNO_LE22_pct=float(lg.LE22_rel_l2_pct),
                projected_LE22_pct=float(proj.LE22_rel_l2_pct),
                U2_safe=u2_safe,gradient_gain_5pct=g_safe,
                constitutive_reference_matches=constitutive_safe,
                interface_preserved=interface_safe,stress_S22_gain_5pct=stress_better,
                preliminary_gate_passed=passed))
    out=ad.safe_output(root,'C_decision')
    df=pd.DataFrame(rows);df.to_csv(out/'65AD_C_comparison.csv',index=False)
    summary={'step':'65AD-C','models_compared':'65AA original, 65AC LGNO, 65AD compatible projection, constitutive stress estimates',
        'independent_test':False,'original_test_and_65W_not_used':True,
        'all_preliminary_gates_passed':bool(df['preliminary_gate_passed'].all()),
        'result_rows':rows,
        'action':'If passed, plan independent Abaqus case and check coupling. If failed, do NOT deploy. Diagnose projection regularization/gradient integrability using only new development data.'}
    (out/'65AD_C_summary.json').write_text(json.dumps(summary,indent=2))
    print(df[['segment','projected_gradient_pct','projected_U2_pct','projected_LE22_pct','projected_constitutive_S22_pct','preliminary_gate_passed']].to_string(index=False))
    print('65AD-C DONE:',out)

if __name__=='__main__':main()
