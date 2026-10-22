#!/usr/bin/env python3
"""B9A11P-A: read-only feasibility audit of frozen B5 operator for inverse work.
This is NOT a fully coupled seven-region FE–NO solver or a digital twin.
Uses existing 65AM-B5 original 44-fit/12-tune region datasets and checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
import numpy as np

REGIONS = ('inner_left','inner_right')


def selftest():
    x=np.array([[4.,7.,8.],[5.,6.,9.]],float)
    assert (x.max(axis=0)-x.min(axis=0)>0).all()
    assert hashlib.sha256(b'a').hexdigest().startswith('ca978')
    print('B9A11P-A SELFTEST PASS')


def audit(root, variant='centered', loss='grad', output=None):
    import torch
    import sys
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    c=importlib.import_module('65AM_B5_common')
    b=c.b
    from src.hard_interface_compatibility_7region import apply_hard_compatibility_7region
    dev=torch.device('cpu')
    state={}; allok=True
    for region in REGIONS:
        d,mu,sd=c.prepare(root,region)
        net,ck=c.load_model(root,variant,loss,region,d,mu,sd,dev)
        ds=d['ds']; phys=np.asarray(ds['branch_inputs'],float)
        if phys.ndim!=2 or phys.shape[1]!=12:
            raise RuntimeError('Expected 12 branch features = E1 E2 G12 + 9 prescribed PCA q')
        fit,tune=d['fit'],d['tune']
        if len(fit)!=44 or len(tune)!=12:
            raise RuntimeError('TRAIN split mismatch')
        lo=phys[fit,:3].min(axis=0); hi=phys[fit,:3].max(axis=0)
        if np.any(hi<=lo) or np.any(lo<=0): raise RuntimeError('Invalid original FIT material ranges')
        # Test whether the prescribed interface lifting changes when just material changes.
        ctx=b.CO.interface_context(root,ds,dev)
        rid=int(tune[0]); q=phys[rid].copy(); q2=q.copy(); q2[:3]*=1.04
        inp=torch.as_tensor(np.stack((q,q2)),dtype=torch.float32)
        zeros=torch.zeros((2,len(ds['coordinates']),3),dtype=torch.float32)
        with torch.no_grad():
            lifts=apply_hard_compatibility_7region(zeros,inp,ctx).cpu().numpy()
        lift_material_dependence=float(np.max(np.abs(lifts[0]-lifts[1])))
        # Compare saved and rebuilt actual tune lift, verifying branch mapping.
        lookup={int(v):i for i,v in enumerate(np.r_[fit,tune])}
        stored=d['lift'][lookup[rid]]
        saved_lift_gap=float(np.max(np.abs(lifts[0]-stored)))
        if saved_lift_gap>2e-5: raise RuntimeError('Lifting mismatch; cannot safely run inverse on '+region)
        if lift_material_dependence>2e-6:
            raise RuntimeError('Material-dependent interface lifting in '+region+'. New forward wrapper required.')
        in_bounds=np.all((phys[tune,:3]>=lo)&(phys[tune,:3]<=hi),axis=1)
        entry=dict(region=region,variant=variant,loss=loss,branch_dimension=int(phys.shape[1]),
            conditioning_features=['E1','E2','G12']+['prescribed_PCA_interface_'+str(j+1) for j in range(9)],
            fitted_cases=44,internal_tune_cases=12,training_range_MPa={k:[float(a),float(z)] for k,a,z in zip(('E1','E2','G12'),lo,hi)},
            in_fit_material_bounds_tune_cases=int(in_bounds.sum()),rank=int(d['rank']),
            sensor_boundary_known_from_input=True,material_change_lift_max_mm=lift_material_dependence,
            reconstruct_lift_gap_mm=saved_lift_gap,checkpoint=str(c.ck_path(root,variant,loss,region)),
            checkpoint_best_tune_objective=float(ck.get('best_tune_objective',float('nan'))),
            full_beam_coupled_inference_supported_by_this_script=False,
            nine_property_input_supported=False)
        state[region]=entry
        print('CHECK PASS',region,'train 44 | tune 12 | in-FIT-range',int(in_bounds.sum()),
              '| fixed lift material dependence',lift_material_dependence)
    saved=root/'results/65AM_B9_coupled_feasibility/A_saved_state/B9_summary.json'
    saved_report=None
    if saved.is_file():
        saved_report=json.loads(saved.read_text())
    summary=dict(stage='B9A11P-A',read_only=True,original_checkpoints_untouched=True,
        models=state,
        coupling_status='NOT validated as a new full-beam B5 material-conditioned coupled solver',
        anchored_65V_B5_report=saved_report and {k:saved_report.get(k) for k in (
            'verified_65V_RMS','not_a_converged_coupled_B5_solution','B5_replacement_approved')},
        limitations=['B5 trained on two INNER subdomains, not entire composite',
            'Prescribed interface/PCA coefficients are known test conditions, not sensor-inferred boundary states',
            'Only E1 E2 G12 are independently varied',
            'Original 12 tune cases are internal development split, not an untouched external test',
            'No physical strain-gauge measurements or online sensor data'])
    if output is not None:
        output.mkdir(parents=True,exist_ok=False)
        (output/'B9A11P_A_readiness.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print('Saved:',output/'B9A11P_A_readiness.json')
    else:print('CHECK COMPLETE: no files changed. NOT a fully coupled solver.')
    return summary


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--stage',required=True,choices=('selftest','check','run'))
    a.add_argument('--project',default='.')
    a.add_argument('--variant',choices=('centered','standardized'),default='centered')
    a.add_argument('--loss',choices=('U','grad'),default='grad')
    a.add_argument('--tag',default='pilot01')
    args=a.parse_args()
    if args.stage=='selftest':selftest();return
    root=Path(args.project).expanduser().resolve()
    out=(root/'results'/'65AM_B9A11P_conditional_inverse'/args.tag/'audit') if args.stage=='run' else None
    audit(root,args.variant,args.loss,out)
if __name__=='__main__':main()