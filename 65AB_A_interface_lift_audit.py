#!/usr/bin/env python3
"""65AB-A: Check the existing hard interfaces and the new zero-end bubble.

python3 65AB_A_interface_lift_audit.py --project . --members 1 --check-only
python3 65AB_A_interface_lift_audit.py --project . --members 1
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
    if a.members<1 or a.members>5:raise ValueError('Use --members 1..5')
    device=ab.aa.co.device_from_option(a.device)
    for seg in ab.SEGMENTS:
        for m in range(1,a.members+1):
            if not ab.aa.source_checkpoint(root,seg,m).is_file():
                raise FileNotFoundError(str(ab.aa.source_checkpoint(root,seg,m)))
        ds=ab.aa.load_data(root,seg)
        if len(ds['splits']['validation'])==0:raise RuntimeError('No validation cases')
    if a.check_only:
        print('65AB-A CHECK PASS: original 64J dataset, 65Z checkpoints, interface metadata available')
        return
    folder=root/'results'/ab.ROOT_NAME/'A_interface_lift_audit'
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Existing output protected. Manually delete ONLY:\n'+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seg in ab.SEGMENTS:
        for m in range(1,a.members+1):
            ds,ck,base,ctx,source=ab.aa.init_case(root,seg,m,device)
            ids=ds['splits']['validation']
            env,ends=ab.envelope_numpy(ds)
            boundary=ab.boundary_local_indices(ctx,ds)
            ubase=ab.aa.predict_numpy(base,ctx,ids,correct=True)
            uraw=ab.aa.predict_numpy(base,ctx,ids,correct=False)
            op=ab.make_operator(ck['branch_dim'],device,seed=65000+100*ab.SEGMENTS.index(seg)+m)
            op.eval()
            phi=torch.as_tensor(env,dtype=torch.float32,device=device)
            with torch.no_grad():
                out=[]
                for start in range(0,len(ids),4):
                    batch=ids[start:start+4]
                    ub=torch.as_tensor(np.array(ubase[start:start+len(batch)],copy=True),dtype=torch.float32,device=device)
                    pred=ab.corrected_from_residual(ub,op,ctx,batch,phi)
                    out.append(pred.cpu().numpy())
                new=np.concatenate(out)
            initerr=float(np.max(np.abs(ubase-new)))
            bderr=float(np.max(np.abs((ubase-new)[:,boundary,:])))
            if initerr>2e-7 or bderr>2e-7:raise RuntimeError('Zero initialization or interface protection failed')
            d0=ab.aa.errors(ubase,ds['U'][ids],ds)
            draw=ab.aa.errors(uraw,ds['U'][ids],ds)
            row=dict(segment=seg,member=m,validation_cases=len(ids),nodes=len(ds['coordinates']),
                     boundary_nodes=len(boundary),zero_envelope_nodes=len(ends),
                     envelope_min=float(env.min()),envelope_max=float(env.max()),
                     init_max_change_mm=initerr,interface_max_change_mm=bderr,
                     raw_U2_error_pct=draw['U2_rel_l2_pct'],
                     hard_corrected_U2_error_pct=d0['U2_rel_l2_pct'],
                     hard_corrected_gradient_error_pct=d0['gradient_rel_l2_pct'])
            rows.append(row)
            print('65AB-A %s member%02d: hard-corrected gradient %.4f%%, bubble initial change %.2e mm'%(
                seg,m,row['hard_corrected_gradient_error_pct'],initerr),flush=True)
    pd.DataFrame(rows).to_csv(folder/'65AB_A_interface_check.csv',index=False)
    summary=dict(step='65AB-A',architecture=ab.ARCHITECTURE,cohort='64J validation only',
                 members=a.members,all_zero_initialization_passed=True,
                 all_prescribed_interfaces_unchanged=True,rows=rows,
                 note='The bubble changes only displacement inside the NO region. All source checkpoint weights stay untouched.')
    (folder/'65AB_A_summary.json').write_text(json.dumps(summary,indent=2))
    print('65AB-A PASSED:',folder)
if __name__=='__main__':main()
