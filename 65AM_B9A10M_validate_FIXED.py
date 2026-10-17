#!/usr/bin/env python3
"""65AM-B9A10-M: read-only 24-case incremental orthotropic stress audit.

python3 65AM_B9A10M_validate.py --project . --stage selftest
python3 65AM_B9A10M_validate.py --project . --stage check
python3 65AM_B9A10M_validate.py --project . --stage validate
python3 65AM_B9A10M_validate_FIXED.py --project . --stage finish  # continue a completed run after JSON error

No fitting, neural operator training or Broyden changes. Compares two pre-specified
objective-increment candidates against unchanged B9A10-L and K controls.
"""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
JOBS=['B9A9_%s%02d'%(r,i) for r in ('L','R') for i in range(1,13)]
OUT=Path('results/65AM_B9_coupled_feasibility/B9A10M_incremental_validation')
SCHEMES=('start','end')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load_module(p,name):
    if not p.is_file():raise FileNotFoundError(str(p))
    spec=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def paths(root):
    names={
      'history':root/'B9A10M_exports/B9A10M_history_fields.npz',
      'meta':root/'B9A10M_exports/B9A10M_windows_summary.json',
      'h':root/'B9A10H_exports/B9A10H_independent_fields.npz',
      'hm':root/'B9A10H_exports/B9A10H_windows_summary.json',
      'l':root/'B9A10L_exports/B9A10L_LE_fields.npz',
      'lm':root/'B9A10L_exports/B9A10L_windows_summary.json',
      'plan':root/'B9A9_transfer_to_windows/B9A9_case_plan.npz',
      'anchor':root/'B9_coupled_anchor/B9_65V_anchor.npz',
      'kc':root/'results/65AM_B9_coupled_feasibility/B9A10K_differentiable_proxy/B9A10K_calibrated_coefficient.json',
      'ls':root/'results/65AM_B9_coupled_feasibility/B9A10L_constitutive_validation/B9A10L_summary.json',
      'lc':root/'results/65AM_B9_coupled_feasibility/B9A10L_constitutive_validation/B9A10L_cases.csv',
      'hcode':root/'65AM_B9A10H_mac_weakform_validation.py',
      'lcode':root/'65AM_B9A10L_torch_force.py',
      'kcode':root/'65AM_B9A10K_hourglass_operator.py',
      'mcode':root/'65AM_B9A10M_incremental_force.py'
    }
    for k,p in names.items():
        if not p.is_file():raise FileNotFoundError('Missing '+k+': '+str(p))
    meta=json.loads(names['meta'].read_text());hm=json.loads(names['hm'].read_text())
    lm=json.loads(names['lm'].read_text());ls=json.loads(names['ls'].read_text())
    kc=json.loads(names['kc'].read_text())
    if len(meta['jobs'])!=24 or set(x['job'] for x in meta['jobs'])!=set(JOBS):
        raise RuntimeError('History manifest does not contain 24 original cases')
    if sha(names['history'])!=meta['export_sha256'] or sha(names['hm'])!=meta['source_B9A10H_summary_sha256']:
        raise RuntimeError('History provenance mismatch')
    if sha(names['h'])!=hm['output_sha256'] or sha(names['l'])!=lm['file_sha256']:
        raise RuntimeError('H/L source hashes have changed')
    hd={x['job']:x['odb_sha256'] for x in hm['cases']}
    if any(hd.get(x['job'])!=x['odb_sha256'] for x in meta['jobs']):
        raise RuntimeError('History exported from different ODBs')
    if ls['step']!='65AM-B9A10-L' or not ls['autograd']['autograd_pass']:
        raise RuntimeError('L control not verified')
    if float(kc['beta'])<=0:raise RuntimeError('Non-positive frozen K beta')
    return names,float(kc['beta'])


def pct(a,ref):
    return 100*float(np.linalg.norm(np.asarray(a)))/max(float(np.linalg.norm(np.asarray(ref))),1.e-15)


def json_numpy_default(value):
    """Serialize scalar NumPy results while rejecting unrelated unsupported objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Not JSON serializable: %s" % type(value).__name__)


def writecsv(p,rows):
    if not rows:raise ValueError('No rows')
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def make_data(root,check_only):
    p,beta=paths(root)
    H=load_module(p['hcode'],'m_h_control');L=load_module(p['lcode'],'m_l_control')
    M=load_module(p['mcode'],'m_inc')
    with np.load(p['history'],allow_pickle=False) as nz,np.load(p['h'],allow_pickle=False) as hz,\
         np.load(p['l'],allow_pickle=False) as lz:
        for job in JOBS:
            for s in ('times','U','S','LE','Q','node_labels','element_labels'):
                if job+'_'+s not in nz:raise RuntimeError('Missing '+job+'_'+s)
            source=H.make_case(hz,job)
            u=nz[job+'_U'];times=nz[job+'_times'];s=nz[job+'_S'];le=nz[job+'_LE'];q=nz[job+'_Q']
            n=len(times)
            if n<3 or u.shape!=(n,1008,3) or s.shape!=(n,728,6) or le.shape!=(n,728,6) or q.shape!=(n,728,3,3):
                raise RuntimeError('Unexpected history dimensions '+job)
            if abs(times[0])>1e-12 or abs(times[-1]-1)>1e-6 or not np.all(np.diff(times)>0):
                raise RuntimeError('History time not monotonic/end 1 '+job)
            if not np.array_equal(nz[job+'_node_labels'],source['node_labels']) or \
               not np.array_equal(nz[job+'_element_labels'],source['element_labels']):
                raise RuntimeError('Labels mismatch '+job)
            if np.max(np.abs(u[-1]-source['U']))>2e-6 or \
               np.max(np.abs(s[-1]-source['S']))>1e-4 or \
               np.max(np.abs(le[-1]-lz[job+'_LE']))>1e-4:
                raise RuntimeError('Final H/L vs M ODB field mismatch '+job)
            if np.max(np.abs(q[-1]-source['Q']))>3e-4:
                raise RuntimeError('Orientation history inconsistent with H final '+job)
            if not np.isfinite(u).all() or not np.isfinite(s).all():raise RuntimeError('Nonfinite history '+job)
    print('B9A10M CHECK PASS | 24 unchanged ODB histories | final H and L fields match | beta=%.10g'%beta)
    if check_only:return
    output=root/OUT
    if not output.is_dir():raise RuntimeError('Please create in Finder first: '+str(output))
    outputs=['B9A10M_cases.csv','B9A10M_modes.csv','B9A10M_summary.json',
             'B9A10M_force_comparison.png','B9A10M_stress_history.png','B9A10M_gradients.json']
    if any((output/n).exists() for n in outputs):
        raise RuntimeError('Existing B9A10M results protected; move old outputs manually for another run')
    old={}
    with p['lc'].open(newline='') as f:
        for row in csv.DictReader(f):old[row['job']]=row
    if set(old)!=set(JOBS):raise RuntimeError('L reference has incorrect case rows')
    rows=[];modes=[];time_curves={}
    with np.load(p['history'],allow_pickle=False) as nz,np.load(p['h'],allow_pickle=False) as hz,\
         np.load(p['plan'],allow_pickle=False) as plan,np.load(p['anchor'],allow_pickle=False) as anchor:
        for region in ('L','R'):
            j0='B9A9_'+region+'01';case0=H.make_case(hz,j0)
            model=L.ConstitutiveC3D8R(case0['node_labels'],case0['X'],case0['connectivity'],
                     beta,p['kcode'],device='cpu',dtype=torch.float64)
            inc=M.IncrementalC3D8R(model)
            material_plan=plan['x_inner_left' if region=='L' else 'x_inner_right']
            for idx in range(12):
                job='B9A9_%s%02d'%(region,idx+1)
                d=H.make_case(hz,job)
                if np.max(np.abs(d['X']-case0['X']))>2e-5:raise RuntimeError('Mesh changed '+job)
                uh=torch.as_tensor(nz[job+'_U'],dtype=torch.float64)
                mat=torch.as_tensor(material_plan[idx,:3],dtype=torch.float64)
                b=H.boundary(d['X'])
                denom=max(np.linalg.norm(d['RF'][b]),1e-15)
                def mval(F):
                    err=np.asarray(F)-d['RF']
                    return (100*np.linalg.norm(err)/denom,
                            100*np.linalg.norm(err[b])/denom,
                            100*np.linalg.norm(err[~b])/denom)
                with torch.no_grad():
                    fs,fh,ff=model.parts(uh[-1],mat,stress_kind='cauchy')
                    force0=(fs+fh).numpy();S0=ff['S_local'].numpy()
                oldval=float(old[job]['cauchy_force_all_pct'])
                if abs(mval(force0)[0]-oldval)>0.002:
                    raise RuntimeError('B9A10-L positive control mismatch '+job)
                result=dict(job=job,region='inner_left' if region=='L' else 'inner_right',
                            condition=['material']*4+['interface']*4+['combined']*4,
                            frame_count=len(uh),
                            L_total_force_all_pct=mval(force0)[0],
                            L_total_stress_pct=pct(S0-d['S'],d['S']))
                result['condition']=result['condition'][idx]
                result['K_oracle_force_pct']=float(old[job]['K_oracle_stress_plus_HG_force_pct'])
                for scheme in SCHEMES:
                    with torch.no_grad():
                        fstress,fh,fields=inc.parts(uh,mat,scheme)
                        force=(fstress+fh).numpy()
                    allerr,bnderr,freeerr=mval(force)
                    s_pred=fields['S_history'].numpy()
                    s_true=nz[job+'_S'][1:]
                    result[scheme+'_force_all_pct']=allerr
                    result[scheme+'_force_interface_pct']=bnderr
                    result[scheme+'_force_free_pct']=freeerr
                    result[scheme+'_stress_final_pct']=pct(s_pred[-1]-s_true[-1],s_true[-1])
                    result[scheme+'_stress_history_pct']=pct(s_pred-s_true,s_true)
                    for face in H.FACES[region]:
                        q,scale=H.pca_force(d,force,anchor,face)
                        true,_=H.pca_force(d,d['RF'],anchor,face)
                        for n,v in enumerate((q-true)/scale,1):
                            modes.append(dict(job=job,region=result['region'],method=scheme,
                                              interface=face,mode=n,abs_error_on_65V_scale=abs(float(v)),
                                              signed_error_on_65V_scale=float(v)))
                    if idx==0:
                        error=[pct(s_pred[t-1]-s_true[t-1],s_true[t-1]) for t in range(1,len(uh))]
                        time_curves[(region,scheme)]=(nz[job+'_times'][1:].copy(),error)
                rows.append(result)
                print('CASE',job,'L %.4f'%result['L_total_force_all_pct'],
                      'increment-start %.4f'%result['start_force_all_pct'],
                      'increment-end %.4f'%result['end_force_all_pct'])
    # PyTorch gradient test on the actual mesh, one case, without fitting or B5.
    with np.load(p['history'],allow_pickle=False) as nz,np.load(p['h'],allow_pickle=False) as hz,\
         np.load(p['plan'],allow_pickle=False) as plan:
        d=H.make_case(hz,'B9A9_L01')
        model=L.ConstitutiveC3D8R(d['node_labels'],d['X'],d['connectivity'],beta,p['kcode'])
        inc=M.IncrementalC3D8R(model)
        hist=torch.tensor(nz['B9A9_L01_U'],dtype=torch.float64,requires_grad=True)
        mat=torch.tensor(plan['x_inner_left'][0,:3],dtype=torch.float64,requires_grad=True)
        bm=torch.tensor(H.boundary(d['X']))
        f=inc(hist,mat,'start')
        free_loss=f[~bm].square().mean()
        gH,gM=torch.autograd.grad(free_loss,(hist,mat))
        autograd_pass=bool(torch.isfinite(gH).all() and torch.isfinite(gM).all() and
                           torch.linalg.norm(gH)>0 and torch.linalg.norm(gM)>0)
        if not autograd_pass:raise RuntimeError('PyTorch history autograd failed')
        grad=dict(device='cpu',dtype='float64',autograd_pass=autograd_pass,
                 loss_free=float(free_loss.detach()),
                 grad_norm_history=float(torch.linalg.norm(gH)),
                 grad_norm_material=float(torch.linalg.norm(gM)),
                 not_neural_operator_training=True)
    aggregates={}
    for reg in ('inner_left','inner_right'):
        selected=[r for r in rows if r['region']==reg]
        groups={}
        for m in ('L_total','start','end'):
            col=m+'_force_all_pct'
            groups[m]=float(np.sqrt(np.mean([x[col]**2 for x in selected])))
        groups['K_oracle']=float(np.sqrt(np.mean([x['K_oracle_force_pct']**2 for x in selected])))
        groups['start_stress_final']=float(np.sqrt(np.mean([x['start_stress_final_pct']**2 for x in selected])))
        groups['end_stress_final']=float(np.sqrt(np.mean([x['end_stress_final_pct']**2 for x in selected])))
        groups['start_improves_cases']=int(sum(bool(x['start_force_all_pct']<x['L_total_force_all_pct']) for x in selected))
        groups['end_improves_cases']=int(sum(bool(x['end_force_all_pct']<x['L_total_force_all_pct']) for x in selected))
        aggregates[reg]=groups
    summary=dict(step='B9A10M',case_count=24,method='Objective incremental log-relative-deformation research candidates',
      schemes={'start':'previous corotating material frame','end':'current corotating material frame'},
      frozen_beta=beta,all_stresses_from_U_and_material=True,
      abaqus_S_LE_Q_used_only_for_diagnostics=True,no_fitting=True,
      results=aggregates,
      exploratory_comparison_only=True,
      any_pooled_improvement_both_regions=any(all(aggregates[r][m]<aggregates[r]['L_total'] for r in aggregates)
                                          for m in SCHEMES),
      physics_loss_ready=False,approved_for_B5=False,approved_for_coupling=False,
      limitations=['Not the exact Abaqus/Standard incremental orthotropic stress-rate algorithm.',
                   'Stored Abaqus U history is used here; NO-predicted displacement histories not validated.',
                   'Model selection using these 24 cases would require fresh holdout validation.',
                   'K beta remains mesh-specific empirical approximation.'],
      input_sha256={key:sha(p[key]) for key in ('history','h','plan','kc','lc')})
    writecsv(output/outputs[0],rows);writecsv(output/outputs[1],modes)
    (output/outputs[2]).write_text(json.dumps(summary,indent=2,default=json_numpy_default))
    (output/outputs[5]).write_text(json.dumps(grad,indent=2,default=json_numpy_default))
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for axis,reg in zip(ax,('inner_left','inner_right')):
        vals=[x for x in rows if x['region']==reg]
        for key,label in (('L_total_force_all_pct','L total-strain'),('start_force_all_pct','M incremental START'),
                          ('end_force_all_pct','M incremental END')):
            axis.plot(range(1,13),[v[key] for v in vals],marker='o',label=label)
        axis.set_xlabel('Independent B9A9 case');axis.set_ylabel('Nodal-force error / boundary RF (%)')
        axis.set_title(reg);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/outputs[3],dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for axis,reg in zip(ax,('L','R')):
        for method in SCHEMES:
            times,values=time_curves[(reg,method)]
            axis.plot(times,values,marker='.',label=method)
        axis.set_title('inner_left' if reg=='L' else 'inner_right')
        axis.set_xlabel('Loading step time');axis.set_ylabel('Stress error vs Abaqus S (%)')
        axis.grid(alpha=.2);axis.legend()
    fig.tight_layout();fig.savefig(output/outputs[4],dpi=160);plt.close(fig)
    print('B9A10M COMPLETE | exploratory incremental comparison | no fitted parameters | output:',output)
    for reg,dat in aggregates.items():print(reg,dat)



def finish_existing_run(root):
    """Complete a run interrupted AFTER its two CSV files were written.

    Reads the already-computed 24-case and PCA results without modifying them.
    Recomputes just the first history for each region and one autograd test.
    Never deletes, renames, or overwrites existing output files.
    """
    # The standard 24-case provenance and final-state checks stay mandatory.
    make_data(root,check_only=True)
    p,beta=paths(root)
    out=root/OUT
    if not out.is_dir():
        raise RuntimeError('Missing result folder: '+str(out))
    case_path=out/'B9A10M_cases.csv'
    modes_path=out/'B9A10M_modes.csv'
    missing=[str(q) for q in (case_path,modes_path) if not q.is_file()]
    if missing:
        raise RuntimeError('Completed CSV files not found. Run --stage validate if no CSV files exist: '+', '.join(missing))
    planned=['B9A10M_summary.json','B9A10M_force_comparison.png',
             'B9A10M_stress_history.png','B9A10M_gradients.json']
    already=[x for x in planned if (out/x).exists()]
    if already:
        raise RuntimeError('The output files already exist; protected from replacement: '+', '.join(already))
    with case_path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    with modes_path.open(newline='') as f:
        modes=list(csv.DictReader(f))
    if len(rows)!=24 or {r['job'] for r in rows}!=set(JOBS) or len(set(r['job'] for r in rows))!=24:
        raise RuntimeError('Existing cases CSV does not contain the 24 distinct B9A9 jobs')
    if not modes or set(r['job'] for r in modes)!=set(JOBS) or set(r['method'] for r in modes)!=set(SCHEMES):
        raise RuntimeError('Existing PCA mode CSV is incomplete')
    for r in rows:
        expected='inner_left' if r['job'][5]=='L' else 'inner_right'
        if r['region']!=expected:
            raise RuntimeError('Incorrect region in '+r['job'])
        if not all(np.isfinite(float(r[k])) for k in (
            'L_total_force_all_pct','start_force_all_pct','end_force_all_pct',
            'K_oracle_force_pct','start_stress_final_pct','end_stress_final_pct')):
            raise RuntimeError('Nonfinite CSV metrics for '+r['job'])
    H=load_module(p['hcode'],'m_h_finish')
    L=load_module(p['lcode'],'m_l_finish')
    M=load_module(p['mcode'],'m_inc_finish')
    time_curves={}
    with np.load(p['history'],allow_pickle=False) as nz,np.load(p['h'],allow_pickle=False) as hz,\
         np.load(p['plan'],allow_pickle=False) as plan:
        for region in ('L','R'):
            job='B9A9_'+region+'01'
            d=H.make_case(hz,job)
            model=L.ConstitutiveC3D8R(d['node_labels'],d['X'],d['connectivity'],
                   beta,p['kcode'],device='cpu',dtype=torch.float64)
            inc=M.IncrementalC3D8R(model)
            uh=torch.as_tensor(nz[job+'_U'],dtype=torch.float64)
            mat=torch.as_tensor(plan['x_inner_left' if region=='L' else 'x_inner_right'][0,:3],dtype=torch.float64)
            true_s=nz[job+'_S'][1:]
            for scheme in SCHEMES:
                with torch.no_grad():
                    _,_,fields=inc.parts(uh,mat,scheme)
                    predicted=fields['S_history'].numpy()
                errors=[pct(predicted[i]-true_s[i],true_s[i]) for i in range(len(true_s))]
                time_curves[(region,scheme)]=(nz[job+'_times'][1:].copy(),errors)
        # Reproduce the single autograd check from the original full run.
        d=H.make_case(hz,'B9A9_L01')
        model=L.ConstitutiveC3D8R(d['node_labels'],d['X'],d['connectivity'],beta,p['kcode'])
        inc=M.IncrementalC3D8R(model)
        hist=torch.tensor(nz['B9A9_L01_U'],dtype=torch.float64,requires_grad=True)
        mat=torch.tensor(plan['x_inner_left'][0,:3],dtype=torch.float64,requires_grad=True)
        bm=torch.tensor(H.boundary(d['X']))
        f=inc(hist,mat,'start')
        free_loss=f[~bm].square().mean()
        gH,gM=torch.autograd.grad(free_loss,(hist,mat))
        autograd_pass=bool(torch.isfinite(gH).all() and torch.isfinite(gM).all() and
                           torch.linalg.norm(gH)>0 and torch.linalg.norm(gM)>0)
        if not autograd_pass:
            raise RuntimeError('PyTorch history autograd failed')
        grad=dict(device='cpu',dtype='float64',autograd_pass=autograd_pass,
                  loss_free=float(free_loss.detach()),
                  grad_norm_history=float(torch.linalg.norm(gH)),
                  grad_norm_material=float(torch.linalg.norm(gM)),
                  not_neural_operator_training=True)
    aggregates={}
    for reg in ('inner_left','inner_right'):
        selected=[r for r in rows if r['region']==reg]
        groups={}
        for m in ('L_total','start','end'):
            groups[m]=float(np.sqrt(np.mean([float(x[m+'_force_all_pct'])**2 for x in selected])))
        groups['K_oracle']=float(np.sqrt(np.mean([float(x['K_oracle_force_pct'])**2 for x in selected])))
        groups['start_stress_final']=float(np.sqrt(np.mean([float(x['start_stress_final_pct'])**2 for x in selected])))
        groups['end_stress_final']=float(np.sqrt(np.mean([float(x['end_stress_final_pct'])**2 for x in selected])))
        groups['start_improves_cases']=int(sum(float(x['start_force_all_pct'])<float(x['L_total_force_all_pct']) for x in selected))
        groups['end_improves_cases']=int(sum(float(x['end_force_all_pct'])<float(x['L_total_force_all_pct']) for x in selected))
        aggregates[reg]=groups
    summary=dict(step='B9A10M',case_count=24,method='Objective incremental log-relative-deformation research candidates',
      schemes={'start':'previous corotating material frame','end':'current corotating material frame'},
      frozen_beta=beta,all_stresses_from_U_and_material=True,
      abaqus_S_LE_Q_used_only_for_diagnostics=True,no_fitting=True,
      results=aggregates,exploratory_comparison_only=True,
      any_pooled_improvement_both_regions=any(all(aggregates[r][m]<aggregates[r]['L_total'] for r in aggregates)
                                          for m in SCHEMES),
      physics_loss_ready=False,approved_for_B5=False,approved_for_coupling=False,
      limitations=['Not the exact Abaqus/Standard incremental orthotropic stress-rate algorithm.',
                   'Stored Abaqus U history is used here; NO-predicted displacement histories not validated.',
                   'Model selection using these 24 cases would require fresh holdout validation.',
                   'K beta remains mesh-specific empirical approximation.'],
      input_sha256={key:sha(p[key]) for key in ('history','h','plan','kc','lc')},
      finished_from_existing_case_and_mode_csv=True,
      resumed_case_csv_sha256=sha(case_path),resumed_mode_csv_sha256=sha(modes_path))
    # Draw figures before writing output; no pre-existing files are modified.
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for axis,reg in zip(ax,('inner_left','inner_right')):
        vals=sorted((x for x in rows if x['region']==reg),key=lambda x:x['job'])
        for key,label in (('L_total_force_all_pct','L total-strain'),
                          ('start_force_all_pct','M incremental START'),
                          ('end_force_all_pct','M incremental END')):
            axis.plot(range(1,13),[float(v[key]) for v in vals],marker='o',label=label)
        axis.set_xlabel('Independent B9A9 case')
        axis.set_ylabel('Nodal-force error / boundary RF (%)')
        axis.set_title(reg);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'B9A10M_force_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for axis,reg in zip(ax,('L','R')):
        for method in SCHEMES:
            times,values=time_curves[(reg,method)]
            axis.plot(times,values,marker='.',label=method)
        axis.set_title('inner_left' if reg=='L' else 'inner_right')
        axis.set_xlabel('Loading step time')
        axis.set_ylabel('Stress error vs Abaqus S (%)')
        axis.grid(alpha=.2);axis.legend()
    fig.tight_layout();fig.savefig(out/'B9A10M_stress_history.png',dpi=160);plt.close(fig)
    (out/'B9A10M_summary.json').write_text(json.dumps(summary,indent=2,default=json_numpy_default))
    (out/'B9A10M_gradients.json').write_text(json.dumps(grad,indent=2,default=json_numpy_default))
    print('B9A10M FINISH PASS | 24 existing case CSV rows reused; no old files overwritten')
    for reg,values in aggregates.items():
        print(reg,values)
    print('Results:',out)


def selftest(root):
    M=load_module(root/'65AM_B9A10M_incremental_force.py','m_inc_test')
    M.selftest()
    print('B9A10M VALIDATION SELFTEST PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.')
    ap.add_argument('--stage',choices=('selftest','check','validate','finish'),required=True)
    a=ap.parse_args();root=Path(a.project).resolve()
    if a.stage=='selftest':selftest(root)
    elif a.stage=='finish':finish_existing_run(root)
    else:make_data(root,check_only=a.stage=='check')
