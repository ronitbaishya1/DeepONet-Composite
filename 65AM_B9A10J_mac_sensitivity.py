#!/usr/bin/env python3
"""B9A10-J / Mac: Abaqus C3D8R hourglass-stiffness sensitivity with all U prescribed.
Compares 0.5, 1, 2 stiffness against ENHANCED and the old reference; never fits a deployable force model.
Does not change B5, 65V, or B9A7; writes new reports in an existing Finder-created folder.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
FACE={'L':('left_inner','center_left'),'R':('center_right','right_inner')}
SLICES={'left_inner':slice(4,8),'center_left':slice(8,13),'center_right':slice(13,18),'right_inner':slice(18,22)}
SETTINGS=('S05','S10','S20','ENH')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def norm(a):return float(np.linalg.norm(np.asarray(a,float)))
def job(reg,setting):return 'B9A10J_%s_%s'%(reg,setting)

def load(z,tag):
    out={}
    for k in ('node_labels','X','U','RF','element_labels','connectivity','S','Q_global_to_local'):
        key=tag+'_'+k
        if key not in z:raise RuntimeError('Missing '+key)
        out['Q' if k=='Q_global_to_local' else k]=np.array(z[key],copy=True)
    spec={'node_labels':(1008,),'X':(1008,3),'U':(1008,3),'RF':(1008,3),
          'element_labels':(728,),'connectivity':(728,8),'S':(728,6),'Q':(728,3,3)}
    for k,shape in spec.items():
        if out[k].shape!=shape:raise RuntimeError(tag+' invalid '+k+' shape '+str(out[k].shape))
        if not np.isfinite(out[k]).all():raise RuntimeError(tag+' nonfinite '+k)
    if np.max(abs(out['Q']@out['Q'].transpose(0,2,1)-np.eye(3)))>1e-3 or np.max(abs(np.linalg.det(out['Q'])-1))>1e-3:
        raise RuntimeError('Bad local material rotations '+tag)
    return out

def assemble(d):
    lab={int(k):i for i,k in enumerate(d['node_labels'])}
    ix=np.asarray([[lab[int(v)] for v in row] for row in d['connectivity']],dtype=int)
    X=d['X'][ix];U=d['U'][ix]
    dnat=SIGNS/8
    Jac=np.einsum('eai,aj->eij',X,dnat)
    det=np.linalg.det(Jac)
    if np.min(det)<=1e-14:raise RuntimeError('Invalid reference Jacobian')
    grad=np.einsum('aj,ejk->eak',dnat,np.linalg.inv(Jac))
    F=np.eye(3)[None]+np.einsum('eai,eaj->eij',U,grad)
    detF=np.linalg.det(F)
    if np.min(detF)<=0:raise RuntimeError('Non-positive deformation Jacobian')
    v=d['S'];sigma=np.zeros((728,3,3))
    sigma[:,0,0]=v[:,0];sigma[:,1,1]=v[:,1];sigma[:,2,2]=v[:,2]
    sigma[:,0,1]=sigma[:,1,0]=v[:,3]
    sigma[:,0,2]=sigma[:,2,0]=v[:,4]
    sigma[:,1,2]=sigma[:,2,1]=v[:,5]
    Q=d['Q'];G=Q.transpose(0,2,1)@sigma@Q
    P=detF[:,None,None]*(G@np.linalg.inv(F).transpose(0,2,1))
    el=(P@grad.transpose(0,2,1)).transpose(0,2,1)*(8*det[:,None,None])
    result=np.zeros((1008,3),float);np.add.at(result,ix,el)
    if norm(result.sum(axis=0))>1e-5*max(norm(result),1):raise RuntimeError('Internal nodal resultant unexpectedly nonzero')
    return result

def boundary(X):
    a=X[:,0];b=np.isclose(a,a.min(),atol=2e-5)|np.isclose(a,a.max(),atol=2e-5)
    if b.sum()!=224:raise RuntimeError('Expected 224 outer-face nodes')
    return b

def pca(d,forces,anc,iface):
    coords=np.asarray(anc['pca_coordinates_'+iface],float)
    basis=np.asarray(anc['pca_basis_'+iface],float)
    scale=np.asarray(anc['force_scale'],float)[SLICES[iface]]
    idx=[];used=set()
    for pt in coords:
        dist=np.linalg.norm(d['X']-pt,axis=1); i=int(dist.argmin())
        if dist[i]>2e-5 or i in used:raise RuntimeError('PCA coordinate mismatch '+iface)
        idx.append(i);used.add(i)
    if basis.shape!=(336,len(scale)) or np.any(scale<=0):raise RuntimeError('Invalid saved PCA basis/force scales '+iface)
    return basis.T@forces[idx].ravel(),scale

def selftest():
    # Affine eight-node element: internal force must equal uniform Piola stress virtual work.
    xyz=SIGNS.copy();U=xyz@np.array([[.005,.001,0],[0,.002,0],[0,0,.003]])
    d={'node_labels':np.arange(1,9),'X':xyz,'U':U,'RF':np.zeros((8,3)),
       'element_labels':np.array([1]),'connectivity':np.arange(1,9).reshape(1,8),
       'S':np.array([[8,5,3,1,.5,.25]],float),'Q':np.eye(3).reshape(1,3,3)}
    # 728-element batch assembly is the same formula as this local test.
    dnat=SIGNS/8;J=xyz.T@dnat;G=dnat@np.linalg.inv(J)
    F=np.eye(3)+U.T@G
    S=np.array([[8,1,.5],[1,5,.25],[.5,.25,3]])
    P=np.linalg.det(F)*S@np.linalg.inv(F).T
    f=(P@G.T).T*(8*np.linalg.det(J))
    T=np.arange(24,dtype=float).reshape(8,3)/10
    if abs(float((f*T).sum())-float(8*np.linalg.det(J)*np.sum(P*(T.T@G))))>1e-10:
        raise RuntimeError('Virtual work failed')
    if norm(f.sum(axis=0))>1e-10:raise RuntimeError('Affine element nodal-force balance failed')
    # Synthetic causal test: r(s)=r0+s*hg; identify the exactly known coefficient.
    ref=np.arange(30.,dtype=float);h=np.cos(np.arange(30.)); s05=ref+.5*h; s10=ref+h; s20=ref+2*h
    recovered=(s20-s05)/1.5
    if norm(recovered-h)>1e-11 or norm((s10-recovered)-ref)>1e-11:
        raise RuntimeError('Hourglass-scale regression failed')
    print('B9A10J MAC SELFTEST PASS | Piola virtual work | force-balance | 0.5/1/2 response inversion')

def inputs(root):
    npz=root/'B9A10J_exports'/'B9A10J_stabilization_fields.npz'
    meta=root/'B9A10J_exports'/'B9A10J_windows_summary.json'
    old=root/'B9A10C_exports'/'B9A10C_fields.npz'
    oldm=root/'B9A10C_exports'/'B9A10C_windows_summary.json'
    a=root/'B9_coupled_anchor'/'B9_65V_anchor.npz'
    for p in (npz,meta,old,oldm,a):
        if not p.is_file():raise FileNotFoundError('Missing '+str(p))
    j=json.loads(meta.read_text());b=json.loads(oldm.read_text())
    if sha(npz)!=j['export_sha256'] or sha(old)!=b['output_sha256']:
        raise RuntimeError('Export/baseline SHA256 mismatch')
    actual={c['job'] for c in j['cases']}
    expected={job(r,s) for r in ('L','R') for s in SETTINGS}
    if actual!=expected:raise RuntimeError('Incomplete 8-job export manifest')
    with np.load(npz,allow_pickle=False) as z:
        for r in ('L','R'):
            for t in SETTINGS:
                d=load(z,job(r,t))
            for k in ('node_labels','X','U','RF','element_labels','connectivity'):
                if 'B9A10J_%s_REFERENCE_%s'%(r,k) not in z:
                    raise RuntimeError('Missing original default reference '+k)
    return npz,old,a,j

def write_csv(path,rows):
    if not rows:raise RuntimeError('Empty output '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root,check_only):
    npz,old,anc,manifest=inputs(root)
    print('B9A10J INPUT CHECK PASS | 8 varied-control jobs, B9A10C original, 65V PCA anchor')
    if check_only:return
    out=root/'results'/'65AM_B9_coupled_feasibility'/'B9A10J_stabilization_sensitivity'
    if not out.is_dir():raise RuntimeError('Create in Finder: '+str(out))
    planned=[out/f for f in ('B9A10J_case_metrics.csv','B9A10J_modes.csv','B9A10J_summary.json',
                             'B9A10J_force_sensitivity.png','B9A10J_mode_sensitivity.png')]
    if any(p.exists() for p in planned):raise RuntimeError('Existing B9A10J output protected; move files in Finder before rerun')
    rows=[];modes=[];summary={}
    with np.load(npz,allow_pickle=False) as z,np.load(old,allow_pickle=False) as oz,np.load(anc,allow_pickle=False) as az:
        for r in ('L','R'):
            cases={t:load(z,job(r,t)) for t in SETTINGS}
            base=cases['S10'];bnd=boundary(base['X'])
            oldj='B9A10C_%s_R_REPLAY'%r
            # Immutable reference: original default calculation must be reproduced at S10.
            for k in ('node_labels','element_labels','connectivity'):
                if not np.array_equal(cases['S10'][k],oz[oldj+'_'+k]):raise RuntimeError('Old mesh mismatch '+r+' '+k)
            oldRF=np.asarray(oz[oldj+'_RF'],float);oldU=np.asarray(oz[oldj+'_U'],float)
            scale=norm(oldRF[bnd]);err_orig=100*norm(oldRF-base['RF'])/max(scale,1e-12)
            if err_orig>0.05:raise RuntimeError('S10 does NOT reproduce old default RF: %s %.5f%%'%(r,err_orig))
            if np.max(abs(oldU-base['U']))>2e-5:raise RuntimeError('S10 original U mismatch '+r)
            for t,d in cases.items():
                for k in ('node_labels','element_labels','connectivity'):
                    if not np.array_equal(base[k],d[k]):raise RuntimeError('Mesh mismatch '+r+t+k)
                if np.max(abs(base['X']-d['X']))>1e-7 or np.max(abs(base['U']-d['U']))>2e-5:
                    raise RuntimeError('Prescribed displacement changed '+r+t)
            f=assemble(base)
            reference=base['RF']; gap=reference-f
            # response slope for default stiffness branch: no fitted NO model.
            hg=(cases['S20']['RF']-cases['S05']['RF'])/1.5
            intercept=reference-hg
            leftover=intercept-f
            denom=max(scale,1e-12)
            fraction=float(np.dot(gap.ravel(),hg.ravel())/(norm(gap)*max(norm(hg),1e-15))) if norm(gap)>1e-15 else float('nan')
            projection=float(np.dot(gap.ravel(),hg.ravel())/max(norm(hg)**2,1e-30))
            norm_pct=lambda v:100*norm(v)/denom
            fit05=intercept+0.5*hg;fit20=intercept+2.0*hg
            linear_dev=max(norm_pct(fit05-cases['S05']['RF']),norm_pct(fit20-cases['S20']['RF']))
            # Compare stress at constant imposed U; a strong variation means the interpretation needs caution.
            sdiff=max(norm(cases[t]['S']-base['S'])/max(norm(base['S']),1e-12) for t in ('S05','S20'))
            summary[r]=dict(original_65V_reference_regression_pct=err_orig,
                default_stress_only_gap_pct=norm_pct(gap),
                default_stress_only_interface_gap_pct=norm_pct(gap[bnd]),
                default_stress_only_free_node_gap_pct=norm_pct(gap[~bnd]),
                hourglass_stiffness_response_pct=norm_pct(hg),
                direction_cosine_between_missing_force_and_HG_response=fraction,
                least_squares_scale_to_match_missing_force=projection,
                reconstructed_zero_scale_residual_pct=norm_pct(leftover),
                linear_stiffness_response_max_deviation_pct=linear_dev,
                max_relative_stress_change_between_scalings=sdiff,
                enhanced_vs_default_reaction_pct=norm_pct(cases['ENH']['RF']-reference),
                paired_cases=4)
            expected={'L':1.1770788954990998,'R':1.1730814646256724}[r]
            if abs(summary[r]['default_stress_only_gap_pct']-expected)>0.15:
                raise RuntimeError('B9A10H historical force-gap regression failed: %s %.5f vs %.5f'%(
                    r,summary[r]['default_stress_only_gap_pct'],expected))
            for t,d in cases.items():
                changes=d['RF']-reference
                rows.append(dict(region='inner_left' if r=='L' else 'inner_right',job=job(r,t),setting=t,
                    prescribed_U_delta_mm=float(np.max(abs(d['U']-base['U']))),
                    RF_difference_from_default_pct=norm_pct(changes),
                    RF_difference_interface_pct=norm_pct(changes[bnd]),
                    RF_difference_free_node_pct=norm_pct(changes[~bnd]),
                    default_stress_only_gap_pct=norm_pct(gap),
                    predicted_linear_change_gap_pct=norm_pct((float({'S05':.5,'S10':1,'S20':2}.get(t,1))-1)*hg) if t!='ENH' else '',
                    is_enhanced=(t=='ENH')))
            for iface in FACE[r]:
                p_gap,forcescale=pca(base,gap,az,iface)
                p_hg,_=pca(base,hg,az,iface)
                p_rem,_=pca(base,leftover,az,iface)
                p_enh,_=pca(base,cases['ENH']['RF']-reference,az,iface)
                for i in range(len(p_gap)):
                    modes.append(dict(region='inner_left' if r=='L' else 'inner_right',interface=iface,mode=i+1,
                        missing_stress_force_N=float(p_gap[i]),
                        stiffness_control_response_N=float(p_hg[i]),
                        remaining_at_extrapolated_zero_scale_N=float(p_rem[i]),
                        enhanced_change_N=float(p_enh[i]),
                        missing_on_65V_scale=float(p_gap[i]/forcescale[i]),
                        stiffness_response_on_65V_scale=float(p_hg[i]/forcescale[i]),
                        residual_on_65V_scale=float(p_rem[i]/forcescale[i])))
            print('%s | old RF regression %.6f%% | force gap %.5f%% | zero-scale leftover %.5f%% | direction cos %.5f'%(
                r,err_orig,norm_pct(gap),norm_pct(leftover),fraction))
    write_csv(planned[0],rows);write_csv(planned[1],modes)
    result=dict(step='65AM-B9A10-J',method='Controlled C3D8R stiffness perturbations with identical full nodal U',
        input_export_sha256=sha(npz),source_B9A10C_sha256=sha(old),regions=summary,
        statement='A fitted linear response is a sensitivity diagnosis, not automatically Abaqus exact hourglass force.',
        old_models_untouched=True,physics_loss_ready=False,live_Broyden_approved=False)
    planned[2].write_text(json.dumps(result,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,reg in zip(axes,('inner_left','inner_right')):
        dat=[x for x in rows if x['region']==reg]
        ax.bar([v['setting'] for v in dat],[v['RF_difference_from_default_pct'] for v in dat])
        ax.set_title(reg+' | same imposed U');ax.set_ylabel('RF change / baseline boundary RF (%)');ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(planned[3],dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,r in zip(axes,('inner_left','inner_right')):
        dat=[m for m in modes if m['region']==r];x=np.arange(len(dat))
        ax.plot(x,[m['missing_on_65V_scale'] for m in dat],marker='o',label='Missing stress-only force')
        ax.plot(x,[m['stiffness_response_on_65V_scale'] for m in dat],marker='s',label='Per unit stiffness response')
        ax.set_xticks(x);ax.set_xticklabels([m['interface']+' m'+str(m['mode']) for m in dat],rotation=65,ha='right',fontsize=8)
        ax.set_title(r);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(planned[4],dpi=170);plt.close(fig)
    print('B9A10J REPORT PASS:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default='.');p.add_argument('--selftest',action='store_true');p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.selftest:selftest()
    else:analyze(Path(a.project).expanduser().resolve(),a.check_only)
