#!/usr/bin/env python3
"""65AG: shared LOCAL strain-stress refinement, with constitutive ablation.

Imports the ACTUAL working 65AF/65AC/65AA/65Z project helpers.
No original checkpoints or FE-NO coupling scripts are changed.

The physics term is APPROXIMATE: Abaqus-reported Cauchy stress S ~ C@HenckyLE
for the studied small-deformation cases. LE and S here are separately learned
and NOT guaranteed kinematically compatible or exactly constitutively equal.
"""
from pathlib import Path
import importlib, hashlib, json
import numpy as np
import torch
from torch import nn

f=importlib.import_module('65AF_common')
ac=f.ac
REGIONS=f.REGIONS
ROOT='65AG_joint_mechanics'
ARMS=('uncoupled','coupled')
ARCH='65AG_joint_pointwise_local_multitask_v1'
INPUTS=f.N_FEATURES+6


def safe_output(root,sub):
    out=root/'results'/ROOT/sub
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Existing output protected; to rerun, delete ONLY this folder manually:\n'+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out


def sha(path):
    return ac.aa.co.sha256(path)


def member_sources(root,reg,m,device,need_predictions=True):
    st=f.cases(root,reg,m,device,need_predictions)
    p=f.checkpoint_path(root,'pointwise',reg,m)
    q=f.checkpoint_path(root,'lgno',reg,m)
    for src in (p,q):
        if not src.is_file():raise FileNotFoundError('Missing frozen 65AF trained model: '+str(src))
    policy=root/'results'/f.ROOT/'C_validation_comparison'/'65AF_C_summary.json'
    if not policy.is_file():raise FileNotFoundError('Run completed 65AF-C first: '+str(policy))
    summary=json.loads(policy.read_text())
    row=next((r for r in summary['decisions'] if r['region']==reg and int(r['member'])==m),None)
    if row is None:raise RuntimeError('No 65AF-C frozen alpha for '+reg+' member '+str(m))
    for k in ('pointwise_alpha','lgno_alpha'):
        if not 0<=float(row[k])<=1:raise RuntimeError('Invalid frozen alpha')
    st.update(pointwise_path=p,lgno65af_path=q,alpha_policy=policy,alphas={
        'pointwise':float(row['pointwise_alpha']), 'lgno':float(row['lgno_alpha'])},
        af_p_sha=sha(p),af_l_sha=sha(q),af_summary_sha=sha(policy))
    return st


def train_tune_split(ds,reg,m,tune_cases=12):
    """Reproducible 44/12 split within ORIGINAL 64J TRAIN (56 cases).

    Note upstream frozen pretrained 65AF model previously used all 56 train
    cases, so the internal subset is NOT independent of the entire pipeline.
    This split is for NEW 65AG weight selection, NOT generalization claims.
    """
    original=np.asarray(ds['splits']['train'],dtype=np.int64)
    n=len(original)
    if n<=tune_cases+5:raise RuntimeError('Insufficient original training cases')
    seed=651357+100*REGIONS.index(reg)+int(m)
    perm=np.random.default_rng(seed).permutation(original)
    fit=np.sort(perm[:-tune_cases]);tune=np.sort(perm[-tune_cases:])
    oldval=set(map(int,ds['splits']['validation']))
    oldtest=set(map(int,ds['splits']['test']))
    if (set(map(int,fit))|set(map(int,tune))) & (oldval|oldtest):
        raise RuntimeError('65AG split leaked original validation/test')
    if len(set(fit)|set(tune))!=n:raise RuntimeError('65AG train split invalid')
    return fit,tune


def component_percent(y,ref,c):return f.relative(np.asarray(y)[...,c],np.asarray(ref)[...,c])


def report(L,S,LEref,Sref,C=None):
    out={}
    for i,key in [(0,'11'),(1,'22'),(2,'33'),(3,'12'),(4,'13'),(5,'23')]:
        out['LE'+key+'_pct']=component_percent(L,LEref,i)
        out['S'+key+'_pct']=component_percent(S,Sref,i)
    out['S22_rmse_MPa']=float(np.sqrt(np.mean((S[...,1]-Sref[...,1])**2)))
    if C is not None:
        cL=f.C_LE(L,C)
        out['constitutive_S22_gap_pct']=component_percent(S,cL,1)
        out['constitutive_all_gap_pct']=f.relative(S,cL)
    return out


class JointLocalMechanics(nn.Module):
    """Shared network outputs 6 strain and 6 stress corrections.

    Both arms use exactly the same architecture. 65AG-B only changes
    constitutive-consistency weight, isolating the proposed mechanics coupling.
    Zero output at initialization means exact 65AF-pointwise blended baseline.
    """
    def __init__(self,hidden=96,cap=1.):
        super().__init__()
        self.hidden=int(hidden);self.cap=float(cap)
        self.shared=nn.Sequential(nn.Linear(INPUTS,hidden),nn.SiLU(),
                                  nn.Linear(hidden,hidden),nn.SiLU())
        self.strain=nn.Linear(hidden,6)
        self.stress=nn.Linear(hidden,6)
        for head in (self.strain,self.stress):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,features,LEbase,Sbase,LEscale,Sscale):
        # normalize original learned strain & stress in features, all LOCAL
        bs=(Sbase/Sscale.view(1,1,6)).clamp(-100,100)
        z=self.shared(torch.cat((features,bs),dim=-1))
        L=LEbase+self.cap*torch.tanh(self.strain(z))*LEscale.view(1,1,6)
        S=Sbase+self.cap*torch.tanh(self.stress(z))*Sscale.view(1,1,6)
        return L,S


def to_tensor(a,device):
    return torch.tensor(np.array(a,copy=True),dtype=torch.float32,device=device)


def make_dataset(st,ids,device,batch=4):
    """Uses frozen prior model predictions ONLY for features and baselines.

    True FEM LE/S are deliberately NOT passed to inference model.
    """
    ids=np.asarray(ids,dtype=np.int64)
    F,du,HL,HS,GL=f.build_features(st,ids,device)
    le_sc=f.strain_scales(st['ds']);s_sc=f.stress_scales(st['ds'])
    C=f.stiffness(st['ds'],ids)
    out=dict(ids=ids,F=F,du=du,HL=HL,HS=HS,GL=GL,C=C,
             LEref=np.array(st['ds']['LE'][ids],copy=True).astype(np.float32),
             Sref=np.array(st['ds']['S'][ids],copy=True).astype(np.float32))
    for arm in ('pointwise','lgno'):
        path=st['pointwise_path'] if arm=='pointwise' else st['lgno65af_path']
        mdl,ck=f.load_model(path,arm,st['region'],st['member'],st['source_sha'],st['lgno_sha'],device)
        L=f.predict(mdl,F,du,HL,le_sc,device,batch=batch)
        alpha=st['alphas'][arm]
        S=(1.-alpha)*HS+alpha*f.C_LE(L,C)
        out['LE_'+arm]=L.astype(np.float32)
        out['S_'+arm]=S.astype(np.float32)
        del mdl
    return out


def forward_numpy(model,data,le_sc,s_sc,device,batch=4):
    model.eval();ll=[];ss=[]
    tLE=to_tensor(le_sc,device);tS=to_tensor(s_sc,device)
    with torch.no_grad():
        for start in range(0,len(data['ids']),batch):
            sl=slice(start,start+batch)
            F=to_tensor(data['F'][sl],device)
            l0=to_tensor(data['LE_pointwise'][sl],device)
            s0=to_tensor(data['S_pointwise'][sl],device)
            L,S=model(F,l0,s0,tLE,tS)
            ll.append(L.detach().cpu().numpy());ss.append(S.detach().cpu().numpy())
    return np.concatenate(ll),np.concatenate(ss)


def model_path(root,arm,reg,member):
    return root/'results'/ROOT/('B_'+arm)/reg/('member_%02d'%member)/'joint_mechanics.pt'


def load_model(root,arm,st,device):
    path=model_path(root,arm,st['region'],st['member'])
    if not path.is_file():raise FileNotFoundError('Run 65AG-B for '+arm+': '+str(path))
    ck=torch.load(path,map_location='cpu',weights_only=False) # trusted own checkpoint only
    expected={'arch':ARCH,'arm':arm,'region':st['region'],'member':st['member'],
              'source_sha':st['source_sha'],'af_pointwise_sha':st['af_p_sha'],
              'af_lgno_sha':st['af_l_sha'],'af_policy_sha':st['af_summary_sha']}
    for k,v in expected.items():
        if ck.get(k)!=v:raise RuntimeError('65AG checkpoint source mismatch for %s at %s'%(k,path))
    mdl=JointLocalMechanics(ck['hidden'],ck['cap']).to(device)
    mdl.load_state_dict(ck['state_dict'],strict=True)
    return mdl,ck


def attach_st(st,reg,member):
    st['region']=reg;st['member']=member
    return st


def baselines_from_data(data):
    return {
        'original_LE_S_heads':(data['HL'],data['HS']),
        '65AF_pointwise_blend':(data['LE_pointwise'],data['S_pointwise']),
        '65AF_lgno_blend':(data['LE_lgno'],data['S_lgno'])
    }


def safe_gates(new,base,orig):
    # safety relative to leading 65AF pointwise strain/stress and original S heads
    return bool(new['LE11_pct'] <= 1.10*base['LE11_pct']+.3 and
                new['LE12_pct'] <= 1.10*base['LE12_pct']+.3 and
                new['S11_pct'] <= 1.10*orig['S11_pct']+.5 and
                new['S12_pct'] <= 1.10*orig['S12_pct']+.5 and
                new['LE22_pct'] <= 1.05*base['LE22_pct']+.3)
