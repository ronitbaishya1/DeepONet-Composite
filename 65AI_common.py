#!/usr/bin/env python3
"""65AI: Displacement-conditioned local joint strain/stress residual operators.

Scientific references / inspirations:
  Shin et al. (JCDE 2025): global displacement -> local stress.
  Rashid et al. (iScience 2022): multiple tensor components predicted together.
  Zhang et al. (LGNO 2026): local displacement-dependent stencil weights.
Adaptation below is original PyTorch code for nonperiodic 3D C3D8R elements.

Inputs at INFERENCE: frozen 65AA hard-compatible predicted nodal U,
[E1,E2,G12] and actual interface-PCA branch values, local element geometry,
frozen 65AC gradient features and frozen 65AG uncoupled LE/S predictions.
Never pass reference FEM U/LE/S to model features. Reference LE/S is used only
for supervised TRAINING and for evaluation on developer cases.

No model modification to 65V solver, source DeepONet, displacement or force.
Predicted strain is NOT automatically compatible with source displacement.
"""
from pathlib import Path
import importlib
import json
import numpy as np
import torch
from torch import nn

ag = importlib.import_module('65AG_common')
co = ag.ac.aa.co
REGIONS = ag.REGIONS
ROOT = '65AI_displacement_conditioned'
ARCH = '65AI_frozenU_local_U_to_joint_LE_S_v1'
ARMS = ('pointwise', 'local_lgno')
TENSOR_COMPONENTS=('11','22','33','12','13','23')
# 65AF features (99: includes FEM cell geometry/DU/gradient) + branch
# plus predicted LE and S baseline vectors (6+6).
FEATURES_BASE = ag.f.N_FEATURES


def no_overwrite(root,sub):
    folder=root/'results'/ROOT/sub
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Output folder already contains results; not overwriting:\n'+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def tensor(a,device):
    return torch.as_tensor(np.array(a,copy=True),dtype=torch.float32,device=device)


def member_source(root,region,member,device,need_predictions=True):
    st=ag.attach_st(ag.member_sources(root,region,member,device,need_predictions),region,member)
    mdl,ck=ag.load_model(root,'uncoupled',st,device)
    st['root']=root
    st['prior_ck']=ck
    st['prior_path']=ag.model_path(root,'uncoupled',region,member)
    st['prior_sha']=ag.sha(st['prior_path'])
    del mdl
    return st


def internal_split(ds,region,member,tune=12):
    return ag.train_tune_split(ds,region,member,tune)


def train_scales(ds):
    return (ag.f.strain_scales(ds).astype(np.float32),
            ag.f.stress_scales(ds).astype(np.float32))


def base_features(st,ids,device,batch=4):
    """Compute cached, FROZEN predictions. Abaqus LE/S do NOT enter inputs."""
    ids=np.asarray(ids,dtype=np.int64)
    if np.intersect1d(ids,np.asarray(st['ds']['splits']['test'],dtype=np.int64)).size:
        raise RuntimeError('Original 64J test cases are prohibited for 65AI')
    legacy=ag.make_dataset(st,ids,device,batch)
    mdl,ck=ag.load_model(st['root'],'uncoupled',st,device)
    L0,S0=ag.forward_numpy(mdl,legacy,*train_scales(st['ds']),device,batch)
    del mdl
    bn,_,_=co.scaled_inputs(st['ck'],st['ds'])
    bnp=np.asarray(bn[ids],np.float32)
    LEscale,Sscale=train_scales(st['ds'])
    E=legacy['F'].shape[1]
    normalizedLE=L0/LEscale[None,None,:]
    normalizedS=S0/Sscale[None,None,:]
    branch=np.broadcast_to(bnp[:,None,:],(len(ids),E,bnp.shape[-1]))
    features=np.concatenate((legacy['F'],normalizedLE,normalizedS,branch),axis=-1).astype(np.float32)
    # local 'du' is normalized local displacement differences from the HARD-
    # corrected, pretrained 65AA model, never from Abaqus reference U.
    du=np.asarray(legacy['du'],np.float32)
    if du.shape!=(len(ids),E,8,3):
        raise RuntimeError('Predicted displacement neighborhood wrong shape: '+str(du.shape))
    if not np.isfinite(features).all() or not np.isfinite(du).all():
        raise RuntimeError('Nonfinite predicted local input features')
    # Optional source-U comparison reads FEM nodal U only as an EVALUATION label.
    pos=np.array([st['index_map'][int(i)] for i in ids],dtype=int)
    U0=np.asarray(st['pred'][pos],np.float32)
    return dict(ds=st['ds'],ids=ids,features=features,du=du,L0=L0.astype(np.float32),S0=S0.astype(np.float32),
                Lref=np.asarray(st['ds']['LE'][ids],np.float32),
                Sref=np.asarray(st['ds']['S'][ids],np.float32),
                U0=U0,Uref=np.asarray(st['ds']['U'][ids],np.float32),
                head_L=np.asarray(legacy['HL'],np.float32),
                head_S=np.asarray(legacy['HS'],np.float32),
                af_L=np.asarray(legacy['LE_pointwise'],np.float32),
                af_S=np.asarray(legacy['S_pointwise'],np.float32),
                C=np.asarray(legacy['C'],np.float32))


class DisplacementToLocalMechanics(nn.Module):
    """Shared local encoder + two LE/S heads. Local LGNO arm adds U stencils.

    SAME input features, normalization, optimizer and output heads in both arms.
    The local_lgno arm adds trainable displacement-weight contractions. Because
    it adds capacity (extra stencil heads), treat comparisons as architecture
    ablation, not a matched-parameter-count result.
    """
    def __init__(self,input_dim,arm,hidden=96,cap=1.0):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unsupported 65AI arm '+str(arm))
        self.input_dim=int(input_dim)
        self.arm=arm;self.hidden=int(hidden);self.cap=float(cap)
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.SiLU())
        self.LE_head=nn.Linear(hidden,6)
        self.S_head=nn.Linear(hidden,6)
        if arm=='local_lgno':
            self.LE_stencil=nn.Linear(hidden,6*8*3)
            self.S_stencil=nn.Linear(hidden,6*8*3)
        for head in ([self.LE_head,self.S_head]+
                     ([self.LE_stencil,self.S_stencil] if arm=='local_lgno' else [])):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,features,du,LEbase,Sbase,LEscale,Sscale):
        if features.shape[-1]!=self.input_dim:raise RuntimeError('65AI feature width mismatch')
        h=self.encoder(features)
        le_norm=torch.tanh(self.LE_head(h))
        s_norm=torch.tanh(self.S_head(h))
        if self.arm=='local_lgno':
            # LGNO-inspired adaptive local weights on centered U differences.
            # Because du is centered, stencil terms vanish for constant U.
            shape=(*du.shape[:2],6,8,3)
            weightsLE=torch.tanh(self.LE_stencil(h)).reshape(shape)
            weightsS=torch.tanh(self.S_stencil(h)).reshape(shape)
            le_norm=le_norm+torch.einsum('beqic,beic->beq',weightsLE,du)/np.sqrt(24.)
            s_norm=s_norm+torch.einsum('beqic,beic->beq',weightsS,du)/np.sqrt(24.)
        L=LEbase+self.cap*le_norm*LEscale.view(1,1,6)
        S=Sbase+self.cap*s_norm*Sscale.view(1,1,6)
        return L,S


def predict(model,data,le_scale,s_scale,device,batch=4):
    model.eval();outs=[[],[]]
    with torch.no_grad():
        for start in range(0,len(data['ids']),batch):
            sl=slice(start,start+batch)
            l,s=model(tensor(data['features'][sl],device),tensor(data['du'][sl],device),
                      tensor(data['L0'][sl],device),tensor(data['S0'][sl],device),
                      tensor(le_scale,device),tensor(s_scale,device))
            outs[0].append(l.detach().cpu().numpy());outs[1].append(s.detach().cpu().numpy())
    return np.concatenate(outs[0]),np.concatenate(outs[1])


def error_metrics(L,S,data):
    report=ag.report(L,S,data['Lref'],data['Sref'],data['C'])
    report['U2_pct']=ag.f.relative(data['U0'][...,1],data['Uref'][...,1])
    report['U_vector_pct']=ag.f.relative(data['U0'],data['Uref'])
    report['LE_vector_pct']=ag.f.relative(L,data['Lref'])
    report['S_vector_pct']=ag.f.relative(S,data['Sref'])
    report['LE22_rmse']=float(np.sqrt(np.mean((L[...,1]-data['Lref'][...,1])**2)))
    report['S22_rmse_MPa']=float(np.sqrt(np.mean((S[...,1]-data['Sref'][...,1])**2)))
    # Kinematic mismatch: independent LE estimate vs strain derived from frozen U.
    estimated_log=ag.ac.logstrain_from_gradient(ag.ac.aa.gradient_numpy(data['U0'],data['ds'])) if 'ds' in data else None
    if estimated_log is not None:
        report['kinematic_LE22_gap_pct']=ag.f.relative(L[...,1],estimated_log[...,1])
    return report


def objective(m):
    return float(m['S22_pct']+.35*m['LE22_pct']+.10*m['S11_pct']+.10*m['S12_pct']+.10*m['LE11_pct'])


def safety(m,base):
    # Keep key non-target fields close to strongest prior 65AG model.
    return bool(m['S11_pct']<=1.10*base['S11_pct']+.20 and
                m['S12_pct']<=1.10*base['S12_pct']+.30 and
                m['LE11_pct']<=1.10*base['LE11_pct']+.20 and
                m['LE12_pct']<=1.10*base['LE12_pct']+.30 and
                m['LE22_pct']<=1.05*base['LE22_pct']+.25 and
                m['S22_pct']<=1.01*base['S22_pct']+.10)


def ck_path(root,arm,region,member):
    return root/'results'/ROOT/('B_'+arm)/region/('member_%02d'%member)/'local_mechanics.pt'


def load_candidate(root,arm,st,device):
    p=ck_path(root,arm,st['region'],st['member'])
    if not p.is_file():raise FileNotFoundError('65AI-B checkpoint absent: '+str(p))
    ck=torch.load(p,map_location='cpu',weights_only=False)  # trusted locally generated model only
    for key,value in [('architecture',ARCH),('arm',arm),('region',st['region']),
                      ('member',st['member']),('prior_sha',st['prior_sha']),
                      ('source_sha',st['source_sha'])]:
        if ck.get(key)!=value:raise RuntimeError('65AI provenance mismatch for '+key)
    mdl=DisplacementToLocalMechanics(ck['input_dim'],arm,ck['hidden'],ck['cap']).to(device)
    mdl.load_state_dict(ck['state_dict'],strict=True)
    mdl.eval()
    return mdl,ck
