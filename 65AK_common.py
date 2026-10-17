#!/usr/bin/env python3
"""Step 65AK: read-only physics diagnostics of six previously packaged FEM cases.

Requires existing 65AJ, 65AI and older source files IN THE MAC PROJECT ROOT.
Existing model files, checkpoints, 65V and 65AJ output are never modified.
"""
from pathlib import Path
import hashlib, importlib, json
import numpy as np
import pandas as pd

aj=importlib.import_module('65AJ_common')
ai=aj.ai
REGIONS=('inner_left','inner_right')
ARMS=('65AG_uncoupled','65AI_pointwise','65AI_local_lgno','65AJ_matched_mlp')
DIR='65AK_diagnostics'
COMPONENTS=('11','22','33','12','13','23')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def new_output(root,name):
    path=Path(root)/'results'/DIR/name
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('65AK will not overwrite results. Delete ONLY this folder to rerun:\n'+str(path))
    path.mkdir(parents=True,exist_ok=True)
    return path


def input_npz(root,region):
    return Path(root)/'results'/aj.ROOT/'E_external_npz'/(region+'.npz')


def a_npz(root,region):
    return Path(root)/'results'/DIR/'A_predictions'/(region+'.npz')


def open_external(root,region):
    path=input_npz(root,region)
    if not path.is_file():raise FileNotFoundError('Run completed 65AJ-E first: '+str(path))
    with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    must=('U','LE','S','coordinates','ip_coordinates','case_ids','materials','source_odb_paths')
    for key in must:
        if key not in data:raise RuntimeError('65AJ-E package missing: '+region+'/'+key)
    return data


def load_A(root,region):
    p=a_npz(root,region)
    if not p.is_file():raise FileNotFoundError('Run 65AK_A_collect.py first: '+str(p))
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def relpct(a,b):
    """Relative L2 (%) with b as reference. Aggregate over supplied axes."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape:raise ValueError('relative: shapes do not match '+str((a.shape,b.shape)))
    return float(100*np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-14))


def rmse(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.sqrt(np.mean((a-b)**2)))


def per_case_fields(z,model,idx):
    return z['LE_'+model][idx], z['S_'+model][idx]


def atomic_summary(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))


def provenance_check(root,region,z):
    source=input_npz(root,region)
    if str(z['source_npz_sha256'])!=sha256(source):
        raise RuntimeError('65AJ-E source changed after Step 65AK-A for '+region+'; rerun 65AK-A on a clean subfolder.')
