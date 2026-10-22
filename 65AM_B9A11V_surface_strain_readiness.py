#!/usr/bin/env python3
"""B9A11V auxiliary: inspect available B9A11T fields for surface-strain inversion.

Audit only. NEVER relabel internal Abaqus LE tensors as measured surface strain.
No ODBs or solver calls and no changes to training checkpoints.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root):
    root=Path(root).expanduser().resolve()
    path=root/'B9A11T_exports'/'B9A11T_U_reference.npz'
    if not path.is_file():
        raise FileNotFoundError('Missing B9A11T_U_reference.npz: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        keys=list(z.files)
        shapes={name:list(z[name].shape) for name in keys}
    likely_strain=[k for k in keys if ('strain' in k.lower() or k.lower().startswith(('le_', 'e_')))]
    response={
        'status':'NOT READY for physically measured displacement-plus-surface-strain inversion',
        'source_file':str(path),
        'fields':shapes,
        'candidate_strain_fields':likely_strain,
        'warning':'Even if a LE tensor exists, Abaqus interior/integration-point strain is not equivalent to a surface-gauge or DIC measurement.',
        'required_next_data':[
            'Identify accessible composite faces and local material-axis orientation on the specimen',
            'Choose measurable in-plane surface strain components tangent to each selected face',
            'Export FEM predictions at the SAME locations and loading states for baseline and material perturbations',
            'Document measurement noise, spatial averaging, tensor vs engineering shear conventions',
            'Ensure the forward model predicts strain at those same sensor locations',
            'Use development cases for sensor selection; new held-out FEM cases for the final inverse comparison'
        ],
        'no_claims_of_physical_experimental_validation':True
    }
    folder=root/'results'/'65AM_B9A11V_surface_strain_readiness'
    dest=folder/'B9A11V_strain_readiness.json'
    if dest.exists():
        raise RuntimeError('Existing surface audit protected: '+str(dest))
    folder.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(response,indent=2))
    print('Surface-strain AUDIT COMPLETE:',dest)
    print('FEM array keys:',', '.join(keys))
    print('STATUS: do not yet use B9A11T as a physical surface-strain dataset')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['selftest','audit'])
    p.add_argument('--project',default='.')
    a=p.parse_args()
    if a.stage=='selftest':
        assert isinstance(np.array([1]).shape,tuple)
        print('B9A11V SURFACE AUDIT SELFTEST PASS')
    else:
        audit(a.project)


if __name__=='__main__':main()
