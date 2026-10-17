#!/usr/bin/env python3
"""Package ACTUALLY solved Abaqus FEM arrays; never synthesizes reference labels.

Requires project/new_FEM_exports/<region>/
  U_stack.npy               [cases,1008,3]
  LE_stack.npy              [cases,728,6] Abaqus LE material orientation
  S_stack.npy               [cases,728,6] Abaqus S material orientation MPa
  coordinates.npy           [1008,3] as Abaqus exported/ordered
  ip_coordinates.npy        [728,3] as Abaqus exported/ordered
  source_odb_paths.txt      one distinct full-FEM ODB path PER material case
These must come from fresh ODBs and be mapped to the original region node/IP
ordering. The script verifies shapes, geometry and finite values, NOT that an
ODB actually contains these fields or that the source model was physically
solved. Provenance must be independently inspected before publication.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import importlib
aj=importlib.import_module('65AJ_common')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',default='.')
 p.add_argument('--source-dir',default='new_FEM_exports')
 a=p.parse_args();root=Path(a.project).resolve()
 manifest=root/'results'/aj.ROOT/'D_new_material_manifest'/'65AJ_D_new_cases.csv'
 if not manifest.is_file():raise FileNotFoundError('Run 65AJ-D first: '+str(manifest))
 with manifest.open() as f:rows=list(csv.DictReader(f))
 ids=np.asarray([r['case_id'] for r in rows],dtype=str)
 mats=np.asarray([[float(r[z]) for z in ('E1_MPa','E2_MPa','G12_MPa')] for r in rows],np.float32)
 allin={};srcdir=root/a.source_dir
 for reg in aj.REGIONS:
  folder=srcdir/reg
  need=['U_stack.npy','LE_stack.npy','S_stack.npy','coordinates.npy','ip_coordinates.npy','source_odb_paths.txt']
  if any(not (folder/x).is_file() for x in need):
   raise FileNotFoundError('Missing REAL Abaqus exports for '+reg+': '+', '.join(x for x in need if not (folder/x).is_file()))
  z={k:np.load(folder/(k+'.npy'),allow_pickle=False) for k in ('coordinates','ip_coordinates')}
  z['U']=np.load(folder/'U_stack.npy',allow_pickle=False)
  z['LE']=np.load(folder/'LE_stack.npy',allow_pickle=False)
  z['S']=np.load(folder/'S_stack.npy',allow_pickle=False)
  z['source_odb_paths']=np.asarray([x.strip() for x in (folder/'source_odb_paths.txt').read_text().splitlines() if x.strip()],dtype=str)
  ds=aj.ai.ag.ac.aa.load_data(root,reg)
  refcoords=ds['coordinates'];refip=ds['ip_coordinates']
  k=len(ids)
  if any(z[x].shape!=s for x,s in [('U',(k,len(refcoords),3)),('LE',(k,len(refip),6)),
                   ('S',(k,len(refip),6)),('coordinates',refcoords.shape),('ip_coordinates',refip.shape),
                   ('source_odb_paths',(k,))]):
   raise RuntimeError('Incorrect exported arrays in '+str(folder))
  if not all(np.isfinite(z[x]).all() for x in ('U','LE','S','coordinates','ip_coordinates')):
   raise RuntimeError('Nonfinite Abaqus arrays in '+str(folder))
  if not np.allclose(z['coordinates'],refcoords,rtol=0,atol=1e-5) or not np.allclose(z['ip_coordinates'],refip,rtol=0,atol=1e-5):
   raise RuntimeError('External data must be in identical original region node and IP ordering: '+reg)
  if len(set(z['source_odb_paths'].tolist()))!=k:
   raise RuntimeError('New independent cases must have distinct source ODB names for each material: '+reg)
  z.update(materials=mats,case_ids=ids)
  allin[reg]=z
 out=aj.new_folder(root,'E_external_npz')
 for reg,z in allin.items():np.savez_compressed(out/(reg+'.npz'),**z)
 (out/'65AJ_E_status.json').write_text(json.dumps(dict(step='65AJ-E',case_count=len(ids),
    status='Unverified user-supplied full-FEM arrays packaged after checks; NOT a proof of independent physical correctness',
    manual_checks='Review ODB material assignments/mesh/integration-point ordering, force/displacement, engineering shear, local material direction'),indent=2))
 print('Export packages ready:',out)
if __name__=='__main__':main()
