#!/usr/bin/env python3
"""WINDOWS Python: stack already exported, genuinely new Abaqus cases.
Copy 65AJ_D_new_cases.csv to Windows before running.
Example:
 python 65AJ_W_stack_exports.py --manifest C:\\...\\65AJ_D_new_cases.csv \
  --target-dir C:\\...\\new_FEM_exports\\inner_left --odb-dir C:\\...\\new_ODB
The ODB names must match AJ_new_01.odb, etc. It verifies their existence but
cannot verify that those ODBs actually contain the requested material values.
"""
import argparse,csv,os
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',required=True);p.add_argument('--target-dir',required=True)
 p.add_argument('--odb-dir',required=True)
 a=p.parse_args();folder=Path(a.target_dir).resolve();odbdir=Path(a.odb_dir).resolve()
 with open(a.manifest,newline='') as f:cases=list(csv.DictReader(f))
 if len(cases)<3:raise RuntimeError('Need >=3 new full-FEM cases')
 outputs=[folder/(s+'_stack.npy') for s in ('U','LE','S')]+[folder/'source_odb_paths.txt']
 if any(f.exists() for f in outputs):raise RuntimeError('Refusing to overwrite prior exports')
 data={name:[] for name in ('U','LE','S')};odbpaths=[]
 for row in cases:
  case=row['case_id'];odb=odbdir/(case+'.odb')
  if not odb.is_file():raise FileNotFoundError('No real Abaqus ODB: '+str(odb))
  odbpaths.append(str(odb))
  for kind in data:
   fname=folder/(kind+'_'+case+'.npy')
   if not fname.is_file():raise FileNotFoundError('Run 65AJ_W_export_from_odb first: '+str(fname))
   data[kind].append(np.load(fname,allow_pickle=False))
 for kind,arr in data.items():
  x=np.stack(arr).astype(np.float32)
  if not np.isfinite(x).all():raise RuntimeError('Nonfinite '+kind)
  np.save(folder/(kind+'_stack.npy'),x)
 (folder/'source_odb_paths.txt').write_text('\n'.join(odbpaths)+'\n')
 print('PASS: stacked',len(cases),'actual Abaqus exports from',folder)
if __name__=='__main__':main()
