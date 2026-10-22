#!/usr/bin/env python3
"""B9A11N-A: 18 SAFE orthotropic finite-difference perturbations from 3Point.inp.

Must run on Windows, from ODBS root, with the existing B9A11I input reader in scripts/.
Generate ONLY INPs and command files. Never runs Abaqus and never edits 3Point.inp.
"""
from __future__ import print_function
import argparse, hashlib, importlib.util, json, math, os, re, sys
from pathlib import Path

PARS=['E1','E2','E3','nu12','nu13','nu23','G12','G13','G23']
NOMINAL=[45000.,12000.,12000.,.28,.28,.40,4500.,4500.,3500.]
AUDIT_SHA='61272f565806ce0c1a066574dc2f6362f25885c411acdd2b86ecaf3af9c9ad6a'
ROOT=Path.cwd()

def load_reader(root):
    p=root/'scripts'/'65AM_B9A11I_C_W_damage_inp.py'
    if not p.is_file():raise FileNotFoundError('Missing original verified reader: '+str(p))
    spec=importlib.util.spec_from_file_location('verified_b9i',str(p))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def elastic_spd(vals):
    E1,E2,E3,n12,n13,n23,g12,g13,g23=vals
    if min(E1,E2,E3,g12,g13,g23)<=0:return False
    a,b,c=1/E1,1/E2,1/E3
    d,e,f=-n12/E1,-n13/E1,-n23/E2
    minor=a*b-d*d
    det=a*b*c+2*d*e*f-a*f*f-b*e*e-c*d*d
    return a>0 and minor>0 and det>0

def parse_base(root,path):
    reader=load_reader(root)
    lines,sha,enc=reader.read_inp(path)
    info=reader.find_composite(lines)
    vals=[float(x) for x in info['const']]
    if any(abs(x-y)>max(1e-7,abs(y)*1e-7) for x,y in zip(vals,NOMINAL)):
        raise RuntimeError('3Point.inp healthy constants changed from previously audited values: '+str(vals))
    if not elastic_spd(vals):raise RuntimeError('Unstable healthy orthotropic elasticity')
    return reader,lines,sha,enc,info,vals

def write_batch(out,root,jobs,cpus):
    b=[];b.append('@echo off\r\nsetlocal\r\ncd /d "'+str(root)+'"\r\n')
    b.append('echo B9A11N: jobs will run SEQUENTIALLY. Existing completed ODBs will be skipped.\r\n')
    for job in jobs:
        # Guard against loss of already computed simulations; do not send overwrite prompts.
        b += ['if exist "'+job+'.odb" (\r\n',
              '  findstr /C:"THE ANALYSIS HAS COMPLETED SUCCESSFULLY" "'+job+'.sta" >nul 2>&1\r\n',
              '  if errorlevel 1 (echo ERROR: existing '+job+'.odb without successful STA. Inspect manually.&exit /b 1)\r\n',
              '  echo SKIP finished '+job+'\r\n',
              ') else (\r\n',
              '  if exist "'+job+'.lck" (echo ERROR: locked job '+job+'&exit /b 1)\r\n',
              '  abaqus job='+job+' input="'+str(out.relative_to(root))+'\\'+job+'.inp" cpus='+str(cpus)+' interactive\r\n',
              '  if errorlevel 1 (echo ERROR: Abaqus failed '+job+'&exit /b 1)\r\n',
              '  findstr /C:"THE ANALYSIS HAS COMPLETED SUCCESSFULLY" "'+job+'.sta" >nul 2>&1\r\n',
              '  if errorlevel 1 (echo ERROR: '+job+' did not finish successfully&exit /b 1)\r\n',
              ')\r\n']
    b.append('echo ALL 18 JOBS COMPLETED SUCCESSFULLY\r\n')
    (out/'B9A11N_run_18_jobs.cmd').write_bytes(''.join(b).encode('ascii'))
    b=['@echo off\r\nsetlocal\r\ncd /d "'+str(root)+'"\r\n']
    for job in jobs:
        tag_dir='online_results_7region_65W_middle\\B_odb_exports\\'+job
        b += ['findstr /C:"THE ANALYSIS HAS COMPLETED SUCCESSFULLY" "'+job+'.sta" >nul 2>&1\r\n',
              'if errorlevel 1 (echo ERROR: missing successful STA for '+job+'&exit /b 1)\r\n',
              'if exist "'+tag_dir+'\\'+job+'_manifest.json" (\r\n',
              '  echo SKIP exported '+job+'\r\n',
              ') else (\r\n',
              '  abaqus python scripts\\65W_B_export_odb.py --stage export --odb '+job+'.odb --tag '+job+' --instances COMPOSITE-1 --step Loading\r\n',
              '  if errorlevel 1 (echo ERROR: export failed '+job+'&exit /b 1)\r\n',
              ')\r\n']
    b.append('echo ALL 18 EXPORTS COMPLETED\r\n')
    (out/'B9A11N_export_18_jobs.cmd').write_bytes(''.join(b).encode('ascii'))

def run(args):
    root=Path(args.root).resolve();source=(root/args.base).resolve()
    if not source.is_file():raise FileNotFoundError(str(source))
    reader,lines,sha,encoding,info,nominal=parse_base(root,source)
    if sha!=AUDIT_SHA:
        print('WARNING: 3Point.inp SHA-256 differs from previous audited file. Compare source changes before simulation.')
        print('previous:',AUDIT_SHA,'current:',sha)
        if args.stage=='generate' and not args.accept_changed_source:
            raise RuntimeError('Refusing generation without --accept-changed-source after manual audit.')
    details={'source':str(source),'source_sha256':sha,'previous_source_sha256':AUDIT_SHA,
             'source_hash_matches_prior_audit':sha==AUDIT_SHA,'material_name':'GlassEpoxy',
             'nominal_constants':dict(zip(PARS,nominal)), 'elastic_pos_definite':True,
             'n_elements':len(info['original_set']), 'perturb_fraction':args.fraction,
             'warning':'One-at-a-time forward sensitivities (not fitted inverse model and not a causal universal ranking).'}
    print(json.dumps(details,indent=2))
    if args.stage=='audit':return
    if not 0<args.fraction<=.1:raise RuntimeError('Perturbation must be in (0,0.1]')
    out=(root/args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Nonempty output directory protected: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    nl='\r\n' if any(x.endswith('\r\n') for x in lines) else '\n'
    elastic_index=info['elastic']
    stop=reader.next_keyword(lines,elastic_index)
    if stop <= elastic_index+1:raise RuntimeError('Missing material data rows')
    cases=[];jobs=[]
    for j,property in enumerate(PARS):
        for direction,sign in [('M',-1.),('P',1.)]:
            value=list(nominal);value[j]*=(1+sign*args.fraction)
            if abs(nominal[j])<1e-12:raise RuntimeError('Cannot use relative sensitivity for zero '+property)
            if not elastic_spd(value):raise RuntimeError('Perturbed elastic constants fail SPD at '+property+' '+direction)
            job='N9_'+property+'_'+direction
            updated=lines[:elastic_index+1]+[', '.join('%.12g'%v for v in value[:8])+nl, '%.12g'%value[8]+nl]+lines[stop:]
            content=''.join(updated)
            path=out/(job+'.inp')
            path.write_bytes(content.encode(encoding))
            _,_,_,_,check,parsed=parse_base_perturbed(reader,path)
            if any(abs(a-b)>max(1e-7,abs(b)*1e-7) for a,b in zip(parsed,value)):
                raise RuntimeError('Generated material constants mismatch: '+job)
            if check['original_set']!=info['original_set'] or check['nodes']!=info['nodes'] or check['elements']!=info['elements']:
                raise RuntimeError('Generated mesh changed: '+job)
            jobs.append(job)
            cases.append({'tag':job,'parameter':property,'direction':direction,'relative_change':sign*args.fraction,
                          'parameter_nominal':nominal[j],'parameter_value':value[j],
                          'inp_path':str(path),'inp_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'all_constants':dict(zip(PARS,value))})
    write_batch(out,root,jobs,args.cpus)
    details['cases']=cases
    (out/'B9A11N_input_audit.json').write_text(json.dumps(details,indent=2))
    print('GENERATED:',len(jobs),'INPs + 2 batch scripts at',str(out))

def parse_base_perturbed(reader,path):
    l,h,e=reader.read_inp(path);i=reader.find_composite(l)
    return reader,l,h,e,i,[float(x) for x in i['const']]

def selftest():
    assert elastic_spd(NOMINAL)
    for j in range(9):
        for s in [-1,1]:
            v=list(NOMINAL);v[j]*=(1+s*.05)
            assert elastic_spd(v),(j,s)
    bad=list(NOMINAL);bad[0]=-1
    assert not elastic_spd(bad)
    print('SELFTEST PASS: 19 stable elastic tensors / instability rejected')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['audit','generate','selftest'],required=True)
    p.add_argument('--root',default='.')
    p.add_argument('--base',default='3Point.inp')
    p.add_argument('--fraction',type=float,default=.05)
    p.add_argument('--out',default='B9A11N_9param_INP')
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--accept-changed-source',action='store_true',help='Only after manually checking source INP against prior approved healthy analysis')
    a=p.parse_args()
    if a.stage=='selftest':selftest()
    else:run(a)
if __name__=='__main__':main()
