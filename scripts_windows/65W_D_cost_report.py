"""65W-D: audit historical Abaqus patch cost; generate controlled-benchmark protocol.

No Abaqus runs. These are DEVELOPMENT costs and NOT a fair speedup benchmark.
Run from ODBS: python scripts\65W_D_cost_report.py
Requires existing 65P/65R/65S/65T/65U/65V manifest files.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'online_results_7region_65W_middle'/'D_cost_report'
ENTRIES={
 '65P':ROOT/'online_results_7region_65P_middle_adaptive_selective'/'rom_solution_manifest.json',
 '65R':ROOT/'online_results_7region_65R_middle'/'65R_D_retry'/'rom_solution_manifest.json',
 '65S_D2':ROOT/'online_results_7region_65S_middle'/'65S_D_from_verified'/'rom_solution_manifest.json',
 '65T':ROOT/'online_results_7region_65T_middle'/'65T_D_two_patch_retry'/'rom_solution_manifest.json',
 '65U':ROOT/'online_results_7region_65U_middle'/'65U_C_lm_continuation'/'rom_solution_manifest.json',
 '65V':ROOT/'online_results_7region_65V_middle'/'65V_D_lm_convergence'/'rom_solution_manifest.json',
}


def lookup(obj,keys):
    for k in keys:
        if k in obj:return obj[k]
    return None


def report():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Report dir not empty. Existing report protected: '+str(OUT))
    OUT.mkdir(parents=True)
    rows=[]
    for name,path in ENTRIES.items():
        if not path.is_file():
            rows.append(dict(experiment=name,exists=False,final_verified_rms='',
                     converged='',patch_jobs_reported='',runtime_seconds='',source=str(path)))
            continue
        with open(path) as f:d=json.load(f)
        if name=='65R':
            jobs=d.get('total_Abaqus_patch_jobs_in_65R')
        elif name=='65T':jobs=d.get('total_new_patch_jobs')
        elif name=='65V':jobs=d.get('total_65V_patch_jobs')
        elif name=='65S_D2':jobs=d.get('patch_jobs_this_continuation')
        else:jobs=lookup(d,['abaqus_patch_jobs','high_fidelity_abaqus_jobs'])
        rows.append(dict(experiment=name,exists=True,
                 final_verified_rms=lookup(d,['best_verified_rms','final_high_fidelity_rms']),
                 converged=d.get('converged'),patch_jobs_reported=jobs,
                 runtime_seconds=lookup(d,['runtime_seconds','run_seconds','total_seconds','runtime_retry_seconds']),
                 source=str(path)))
    with open(OUT/'65W_D_development_costs.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p={
      'what_can_be_claimed':'The 65V solution converged in 51 new high-fidelity patch jobs within that step; each historical experiment has its own warm start and calibrations.',
      'what_cannot_be_claimed':'Do not add all listed jobs to infer an algorithm speedup; do not treat development milestones as matched comparisons. Do not compare with the 9-job 65H baseline directly because material cases and warm starts differ.',
      'controlled_benchmark_design':{
          'same_material_MPa':[46500,12500,4750],
          'identical_start':'Select ONE 26-coordinate start solution and verify full Abaqus residual before each independent solver run',
          'identical_stopping':'normalized 26-component HF RMS <= 0.02',
          'identical_budget':'Recommend 60 Abaqus PATCH jobs inclusive of calibrations and repro checks',
          'identical_cpu_and_license':'same university server allocation/cpus/Abaqus version',
          'identical_models':'same trained NO weights and FE/ROM templates',
          'randomness':'deterministic seeds/repeats where feasible; report variability',
          'treat_cost_as':'number of Abaqus patch jobs, full-FEM jobs separately, and actual wall-clock times',
          'policies_to_compare':['Direct-FEM Broyden','ROM-only LM with final HF verification','HF-calibrated LM without refresh','HF-calibrated LM with dynamic refresh (65V)'],
          'fairness_note':'Existing 65Q code uses a different starting state; it is not a matched 65V controlled experiment as-is.',
          'do_not_claim_speedup_until':'identical-condition baseline runs are implemented and checked'
      },
      'historical_experiments':rows
    }
    with open(OUT/'65W_D_benchmark_protocol.json','w') as f:json.dump(p,f,indent=2)
    print('Wrote:',OUT)
    for item in rows:print(item['experiment'],item['final_verified_rms'],item['patch_jobs_reported'])
    print('These are research development costs, NOT matched solver speedups.')


def selftest():
    assert lookup({'A':0},('A','B'))==0
    assert lookup({'B':2},('A','B'))==2
    assert lookup({},('A',)) is None
    print('65W-D selftest PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('selftest','report'),default='report')
    args=p.parse_args()
    if args.stage=='selftest':selftest()
    else:report()

if __name__=='__main__':main()
