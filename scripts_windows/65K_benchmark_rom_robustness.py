
# Windows / Abaqus Python 2025
# Benchmarks additional ROM-corrected material cases.
#
# Each case gets a separate result directory.
# The existing successful baseline is not overwritten.

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import time


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(SCRIPT_DIR)

SOLVER_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "65H_solve_rom_accelerated_hybrid.py",
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "rom_robustness_benchmark_65K",
)

LOG_DIR = os.path.join(
    OUT_DIR, "logs"
)

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)


# ============================================================
# ADDITIONAL MATERIAL CASES
# ============================================================

CASES = [
    (
        "case_soft",
        42000.0,
        11700.0,
        4250.0,
    ),
    (
        "case_middle",
        46500.0,
        12500.0,
        4750.0,
    ),
    (
        "case_stiff",
        49000.0,
        10900.0,
        4500.0,
    ),
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cpus",
    type=int,
    default=4,
)

parser.add_argument(
    "--rom_max_iter",
    type=int,
    default=25,
)

parser.add_argument(
    "--hf_max_iter",
    type=int,
    default=8,
)

parser.add_argument(
    "--tol",
    type=float,
    default=0.02,
)

parser.add_argument(
    "--rerun",
    action="store_true",
)

parser.add_argument(
    "--dry_run",
    action="store_true",
)

parser.add_argument(
    "--old_direct_seconds",
    type=float,
    default=None,
    help=(
        "Independently measured original 64T "
        "baseline runtime on the same machine"
    ),
)

args = parser.parse_args()


# ============================================================
# CHECK EXISTING 65H SOURCE
# ============================================================

if not os.path.isfile(SOLVER_SCRIPT):
    raise FileNotFoundError(SOLVER_SCRIPT)

with open(SOLVER_SCRIPT, "r") as f:
    solver_text = f.read()

source_token = '"online_results_7region_rom"'

if solver_text.count(source_token) != 1:
    raise RuntimeError(
        "65H output path has changed. "
        "Expected exactly one {}"
        .format(source_token)
    )


# ============================================================
# INCLUDE EXISTING BASELINE RESULT
# ============================================================

rows = []

existing_manifest_file = os.path.join(
    BASE_DIR,
    "online_results_7region_rom",
    "rom_solution_manifest.json",
)

if os.path.isfile(existing_manifest_file):

    with open(existing_manifest_file, "r") as f:
        m = json.load(f)

    rows.append({
        "Case": "baseline_existing_65H",
        "Source": "already completed (not rerun)",
        "E1_MPa": 45000.0,
        "E2_MPa": 12000.0,
        "G12_MPa": 4500.0,
        "Converged": bool(
            m.get("converged", False)
        ),
        "FinalHighFidelityRMS":
            m.get("final_high_fidelity_rms"),
        "ROMResidualEvaluations":
            m.get("rom_residual_evaluations"),
        "RealFEMResidualEvaluations":
            m.get("high_fidelity_residual_evaluations"),
        "AbaqusPatchJobs":
            m.get("high_fidelity_abaqus_jobs"),
        "ROMSeconds":
            m.get("rom_phase_seconds"),
        "HighFidelitySeconds":
            m.get("high_fidelity_phase_seconds"),
        "SolverSeconds":
            m.get("total_seconds"),
        "ProcessWallSeconds": "",
        "ExitCode": 0,
        "ResultFolder": os.path.join(
            BASE_DIR,
            "online_results_7region_rom",
        ),
    })


# ============================================================
# RUN NEW CASES
# ============================================================

for case_name, E1, E2, G12 in CASES:

    result_folder_name = (
        "online_results_7region_rom_65K_"
        + case_name
    )

    result_folder = os.path.join(
        BASE_DIR, result_folder_name
    )

    manifest_file = os.path.join(
        result_folder,
        "rom_solution_manifest.json",
    )

    generated_script = os.path.join(
        SCRIPT_DIR,
        "_65K_generated_" + case_name + ".py",
    )

    log_file = os.path.join(
        LOG_DIR,
        case_name + ".log",
    )

    # Isolate outputs by creating a temporary complete
    # copy of the original 65H solver with only its
    # result-directory name changed.
    adapted_text = solver_text.replace(
        source_token,
        '"{}"'.format(result_folder_name),
    )

    command = (
        'abaqus python "{}" '
        '--E1 {:.12g} '
        '--E2 {:.12g} '
        '--G12 {:.12g} '
        '--cpus {} '
        '--tol {:.12g} '
        '--rom_max_iter {} '
        '--hf_max_iter {}'
    ).format(
        generated_script,
        E1,
        E2,
        G12,
        args.cpus,
        args.tol,
        args.rom_max_iter,
        args.hf_max_iter,
    )

    print("")
    print(
        "CASE:", case_name,
        "material:", E1, E2, G12
    )

    if args.dry_run:
        print("Would run:", command)
        continue

    # Skip previously finished experiments unless
    # the user explicitly requests --rerun.
    if (
        os.path.isfile(manifest_file)
        and not args.rerun
    ):
        print(
            "Using existing case manifest. "
            "Use --rerun to repeat."
        )

        elapsed = None
        return_code = 0

    else:

        if (
            args.rerun
            and os.path.isfile(manifest_file)
        ):
            # Avoid treating an old successful manifest
            # as the result of a failed new run.
            backup = (
                manifest_file
                + ".before_65K_rerun"
            )

            if os.path.isfile(backup):
                os.remove(backup)

            os.rename(
                manifest_file,
                backup,
            )

        with open(generated_script, "w") as f:
            f.write(adapted_text)

        t0 = time.time()

        try:
            with open(log_file, "w") as log:
                return_code = subprocess.call(
                    command,
                    cwd=BASE_DIR,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

        finally:
            if os.path.isfile(generated_script):
                os.remove(generated_script)

        elapsed = time.time() - t0

    # ========================================================
    # READ SOLVER RESULT
    # ========================================================

    manifest = {}

    if os.path.isfile(manifest_file):
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

    if not manifest:
        print(
            "FAILED: no manifest. See log:",
            log_file,
        )

    else:
        print(
            "Converged:",
            manifest.get("converged"),
            "RMS:",
            manifest.get("final_high_fidelity_rms"),
            "Abaqus jobs:",
            manifest.get("high_fidelity_abaqus_jobs"),
        )

    rows.append({
        "Case": case_name,
        "Source": (
            "new run"
            if elapsed is not None
            else "existing case"
        ),
        "E1_MPa": E1,
        "E2_MPa": E2,
        "G12_MPa": G12,
        "Converged": (
            bool(
                manifest.get("converged", False)
            )
            and return_code == 0
        ),
        "FinalHighFidelityRMS":
            manifest.get(
                "final_high_fidelity_rms", ""
            ),
        "ROMResidualEvaluations":
            manifest.get(
                "rom_residual_evaluations", ""
            ),
        "RealFEMResidualEvaluations":
            manifest.get(
                "high_fidelity_residual_evaluations", ""
            ),
        "AbaqusPatchJobs":
            manifest.get(
                "high_fidelity_abaqus_jobs", ""
            ),
        "ROMSeconds":
            manifest.get(
                "rom_phase_seconds", ""
            ),
        "HighFidelitySeconds":
            manifest.get(
                "high_fidelity_phase_seconds", ""
            ),
        "SolverSeconds":
            manifest.get(
                "total_seconds", ""
            ),
        "ProcessWallSeconds": (
            ""
            if elapsed is None
            else elapsed
        ),
        "ExitCode": return_code,
        "ResultFolder": result_folder,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

if args.dry_run:
    print(
        "DRY RUN COMPLETE. "
        "No Abaqus solves were started."
    )

else:
    destination = os.path.join(
        OUT_DIR,
        "benchmark_results_65K.csv",
    )

    with open(destination, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    new_cases = [
        r for r in rows
        if r["Case"].startswith("case_")
    ]

    successful = sum(
        bool(r["Converged"])
        for r in new_cases
    )

    summary = {
        "new_cases": len(new_cases),
        "new_cases_converged": successful,
        "all_new_cases_converged": (
            successful == len(new_cases)
            and len(new_cases) == len(CASES)
        ),
        "baseline_direct_64T_abaqus_jobs": 54,
        "meaning_of_54":
            "Original baseline reference only; "
            "not a case-matched direct solve.",
        "benchmark_scope":
            "Online equilibrium convergence, "
            "Abaqus patch job counts and runtime; "
            "NOT full-field error for new materials.",
        "old_direct_seconds_if_provided":
            args.old_direct_seconds,
    }

    if (
        args.old_direct_seconds is not None
        and rows
        and rows[0]["Case"]
        == "baseline_existing_65H"
    ):
        new_seconds = float(
            rows[0]["SolverSeconds"]
        )

        summary[
            "baseline_wall_clock_speedup_vs_direct"
        ] = (
            args.old_direct_seconds
            / new_seconds
        )

    with open(
        os.path.join(
            OUT_DIR,
            "benchmark_summary_65K.json",
        ),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print("")
    print("STEP 65K COMPLETE")
    print(json.dumps(summary, indent=2))
    print("CSV:", destination)
