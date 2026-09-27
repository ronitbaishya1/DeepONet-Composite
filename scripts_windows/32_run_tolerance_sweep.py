from __future__ import print_function

import os
import shutil
import subprocess


# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


SOLVER_FILE = os.path.join(
    SCRIPT_DIR,
    "28_solve_online_hybrid.py",
)


CURRENT_RESULTS = os.path.join(
    BASE_DIR,
    "online_results",
)


ARCHIVE_ROOT = os.path.join(
    BASE_DIR,
    "online_results_archive",
)


BASELINE_BACKUP = os.path.join(
    ARCHIVE_ROOT,
    "baseline_original",
)


if not os.path.isdir(
    ARCHIVE_ROOT
):

    os.makedirs(
        ARCHIVE_ROOT
    )


# ============================================================
# BACK UP EXISTING SUCCESSFUL BASELINE
# ============================================================

if (
    os.path.isdir(
        CURRENT_RESULTS
    )
    and
    not os.path.isdir(
        BASELINE_BACKUP
    )
):

    print(
        "Backing up current successful baseline..."
    )

    shutil.copytree(
        CURRENT_RESULTS,
        BASELINE_BACKUP,
    )


# ============================================================
# TOLERANCES
# ============================================================

tolerances = [
    0.02,
    0.01,
    0.005,
]


for tolerance in tolerances:

    tag = (
        "tol_{:04d}".format(
            int(
                round(
                    tolerance
                    *
                    10000
                )
            )
        )
    )


    archive_directory = os.path.join(
        ARCHIVE_ROOT,
        tag,
    )


    if os.path.isdir(
        archive_directory
    ):

        print("")
        print(
            "Already exists, skipping:",
            archive_directory,
        )

        continue


    if os.path.isdir(
        CURRENT_RESULTS
    ):

        shutil.rmtree(
            CURRENT_RESULTS
        )


    command = (
        'abaqus python "{}" '
        '--E1 45000 '
        '--E2 12000 '
        '--G12 4500 '
        '--cpus 4 '
        '--max_iter 20 '
        '--tol {}'
    ).format(
        SOLVER_FILE,
        tolerance,
    )


    print("")
    print(
        "============================================"
    )

    print(
        "RUNNING TOLERANCE:",
        tolerance,
    )

    print(
        "============================================"
    )


    return_code = subprocess.call(
        command,
        cwd=BASE_DIR,
        shell=True,
    )


    if return_code != 0:

        raise RuntimeError(
            "Tolerance run failed: {}".format(
                tolerance
            )
        )


    if not os.path.isdir(
        CURRENT_RESULTS
    ):

        raise RuntimeError(
            "Expected online_results was not created."
        )


    shutil.move(
        CURRENT_RESULTS,
        archive_directory,
    )


    print(
        "Archived:",
        archive_directory,
    )


# ============================================================
# RESTORE ORIGINAL BASELINE
# ============================================================

if os.path.isdir(
    CURRENT_RESULTS
):

    shutil.rmtree(
        CURRENT_RESULTS
    )


if os.path.isdir(
    BASELINE_BACKUP
):

    shutil.copytree(
        BASELINE_BACKUP,
        CURRENT_RESULTS,
    )


print("")
print(
    "============================================"
)

print(
    "TOLERANCE SWEEP COMPLETE"
)

print(
    "============================================"
)

print(
    "Archived runs:"
)

print(
    ARCHIVE_ROOT
)

print("")
print(
    "Original successful baseline restored to:"
)

print(
    CURRENT_RESULTS
)