from __future__ import print_function

import os
import csv
import time
import argparse
import subprocess


# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


INPUT_ROOT = os.path.join(
    BASE_DIR,
    "candidate_inp_7region"
)


SUMMARY_ROOT = os.path.join(
    BASE_DIR,
    "candidate_job_summaries_7region"
)


os.makedirs(
    SUMMARY_ROOT,
    exist_ok=True
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "outer_left",
        "inner_left",
        "inner_right",
        "outer_right",
    ],
    required=True,
)


parser.add_argument(
    "--cpus",
    type=int,
    default=4,
)


parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help=(
        "0 = run all available INPs. "
        "Use --limit 1 for smoke test."
    ),
)


args = parser.parse_args()


# ============================================================
# INPUT DIRECTORY
# ============================================================

INPUT_DIR = os.path.join(
    INPUT_ROOT,
    args.segment
)


if not os.path.isdir(
    INPUT_DIR
):

    raise RuntimeError(
        "Missing input directory: {}".format(
            INPUT_DIR
        )
    )


inp_files = sorted(
    [
        file_name

        for file_name
        in os.listdir(
            INPUT_DIR
        )

        if file_name.lower().endswith(
            ".inp"
        )
    ]
)


if args.limit > 0:

    inp_files = inp_files[
        :args.limit
    ]


if len(
    inp_files
) == 0:

    raise RuntimeError(
        "No INP files found."
    )


# ============================================================
# STATUS CHECK
# ============================================================

def analysis_completed(
    sta_file
):

    if not os.path.isfile(
        sta_file
    ):

        return False


    try:

        with open(
            sta_file,
            "r"
        ) as file_object:

            contents = (
                file_object
                .read()
                .upper()
            )


        return (
            "COMPLETED SUCCESSFULLY"
            in
            contents
        )


    except Exception:

        return False


# ============================================================
# RUN
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64G — RUN 7-REGION ABAQUS JOBS"
)

print(
    "=========================================="
)

print(
    "Segment:",
    args.segment
)

print(
    "Jobs:",
    len(
        inp_files
    )
)

print(
    "CPUs/job:",
    args.cpus
)

print("")


results = []


for index, inp_name in enumerate(
    inp_files
):

    job_name = os.path.splitext(
        inp_name
    )[0]


    odb_file = os.path.join(
        INPUT_DIR,
        job_name
        +
        ".odb"
    )


    sta_file = os.path.join(
        INPUT_DIR,
        job_name
        +
        ".sta"
    )


    # ========================================================
    # EXISTING SUCCESSFUL JOB
    # ========================================================

    if (
        os.path.isfile(
            odb_file
        )
        and
        analysis_completed(
            sta_file
        )
    ):

        print(
            "[{}/{}] {} already complete — skipping."
            .format(
                index + 1,
                len(
                    inp_files
                ),
                job_name,
            )
        )


        results.append(
            {
                "Job":
                    job_name,

                "Status":
                    "SKIPPED_COMPLETE",

                "Seconds":
                    0.0,
            }
        )


        continue


    # ========================================================
    # PROTECT AGAINST STALE FAILED ODB
    # ========================================================

    if (
        os.path.isfile(
            odb_file
        )
        and
        not analysis_completed(
            sta_file
        )
    ):

        raise RuntimeError(
            (
                "Job {} already has an ODB but its STA does "
                "not show successful completion. "
                "Inspect/remove its stale Abaqus files before "
                "rerunning."
            ).format(
                job_name
            )
        )


    # ========================================================
    # REMOVE STALE LOCK
    # ========================================================

    lock_file = os.path.join(
        INPUT_DIR,
        job_name
        +
        ".lck"
    )


    if os.path.isfile(
        lock_file
    ):

        try:

            os.remove(
                lock_file
            )

        except Exception:

            pass


    print("")
    print(
        "[{}/{}] Running {}"
        .format(
            index + 1,
            len(
                inp_files
            ),
            job_name,
        )
    )


    command = (
        'abaqus job="{}" '
        'input="{}" '
        'cpus={} '
        'interactive'
    ).format(
        job_name,
        inp_name,
        args.cpus,
    )


    start_time = time.time()


    return_code = subprocess.call(
        command,
        cwd=
            INPUT_DIR,
        shell=True,
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    success = (
        return_code == 0
        and
        os.path.isfile(
            odb_file
        )
        and
        analysis_completed(
            sta_file
        )
    )


    if success:

        status = "SUCCESS"


        print(
            "SUCCESS: {} | {:.2f} s"
            .format(
                job_name,
                elapsed,
            )
        )


    else:

        status = "FAILED"


        print(
            "FAILED:",
            job_name
        )


        print(
            "Return code:",
            return_code
        )


        if os.path.isfile(
            sta_file
        ):

            print(
                "Check:",
                sta_file
            )


    results.append(
        {
            "Job":
                job_name,

            "Status":
                status,

            "Seconds":
                elapsed,
        }
    )


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

summary_file = os.path.join(
    SUMMARY_ROOT,
    "{}_run_summary.csv".format(
        args.segment
    )
)


with open(
    summary_file,
    "w",
    newline=""
) as file_object:

    writer = csv.DictWriter(
        file_object,
        fieldnames=[
            "Job",
            "Status",
            "Seconds",
        ],
    )


    writer.writeheader()


    for row in results:

        writer.writerow(
            row
        )


number_success = len(
    [
        row
        for row
        in results
        if row[
            "Status"
        ]
        in [
            "SUCCESS",
            "SKIPPED_COMPLETE",
        ]
    ]
)


number_failed = len(
    results
) - number_success


print("")
print(
    "=========================================="
)

print(
    "STEP 64G COMPLETE"
)

print(
    "=========================================="
)

print(
    "Successful/complete:",
    number_success
)

print(
    "Failed:",
    number_failed
)

print(
    "Summary:",
    summary_file
)

print("")