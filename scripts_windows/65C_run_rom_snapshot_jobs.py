from __future__ import print_function

import os
import csv
import time
import argparse
import subprocess


# ============================================================
# ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--patch",
    choices=[
        "left",
        "center",
        "right",
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
)


args = parser.parse_args()


JOB_DIR = os.path.join(
    BASE_DIR,
    "rom_snapshot_jobs_7region",
    args.patch,
)


SUMMARY_DIR = os.path.join(
    BASE_DIR,
    "rom_snapshot_summaries_7region",
)


if not os.path.isdir(
    SUMMARY_DIR
):

    os.makedirs(
        SUMMARY_DIR
    )


# ============================================================
# FILES
# ============================================================

inp_files = sorted(
    [
        filename

        for filename
        in os.listdir(
            JOB_DIR
        )

        if filename.lower().endswith(
            ".inp"
        )
    ]
)


if args.limit > 0:

    inp_files = inp_files[
        :args.limit
    ]


# ============================================================
# COMPLETION CHECK
# ============================================================

def successful(
    sta_file,
):

    if not os.path.isfile(
        sta_file
    ):

        return False


    with open(
        sta_file,
        "r"
    ) as file_object:

        text = (
            file_object
            .read()
            .upper()
        )


    return (
        "COMPLETED SUCCESSFULLY"
        in
        text
    )


# ============================================================
# RUN
# ============================================================

summary = []


for index, inp_name in enumerate(
    inp_files
):

    job_name = os.path.splitext(
        inp_name
    )[0]


    odb_file = os.path.join(
        JOB_DIR,
        job_name
        +
        ".odb",
    )


    sta_file = os.path.join(
        JOB_DIR,
        job_name
        +
        ".sta",
    )


    if (
        os.path.isfile(
            odb_file
        )
        and
        successful(
            sta_file
        )
    ):

        print(
            "[{}/{}] {} already complete."
            .format(
                index + 1,
                len(
                    inp_files
                ),
                job_name,
            )
        )


        summary.append(
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


    start = time.time()


    return_code = subprocess.call(
        command,
        cwd=JOB_DIR,
        shell=True,
    )


    elapsed = (
        time.time()
        -
        start
    )


    status = (
        "SUCCESS"

        if (
            return_code == 0
            and
            os.path.isfile(
                odb_file
            )
            and
            successful(
                sta_file
            )
        )

        else
        "FAILED"
    )


    print(
        status,
        "|",
        elapsed,
        "seconds"
    )


    summary.append(
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
# SAVE
# ============================================================

output_file = os.path.join(
    SUMMARY_DIR,
    "{}_snapshot_run_summary.csv".format(
        args.patch
    ),
)


with open(
    output_file,
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


    writer.writerows(
        summary
    )


failures = [
    row

    for row
    in summary

    if row[
        "Status"
    ]
    ==
    "FAILED"
]


print("")
print(
    "=========================================="
)

print(
    "STEP 65C COMPLETE"
)

print(
    "=========================================="
)

print(
    "Patch:",
    args.patch
)

print(
    "Failures:",
    len(
        failures
    )
)