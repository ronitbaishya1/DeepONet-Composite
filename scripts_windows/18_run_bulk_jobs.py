import os
import argparse
import subprocess
import time


# ============================================================
# AUTOMATIC PROJECT ROOT
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
    "--segment",
    choices=[
        "left",
        "right",
    ],
    required=True
)


parser.add_argument(
    "--cpus",
    type=int,
    default=4
)


args = parser.parse_args()


# ============================================================
# INPUT DIRECTORY
# ============================================================

if args.segment == "left":

    INPUT_DIR = os.path.join(
        BASE_DIR,
        "left_inp"
    )

else:

    INPUT_DIR = os.path.join(
        BASE_DIR,
        "right_inp"
    )


if not os.path.isdir(
    INPUT_DIR
):

    raise RuntimeError(
        "Input directory does not exist: {}".format(
            INPUT_DIR
        )
    )


# ============================================================
# INPUT FILES
# ============================================================

inp_files = sorted(
    [
        file_name

        for file_name in os.listdir(
            INPUT_DIR
        )

        if file_name.lower().endswith(
            ".inp"
        )
    ]
)


if len(
    inp_files
) == 0:

    raise RuntimeError(
        "No .inp files found in {}".format(
            INPUT_DIR
        )
    )


print("")
print("========================================")
print("STEP 18 - RUN BULK ABAQUS JOBS")
print("========================================")
print("BASE_DIR  =", BASE_DIR)
print("Segment   =", args.segment)
print("INPUT_DIR =", INPUT_DIR)
print("Jobs      =", len(inp_files))
print("CPUs/job  =", args.cpus)
print("")


# ============================================================
# JOB SUMMARY
# ============================================================

successful_jobs = []

failed_jobs = []

skipped_jobs = []


# ============================================================
# RUN JOBS
# ============================================================

for index, inp_name in enumerate(
    inp_files
):

    job_name = os.path.splitext(
        inp_name
    )[
        0
    ]


    inp_path = os.path.join(
        INPUT_DIR,
        inp_name
    )


    odb_path = os.path.join(
        INPUT_DIR,
        job_name
        +
        ".odb"
    )


    # --------------------------------------------------------
    # SKIP EXISTING ODB
    # --------------------------------------------------------

    if os.path.isfile(
        odb_path
    ):

        print(
            "[{}/{}] {} already has ODB -- skipping.".format(
                index + 1,
                len(
                    inp_files
                ),
                job_name
            )
        )


        skipped_jobs.append(
            job_name
        )


        continue


    # --------------------------------------------------------
    # REMOVE STALE LOCK FILE
    # --------------------------------------------------------

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
        "[{}/{}] Running {}".format(
            index + 1,
            len(
                inp_files
            ),
            job_name
        )
    )


    print(
        "Input:",
        inp_path
    )


    # --------------------------------------------------------
    # ABAQUS COMMAND
    # --------------------------------------------------------

    command = (
        'abaqus job="{}" '
        'input="{}" '
        'cpus={} '
        'interactive'
    ).format(
        job_name,
        inp_name,
        args.cpus
    )


    start_time = time.time()


    return_code = subprocess.call(
        command,
        cwd=INPUT_DIR,
        shell=True
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    # --------------------------------------------------------
    # CHECK RESULT
    # --------------------------------------------------------

    if (
        return_code == 0
        and os.path.isfile(
            odb_path
        )
    ):

        print(
            "SUCCESS: {} | {:.1f} seconds".format(
                job_name,
                elapsed
            )
        )


        successful_jobs.append(
            job_name
        )


    else:

        print(
            "FAILED: {} | return code = {}".format(
                job_name,
                return_code
            )
        )


        # Check .sta and .msg for debugging.
        sta_file = os.path.join(
            INPUT_DIR,
            job_name
            +
            ".sta"
        )


        msg_file = os.path.join(
            INPUT_DIR,
            job_name
            +
            ".msg"
        )


        if os.path.isfile(
            sta_file
        ):

            print(
                "Check:",
                sta_file
            )


        if os.path.isfile(
            msg_file
        ):

            print(
                "Check:",
                msg_file
            )


        failed_jobs.append(
            job_name
        )


# ============================================================
# SUMMARY
# ============================================================

print("")
print("========================================")
print("STEP 18 COMPLETE")
print("========================================")
print(
    "Successful:",
    len(
        successful_jobs
    )
)
print(
    "Skipped existing:",
    len(
        skipped_jobs
    )
)
print(
    "Failed:",
    len(
        failed_jobs
    )
)


if len(
    failed_jobs
) > 0:

    print("")
    print("Failed jobs:")

    for job_name in failed_jobs:

        print(
            "  ",
            job_name
        )


print("")