from __future__ import print_function

import os
import csv
import argparse
import subprocess


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
    default=None,
    help="Optional number of jobs to run for smoke testing.",
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


INP_DIR = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
    "{}_inp".format(
        args.segment
    ),
)


STATUS_FILE = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
    "{}_run_status.csv".format(
        args.segment
    ),
)


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not os.path.isdir(
    INP_DIR
):

    raise RuntimeError(
        "Input directory not found:\n{}".format(
            INP_DIR
        )
    )


# ============================================================
# FIND INPUT FILES
# ============================================================

inp_files = sorted(
    [
        filename

        for filename in os.listdir(
            INP_DIR
        )

        if filename.lower().endswith(
            ".inp"
        )
    ]
)


if len(
    inp_files
) == 0:

    raise RuntimeError(
        "No .inp files found in:\n{}".format(
            INP_DIR
        )
    )


if args.limit is not None:

    inp_files = inp_files[
        :args.limit
    ]


# ============================================================
# CHECK WHETHER A JOB REALLY COMPLETED
# ============================================================

def analysis_completed(
    case_id,
):

    sta_file = os.path.join(
        INP_DIR,
        "{}.sta".format(
            case_id
        ),
    )


    if not os.path.isfile(
        sta_file
    ):

        return False


    try:

        with open(
            sta_file,
            "r"
        ) as file_object:

            text = (
                file_object
                .read()
                .upper()
            )


    except Exception:

        return False


    success_phrases = [

        "THE ANALYSIS HAS COMPLETED SUCCESSFULLY",

        "ANALYSIS COMPLETED SUCCESSFULLY",
    ]


    for phrase in success_phrases:

        if phrase in text:

            return True


    return False


# ============================================================
# REMOVE OLD FAILED OUTPUTS
#
# IMPORTANT:
#
# We DO NOT delete the .inp file.
#
# We only delete old Abaqus result files so the corrected
# input file can be rerun cleanly.
# ============================================================

def remove_old_results(
    case_id,
):

    extensions = [

        ".odb",
        ".odb_f",
        ".sta",
        ".msg",
        ".dat",
        ".com",
        ".prt",
        ".sim",
        ".stt",
        ".mdl",
        ".pac",
        ".res",
        ".sel",
        ".ipm",
        ".log",
        ".lck",
    ]


    removed_anything = False


    for extension in extensions:

        filename = os.path.join(
            INP_DIR,
            case_id
            +
            extension,
        )


        if os.path.isfile(
            filename
        ):

            try:

                os.remove(
                    filename
                )


                print(
                    "Removed old file:",
                    os.path.basename(
                        filename
                    )
                )


                removed_anything = True


            except Exception as error:

                print(
                    "Could not delete:",
                    filename
                )


                print(
                    str(
                        error
                    )
                )


    return removed_anything


# ============================================================
# SAVE STATUS FILE
# ============================================================

def save_status(
    rows,
):

    with open(
        STATUS_FILE,
        "w",
        newline=""
    ) as file_object:

        writer = csv.writer(
            file_object
        )


        writer.writerow(
            [
                "CaseID",
                "Status",
                "ReturnCode",
            ]
        )


        writer.writerows(
            rows
        )


# ============================================================
# START
# ============================================================

print("")
print(
    "Input directory:"
)

print(
    INP_DIR
)


print("")
print(
    "Jobs to run:",
    len(
        inp_files
    )
)


status_rows = []


# ============================================================
# LOOP OVER JOBS
# ============================================================

for index, inp_filename in enumerate(
    inp_files
):

    case_id = os.path.splitext(
        inp_filename
    )[0]


    inp_file = os.path.join(
        INP_DIR,
        inp_filename,
    )


    odb_file = os.path.join(
        INP_DIR,
        "{}.odb".format(
            case_id
        ),
    )


    print("")
    print(
        "================================================"
    )


    print(
        "[{}/{}] {}".format(
            index + 1,
            len(
                inp_files
            ),
            case_id,
        )
    )


    print(
        "================================================"
    )


    # ========================================================
    # INPUT FILE CHECK
    # ========================================================

    if not os.path.isfile(
        inp_file
    ):

        print(
            "ERROR: input file missing."
        )


        status_rows.append(
            [
                case_id,
                "missing_inp",
                "",
            ]
        )


        save_status(
            status_rows
        )


        continue


    # ========================================================
    # ONLY SKIP IF JOB REALLY FINISHED SUCCESSFULLY
    # ========================================================

    if (
        os.path.isfile(
            odb_file
        )
        and
        analysis_completed(
            case_id
        )
    ):

        print(
            "Job already completed successfully. Skipping."
        )


        status_rows.append(
            [
                case_id,
                "existing_success",
                0,
            ]
        )


        save_status(
            status_rows
        )


        continue


    # ========================================================
    # OLD ODB EXISTS BUT JOB FAILED
    #
    # Remove the old partial result files automatically.
    # ========================================================

    if os.path.isfile(
        odb_file
    ):

        print(
            "Old ODB exists, but analysis was not successful."
        )


        print(
            "Removing old failed result files..."
        )


    remove_old_results(
        case_id
    )


    # ========================================================
    # WINDOWS ABAQUS COMMAND
    #
    # We run through cmd /c because Abaqus on Windows is
    # normally launched through a command/batch wrapper.
    # ========================================================

    command = (

        'cmd /c abaqus '
        'job="{job}" '
        'input="{inp}" '
        'cpus={cpus} '
        'interactive'

    ).format(

        job=
            case_id,

        inp=
            inp_filename,

        cpus=
            args.cpus,
    )


    print("")
    print(
        "Running:"
    )


    print(
        command
    )


    # ========================================================
    # RUN ABAQUS
    # ========================================================

    try:

        return_code = subprocess.call(

            command,

            cwd=
                INP_DIR,

            shell=True,
        )


    except Exception as error:

        print("")
        print(
            "FAILED TO START ABAQUS"
        )


        print(
            str(
                error
            )
        )


        status_rows.append(
            [
                case_id,
                "launch_failed",
                "",
            ]
        )


        save_status(
            status_rows
        )


        continue


    # ========================================================
    # CHECK TRUE SUCCESS
    # ========================================================

    odb_exists = os.path.isfile(
        odb_file
    )


    completed = analysis_completed(
        case_id
    )


    if (
        return_code == 0
        and
        odb_exists
        and
        completed
    ):

        status = "success"


        print("")
        print(
            "SUCCESS"
        )


        print(
            "ODB:"
        )


        print(
            odb_file
        )


    else:

        status = "failed"


        print("")
        print(
            "FAILED"
        )


        print(
            "Return code:",
            return_code
        )


        print(
            "ODB exists:",
            odb_exists
        )


        print(
            "STA confirms successful completion:",
            completed
        )


        print("")
        print(
            "Check these files:"
        )


        print(
            os.path.join(
                INP_DIR,
                "{}.dat".format(
                    case_id
                ),
            )
        )


        print(
            os.path.join(
                INP_DIR,
                "{}.msg".format(
                    case_id
                ),
            )
        )


        print(
            os.path.join(
                INP_DIR,
                "{}.sta".format(
                    case_id
                ),
            )
        )


    status_rows.append(
        [
            case_id,
            status,
            return_code,
        ]
    )


    save_status(
        status_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

successful_rows = [

    row

    for row in status_rows

    if row[
        1
    ] in [
        "success",
        "existing_success",
    ]
]


failed_rows = [

    row

    for row in status_rows

    if row[
        1
    ]
    not in [
        "success",
        "existing_success",
    ]
]


print("")
print(
    "================================================"
)

print(
    "ADAPTIVE FEM RUN COMPLETE"
)

print(
    "================================================"
)


print(
    "Successful/existing:",
    len(
        successful_rows
    )
)


print(
    "Failed:",
    len(
        failed_rows
    )
)


print("")
print(
    "Status file:"
)


print(
    STATUS_FILE
)