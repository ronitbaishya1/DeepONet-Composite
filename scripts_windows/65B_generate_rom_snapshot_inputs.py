from __future__ import print_function

import os
import csv
import argparse

import numpy as np

from hybrid_runtime_7region import (
    BASE_DIR,
    render_patch_input,
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
    "--limit",
    type=int,
    default=0,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DESIGN_DIR = os.path.join(
    BASE_DIR,
    "rom_designs_7region",
)


OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "rom_snapshot_jobs_7region",
)


OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    args.patch,
)


if not os.path.isdir(
    OUTPUT_DIR
):

    os.makedirs(
        OUTPUT_DIR
    )


DESIGN_FILE = os.path.join(
    DESIGN_DIR,
    "{}_fe_rom_design.csv".format(
        args.patch
    ),
)


# ============================================================
# INTERFACES
# ============================================================

PATCH_INTERFACES = {

    "left": [
        "left_outer",
        "left_inner",
    ],

    "center": [
        "center_left",
        "center_right",
    ],

    "right": [
        "right_inner",
        "right_outer",
    ],
}


# ============================================================
# LOAD DESIGN
# ============================================================

with open(
    DESIGN_FILE,
    "r",
    newline=""
) as file_object:

    rows = list(
        csv.DictReader(
            file_object
        )
    )


if args.limit > 0:

    rows = rows[
        :args.limit
    ]


# ============================================================
# GENERATE
# ============================================================

for row_index, row in enumerate(
    rows
):

    case_id = row[
        "CaseID"
    ]


    E1 = float(
        row[
            "E1_MPa"
        ]
    )


    E2 = float(
        row[
            "E2_MPa"
        ]
    )


    G12 = float(
        row[
            "G12_MPa"
        ]
    )


    coefficient_dictionary = {}


    for interface_name in PATCH_INTERFACES[
        args.patch
    ]:

        values = []


        mode_index = 1


        while True:

            column = (
                "c_{}_{:02d}"
                .format(
                    interface_name,
                    mode_index,
                )
            )


            if column not in row:

                break


            values.append(
                float(
                    row[
                        column
                    ]
                )
            )


            mode_index += 1


        if len(
            values
        ) == 0:

            raise RuntimeError(
                "No coefficients found for {}."
                .format(
                    interface_name
                )
            )


        coefficient_dictionary[
            interface_name
        ] = np.asarray(
            values,
            dtype=np.float64,
        )


    output_file = os.path.join(
        OUTPUT_DIR,
        case_id
        +
        ".inp",
    )


    render_patch_input(

        patch_name=
            args.patch,

        E1=
            E1,

        E2=
            E2,

        G12=
            G12,

        coefficient_dictionary=
            coefficient_dictionary,

        output_file=
            output_file,
    )


    print(
        "[{}/{}] {}"
        .format(
            row_index + 1,
            len(
                rows
            ),
            case_id,
        )
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 65B COMPLETE"
)

print(
    "=========================================="
)

print(
    "Patch:",
    args.patch
)

print(
    "Inputs:",
    len(
        rows
    )
)