from __future__ import print_function

import os
import csv
import json
import glob
import shutil
import subprocess

import numpy as np

from odbAccess import openOdb


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


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results",
)


FINAL_PATCH_DIR = os.path.join(
    RESULTS_DIR,
    "final",
)


VALIDATION_DIR = os.path.join(
    RESULTS_DIR,
    "validation_fields",
)


REFERENCE_DIR = os.path.join(
    RESULTS_DIR,
    "full_reference",
)


BASELINE_INP = os.path.join(
    BASE_DIR,
    "baseline",
    "3Point.inp",
)


for directory in [
    VALIDATION_DIR,
    REFERENCE_DIR,
]:

    if not os.path.isdir(
        directory
    ):

        os.makedirs(
            directory
        )


# ============================================================
# HELPERS
# ============================================================

def find_instance(
    odb,
    requested_name,
):

    target = requested_name.lower()


    for key, value in odb.rootAssembly.instances.items():

        if key.lower() == target:

            return value


    raise RuntimeError(
        "Instance not found: {}".format(
            requested_name
        )
    )


def extract_composite_nodes(
    odb_file,
    output_csv,
):

    odb = openOdb(
        odb_file,
        readOnly=True,
    )


    instance = find_instance(
        odb,
        "Composite-1",
    )


    frame = odb.steps[
        "Loading"
    ].frames[
        -1
    ]


    U_field = frame.fieldOutputs[
        "U"
    ].getSubset(
        region=instance
    )


    U_map = {}


    for value in U_field.values:

        U_map[
            value.nodeLabel
        ] = value.data


    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "NodeLabel",
                "X",
                "Y",
                "Z",
                "U1",
                "U2",
                "U3",
            ]
        )


        for node in instance.nodes:

            displacement = U_map[
                node.label
            ]


            writer.writerow(
                [
                    node.label,

                    node.coordinates[
                        0
                    ],

                    node.coordinates[
                        1
                    ],

                    node.coordinates[
                        2
                    ],

                    displacement[
                        0
                    ],

                    displacement[
                        1
                    ],

                    displacement[
                        2
                    ],
                ]
            )


    odb.close()


def extract_reference_nose_reaction(
    odb_file,
):

    odb = openOdb(
        odb_file,
        readOnly=True,
    )


    instance = find_instance(
        odb,
        "Point-2",
    )


    frame = odb.steps[
        "Loading"
    ].frames[
        -1
    ]


    RF_field = frame.fieldOutputs[
        "RF"
    ].getSubset(
        region=instance
    )


    value_found = None


    for value in RF_field.values:

        if value.nodeLabel == 4852:

            value_found = float(
                value.data[
                    1
                ]
            )

            break


    odb.close()


    return value_found


# ============================================================
# EXTRACT FINAL HYBRID PATCHES
# ============================================================

for patch_name in [
    "left",
    "center",
    "right",
]:

    odb_file = os.path.join(
        FINAL_PATCH_DIR,
        "{}.odb".format(
            patch_name
        ),
    )


    if not os.path.isfile(
        odb_file
    ):

        raise FileNotFoundError(
            odb_file
        )


    output_csv = os.path.join(
        VALIDATION_DIR,
        "{}_nodes.csv".format(
            patch_name
        ),
    )


    extract_composite_nodes(
        odb_file,
        output_csv,
    )


    print(
        "Extracted hybrid patch:",
        patch_name,
    )


# ============================================================
# CENTER PATCH NOSE REACTION
# ============================================================

hybrid_nose_reaction = (
    extract_reference_nose_reaction(
        os.path.join(
            FINAL_PATCH_DIR,
            "center.odb",
        )
    )
)


# ============================================================
# RUN FULL BASELINE REFERENCE
# ============================================================

reference_job = "full_reference"


reference_odb = os.path.join(
    REFERENCE_DIR,
    reference_job
    +
    ".odb",
)


if not os.path.isfile(
    reference_odb
):

    # Remove stale files
    for path in glob.glob(
        os.path.join(
            REFERENCE_DIR,
            reference_job
            +
            ".*"
        )
    ):

        try:

            os.remove(
                path
            )

        except Exception:

            pass


    command = (
        'abaqus job="{}" '
        'input="{}" '
        'cpus=4 '
        'interactive'
    ).format(
        reference_job,
        BASELINE_INP,
    )


    print("")
    print(
        "Running complete baseline FEM reference..."
    )


    return_code = subprocess.call(
        command,
        cwd=REFERENCE_DIR,
        shell=True,
    )


    if (
        return_code != 0
        or not os.path.isfile(
            reference_odb
        )
    ):

        raise RuntimeError(
            "Full reference FEM job failed."
        )


# ============================================================
# EXTRACT FULL REFERENCE
# ============================================================

reference_csv = os.path.join(
    VALIDATION_DIR,
    "full_reference_nodes.csv",
)


extract_composite_nodes(
    reference_odb,
    reference_csv,
)


full_nose_reaction = (
    extract_reference_nose_reaction(
        reference_odb
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "hybrid_center_nose_RF2_N":
        hybrid_nose_reaction,

    "full_FEM_nose_RF2_N":
        full_nose_reaction,

    "absolute_reaction_error_N":
        abs(
            hybrid_nose_reaction
            -
            full_nose_reaction
        ),

    "relative_reaction_error":
        abs(
            hybrid_nose_reaction
            -
            full_nose_reaction
        )
        /
        (
            abs(
                full_nose_reaction
            )
            +
            1.0e-14
        ),
}


summary_file = os.path.join(
    VALIDATION_DIR,
    "reaction_summary.json",
)


with open(
    summary_file,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


print("")
print(
    "============================================"
)

print(
    "VALIDATION EXTRACTION COMPLETE"
)

print(
    "============================================"
)

print(
    "Hybrid nose RF2:",
    hybrid_nose_reaction,
)

print(
    "Full FEM nose RF2:",
    full_nose_reaction,
)

print(
    "Relative reaction error:",
    summary[
        "relative_reaction_error"
    ],
)

print(
    VALIDATION_DIR
)