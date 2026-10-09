from __future__ import print_function

import os
import csv
import json
import glob
import subprocess

from odbAccess import openOdb


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


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results_ensemble",
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
    ) as file_object:

        writer = csv.writer(
            file_object
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

                    node.coordinates[0],
                    node.coordinates[1],
                    node.coordinates[2],

                    displacement[0],
                    displacement[1],
                    displacement[2],
                ]
            )


    odb.close()


def extract_nose_reaction(
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


    reaction = None


    for value in RF_field.values:

        if value.nodeLabel == 4852:

            reaction = float(
                value.data[1]
            )

            break


    odb.close()


    return reaction


# ============================================================
# EXTRACT ENSEMBLE HYBRID PATCHES
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
        "Extracted:",
        patch_name
    )


hybrid_nose_reaction = (
    extract_nose_reaction(
        os.path.join(
            FINAL_PATCH_DIR,
            "center.odb",
        )
    )
)


# ============================================================
# REUSE OLD FULL-FEM REFERENCE IF AVAILABLE
# ============================================================

old_reference_odb = os.path.join(
    BASE_DIR,
    "online_results",
    "full_reference",
    "full_reference.odb",
)


if os.path.isfile(
    old_reference_odb
):

    reference_odb = old_reference_odb


    print(
        "Reusing existing full FEM reference:"
    )

    print(
        reference_odb
    )


else:

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


        print(
            "Running full FEM reference..."
        )


        return_code = subprocess.call(
            command,
            cwd=REFERENCE_DIR,
            shell=True,
        )


        if (
            return_code != 0

            or

            not os.path.isfile(
                reference_odb
            )
        ):

            raise RuntimeError(
                "Full reference FEM failed."
            )


# ============================================================
# EXTRACT REFERENCE
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
    extract_nose_reaction(
        reference_odb
    )
)


# ============================================================
# REACTION SUMMARY
# ============================================================

absolute_error = abs(
    hybrid_nose_reaction
    -
    full_nose_reaction
)


relative_error = (

    absolute_error

    /

    (
        abs(
            full_nose_reaction
        )

        +

        1.0e-14
    )
)


summary = {

    "hybrid_center_nose_RF2_N":
        hybrid_nose_reaction,

    "full_FEM_nose_RF2_N":
        full_nose_reaction,

    "absolute_reaction_error_N":
        absolute_error,

    "relative_reaction_error":
        relative_error,
}


with open(
    os.path.join(
        VALIDATION_DIR,
        "reaction_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        summary,
        file_object,
        indent=4,
    )


print("")
print(
    "================================================"
)

print(
    "STEP 59B COMPLETE"
)

print(
    "================================================"
)


print(
    "Hybrid nose reaction:",
    hybrid_nose_reaction
)


print(
    "Full FEM reaction:",
    full_nose_reaction
)


print(
    "Relative reaction error:",
    relative_error
)