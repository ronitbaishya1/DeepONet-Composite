from __future__ import print_function

import os
import csv
import json
import glob
import argparse
import subprocess

import numpy as np

from odbAccess import openOdb

from abaqusConstants import (
    INTEGRATION_POINT,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--cpus",
    type=int,
    default=4,
)


args = parser.parse_args()


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
    "online_results_7region_direct",
)


FINAL_PATCH_DIR = os.path.join(
    RESULTS_DIR,
    "final",
)


VALIDATION_DIR = os.path.join(
    RESULTS_DIR,
    "validation_fields",
)


REFERENCE_RUN_DIR = os.path.join(
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
    REFERENCE_RUN_DIR,
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


    for key, instance in (
        odb.rootAssembly.instances.items()
    ):

        if key.lower() == target:

            return instance


    raise RuntimeError(
        "ODB instance not found: {}".format(
            requested_name
        )
    )


# ============================================================
# AVERAGE IP VALUES BY ELEMENT
# ============================================================

def average_element_values(
    field_output,
):

    storage = {}


    for value in field_output.values:

        label = value.elementLabel


        if label not in storage:

            storage[
                label
            ] = []


        storage[
            label
        ].append(

            np.asarray(
                value.data,
                dtype=np.float64,
            )
        )


    result = {}


    for label, values in (
        storage.items()
    ):

        result[
            label
        ] = np.mean(

            np.asarray(
                values,
                dtype=np.float64,
            ),

            axis=0,
        )


    return result


# ============================================================
# EXTRACT COMPOSITE NODES
# ============================================================

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


    RF_field = frame.fieldOutputs[
        "RF"
    ].getSubset(
        region=instance
    )


    U_map = {}


    for value in U_field.values:

        U_map[
            value.nodeLabel
        ] = value.data


    RF_map = {}


    for value in RF_field.values:

        RF_map[
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
                "RF1",
                "RF2",
                "RF3",
            ]
        )


        for node in instance.nodes:

            displacement = U_map[
                node.label
            ]


            reaction = RF_map.get(

                node.label,

                (
                    0.0,
                    0.0,
                    0.0,
                ),
            )


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

                    reaction[
                        0
                    ],

                    reaction[
                        1
                    ],

                    reaction[
                        2
                    ],
                ]
            )


    number_nodes = len(
        instance.nodes
    )


    odb.close()


    return number_nodes


# ============================================================
# EXTRACT S / LE
# ============================================================

def extract_composite_ip(
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


    S_field = frame.fieldOutputs[
        "S"
    ].getSubset(
        region=instance,
        position=INTEGRATION_POINT,
    )


    LE_field = frame.fieldOutputs[
        "LE"
    ].getSubset(
        region=instance,
        position=INTEGRATION_POINT,
    )


    S_map = average_element_values(
        S_field
    )


    LE_map = average_element_values(
        LE_field
    )


    # ========================================================
    # NODE COORDINATES
    # ========================================================

    node_coordinates = {}


    for node in instance.nodes:

        node_coordinates[
            node.label
        ] = np.asarray(
            node.coordinates,
            dtype=np.float64,
        )


    # ========================================================
    # WRITE ELEMENT-CENTROID VALUES
    # ========================================================

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
                "ElementLabel",
                "X",
                "Y",
                "Z",

                "S11",
                "S22",
                "S33",
                "S12",
                "S13",
                "S23",

                "LE11",
                "LE22",
                "LE33",
                "LE12",
                "LE13",
                "LE23",
            ]
        )


        count = 0


        for element in instance.elements:

            label = element.label


            if (
                label not in S_map
                or
                label not in LE_map
            ):

                continue


            coordinates = np.asarray(
                [
                    node_coordinates[
                        node_label
                    ]

                    for node_label
                    in element.connectivity
                ],
                dtype=np.float64,
            )


            centroid = coordinates.mean(
                axis=0
            )


            stress = S_map[
                label
            ]


            strain = LE_map[
                label
            ]


            writer.writerow(
                [
                    label,

                    centroid[
                        0
                    ],

                    centroid[
                        1
                    ],

                    centroid[
                        2
                    ],

                    stress[
                        0
                    ],

                    stress[
                        1
                    ],

                    stress[
                        2
                    ],

                    stress[
                        3
                    ],

                    stress[
                        4
                    ],

                    stress[
                        5
                    ],

                    strain[
                        0
                    ],

                    strain[
                        1
                    ],

                    strain[
                        2
                    ],

                    strain[
                        3
                    ],

                    strain[
                        4
                    ],

                    strain[
                        5
                    ],
                ]
            )


            count += 1


    odb.close()


    return count


# ============================================================
# NOSE REACTION
# ============================================================

def extract_nose_rf2(
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


    values = []


    for value in RF_field.values:

        values.append(
            float(
                value.data[
                    1
                ]
            )
        )


    odb.close()


    if len(
        values
    ) == 0:

        raise RuntimeError(
            "No nose RF2 values found."
        )


    return sum(
        values
    )


# ============================================================
# EXTRACT THREE FINAL PATCHES
# ============================================================

patch_summary = {}


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


    node_file = os.path.join(
        VALIDATION_DIR,
        "{}_nodes.csv".format(
            patch_name
        ),
    )


    ip_file = os.path.join(
        VALIDATION_DIR,
        "{}_ip.csv".format(
            patch_name
        ),
    )


    number_nodes = extract_composite_nodes(
        odb_file,
        node_file,
    )


    number_elements = extract_composite_ip(
        odb_file,
        ip_file,
    )


    patch_summary[
        patch_name
    ] = {
        "nodes":
            int(
                number_nodes
            ),

        "elements":
            int(
                number_elements
            ),
    }


    print(
        "Extracted:",
        patch_name,
        "| nodes:",
        number_nodes,
        "| elements:",
        number_elements,
    )


# ============================================================
# HYBRID NOSE REACTION
# ============================================================

hybrid_nose_rf2 = extract_nose_rf2(

    os.path.join(
        FINAL_PATCH_DIR,
        "center.odb",
    )
)


# ============================================================
# FIND EXISTING FULL REFERENCE ODB
# ============================================================

REFERENCE_CANDIDATES = [

    os.path.join(
        BASE_DIR,
        "online_results",
        "full_reference",
        "full_reference.odb",
    ),

    os.path.join(
        BASE_DIR,
        "online_results_archive",
        "baseline_original",
        "full_reference",
        "full_reference.odb",
    ),

    os.path.join(
        REFERENCE_RUN_DIR,
        "full_reference_7region.odb",
    ),
]


reference_odb = None


for candidate in REFERENCE_CANDIDATES:

    if os.path.isfile(
        candidate
    ):

        reference_odb = candidate

        break


# ============================================================
# IF NECESSARY, RUN FULL FEM REFERENCE
# ============================================================

if reference_odb is None:

    if not os.path.isfile(
        BASELINE_INP
    ):

        raise FileNotFoundError(
            BASELINE_INP
        )


    job_name = (
        "full_reference_7region"
    )


    for path in glob.glob(
        os.path.join(
            REFERENCE_RUN_DIR,
            job_name
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
        'cpus={} '
        'interactive'
    ).format(
        job_name,
        BASELINE_INP,
        args.cpus,
    )


    print("")
    print(
        "Running full FEM reference..."
    )


    return_code = subprocess.call(
        command,
        cwd=REFERENCE_RUN_DIR,
        shell=True,
    )


    reference_odb = os.path.join(
        REFERENCE_RUN_DIR,
        job_name
        +
        ".odb",
    )


    if (
        return_code != 0
        or
        not os.path.isfile(
            reference_odb
        )
    ):

        raise RuntimeError(
            "Full reference FEM job failed."
        )


else:

    print("")
    print(
        "Using existing full reference:"
    )

    print(
        reference_odb
    )


# ============================================================
# EXTRACT FULL FEM
# ============================================================

reference_node_file = os.path.join(
    VALIDATION_DIR,
    "full_reference_nodes.csv",
)


reference_ip_file = os.path.join(
    VALIDATION_DIR,
    "full_reference_ip.csv",
)


reference_nodes = extract_composite_nodes(
    reference_odb,
    reference_node_file,
)


reference_elements = extract_composite_ip(
    reference_odb,
    reference_ip_file,
)


full_nose_rf2 = extract_nose_rf2(
    reference_odb
)


# ============================================================
# REACTION SUMMARY
# ============================================================

absolute_reaction_error = abs(
    hybrid_nose_rf2
    -
    full_nose_rf2
)


relative_reaction_error = (
    absolute_reaction_error
    /
    (
        abs(
            full_nose_rf2
        )
        +
        1.0e-14
    )
)


reaction_summary = {

    "hybrid_center_nose_RF2_N":
        hybrid_nose_rf2,

    "full_FEM_nose_RF2_N":
        full_nose_rf2,

    "absolute_reaction_error_N":
        absolute_reaction_error,

    "relative_reaction_error":
        relative_reaction_error,

    "relative_reaction_error_percent":
        100.0
        *
        relative_reaction_error,
}


with open(
    os.path.join(
        VALIDATION_DIR,
        "reaction_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        reaction_summary,
        file_object,
        indent=4,
    )


# ============================================================
# EXTRACTION SUMMARY
# ============================================================

extraction_summary = {

    "patches":
        patch_summary,

    "full_reference_nodes":
        int(
            reference_nodes
        ),

    "full_reference_elements":
        int(
            reference_elements
        ),

    "reference_odb":
        reference_odb,
}


with open(
    os.path.join(
        VALIDATION_DIR,
        "extraction_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        extraction_summary,
        file_object,
        indent=4,
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64U COMPLETE"
)

print(
    "=========================================="
)

print(
    "Full FEM nodes:",
    reference_nodes
)

print(
    "Full FEM elements:",
    reference_elements
)

print(
    "Hybrid nose RF2:",
    hybrid_nose_rf2
)

print(
    "Full FEM nose RF2:",
    full_nose_rf2
)

print(
    "Reaction error (%):",
    100.0
    *
    relative_reaction_error
)

print(
    "Output:"
)

print(
    VALIDATION_DIR
)