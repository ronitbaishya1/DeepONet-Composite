from __future__ import print_function

import os
import csv
import argparse

import numpy as np

from odbAccess import openOdb

from abaqusConstants import (
    INTEGRATION_POINT,
)


# ============================================================
# PROJECT ROOT
#
# Expected:
#
# ODBS/
# ├── scripts/
# │   └── 64A_extract_full_reference_mechanics.py
# ├── online_results_ensemble/
# │   └── full_reference/
# │       └── full_reference.odb
# └── ...
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
    "--odb",
    default=None,
    help=(
        "Optional explicit path to full-reference ODB. "
        "If omitted, common project locations are searched."
    ),
)


parser.add_argument(
    "--output",
    default=None,
    help=(
        "Optional output CSV path. "
        "Default = BASE_DIR/full_reference_mechanics.csv"
    ),
)


args = parser.parse_args()


# ============================================================
# LOCATE ODB
# ============================================================

if args.odb is not None:

    ODB_FILE = os.path.abspath(
        args.odb
    )

else:

    candidates = [

        os.path.join(
            BASE_DIR,
            "online_results_ensemble",
            "full_reference",
            "full_reference.odb",
        ),

        os.path.join(
            BASE_DIR,
            "online_results",
            "full_reference",
            "full_reference.odb",
        ),

        os.path.join(
            BASE_DIR,
            "full_reference",
            "full_reference.odb",
        ),
    ]


    ODB_FILE = None


    for candidate in candidates:

        if os.path.isfile(
            candidate
        ):

            ODB_FILE = candidate

            break


if ODB_FILE is None:

    raise RuntimeError(
        "Could not locate full_reference.odb."
    )


if not os.path.isfile(
    ODB_FILE
):

    raise RuntimeError(
        "ODB does not exist: {}".format(
            ODB_FILE
        )
    )


# ============================================================
# OUTPUT
# ============================================================

if args.output is None:

    OUTPUT_FILE = os.path.join(
        BASE_DIR,
        "full_reference_mechanics.csv",
    )

else:

    OUTPUT_FILE = os.path.abspath(
        args.output
    )


# ============================================================
# PRINT PATHS
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64A"
)

print(
    "FULL REFERENCE MECHANICS EXTRACTION"
)

print(
    "=========================================="
)

print(
    "BASE_DIR :",
    BASE_DIR,
)

print(
    "ODB      :",
    ODB_FILE,
)

print(
    "OUTPUT   :",
    OUTPUT_FILE,
)

print("")


# ============================================================
# OPEN ODB
# ============================================================

odb = openOdb(
    ODB_FILE,
    readOnly=True,
)


# ============================================================
# FIND COMPOSITE INSTANCE
# ============================================================

instance = None


for instance_name, candidate in (
    odb.rootAssembly.instances.items()
):

    if instance_name.lower() == (
        "composite-1"
    ):

        instance = candidate

        break


if instance is None:

    odb.close()

    raise RuntimeError(
        "Could not find Composite-1 instance."
    )


# ============================================================
# LOADING STEP
# ============================================================

if "Loading" not in odb.steps:

    odb.close()

    raise RuntimeError(
        "Loading step not found."
    )


step = odb.steps[
    "Loading"
]


if len(
    step.frames
) == 0:

    odb.close()

    raise RuntimeError(
        "Loading step contains no frames."
    )


frame = step.frames[
    -1
]


# ============================================================
# REQUIRED OUTPUTS
# ============================================================

for required_field in [
    "S",
    "LE",
]:

    if required_field not in (
        frame.fieldOutputs
    ):

        odb.close()

        raise RuntimeError(
            "Required field output {} not found."
            .format(
                required_field
            )
        )


# ============================================================
# GET INTEGRATION-POINT OUTPUTS
# ============================================================

S_field = (
    frame.fieldOutputs[
        "S"
    ]
    .getSubset(
        region=
            instance,

        position=
            INTEGRATION_POINT,
    )
)


LE_field = (
    frame.fieldOutputs[
        "LE"
    ]
    .getSubset(
        region=
            instance,

        position=
            INTEGRATION_POINT,
    )
)


print(
    "Stress component labels:",
    S_field.componentLabels,
)


print(
    "Strain component labels:",
    LE_field.componentLabels,
)


# ============================================================
# COMPONENT INDEX MAPS
# ============================================================

def build_component_map(
    component_labels,
    required_labels,
):

    result = {}


    labels = list(
        component_labels
    )


    for required in required_labels:

        if required not in labels:

            raise RuntimeError(
                "Required component {} not found in {}"
                .format(
                    required,
                    labels,
                )
            )


        result[
            required
        ] = labels.index(
            required
        )


    return result


stress_component_map = (
    build_component_map(
        S_field.componentLabels,

        [
            "S11",
            "S22",
            "S33",
            "S12",
            "S13",
            "S23",
        ],
    )
)


strain_component_map = (
    build_component_map(
        LE_field.componentLabels,

        [
            "LE11",
            "LE22",
            "LE33",
            "LE12",
            "LE13",
            "LE23",
        ],
    )
)


# ============================================================
# NODE COORDINATES
# ============================================================

node_coordinates = {}


for node in instance.nodes:

    node_coordinates[
        node.label
    ] = np.asarray(
        node.coordinates,
        dtype=np.float64,
    )


# ============================================================
# ELEMENT CENTROIDS
# ============================================================

element_centroids = {}


for element in instance.elements:

    element_node_coordinates = np.asarray(
        [
            node_coordinates[
                node_label
            ]

            for node_label
            in element.connectivity
        ],
        dtype=np.float64,
    )


    element_centroids[
        element.label
    ] = np.mean(
        element_node_coordinates,
        axis=0,
    )


# ============================================================
# ACCUMULATE FIELD VALUES
#
# C3D8R normally has one integration point.
#
# But this implementation also works if an element has
# multiple integration points: values are averaged.
# ============================================================

def accumulate_element_values(
    field,
):

    storage = {}


    for value in field.values:

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


    averaged = {}


    for label, values in storage.items():

        averaged[
            label
        ] = np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            axis=0,
        )


    return averaged


S_map = accumulate_element_values(
    S_field
)


LE_map = accumulate_element_values(
    LE_field
)


# ============================================================
# COMMON ELEMENTS
# ============================================================

common_labels = sorted(
    set(
        S_map.keys()
    )
    &
    set(
        LE_map.keys()
    )
    &
    set(
        element_centroids.keys()
    )
)


if len(
    common_labels
) == 0:

    odb.close()

    raise RuntimeError(
        "No common stress/strain elements found."
    )


print(
    "Extracting elements:",
    len(
        common_labels
    ),
)


# ============================================================
# WRITE CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
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


    for label in common_labels:

        centroid = element_centroids[
            label
        ]


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
                    stress_component_map[
                        "S11"
                    ]
                ],

                stress[
                    stress_component_map[
                        "S22"
                    ]
                ],

                stress[
                    stress_component_map[
                        "S33"
                    ]
                ],

                stress[
                    stress_component_map[
                        "S12"
                    ]
                ],

                stress[
                    stress_component_map[
                        "S13"
                    ]
                ],

                stress[
                    stress_component_map[
                        "S23"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE11"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE22"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE33"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE12"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE13"
                    ]
                ],

                strain[
                    strain_component_map[
                        "LE23"
                    ]
                ],
            ]
        )


odb.close()


print("")
print(
    "=========================================="
)

print(
    "STEP 64A COMPLETE"
)

print(
    "=========================================="
)

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)