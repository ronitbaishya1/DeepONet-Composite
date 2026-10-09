from __future__ import print_function

import os
import csv
import json
import argparse

import numpy as np

from odbAccess import openOdb


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


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DESIGN_FILE = os.path.join(
    BASE_DIR,
    "rom_designs_7region",
    "{}_fe_rom_design.csv".format(
        args.patch
    ),
)


JOB_DIR = os.path.join(
    BASE_DIR,
    "rom_snapshot_jobs_7region",
    args.patch,
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "rom_snapshots_7region",
    args.patch,
)


PATCH_METADATA_FILE = os.path.join(
    BASE_DIR,
    "patch_templates_7region",
    "patch_metadata.json",
)


INTERFACE_DIR = os.path.join(
    BASE_DIR,
    "interfaces_7region",
)


if not os.path.isdir(
    OUTPUT_DIR
):

    os.makedirs(
        OUTPUT_DIR
    )


# ============================================================
# PATCH CONFIGURATION
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
# LOAD METADATA
# ============================================================

with open(
    PATCH_METADATA_FILE,
    "r"
) as file_object:

    patch_metadata = json.load(
        file_object
    )[
        "patches"
    ][
        args.patch
    ]


# ============================================================
# LOAD DESIGN
# ============================================================

with open(
    DESIGN_FILE,
    "r",
    newline=""
) as file_object:

    design_rows = list(
        csv.DictReader(
            file_object
        )
    )


# ============================================================
# FIND INSTANCE
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
        "Instance not found: {}"
        .format(
            requested_name
        )
    )


# ============================================================
# LOAD INTERFACE BASES
# ============================================================

interface_bases = {}


for interface_name in PATCH_INTERFACES[
    args.patch
]:

    data = np.load(
        os.path.join(
            INTERFACE_DIR,
            "interface_{}.npz".format(
                interface_name
            ),
        )
    )


    interface_bases[
        interface_name
    ] = data[
        "basis"
    ].astype(
        np.float64
    )


# ============================================================
# STORAGE
# ============================================================

all_U = []

all_g = []

all_reaction = []

successful_case_ids = []


reference_node_labels = None

reference_coordinates = None


# ============================================================
# CASE LOOP
# ============================================================

for row_index, row in enumerate(
    design_rows
):

    case_id = row[
        "CaseID"
    ]


    odb_file = os.path.join(
        JOB_DIR,
        case_id
        +
        ".odb",
    )


    if not os.path.isfile(
        odb_file
    ):

        raise FileNotFoundError(
            odb_file
        )


    odb = openOdb(
        odb_file,
        readOnly=True,
    )


    composite = find_instance(
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
        region=composite
    )


    RF_field = frame.fieldOutputs[
        "RF"
    ].getSubset(
        region=composite
    )


    U_map = {

        value.nodeLabel:
            np.asarray(
                value.data,
                dtype=np.float64,
            )

        for value in U_field.values
    }


    RF_map = {

        value.nodeLabel:
            np.asarray(
                value.data,
                dtype=np.float64,
            )

        for value in RF_field.values
    }


    node_records = sorted(

        [
            (
                node.label,
                np.asarray(
                    node.coordinates,
                    dtype=np.float64,
                ),
            )

            for node
            in composite.nodes
        ],

        key=lambda item:
            item[
                0
            ],
    )


    node_labels = np.asarray(
        [
            item[
                0
            ]

            for item
            in node_records
        ],
        dtype=np.int64,
    )


    coordinates = np.asarray(
        [
            item[
                1
            ]

            for item
            in node_records
        ],
        dtype=np.float64,
    )


    U = np.asarray(
        [
            U_map[
                int(
                    label
                )
            ]

            for label
            in node_labels
        ],
        dtype=np.float64,
    )


    if reference_node_labels is None:

        reference_node_labels = (
            node_labels.copy()
        )


        reference_coordinates = (
            coordinates.copy()
        )


    else:

        if not np.array_equal(
            reference_node_labels,
            node_labels,
        ):

            odb.close()


            raise RuntimeError(
                "Node labels changed between snapshots."
            )


        if not np.allclose(
            reference_coordinates,
            coordinates,
            atol=1.0e-10,
        ):

            odb.close()


            raise RuntimeError(
                "Node coordinates changed."
            )


    # ========================================================
    # GENERALIZED INTERFACE FORCES
    # ========================================================

    g_parts = []


    for interface_name in PATCH_INTERFACES[
        args.patch
    ]:

        interface_metadata = patch_metadata[
            "interfaces"
        ][
            interface_name
        ]


        node_labels_interface = (
            interface_metadata[
                "node_labels"
            ]
        )


        reaction_matrix = np.asarray(
            [
                RF_map[
                    int(
                        node_label
                    )
                ]

                for node_label
                in node_labels_interface
            ],
            dtype=np.float64,
        )


        flattened = reaction_matrix.reshape(
            -1
        )


        basis = interface_bases[
            interface_name
        ]


        g = (
            basis.T
            @
            flattened
        )


        g_parts.append(
            g
        )


    g_all = np.concatenate(
        g_parts
    )


    # ========================================================
    # CENTER NOSE REACTION
    # ========================================================

    nose_reaction = 0.0


    if args.patch == "center":

        point = find_instance(
            odb,
            "Point-2",
        )


        point_RF = frame.fieldOutputs[
            "RF"
        ].getSubset(
            region=point
        )


        nose_reaction = sum(
            [
                float(
                    value.data[
                        1
                    ]
                )

                for value
                in point_RF.values
            ]
        )


    all_U.append(
        U
    )


    all_g.append(
        g_all
    )


    all_reaction.append(
        nose_reaction
    )


    successful_case_ids.append(
        case_id
    )


    odb.close()


    print(
        "[{}/{}] {}"
        .format(
            row_index + 1,
            len(
                design_rows
            ),
            case_id,
        )
    )


# ============================================================
# ARRAYS
# ============================================================

all_U = np.asarray(
    all_U,
    dtype=np.float32,
)


all_g = np.asarray(
    all_g,
    dtype=np.float32,
)


all_reaction = np.asarray(
    all_reaction,
    dtype=np.float32,
)


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "U_snapshots.npy",
    ),
    all_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "g_snapshots.npy",
    ),
    all_g,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "nose_rf2.npy",
    ),
    all_reaction,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "node_labels.npy",
    ),
    reference_node_labels,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "node_coordinates.npy",
    ),
    reference_coordinates.astype(
        np.float32
    ),
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "case_ids.csv",
    ),
    "w",
    newline=""
) as file_object:

    writer = csv.writer(
        file_object
    )


    writer.writerow(
        [
            "Index",
            "CaseID",
        ]
    )


    for index, case_id in enumerate(
        successful_case_ids
    ):

        writer.writerow(
            [
                index,
                case_id,
            ]
        )


summary = {

    "patch":
        args.patch,

    "cases":
        int(
            len(
                successful_case_ids
            )
        ),

    "nodes":
        int(
            all_U.shape[
                1
            ]
        ),

    "displacement_dofs":
        int(
            all_U.shape[
                1
            ]
            *
            3
        ),

    "generalized_force_dimension":
        int(
            all_g.shape[
                1
            ]
        ),
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "snapshot_summary.json",
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
    "=========================================="
)

print(
    "STEP 65D COMPLETE"
)

print(
    "=========================================="
)

print(
    json.dumps(
        summary,
        indent=4,
    )
)