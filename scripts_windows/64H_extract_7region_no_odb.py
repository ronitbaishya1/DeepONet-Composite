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
# PROJECT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


ODB_ROOT = os.path.join(
    BASE_DIR,
    "candidate_inp_7region"
)


OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "candidate_extracted_7region"
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
    "--limit",
    type=int,
    default=0,
)


args = parser.parse_args()


ODB_DIR = os.path.join(
    ODB_ROOT,
    args.segment
)


OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    args.segment
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


if not os.path.isdir(
    ODB_DIR
):

    raise RuntimeError(
        "ODB directory missing: {}".format(
            ODB_DIR
        )
    )


# ============================================================
# ODB FILES
# ============================================================

odb_files = sorted(
    [
        name

        for name
        in os.listdir(
            ODB_DIR
        )

        if name.lower().endswith(
            ".odb"
        )
    ]
)


if args.limit > 0:

    odb_files = odb_files[
        :args.limit
    ]


if len(
    odb_files
) == 0:

    raise RuntimeError(
        "No ODB files found."
    )


# ============================================================
# HELPER: AVERAGE ELEMENT OUTPUTS
# ============================================================

def average_element_values(
    field_output
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


    for label, values in storage.items():

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
# PROCESS
# ============================================================

summary_rows = []


print("")
print(
    "=========================================="
)

print(
    "STEP 64H — EXTRACT 7-REGION ODB"
)

print(
    "=========================================="
)

print(
    "Segment:",
    args.segment
)

print(
    "ODB count:",
    len(
        odb_files
    )
)

print("")


for file_index, odb_name in enumerate(
    odb_files
):

    case_id = os.path.splitext(
        odb_name
    )[0]


    odb_path = os.path.join(
        ODB_DIR,
        odb_name
    )


    print(
        "[{}/{}] {}"
        .format(
            file_index + 1,
            len(
                odb_files
            ),
            case_id,
        )
    )


    odb = None


    try:

        odb = openOdb(
            odb_path,
            readOnly=True,
        )


        if "BLOCK-1" not in (
            odb.rootAssembly.instances
        ):

            raise RuntimeError(
                "BLOCK-1 not found."
            )


        instance = odb.rootAssembly.instances[
            "BLOCK-1"
        ]


        if "Loading" not in (
            odb.steps
        ):

            raise RuntimeError(
                "Loading step missing."
            )


        step = odb.steps[
            "Loading"
        ]


        if len(
            step.frames
        ) == 0:

            raise RuntimeError(
                "Loading has no frames."
            )


        frame = step.frames[
            -1
        ]


        for field_name in [
            "U",
            "RF",
            "S",
            "LE",
        ]:

            if field_name not in (
                frame.fieldOutputs
            ):

                raise RuntimeError(
                    "{} field missing."
                    .format(
                        field_name
                    )
                )


        # ====================================================
        # NODAL FIELDS
        # ====================================================

        U_field = frame.fieldOutputs[
            "U"
        ].getSubset(
            region=
                instance
        )


        RF_field = frame.fieldOutputs[
            "RF"
        ].getSubset(
            region=
                instance
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


        node_file = os.path.join(
            OUTPUT_DIR,
            case_id
            +
            "_nodes.csv"
        )


        x_values = []


        with open(
            node_file,
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

                label = node.label

                xyz = node.coordinates


                x_values.append(
                    xyz[
                        0
                    ]
                )


                displacement = U_map.get(
                    label,
                    (
                        0.0,
                        0.0,
                        0.0,
                    )
                )


                reaction = RF_map.get(
                    label,
                    (
                        0.0,
                        0.0,
                        0.0,
                    )
                )


                writer.writerow(
                    [
                        label,
                        xyz[
                            0
                        ],
                        xyz[
                            1
                        ],
                        xyz[
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


        # ====================================================
        # ELEMENT CENTROIDS
        # ====================================================

        node_coordinate_map = {}


        for node in instance.nodes:

            node_coordinate_map[
                node.label
            ] = np.asarray(
                node.coordinates,
                dtype=np.float64,
            )


        element_centroid_map = {}


        for element in instance.elements:

            element_coordinates = np.asarray(
                [
                    node_coordinate_map[
                        node_label
                    ]

                    for node_label
                    in element.connectivity
                ],
                dtype=np.float64,
            )


            element_centroid_map[
                element.label
            ] = np.mean(
                element_coordinates,
                axis=0,
            )


        # ====================================================
        # STRESS / STRAIN
        # ====================================================

        S_field = frame.fieldOutputs[
            "S"
        ].getSubset(
            region=
                instance,
            position=
                INTEGRATION_POINT,
        )


        LE_field = frame.fieldOutputs[
            "LE"
        ].getSubset(
            region=
                instance,
            position=
                INTEGRATION_POINT,
        )


        S_map = average_element_values(
            S_field
        )


        LE_map = average_element_values(
            LE_field
        )


        ip_file = os.path.join(
            OUTPUT_DIR,
            case_id
            +
            "_ip.csv"
        )


        with open(
            ip_file,
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


            for element in instance.elements:

                label = element.label


                if label not in S_map:

                    continue


                if label not in LE_map:

                    continue


                xyz = element_centroid_map[
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

                        xyz[
                            0
                        ],
                        xyz[
                            1
                        ],
                        xyz[
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


        # ====================================================
        # FACE REACTION SUMMARY
        # ====================================================

        x_min = float(
            min(
                x_values
            )
        )


        x_max = float(
            max(
                x_values
            )
        )


        left_resultant = np.zeros(
            3,
            dtype=np.float64,
        )


        right_resultant = np.zeros(
            3,
            dtype=np.float64,
        )


        left_max_rf = 0.0

        right_max_rf = 0.0


        for node in instance.nodes:

            x = float(
                node.coordinates[
                    0
                ]
            )


            reaction = np.asarray(
                RF_map.get(
                    node.label,
                    (
                        0.0,
                        0.0,
                        0.0,
                    )
                ),
                dtype=np.float64,
            )


            if np.isclose(
                x,
                x_min,
                atol=1.0e-7,
            ):

                left_resultant += (
                    reaction
                )


                left_max_rf = max(
                    left_max_rf,
                    float(
                        np.max(
                            np.abs(
                                reaction
                            )
                        )
                    ),
                )


            if np.isclose(
                x,
                x_max,
                atol=1.0e-7,
            ):

                right_resultant += (
                    reaction
                )


                right_max_rf = max(
                    right_max_rf,
                    float(
                        np.max(
                            np.abs(
                                reaction
                            )
                        )
                    ),
                )


        summary_rows.append(
            {
                "CaseID":
                    case_id,

                "Nodes":
                    len(
                        instance.nodes
                    ),

                "Elements":
                    len(
                        instance.elements
                    ),

                "XMin":
                    x_min,

                "XMax":
                    x_max,

                "LeftRF1":
                    left_resultant[
                        0
                    ],

                "LeftRF2":
                    left_resultant[
                        1
                    ],

                "LeftRF3":
                    left_resultant[
                        2
                    ],

                "RightRF1":
                    right_resultant[
                        0
                    ],

                "RightRF2":
                    right_resultant[
                        1
                    ],

                "RightRF3":
                    right_resultant[
                        2
                    ],

                "LeftMaxAbsRF":
                    left_max_rf,

                "RightMaxAbsRF":
                    right_max_rf,
            }
        )


        odb.close()

        odb = None


    except Exception as error:

        print(
            "FAILED:",
            case_id
        )

        print(
            "Reason:",
            str(
                error
            )
        )


        if odb is not None:

            try:

                odb.close()

            except Exception:

                pass


# ============================================================
# SUMMARY CSV
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "extraction_summary.csv"
)


if len(
    summary_rows
) > 0:

    with open(
        summary_file,
        "w",
        newline=""
    ) as file_object:

        writer = csv.DictWriter(
            file_object,
            fieldnames=list(
                summary_rows[
                    0
                ].keys()
            ),
        )


        writer.writeheader()


        for row in summary_rows:

            writer.writerow(
                row
            )


print("")
print(
    "=========================================="
)

print(
    "STEP 64H COMPLETE"
)

print(
    "=========================================="
)

print(
    "Extracted:",
    len(
        summary_rows
    )
)

print(
    "Folder:",
    OUTPUT_DIR
)

print("")