from __future__ import print_function

import os
import csv
import argparse

from odbAccess import openOdb


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


args = parser.parse_args()


# ============================================================
# DIRECTORIES
# ============================================================

if args.segment == "left":

    ODB_DIR = os.path.join(
        BASE_DIR,
        "left_inp"
    )

    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        "left_extracted"
    )

else:

    ODB_DIR = os.path.join(
        BASE_DIR,
        "right_inp"
    )

    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        "right_extracted"
    )


if not os.path.isdir(
    ODB_DIR
):

    raise RuntimeError(
        "ODB directory does not exist: {}".format(
            ODB_DIR
        )
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND ODB FILES
# ============================================================

odb_files = sorted(
    [
        file_name

        for file_name in os.listdir(
            ODB_DIR
        )

        if file_name.lower().endswith(
            ".odb"
        )
    ]
)


if len(
    odb_files
) == 0:

    raise RuntimeError(
        "No ODB files found in {}".format(
            ODB_DIR
        )
    )


print("")
print("========================================")
print("STEP 19 - EXTRACT BULK ODB DATA")
print("========================================")
print("BASE_DIR   =", BASE_DIR)
print("Segment    =", args.segment)
print("ODB_DIR    =", ODB_DIR)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("ODB count  =", len(odb_files))
print("")


# ============================================================
# PROCESS EACH ODB
# ============================================================

successful_cases = []

failed_cases = []


for file_index, odb_name in enumerate(
    odb_files
):

    case_id = os.path.splitext(
        odb_name
    )[
        0
    ]


    odb_path = os.path.join(
        ODB_DIR,
        odb_name
    )


    print(
        "[{}/{}] Extracting {}".format(
            file_index + 1,
            len(
                odb_files
            ),
            case_id
        )
    )


    try:

        odb = openOdb(
            odb_path,
            readOnly=True
        )


        # ====================================================
        # INSTANCE
        # ====================================================

        if "BLOCK-1" not in odb.rootAssembly.instances:

            raise RuntimeError(
                "BLOCK-1 instance not found."
            )


        instance = odb.rootAssembly.instances[
            "BLOCK-1"
        ]


        # ====================================================
        # STEP
        # ====================================================

        if "Loading" not in odb.steps:

            raise RuntimeError(
                "Loading step not found."
            )


        step = odb.steps[
            "Loading"
        ]


        if len(
            step.frames
        ) == 0:

            raise RuntimeError(
                "No frames available in Loading step."
            )


        frame = step.frames[
            -1
        ]


        # ====================================================
        # REQUIRED OUTPUTS
        # ====================================================

        required_outputs = [
            "U",
            "RF",
            "S",
            "LE"
        ]


        for output_name in required_outputs:

            if output_name not in frame.fieldOutputs:

                raise RuntimeError(
                    "{} field output not found.".format(
                        output_name
                    )
                )


        # ====================================================
        # FIELD OUTPUTS
        # ====================================================

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


        S_field = frame.fieldOutputs[
            "S"
        ].getSubset(
            region=instance
        )


        LE_field = frame.fieldOutputs[
            "LE"
        ].getSubset(
            region=instance
        )


        # ====================================================
        # MAP NODAL U
        # ====================================================

        U_map = {}


        for value in U_field.values:

            U_map[
                value.nodeLabel
            ] = value.data


        # ====================================================
        # MAP NODAL RF
        # ====================================================

        RF_map = {}


        for value in RF_field.values:

            RF_map[
                value.nodeLabel
            ] = value.data


        # ====================================================
        # NODE OUTPUT CSV
        # ====================================================

        node_output_file = os.path.join(
            OUTPUT_DIR,
            case_id
            +
            "_nodes.csv"
        )


        with open(
            node_output_file,
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
                    "RF3"
                ]
            )


            for node in instance.nodes:

                label = node.label

                xyz = node.coordinates


                displacement = U_map.get(
                    label,
                    (
                        0.0,
                        0.0,
                        0.0
                    )
                )


                reaction = RF_map.get(
                    label,
                    (
                        0.0,
                        0.0,
                        0.0
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
                        ]
                    ]
                )


        # ====================================================
        # NODE COORDINATE MAP
        # ====================================================

        node_coordinate_map = {}


        for node in instance.nodes:

            node_coordinate_map[
                node.label
            ] = node.coordinates


        # ====================================================
        # ELEMENT CENTROIDS
        # ====================================================

        element_centroid_map = {}


        for element in instance.elements:

            x_sum = 0.0

            y_sum = 0.0

            z_sum = 0.0


            for node_label in element.connectivity:

                xyz = node_coordinate_map[
                    node_label
                ]


                x_sum += xyz[
                    0
                ]

                y_sum += xyz[
                    1
                ]

                z_sum += xyz[
                    2
                ]


            number_nodes = float(
                len(
                    element.connectivity
                )
            )


            element_centroid_map[
                element.label
            ] = (
                x_sum
                /
                number_nodes,

                y_sum
                /
                number_nodes,

                z_sum
                /
                number_nodes
            )


        # ====================================================
        # MAP STRESS
        # ====================================================

        S_map = {}


        for value in S_field.values:

            S_map[
                value.elementLabel
            ] = value.data


        # ====================================================
        # MAP STRAIN
        # ====================================================

        LE_map = {}


        for value in LE_field.values:

            LE_map[
                value.elementLabel
            ] = value.data


        # ====================================================
        # ELEMENT/IP OUTPUT CSV
        #
        # C3D8R has one reduced integration point per element.
        # Element centroid is therefore used as its location.
        # ====================================================

        ip_output_file = os.path.join(
            OUTPUT_DIR,
            case_id
            +
            "_ip.csv"
        )


        with open(
            ip_output_file,
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
                    "LE23"
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
                        ]
                    ]
                )


        # ====================================================
        # CLOSE ODB
        # ====================================================

        odb.close()


        successful_cases.append(
            case_id
        )


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


        failed_cases.append(
            case_id
        )


        try:

            odb.close()

        except Exception:

            pass


# ============================================================
# SUMMARY
# ============================================================

print("")
print("========================================")
print("STEP 19 COMPLETE")
print("========================================")
print(
    "Successful:",
    len(
        successful_cases
    )
)
print(
    "Failed:",
    len(
        failed_cases
    )
)
print(
    "Output directory:",
    OUTPUT_DIR
)


if len(
    failed_cases
) > 0:

    print("")
    print("Failed cases:")

    for case_id in failed_cases:

        print(
            "  ",
            case_id
        )


print("")