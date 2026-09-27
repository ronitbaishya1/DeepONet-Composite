from __future__ import print_function

import os
import csv
import argparse

from odbAccess import openOdb


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


ODB_DIR = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
    "{}_inp".format(
        args.segment
    ),
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
    "{}_extracted".format(
        args.segment
    ),
)


STATUS_FILE = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
    "{}_extraction_status.csv".format(
        args.segment
    ),
)


if not os.path.isdir(
    OUTPUT_DIR
):

    os.makedirs(
        OUTPUT_DIR
    )


# ============================================================
# FIND ODB FILES
# ============================================================

odb_files = sorted(
    [
        filename

        for filename in os.listdir(
            ODB_DIR
        )

        if filename.lower().endswith(
            ".odb"
        )
    ]
)


print("")
print(
    "ODB files found:",
    len(
        odb_files
    )
)


status_rows = []


# ============================================================
# PROCESS EACH ODB
# ============================================================

for filename in odb_files:

    case_id = os.path.splitext(
        filename
    )[0]


    odb_file = os.path.join(
        ODB_DIR,
        filename,
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "{}_nodes.csv".format(
            case_id
        ),
    )


    print("")
    print(
        "Extracting:",
        case_id
    )


    odb = None


    try:

        odb = openOdb(
            odb_file,
            readOnly=True,
        )


        # ====================================================
        # CHECK THAT LOADING STEP EXISTS
        # ====================================================

        if "Loading" not in odb.steps:

            print(
                "  SKIPPED: Loading step not found."
            )


            status_rows.append(
                [
                    case_id,
                    "missing_loading_step",
                    0,
                ]
            )


            odb.close()

            continue


        step = odb.steps[
            "Loading"
        ]


        number_frames = len(
            step.frames
        )


        # ====================================================
        # CHECK FOR ACTUAL RESULT FRAME
        # ====================================================

        if number_frames == 0:

            print(
                "  SKIPPED: Loading step has zero frames."
            )


            status_rows.append(
                [
                    case_id,
                    "zero_frames",
                    0,
                ]
            )


            odb.close()

            continue


        frame = step.frames[
            -1
        ]


        # ====================================================
        # CHECK OUTPUT FIELDS
        # ====================================================

        if "U" not in frame.fieldOutputs:

            print(
                "  SKIPPED: U output missing."
            )


            status_rows.append(
                [
                    case_id,
                    "missing_U",
                    number_frames,
                ]
            )


            odb.close()

            continue


        if "RF" not in frame.fieldOutputs:

            print(
                "  SKIPPED: RF output missing."
            )


            status_rows.append(
                [
                    case_id,
                    "missing_RF",
                    number_frames,
                ]
            )


            odb.close()

            continue


        # ====================================================
        # FIND MAIN INSTANCE
        # ====================================================

        main_instance = None

        maximum_nodes = -1


        for instance in odb.rootAssembly.instances.values():

            number_nodes = len(
                instance.nodes
            )


            if number_nodes > maximum_nodes:

                maximum_nodes = number_nodes

                main_instance = instance


        if main_instance is None:

            print(
                "  SKIPPED: no instance found."
            )


            status_rows.append(
                [
                    case_id,
                    "missing_instance",
                    number_frames,
                ]
            )


            odb.close()

            continue


        # ====================================================
        # READ DISPLACEMENT AND REACTION FORCE
        # ====================================================

        U_field = frame.fieldOutputs[
            "U"
        ].getSubset(
            region=
                main_instance
        )


        RF_field = frame.fieldOutputs[
            "RF"
        ].getSubset(
            region=
                main_instance
        )


        U_map = {

            value.nodeLabel:
                value.data

            for value in U_field.values
        }


        RF_map = {

            value.nodeLabel:
                value.data

            for value in RF_field.values
        }


        # ====================================================
        # MAKE SURE RESULTS ARE NOT EMPTY
        # ====================================================

        if len(
            U_map
        ) == 0:

            print(
                "  SKIPPED: displacement field is empty."
            )


            status_rows.append(
                [
                    case_id,
                    "empty_U",
                    number_frames,
                ]
            )


            odb.close()

            continue


        # ====================================================
        # WRITE CSV
        # ====================================================

        with open(
            output_file,
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


            for node in main_instance.nodes:

                if node.label not in U_map:

                    continue


                displacement = U_map[
                    node.label
                ]


                reaction = RF_map.get(
                    node.label,
                    (
                        0.0,
                        0.0,
                        0.0,
                    )
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


        print(
            "  SUCCESS"
        )


        print(
            "  Frames:",
            number_frames
        )


        print(
            "  Nodes:",
            len(
                U_map
            )
        )


        status_rows.append(
            [
                case_id,
                "success",
                number_frames,
            ]
        )


        odb.close()


    except Exception as error:

        print(
            "  EXTRACTION ERROR:"
        )


        print(
            " ",
            str(
                error
            )
        )


        status_rows.append(
            [
                case_id,
                "exception",
                -1,
            ]
        )


        if odb is not None:

            try:

                odb.close()

            except Exception:

                pass


# ============================================================
# SAVE EXTRACTION STATUS
# ============================================================

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
            "NumberFrames",
        ]
    )


    writer.writerows(
        status_rows
    )


# ============================================================
# SUMMARY
# ============================================================

successful = [

    row

    for row in status_rows

    if row[
        1
    ]
    ==
    "success"
]


failed = [

    row

    for row in status_rows

    if row[
        1
    ]
    !=
    "success"
]


print("")
print(
    "================================================"
)

print(
    "EXTRACTION SUMMARY"
)

print(
    "================================================"
)


print(
    "Successful:",
    len(
        successful
    )
)


print(
    "Skipped/failed:",
    len(
        failed
    )
)


print("")
print(
    "Status file:"
)


print(
    STATUS_FILE
)