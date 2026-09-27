from __future__ import print_function

import os
import csv
import json
import shutil

from odbAccess import openOdb


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


ARCHIVE_DIR = os.path.join(
    BASE_DIR,
    "online_results_archive",
)


OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "tolerance_comparison",
)


if not os.path.isdir(
    OUTPUT_ROOT
):

    os.makedirs(
        OUTPUT_ROOT
    )


# ============================================================
# TOLERANCE RUNS
# ============================================================

TOLERANCE_RUNS = [

    "tol_0050",

    "tol_0100",

    "tol_0200",
]


# ============================================================
# FIND ODB INSTANCE CASE-INSENSITIVELY
# ============================================================

def find_instance(
    odb,
    requested_name,
):

    requested_lower = requested_name.lower()


    for key, instance in odb.rootAssembly.instances.items():

        if key.lower() == requested_lower:

            return instance


    raise RuntimeError(
        "Instance not found: {}".format(
            requested_name
        )
    )


# ============================================================
# EXTRACT COMPOSITE NODAL FIELD
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


    step = odb.steps[
        "Loading"
    ]


    frame = step.frames[
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


    odb.close()


# ============================================================
# EXTRACT NOSE REACTION FROM CENTER PATCH
# ============================================================

def extract_nose_rf2(
    center_odb_file,
):

    odb = openOdb(
        center_odb_file,
        readOnly=True,
    )


    point_instance = find_instance(
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
        region=point_instance
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
            "No RF2 values found for Point-2."
        )


    # There is normally one rigid-body RP node.
    return sum(
        values
    )


# ============================================================
# FIND FULL FEM REFERENCE CSV
# ============================================================

reference_candidates = [

    os.path.join(
        BASE_DIR,
        "online_results",
        "validation_fields",
        "full_reference_nodes.csv",
    ),

    os.path.join(
        ARCHIVE_DIR,
        "baseline_original",
        "validation_fields",
        "full_reference_nodes.csv",
    ),

    os.path.join(
        ARCHIVE_DIR,
        "baseline_original",
        "full_reference_nodes.csv",
    ),
]


REFERENCE_FILE = None


for candidate in reference_candidates:

    if os.path.isfile(
        candidate
    ):

        REFERENCE_FILE = candidate

        break


if REFERENCE_FILE is None:

    print("")
    print(
        "WARNING:"
    )

    print(
        "full_reference_nodes.csv was not found automatically."
    )

    print(
        "The tolerance patch extraction will still run."
    )


# ============================================================
# FIND ORIGINAL FULL FEM REACTION
# ============================================================

reaction_candidates = [

    os.path.join(
        BASE_DIR,
        "online_results",
        "validation_fields",
        "reaction_summary.json",
    ),

    os.path.join(
        ARCHIVE_DIR,
        "baseline_original",
        "validation_fields",
        "reaction_summary.json",
    ),
]


FULL_FEM_RF2 = None


for candidate in reaction_candidates:

    if os.path.isfile(
        candidate
    ):

        with open(
            candidate,
            "r"
        ) as file_object:

            reaction_data = json.load(
                file_object
            )


        if "full_FEM_nose_RF2_N" in reaction_data:

            FULL_FEM_RF2 = float(
                reaction_data[
                    "full_FEM_nose_RF2_N"
                ]
            )

            break


# ============================================================
# COPY REFERENCE
# ============================================================

if REFERENCE_FILE is not None:

    shutil.copyfile(

        REFERENCE_FILE,

        os.path.join(
            OUTPUT_ROOT,
            "full_reference_nodes.csv",
        ),
    )


# ============================================================
# PROCESS EACH TOLERANCE
# ============================================================

for tolerance_name in TOLERANCE_RUNS:

    print("")
    print(
        "============================================"
    )

    print(
        "PROCESSING:",
        tolerance_name,
    )

    print(
        "============================================"
    )


    source_directory = os.path.join(
        ARCHIVE_DIR,
        tolerance_name,
    )


    final_directory = os.path.join(
        source_directory,
        "final",
    )


    output_directory = os.path.join(
        OUTPUT_ROOT,
        tolerance_name,
    )


    if not os.path.isdir(
        output_directory
    ):

        os.makedirs(
            output_directory
        )


    # --------------------------------------------------------
    # COPY METADATA FILES
    # --------------------------------------------------------

    metadata_files = [

        "online_solution.npz",

        "solution_manifest.json",

        "broyden_history.csv",
    ]


    for filename in metadata_files:

        source_file = os.path.join(
            source_directory,
            filename,
        )


        destination_file = os.path.join(
            output_directory,
            filename,
        )


        if not os.path.isfile(
            source_file
        ):

            raise FileNotFoundError(
                source_file
            )


        shutil.copyfile(
            source_file,
            destination_file,
        )


    # --------------------------------------------------------
    # PATCH ODBS
    # --------------------------------------------------------

    patch_files = {

        "left":
            os.path.join(
                final_directory,
                "left.odb",
            ),

        "center":
            os.path.join(
                final_directory,
                "center.odb",
            ),

        "right":
            os.path.join(
                final_directory,
                "right.odb",
            ),
    }


    for patch_name, odb_file in patch_files.items():

        if not os.path.isfile(
            odb_file
        ):

            raise FileNotFoundError(
                odb_file
            )


        output_csv = os.path.join(
            output_directory,
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
            output_csv,
        )


    # --------------------------------------------------------
    # CENTER NOSE REACTION
    # --------------------------------------------------------

    hybrid_rf2 = extract_nose_rf2(
        patch_files[
            "center"
        ]
    )


    reaction_summary = {

        "tolerance_run":
            tolerance_name,

        "hybrid_center_nose_RF2_N":
            hybrid_rf2,
    }


    if FULL_FEM_RF2 is not None:

        absolute_error = abs(
            hybrid_rf2
            -
            FULL_FEM_RF2
        )


        relative_error = (
            absolute_error
            /
            (
                abs(
                    FULL_FEM_RF2
                )
                +
                1.0e-14
            )
        )


        reaction_summary[
            "full_FEM_nose_RF2_N"
        ] = FULL_FEM_RF2


        reaction_summary[
            "absolute_reaction_error_N"
        ] = absolute_error


        reaction_summary[
            "relative_reaction_error"
        ] = relative_error


    reaction_file = os.path.join(
        output_directory,
        "reaction_summary.json",
    )


    with open(
        reaction_file,
        "w"
    ) as file_object:

        json.dump(
            reaction_summary,
            file_object,
            indent=4,
        )


    print(
        "Reaction summary:",
        reaction_file,
    )


print("")
print(
    "============================================"
)

print(
    "TOLERANCE FIELD EXTRACTION COMPLETE"
)

print(
    "============================================"
)

print(
    "Output directory:"
)

print(
    OUTPUT_ROOT
)