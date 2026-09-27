import os
import json
import argparse

import numpy as np
import pandas as pd


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

parser.add_argument(
    "--coordinate_tolerance",
    type=float,
    default=1.0e-4,
)

args = parser.parse_args()


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.getcwd()


# ============================================================
# SEGMENT SETTINGS
# ============================================================

if args.segment == "left":

    BASE_DATA_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "hybrid_bulk_left",
    )

    DESIGN_FILE = os.path.join(
        PROJECT_ROOT,
        "data",
        "adaptive_enrichment",
        "left_enrichment_design_40.csv",
    )

    EXTRACTED_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "adaptive_enrichment",
        "windows",
        "left_extracted",
    )

    OUTPUT_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "hybrid_bulk_left_adaptive120",
    )

    interface_names = [
        "m6",
        "m2",
    ]

    coefficient_columns = (

        [
            "c_m6_{}".format(i)
            for i in range(4)
        ]

        +

        [
            "c_m2_{}".format(i)
            for i in range(5)
        ]
    )


else:

    BASE_DATA_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "hybrid_bulk_right",
    )

    DESIGN_FILE = os.path.join(
        PROJECT_ROOT,
        "data",
        "adaptive_enrichment",
        "right_enrichment_design_40.csv",
    )

    EXTRACTED_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "adaptive_enrichment",
        "windows",
        "right_extracted",
    )

    OUTPUT_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "hybrid_bulk_right_adaptive120",
    )

    interface_names = [
        "p2",
        "p6",
    ]

    coefficient_columns = (

        [
            "c_p2_{}".format(i)
            for i in range(5)
        ]

        +

        [
            "c_p6_{}".format(i)
            for i in range(4)
        ]
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


INTERFACE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_interfaces_trainonly",
)


FORCE_PCA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_force_pca",
)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

branch_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "branch_inputs.npy",
    )
).astype(
    np.float32
)


coordinates = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "coordinates.npy",
    )
).astype(
    np.float64
)


U1_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "U1.npy",
    )
).astype(
    np.float32
)


U2_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "U2.npy",
    )
).astype(
    np.float32
)


U3_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "U3.npy",
    )
).astype(
    np.float32
)


g_left_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "g_left.npy",
    )
).astype(
    np.float32
)


g_right_original = np.load(
    os.path.join(
        BASE_DATA_DIR,
        "g_right.npy",
    )
).astype(
    np.float32
)


split_original = pd.read_csv(
    os.path.join(
        BASE_DATA_DIR,
        "split_assignment.csv",
    )
)


force_coeff_original = np.load(
    os.path.join(
        FORCE_PCA_DIR,
        "{}_force_coefficients.npy".format(
            args.segment
        ),
    )
).astype(
    np.float32
)


# ============================================================
# BASIC ORIGINAL DATA CHECKS
# ============================================================

number_original_cases = branch_original.shape[
    0
]


number_nodes = coordinates.shape[
    0
]


if U1_original.shape != (
    number_original_cases,
    number_nodes,
):

    raise RuntimeError(
        "Original U1 shape does not match branch/coordinate data."
    )


if U2_original.shape != U1_original.shape:

    raise RuntimeError(
        "Original U2 shape mismatch."
    )


if U3_original.shape != U1_original.shape:

    raise RuntimeError(
        "Original U3 shape mismatch."
    )


print("")
print(
    "================================================"
)

print(
    "ORIGINAL DATASET"
)

print(
    "================================================"
)


print(
    "Cases:",
    number_original_cases
)


print(
    "Nodes per case:",
    number_nodes
)


print(
    "Branch dimension:",
    branch_original.shape[
        1
    ]
)


# ============================================================
# LOAD INTERFACE INFORMATION
# ============================================================

interface_data = []


for interface_name in interface_names:

    displacement_file = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    force_file = os.path.join(
        FORCE_PCA_DIR,
        "force_pca_{}.npz".format(
            interface_name
        ),
    )


    displacement_interface = np.load(
        displacement_file
    )


    force_interface = np.load(
        force_file
    )


    interface_data.append(
        {
            "name":
                interface_name,

            "coordinates":
                displacement_interface[
                    "coordinates"
                ].astype(
                    np.float64
                ),

            "displacement_basis":
                displacement_interface[
                    "basis"
                ].astype(
                    np.float64
                ),

            "force_mean":
                force_interface[
                    "mean"
                ].astype(
                    np.float64
                ),

            "force_basis":
                force_interface[
                    "basis"
                ].astype(
                    np.float64
                ),

            "g_mean":
                force_interface[
                    "g_mean"
                ].astype(
                    np.float64
                ),

            "g_matrix":
                force_interface[
                    "g_matrix"
                ].astype(
                    np.float64
                ),
        }
    )


# ============================================================
# LOAD DESIGN
# ============================================================

design = pd.read_csv(
    DESIGN_FILE
)


if len(
    design
) != 40:

    print(
        "WARNING: expected 40 enrichment cases, found",
        len(
            design
        ),
    )


# ============================================================
# NEAREST-NODE MATCHING
#
# IMPORTANT:
#
# We are NOT interpolating.
#
# We find the closest actual Abaqus node and only accept it
# when the coordinate difference is extremely small.
# ============================================================

def nearest_node_mapping(
    reference_coordinates,
    source_coordinates,
    tolerance,
):

    reference_coordinates = np.asarray(
        reference_coordinates,
        dtype=np.float64,
    )


    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64,
    )


    try:

        from scipy.spatial import cKDTree


        tree = cKDTree(
            source_coordinates
        )


        distances, indices = tree.query(
            reference_coordinates,
            k=1,
        )


    except ImportError:

        print(
            "SciPy not available. "
            "Using slower NumPy coordinate matching."
        )


        distances = []

        indices = []


        for reference_coordinate in reference_coordinates:

            difference = (

                source_coordinates

                -

                reference_coordinate.reshape(
                    1,
                    3,
                )
            )


            distance = np.sqrt(
                np.sum(
                    difference
                    ** 2,
                    axis=1,
                )
            )


            nearest_index = int(
                np.argmin(
                    distance
                )
            )


            indices.append(
                nearest_index
            )


            distances.append(
                distance[
                    nearest_index
                ]
            )


        distances = np.asarray(
            distances
        )


        indices = np.asarray(
            indices,
            dtype=int,
        )


    maximum_distance = float(
        np.max(
            distances
        )
    )


    mean_distance = float(
        np.mean(
            distances
        )
    )


    if maximum_distance > tolerance:

        worst_index = int(
            np.argmax(
                distances
            )
        )


        raise RuntimeError(

            "\nCoordinate matching failed.\n"
            "Maximum nearest-node distance = {:.8e}\n"
            "Allowed tolerance             = {:.8e}\n"
            "Reference coordinate          = {}\n"
            "Nearest source coordinate     = {}\n"
            "\n"
            "This is larger than a floating-point "
            "rounding difference, so we should NOT "
            "silently merge these meshes."
            .format(

                maximum_distance,

                tolerance,

                reference_coordinates[
                    worst_index
                ],

                source_coordinates[
                    indices[
                        worst_index
                    ]
                ],
            )
        )


    # --------------------------------------------------------
    # Make sure two reference nodes did not map onto the same
    # source node.
    # --------------------------------------------------------

    if len(
        np.unique(
            indices
        )
    ) != len(
        indices
    ):

        raise RuntimeError(

            "Coordinate mapping is not one-to-one. "
            "At least two original nodes mapped to the "
            "same adaptive FEM node."
        )


    return (
        indices,
        distances,
        maximum_distance,
        mean_distance,
    )


# ============================================================
# NEW DATA CONTAINERS
# ============================================================

new_branch = []

new_U1 = []

new_U2 = []

new_U3 = []

new_g_left = []

new_g_right = []

new_force_coefficients = []


matching_audit_rows = []

force_audit_rows = []


# ============================================================
# PROCESS EACH NEW FEM CASE
# ============================================================

for case_counter, design_row in design.iterrows():

    case_id = str(
        design_row[
            "CaseID"
        ]
    )


    node_file = os.path.join(
        EXTRACTED_DIR,
        "{}_nodes.csv".format(
            case_id
        ),
    )


    if not os.path.isfile(
        node_file
    ):

        raise FileNotFoundError(
            node_file
        )


    nodes = pd.read_csv(
        node_file
    )


    required_columns = [

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


    for required_column in required_columns:

        if required_column not in nodes.columns:

            raise RuntimeError(
                "{} is missing column {}"
                .format(
                    node_file,
                    required_column,
                )
            )


    source_coordinates = nodes[
        [
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    source_U = nodes[
        [
            "U1",
            "U2",
            "U3",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    source_RF = nodes[
        [
            "RF1",
            "RF2",
            "RF3",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    # ========================================================
    # MATCH COMPLETE LOCAL FIELD
    # ========================================================

    (
        full_mapping,
        full_distances,
        full_max_distance,
        full_mean_distance,
    ) = nearest_node_mapping(

        coordinates,

        source_coordinates,

        args.coordinate_tolerance,
    )


    case_U = source_U[
        full_mapping
    ]


    # ========================================================
    # BRANCH INPUT
    # ========================================================

    branch_row = [

        float(
            design_row[
                "E1"
            ]
        ),

        float(
            design_row[
                "E2"
            ]
        ),

        float(
            design_row[
                "G12"
            ]
        ),
    ]


    for column in coefficient_columns:

        branch_row.append(
            float(
                design_row[
                    column
                ]
            )
        )


    branch_row = np.asarray(
        branch_row,
        dtype=np.float32,
    )


    if len(
        branch_row
    ) != branch_original.shape[
        1
    ]:

        raise RuntimeError(

            "{} branch dimension is {}, "
            "but original branch dimension is {}."
            .format(

                case_id,

                len(
                    branch_row
                ),

                branch_original.shape[
                    1
                ],
            )
        )


    new_branch.append(
        branch_row
    )


    new_U1.append(
        case_U[
            :,
            0
        ]
    )


    new_U2.append(
        case_U[
            :,
            1
        ]
    )


    new_U3.append(
        case_U[
            :,
            2
        ]
    )


    matching_audit_rows.append(
        {
            "CaseID":
                case_id,

            "NodeCountOriginal":
                int(
                    len(
                        coordinates
                    )
                ),

            "NodeCountAdaptiveCSV":
                int(
                    len(
                        source_coordinates
                    )
                ),

            "MeanCoordinateDistance":
                full_mean_distance,

            "MaxCoordinateDistance":
                full_max_distance,
        }
    )


    # ========================================================
    # INTERFACE FORCES
    # ========================================================

    generalized_forces_case = []

    force_coefficients_case = []


    for interface in interface_data:

        (
            interface_mapping,
            interface_distances,
            interface_max_distance,
            interface_mean_distance,
        ) = nearest_node_mapping(

            interface[
                "coordinates"
            ],

            source_coordinates,

            args.coordinate_tolerance,
        )


        interface_force_matrix = source_RF[
            interface_mapping
        ]


        interface_force_vector = (
            interface_force_matrix
            .reshape(
                -1
            )
        )


        # ----------------------------------------------------
        # Generalized force used by the original coupling.
        # ----------------------------------------------------

        g_direct = (

            interface[
                "displacement_basis"
            ].T

            @

            interface_force_vector
        )


        # ----------------------------------------------------
        # 8-mode force PCA coefficient.
        # ----------------------------------------------------

        force_coefficient = (

            interface[
                "force_basis"
            ].T

            @

            (
                interface_force_vector

                -

                interface[
                    "force_mean"
                ]
            )
        )


        # ----------------------------------------------------
        # Reconstruct nodal force from 8 force modes.
        # ----------------------------------------------------

        reconstructed_force = (

            interface[
                "force_mean"
            ]

            +

            interface[
                "force_basis"
            ]

            @

            force_coefficient
        )


        force_reconstruction_error = (

            np.linalg.norm(

                reconstructed_force

                -

                interface_force_vector
            )

            /

            (
                np.linalg.norm(
                    interface_force_vector
                )

                +

                1.0e-14
            )
        )


        # ----------------------------------------------------
        # Check that force PCA still reproduces the generalized
        # force for these NEW enrichment cases.
        # ----------------------------------------------------

        g_from_force_pca = (

            interface[
                "g_mean"
            ]

            +

            interface[
                "g_matrix"
            ]

            @

            force_coefficient
        )


        generalized_force_mapping_error = (

            np.linalg.norm(

                g_from_force_pca

                -

                g_direct
            )

            /

            (
                np.linalg.norm(
                    g_direct
                )

                +

                1.0e-14
            )
        )


        generalized_forces_case.append(
            g_direct
        )


        force_coefficients_case.append(
            force_coefficient
        )


        force_audit_rows.append(
            {
                "CaseID":
                    case_id,

                "Interface":
                    interface[
                        "name"
                    ],

                "InterfaceMeanCoordinateDistance":
                    interface_mean_distance,

                "InterfaceMaxCoordinateDistance":
                    interface_max_distance,

                "ForcePCAReconstructionError_percent":
                    100.0
                    *
                    force_reconstruction_error,

                "GeneralizedForceMappingError_percent":
                    100.0
                    *
                    generalized_force_mapping_error,
            }
        )


    new_g_left.append(
        generalized_forces_case[
            0
        ]
    )


    new_g_right.append(
        generalized_forces_case[
            1
        ]
    )


    new_force_coefficients.append(

        np.concatenate(
            force_coefficients_case
        )
    )


    print(
        "[{}/{}] {} | max node distance = {:.3e}"
        .format(

            case_counter + 1,

            len(
                design
            ),

            case_id,

            full_max_distance,
        )
    )


# ============================================================
# CONVERT NEW DATA TO ARRAYS
# ============================================================

new_branch = np.asarray(
    new_branch,
    dtype=np.float32,
)


new_U1 = np.asarray(
    new_U1,
    dtype=np.float32,
)


new_U2 = np.asarray(
    new_U2,
    dtype=np.float32,
)


new_U3 = np.asarray(
    new_U3,
    dtype=np.float32,
)


new_g_left = np.asarray(
    new_g_left,
    dtype=np.float32,
)


new_g_right = np.asarray(
    new_g_right,
    dtype=np.float32,
)


new_force_coefficients = np.asarray(
    new_force_coefficients,
    dtype=np.float32,
)


# ============================================================
# CHECK FOR NAN / INF
# ============================================================

arrays_to_check = {

    "new_branch":
        new_branch,

    "new_U1":
        new_U1,

    "new_U2":
        new_U2,

    "new_U3":
        new_U3,

    "new_g_left":
        new_g_left,

    "new_g_right":
        new_g_right,

    "new_force_coefficients":
        new_force_coefficients,
}


for array_name, array in arrays_to_check.items():

    if not np.all(
        np.isfinite(
            array
        )
    ):

        raise RuntimeError(
            "{} contains NaN or Inf."
            .format(
                array_name
            )
        )


# ============================================================
# MERGE ORIGINAL + 40 NEW CASES
# ============================================================

branch_combined = np.concatenate(
    [
        branch_original,
        new_branch,
    ],
    axis=0,
)


U1_combined = np.concatenate(
    [
        U1_original,
        new_U1,
    ],
    axis=0,
)


U2_combined = np.concatenate(
    [
        U2_original,
        new_U2,
    ],
    axis=0,
)


U3_combined = np.concatenate(
    [
        U3_original,
        new_U3,
    ],
    axis=0,
)


g_left_combined = np.concatenate(
    [
        g_left_original,
        new_g_left,
    ],
    axis=0,
)


g_right_combined = np.concatenate(
    [
        g_right_original,
        new_g_right,
    ],
    axis=0,
)


force_coeff_combined = np.concatenate(
    [
        force_coeff_original,
        new_force_coefficients,
    ],
    axis=0,
)


# ============================================================
# KEEP OLD VALIDATION AND TEST SETS UNCHANGED
#
# Original:
#
# train = 56
# val   = 12
# test  = 12
#
# Add 40 enrichment cases only to training:
#
# train = 96
# val   = 12
# test  = 12
#
# total = 120
# ============================================================

split_combined = split_original.copy()


new_split_rows = []


start_index = number_original_cases


for local_index, design_row in design.iterrows():

    new_row = {

        column:
            np.nan

        for column in split_combined.columns
    }


    if "Index" in new_row:

        new_row[
            "Index"
        ] = (

            start_index

            +

            local_index
        )


    if "CaseID" in new_row:

        new_row[
            "CaseID"
        ] = str(
            design_row[
                "CaseID"
            ]
        )


    if "Split" in new_row:

        new_row[
            "Split"
        ] = "train"


    if "split" in new_row:

        new_row[
            "split"
        ] = "train"


    new_split_rows.append(
        new_row
    )


split_combined = pd.concat(
    [
        split_combined,

        pd.DataFrame(
            new_split_rows
        ),
    ],
    ignore_index=True,
)


# ============================================================
# SAVE DATASET
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "branch_inputs.npy",
    ),
    branch_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "coordinates.npy",
    ),
    coordinates.astype(
        np.float32
    ),
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "U1.npy",
    ),
    U1_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "U2.npy",
    ),
    U2_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "U3.npy",
    ),
    U3_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "g_left.npy",
    ),
    g_left_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "g_right.npy",
    ),
    g_right_combined,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "force_coefficients.npy",
    ),
    force_coeff_combined,
)


split_combined.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "split_assignment.csv",
    ),
    index=False,
)


# ============================================================
# SAVE AUDITS
# ============================================================

matching_audit = pd.DataFrame(
    matching_audit_rows
)


force_audit = pd.DataFrame(
    force_audit_rows
)


matching_audit_file = os.path.join(
    OUTPUT_DIR,
    "node_matching_audit.csv",
)


force_audit_file = os.path.join(
    OUTPUT_DIR,
    "new_force_pca_audit.csv",
)


matching_audit.to_csv(
    matching_audit_file,
    index=False,
)


force_audit.to_csv(
    force_audit_file,
    index=False,
)


# ============================================================
# SPLIT COUNTS
# ============================================================

if "Split" in split_combined.columns:

    final_split_column = "Split"

else:

    final_split_column = "split"


split_counts = (

    split_combined[
        final_split_column
    ]
    .astype(str)
    .str.lower()
    .value_counts()
    .to_dict()
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "Segment":
        args.segment,

    "OriginalTotalCases":
        int(
            number_original_cases
        ),

    "NewEnrichmentCases":
        int(
            len(
                new_branch
            )
        ),

    "FinalTotalCases":
        int(
            len(
                branch_combined
            )
        ),

    "FinalTrainCases":
        int(
            split_counts.get(
                "train",
                0,
            )
        ),

    "FinalValidationCases":
        int(
            split_counts.get(
                "validation",
                split_counts.get(
                    "val",
                    0,
                ),
            )
        ),

    "FinalTestCases":
        int(
            split_counts.get(
                "test",
                0,
            )
        ),

    "CoordinateTolerance":
        float(
            args.coordinate_tolerance
        ),

    "MaximumObservedNodeMatchDistance":
        float(
            matching_audit[
                "MaxCoordinateDistance"
            ].max()
        ),

    "MeanNewForcePCAReconstructionError_percent":
        float(
            force_audit[
                "ForcePCAReconstructionError_percent"
            ].mean()
        ),

    "MaximumNewForcePCAReconstructionError_percent":
        float(
            force_audit[
                "ForcePCAReconstructionError_percent"
            ].max()
        ),

    "MeanNewGeneralizedForceMappingError_percent":
        float(
            force_audit[
                "GeneralizedForceMappingError_percent"
            ].mean()
        ),

    "MaximumNewGeneralizedForceMappingError_percent":
        float(
            force_audit[
                "GeneralizedForceMappingError_percent"
            ].max()
        ),
}


metadata_file = os.path.join(
    OUTPUT_DIR,
    "enrichment_metadata.json",
)


with open(
    metadata_file,
    "w",
) as file_object:

    json.dump(
        metadata,
        file_object,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("")
print(
    "================================================"
)

print(
    "ENRICHED DATASET COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    "Segment:",
    args.segment
)


print(
    "Original total cases:",
    number_original_cases
)


print(
    "New FEM cases:",
    len(
        new_branch
    )
)


print(
    "Final total cases:",
    len(
        branch_combined
    )
)


print("")
print(
    "Final split:"
)


print(
    "Train:",
    metadata[
        "FinalTrainCases"
    ]
)


print(
    "Validation:",
    metadata[
        "FinalValidationCases"
    ]
)


print(
    "Test:",
    metadata[
        "FinalTestCases"
    ]
)


print("")
print(
    "Maximum coordinate mismatch:"
)


print(
    "{:.8e}".format(
        metadata[
            "MaximumObservedNodeMatchDistance"
        ]
    )
)


print("")
print(
    "New force PCA reconstruction error:"
)


print(
    "Mean = {:.6f}%"
    .format(
        metadata[
            "MeanNewForcePCAReconstructionError_percent"
        ]
    )
)


print(
    "Max  = {:.6f}%"
    .format(
        metadata[
            "MaximumNewForcePCAReconstructionError_percent"
        ]
    )
)


print("")
print(
    "New generalized-force mapping error:"
)


print(
    "Mean = {:.6f}%"
    .format(
        metadata[
            "MeanNewGeneralizedForceMappingError_percent"
        ]
    )
)


print(
    "Max  = {:.6f}%"
    .format(
        metadata[
            "MaximumNewGeneralizedForceMappingError_percent"
        ]
    )
)


print("")
print(
    "Output directory:"
)


print(
    OUTPUT_DIR
)