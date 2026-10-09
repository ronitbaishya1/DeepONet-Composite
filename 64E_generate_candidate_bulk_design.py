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
        "all",
        "outer_left",
        "inner_left",
        "inner_right",
        "outer_right",
    ],
    default=
        "all",
)


parser.add_argument(
    "--interface_dir",
    default=
        "data/candidate_partition_interfaces_7region",
)


parser.add_argument(
    "--partition",
    default=
        "results/saint_venant_partition/"
        "selected_partition_7region.json",
)


parser.add_argument(
    "--parameters",
    default=
        "data/parameters.npy",
)


parser.add_argument(
    "--output_dir",
    default=
        "data/candidate_partition_designs_7region",
)


parser.add_argument(
    "--n_total",
    type=int,
    default=
        80,
)


parser.add_argument(
    "--seed",
    type=int,
    default=
        6401,
)


parser.add_argument(
    "--inflation",
    type=float,
    default=
        1.25,
)


# ============================================================
# MATERIAL RANGES
#
# Baseline +/- 10%
# ============================================================

parser.add_argument(
    "--E1_min",
    type=float,
    default=
        40500.0,
)


parser.add_argument(
    "--E1_max",
    type=float,
    default=
        49500.0,
)


parser.add_argument(
    "--E2_min",
    type=float,
    default=
        10800.0,
)


parser.add_argument(
    "--E2_max",
    type=float,
    default=
        13200.0,
)


parser.add_argument(
    "--G12_min",
    type=float,
    default=
        4050.0,
)


parser.add_argument(
    "--G12_max",
    type=float,
    default=
        4950.0,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# LOAD PARTITION
# ============================================================

with open(
    args.partition,
    "r",
) as file_object:

    partition = json.load(
        file_object
    )


interfaces_x = partition[
    "interfaces"
]


domain_x_min = float(
    partition[
        "DomainXMin"
    ]
)


domain_x_max = float(
    partition[
        "DomainXMax"
    ]
)


# ============================================================
# FOUR NO SEGMENTS
#
# None means physical exterior beam boundary.
#
# outer-left:
#
#   free surface | NO | left_outer interface
#
# outer-right:
#
#   right_outer interface | NO | free surface
# ============================================================

SEGMENTS = {

    "outer_left": {

        "left_interface":
            None,

        "right_interface":
            "left_outer",

        "x_min":
            domain_x_min,

        "x_max":
            float(
                interfaces_x[
                    "left_outer"
                ]
            ),

        "left_boundary_type":
            "free",

        "right_boundary_type":
            "interface",
    },

    "inner_left": {

        "left_interface":
            "left_inner",

        "right_interface":
            "center_left",

        "x_min":
            float(
                interfaces_x[
                    "left_inner"
                ]
            ),

        "x_max":
            float(
                interfaces_x[
                    "center_left"
                ]
            ),

        "left_boundary_type":
            "interface",

        "right_boundary_type":
            "interface",
    },

    "inner_right": {

        "left_interface":
            "center_right",

        "right_interface":
            "right_inner",

        "x_min":
            float(
                interfaces_x[
                    "center_right"
                ]
            ),

        "x_max":
            float(
                interfaces_x[
                    "right_inner"
                ]
            ),

        "left_boundary_type":
            "interface",

        "right_boundary_type":
            "interface",
    },

    "outer_right": {

        "left_interface":
            "right_outer",

        "right_interface":
            None,

        "x_min":
            float(
                interfaces_x[
                    "right_outer"
                ]
            ),

        "x_max":
            domain_x_max,

        "left_boundary_type":
            "interface",

        "right_boundary_type":
            "free",
    },
}


# ============================================================
# SELECT SEGMENTS
# ============================================================

if args.segment == "all":

    selected_segments = list(
        SEGMENTS.keys()
    )

else:

    selected_segments = [
        args.segment
    ]


# ============================================================
# MATERIAL PARAMETERS
# ============================================================

parameters = np.load(
    args.parameters
).astype(
    np.float64
)


if (
    parameters.ndim != 2
    or
    parameters.shape[1] < 3
):

    raise RuntimeError(
        "parameters.npy must have at least three columns."
    )


# ============================================================
# LHS FUNCTION
# ============================================================

def latin_hypercube(
    number_samples,
    dimensions,
    rng,
):

    if number_samples == 0:

        return np.zeros(
            (
                0,
                dimensions,
            ),
            dtype=np.float64,
        )


    result = np.zeros(
        (
            number_samples,
            dimensions,
        ),
        dtype=np.float64,
    )


    edges = np.linspace(
        0.0,
        1.0,
        number_samples + 1,
    )


    for dimension in range(
        dimensions
    ):

        values = rng.uniform(
            edges[
                :-1
            ],
            edges[
                1:
            ],
        )


        rng.shuffle(
            values
        )


        result[
            :,
            dimension
        ] = values


    return result


# ============================================================
# INTERFACE LOADER
# ============================================================

def load_interface(
    interface_name
):

    if interface_name is None:

        return None


    filename = os.path.join(
        args.interface_dir,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    if not os.path.isfile(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


    return np.load(
        filename
    )


# ============================================================
# BUILD ONE SEGMENT
# ============================================================

def build_segment_design(
    segment_name,
    config,
    seed_offset,
):

    print("")
    print(
        "=========================================="
    )

    print(
        "SEGMENT:",
        segment_name.upper()
    )

    print(
        "=========================================="
    )


    left_name = config[
        "left_interface"
    ]


    right_name = config[
        "right_interface"
    ]


    left_interface = load_interface(
        left_name
    )


    right_interface = load_interface(
        right_name
    )


    # ========================================================
    # DETERMINE TRAIN INDICES
    # ========================================================

    available_interfaces = [

        interface

        for interface in [
            left_interface,
            right_interface,
        ]

        if interface is not None
    ]


    if len(
        available_interfaces
    ) == 0:

        raise RuntimeError(
            "NO segment has no interface."
        )


    train_indices = (
        available_interfaces[
            0
        ][
            "train_case_indices"
        ]
        .astype(
            int
        )
    )


    for interface in available_interfaces[
        1:
    ]:

        candidate_indices = (
            interface[
                "train_case_indices"
            ]
            .astype(
                int
            )
        )


        if not np.array_equal(
            train_indices,
            candidate_indices,
        ):

            raise RuntimeError(
                (
                    "Train indices differ between "
                    "interfaces for {}."
                ).format(
                    segment_name
                )
            )


    number_anchor_cases = len(
        train_indices
    )


    number_augmented_cases = (
        args.n_total
        -
        number_anchor_cases
    )


    if number_augmented_cases < 0:

        raise RuntimeError(
            (
                "n_total={} is smaller than "
                "{} anchor cases."
            ).format(
                args.n_total,
                number_anchor_cases,
            )
        )


    # ========================================================
    # INTERFACE PCA SCORES
    # ========================================================

    score_parts = []


    left_number_modes = 0

    right_number_modes = 0


    if left_interface is not None:

        left_scores = left_interface[
            "scores_train"
        ].astype(
            np.float64
        )


        left_number_modes = (
            left_scores.shape[
                1
            ]
        )


        score_parts.append(
            left_scores
        )


    if right_interface is not None:

        right_scores = right_interface[
            "scores_train"
        ].astype(
            np.float64
        )


        right_number_modes = (
            right_scores.shape[
                1
            ]
        )


        score_parts.append(
            right_scores
        )


    joint_scores = np.concatenate(
        score_parts,
        axis=1,
    )


    # ========================================================
    # MATERIAL ANCHORS
    # ========================================================

    if np.max(
        train_indices
    ) >= parameters.shape[
        0
    ]:

        raise RuntimeError(
            "Train index exceeds parameters.npy."
        )


    anchor_parameters = parameters[
        train_indices,
        :3,
    ]


    # ========================================================
    # RANDOM GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        args.seed
        +
        seed_offset
    )


    # ========================================================
    # AUGMENTED MATERIALS
    # ========================================================

    lhs = latin_hypercube(
        number_augmented_cases,
        3,
        rng,
    )


    augmented_parameters = np.zeros(
        (
            number_augmented_cases,
            3,
        ),
        dtype=np.float64,
    )


    if number_augmented_cases > 0:

        augmented_parameters[
            :,
            0
        ] = (
            args.E1_min
            +
            lhs[
                :,
                0
            ]
            *
            (
                args.E1_max
                -
                args.E1_min
            )
        )


        augmented_parameters[
            :,
            1
        ] = (
            args.E2_min
            +
            lhs[
                :,
                1
            ]
            *
            (
                args.E2_max
                -
                args.E2_min
            )
        )


        augmented_parameters[
            :,
            2
        ] = (
            args.G12_min
            +
            lhs[
                :,
                2
            ]
            *
            (
                args.G12_max
                -
                args.G12_min
            )
        )


    # ========================================================
    # JOINT PCA SCORE DISTRIBUTION
    # ========================================================

    joint_mean = np.mean(
        joint_scores,
        axis=0,
    )


    joint_covariance = np.cov(
        joint_scores,
        rowvar=False,
    )


    # A one-dimensional covariance may return scalar.
    if np.ndim(
        joint_covariance
    ) == 0:

        joint_covariance = np.asarray(
            [
                [
                    float(
                        joint_covariance
                    )
                ]
            ],
            dtype=np.float64,
        )


    trace_value = float(
        np.trace(
            joint_covariance
        )
    )


    jitter = (
        1.0e-10
        *
        max(
            trace_value,
            1.0,
        )
    )


    joint_covariance = (
        joint_covariance
        +
        jitter
        *
        np.eye(
            joint_covariance.shape[
                0
            ]
        )
    )


    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            joint_covariance
        )
    )


    eigenvalues = np.maximum(
        eigenvalues,
        1.0e-14,
    )


    sqrt_covariance = (
        eigenvectors
        @
        np.diag(
            np.sqrt(
                eigenvalues
            )
        )
        @
        eigenvectors.T
    )


    if number_augmented_cases > 0:

        latent = rng.normal(
            size=(
                number_augmented_cases,
                joint_scores.shape[
                    1
                ],
            )
        )


        # Avoid extreme Gaussian samples.
        latent = np.clip(
            latent,
            -2.5,
            2.5,
        )


        augmented_scores = (
            joint_mean.reshape(
                1,
                -1,
            )
            +
            args.inflation
            *
            latent
            @
            sqrt_covariance.T
        )


        # -----------------------------------------------
        # Additional component-wise safety bounds.
        # -----------------------------------------------

        joint_std = np.std(
            joint_scores,
            axis=0,
            ddof=1,
        )


        lower = (
            joint_mean
            -
            2.75
            *
            joint_std
        )


        upper = (
            joint_mean
            +
            2.75
            *
            joint_std
        )


        augmented_scores = np.clip(
            augmented_scores,
            lower,
            upper,
        )


    else:

        augmented_scores = np.zeros(
            (
                0,
                joint_scores.shape[
                    1
                ],
            ),
            dtype=np.float64,
        )


    # ========================================================
    # COMBINE
    # ========================================================

    all_parameters = np.concatenate(
        [
            anchor_parameters,
            augmented_parameters,
        ],
        axis=0,
    )


    all_scores = np.concatenate(
        [
            joint_scores,
            augmented_scores,
        ],
        axis=0,
    )


    if all_parameters.shape[
        0
    ] != args.n_total:

        raise RuntimeError(
            "Material design count mismatch."
        )


    if all_scores.shape[
        0
    ] != args.n_total:

        raise RuntimeError(
            "Interface score count mismatch."
        )


    # ========================================================
    # SCORE SLICES
    # ========================================================

    cursor = 0


    left_slice = None

    right_slice = None


    if left_interface is not None:

        left_slice = slice(
            cursor,
            cursor
            +
            left_number_modes,
        )


        cursor += (
            left_number_modes
        )


    if right_interface is not None:

        right_slice = slice(
            cursor,
            cursor
            +
            right_number_modes,
        )


        cursor += (
            right_number_modes
        )


    # ========================================================
    # BUILD CSV
    #
    # We deliberately keep cL_XX / cR_XX naming.
    #
    # Outer-left has only cR.
    # Outer-right has only cL.
    # ========================================================

    rows = []


    for case_index in range(
        args.n_total
    ):

        is_anchor = (
            case_index
            <
            number_anchor_cases
        )


        if is_anchor:

            parent_index = int(
                train_indices[
                    case_index
                ]
            )

        else:

            parent_index = -1


        row = {

            "CaseID":
                "candidate_{}_{:04d}"
                .format(
                    segment_name,
                    case_index + 1,
                ),

            "Source":
                (
                    "anchor"
                    if is_anchor
                    else
                    "augmented"
                ),

            "ParentGlobalIndex":
                parent_index,

            "Segment":
                segment_name,

            "XMin":
                config[
                    "x_min"
                ],

            "XMax":
                config[
                    "x_max"
                ],

            "LeftBoundaryType":
                config[
                    "left_boundary_type"
                ],

            "RightBoundaryType":
                config[
                    "right_boundary_type"
                ],

            "LeftInterface":
                (
                    left_name
                    if left_name is not None
                    else
                    "NONE"
                ),

            "RightInterface":
                (
                    right_name
                    if right_name is not None
                    else
                    "NONE"
                ),

            "E1_MPa":
                float(
                    all_parameters[
                        case_index,
                        0
                    ]
                ),

            "E2_MPa":
                float(
                    all_parameters[
                        case_index,
                        1
                    ]
                ),

            "G12_MPa":
                float(
                    all_parameters[
                        case_index,
                        2
                    ]
                ),
        }


        # ====================================================
        # LEFT INTERFACE COEFFICIENTS
        # ====================================================

        if left_slice is not None:

            left_values = all_scores[
                case_index,
                left_slice
            ]


            for mode_index, value in enumerate(
                left_values
            ):

                row[
                    "cL_{:02d}".format(
                        mode_index + 1
                    )
                ] = float(
                    value
                )


        # ====================================================
        # RIGHT INTERFACE COEFFICIENTS
        # ====================================================

        if right_slice is not None:

            right_values = all_scores[
                case_index,
                right_slice
            ]


            for mode_index, value in enumerate(
                right_values
            ):

                row[
                    "cR_{:02d}".format(
                        mode_index + 1
                    )
                ] = float(
                    value
                )


        rows.append(
            row
        )


    design = pd.DataFrame(
        rows
    )


    output_file = os.path.join(
        args.output_dir,
        "{}_candidate_design_80.csv".format(
            segment_name
        ),
    )


    design.to_csv(
        output_file,
        index=False,
    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "segment":
            segment_name,

        "x_min":
            float(
                config[
                    "x_min"
                ]
            ),

        "x_max":
            float(
                config[
                    "x_max"
                ]
            ),

        "length":
            float(
                config[
                    "x_max"
                ]
                -
                config[
                    "x_min"
                ]
            ),

        "left_boundary_type":
            config[
                "left_boundary_type"
            ],

        "right_boundary_type":
            config[
                "right_boundary_type"
            ],

        "left_interface":
            left_name,

        "right_interface":
            right_name,

        "left_modes":
            int(
                left_number_modes
            ),

        "right_modes":
            int(
                right_number_modes
            ),

        "material_dimensions":
            3,

        "interface_dimensions":
            int(
                left_number_modes
                +
                right_number_modes
            ),

        "branch_dimension":
            int(
                3
                +
                left_number_modes
                +
                right_number_modes
            ),

        "anchor_cases":
            int(
                number_anchor_cases
            ),

        "augmented_cases":
            int(
                number_augmented_cases
            ),

        "total_cases":
            int(
                args.n_total
            ),

        "design_file":
            output_file,
    }


    metadata_file = os.path.join(
        args.output_dir,
        "{}_design_metadata.json".format(
            segment_name
        ),
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


    print("")
    print(
        "Saved:"
    )

    print(
        output_file
    )

    print(
        "Length:",
        metadata[
            "length"
        ],
        "mm",
    )

    print(
        "Left modes:",
        left_number_modes,
    )

    print(
        "Right modes:",
        right_number_modes,
    )

    print(
        "Branch dimension:",
        metadata[
            "branch_dimension"
        ],
    )


# ============================================================
# BUILD REQUESTED SEGMENTS
# ============================================================

seed_offsets = {

    "outer_left":
        0,

    "inner_left":
        1000,

    "inner_right":
        2000,

    "outer_right":
        3000,
}


for segment_name in selected_segments:

    build_segment_design(

        segment_name=
            segment_name,

        config=
            SEGMENTS[
                segment_name
            ],

        seed_offset=
            seed_offsets[
                segment_name
            ],
    )


# ============================================================
# SAVE OVERALL CONFIGURATION
# ============================================================

overall_file = os.path.join(
    args.output_dir,
    "seven_region_no_configuration.json",
)


with open(
    overall_file,
    "w",
) as file_object:

    json.dump(
        SEGMENTS,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64E COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    "Output directory:"
)

print(
    args.output_dir
)