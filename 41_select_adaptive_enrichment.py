import os
import json
import argparse

import numpy as np
import pandas as pd

import torch

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
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
    required=True,
)


parser.add_argument(
    "--data_dir",
    required=True,
)


parser.add_argument(
    "--ensemble_root",
    required=True,
)


parser.add_argument(
    "--output_dir",
    default=
        "data/adaptive_enrichment",
)


parser.add_argument(
    "--pool_size",
    type=int,
    default=5000,
)


parser.add_argument(
    "--select",
    type=int,
    default=40,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

branch = np.load(
    os.path.join(
        args.data_dir,
        "branch_inputs.npy",
    )
).astype(
    np.float32
)


coordinates = np.load(
    os.path.join(
        args.data_dir,
        "coordinates.npy",
    )
).astype(
    np.float32
)


split_dataframe = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# FIND SPLIT COLUMN
# ============================================================

if "Split" in split_dataframe.columns:

    split_column = "Split"

elif "split" in split_dataframe.columns:

    split_column = "split"

else:

    raise RuntimeError(
        "Could not find Split column."
    )


split_values = (
    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()
)


def get_indices(
    names,
):

    mask = split_values.isin(
        names
    )


    if "Index" in split_dataframe.columns:

        return split_dataframe.loc[
            mask,
            "Index",
        ].to_numpy(
            dtype=int
        )


    return np.where(
        mask.to_numpy()
    )[0]


train_indices = get_indices(
    [
        "train",
    ]
)


validation_indices = get_indices(
    [
        "validation",
        "val",
    ]
)


training_branch = branch[
    train_indices
]


validation_branch = branch[
    validation_indices
]


print("")
print(
    "Train cases:",
    len(
        train_indices
    )
)


print(
    "Validation cases:",
    len(
        validation_indices
    )
)


# ============================================================
# LOAD ENSEMBLE
# ============================================================

member_directories = sorted(
    [
        name

        for name in os.listdir(
            args.ensemble_root
        )

        if name.startswith(
            "member_"
        )
    ]
)


if len(
    member_directories
) < 2:

    raise RuntimeError(
        "At least two ensemble members are required."
    )


models = []

checkpoints = []


for member_directory in member_directories:

    checkpoint_file = os.path.join(
        args.ensemble_root,
        member_directory,
        "best_v3_ensemble.pt",
    )


    checkpoint = torch.load(

        checkpoint_file,

        map_location=DEVICE,

        weights_only=False,
    )


    model = HybridBulkOperatorV3(

        branch_dim=
            checkpoint[
                "branch_dim"
            ],

        number_force_coefficients=
            checkpoint[
                "number_force_coefficients"
            ],

        hidden_dim=
            checkpoint[
                "hidden_dim"
            ],

        latent_dim=
            checkpoint[
                "latent_dim"
            ],

        depth=
            checkpoint[
                "depth"
            ],
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    models.append(
        model
    )


    checkpoints.append(
        checkpoint
    )


print(
    "Ensemble members:",
    len(
        models
    )
)


# ============================================================
# NORMALIZATION
# ============================================================

checkpoint = checkpoints[
    0
]


branch_mean = np.asarray(
    checkpoint[
        "branch_mean"
    ],
    dtype=np.float64,
)


branch_std = np.asarray(
    checkpoint[
        "branch_std"
    ],
    dtype=np.float64,
)


coordinate_min = np.asarray(
    checkpoint[
        "coordinate_min"
    ],
    dtype=np.float64,
)


coordinate_max = np.asarray(
    checkpoint[
        "coordinate_max"
    ],
    dtype=np.float64,
)


coordinate_range = np.maximum(

    coordinate_max
    -
    coordinate_min,

    1.0e-8,
)


coordinate_normalized = (

    2.0
    *
    (
        coordinates
        -
        coordinate_min
    )
    /
    coordinate_range

    -
    1.0
)


coordinate_tensor = torch.tensor(

    coordinate_normalized,

    dtype=torch.float32,

    device=DEVICE,
)


# ============================================================
# STANDARDIZED TRAINING SPACE
# ============================================================

z_train = (

    training_branch
    -
    branch_mean

) / branch_std


z_validation = (

    validation_branch
    -
    branch_mean

) / branch_std


number_dimensions = branch.shape[
    1
]


distance_scale = np.sqrt(
    number_dimensions
)


# ============================================================
# NEAREST-TRAINING-CASE DISTANCE
#
# This replaces the previous unstable covariance OOD score.
#
# Small number:
# candidate looks similar to existing training data.
#
# Large number:
# candidate is relatively unfamiliar.
# ============================================================

def nearest_training_distance(
    z_points,
):

    distances = []


    for point in z_points:

        difference = (

            z_train

            -

            point.reshape(
                1,
                -1
            )
        )


        point_distances = np.linalg.norm(

            difference,

            axis=1,
        ) / distance_scale


        distances.append(
            np.min(
                point_distances
            )
        )


    return np.asarray(
        distances,
        dtype=np.float64,
    )


# ============================================================
# ENSEMBLE DISAGREEMENT
# ============================================================

def ensemble_indicators(
    physical_branch,
    batch_size=64,
):

    all_u2_spread = []

    all_force_spread = []


    for start in range(
        0,
        len(
            physical_branch
        ),
        batch_size,
    ):

        batch = physical_branch[
            start:
            start
            +
            batch_size
        ]


        batch_normalized = (

            batch
            -
            branch_mean

        ) / branch_std


        batch_tensor = torch.tensor(

            batch_normalized,

            dtype=torch.float32,

            device=DEVICE,
        )


        member_u2 = []

        member_force_coefficients = []


        with torch.no_grad():

            for model in models:

                (
                    U_prediction,
                    force_prediction,
                ) = model(

                    batch_tensor,

                    coordinate_tensor,
                )


                member_u2.append(

                    U_prediction[
                        :,
                        :,
                        1
                    ]
                    .cpu()
                    .numpy()
                )


                member_force_coefficients.append(

                    force_prediction
                    .cpu()
                    .numpy()
                )


        member_u2 = np.asarray(
            member_u2
        )


        member_force_coefficients = np.asarray(
            member_force_coefficients
        )


        # ----------------------------------------------------
        # HOW MUCH DO THE 5 MODELS DISAGREE ABOUT U2?
        # ----------------------------------------------------

        u2_std = np.std(
            member_u2,
            axis=0,
        )


        u2_spread = np.sqrt(

            np.mean(
                u2_std
                ** 2,
                axis=1,
            )
        )


        # ----------------------------------------------------
        # HOW MUCH DO THE 5 MODELS DISAGREE ABOUT FORCE PCA?
        # ----------------------------------------------------

        force_std = np.std(
            member_force_coefficients,
            axis=0,
        )


        force_spread = np.sqrt(

            np.mean(
                force_std
                ** 2,
                axis=1,
            )
        )


        all_u2_spread.append(
            u2_spread
        )


        all_force_spread.append(
            force_spread
        )


    return (

        np.concatenate(
            all_u2_spread
        ),

        np.concatenate(
            all_force_spread
        ),
    )


# ============================================================
# CALIBRATE NORMAL VALUES USING VALIDATION DATA
# ============================================================

(
    validation_u2_spread,
    validation_force_spread,
) = ensemble_indicators(
    validation_branch
)


validation_distance = nearest_training_distance(
    z_validation
)


u2_threshold = max(

    float(
        np.percentile(
            validation_u2_spread,
            95,
        )
    ),

    1.0e-8,
)


force_threshold = max(

    float(
        np.percentile(
            validation_force_spread,
            95,
        )
    ),

    1.0e-8,
)


distance_threshold = max(

    float(
        np.percentile(
            validation_distance,
            95,
        )
    ),

    1.0e-8,
)


print("")
print(
    "Normal validation thresholds:"
)


print(
    "U2 disagreement:",
    u2_threshold
)


print(
    "Force disagreement:",
    force_threshold
)


print(
    "Nearest-training distance:",
    distance_threshold
)


# ============================================================
# TRAINING COVARIANCE
#
# We use this only to generate realistic correlated
# perturbations.
#
# We DO NOT invert it for an OOD score anymore.
# ============================================================

covariance = np.cov(
    z_train,
    rowvar=False,
)


covariance = (

    covariance
    +
    covariance.T

) / 2.0


eigenvalues, eigenvectors = np.linalg.eigh(
    covariance
)


eigenvalues = np.maximum(
    eigenvalues,
    1.0e-5,
)


square_root_covariance = (

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


# ============================================================
# ALLOWED INPUT RANGE
# ============================================================

coefficient_min = np.min(
    training_branch[
        :,
        3:
    ],
    axis=0,
)


coefficient_max = np.max(
    training_branch[
        :,
        3:
    ],
    axis=0,
)


coefficient_range = (

    coefficient_max
    -
    coefficient_min
)


# Allow only a modest 15% expansion beyond training.
coefficient_lower = (

    coefficient_min
    -
    0.15
    *
    coefficient_range
)


coefficient_upper = (

    coefficient_max
    +
    0.15
    *
    coefficient_range
)


# ============================================================
# GENERATE REALISTIC CANDIDATES
#
# We build each candidate from:
#
# 1. interpolation between two real training cases
# 2. a small correlated perturbation
#
# This keeps candidates much closer to physically observed
# combinations of interface coefficients.
# ============================================================

rng = np.random.default_rng(
    2026
)


candidate_batches = []


required_candidates = args.pool_size


attempt = 0


while sum(
    len(
        batch
    )
    for batch in candidate_batches
) < required_candidates:

    attempt += 1


    generation_size = max(
        3000,
        required_candidates,
    )


    first_indices = rng.integers(

        0,

        len(
            z_train
        ),

        size=
            generation_size,
    )


    second_indices = rng.integers(

        0,

        len(
            z_train
        ),

        size=
            generation_size,
    )


    alpha = rng.uniform(

        0.0,

        1.0,

        size=(
            generation_size,
            1,
        ),
    )


    base_z = (

        alpha
        *
        z_train[
            first_indices
        ]

        +

        (
            1.0
            -
            alpha
        )
        *
        z_train[
            second_indices
        ]
    )


    random_direction = rng.normal(

        size=(
            generation_size,
            number_dimensions,
        )
    )


    correlated_perturbation = (

        random_direction

        @

        square_root_covariance.T
    )


    # Moderate perturbation.
    candidate_z = (

        base_z

        +

        0.35
        *
        correlated_perturbation
    )


    candidate_physical = (

        branch_mean

        +

        candidate_z
        *
        branch_std
    )


    # ========================================================
    # REJECT, DO NOT CLIP
    #
    # This is important.
    #
    # The previous script clipped many candidates onto the
    # exact same PCA limits.
    #
    # Now unrealistic candidates are simply removed.
    # ========================================================

    valid = np.ones(
        generation_size,
        dtype=bool,
    )


    # Material ranges.
    valid &= (
        candidate_physical[
            :,
            0
        ]
        >=
        40500.0
    )


    valid &= (
        candidate_physical[
            :,
            0
        ]
        <=
        49500.0
    )


    valid &= (
        candidate_physical[
            :,
            1
        ]
        >=
        10800.0
    )


    valid &= (
        candidate_physical[
            :,
            1
        ]
        <=
        13200.0
    )


    valid &= (
        candidate_physical[
            :,
            2
        ]
        >=
        4050.0
    )


    valid &= (
        candidate_physical[
            :,
            2
        ]
        <=
        4950.0
    )


    # PCA coefficient ranges.
    for dimension in range(
        branch.shape[
            1
        ]
        -
        3
    ):

        physical_dimension = (
            dimension
            +
            3
        )


        valid &= (

            candidate_physical[
                :,
                physical_dimension
            ]

            >=

            coefficient_lower[
                dimension
            ]
        )


        valid &= (

            candidate_physical[
                :,
                physical_dimension
            ]

            <=

            coefficient_upper[
                dimension
            ]
        )


    accepted = candidate_physical[
        valid
    ]


    if len(
        accepted
    ) > 0:

        candidate_batches.append(
            accepted
        )


    if attempt > 20:

        raise RuntimeError(
            "Could not generate enough valid candidates."
        )


candidate_branch = np.concatenate(
    candidate_batches,
    axis=0,
)[
    :args.pool_size
]


candidate_z = (

    candidate_branch
    -
    branch_mean

) / branch_std


print("")
print(
    "Valid candidate pool:",
    len(
        candidate_branch
    )
)


# ============================================================
# DISTANCE FROM TRAINING DATA
# ============================================================

candidate_distance = nearest_training_distance(
    candidate_z
)


candidate_distance_ratio = (

    candidate_distance
    /
    distance_threshold
)


# ============================================================
# REMOVE:
#
# 1. cases almost identical to training samples
# 2. cases excessively far from training data
# ============================================================

keep = (

    (
        candidate_distance_ratio
        >
        0.20
    )

    &

    (
        candidate_distance_ratio
        <=
        2.50
    )
)


candidate_branch = candidate_branch[
    keep
]


candidate_z = candidate_z[
    keep
]


candidate_distance = candidate_distance[
    keep
]


candidate_distance_ratio = (
    candidate_distance_ratio[
        keep
    ]
)


print(
    "Candidate pool after distance filter:",
    len(
        candidate_branch
    )
)


if len(
    candidate_branch
) < args.select:

    raise RuntimeError(
        "Not enough candidates remain after filtering."
    )


# ============================================================
# MODEL UNCERTAINTY
# ============================================================

(
    candidate_u2_spread,
    candidate_force_spread,
) = ensemble_indicators(
    candidate_branch
)


candidate_u2_ratio = (

    candidate_u2_spread
    /
    u2_threshold
)


candidate_force_ratio = (

    candidate_force_spread
    /
    force_threshold
)


# ============================================================
# FINAL SCORE
#
# Main importance:
#
# 50% = U2 disagreement
# 30% = force disagreement
# 20% = novelty relative to training data
#
# We cap each ratio at 3 so no single metric can completely
# dominate selection.
# ============================================================

u2_score = np.clip(
    candidate_u2_ratio,
    0.0,
    3.0,
)


force_score = np.clip(
    candidate_force_ratio,
    0.0,
    3.0,
)


distance_score = np.clip(
    candidate_distance_ratio,
    0.0,
    3.0,
)


reliability_score = (

    0.50
    *
    u2_score

    +

    0.30
    *
    force_score

    +

    0.20
    *
    distance_score
)


# ============================================================
# KEEP TOP UNCERTAIN CANDIDATES
# ============================================================

top_count = min(
    800,
    len(
        candidate_branch
    ),
)


top_indices = np.argsort(
    reliability_score
)[
    -top_count:
]


top_branch = candidate_branch[
    top_indices
]


top_z = candidate_z[
    top_indices
]


top_reliability = reliability_score[
    top_indices
]


top_u2_spread = candidate_u2_spread[
    top_indices
]


top_force_spread = candidate_force_spread[
    top_indices
]


top_distance = candidate_distance[
    top_indices
]


top_u2_ratio = candidate_u2_ratio[
    top_indices
]


top_force_ratio = candidate_force_ratio[
    top_indices
]


top_distance_ratio = candidate_distance_ratio[
    top_indices
]


# ============================================================
# DIVERSITY SELECTION
#
# We want useful cases, but we also don't want all 40 cases
# to sit in exactly the same part of the input space.
# ============================================================

selected = []


first_index = int(
    np.argmax(
        top_reliability
    )
)


selected.append(
    first_index
)


while len(
    selected
) < args.select:

    best_index = None

    best_value = -np.inf


    score_min = np.min(
        top_reliability
    )


    score_max = np.max(
        top_reliability
    )


    for candidate_index in range(
        len(
            top_branch
        )
    ):

        if candidate_index in selected:

            continue


        selected_distances = []


        for selected_index in selected:

            distance = (

                np.linalg.norm(

                    top_z[
                        candidate_index
                    ]

                    -

                    top_z[
                        selected_index
                    ]
                )

                /

                distance_scale
            )


            selected_distances.append(
                distance
            )


        diversity = np.min(
            selected_distances
        )


        normalized_reliability = (

            top_reliability[
                candidate_index
            ]

            -
            score_min

        ) / (

            score_max
            -
            score_min
            +
            1.0e-12
        )


        normalized_diversity = np.clip(
            diversity,
            0.0,
            1.0,
        )


        combined_value = (

            0.80
            *
            normalized_reliability

            +

            0.20
            *
            normalized_diversity
        )


        if combined_value > best_value:

            best_value = combined_value

            best_index = candidate_index


    selected.append(
        best_index
    )


selected = np.asarray(
    selected,
    dtype=int,
)


# ============================================================
# SELECTED ARRAYS
# ============================================================

selected_branch = top_branch[
    selected
]


selected_reliability = top_reliability[
    selected
]


selected_u2_spread = top_u2_spread[
    selected
]


selected_force_spread = top_force_spread[
    selected
]


selected_distance = top_distance[
    selected
]


selected_u2_ratio = top_u2_ratio[
    selected
]


selected_force_ratio = top_force_ratio[
    selected
]


selected_distance_ratio = top_distance_ratio[
    selected
]


# ============================================================
# INPUT COLUMN NAMES
# ============================================================

if args.segment == "left":

    coefficient_names = (

        [
            "c_m6_{}".format(
                index
            )
            for index in range(
                4
            )
        ]

        +

        [
            "c_m2_{}".format(
                index
            )
            for index in range(
                5
            )
        ]
    )


    case_prefix = "AE_L"


else:

    coefficient_names = (

        [
            "c_p2_{}".format(
                index
            )
            for index in range(
                5
            )
        ]

        +

        [
            "c_p6_{}".format(
                index
            )
            for index in range(
                4
            )
        ]
    )


    case_prefix = "AE_R"


input_columns = [

    "E1",
    "E2",
    "G12",

] + coefficient_names


# ============================================================
# BUILD OUTPUT TABLE
# ============================================================

rows = []


for index in range(
    len(
        selected_branch
    )
):

    row = {

        "CaseID":
            "{}_{:03d}".format(
                case_prefix,
                index
                +
                1,
            )
    }


    for column_index, column_name in enumerate(
        input_columns
    ):

        row[
            column_name
        ] = float(

            selected_branch[
                index,
                column_index
            ]
        )


    indicator_values = {

        "U2":
            selected_u2_ratio[
                index
            ],

        "Force":
            selected_force_ratio[
                index
            ],

        "Distance":
            selected_distance_ratio[
                index
            ],
    }


    dominant_indicator = max(

        indicator_values,

        key=
            indicator_values.get,
    )


    row[
        "ReliabilityScore"
    ] = float(
        selected_reliability[
            index
        ]
    )


    row[
        "U2Spread"
    ] = float(
        selected_u2_spread[
            index
        ]
    )


    row[
        "ForceCoefficientSpread"
    ] = float(
        selected_force_spread[
            index
        ]
    )


    row[
        "MinDistanceToTraining"
    ] = float(
        selected_distance[
            index
        ]
    )


    row[
        "U2Ratio"
    ] = float(
        selected_u2_ratio[
            index
        ]
    )


    row[
        "ForceRatio"
    ] = float(
        selected_force_ratio[
            index
        ]
    )


    row[
        "DistanceRatio"
    ] = float(
        selected_distance_ratio[
            index
        ]
    )


    row[
        "DominantIndicator"
    ] = dominant_indicator


    rows.append(
        row
    )


design_dataframe = pd.DataFrame(
    rows
)


# ============================================================
# SAVE DESIGN
# ============================================================

design_file = os.path.join(

    args.output_dir,

    "{}_enrichment_design_40.csv".format(
        args.segment
    ),
)


design_dataframe.to_csv(
    design_file,
    index=False,
)


# ============================================================
# CALIBRATION FILE
# ============================================================

calibration = {

    "Segment":
        args.segment,

    "EnsembleMembers":
        len(
            models
        ),

    "CandidatePoolRequested":
        int(
            args.pool_size
        ),

    "CandidatePoolAfterFiltering":
        int(
            len(
                candidate_branch
            )
        ),

    "SelectedCases":
        int(
            args.select
        ),

    "U2Threshold":
        float(
            u2_threshold
        ),

    "ForceCoefficientThreshold":
        float(
            force_threshold
        ),

    "NearestTrainingDistanceThreshold":
        float(
            distance_threshold
        ),

    "Scoring":
        "0.50 U2 disagreement + 0.30 force disagreement + 0.20 nearest-training distance",

    "MaximumAllowedDistanceRatio":
        2.5,
}


calibration_file = os.path.join(

    args.output_dir,

    "{}_reliability_calibration.json".format(
        args.segment
    ),
)


with open(
    calibration_file,
    "w",
) as file_object:

    json.dump(
        calibration,
        file_object,
        indent=4,
    )


# ============================================================
# PRINT DIAGNOSTICS
# ============================================================

print("")
print(
    "================================================"
)

print(
    "CORRECTED ADAPTIVE SELECTION COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    "Selected cases:",
    len(
        design_dataframe
    )
)


print("")
print(
    "Reliability score:"
)

print(
    "  min  =",
    design_dataframe[
        "ReliabilityScore"
    ].min()
)

print(
    "  mean =",
    design_dataframe[
        "ReliabilityScore"
    ].mean()
)

print(
    "  max  =",
    design_dataframe[
        "ReliabilityScore"
    ].max()
)


print("")
print(
    "Distance ratio:"
)

print(
    "  min  =",
    design_dataframe[
        "DistanceRatio"
    ].min()
)

print(
    "  mean =",
    design_dataframe[
        "DistanceRatio"
    ].mean()
)

print(
    "  max  =",
    design_dataframe[
        "DistanceRatio"
    ].max()
)


print("")
print(
    "Dominant indicators:"
)

print(
    design_dataframe[
        "DominantIndicator"
    ].value_counts()
)


print("")
print(
    "Saved:"
)

print(
    design_file
)

print(
    calibration_file
)