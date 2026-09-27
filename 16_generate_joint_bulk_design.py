import os
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
    "--n_total",
    type=int,
    default=80,
)


parser.add_argument(
    "--seed",
    type=int,
    default=42,
)


parser.add_argument(
    "--inflation",
    type=float,
    default=1.25,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"


INTERFACE_DIR = os.path.join(
    DATA_DIR,
    "hybrid_interfaces_trainonly",
)


OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "hybrid_bulk_designs",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# INTERFACE PAIR
# ============================================================

if args.segment == "left":

    LEFT_NAME = "m6"
    RIGHT_NAME = "m2"

else:

    LEFT_NAME = "p2"
    RIGHT_NAME = "p6"


# ============================================================
# LOAD
# ============================================================

left = np.load(
    os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            LEFT_NAME
        ),
    )
)


right = np.load(
    os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            RIGHT_NAME
        ),
    )
)


parameters = np.load(
    os.path.join(
        DATA_DIR,
        "parameters.npy",
    )
).astype(
    np.float64
)


train_indices = left[
    "train_case_indices"
].astype(
    int
)


if not np.array_equal(
    train_indices,
    right[
        "train_case_indices"
    ].astype(
        int
    ),
):

    raise RuntimeError(
        "Train case indices differ between interfaces."
    )


left_scores = left[
    "scores_train"
]


right_scores = right[
    "scores_train"
]


joint_scores = np.concatenate(
    [
        left_scores,
        right_scores,
    ],
    axis=1,
)


number_training_anchors = (
    joint_scores.shape[
        0
    ]
)


n_left_modes = (
    left_scores.shape[
        1
    ]
)


n_right_modes = (
    right_scores.shape[
        1
    ]
)


if args.n_total < number_training_anchors:

    raise ValueError(
        "n_total must be at least {} because all training "
        "anchors are retained.".format(
            number_training_anchors
        )
    )


n_augmented = (
    args.n_total
    - number_training_anchors
)


rng = np.random.default_rng(
    args.seed
)


# ============================================================
# MATERIAL LHS FOR AUGMENTED CASES
# ============================================================

def lhs(
    n,
    dimensions,
):

    if n == 0:

        return np.zeros(
            (
                0,
                dimensions,
            )
        )


    result = np.empty(
        (
            n,
            dimensions,
        ),
        dtype=np.float64,
    )


    for j in range(
        dimensions
    ):

        edges = np.linspace(
            0.0,
            1.0,
            n + 1,
        )


        samples = rng.uniform(
            edges[:-1],
            edges[1:],
        )


        rng.shuffle(
            samples
        )


        result[
            :,
            j
        ] = samples


    return result


lhs_values = lhs(
    n_augmented,
    3,
)


augmented_parameters = np.zeros(
    (
        n_augmented,
        3,
    ),
    dtype=np.float64,
)


if n_augmented > 0:

    augmented_parameters[
        :,
        0
    ] = (
        40500.0
        +
        lhs_values[
            :,
            0
        ]
        * (
            49500.0
            - 40500.0
        )
    )


    augmented_parameters[
        :,
        1
    ] = (
        10800.0
        +
        lhs_values[
            :,
            1
        ]
        * (
            13200.0
            - 10800.0
        )
    )


    augmented_parameters[
        :,
        2
    ] = (
        4050.0
        +
        lhs_values[
            :,
            2
        ]
        * (
            4950.0
            - 4050.0
        )
    )


# ============================================================
# COVARIANCE-AWARE JOINT INTERFACE SAMPLING
# ============================================================

joint_mean = joint_scores.mean(
    axis=0
)


joint_covariance = np.cov(
    joint_scores,
    rowvar=False,
)


# Small stabilization for numerical robustness.
jitter = (
    1.0e-10
    *
    max(
        float(
            np.trace(
                joint_covariance
            )
        ),
        1.0,
    )
)


joint_covariance = (
    joint_covariance
    +
    jitter
    * np.eye(
        joint_covariance.shape[
            0
        ]
    )
)


eigenvalues, eigenvectors = np.linalg.eigh(
    joint_covariance
)


eigenvalues = np.maximum(
    eigenvalues,
    1.0e-14,
)


sqrt_covariance = (
    eigenvectors
    @ np.diag(
        np.sqrt(
            eigenvalues
        )
    )
    @ eigenvectors.T
)


if n_augmented > 0:

    latent = rng.normal(
        size=(
            n_augmented,
            joint_scores.shape[
                1
            ],
        )
    )


    # Truncate extreme latent samples.
    latent = np.clip(
        latent,
        -2.5,
        2.5,
    )


    augmented_scores = (
        joint_mean.reshape(
            1,
            -1
        )
        +
        args.inflation
        * (
            latent
            @ sqrt_covariance.T
        )
    )

else:

    augmented_scores = np.zeros(
        (
            0,
            joint_scores.shape[
                1
            ],
        )
    )


# ============================================================
# COMBINE:
#
# 35 real full-beam training states
# +
# covariance-aware augmented states
# ============================================================

anchor_parameters = parameters[
    train_indices
]


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


sources = (
    ["anchor"]
    * number_training_anchors

    +
    ["augmented"]
    * n_augmented
)


# ============================================================
# TABLE
# ============================================================

rows = []


for i in range(
    args.n_total
):

    row = {

        "CaseID":
            "{}_bulk_{:04d}".format(
                args.segment,
                i + 1,
            ),

        "Source":
            sources[
                i
            ],

        "E1_MPa":
            all_parameters[
                i,
                0
            ],

        "E2_MPa":
            all_parameters[
                i,
                1
            ],

        "G12_MPa":
            all_parameters[
                i,
                2
            ],
    }


    left_values = all_scores[
        i,
        :n_left_modes
    ]


    right_values = all_scores[
        i,
        n_left_modes:
    ]


    for j in range(
        n_left_modes
    ):

        row[
            "cL_{:02d}".format(
                j + 1
            )
        ] = left_values[
            j
        ]


    for j in range(
        n_right_modes
    ):

        row[
            "cR_{:02d}".format(
                j + 1
            )
        ] = right_values[
            j
        ]


    rows.append(
        row
    )


table = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "{}_bulk_design_{}.csv".format(
        args.segment,
        args.n_total,
    ),
)


table.to_csv(
    output_file,
    index=False,
)


np.savez(
    os.path.join(
        OUTPUT_DIR,
        "{}_joint_sampling_info.npz".format(
            args.segment
        ),
    ),

    joint_mean=joint_mean,

    joint_covariance=joint_covariance,

    n_left_modes=np.array(
        [
            n_left_modes
        ]
    ),

    n_right_modes=np.array(
        [
            n_right_modes
        ]
    ),

    train_indices=train_indices,
)


print("")
print(
    "========================================"
)

print(
    "JOINT BULK DESIGN COMPLETE"
)

print(
    "========================================"
)

print(
    "Segment:",
    args.segment
)

print(
    "Anchor cases:",
    number_training_anchors
)

print(
    "Augmented cases:",
    n_augmented
)

print(
    "Left modes:",
    n_left_modes
)

print(
    "Right modes:",
    n_right_modes
)

print(
    "Total branch dimension:",
    3
    + n_left_modes
    + n_right_modes
)

print(
    "Saved:"
)

print(
    output_file
)