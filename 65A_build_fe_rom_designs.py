import os
import json

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INTERFACE_DIR = os.path.join(
    "data",
    "candidate_partition_interfaces_7region",
)


OUTPUT_DIR = os.path.join(
    "data",
    "fe_rom_designs_7region",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


NUMBER_AUGMENTED = 70

SEED = 6501

INFLATION = 1.10


MATERIAL_BOUNDS = {

    "E1":
        (
            40500.0,
            49500.0,
        ),

    "E2":
        (
            10800.0,
            13200.0,
        ),

    "G12":
        (
            4050.0,
            4950.0,
        ),
}


INTERFACE_ORDER = [

    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
]


PATCHES = {

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
# LOAD MATERIAL PARAMETERS
# ============================================================

parameters_npy = os.path.join(
    "data",
    "parameters.npy",
)


parameters_csv = os.path.join(
    "Data",
    "parameters.csv",
)


if os.path.isfile(
    parameters_npy
):

    parameters = np.load(
        parameters_npy
    ).astype(
        np.float64
    )


    case_ids = np.asarray(
        [
            "Case_{:04d}".format(
                index + 1
            )

            for index
            in range(
                parameters.shape[
                    0
                ]
            )
        ]
    )


elif os.path.isfile(
    parameters_csv
):

    parameter_table = pd.read_csv(
        parameters_csv
    )


    parameters = parameter_table[
        [
            "E1_MPa",
            "E2_MPa",
            "G12_MPa",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    case_ids = parameter_table[
        "CaseID"
    ].astype(
        str
    ).to_numpy()


else:

    raise FileNotFoundError(
        "Could not find parameters.npy or parameters.csv."
    )


if parameters.shape != (
    50,
    3,
):

    raise RuntimeError(
        "Expected 50 x 3 parameter table. Got {}."
        .format(
            parameters.shape
        )
    )


# ============================================================
# LOAD GLOBAL SPLIT
# ============================================================

split_candidates = [

    os.path.join(
        "data",
        "split_assignment.csv",
    ),

    os.path.join(
        "Data",
        "split_assignment.csv",
    ),
]


split_file = None


for candidate in split_candidates:

    if os.path.isfile(
        candidate
    ):

        split_file = candidate

        break


if split_file is None:

    raise FileNotFoundError(
        "Global split_assignment.csv not found."
    )


split_table = pd.read_csv(
    split_file
)


if len(
    split_table
) != 50:

    raise RuntimeError(
        "Expected 50 global split rows."
    )


split_labels = (
    split_table[
        "Split"
    ]
    .astype(
        str
    )
    .str.lower()
    .to_numpy()
)


train_indices = np.where(
    split_labels
    ==
    "train"
)[0]


validation_indices = np.where(
    split_labels
    ==
    "validation"
)[0]


test_indices = np.where(
    split_labels
    ==
    "test"
)[0]


print(
    "Global real states:"
)

print(
    "Train:",
    len(
        train_indices
    )
)

print(
    "Validation:",
    len(
        validation_indices
    )
)

print(
    "Test:",
    len(
        test_indices
    )
)


if len(
    train_indices
) != 35:

    raise RuntimeError(
        "Expected 35 global training cases."
    )


# ============================================================
# LOAD SIX INTERFACE SCORE MATRICES
# ============================================================

interface_data = {}

interface_scores_all = []

interface_mode_counts = {}


for interface_name in INTERFACE_ORDER:

    filename = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    data = np.load(
        filename
    )


    scores_all = data[
        "scores_all"
    ].astype(
        np.float64
    )


    if scores_all.shape[
        0
    ] != 50:

        raise RuntimeError(
            "{} does not have 50 scores."
            .format(
                interface_name
            )
        )


    interface_data[
        interface_name
    ] = data


    interface_scores_all.append(
        scores_all
    )


    interface_mode_counts[
        interface_name
    ] = scores_all.shape[
        1
    ]


# ============================================================
# COMPLETE 26-D INTERFACE STATE
# ============================================================

all_interface_scores = np.concatenate(
    interface_scores_all,
    axis=1,
)


if all_interface_scores.shape[
    1
] != 26:

    raise RuntimeError(
        "Expected 26 interface PCA scores."
    )


# ============================================================
# COMPLETE PHYSICAL STATE
#
# [E1,E2,G12, 26 interface coefficients]
# ============================================================

real_state = np.concatenate(
    [
        parameters,
        all_interface_scores,
    ],
    axis=1,
)


# ============================================================
# NORMALIZE USING TRAINING STATES ONLY
# ============================================================

train_state = real_state[
    train_indices
]


state_mean = train_state.mean(
    axis=0
)


state_std = train_state.std(
    axis=0,
    ddof=1,
)


state_std = np.maximum(
    state_std,
    1.0e-12,
)


train_standardized = (
    train_state
    -
    state_mean
) / state_std


# ============================================================
# JOINT COVARIANCE
#
# This preserves correlation between:
#
# material properties
# +
# all six interface displacement states.
# ============================================================

covariance = np.cov(
    train_standardized,
    rowvar=False,
)


jitter = (
    1.0e-10
    *
    max(
        float(
            np.trace(
                covariance
            )
        ),
        1.0,
    )
)


covariance = (

    covariance

    +

    jitter
    *
    np.eye(
        covariance.shape[
            0
        ]
    )
)


eigenvalues, eigenvectors = np.linalg.eigh(
    covariance
)


eigenvalues = np.maximum(
    eigenvalues,
    1.0e-12,
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


# ============================================================
# SAMPLE AUGMENTED TRAINING STATES
# ============================================================

rng = np.random.default_rng(
    SEED
)


latent = rng.normal(
    size=(
        NUMBER_AUGMENTED,
        real_state.shape[
            1
        ],
    )
)


latent = np.clip(
    latent,
    -2.5,
    2.5,
)


augmented_standardized = (

    INFLATION

    *

    (
        latent
        @
        sqrt_covariance.T
    )
)


# Keep interface-state normalized magnitudes moderate.
augmented_standardized[
    :,
    3:
] = np.clip(
    augmented_standardized[
        :,
        3:
    ],
    -2.5,
    2.5,
)


augmented_state = (

    state_mean.reshape(
        1,
        -1
    )

    +

    augmented_standardized

    *

    state_std.reshape(
        1,
        -1
    )
)


# ============================================================
# MATERIAL PHYSICAL BOUNDS
# ============================================================

augmented_state[
    :,
    0
] = np.clip(

    augmented_state[
        :,
        0
    ],

    MATERIAL_BOUNDS[
        "E1"
    ][0],

    MATERIAL_BOUNDS[
        "E1"
    ][1],
)


augmented_state[
    :,
    1
] = np.clip(

    augmented_state[
        :,
        1
    ],

    MATERIAL_BOUNDS[
        "E2"
    ][0],

    MATERIAL_BOUNDS[
        "E2"
    ][1],
)


augmented_state[
    :,
    2
] = np.clip(

    augmented_state[
        :,
        2
    ],

    MATERIAL_BOUNDS[
        "G12"
    ][0],

    MATERIAL_BOUNDS[
        "G12"
    ][1],
)


# ============================================================
# BUILD OFFSET TABLE FOR 26 COEFFICIENTS
# ============================================================

interface_slices = {}


cursor = 0


for interface_name in INTERFACE_ORDER:

    count = interface_mode_counts[
        interface_name
    ]


    interface_slices[
        interface_name
    ] = slice(
        cursor,
        cursor
        +
        count,
    )


    cursor += count


if cursor != 26:

    raise RuntimeError(
        "Interface mode total is not 26."
    )


# ============================================================
# BUILD PATCH DESIGN
# ============================================================

def build_patch_table(
    patch_name,
    patch_interfaces,
):

    rows = []


    # ========================================================
    # ALL 50 REAL FULL-BEAM STATES
    # ========================================================

    for global_index in range(
        50
    ):

        row = {

            "CaseID":
                "{}_real_{:04d}".format(
                    patch_name,
                    global_index + 1,
                ),

            "Source":
                "real",

            "GlobalIndex":
                global_index,

            "GlobalCaseID":
                str(
                    case_ids[
                        global_index
                    ]
                ),

            "Split":
                split_labels[
                    global_index
                ],

            "E1_MPa":
                parameters[
                    global_index,
                    0
                ],

            "E2_MPa":
                parameters[
                    global_index,
                    1
                ],

            "G12_MPa":
                parameters[
                    global_index,
                    2
                ],
        }


        for interface_name in patch_interfaces:

            values = all_interface_scores[
                global_index,
                interface_slices[
                    interface_name
                ],
            ]


            for mode_index, value in enumerate(
                values
            ):

                row[
                    "c_{}_{:02d}".format(
                        interface_name,
                        mode_index + 1,
                    )
                ] = value


        rows.append(
            row
        )


    # ========================================================
    # AUGMENTED TRAINING STATES
    # ========================================================

    for augmented_index in range(
        NUMBER_AUGMENTED
    ):

        state = augmented_state[
            augmented_index
        ]


        row = {

            "CaseID":
                "{}_aug_{:04d}".format(
                    patch_name,
                    augmented_index + 1,
                ),

            "Source":
                "augmented",

            "GlobalIndex":
                -1,

            "GlobalCaseID":
                "NONE",

            "Split":
                "train",

            "E1_MPa":
                state[
                    0
                ],

            "E2_MPa":
                state[
                    1
                ],

            "G12_MPa":
                state[
                    2
                ],
        }


        full_interface_state = state[
            3:
        ]


        for interface_name in patch_interfaces:

            values = full_interface_state[
                interface_slices[
                    interface_name
                ]
            ]


            for mode_index, value in enumerate(
                values
            ):

                row[
                    "c_{}_{:02d}".format(
                        interface_name,
                        mode_index + 1,
                    )
                ] = value


        rows.append(
            row
        )


    table = pd.DataFrame(
        rows
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "{}_fe_rom_design.csv".format(
            patch_name
        ),
    )


    table.to_csv(
        output_file,
        index=False,
    )


    print("")
    print(
        patch_name.upper()
    )

    print(
        "Total:",
        len(
            table
        )
    )

    print(
        table[
            "Split"
        ].value_counts()
    )

    print(
        "Saved:",
        output_file
    )


    return table


# ============================================================
# ALL THREE PATCHES
# ============================================================

for patch_name, patch_interfaces in (
    PATCHES.items()
):

    build_patch_table(
        patch_name,
        patch_interfaces,
    )


# ============================================================
# SAVE GLOBAL SAMPLING INFORMATION
# ============================================================

np.savez_compressed(

    os.path.join(
        OUTPUT_DIR,
        "rom_sampling_info.npz",
    ),

    state_mean=
        state_mean,

    state_std=
        state_std,

    covariance=
        covariance,

    augmented_state=
        augmented_state,

    train_indices=
        train_indices,

    validation_indices=
        validation_indices,

    test_indices=
        test_indices,
)


metadata = {

    "real_cases":
        50,

    "real_train":
        int(
            len(
                train_indices
            )
        ),

    "real_validation":
        int(
            len(
                validation_indices
            )
        ),

    "real_test":
        int(
            len(
                test_indices
            )
        ),

    "augmented_training_cases":
        NUMBER_AUGMENTED,

    "total_cases_per_patch":
        int(
            50
            +
            NUMBER_AUGMENTED
        ),

    "joint_state_dimension":
        int(
            real_state.shape[
                1
            ]
        ),

    "interface_dimension":
        26,

    "inflation":
        INFLATION,

    "seed":
        SEED,
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "rom_design_metadata.json",
    ),
    "w",
) as file_object:

    json.dump(
        metadata,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 65A COMPLETE"
)

print(
    "=========================================="
)

print(
    json.dumps(
        metadata,
        indent=4,
    )
)