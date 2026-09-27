import os
import json

import numpy as np
import pandas as pd

import torch

from src.hybrid_bulk_operator import (
    HybridBulkOperator
)


# ============================================================
# PATHS
# ============================================================

ONLINE_DIR = os.path.join(
    "data",
    "hybrid_online_baseline",
)


VALIDATION_DIR = os.path.join(
    ONLINE_DIR,
    "validation_fields",
)


SOLUTION_FILE = os.path.join(
    ONLINE_DIR,
    "online_solution.npz",
)


OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_online_validation",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD SOLUTION
# ============================================================

solution = np.load(
    SOLUTION_FILE
)


E1 = float(
    solution[
        "E1"
    ][
        0
    ]
)


E2 = float(
    solution[
        "E2"
    ][
        0
    ]
)


G12 = float(
    solution[
        "G12"
    ][
        0
    ]
)


c_m6 = solution[
    "c_m6"
].astype(
    np.float32
)


c_m2 = solution[
    "c_m2"
].astype(
    np.float32
)


c_p2 = solution[
    "c_p2"
].astype(
    np.float32
)


c_p6 = solution[
    "c_p6"
].astype(
    np.float32
)


print(
    "Material:",
    E1,
    E2,
    G12,
)


# ============================================================
# LOAD ONE LOCAL OPERATOR
# ============================================================

def load_local_operator(
    data_dir,
    model_dir,
):

    with open(
        os.path.join(
            model_dir,
            "normalization.json",
        ),
        "r",
    ) as f:

        normalization = json.load(
            f
        )


    checkpoint = torch.load(
        os.path.join(
            model_dir,
            "best_hybrid_bulk_operator.pt",
        ),
        map_location=DEVICE,
    )


    model = HybridBulkOperator(
        branch_dim=checkpoint[
            "branch_dim"
        ],
        number_force_outputs=checkpoint[
            "number_force_outputs"
        ],
        latent_dim=checkpoint[
            "latent_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
    ).to(
        DEVICE
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    coordinates = np.load(
        os.path.join(
            data_dir,
            "coordinates.npy",
        )
    ).astype(
        np.float32
    )


    return (
        model,
        normalization,
        coordinates,
    )


# ============================================================
# PREDICT LOCAL DISPLACEMENT
# ============================================================

def predict_local_displacement(
    model,
    normalization,
    coordinates,
    branch_physical,
):

    branch_mean = np.asarray(
        normalization[
            "branch_mean"
        ],
        dtype=np.float32,
    )


    branch_std = np.asarray(
        normalization[
            "branch_std"
        ],
        dtype=np.float32,
    )


    coordinate_min = np.asarray(
        normalization[
            "coordinate_min"
        ],
        dtype=np.float32,
    )


    coordinate_max = np.asarray(
        normalization[
            "coordinate_max"
        ],
        dtype=np.float32,
    )


    output_mean = np.asarray(
        normalization[
            "output_mean"
        ],
        dtype=np.float32,
    )


    output_std = np.asarray(
        normalization[
            "output_std"
        ],
        dtype=np.float32,
    )


    branch_normalized = (
        branch_physical
        -
        branch_mean
    ) / branch_std


    coordinate_range = (
        coordinate_max
        -
        coordinate_min
    )


    coordinates_normalized = (
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


    with torch.no_grad():

        U_normalized, _ = model(
            torch.tensor(
                branch_normalized,
                dtype=torch.float32,
            ).unsqueeze(
                0
            ),

            torch.tensor(
                coordinates_normalized,
                dtype=torch.float32,
            ),
        )


    U = (
        U_normalized[
            0
        ]
        .cpu()
        .numpy()
        *
        output_std.reshape(
            1,
            3,
        )
        +
        output_mean.reshape(
            1,
            3,
        )
    )


    return U


# ============================================================
# LEFT NO
# ============================================================

(
    left_model,
    left_normalization,
    left_coordinates,
) = load_local_operator(

    "data/hybrid_bulk_left",

    "results/hybrid_bulk_left",
)


left_branch = np.concatenate(
    [

        np.array(
            [
                E1,
                E2,
                G12,
            ],
            dtype=np.float32,
        ),

        c_m6,

        c_m2,
    ]
)


left_U = predict_local_displacement(
    left_model,
    left_normalization,
    left_coordinates,
    left_branch,
)


# ============================================================
# RIGHT NO
# ============================================================

(
    right_model,
    right_normalization,
    right_coordinates,
) = load_local_operator(

    "data/hybrid_bulk_right",

    "results/hybrid_bulk_right",
)


right_branch = np.concatenate(
    [

        np.array(
            [
                E1,
                E2,
                G12,
            ],
            dtype=np.float32,
        ),

        c_p2,

        c_p6,
    ]
)


right_U = predict_local_displacement(
    right_model,
    right_normalization,
    right_coordinates,
    right_branch,
)


# ============================================================
# LOAD FE PATCH FIELDS
# ============================================================

left_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "left_nodes.csv",
    )
)


center_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "center_nodes.csv",
    )
)


right_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "right_nodes.csv",
    )
)


reference = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "full_reference_nodes.csv",
    )
)


# ============================================================
# COORDINATE KEY
#
# Mesh coordinates have tiny floating-point differences,
# therefore round before dictionary matching.
# ============================================================

def coordinate_key(
    x,
    y,
    z,
):

    return (
        round(
            float(
                x
            ),
            5,
        ),

        round(
            float(
                y
            ),
            5,
        ),

        round(
            float(
                z
            ),
            5,
        ),
    )


# ============================================================
# BUILD HYBRID FIELD DICTIONARY
#
# FE owns all interfaces.
# NO is used only in strict interiors:
#
# -6 < x < -1.8
#  1.8 < x < 6
# ============================================================

hybrid = {}


def add_dataframe(
    dataframe
):

    for _, row in dataframe.iterrows():

        key = coordinate_key(
            row[
                "X"
            ],
            row[
                "Y"
            ],
            row[
                "Z"
            ],
        )


        hybrid[
            key
        ] = np.array(
            [
                row[
                    "U1"
                ],
                row[
                    "U2"
                ],
                row[
                    "U3"
                ],
            ],
            dtype=np.float64,
        )


# Left FE
add_dataframe(
    left_fe
)


# Center FE
add_dataframe(
    center_fe
)


# Right FE
add_dataframe(
    right_fe
)


# ------------------------------------------------------------
# LEFT NO STRICT INTERIOR
# ------------------------------------------------------------

for index in range(
    len(
        left_coordinates
    )
):

    xyz = left_coordinates[
        index
    ]


    x = xyz[
        0
    ]


    if (
        x
        >
        -6.0
        +
        1.0e-5

        and

        x
        <
        -1.8
        -
        1.0e-5
    ):

        key = coordinate_key(
            xyz[
                0
            ],
            xyz[
                1
            ],
            xyz[
                2
            ],
        )


        hybrid[
            key
        ] = left_U[
            index
        ].astype(
            np.float64
        )


# ------------------------------------------------------------
# RIGHT NO STRICT INTERIOR
# ------------------------------------------------------------

for index in range(
    len(
        right_coordinates
    )
):

    xyz = right_coordinates[
        index
    ]


    x = xyz[
        0
    ]


    if (
        x
        >
        1.8
        +
        1.0e-5

        and

        x
        <
        6.0
        -
        1.0e-5
    ):

        key = coordinate_key(
            xyz[
                0
            ],
            xyz[
                1
            ],
            xyz[
                2
            ],
        )


        hybrid[
            key
        ] = right_U[
            index
        ].astype(
            np.float64
        )


# ============================================================
# MAP TO FULL FEM REFERENCE ORDER
# ============================================================

hybrid_U = []

reference_U = []

missing = []


for _, row in reference.iterrows():

    key = coordinate_key(
        row[
            "X"
        ],
        row[
            "Y"
        ],
        row[
            "Z"
        ],
    )


    if key not in hybrid:

        missing.append(
            key
        )

        continue


    hybrid_U.append(
        hybrid[
            key
        ]
    )


    reference_U.append(
        np.array(
            [
                row[
                    "U1"
                ],
                row[
                    "U2"
                ],
                row[
                    "U3"
                ],
            ],
            dtype=np.float64,
        )
    )


if len(
    missing
) > 0:

    print(
        "Missing nodes:",
        len(
            missing
        )
    )

    print(
        "First missing:",
        missing[
            :10
        ]
    )

    raise RuntimeError(
        "Hybrid field does not cover the complete reference mesh."
    )


hybrid_U = np.asarray(
    hybrid_U,
    dtype=np.float64,
)


reference_U = np.asarray(
    reference_U,
    dtype=np.float64,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
):

    error = (
        prediction
        -
        truth
    )


    relative_l2 = (
        np.linalg.norm(
            error
        )
        /
        (
            np.linalg.norm(
                truth
            )
            +
            1.0e-14
        )
    )


    rmse = np.sqrt(
        np.mean(
            error
            ** 2
        )
    )


    mae = np.mean(
        np.abs(
            error
        )
    )


    ss_res = np.sum(
        error
        ** 2
    )


    ss_tot = np.sum(
        (
            truth
            -
            truth.mean()
        )
        ** 2
    )


    if ss_tot > 1.0e-14:

        r2 = (
            1.0
            -
            ss_res
            /
            ss_tot
        )

    else:

        r2 = np.nan


    return (
        relative_l2,
        rmse,
        mae,
        r2,
    )


rows = []


for component, name in enumerate(
    [
        "U1",
        "U2",
        "U3",
    ]
):

    (
        relative_l2,
        rmse,
        mae,
        r2,
    ) = calculate_metrics(

        hybrid_U[
            :,
            component
        ],

        reference_U[
            :,
            component
        ],
    )


    rows.append(
        {

            "Component":
                name,

            "Relative_L2":
                relative_l2,

            "RMSE_mm":
                rmse,

            "MAE_mm":
                mae,

            "R2":
                r2,
        }
    )


metrics = pd.DataFrame(
    rows
)


# ============================================================
# REACTION COMPARISON
# ============================================================

with open(
    os.path.join(
        VALIDATION_DIR,
        "reaction_summary.json",
    ),
    "r",
) as f:

    reaction_summary = json.load(
        f
    )


# ============================================================
# SAVE
# ============================================================

metrics_file = os.path.join(
    OUTPUT_DIR,
    "hybrid_displacement_metrics.csv",
)


metrics.to_csv(
    metrics_file,
    index=False,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_U.npy",
    ),
    hybrid_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "reference_U.npy",
    ),
    reference_U,
)


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "============================================"
)

print(
    "HYBRID vs FULL FEM"
)

print(
    "============================================"
)

print(
    metrics.to_string(
        index=False
    )
)


print("")
print(
    "Full FEM nose RF2:",
    reaction_summary[
        "full_FEM_nose_RF2_N"
    ],
)

print(
    "Hybrid nose RF2:",
    reaction_summary[
        "hybrid_center_nose_RF2_N"
    ],
)

print(
    "Reaction relative error:",
    reaction_summary[
        "relative_reaction_error"
    ],
)


print("")
print(
    "Saved:"
)

print(
    metrics_file
)