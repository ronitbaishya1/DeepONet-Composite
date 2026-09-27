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
    "--data_dir",
    required=True,
)


parser.add_argument(
    "--checkpoint",
    required=True,
)


parser.add_argument(
    "--output_dir",
    required=True,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD CHECKPOINT
#
# PyTorch 2.6+ defaults to weights_only=True.
#
# Our checkpoint contains NumPy normalization arrays in
# addition to model weights, so we explicitly use:
#
# weights_only=False
#
# This is safe here because this checkpoint was created by
# our own training script.
# ============================================================

checkpoint = torch.load(

    args.checkpoint,

    map_location=DEVICE,

    weights_only=False,
)


# ============================================================
# BUILD MODEL
# ============================================================

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


# ============================================================
# LOAD LOCAL DATA
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


U1 = np.load(
    os.path.join(
        args.data_dir,
        "U1.npy",
    )
).astype(
    np.float32
)


U2 = np.load(
    os.path.join(
        args.data_dir,
        "U2.npy",
    )
).astype(
    np.float32
)


U3 = np.load(
    os.path.join(
        args.data_dir,
        "U3.npy",
    )
).astype(
    np.float32
)


U = np.stack(
    [
        U1,
        U2,
        U3,
    ],
    axis=-1,
)


g_left = np.load(
    os.path.join(
        args.data_dir,
        "g_left.npy",
    )
).astype(
    np.float32
)


g_right = np.load(
    os.path.join(
        args.data_dir,
        "g_right.npy",
    )
).astype(
    np.float32
)


g = np.concatenate(
    [
        g_left,
        g_right,
    ],
    axis=1,
)


split_dataframe = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# SEGMENT
# ============================================================

segment = checkpoint[
    "segment"
]


print("")
print(
    "Segment:",
    segment
)


# ============================================================
# LOAD FORCE-PCA TARGETS
# ============================================================

force_pca_dir = os.path.join(
    "data",
    "hybrid_force_pca",
)


force_coefficients = np.load(
    os.path.join(
        force_pca_dir,
        "{}_force_coefficients.npy".format(
            segment
        ),
    )
).astype(
    np.float32
)


# ============================================================
# SELECT INTERFACES
# ============================================================

if segment == "left":

    interface_1 = "m6"

    interface_2 = "m2"

else:

    interface_1 = "p2"

    interface_2 = "p6"


pca_1 = np.load(
    os.path.join(
        force_pca_dir,
        "force_pca_{}.npz".format(
            interface_1
        ),
    )
)


pca_2 = np.load(
    os.path.join(
        force_pca_dir,
        "force_pca_{}.npz".format(
            interface_2
        ),
    )
)


# ============================================================
# TEST INDICES
# ============================================================

if "Split" in split_dataframe.columns:

    split_column = "Split"

elif "split" in split_dataframe.columns:

    split_column = "split"

else:

    raise RuntimeError(
        "Could not find Split column."
    )


test_mask = (

    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()

    ==
    "test"
)


if "Index" in split_dataframe.columns:

    test_indices = split_dataframe.loc[
        test_mask,
        "Index",
    ].to_numpy(
        dtype=int
    )

else:

    test_indices = np.where(
        test_mask.to_numpy()
    )[0]


print(
    "Number of test cases:",
    len(
        test_indices
    )
)


# ============================================================
# LOAD NORMALIZATION FROM CHECKPOINT
# ============================================================

branch_mean = np.asarray(
    checkpoint[
        "branch_mean"
    ],
    dtype=np.float32,
)


branch_std = np.asarray(
    checkpoint[
        "branch_std"
    ],
    dtype=np.float32,
)


coord_min = np.asarray(
    checkpoint[
        "coord_min"
    ],
    dtype=np.float32,
)


coord_max = np.asarray(
    checkpoint[
        "coord_max"
    ],
    dtype=np.float32,
)


U_mean = np.asarray(
    checkpoint[
        "U_mean"
    ],
    dtype=np.float32,
)


U_std = np.asarray(
    checkpoint[
        "U_std"
    ],
    dtype=np.float32,
)


force_coeff_mean = np.asarray(
    checkpoint[
        "force_coeff_mean"
    ],
    dtype=np.float32,
)


force_coeff_std = np.asarray(
    checkpoint[
        "force_coeff_std"
    ],
    dtype=np.float32,
)


# ============================================================
# NORMALIZE BRANCH INPUT
# ============================================================

branch_normalized = (

    branch
    -
    branch_mean

) / branch_std


# ============================================================
# NORMALIZE COORDINATES
# ============================================================

coordinate_range = np.maximum(

    coord_max
    -
    coord_min,

    1.0e-8,
)


coordinate_normalized = (

    2.0
    *
    (
        coordinates
        -
        coord_min
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
# FORCE PCA INFORMATION
# ============================================================

number_force_modes_1 = pca_1[
    "basis"
].shape[
    1
]


number_force_modes_2 = pca_2[
    "basis"
].shape[
    1
]


print(
    "Force modes at",
    interface_1,
    "=",
    number_force_modes_1,
)


print(
    "Force modes at",
    interface_2,
    "=",
    number_force_modes_2,
)


# ============================================================
# CHECK OUTPUT SIZE
# ============================================================

expected_force_coefficients = (

    number_force_modes_1
    +
    number_force_modes_2
)


if (
    force_coefficients.shape[
        1
    ]
    !=
    expected_force_coefficients
):

    raise RuntimeError(

        "Force coefficient dimension mismatch.\n"
        "Expected {}, found {}."
        .format(

            expected_force_coefficients,

            force_coefficients.shape[
                1
            ],
        )
    )


# ============================================================
# EVALUATION
# ============================================================

rows = []


for case_index in test_indices:

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    branch_tensor = torch.tensor(

        branch_normalized[
            case_index
        ],

        dtype=torch.float32,

        device=DEVICE,

    ).reshape(
        1,
        -1,
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    with torch.no_grad():

        (
            U_prediction_normalized,
            coefficient_prediction_normalized,
        ) = model(

            branch_tensor,

            coordinate_tensor,
        )


    # ========================================================
    # CONVERT DISPLACEMENT BACK TO PHYSICAL UNITS
    # ========================================================

    U_prediction = (

        U_prediction_normalized[
            0
        ]
        .cpu()
        .numpy()

        *
        U_std.reshape(
            1,
            3,
        )

        +

        U_mean.reshape(
            1,
            3,
        )
    )


    # ========================================================
    # CONVERT FORCE COEFFICIENTS BACK TO PHYSICAL VALUES
    # ========================================================

    coefficient_prediction = (

        coefficient_prediction_normalized[
            0
        ]
        .cpu()
        .numpy()

        *
        force_coeff_std

        +

        force_coeff_mean
    )


    # ========================================================
    # SPLIT FORCE COEFFICIENTS BY INTERFACE
    # ========================================================

    coefficients_1 = coefficient_prediction[
        :number_force_modes_1
    ]


    coefficients_2 = coefficient_prediction[

        number_force_modes_1:

        number_force_modes_1
        +
        number_force_modes_2
    ]


    # ========================================================
    # CONVERT FORCE PCA -> OLD GENERALIZED FORCE
    # ========================================================

    g1_prediction = (

        pca_1[
            "g_mean"
        ]

        +

        pca_1[
            "g_matrix"
        ]
        @
        coefficients_1
    )


    g2_prediction = (

        pca_2[
            "g_mean"
        ]

        +

        pca_2[
            "g_matrix"
        ]
        @
        coefficients_2
    )


    g_prediction = np.concatenate(
        [
            g1_prediction,
            g2_prediction,
        ]
    )


    # ========================================================
    # SAVE CASE METRICS
    # ========================================================

    row = {

        "CaseIndex":
            int(
                case_index
            ),
    }


    # ========================================================
    # DISPLACEMENT ERRORS
    # ========================================================

    for component_index, component_name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        truth = U[
            case_index,
            :,
            component_index
        ]


        prediction = U_prediction[
            :,
            component_index
        ]


        relative_l2 = (

            np.linalg.norm(
                prediction
                -
                truth
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
                (
                    prediction
                    -
                    truth
                )
                ** 2
            )
        )


        row[
            "{}_RelL2_percent".format(
                component_name
            )
        ] = (
            100.0
            *
            relative_l2
        )


        row[
            "{}_RMSE".format(
                component_name
            )
        ] = rmse


    # ========================================================
    # GENERALIZED FORCE ERROR
    # ========================================================

    generalized_force_error = (

        np.linalg.norm(

            g_prediction

            -

            g[
                case_index
            ]
        )

        /

        (
            np.linalg.norm(
                g[
                    case_index
                ]
            )
            +
            1.0e-14
        )
    )


    row[
        "GeneralizedForce_RelL2_percent"
    ] = (

        100.0
        *
        generalized_force_error
    )


    # ========================================================
    # FORCE PCA COEFFICIENT ERROR
    # ========================================================

    force_coefficient_error = (

        np.linalg.norm(

            coefficient_prediction

            -

            force_coefficients[
                case_index
            ]
        )

        /

        (
            np.linalg.norm(
                force_coefficients[
                    case_index
                ]
            )
            +
            1.0e-14
        )
    )


    row[
        "ForceCoefficient_RelL2_percent"
    ] = (

        100.0
        *
        force_coefficient_error
    )


    rows.append(
        row
    )


# ============================================================
# DATAFRAME
# ============================================================

case_dataframe = pd.DataFrame(
    rows
)


case_file = os.path.join(
    args.output_dir,
    "test_case_metrics.csv",
)


case_dataframe.to_csv(
    case_file,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "Segment":
        segment,

    "PhysicsWeight":
        float(
            checkpoint[
                "physics_weight"
            ]
        ),

    "BestEpoch":
        int(
            checkpoint[
                "best_epoch"
            ]
        ),

    "NumberTestCases":
        int(
            len(
                test_indices
            )
        ),

    "Mean_U1_percent":
        float(
            case_dataframe[
                "U1_RelL2_percent"
            ].mean()
        ),

    "Mean_U2_percent":
        float(
            case_dataframe[
                "U2_RelL2_percent"
            ].mean()
        ),

    "Mean_U3_percent":
        float(
            case_dataframe[
                "U3_RelL2_percent"
            ].mean()
        ),

    "Mean_GeneralizedForce_percent":
        float(
            case_dataframe[
                "GeneralizedForce_RelL2_percent"
            ].mean()
        ),

    "Mean_ForceCoefficient_percent":
        float(
            case_dataframe[
                "ForceCoefficient_RelL2_percent"
            ].mean()
        ),
}


summary_file = os.path.join(
    args.output_dir,
    "summary.json",
)


with open(
    summary_file,
    "w",
) as file_object:

    json.dump(
        summary,
        file_object,
        indent=4,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("")
print(
    "================================================"
)

print(
    "EVALUATION COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    "Physics weight:",
    summary[
        "PhysicsWeight"
    ]
)


print(
    "U1 mean test error: {:.4f}%".format(
        summary[
            "Mean_U1_percent"
        ]
    )
)


print(
    "U2 mean test error: {:.4f}%".format(
        summary[
            "Mean_U2_percent"
        ]
    )
)


print(
    "U3 mean test error: {:.4f}%".format(
        summary[
            "Mean_U3_percent"
        ]
    )
)


print(
    "Generalized force mean test error: {:.4f}%".format(
        summary[
            "Mean_GeneralizedForce_percent"
        ]
    )
)


print(
    "Force PCA coefficient mean test error: {:.4f}%".format(
        summary[
            "Mean_ForceCoefficient_percent"
        ]
    )
)


print("")
print(
    "Saved:"
)


print(
    case_file
)


print(
    summary_file
)