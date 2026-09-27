import os
import json
import argparse

import numpy as np
import pandas as pd

import torch

from src.hybrid_bulk_operator import (
    HybridBulkOperator
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
    "--model_dir",
    required=True,
)


args = parser.parse_args()


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD DATA
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


U = np.stack(
    [
        np.load(
            os.path.join(
                args.data_dir,
                "U1.npy",
            )
        ),

        np.load(
            os.path.join(
                args.data_dir,
                "U2.npy",
            )
        ),

        np.load(
            os.path.join(
                args.data_dir,
                "U3.npy",
            )
        ),
    ],
    axis=-1,
).astype(
    np.float32
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


split = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


test_indices = split.loc[
    split[
        "Split"
    ] == "test",
    "Index",
].to_numpy(
    dtype=int
)


# ============================================================
# NORMALIZATION
# ============================================================

with open(
    os.path.join(
        args.model_dir,
        "normalization.json",
    ),
    "r",
) as f:

    normalization = json.load(
        f
    )


branch_mean = np.asarray(
    normalization[
        "branch_mean"
    ],
    dtype=np.float32
)


branch_std = np.asarray(
    normalization[
        "branch_std"
    ],
    dtype=np.float32
)


coordinate_min = np.asarray(
    normalization[
        "coordinate_min"
    ],
    dtype=np.float32
)


coordinate_max = np.asarray(
    normalization[
        "coordinate_max"
    ],
    dtype=np.float32
)


output_mean = np.asarray(
    normalization[
        "output_mean"
    ],
    dtype=np.float32
)


output_std = np.asarray(
    normalization[
        "output_std"
    ],
    dtype=np.float32
)


force_mean = np.asarray(
    normalization[
        "force_mean"
    ],
    dtype=np.float32
)


force_std = np.asarray(
    normalization[
        "force_std"
    ],
    dtype=np.float32
)


n_left_modes = int(
    normalization[
        "n_left_force_modes"
    ]
)


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    os.path.join(
        args.model_dir,
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


# ============================================================
# NORMALIZE COORDINATES
# ============================================================

coordinate_range = (
    coordinate_max
    -
    coordinate_min
)


coordinate_range = np.maximum(
    coordinate_range,
    1.0e-8,
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


coordinate_tensor = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
)


# ============================================================
# METRICS
# ============================================================

def relative_l2(
    prediction,
    truth,
):

    return (
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


def rmse(
    prediction,
    truth,
):

    return np.sqrt(
        np.mean(
            (
                prediction
                -
                truth
            )
            ** 2
        )
    )


def mae(
    prediction,
    truth,
):

    return np.mean(
        np.abs(
            prediction
            -
            truth
        )
    )


# ============================================================
# EVALUATE
# ============================================================

rows = []


for index in test_indices:

    branch_normalized = (
        branch[
            index
        ]
        -
        branch_mean
    ) / branch_std


    with torch.no_grad():

        (
            U_prediction_normalized,
            g_prediction_normalized,
        ) = model(
            torch.tensor(
                branch_normalized,
                dtype=torch.float32,
            ).unsqueeze(
                0
            ),
            coordinate_tensor,
        )


    U_prediction = (
        U_prediction_normalized[
            0
        ].cpu().numpy()
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


    g_prediction = (
        g_prediction_normalized[
            0
        ].cpu().numpy()
        *
        force_std
        +
        force_mean
    )


    truth_U = U[
        index
    ]


    truth_g = g[
        index
    ]


    row = {
        "Index":
            int(
                index
            ),
    }


    for component in range(
        3
    ):

        prediction_component = U_prediction[
            :,
            component
        ]


        truth_component = truth_U[
            :,
            component
        ]


        row[
            "U{}_RelativeL2".format(
                component + 1
            )
        ] = relative_l2(
            prediction_component,
            truth_component,
        )


        row[
            "U{}_RMSE_mm".format(
                component + 1
            )
        ] = rmse(
            prediction_component,
            truth_component,
        )


        row[
            "U{}_MAE_mm".format(
                component + 1
            )
        ] = mae(
            prediction_component,
            truth_component,
        )


    # --------------------------------------------------------
    # Forces
    # --------------------------------------------------------

    row[
        "ForceLeft_RelativeL2"
    ] = relative_l2(
        g_prediction[
            :n_left_modes
        ],
        truth_g[
            :n_left_modes
        ],
    )


    row[
        "ForceRight_RelativeL2"
    ] = relative_l2(
        g_prediction[
            n_left_modes:
        ],
        truth_g[
            n_left_modes:
        ],
    )


    row[
        "ForceAll_RelativeL2"
    ] = relative_l2(
        g_prediction,
        truth_g,
    )


    row[
        "ForceAll_RMSE"
    ] = rmse(
        g_prediction,
        truth_g,
    )


    row[
        "ForceAll_MAE"
    ] = mae(
        g_prediction,
        truth_g,
    )


    rows.append(
        row
    )


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(
    rows
)


metrics_file = os.path.join(
    args.model_dir,
    "test_metrics.csv",
)


results.to_csv(
    metrics_file,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = results.mean(
    numeric_only=True
)


summary_dataframe = pd.DataFrame(
    {
        "Metric":
            summary.index,

        "Mean":
            summary.values,
    }
)


summary_file = os.path.join(
    args.model_dir,
    "test_summary.csv",
)


summary_dataframe.to_csv(
    summary_file,
    index=False,
)


print("")
print(
    "=============================================="
)

print(
    "HYBRID BULK TEST SUMMARY"
)

print(
    "=============================================="
)


for metric, value in summary.items():

    if metric == "Index":

        continue


    print(
        "{:<30s} {:.8e}".format(
            metric,
            value,
        )
    )


print("")
print(
    "Saved:"
)

print(
    metrics_file
)

print(
    summary_file
)