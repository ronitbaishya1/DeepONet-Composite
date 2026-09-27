import os
import json
import argparse

import numpy as np
import pandas as pd

import torch

from src.separate_vector_deeponet import (
    SeparateVectorDeepONet
)

from src.mechanics_losses import (
    derived_mechanics_single,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
)


parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
)


parser.add_argument(
    "--tag",
    type=str,
    required=True,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"


OUTPUT_DIR = os.path.join(
    args.model_dir,
    "evaluation_" + args.tag,
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

parameters = np.load(
    os.path.join(
        DATA_DIR,
        "parameters.npy",
    )
).astype(
    np.float32
)


coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "coordinates.npy",
    )
).astype(
    np.float32
)


ip_coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "ip_coordinates.npy",
    )
).astype(
    np.float32
)


U1 = np.load(
    os.path.join(
        DATA_DIR,
        "U1.npy",
    )
).astype(
    np.float32
)


U2 = np.load(
    os.path.join(
        DATA_DIR,
        "U2.npy",
    )
).astype(
    np.float32
)


U3 = np.load(
    os.path.join(
        DATA_DIR,
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


LE_tensor = np.load(
    os.path.join(
        DATA_DIR,
        "LE_tensor.npy",
    )
).astype(
    np.float32
)


S_tensor = np.load(
    os.path.join(
        DATA_DIR,
        "S_tensor.npy",
    )
).astype(
    np.float32
)


split_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)


case_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "case_ids.csv",
    )
)


test_indices = split_table.loc[
    split_table[
        "Split"
    ] == "test",
    "Index",
].to_numpy(
    dtype=int
)


# ============================================================
# NORMALIZATION
# ============================================================

normalization_path = os.path.join(
    args.model_dir,
    "normalization.json",
)


if not os.path.exists(
    normalization_path
):

    normalization_path = os.path.join(
        "results",
        "vector_deeponet_separate",
        "normalization.json",
    )


with open(
    normalization_path,
    "r",
) as f:

    normalization = json.load(
        f
    )


parameter_mean = torch.tensor(
    normalization[
        "parameter_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


parameter_std = torch.tensor(
    normalization[
        "parameter_std"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


coordinate_min = torch.tensor(
    normalization[
        "coordinate_min"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


coordinate_max = torch.tensor(
    normalization[
        "coordinate_max"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


output_mean = torch.tensor(
    normalization[
        "output_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


output_std = torch.tensor(
    normalization[
        "output_std"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MODEL
# ============================================================

checkpoint_path = os.path.join(
    args.model_dir,
    args.checkpoint,
)


checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
)


model = SeparateVectorDeepONet(
    branch_dim=checkpoint.get(
        "branch_dim",
        3,
    ),
    trunk_dim=3,
    latent_dim=checkpoint.get(
        "latent_dim",
        128,
    ),
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
# NODAL COORDINATES
# ============================================================

coordinates_tensor = torch.tensor(
    coordinates,
    dtype=torch.float32,
    device=DEVICE,
)


coordinates_normalized = (
    2.0
    * (
        coordinates_tensor
        - coordinate_min
    )
    / (
        coordinate_max
        - coordinate_min
    )
    - 1.0
)


ip_coordinates_tensor = torch.tensor(
    ip_coordinates,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# METRICS
# ============================================================

def metrics(
    prediction,
    truth,
):

    error = (
        prediction
        - truth
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
            + 1.0e-14
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
            - truth.mean()
        )
        ** 2
    )


    if ss_tot > 1.0e-14:

        r2 = (
            1.0
            - ss_res
            / ss_tot
        )

    else:

        r2 = np.nan


    return (
        relative_l2,
        rmse,
        mae,
        r2,
    )


# ============================================================
# EVALUATE
# ============================================================

displacement_rows = []

mechanics_rows = []


component_names = [
    "11",
    "22",
    "33",
    "12",
    "13",
    "23",
]


for global_index in test_indices:

    case_id = case_table.loc[
        case_table[
            "Index"
        ] == global_index,
        "CaseID",
    ].iloc[0]


    parameter_physical = torch.tensor(
        parameters[
            global_index
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


    parameter_normalized = (
        parameter_physical
        - parameter_mean
    ) / parameter_std


    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    with torch.no_grad():

        prediction_normalized = model(
            parameter_normalized.unsqueeze(
                0
            ),
            coordinates_normalized,
        )[0]


        prediction_U = (
            prediction_normalized
            * output_std
            + output_mean
        ).cpu().numpy()


    truth_U = U[
        global_index
    ]


    for c, name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        (
            rel,
            rmse,
            mae,
            r2,
        ) = metrics(
            prediction_U[
                :,
                c
            ],
            truth_U[
                :,
                c
            ],
        )


        displacement_rows.append(
            {
                "CaseID":
                    case_id,

                "Component":
                    name,

                "Relative_L2":
                    rel,

                "RMSE_mm":
                    rmse,

                "MAE_mm":
                    mae,

                "R2":
                    r2,
            }
        )


    # --------------------------------------------------------
    # MECHANICS
    # --------------------------------------------------------

    with torch.enable_grad():

        (
            _,
            strain_prediction,
            stress_prediction,
        ) = derived_mechanics_single(
            model=model,
            parameter_physical=parameter_physical,
            coordinates_physical=ip_coordinates_tensor,
            normalization=normalization,
            create_graph=False,
        )


    strain_prediction = (
        strain_prediction
        .detach()
        .cpu()
        .numpy()
    )


    stress_prediction = (
        stress_prediction
        .detach()
        .cpu()
        .numpy()
    )


    for c, name in enumerate(
        component_names
    ):

        (
            rel,
            rmse,
            mae,
            r2,
        ) = metrics(
            strain_prediction[
                :,
                c
            ],
            LE_tensor[
                global_index,
                :,
                c
            ],
        )


        mechanics_rows.append(
            {
                "CaseID":
                    case_id,

                "Quantity":
                    "LE" + name,

                "Relative_L2":
                    rel,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "R2":
                    r2,
            }
        )


        (
            rel,
            rmse,
            mae,
            r2,
        ) = metrics(
            stress_prediction[
                :,
                c
            ],
            S_tensor[
                global_index,
                :,
                c
            ],
        )


        mechanics_rows.append(
            {
                "CaseID":
                    case_id,

                "Quantity":
                    "S" + name,

                "Relative_L2":
                    rel,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "R2":
                    r2,
            }
        )


    print(
        "Evaluated:",
        case_id
    )


# ============================================================
# SAVE
# ============================================================

displacement_df = pd.DataFrame(
    displacement_rows
)


mechanics_df = pd.DataFrame(
    mechanics_rows
)


displacement_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "displacement_metrics.csv",
    ),
    index=False,
)


mechanics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mechanics_metrics.csv",
    ),
    index=False,
)


displacement_summary = (
    displacement_df
    .groupby(
        "Component"
    )[
        [
            "Relative_L2",
            "RMSE_mm",
            "MAE_mm",
            "R2",
        ]
    ]
    .mean()
    .reset_index()
)


mechanics_summary = (
    mechanics_df
    .groupby(
        "Quantity"
    )[
        [
            "Relative_L2",
            "RMSE",
            "MAE",
            "R2",
        ]
    ]
    .mean()
    .reset_index()
)


displacement_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "displacement_summary.csv",
    ),
    index=False,
)


mechanics_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mechanics_summary.csv",
    ),
    index=False,
)


print("")
print(
    "=========================================="
)

print(
    "DISPLACEMENT SUMMARY"
)

print(
    "=========================================="
)

print(
    displacement_summary.to_string(
        index=False
    )
)


print("")
print(
    "=========================================="
)

print(
    "MECHANICS SUMMARY"
)

print(
    "=========================================="
)

print(
    mechanics_summary.to_string(
        index=False
    )
)