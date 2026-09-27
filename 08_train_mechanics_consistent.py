import os
import json
import random
import argparse
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from src.separate_vector_deeponet import (
    SeparateVectorDeepONet
)

from src.mechanics_losses import (
    calculate_mechanics_scales,
    mechanics_supervision_loss,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--lambda_strain",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--lambda_stress",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=2000,
)


parser.add_argument(
    "--mechanics_points",
    type=int,
    default=256,
)


parser.add_argument(
    "--components",
    choices=[
        "key",
        "all",
    ],
    default="key",
)


parser.add_argument(
    "--train_fraction",
    type=float,
    default=1.0,
)


parser.add_argument(
    "--device",
    choices=[
        "auto",
        "cpu",
        "mps",
    ],
    default="auto",
)


args = parser.parse_args()


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


rng = np.random.default_rng(
    SEED
)


DATA_DIR = "data"


BASE_DIR = os.path.join(
    "results",
    "vector_deeponet_separate",
)


RESULTS_DIR = os.path.join(
    "results",
    "mechanics_consistent",
    (
        "eps_{:g}_sig_{:g}_frac_{:g}"
        .format(
            args.lambda_strain,
            args.lambda_stress,
            args.train_fraction,
        )
    ),
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

if args.device == "cpu":

    DEVICE = torch.device(
        "cpu"
    )

elif args.device == "mps":

    DEVICE = torch.device(
        "mps"
    )

else:

    if torch.backends.mps.is_available():

        DEVICE = torch.device(
            "mps"
        )

    else:

        DEVICE = torch.device(
            "cpu"
        )


print(
    "Device:",
    DEVICE,
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


full_train_indices = split_table.loc[
    split_table[
        "Split"
    ] == "train",
    "Index",
].to_numpy(
    dtype=int
)


validation_indices = split_table.loc[
    split_table[
        "Split"
    ] == "validation",
    "Index",
].to_numpy(
    dtype=int
)


# ============================================================
# OPTIONAL REDUCED TRAINING SET
# ============================================================

shuffled_train = (
    full_train_indices.copy()
)

rng.shuffle(
    shuffled_train
)


number_train = max(
    1,
    int(
        round(
            args.train_fraction
            * len(
                shuffled_train
            )
        )
    ),
)


train_indices = np.sort(
    shuffled_train[
        :number_train
    ]
)


print("")
print(
    "Training cases:",
    len(
        train_indices
    ),
    "/",
    len(
        full_train_indices
    ),
)


# ============================================================
# LOAD ORIGINAL NORMALIZATION
# ============================================================

with open(
    os.path.join(
        BASE_DIR,
        "normalization.json",
    ),
    "r",
) as f:

    normalization = json.load(
        f
    )


parameter_mean = np.asarray(
    normalization[
        "parameter_mean"
    ],
    dtype=np.float32,
)


parameter_std = np.asarray(
    normalization[
        "parameter_std"
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


parameters_normalized = (
    parameters
    - parameter_mean
) / parameter_std


coordinates_normalized = (
    2.0
    * (
        coordinates
        - coordinate_min
    )
    / (
        coordinate_max
        - coordinate_min
    )
    - 1.0
)


U_normalized = (
    U
    - output_mean.reshape(
        1,
        1,
        3,
    )
) / output_std.reshape(
    1,
    1,
    3,
)


# ============================================================
# MECHANICS COMPONENTS
#
# Order:
# [11,22,33,12,13,23]
# ============================================================

if args.components == "key":

    COMPONENT_INDICES = [
        0,
        3,
    ]

else:

    COMPONENT_INDICES = [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


print(
    "Mechanics components:",
    COMPONENT_INDICES,
)


# ============================================================
# MECHANICS SCALES
# ============================================================

(
    strain_scale,
    stress_scale,
) = calculate_mechanics_scales(
    LE_tensor,
    S_tensor,
    train_indices,
)


print(
    "Strain scale:",
    strain_scale,
)


print(
    "Stress scale:",
    stress_scale,
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "lambda_strain":
        args.lambda_strain,

    "lambda_stress":
        args.lambda_stress,

    "train_fraction":
        args.train_fraction,

    "components":
        args.components,

    "component_indices":
        COMPONENT_INDICES,

    "strain_scale":
        strain_scale.tolist(),

    "stress_scale":
        stress_scale.tolist(),

    "train_indices":
        train_indices.tolist(),

    "validation_indices":
        validation_indices.tolist(),
}


with open(
    os.path.join(
        RESULTS_DIR,
        "mechanics_config.json",
    ),
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


shutil.copyfile(
    os.path.join(
        BASE_DIR,
        "normalization.json",
    ),
    os.path.join(
        RESULTS_DIR,
        "normalization.json",
    ),
)


# ============================================================
# DATASET
# ============================================================

class CaseDataset(Dataset):

    def __init__(
        self,
        indices,
    ):

        self.indices = np.asarray(
            indices,
            dtype=int,
        )


    def __len__(
        self,
    ):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        local_index,
    ):

        global_index = int(
            self.indices[
                local_index
            ]
        )

        return (
            global_index,

            torch.tensor(
                parameters_normalized[
                    global_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                parameters[
                    global_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                U_normalized[
                    global_index
                ],
                dtype=torch.float32,
            ),
        )


train_loader = DataLoader(
    CaseDataset(
        train_indices
    ),
    batch_size=5,
    shuffle=True,
)


validation_loader = DataLoader(
    CaseDataset(
        validation_indices
    ),
    batch_size=5,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

baseline_checkpoint = torch.load(
    os.path.join(
        BASE_DIR,
        "best_separate_vector_deeponet.pt",
    ),
    map_location=DEVICE,
)


model = SeparateVectorDeepONet(
    branch_dim=3,
    trunk_dim=3,
    latent_dim=baseline_checkpoint.get(
        "latent_dim",
        128,
    ),
).to(
    DEVICE
)


model.load_state_dict(
    baseline_checkpoint[
        "model_state_dict"
    ]
)


print("")
print(
    "Starting from baseline epoch:",
    baseline_checkpoint[
        "epoch"
    ],
)


# ============================================================
# FIXED DATA COORDINATES
# ============================================================

nodal_coordinates_tensor = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MECHANICS ARRAYS ON DEVICE
# ============================================================

ip_coordinates_tensor = torch.tensor(
    ip_coordinates,
    dtype=torch.float32,
    device=DEVICE,
)


LE_tensor_torch = torch.tensor(
    LE_tensor,
    dtype=torch.float32,
    device=DEVICE,
)


S_tensor_torch = torch.tensor(
    S_tensor,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# VALIDATION MECHANICS POINTS
# ============================================================

number_validation_points = min(
    args.mechanics_points,
    len(
        ip_coordinates
    ),
)


validation_ip_indices = np.linspace(
    0,
    len(
        ip_coordinates
    ) - 1,
    number_validation_points,
).astype(
    int
)


validation_ip_indices_torch = torch.tensor(
    validation_ip_indices,
    dtype=torch.long,
    device=DEVICE,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2.0e-4,
    weight_decay=1.0e-6,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        min_lr=1.0e-6,
    )
)


mse = nn.MSELoss()


best_validation = np.inf


best_path = os.path.join(
    RESULTS_DIR,
    "best_mechanics_consistent.pt",
)


history = []


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    1,
    args.epochs + 1,
):

    model.train()

    sum_data = 0.0

    sum_strain = 0.0

    sum_stress = 0.0

    count = 0


    for (
        global_indices,
        parameter_normalized_batch,
        parameter_physical_batch,
        displacement_target_batch,
    ) in train_loader:

        parameter_normalized_batch = (
            parameter_normalized_batch.to(
                DEVICE
            )
        )

        parameter_physical_batch = (
            parameter_physical_batch.to(
                DEVICE
            )
        )

        displacement_target_batch = (
            displacement_target_batch.to(
                DEVICE
            )
        )

        global_indices = (
            global_indices
            .cpu()
            .numpy()
            .astype(
                int
            )
        )


        optimizer.zero_grad()


        # ----------------------------------------------------
        # DISPLACEMENT LOSS
        # ----------------------------------------------------

        displacement_prediction = model(
            parameter_normalized_batch,
            nodal_coordinates_tensor,
        )


        data_loss = mse(
            displacement_prediction,
            displacement_target_batch,
        )


        # ----------------------------------------------------
        # RANDOM MECHANICS POINTS
        # ----------------------------------------------------

        number_points = min(
            args.mechanics_points,
            len(
                ip_coordinates
            ),
        )


        ip_indices = rng.choice(
            len(
                ip_coordinates
            ),
            size=number_points,
            replace=False,
        )


        ip_indices_torch = torch.tensor(
            ip_indices,
            dtype=torch.long,
            device=DEVICE,
        )


        mechanics_coordinates = (
            ip_coordinates_tensor[
                ip_indices_torch
            ]
        )


        strain_target_batch = (
            LE_tensor_torch[
                global_indices
            ][
                :,
                ip_indices_torch,
                :
            ]
        )


        stress_target_batch = (
            S_tensor_torch[
                global_indices
            ][
                :,
                ip_indices_torch,
                :
            ]
        )


        (
            strain_loss,
            stress_loss,
        ) = mechanics_supervision_loss(
            model=model,
            parameters_physical=parameter_physical_batch,
            coordinates_physical=mechanics_coordinates,
            strain_target=strain_target_batch,
            stress_target=stress_target_batch,
            normalization=normalization,
            strain_scale=strain_scale,
            stress_scale=stress_scale,
            component_indices=COMPONENT_INDICES,
            create_graph=True,
        )


        total_loss = (
            data_loss
            +
            args.lambda_strain
            * strain_loss
            +
            args.lambda_stress
            * stress_loss
        )


        total_loss.backward()


        optimizer.step()


        batch_size = len(
            parameter_normalized_batch
        )


        sum_data += (
            data_loss.item()
            * batch_size
        )


        sum_strain += (
            strain_loss.item()
            * batch_size
        )


        sum_stress += (
            stress_loss.item()
            * batch_size
        )


        count += batch_size


    train_data = (
        sum_data
        / count
    )


    train_strain = (
        sum_strain
        / count
    )


    train_stress = (
        sum_stress
        / count
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_data_sum = 0.0

    val_strain_sum = 0.0

    val_stress_sum = 0.0

    val_count = 0


    # Do not use torch.no_grad().
    # Spatial derivatives are needed.
    with torch.enable_grad():

        for (
            global_indices,
            parameter_normalized_batch,
            parameter_physical_batch,
            displacement_target_batch,
        ) in validation_loader:

            parameter_normalized_batch = (
                parameter_normalized_batch.to(
                    DEVICE
                )
            )

            parameter_physical_batch = (
                parameter_physical_batch.to(
                    DEVICE
                )
            )

            displacement_target_batch = (
                displacement_target_batch.to(
                    DEVICE
                )
            )

            global_indices_np = (
                global_indices
                .cpu()
                .numpy()
                .astype(
                    int
                )
            )


            prediction = model(
                parameter_normalized_batch,
                nodal_coordinates_tensor,
            )


            validation_data_loss = mse(
                prediction,
                displacement_target_batch,
            )


            strain_target = (
                LE_tensor_torch[
                    global_indices_np
                ][
                    :,
                    validation_ip_indices_torch,
                    :
                ]
            )


            stress_target = (
                S_tensor_torch[
                    global_indices_np
                ][
                    :,
                    validation_ip_indices_torch,
                    :
                ]
            )


            (
                validation_strain_loss,
                validation_stress_loss,
            ) = mechanics_supervision_loss(
                model=model,
                parameters_physical=parameter_physical_batch,
                coordinates_physical=ip_coordinates_tensor[
                    validation_ip_indices_torch
                ],
                strain_target=strain_target,
                stress_target=stress_target,
                normalization=normalization,
                strain_scale=strain_scale,
                stress_scale=stress_scale,
                component_indices=COMPONENT_INDICES,
                create_graph=False,
            )


            batch_size = len(
                parameter_normalized_batch
            )


            val_data_sum += (
                validation_data_loss.item()
                * batch_size
            )


            val_strain_sum += (
                validation_strain_loss.item()
                * batch_size
            )


            val_stress_sum += (
                validation_stress_loss.item()
                * batch_size
            )


            val_count += batch_size


    val_data = (
        val_data_sum
        / val_count
    )


    val_strain = (
        val_strain_sum
        / val_count
    )


    val_stress = (
        val_stress_sum
        / val_count
    )


    validation_score = (
        val_data
        +
        args.lambda_strain
        * val_strain
        +
        args.lambda_stress
        * val_stress
    )


    scheduler.step(
        validation_score
    )


    history.append(
        {
            "Epoch":
                epoch,

            "TrainDataLoss":
                train_data,

            "TrainStrainLoss":
                train_strain,

            "TrainStressLoss":
                train_stress,

            "ValidationDataLoss":
                val_data,

            "ValidationStrainLoss":
                val_strain,

            "ValidationStressLoss":
                val_stress,

            "ValidationScore":
                validation_score,
        }
    )


    if (
        validation_score
        < best_validation
    ):

        best_validation = (
            validation_score
        )


        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "validation_score":
                    validation_score,

                "branch_dim":
                    3,

                "latent_dim":
                    128,

                "lambda_strain":
                    args.lambda_strain,

                "lambda_stress":
                    args.lambda_stress,
            },
            best_path,
        )


    if (
        epoch == 1
        or epoch % 25 == 0
    ):

        print(
            "Epoch {:5d} | "
            "D {:.3e} | "
            "eps {:.3e} | "
            "sig {:.3e} | "
            "Val {:.3e}"
            .format(
                epoch,
                train_data,
                train_strain,
                train_stress,
                validation_score,
            )
        )


# ============================================================
# SAVE HISTORY
# ============================================================

pd.DataFrame(
    history
).to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv",
    ),
    index=False,
)


print("")
print(
    "Training complete."
)

print(
    "Best validation score:",
    best_validation,
)

print(
    "Best model:",
    best_path,
)