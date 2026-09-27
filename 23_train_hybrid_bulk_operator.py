import os
import json
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

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
    "--output_dir",
    required=True,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=3000,
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
)


parser.add_argument(
    "--lambda_force",
    type=float,
    default=1.0,
)


parser.add_argument(
    "--learning_rate",
    type=float,
    default=2.0e-4,
)


parser.add_argument(
    "--patience",
    type=int,
    default=400,
)


parser.add_argument(
    "--device",
    choices=[
        "auto",
        "mps",
        "cpu",
    ],
    default="auto",
)


args = parser.parse_args()


# ============================================================
# REPRODUCIBILITY
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


# ============================================================
# DEVICE
# ============================================================

if args.device == "cpu":

    DEVICE = torch.device(
        "cpu"
    )

elif args.device == "mps":

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "MPS requested but unavailable."
        )

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
    DEVICE
)


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

branch_inputs = np.load(
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


split_table = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


train_indices = split_table.loc[
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


print("")
print(
    "Branch:",
    branch_inputs.shape
)

print(
    "Coordinates:",
    coordinates.shape
)

print(
    "U:",
    U.shape
)

print(
    "g:",
    g.shape
)

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
# TRAIN-ONLY NORMALIZATION
# ============================================================

branch_mean = branch_inputs[
    train_indices
].mean(
    axis=0
)


branch_std = branch_inputs[
    train_indices
].std(
    axis=0
)


branch_std = np.maximum(
    branch_std,
    1.0e-8,
)


branch_normalized = (
    branch_inputs
    -
    branch_mean
) / branch_std


coordinate_min = coordinates.min(
    axis=0
)


coordinate_max = coordinates.max(
    axis=0
)


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


U_train = U[
    train_indices
]


output_mean = U_train.reshape(
    -1,
    3,
).mean(
    axis=0
)


output_std = U_train.reshape(
    -1,
    3,
).std(
    axis=0
)


output_std = np.maximum(
    output_std,
    1.0e-8,
)


U_normalized = (
    U
    -
    output_mean.reshape(
        1,
        1,
        3,
    )
) / output_std.reshape(
    1,
    1,
    3,
)


force_mean = g[
    train_indices
].mean(
    axis=0
)


force_std = g[
    train_indices
].std(
    axis=0
)


force_std = np.maximum(
    force_std,
    1.0e-6,
)


g_normalized = (
    g
    -
    force_mean
) / force_std


# ============================================================
# SAVE NORMALIZATION
# ============================================================

normalization = {

    "branch_mean":
        branch_mean.tolist(),

    "branch_std":
        branch_std.tolist(),

    "coordinate_min":
        coordinate_min.tolist(),

    "coordinate_max":
        coordinate_max.tolist(),

    "output_mean":
        output_mean.tolist(),

    "output_std":
        output_std.tolist(),

    "force_mean":
        force_mean.tolist(),

    "force_std":
        force_std.tolist(),

    "n_left_force_modes":
        int(
            g_left.shape[
                1
            ]
        ),

    "n_right_force_modes":
        int(
            g_right.shape[
                1
            ]
        ),
}


with open(
    os.path.join(
        args.output_dir,
        "normalization.json",
    ),
    "w",
) as f:

    json.dump(
        normalization,
        f,
        indent=4,
    )


# ============================================================
# DATASET
# ============================================================

class BulkDataset(Dataset):

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
            torch.tensor(
                branch_normalized[
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

            torch.tensor(
                g_normalized[
                    global_index
                ],
                dtype=torch.float32,
            ),
        )


train_loader = DataLoader(
    BulkDataset(
        train_indices
    ),
    batch_size=args.batch_size,
    shuffle=True,
)


validation_loader = DataLoader(
    BulkDataset(
        validation_indices
    ),
    batch_size=args.batch_size,
    shuffle=False,
)


coordinate_tensor = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MODEL
# ============================================================

model = HybridBulkOperator(
    branch_dim=branch_inputs.shape[
        1
    ],
    number_force_outputs=g.shape[
        1
    ],
    latent_dim=128,
    hidden_dim=128,
).to(
    DEVICE
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=1.0e-6,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=120,
        min_lr=1.0e-6,
    )
)


mse = nn.MSELoss()


best_validation = np.inf

epochs_without_improvement = 0


best_file = os.path.join(
    args.output_dir,
    "best_hybrid_bulk_operator.pt",
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


    train_U_sum = 0.0

    train_g_sum = 0.0

    train_count = 0


    for (
        branch_batch,
        U_target,
        g_target,
    ) in train_loader:

        branch_batch = branch_batch.to(
            DEVICE
        )


        U_target = U_target.to(
            DEVICE
        )


        g_target = g_target.to(
            DEVICE
        )


        optimizer.zero_grad()


        (
            U_prediction,
            g_prediction,
        ) = model(
            branch_batch,
            coordinate_tensor,
        )


        loss_U = mse(
            U_prediction,
            U_target,
        )


        loss_g = mse(
            g_prediction,
            g_target,
        )


        total_loss = (
            loss_U
            +
            args.lambda_force
            *
            loss_g
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        batch_size = branch_batch.shape[
            0
        ]


        train_U_sum += (
            loss_U.item()
            *
            batch_size
        )


        train_g_sum += (
            loss_g.item()
            *
            batch_size
        )


        train_count += (
            batch_size
        )


    train_U = (
        train_U_sum
        /
        train_count
    )


    train_g = (
        train_g_sum
        /
        train_count
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_U_sum = 0.0

    val_g_sum = 0.0

    val_count = 0


    with torch.no_grad():

        for (
            branch_batch,
            U_target,
            g_target,
        ) in validation_loader:

            branch_batch = branch_batch.to(
                DEVICE
            )


            U_target = U_target.to(
                DEVICE
            )


            g_target = g_target.to(
                DEVICE
            )


            (
                U_prediction,
                g_prediction,
            ) = model(
                branch_batch,
                coordinate_tensor,
            )


            val_U_loss = mse(
                U_prediction,
                U_target,
            )


            val_g_loss = mse(
                g_prediction,
                g_target,
            )


            batch_size = branch_batch.shape[
                0
            ]


            val_U_sum += (
                val_U_loss.item()
                *
                batch_size
            )


            val_g_sum += (
                val_g_loss.item()
                *
                batch_size
            )


            val_count += (
                batch_size
            )


    val_U = (
        val_U_sum
        /
        val_count
    )


    val_g = (
        val_g_sum
        /
        val_count
    )


    validation_score = (
        val_U
        +
        args.lambda_force
        *
        val_g
    )


    scheduler.step(
        validation_score
    )


    history.append(
        {
            "Epoch":
                epoch,

            "TrainULoss":
                train_U,

            "TrainForceLoss":
                train_g,

            "ValidationULoss":
                val_U,

            "ValidationForceLoss":
                val_g,

            "ValidationScore":
                validation_score,

            "LearningRate":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],
        }
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if (
        validation_score
        <
        best_validation
    ):

        best_validation = (
            validation_score
        )


        epochs_without_improvement = 0


        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "branch_dim":
                    branch_inputs.shape[
                        1
                    ],

                "number_force_outputs":
                    g.shape[
                        1
                    ],

                "latent_dim":
                    128,

                "hidden_dim":
                    128,

                "lambda_force":
                    args.lambda_force,

                "validation_score":
                    validation_score,

                "validation_U_loss":
                    val_U,

                "validation_force_loss":
                    val_g,
            },
            best_file,
        )


    else:

        epochs_without_improvement += 1


    if (
        epoch == 1
        or epoch % 25 == 0
    ):

        print(
            "Epoch {:5d} | "
            "U {:.3e} | "
            "g {:.3e} | "
            "vU {:.3e} | "
            "vg {:.3e} | "
            "Score {:.3e}"
            .format(
                epoch,
                train_U,
                train_g,
                val_U,
                val_g,
                validation_score,
            )
        )


    if (
        epochs_without_improvement
        >= args.patience
    ):

        print("")
        print(
            "Early stopping at epoch:",
            epoch
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

pd.DataFrame(
    history
).to_csv(
    os.path.join(
        args.output_dir,
        "training_history.csv",
    ),
    index=False,
)


print("")
print(
    "============================================"
)

print(
    "TRAINING COMPLETE"
)

print(
    "============================================"
)

print(
    "Best validation score:",
    best_validation
)

print(
    "Checkpoint:",
    best_file
)