import os
import json
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
)

from src.orthotropic_fe_physics import (
    StructuredHexPhysics
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
    "--output_dir",
    required=True,
)


parser.add_argument(
    "--physics_weight",
    type=float,
    default=0.01,
)


parser.add_argument(
    "--force_coefficient_weight",
    type=float,
    default=0.25,
)


parser.add_argument(
    "--generalized_force_weight",
    type=float,
    default=1.0,
)


parser.add_argument(
    "--boundary_weight",
    type=float,
    default=0.10,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=2200,
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
)


parser.add_argument(
    "--learning_rate",
    type=float,
    default=2.0e-4,
)


parser.add_argument(
    "--patience",
    type=int,
    default=300,
)


parser.add_argument(
    "--physics_warmup",
    type=int,
    default=200,
)


parser.add_argument(
    "--seed",
    type=int,
    default=2026,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# SEED
# ============================================================

random.seed(
    args.seed
)

np.random.seed(
    args.seed
)

torch.manual_seed(
    args.seed
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(

    "mps"

    if torch.backends.mps.is_available()

    else "cpu"
)


print(
    "Device:",
    DEVICE
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


g = np.concatenate(
    [

        np.load(
            os.path.join(
                args.data_dir,
                "g_left.npy",
            )
        ),

        np.load(
            os.path.join(
                args.data_dir,
                "g_right.npy",
            )
        ),
    ],

    axis=1,
).astype(
    np.float32
)


force_pca_dir = os.path.join(
    "data",
    "hybrid_force_pca",
)


force_coefficients = np.load(
    os.path.join(
        force_pca_dir,
        "{}_force_coefficients.npy".format(
            args.segment
        ),
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
# INTERFACE NAMES
# ============================================================

if args.segment == "left":

    INTERFACE_1 = "m6"

    INTERFACE_2 = "m2"

else:

    INTERFACE_1 = "p2"

    INTERFACE_2 = "p6"


force_pca_1 = np.load(
    os.path.join(
        force_pca_dir,
        "force_pca_{}.npz".format(
            INTERFACE_1
        ),
    )
)


force_pca_2 = np.load(
    os.path.join(
        force_pca_dir,
        "force_pca_{}.npz".format(
            INTERFACE_2
        ),
    )
)


# ============================================================
# SPLIT
# ============================================================

split_column = (
    "Split"
    if
    "Split"
    in
    split_dataframe.columns
    else
    "split"
)


split_values = (
    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()
)


def get_indices(names):

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


test_indices = get_indices(
    [
        "test",
    ]
)


print(
    "Train:",
    len(train_indices)
)

print(
    "Validation:",
    len(validation_indices)
)

print(
    "Test:",
    len(test_indices)
)


# ============================================================
# NORMALIZATION
# ============================================================

branch_mean = branch[
    train_indices
].mean(
    axis=0
)


branch_std = branch[
    train_indices
].std(
    axis=0
)


branch_std = np.maximum(
    branch_std,
    1.0e-8,
)


branch_norm = (

    branch
    -
    branch_mean

) / branch_std


coord_min = coordinates.min(
    axis=0
)


coord_max = coordinates.max(
    axis=0
)


coord_range = np.maximum(

    coord_max
    -
    coord_min,

    1.0e-8,
)


coord_norm = (

    2.0
    *
    (
        coordinates
        -
        coord_min
    )
    /
    coord_range

    -
    1.0
)


U_train = U[
    train_indices
]


U_mean = U_train.reshape(
    -1,
    3,
).mean(
    axis=0
)


U_std = U_train.reshape(
    -1,
    3,
).std(
    axis=0
)


U_std = np.maximum(
    U_std,
    1.0e-8,
)


U_norm = (

    U
    -
    U_mean.reshape(
        1,
        1,
        3,
    )

) / U_std.reshape(
    1,
    1,
    3,
)


force_coeff_mean = force_coefficients[
    train_indices
].mean(
    axis=0
)


force_coeff_std = force_coefficients[
    train_indices
].std(
    axis=0
)


force_coeff_std = np.maximum(
    force_coeff_std,
    1.0e-8,
)


force_coeff_norm = (

    force_coefficients
    -
    force_coeff_mean

) / force_coeff_std


g_mean_training = g[
    train_indices
].mean(
    axis=0
)


g_std_training = g[
    train_indices
].std(
    axis=0
)


g_std_training = np.maximum(
    g_std_training,
    1.0e-6,
)


g_norm = (

    g
    -
    g_mean_training

) / g_std_training


# ============================================================
# FORCE PCA -> GENERALIZED FORCE MAPPING
# ============================================================

g_mean_1 = torch.tensor(
    force_pca_1[
        "g_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


g_matrix_1 = torch.tensor(
    force_pca_1[
        "g_matrix"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


g_mean_2 = torch.tensor(
    force_pca_2[
        "g_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


g_matrix_2 = torch.tensor(
    force_pca_2[
        "g_matrix"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


number_force_modes = force_pca_1[
    "basis"
].shape[
    1
]


# ============================================================
# DATASET
# ============================================================

class DatasetLocal(Dataset):

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
        item,
    ):

        index = self.indices[
            item
        ]


        return (

            torch.tensor(
                branch[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                branch_norm[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                U[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                U_norm[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                force_coeff_norm[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                g_norm[
                    index
                ],
                dtype=torch.float32,
            ),
        )


train_loader = DataLoader(

    DatasetLocal(
        train_indices
    ),

    batch_size=
        args.batch_size,

    shuffle=True,
)


validation_loader = DataLoader(

    DatasetLocal(
        validation_indices
    ),

    batch_size=
        args.batch_size,

    shuffle=False,
)


# ============================================================
# TENSORS
# ============================================================

coordinates_tensor = torch.tensor(
    coord_norm,
    dtype=torch.float32,
    device=DEVICE,
)


U_mean_tensor = torch.tensor(
    U_mean,
    dtype=torch.float32,
    device=DEVICE,
)


U_std_tensor = torch.tensor(
    U_std,
    dtype=torch.float32,
    device=DEVICE,
)


force_coeff_mean_tensor = torch.tensor(
    force_coeff_mean,
    dtype=torch.float32,
    device=DEVICE,
)


force_coeff_std_tensor = torch.tensor(
    force_coeff_std,
    dtype=torch.float32,
    device=DEVICE,
)


g_mean_training_tensor = torch.tensor(
    g_mean_training,
    dtype=torch.float32,
    device=DEVICE,
)


g_std_training_tensor = torch.tensor(
    g_std_training,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MODEL
# ============================================================

model = HybridBulkOperatorV3(

    branch_dim=
        branch.shape[
            1
        ],

    number_force_coefficients=
        force_coefficients.shape[
            1
        ],

    hidden_dim=
        128,

    latent_dim=
        128,

    depth=
        4,
).to(
    DEVICE
)


physics = StructuredHexPhysics(

    coordinates,

    DEVICE,
)


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=
        args.learning_rate,

    weight_decay=
        1.0e-6,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=100,

    min_lr=1.0e-6,
)


mse = nn.MSELoss()


# ============================================================
# PHYSICS NORMALIZATION
# ============================================================

print("")
print(
    "Calculating physics scale..."
)


physics_values = []


with torch.no_grad():

    for start in range(
        0,
        len(
            train_indices
        ),
        8,
    ):

        selected = train_indices[
            start:
            start + 8
        ]


        residual = physics.free_residual(

            torch.tensor(
                U[
                    selected
                ],
                dtype=torch.float32,
                device=DEVICE,
            ),

            torch.tensor(
                branch[
                    selected,
                    0:3
                ],
                dtype=torch.float32,
                device=DEVICE,
            ),
        )


        physics_values.append(
            residual.detach().cpu().numpy()
        )


physics_values = np.concatenate(
    physics_values,
    axis=0,
)


physics_scale = float(

    np.sqrt(
        np.mean(
            physics_values
            ** 2
        )
    )
)


physics_scale = max(
    physics_scale,
    1.0,
)


print(
    "Physics scale:",
    physics_scale
)


# ============================================================
# PREDICT GENERALIZED FORCE FROM FORCE PCA
# ============================================================

def force_coefficients_to_g(
    predicted_coeff_norm,
):

    physical_coefficients = (

        predicted_coeff_norm

        *
        force_coeff_std_tensor

        +

        force_coeff_mean_tensor
    )


    coefficients_1 = physical_coefficients[
        :,
        :number_force_modes
    ]


    coefficients_2 = physical_coefficients[
        :,
        number_force_modes:
    ]


    g1 = (

        g_mean_1.reshape(
            1,
            -1
        )

        +

        coefficients_1
        @
        g_matrix_1.T
    )


    g2 = (

        g_mean_2.reshape(
            1,
            -1
        )

        +

        coefficients_2
        @
        g_matrix_2.T
    )


    g_physical = torch.cat(
        [
            g1,
            g2,
        ],
        dim=1,
    )


    g_prediction_norm = (

        g_physical
        -
        g_mean_training_tensor

    ) / g_std_training_tensor


    return (
        g_prediction_norm,
        physical_coefficients,
    )


# ============================================================
# PHYSICS LOSS
# ============================================================

def physics_loss_function(
    prediction_normalized,
    truth_physical,
    branch_physical,
):

    prediction_physical = (

        prediction_normalized

        *
        U_std_tensor.reshape(
            1,
            1,
            3,
        )

        +

        U_mean_tensor.reshape(
            1,
            1,
            3,
        )
    )


    material = branch_physical[
        :,
        0:3
    ]


    predicted_residual = physics.free_residual(

        prediction_physical,

        material,
    )


    with torch.no_grad():

        truth_residual = physics.free_residual(

            truth_physical,

            material,
        )


    difference = (

        predicted_residual
        -
        truth_residual

    ) / physics_scale


    return torch.mean(
        difference
        ** 2
    )


# ============================================================
# TRAINING
# ============================================================

best_validation_score = np.inf

best_epoch = -1

no_improvement = 0


history = []


checkpoint_file = os.path.join(
    args.output_dir,
    "best_force_physics_operator.pt",
)


for epoch in range(
    1,
    args.epochs
    +
    1,
):

    model.train()


    train_total = 0.0

    train_count = 0


    warmup = min(

        1.0,

        epoch
        /
        float(
            max(
                args.physics_warmup,
                1,
            )
        ),
    )


    effective_physics_weight = (

        args.physics_weight
        *
        warmup
    )


    for (
        branch_physical,
        branch_input,
        U_physical,
        U_target,
        force_coeff_target,
        g_target,
    ) in train_loader:

        branch_physical = branch_physical.to(
            DEVICE
        )


        branch_input = branch_input.to(
            DEVICE
        )


        U_physical = U_physical.to(
            DEVICE
        )


        U_target = U_target.to(
            DEVICE
        )


        force_coeff_target = force_coeff_target.to(
            DEVICE
        )


        g_target = g_target.to(
            DEVICE
        )


        optimizer.zero_grad()


        (
            U_prediction,
            force_coeff_prediction,
        ) = model(

            branch_input,

            coordinates_tensor,
        )


        (
            g_prediction,
            _
        ) = force_coefficients_to_g(
            force_coeff_prediction
        )


        displacement_loss = mse(

            U_prediction,

            U_target,
        )


        coefficient_loss = mse(

            force_coeff_prediction,

            force_coeff_target,
        )


        generalized_force_loss = mse(

            g_prediction,

            g_target,
        )


        physics_loss = physics_loss_function(

            U_prediction,

            U_physical,

            branch_physical,
        )


        total_loss = (

            displacement_loss

            +

            args.force_coefficient_weight
            *
            coefficient_loss

            +

            args.generalized_force_weight
            *
            generalized_force_loss

            +

            effective_physics_weight
            *
            physics_loss
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0,
        )


        optimizer.step()


        batch_size = branch_input.shape[
            0
        ]


        train_total += (
            total_loss.item()
            *
            batch_size
        )


        train_count += batch_size


    train_total /= train_count


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_displacement = 0.0

    val_coefficients = 0.0

    val_g = 0.0

    val_physics = 0.0

    val_count = 0


    with torch.no_grad():

        for (
            branch_physical,
            branch_input,
            U_physical,
            U_target,
            force_coeff_target,
            g_target,
        ) in validation_loader:

            branch_physical = branch_physical.to(
                DEVICE
            )


            branch_input = branch_input.to(
                DEVICE
            )


            U_physical = U_physical.to(
                DEVICE
            )


            U_target = U_target.to(
                DEVICE
            )


            force_coeff_target = force_coeff_target.to(
                DEVICE
            )


            g_target = g_target.to(
                DEVICE
            )


            (
                U_prediction,
                force_coeff_prediction,
            ) = model(

                branch_input,

                coordinates_tensor,
            )


            (
                g_prediction,
                _
            ) = force_coefficients_to_g(
                force_coeff_prediction
            )


            displacement_loss = mse(

                U_prediction,

                U_target,
            )


            coefficient_loss = mse(

                force_coeff_prediction,

                force_coeff_target,
            )


            generalized_force_loss = mse(

                g_prediction,

                g_target,
            )


            physics_loss = physics_loss_function(

                U_prediction,

                U_physical,

                branch_physical,
            )


            batch_size = branch_input.shape[
                0
            ]


            val_displacement += (
                displacement_loss.item()
                *
                batch_size
            )


            val_coefficients += (
                coefficient_loss.item()
                *
                batch_size
            )


            val_g += (
                generalized_force_loss.item()
                *
                batch_size
            )


            val_physics += (
                physics_loss.item()
                *
                batch_size
            )


            val_count += batch_size


    val_displacement /= val_count

    val_coefficients /= val_count

    val_g /= val_count

    val_physics /= val_count


    # --------------------------------------------------------
    # We select the best model using prediction accuracy.
    #
    # Physics is used to GUIDE training, but not to decide
    # which checkpoint wins.
    # --------------------------------------------------------

    validation_score = (

        val_displacement

        +

        args.force_coefficient_weight
        *
        val_coefficients

        +

        args.generalized_force_weight
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

            "TrainTotal":
                train_total,

            "ValDisplacement":
                val_displacement,

            "ValForceCoefficients":
                val_coefficients,

            "ValGeneralizedForce":
                val_g,

            "ValPhysics":
                val_physics,

            "ValidationScore":
                validation_score,
        }
    )


    if validation_score < best_validation_score:

        best_validation_score = validation_score

        best_epoch = epoch

        no_improvement = 0


        torch.save(
            {

                "model_state_dict":
                    model.state_dict(),

                "segment":
                    args.segment,

                "branch_dim":
                    branch.shape[
                        1
                    ],

                "number_force_coefficients":
                    force_coefficients.shape[
                        1
                    ],

                "hidden_dim":
                    128,

                "latent_dim":
                    128,

                "depth":
                    4,

                "physics_weight":
                    args.physics_weight,

                "force_coefficient_weight":
                    args.force_coefficient_weight,

                "generalized_force_weight":
                    args.generalized_force_weight,

                "best_epoch":
                    epoch,

                "branch_mean":
                    branch_mean,

                "branch_std":
                    branch_std,

                "coord_min":
                    coord_min,

                "coord_max":
                    coord_max,

                "U_mean":
                    U_mean,

                "U_std":
                    U_std,

                "force_coeff_mean":
                    force_coeff_mean,

                "force_coeff_std":
                    force_coeff_std,

                "g_training_mean":
                    g_mean_training,

                "g_training_std":
                    g_std_training,

                "physics_scale":
                    physics_scale,
            },

            checkpoint_file,
        )


    else:

        no_improvement += 1


    if (
        epoch == 1
        or
        epoch % 50 == 0
    ):

        print(

            "Epoch {:4d} | "
            "U {:.3e} | "
            "Coeff {:.3e} | "
            "g {:.3e} | "
            "Phys {:.3e} | "
            "Score {:.3e}".format(

                epoch,

                val_displacement,

                val_coefficients,

                val_g,

                val_physics,

                validation_score,
            )
        )


    if no_improvement >= args.patience:

        print("")
        print(
            "Early stopping at epoch",
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
    "================================================"
)

print(
    "TRAINING COMPLETE"
)

print(
    "================================================"
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation score:",
    best_validation_score
)

print(
    "Saved:",
    checkpoint_file
)