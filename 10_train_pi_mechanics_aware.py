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
    sample_contact_aware_interior,
    equilibrium_loss_single,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--base_model_dir",
    type=str,
    default=(
        "results/mechanics_consistent/"
        "eps_0.1_sig_0.1_frac_1"
    ),
)


parser.add_argument(
    "--base_checkpoint",
    type=str,
    default="best_mechanics_consistent.pt",
)


# ------------------------------------------------------------
# TRAINING LOSS WEIGHTS
# ------------------------------------------------------------

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
    "--lambda_eq",
    type=float,
    default=0.01,
)


# ------------------------------------------------------------
# FIXED VALIDATION WEIGHTS
#
# IMPORTANT:
# These remain the same for ALL lambda_eq experiments.
# This makes validation scores directly comparable.
# ------------------------------------------------------------

parser.add_argument(
    "--val_weight_strain",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--val_weight_stress",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--val_weight_eq",
    type=float,
    default=0.1,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=1500,
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)


parser.add_argument(
    "--mechanics_points",
    type=int,
    default=128,
)


parser.add_argument(
    "--physics_points",
    type=int,
    default=128,
)


parser.add_argument(
    "--validation_mechanics_points",
    type=int,
    default=256,
)


parser.add_argument(
    "--validation_physics_points",
    type=int,
    default=256,
)


parser.add_argument(
    "--learning_rate",
    type=float,
    default=1.0e-4,
)


parser.add_argument(
    "--patience",
    type=int,
    default=250,
)


parser.add_argument(
    "--residual_reference",
    type=float,
    default=562.5,
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


# Separate RNGs are deliberate:
#
# training_rng:
# changes training collocation points
#
# validation_rng:
# creates one fixed validation set
# ============================================================

training_rng = np.random.default_rng(
    SEED
)


validation_mechanics_rng = (
    np.random.default_rng(
        SEED + 1001
    )
)


validation_physics_rng = (
    np.random.default_rng(
        SEED + 2002
    )
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"


RESULTS_ROOT = os.path.join(
    "results",
    "pi_mechanics_aware",
)


RESULTS_DIR = os.path.join(
    RESULTS_ROOT,
    "eq_{:g}".format(
        args.lambda_eq
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

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "MPS was requested but is not available."
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


print("")
print(
    "========================================"
)

print(
    "PI-DEEPONET TRAINING"
)

print(
    "========================================"
)

print(
    "Device:",
    DEVICE,
)

print(
    "Training lambda_eq:",
    args.lambda_eq,
)

print(
    "Fixed validation equilibrium weight:",
    args.val_weight_eq,
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


print(
    "Train cases:",
    len(
        train_indices
    ),
)

print(
    "Validation cases:",
    len(
        validation_indices
    ),
)


# ============================================================
# NORMALIZATION
# ============================================================

normalization_file = os.path.join(
    args.base_model_dir,
    "normalization.json",
)


with open(
    normalization_file,
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


shutil.copyfile(
    normalization_file,
    os.path.join(
        RESULTS_DIR,
        "normalization.json",
    ),
)


# ============================================================
# MECHANICS COMPONENTS
#
# Tensor order:
# [11,22,33,12,13,23]
#
# Current mechanics supervision:
# 11 -> bending
# 12 -> primary short-beam shear
# ============================================================

COMPONENT_INDICES = [
    0,
    3,
]


# ============================================================
# TRAIN-ONLY MECHANICS NORMALIZATION
# ============================================================

(
    strain_scale,
    stress_scale,
) = calculate_mechanics_scales(
    LE_tensor,
    S_tensor,
    train_indices,
)


print("")
print(
    "Strain scales:",
    strain_scale,
)

print(
    "Stress scales:",
    stress_scale,
)


# ============================================================
# DATASET
# ============================================================

class CaseDataset(Dataset):

    def __init__(
        self,
        case_indices,
    ):

        self.case_indices = np.asarray(
            case_indices,
            dtype=int,
        )


    def __len__(
        self,
    ):

        return len(
            self.case_indices
        )


    def __getitem__(
        self,
        local_index,
    ):

        global_index = int(
            self.case_indices[
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
    batch_size=args.batch_size,
    shuffle=True,
)


validation_loader = DataLoader(
    CaseDataset(
        validation_indices
    ),
    batch_size=args.batch_size,
    shuffle=False,
)


# ============================================================
# LOAD MECHANICS-CONSISTENT STARTING MODEL
# ============================================================

base_checkpoint_path = os.path.join(
    args.base_model_dir,
    args.base_checkpoint,
)


base_checkpoint = torch.load(
    base_checkpoint_path,
    map_location=DEVICE,
)


model = SeparateVectorDeepONet(
    branch_dim=base_checkpoint.get(
        "branch_dim",
        3,
    ),
    trunk_dim=3,
    latent_dim=base_checkpoint.get(
        "latent_dim",
        128,
    ),
).to(
    DEVICE
)


model.load_state_dict(
    base_checkpoint[
        "model_state_dict"
    ]
)


print("")
print(
    "Loaded starting checkpoint:"
)

print(
    base_checkpoint_path
)


# ============================================================
# STATIC TENSORS
# ============================================================

nodal_coordinates_tensor = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


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
# FIXED VALIDATION MECHANICS POINTS
#
# These do NOT change from epoch to epoch.
# ============================================================

number_validation_mechanics = min(
    args.validation_mechanics_points,
    len(
        ip_coordinates
    ),
)


validation_mechanics_indices = (
    validation_mechanics_rng.choice(
        len(
            ip_coordinates
        ),
        size=number_validation_mechanics,
        replace=False,
    )
)


validation_mechanics_indices = np.sort(
    validation_mechanics_indices
)


validation_mechanics_indices_tensor = (
    torch.tensor(
        validation_mechanics_indices,
        dtype=torch.long,
        device=DEVICE,
    )
)


validation_mechanics_coordinates = (
    ip_coordinates_tensor[
        validation_mechanics_indices_tensor
    ]
)


# ============================================================
# FIXED VALIDATION PHYSICS POINTS
#
# Same physical locations every epoch and for every lambda run.
# ============================================================

validation_physics_coordinates = (
    sample_contact_aware_interior(
        nodal_coordinates=coordinates,
        number_points=(
            args.validation_physics_points
        ),
        device=DEVICE,
        rng=validation_physics_rng,
    )
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "validation_mechanics_indices.npy",
    ),
    validation_mechanics_indices,
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "validation_physics_coordinates.npy",
    ),
    (
        validation_physics_coordinates
        .detach()
        .cpu()
        .numpy()
    ),
)


# ============================================================
# OPTIMIZER
# ============================================================

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
        patience=100,
        min_lr=1.0e-6,
    )
)


mse = nn.MSELoss()


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "training_lambda_strain":
        args.lambda_strain,

    "training_lambda_stress":
        args.lambda_stress,

    "training_lambda_eq":
        args.lambda_eq,

    "validation_weight_strain":
        args.val_weight_strain,

    "validation_weight_stress":
        args.val_weight_stress,

    "validation_weight_eq":
        args.val_weight_eq,

    "mechanics_components":
        [
            "11",
            "12",
        ],

    "residual_reference_MPa_per_mm":
        args.residual_reference,

    "training_mechanics_points":
        args.mechanics_points,

    "training_physics_points":
        args.physics_points,

    "validation_mechanics_points":
        args.validation_mechanics_points,

    "validation_physics_points":
        args.validation_physics_points,

    "train_indices":
        train_indices.tolist(),

    "validation_indices":
        validation_indices.tolist(),
}


with open(
    os.path.join(
        RESULTS_DIR,
        "training_config.json",
    ),
    "w",
) as f:

    json.dump(
        config,
        f,
        indent=4,
    )


# ============================================================
# TRAINING STATE
# ============================================================

best_validation_score = np.inf

epochs_without_improvement = 0


best_path = os.path.join(
    RESULTS_DIR,
    "best_pi_deeponet.pt",
)


history = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    args.epochs + 1,
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    train_data_sum = 0.0

    train_strain_sum = 0.0

    train_stress_sum = 0.0

    train_eq_sum = 0.0

    train_count = 0


    for (
        global_indices,
        parameter_normalized_batch,
        parameter_physical_batch,
        displacement_target_batch,
    ) in train_loader:

        global_indices_device = (
            global_indices
            .to(
                DEVICE
            )
            .long()
        )


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


        optimizer.zero_grad()


        # ----------------------------------------------------
        # 1. DISPLACEMENT LOSS
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
        # 2. RANDOM MECHANICS SUPERVISION POINTS
        # ----------------------------------------------------

        number_mechanics = min(
            args.mechanics_points,
            len(
                ip_coordinates
            ),
        )


        mechanics_indices = (
            training_rng.choice(
                len(
                    ip_coordinates
                ),
                size=number_mechanics,
                replace=False,
            )
        )


        mechanics_indices_tensor = (
            torch.tensor(
                mechanics_indices,
                dtype=torch.long,
                device=DEVICE,
            )
        )


        (
            strain_loss,
            stress_loss,
        ) = mechanics_supervision_loss(
            model=model,

            parameters_physical=(
                parameter_physical_batch
            ),

            coordinates_physical=(
                ip_coordinates_tensor[
                    mechanics_indices_tensor
                ]
            ),

            strain_target=(
                LE_tensor_torch[
                    global_indices_device
                ][
                    :,
                    mechanics_indices_tensor,
                    :
                ]
            ),

            stress_target=(
                S_tensor_torch[
                    global_indices_device
                ][
                    :,
                    mechanics_indices_tensor,
                    :
                ]
            ),

            normalization=normalization,

            strain_scale=strain_scale,

            stress_scale=stress_scale,

            component_indices=(
                COMPONENT_INDICES
            ),

            create_graph=True,
        )


        # ----------------------------------------------------
        # 3. RANDOM CONTACT-AWARE PHYSICS POINTS
        # ----------------------------------------------------

        physics_coordinates = (
            sample_contact_aware_interior(
                nodal_coordinates=coordinates,
                number_points=(
                    args.physics_points
                ),
                device=DEVICE,
                rng=training_rng,
            )
        )


        equilibrium_losses = []


        for b in range(
            parameter_physical_batch.shape[
                0
            ]
        ):

            equilibrium_losses.append(
                equilibrium_loss_single(
                    model=model,

                    parameter_physical=(
                        parameter_physical_batch[
                            b
                        ]
                    ),

                    coordinates_physical=(
                        physics_coordinates
                    ),

                    normalization=(
                        normalization
                    ),

                    residual_reference=(
                        args.residual_reference
                    ),

                    create_graph_second=True,
                )
            )


        equilibrium_loss = (
            torch.stack(
                equilibrium_losses
            )
            .mean()
        )


        # ----------------------------------------------------
        # TOTAL TRAINING LOSS
        # ----------------------------------------------------

        total_loss = (
            data_loss

            +
            args.lambda_strain
            * strain_loss

            +
            args.lambda_stress
            * stress_loss

            +
            args.lambda_eq
            * equilibrium_loss
        )


        total_loss.backward()


        # Helps protect second-derivative training
        # from occasional large gradient spikes.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        batch_size = (
            parameter_physical_batch.shape[
                0
            ]
        )


        train_data_sum += (
            data_loss.item()
            * batch_size
        )


        train_strain_sum += (
            strain_loss.item()
            * batch_size
        )


        train_stress_sum += (
            stress_loss.item()
            * batch_size
        )


        train_eq_sum += (
            equilibrium_loss.item()
            * batch_size
        )


        train_count += (
            batch_size
        )


    train_data = (
        train_data_sum
        / train_count
    )


    train_strain = (
        train_strain_sum
        / train_count
    )


    train_stress = (
        train_stress_sum
        / train_count
    )


    train_eq = (
        train_eq_sum
        / train_count
    )


    # ========================================================
    # MECHANICS-AWARE VALIDATION
    # ========================================================

    model.eval()


    val_data_sum = 0.0

    val_strain_sum = 0.0

    val_stress_sum = 0.0

    val_eq_sum = 0.0

    val_count = 0


    for (
        global_indices,
        parameter_normalized_batch,
        parameter_physical_batch,
        displacement_target_batch,
    ) in validation_loader:

        global_indices_device = (
            global_indices
            .to(
                DEVICE
            )
            .long()
        )


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


        # ----------------------------------------------------
        # VALIDATION DISPLACEMENT
        # ----------------------------------------------------

        with torch.no_grad():

            displacement_prediction = model(
                parameter_normalized_batch,
                nodal_coordinates_tensor,
            )


            validation_data_loss = mse(
                displacement_prediction,
                displacement_target_batch,
            )


        # ----------------------------------------------------
        # VALIDATION MECHANICS + EQUILIBRIUM
        #
        # Gradients are required, but NO optimizer step occurs.
        # ----------------------------------------------------

        with torch.enable_grad():

            (
                validation_strain_loss,
                validation_stress_loss,
            ) = mechanics_supervision_loss(
                model=model,

                parameters_physical=(
                    parameter_physical_batch
                ),

                coordinates_physical=(
                    validation_mechanics_coordinates
                ),

                strain_target=(
                    LE_tensor_torch[
                        global_indices_device
                    ][
                        :,
                        validation_mechanics_indices_tensor,
                        :
                    ]
                ),

                stress_target=(
                    S_tensor_torch[
                        global_indices_device
                    ][
                        :,
                        validation_mechanics_indices_tensor,
                        :
                    ]
                ),

                normalization=(
                    normalization
                ),

                strain_scale=(
                    strain_scale
                ),

                stress_scale=(
                    stress_scale
                ),

                component_indices=(
                    COMPONENT_INDICES
                ),

                create_graph=False,
            )


            validation_equilibrium_losses = []


            for b in range(
                parameter_physical_batch.shape[
                    0
                ]
            ):

                validation_equilibrium_losses.append(
                    equilibrium_loss_single(
                        model=model,

                        parameter_physical=(
                            parameter_physical_batch[
                                b
                            ]
                        ),

                        coordinates_physical=(
                            validation_physics_coordinates
                        ),

                        normalization=(
                            normalization
                        ),

                        residual_reference=(
                            args.residual_reference
                        ),

                        create_graph_second=False,
                    )
                )


            validation_equilibrium_loss = (
                torch.stack(
                    validation_equilibrium_losses
                )
                .mean()
            )


        batch_size = (
            parameter_physical_batch.shape[
                0
            ]
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


        val_eq_sum += (
            validation_equilibrium_loss.item()
            * batch_size
        )


        val_count += (
            batch_size
        )


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


    val_eq = (
        val_eq_sum
        / val_count
    )


    # ========================================================
    # FIXED VALIDATION SCORE
    #
    # IMPORTANT:
    #
    # lambda_eq used during TRAINING may change.
    #
    # The weights below remain fixed so that validation scores
    # can be compared across lambda_eq experiments.
    # ========================================================

    validation_score = (
        val_data

        +
        args.val_weight_strain
        * val_strain

        +
        args.val_weight_stress
        * val_stress

        +
        args.val_weight_eq
        * val_eq
    )


    scheduler.step(
        validation_score
    )


    # ========================================================
    # HISTORY
    # ========================================================

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

            "TrainEquilibriumLoss":
                train_eq,

            "ValidationDataLoss":
                val_data,

            "ValidationStrainLoss":
                val_strain,

            "ValidationStressLoss":
                val_stress,

            "ValidationEquilibriumLoss":
                val_eq,

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
    # BEST CHECKPOINT
    # ========================================================

    if (
        validation_score
        < best_validation_score
    ):

        best_validation_score = (
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
                    3,

                "latent_dim":
                    base_checkpoint.get(
                        "latent_dim",
                        128,
                    ),

                "training_lambda_strain":
                    args.lambda_strain,

                "training_lambda_stress":
                    args.lambda_stress,

                "training_lambda_eq":
                    args.lambda_eq,

                "validation_weight_strain":
                    args.val_weight_strain,

                "validation_weight_stress":
                    args.val_weight_stress,

                "validation_weight_eq":
                    args.val_weight_eq,

                "ValidationDataLoss":
                    val_data,

                "ValidationStrainLoss":
                    val_strain,

                "ValidationStressLoss":
                    val_stress,

                "ValidationEquilibriumLoss":
                    val_eq,

                "ValidationScore":
                    validation_score,
            },
            best_path,
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # PRINT
    # ========================================================

    if (
        epoch == 1
        or epoch % 25 == 0
    ):

        print(
            "Epoch {:5d} | "
            "D {:.3e} | "
            "eps {:.3e} | "
            "sig {:.3e} | "
            "eq {:.3e} || "
            "vD {:.3e} | "
            "vEps {:.3e} | "
            "vSig {:.3e} | "
            "vEq {:.3e} | "
            "Score {:.3e}"
            .format(
                epoch,

                train_data,
                train_strain,
                train_stress,
                train_eq,

                val_data,
                val_strain,
                val_stress,
                val_eq,

                validation_score,
            )
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= args.patience
    ):

        print("")
        print(
            "Early stopping at epoch:",
            epoch,
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_dataframe = pd.DataFrame(
    history
)


history_dataframe.to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv",
    ),
    index=False,
)


print("")
print(
    "========================================"
)

print(
    "PI TRAINING COMPLETE"
)

print(
    "========================================"
)

print(
    "Best validation score:",
    best_validation_score,
)

print(
    "Best checkpoint:"
)

print(
    best_path
)