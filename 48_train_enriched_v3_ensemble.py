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

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
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
    "--output_root",
    required=True,
)


parser.add_argument(
    "--members",
    type=int,
    default=5,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=1800,
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


args = parser.parse_args()


os.makedirs(
    args.output_root,
    exist_ok=True,
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
# LOAD ENRICHED DATA
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


force_coefficients = np.load(
    os.path.join(
        args.data_dir,
        "force_coefficients.npy",
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
# INTERFACE CONFIGURATION
# ============================================================

if args.segment == "left":

    interface_1 = "m6"
    interface_2 = "m2"

else:

    interface_1 = "p2"
    interface_2 = "p6"


FORCE_PCA_DIR = os.path.join(
    "data",
    "hybrid_force_pca",
)


pca_1 = np.load(
    os.path.join(
        FORCE_PCA_DIR,
        "force_pca_{}.npz".format(
            interface_1
        ),
    )
)


pca_2 = np.load(
    os.path.join(
        FORCE_PCA_DIR,
        "force_pca_{}.npz".format(
            interface_2
        ),
    )
)


number_modes_1 = pca_1[
    "basis"
].shape[
    1
]


number_modes_2 = pca_2[
    "basis"
].shape[
    1
]


# ============================================================
# SPLITS
# ============================================================

if "Split" in split_dataframe.columns:

    split_column = "Split"

elif "split" in split_dataframe.columns:

    split_column = "split"

else:

    raise RuntimeError(
        "Split column not found."
    )


split_values = (

    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()
)


def get_indices(
    names,
):

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


print("")
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


# ============================================================
# TRAIN-ONLY NORMALIZATION
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


branch_normalized = (

    branch
    -
    branch_mean

) / branch_std


coordinate_min = coordinates.min(
    axis=0
)


coordinate_max = coordinates.max(
    axis=0
)


coordinate_range = np.maximum(

    coordinate_max
    -
    coordinate_min,

    1.0e-8,
)


coordinate_normalized = (

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


U_normalized = (

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


force_mean = force_coefficients[
    train_indices
].mean(
    axis=0
)


force_std = force_coefficients[
    train_indices
].std(
    axis=0
)


force_std = np.maximum(
    force_std,
    1.0e-8,
)


force_normalized = (

    force_coefficients
    -
    force_mean

) / force_std


g_mean = g[
    train_indices
].mean(
    axis=0
)


g_std = g[
    train_indices
].std(
    axis=0
)


g_std = np.maximum(
    g_std,
    1.0e-6,
)


g_normalized = (

    g
    -
    g_mean

) / g_std


# ============================================================
# DATASET
# ============================================================

class LocalDataset(Dataset):

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
                branch_normalized[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                U_normalized[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                force_normalized[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                g_normalized[
                    index
                ],
                dtype=torch.float32,
            ),
        )


train_loader = DataLoader(

    LocalDataset(
        train_indices
    ),

    batch_size=
        args.batch_size,

    shuffle=True,
)


validation_loader = DataLoader(

    LocalDataset(
        validation_indices
    ),

    batch_size=
        args.batch_size,

    shuffle=False,
)


# ============================================================
# CONSTANT TENSORS
# ============================================================

coordinate_tensor = torch.tensor(
    coordinate_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


force_mean_tensor = torch.tensor(
    force_mean,
    dtype=torch.float32,
    device=DEVICE,
)


force_std_tensor = torch.tensor(
    force_std,
    dtype=torch.float32,
    device=DEVICE,
)


g_mean_tensor = torch.tensor(
    g_mean,
    dtype=torch.float32,
    device=DEVICE,
)


g_std_tensor = torch.tensor(
    g_std,
    dtype=torch.float32,
    device=DEVICE,
)


pca_g_mean_1 = torch.tensor(
    pca_1[
        "g_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


pca_g_matrix_1 = torch.tensor(
    pca_1[
        "g_matrix"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


pca_g_mean_2 = torch.tensor(
    pca_2[
        "g_mean"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


pca_g_matrix_2 = torch.tensor(
    pca_2[
        "g_matrix"
    ],
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# FORCE PCA -> GENERALIZED FORCE
# ============================================================

def coefficients_to_g(
    normalized_coefficients,
):

    physical_coefficients = (

        normalized_coefficients

        *

        force_std_tensor

        +

        force_mean_tensor
    )


    coefficient_1 = physical_coefficients[
        :,
        :number_modes_1
    ]


    coefficient_2 = physical_coefficients[
        :,
        number_modes_1:
        number_modes_1
        +
        number_modes_2
    ]


    g1 = (

        pca_g_mean_1.reshape(
            1,
            -1
        )

        +

        coefficient_1

        @

        pca_g_matrix_1.T
    )


    g2 = (

        pca_g_mean_2.reshape(
            1,
            -1
        )

        +

        coefficient_2

        @

        pca_g_matrix_2.T
    )


    g_physical = torch.cat(
        [
            g1,
            g2,
        ],
        dim=1,
    )


    g_normalized_prediction = (

        g_physical

        -

        g_mean_tensor

    ) / g_std_tensor


    return (
        g_normalized_prediction,
        g_physical,
        physical_coefficients,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
):

    model.eval()


    U_predictions = []

    g_predictions = []

    coefficient_predictions = []


    with torch.no_grad():

        for case_index in test_indices:

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


            (
                U_prediction_normalized,
                coefficient_prediction_normalized,
            ) = model(

                branch_tensor,

                coordinate_tensor,
            )


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


            (
                _,
                g_prediction,
                coefficient_prediction,
            ) = coefficients_to_g(
                coefficient_prediction_normalized
            )


            U_predictions.append(
                U_prediction
            )


            g_predictions.append(
                g_prediction[
                    0
                ]
                .cpu()
                .numpy()
            )


            coefficient_predictions.append(
                coefficient_prediction[
                    0
                ]
                .cpu()
                .numpy()
            )


    return (

        np.asarray(
            U_predictions
        ),

        np.asarray(
            g_predictions
        ),

        np.asarray(
            coefficient_predictions
        ),
    )


# ============================================================
# ERROR HELPERS
# ============================================================

def mean_component_error(
    predictions,
    component_index,
):

    errors = []


    for local_index, case_index in enumerate(
        test_indices
    ):

        prediction = predictions[
            local_index,
            :,
            component_index
        ]


        truth = U[
            case_index,
            :,
            component_index
        ]


        error = (

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


        errors.append(
            100.0
            *
            error
        )


    return float(
        np.mean(
            errors
        )
    )


def mean_vector_error(
    predictions,
    truth_array,
):

    errors = []


    for local_index, case_index in enumerate(
        test_indices
    ):

        prediction = predictions[
            local_index
        ]


        truth = truth_array[
            case_index
        ]


        error = (

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


        errors.append(
            100.0
            *
            error
        )


    return float(
        np.mean(
            errors
        )
    )


# ============================================================
# TRAIN ENSEMBLE
# ============================================================

SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


mse = nn.MSELoss()


member_rows = []


all_member_U = []

all_member_g = []

all_member_coefficients = []


for member_index in range(
    args.members
):

    seed = SEEDS[
        member_index
    ]


    print("")
    print(
        "================================================"
    )

    print(
        "MEMBER",
        member_index + 1,
        "SEED",
        seed,
    )

    print(
        "================================================"
    )


    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    member_directory = os.path.join(
        args.output_root,
        "member_{:02d}".format(
            member_index + 1
        ),
    )


    os.makedirs(
        member_directory,
        exist_ok=True,
    )


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


    best_validation = np.inf

    best_epoch = -1

    no_improvement = 0


    history = []


    checkpoint_file = os.path.join(
        member_directory,
        "best_enriched_v3.pt",
    )


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()


        train_sum = 0.0

        train_count = 0


        for (
            branch_batch,
            U_target,
            coefficient_target,
            g_target,
        ) in train_loader:

            branch_batch = branch_batch.to(
                DEVICE
            )


            U_target = U_target.to(
                DEVICE
            )


            coefficient_target = coefficient_target.to(
                DEVICE
            )


            g_target = g_target.to(
                DEVICE
            )


            optimizer.zero_grad()


            (
                U_prediction,
                coefficient_prediction,
            ) = model(

                branch_batch,

                coordinate_tensor,
            )


            (
                g_prediction,
                _,
                _,
            ) = coefficients_to_g(
                coefficient_prediction
            )


            displacement_loss = mse(

                U_prediction,

                U_target,
            )


            coefficient_loss = mse(

                coefficient_prediction,

                coefficient_target,
            )


            generalized_force_loss = mse(

                g_prediction,

                g_target,
            )


            total_loss = (

                displacement_loss

                +

                0.25
                *
                coefficient_loss

                +

                generalized_force_loss
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


            train_sum += (

                total_loss.item()

                *

                batch_size
            )


            train_count += batch_size


        train_loss = (

            train_sum
            /
            train_count
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        validation_sum = 0.0

        validation_count = 0


        with torch.no_grad():

            for (
                branch_batch,
                U_target,
                coefficient_target,
                g_target,
            ) in validation_loader:

                branch_batch = branch_batch.to(
                    DEVICE
                )


                U_target = U_target.to(
                    DEVICE
                )


                coefficient_target = coefficient_target.to(
                    DEVICE
                )


                g_target = g_target.to(
                    DEVICE
                )


                (
                    U_prediction,
                    coefficient_prediction,
                ) = model(

                    branch_batch,

                    coordinate_tensor,
                )


                (
                    g_prediction,
                    _,
                    _,
                ) = coefficients_to_g(
                    coefficient_prediction
                )


                displacement_loss = mse(

                    U_prediction,

                    U_target,
                )


                coefficient_loss = mse(

                    coefficient_prediction,

                    coefficient_target,
                )


                generalized_force_loss = mse(

                    g_prediction,

                    g_target,
                )


                total_loss = (

                    displacement_loss

                    +

                    0.25
                    *
                    coefficient_loss

                    +

                    generalized_force_loss
                )


                batch_size = branch_batch.shape[
                    0
                ]


                validation_sum += (

                    total_loss.item()

                    *

                    batch_size
                )


                validation_count += batch_size


        validation_loss = (

            validation_sum

            /

            validation_count
        )


        scheduler.step(
            validation_loss
        )


        history.append(
            {
                "Epoch":
                    epoch,

                "TrainLoss":
                    train_loss,

                "ValidationLoss":
                    validation_loss,
            }
        )


        if validation_loss < best_validation:

            best_validation = validation_loss

            best_epoch = epoch

            no_improvement = 0


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "segment":
                        args.segment,

                    "seed":
                        seed,

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

                    "best_epoch":
                        epoch,

                    "branch_mean":
                        branch_mean,

                    "branch_std":
                        branch_std,

                    "coordinate_min":
                        coordinate_min,

                    "coordinate_max":
                        coordinate_max,

                    "U_mean":
                        U_mean,

                    "U_std":
                        U_std,

                    "force_coeff_mean":
                        force_mean,

                    "force_coeff_std":
                        force_std,

                    "g_mean":
                        g_mean,

                    "g_std":
                        g_std,
                },

                checkpoint_file,
            )


        else:

            no_improvement += 1


        if (
            epoch == 1
            or
            epoch % 100 == 0
        ):

            print(

                "Epoch {:4d} | "
                "Train {:.4e} | "
                "Val {:.4e}"

                .format(

                    epoch,

                    train_loss,

                    validation_loss,
                )
            )


        if no_improvement >= args.patience:

            print(
                "Early stopping:",
                epoch
            )

            break


    # ========================================================
    # LOAD BEST MEMBER
    # ========================================================

    checkpoint = torch.load(

        checkpoint_file,

        map_location=DEVICE,

        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    (
        member_U,
        member_g,
        member_coefficients,
    ) = evaluate_model(
        model
    )


    all_member_U.append(
        member_U
    )


    all_member_g.append(
        member_g
    )


    all_member_coefficients.append(
        member_coefficients
    )


    member_U1 = mean_component_error(
        member_U,
        0,
    )


    member_U2 = mean_component_error(
        member_U,
        1,
    )


    member_U3 = mean_component_error(
        member_U,
        2,
    )


    member_g_error = mean_vector_error(

        member_g,

        g,
    )


    member_force_error = mean_vector_error(

        member_coefficients,

        force_coefficients,
    )


    member_rows.append(
        {
            "Member":
                member_index + 1,

            "Seed":
                seed,

            "BestEpoch":
                best_epoch,

            "U1_percent":
                member_U1,

            "U2_percent":
                member_U2,

            "U3_percent":
                member_U3,

            "GeneralizedForce_percent":
                member_g_error,

            "ForceCoefficient_percent":
                member_force_error,
        }
    )


    pd.DataFrame(
        history
    ).to_csv(

        os.path.join(
            member_directory,
            "training_history.csv",
        ),

        index=False,
    )


    print(
        "Test U2 = {:.4f}%"
        .format(
            member_U2
        )
    )


# ============================================================
# ENSEMBLE MEAN
# ============================================================

all_member_U = np.asarray(
    all_member_U
)


all_member_g = np.asarray(
    all_member_g
)


all_member_coefficients = np.asarray(
    all_member_coefficients
)


ensemble_U = np.mean(
    all_member_U,
    axis=0,
)


ensemble_g = np.mean(
    all_member_g,
    axis=0,
)


ensemble_coefficients = np.mean(
    all_member_coefficients,
    axis=0,
)


ensemble_U1 = mean_component_error(
    ensemble_U,
    0,
)


ensemble_U2 = mean_component_error(
    ensemble_U,
    1,
)


ensemble_U3 = mean_component_error(
    ensemble_U,
    2,
)


ensemble_g_error = mean_vector_error(

    ensemble_g,

    g,
)


ensemble_force_error = mean_vector_error(

    ensemble_coefficients,

    force_coefficients,
)


# ============================================================
# SAVE
# ============================================================

member_dataframe = pd.DataFrame(
    member_rows
)


member_dataframe.to_csv(

    os.path.join(
        args.output_root,
        "ensemble_member_summary.csv",
    ),

    index=False,
)


summary = {

    "Segment":
        args.segment,

    "Members":
        args.members,

    "TrainCases":
        len(
            train_indices
        ),

    "ValidationCases":
        len(
            validation_indices
        ),

    "TestCases":
        len(
            test_indices
        ),

    "TotalCases":
        len(
            branch
        ),

    "EnsembleMean_U1_percent":
        ensemble_U1,

    "EnsembleMean_U2_percent":
        ensemble_U2,

    "EnsembleMean_U3_percent":
        ensemble_U3,

    "EnsembleMean_GeneralizedForce_percent":
        ensemble_g_error,

    "EnsembleMean_ForceCoefficient_percent":
        ensemble_force_error,
}


summary_file = os.path.join(
    args.output_root,
    "ensemble_summary.json",
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


print("")
print(
    "================================================"
)

print(
    "ENRICHED ENSEMBLE COMPLETE"
)

print(
    "================================================"
)


print(
    json.dumps(
        summary,
        indent=4,
    )
)