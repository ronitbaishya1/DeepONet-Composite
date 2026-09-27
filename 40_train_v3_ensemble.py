import os
import json
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from src.hybrid_bulk_operator_v3 import HybridBulkOperatorV3


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--segment",
    choices=["left", "right"],
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

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

branch = np.load(
    os.path.join(
        args.data_dir,
        "branch_inputs.npy",
    )
).astype(np.float32)


coordinates = np.load(
    os.path.join(
        args.data_dir,
        "coordinates.npy",
    )
).astype(np.float32)


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
).astype(np.float32)


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
).astype(np.float32)


split_dataframe = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# FORCE PCA COEFFICIENTS
#
# If this is an enriched dataset, force_coefficients.npy will
# be stored inside the dataset directory.
#
# Otherwise use the original 80-case file.
# ============================================================

local_force_file = os.path.join(
    args.data_dir,
    "force_coefficients.npy",
)


if os.path.isfile(local_force_file):

    force_coefficients = np.load(
        local_force_file
    ).astype(np.float32)

else:

    force_coefficients = np.load(
        os.path.join(
            "data",
            "hybrid_force_pca",
            "{}_force_coefficients.npy".format(
                args.segment
            ),
        )
    ).astype(np.float32)


# ============================================================
# INTERFACES
# ============================================================

if args.segment == "left":

    interface_1 = "m6"
    interface_2 = "m2"

else:

    interface_1 = "p2"
    interface_2 = "p6"


force_pca_1 = np.load(
    os.path.join(
        "data",
        "hybrid_force_pca",
        "force_pca_{}.npz".format(
            interface_1
        ),
    )
)


force_pca_2 = np.load(
    os.path.join(
        "data",
        "hybrid_force_pca",
        "force_pca_{}.npz".format(
            interface_2
        ),
    )
)


number_force_modes_1 = force_pca_1[
    "basis"
].shape[1]


number_force_modes_2 = force_pca_2[
    "basis"
].shape[1]


# ============================================================
# TRAIN / VALIDATION / TEST SPLITS
# ============================================================

if "Split" in split_dataframe.columns:

    split_column = "Split"

elif "split" in split_dataframe.columns:

    split_column = "split"

else:

    raise RuntimeError(
        "Could not find Split column."
    )


split_values = (
    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()
)


def get_indices(names):

    mask = split_values.isin(names)

    if "Index" in split_dataframe.columns:

        return split_dataframe.loc[
            mask,
            "Index",
        ].to_numpy(dtype=int)

    return np.where(
        mask.to_numpy()
    )[0]


train_indices = get_indices(
    ["train"]
)

validation_indices = get_indices(
    ["validation", "val"]
)

test_indices = get_indices(
    ["test"]
)


print("")
print("Train:", len(train_indices))
print("Validation:", len(validation_indices))
print("Test:", len(test_indices))


# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

branch_mean = branch[
    train_indices
].mean(axis=0)


branch_std = branch[
    train_indices
].std(axis=0)


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
).mean(axis=0)


U_std = U_train.reshape(
    -1,
    3,
).std(axis=0)


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


force_coefficient_mean = force_coefficients[
    train_indices
].mean(axis=0)


force_coefficient_std = force_coefficients[
    train_indices
].std(axis=0)


force_coefficient_std = np.maximum(
    force_coefficient_std,
    1.0e-8,
)


force_coefficient_normalized = (
    force_coefficients
    -
    force_coefficient_mean
) / force_coefficient_std


g_mean = g[
    train_indices
].mean(axis=0)


g_std = g[
    train_indices
].std(axis=0)


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


    def __len__(self):

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
                force_coefficient_normalized[
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


force_coefficient_mean_tensor = torch.tensor(
    force_coefficient_mean,
    dtype=torch.float32,
    device=DEVICE,
)


force_coefficient_std_tensor = torch.tensor(
    force_coefficient_std,
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


# ============================================================
# FORCE COEFFICIENTS -> GENERALIZED FORCE
# ============================================================

def coefficients_to_g(
    coefficient_prediction_normalized,
):

    coefficients = (
        coefficient_prediction_normalized
        *
        force_coefficient_std_tensor
        +
        force_coefficient_mean_tensor
    )


    coefficients_1 = coefficients[
        :,
        :number_force_modes_1
    ]


    coefficients_2 = coefficients[
        :,
        number_force_modes_1:
        number_force_modes_1
        +
        number_force_modes_2
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


    return (
        g_physical
        -
        g_mean_tensor
    ) / g_std_tensor


# ============================================================
# TEST ONE MODEL
# ============================================================

def evaluate_test(
    model,
):

    model.eval()

    predictions_U = []

    predictions_coeff = []

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
                coeff_prediction_normalized,
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
                    3
                )
                +
                U_mean.reshape(
                    1,
                    3
                )
            )


            coeff_prediction = (
                coeff_prediction_normalized[
                    0
                ]
                .cpu()
                .numpy()
                *
                force_coefficient_std
                +
                force_coefficient_mean
            )


            predictions_U.append(
                U_prediction
            )

            predictions_coeff.append(
                coeff_prediction
            )


    predictions_U = np.asarray(
        predictions_U
    )


    predictions_coeff = np.asarray(
        predictions_coeff
    )


    results = {}


    for component_index, component_name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        errors = []

        for local_index, case_index in enumerate(
            test_indices
        ):

            prediction = predictions_U[
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


        results[
            "{}_percent".format(
                component_name
            )
        ] = float(
            np.mean(
                errors
            )
        )


    return (
        results,
        predictions_U,
        predictions_coeff,
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
    606,
    707,
    808,
]


mse = nn.MSELoss()


member_rows = []

ensemble_U_predictions = []


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
        "TRAINING MEMBER",
        member_index + 1,
        "SEED",
        seed,
    )

    print(
        "================================================"
    )


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
            branch.shape[1],
        number_force_coefficients=
            force_coefficients.shape[1],
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ).to(DEVICE)


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


    best_score = np.inf

    best_epoch = -1

    no_improvement = 0

    history = []


    checkpoint_file = os.path.join(
        member_directory,
        "best_v3_ensemble.pt",
    )


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        train_total = 0.0

        train_count = 0


        for (
            branch_batch,
            U_target,
            coeff_target,
            g_target,
        ) in train_loader:

            branch_batch = branch_batch.to(
                DEVICE
            )

            U_target = U_target.to(
                DEVICE
            )

            coeff_target = coeff_target.to(
                DEVICE
            )

            g_target = g_target.to(
                DEVICE
            )


            optimizer.zero_grad()


            (
                U_prediction,
                coeff_prediction,
            ) = model(
                branch_batch,
                coordinate_tensor,
            )


            g_prediction = coefficients_to_g(
                coeff_prediction
            )


            displacement_loss = mse(
                U_prediction,
                U_target,
            )


            coefficient_loss = mse(
                coeff_prediction,
                coeff_target,
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


            train_total += (
                total_loss.item()
                *
                batch_size
            )


            train_count += batch_size


        train_total /= train_count


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_total = 0.0

        validation_count = 0


        with torch.no_grad():

            for (
                branch_batch,
                U_target,
                coeff_target,
                g_target,
            ) in validation_loader:

                branch_batch = branch_batch.to(
                    DEVICE
                )

                U_target = U_target.to(
                    DEVICE
                )

                coeff_target = coeff_target.to(
                    DEVICE
                )

                g_target = g_target.to(
                    DEVICE
                )


                (
                    U_prediction,
                    coeff_prediction,
                ) = model(
                    branch_batch,
                    coordinate_tensor,
                )


                g_prediction = coefficients_to_g(
                    coeff_prediction
                )


                displacement_loss = mse(
                    U_prediction,
                    U_target,
                )


                coefficient_loss = mse(
                    coeff_prediction,
                    coeff_target,
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


                validation_total += (
                    total_loss.item()
                    *
                    batch_size
                )


                validation_count += batch_size


        validation_total /= (
            validation_count
        )


        scheduler.step(
            validation_total
        )


        history.append(
            {
                "Epoch":
                    epoch,
                "TrainLoss":
                    train_total,
                "ValidationLoss":
                    validation_total,
            }
        )


        if validation_total < best_score:

            best_score = validation_total

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
                        branch.shape[1],

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
                        force_coefficient_mean,

                    "force_coeff_std":
                        force_coefficient_std,

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
                "Val {:.4e}".format(
                    epoch,
                    train_total,
                    validation_total,
                )
            )


        if no_improvement >= args.patience:

            print(
                "Early stopping:",
                epoch
            )

            break


    # ========================================================
    # RELOAD BEST CHECKPOINT
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
        test_results,
        test_U_predictions,
        test_coeff_predictions,
    ) = evaluate_test(
        model
    )


    ensemble_U_predictions.append(
        test_U_predictions
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
                test_results[
                    "U1_percent"
                ],

            "U2_percent":
                test_results[
                    "U2_percent"
                ],

            "U3_percent":
                test_results[
                    "U3_percent"
                ],
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
        "Test U2:",
        test_results[
            "U2_percent"
        ],
        "%"
    )


# ============================================================
# ENSEMBLE MEAN TEST RESULT
# ============================================================

ensemble_U_predictions = np.asarray(
    ensemble_U_predictions
)


ensemble_mean_U = np.mean(
    ensemble_U_predictions,
    axis=0,
)


ensemble_results = {}


for component_index, component_name in enumerate(
    [
        "U1",
        "U2",
        "U3",
    ]
):

    errors = []

    for local_index, case_index in enumerate(
        test_indices
    ):

        prediction = ensemble_mean_U[
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


    ensemble_results[
        "{}_percent".format(
            component_name
        )
    ] = float(
        np.mean(
            errors
        )
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


ensemble_summary = {
    "Segment":
        args.segment,

    "Members":
        args.members,

    "TrainCases":
        len(train_indices),

    "ValidationCases":
        len(validation_indices),

    "TestCases":
        len(test_indices),

    "EnsembleMean_U1_percent":
        ensemble_results[
            "U1_percent"
        ],

    "EnsembleMean_U2_percent":
        ensemble_results[
            "U2_percent"
        ],

    "EnsembleMean_U3_percent":
        ensemble_results[
            "U3_percent"
        ],
}


with open(
    os.path.join(
        args.output_root,
        "ensemble_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        ensemble_summary,
        file_object,
        indent=4,
    )


print("")
print(
    "================================================"
)

print(
    "ENSEMBLE COMPLETE"
)

print(
    "================================================"
)


print(
    json.dumps(
        ensemble_summary,
        indent=4,
    )
)