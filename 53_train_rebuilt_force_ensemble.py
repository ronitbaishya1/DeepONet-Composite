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
    WeightedRandomSampler,
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
    "--enrichment_fraction",
    type=float,
    default=-1.0,
    help=(
        "-1 = natural sampling. "
        "0.30 = enrichment contributes about 30%% "
        "of samples per epoch."
    ),
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


split_df = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# NEW FORCE PCA
# ============================================================

PCA_DIR = os.path.join(
    "data",
    "hybrid_force_pca_enriched",
    "final",
)


force_coefficients = np.load(
    os.path.join(
        PCA_DIR,
        "{}_force_coefficients.npy".format(
            args.segment
        ),
    )
).astype(
    np.float32
)


if args.segment == "left":

    interface_1 = "m6"
    interface_2 = "m2"

else:

    interface_1 = "p2"
    interface_2 = "p6"


pca_1 = np.load(
    os.path.join(
        PCA_DIR,
        "force_pca_{}.npz".format(
            interface_1
        ),
    )
)


pca_2 = np.load(
    os.path.join(
        PCA_DIR,
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


print("")
print(
    "Force modes:",
    number_modes_1,
    "+",
    number_modes_2,
)


# ============================================================
# SPLIT
# ============================================================

if "Split" in split_df.columns:
    split_column = "Split"
else:
    split_column = "split"


split_values = (
    split_df[
        split_column
    ]
    .astype(str)
    .str.lower()
)


def get_indices(names):

    mask = split_values.isin(
        names
    )

    if "Index" in split_df.columns:

        return split_df.loc[
            mask,
            "Index",
        ].to_numpy(
            dtype=int
        )

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


print(
    "Train:",
    len(train_indices)
)

print(
    "Validation:",
    len(validation_indices)
)

print(
    "Test held out:",
    len(test_indices)
)


# ============================================================
# IDENTIFY ORIGINAL VS ENRICHMENT TRAINING CASES
# ============================================================

case_ids = (
    split_df[
        "CaseID"
    ]
    .astype(str)
    .to_numpy()
)


train_case_ids = case_ids[
    train_indices
]


train_is_enrichment = np.asarray(
    [
        case_id.startswith(
            "AE_"
        )
        for case_id in train_case_ids
    ],
    dtype=bool,
)


number_original_train = int(
    np.sum(
        ~train_is_enrichment
    )
)


number_enrichment_train = int(
    np.sum(
        train_is_enrichment
    )
)


print(
    "Original training cases:",
    number_original_train
)


print(
    "Enrichment training cases:",
    number_enrichment_train
)


# ============================================================
# NORMALIZATION — TRAIN ONLY
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


train_dataset = LocalDataset(
    train_indices
)


validation_dataset = LocalDataset(
    validation_indices
)


# ============================================================
# OPTIONAL WEIGHTED SAMPLING
# ============================================================

def build_train_loader():

    if args.enrichment_fraction < 0:

        return DataLoader(
            train_dataset,
            batch_size=
                args.batch_size,
            shuffle=True,
        )


    fraction = args.enrichment_fraction


    if not (
        0.0
        <
        fraction
        <
        1.0
    ):

        raise ValueError(
            "enrichment_fraction must be -1 or between 0 and 1."
        )


    weights = np.zeros(
        len(
            train_indices
        ),
        dtype=np.float64,
    )


    original_weight = (
        1.0
        -
        fraction
    ) / max(
        number_original_train,
        1,
    )


    enrichment_weight = (
        fraction
        /
        max(
            number_enrichment_train,
            1,
        )
    )


    weights[
        ~train_is_enrichment
    ] = original_weight


    weights[
        train_is_enrichment
    ] = enrichment_weight


    sampler = WeightedRandomSampler(
        weights=
            torch.tensor(
                weights,
                dtype=torch.double,
            ),

        num_samples=
            len(
                train_dataset
            ),

        replacement=True,
    )


    return DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        sampler=
            sampler,
    )


validation_loader = DataLoader(
    validation_dataset,
    batch_size=
        args.batch_size,
    shuffle=False,
)


# ============================================================
# TENSORS
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


    c1 = physical_coefficients[
        :,
        :number_modes_1
    ]


    c2 = physical_coefficients[
        :,
        number_modes_1:
        number_modes_1
        +
        number_modes_2
    ]


    g1 = (
        pca_g_mean_1.reshape(
            1,
            -1,
        )
        +
        c1
        @
        pca_g_matrix_1.T
    )


    g2 = (
        pca_g_mean_2.reshape(
            1,
            -1,
        )
        +
        c2
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
# VALIDATION METRICS
# ============================================================

def evaluate_validation(
    model,
):

    model.eval()


    U_predictions = []

    g_predictions = []

    coefficient_predictions = []


    with torch.no_grad():

        for case_index in validation_indices:

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
                *
                U_std_tensor.reshape(
                    1,
                    3,
                )
                +
                U_mean_tensor.reshape(
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
                U_prediction.cpu().numpy()
            )


            g_predictions.append(
                g_prediction[
                    0
                ].cpu().numpy()
            )


            coefficient_predictions.append(
                coefficient_prediction[
                    0
                ].cpu().numpy()
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


def mean_component_error(
    predictions,
    component,
):

    errors = []


    for local_index, case_index in enumerate(
        validation_indices
    ):

        truth = U[
            case_index,
            :,
            component
        ]


        prediction = predictions[
            local_index,
            :,
            component
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
    prediction,
    truth_array,
):

    errors = []


    for local_index, case_index in enumerate(
        validation_indices
    ):

        error = (
            np.linalg.norm(
                prediction[
                    local_index
                ]
                -
                truth_array[
                    case_index
                ]
            )
            /
            (
                np.linalg.norm(
                    truth_array[
                        case_index
                    ]
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

ensemble_validation_U = []

ensemble_validation_g = []

ensemble_validation_coefficients = []


for member_index in range(
    args.members
):

    seed = SEEDS[
        member_index
    ]


    print("")
    print(
        "========================================"
    )

    print(
        "MEMBER",
        member_index + 1,
        "SEED",
        seed,
    )

    print(
        "========================================"
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


    train_loader = build_train_loader()


    model = HybridBulkOperatorV3(
        branch_dim=
            branch.shape[
                1
            ],

        number_force_coefficients=
            force_coefficients.shape[
                1
            ],

        hidden_dim=128,

        latent_dim=128,

        depth=4,
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


    member_dir = os.path.join(
        args.output_root,
        "member_{:02d}".format(
            member_index + 1
        ),
    )


    os.makedirs(
        member_dir,
        exist_ok=True,
    )


    checkpoint_file = os.path.join(
        member_dir,
        "best_enriched_forcepca.pt",
    )


    best_validation_loss = np.inf

    best_epoch = -1

    no_improvement = 0

    history = []


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


            loss_U = mse(
                U_prediction,
                U_target,
            )


            loss_coeff = mse(
                coefficient_prediction,
                coefficient_target,
            )


            loss_g = mse(
                g_prediction,
                g_target,
            )


            loss = (
                loss_U
                +
                0.25
                *
                loss_coeff
                +
                loss_g
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            optimizer.step()


            batch_size = branch_batch.shape[
                0
            ]


            train_sum += (
                loss.item()
                *
                batch_size
            )


            train_count += batch_size


        train_loss = (
            train_sum
            /
            train_count
        )


        # ----------------------------------------------------
        # Validation loss for early stopping
        # ----------------------------------------------------

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


                loss = (
                    mse(
                        U_prediction,
                        U_target,
                    )
                    +
                    0.25
                    *
                    mse(
                        coefficient_prediction,
                        coefficient_target,
                    )
                    +
                    mse(
                        g_prediction,
                        g_target,
                    )
                )


                batch_size = branch_batch.shape[
                    0
                ]


                validation_sum += (
                    loss.item()
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


        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

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

                    "force_modes_per_interface":
                        number_modes_1,

                    "hidden_dim":
                        128,

                    "latent_dim":
                        128,

                    "depth":
                        4,

                    "best_epoch":
                        epoch,

                    "enrichment_fraction":
                        args.enrichment_fraction,

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
                "Epoch {:4d} | Train {:.4e} | Val {:.4e}"
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


    # --------------------------------------------------------
    # Load best member
    # --------------------------------------------------------

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
        validation_U,
        validation_g,
        validation_coefficients,
    ) = evaluate_validation(
        model
    )


    U1_error = mean_component_error(
        validation_U,
        0,
    )

    U2_error = mean_component_error(
        validation_U,
        1,
    )

    U3_error = mean_component_error(
        validation_U,
        2,
    )


    g_error = mean_vector_error(
        validation_g,
        g,
    )


    coefficient_error = mean_vector_error(
        validation_coefficients,
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

            "Validation_U1_percent":
                U1_error,

            "Validation_U2_percent":
                U2_error,

            "Validation_U3_percent":
                U3_error,

            "Validation_GeneralizedForce_percent":
                g_error,

            "Validation_ForceCoefficient_percent":
                coefficient_error,
        }
    )


    ensemble_validation_U.append(
        validation_U
    )

    ensemble_validation_g.append(
        validation_g
    )

    ensemble_validation_coefficients.append(
        validation_coefficients
    )


    pd.DataFrame(
        history
    ).to_csv(
        os.path.join(
            member_dir,
            "training_history.csv",
        ),
        index=False,
    )


# ============================================================
# ENSEMBLE VALIDATION RESULT
# ============================================================

ensemble_validation_U = np.mean(
    np.asarray(
        ensemble_validation_U
    ),
    axis=0,
)


ensemble_validation_g = np.mean(
    np.asarray(
        ensemble_validation_g
    ),
    axis=0,
)


ensemble_validation_coefficients = np.mean(
    np.asarray(
        ensemble_validation_coefficients
    ),
    axis=0,
)


ensemble_U1 = mean_component_error(
    ensemble_validation_U,
    0,
)


ensemble_U2 = mean_component_error(
    ensemble_validation_U,
    1,
)


ensemble_U3 = mean_component_error(
    ensemble_validation_U,
    2,
)


ensemble_g_error = mean_vector_error(
    ensemble_validation_g,
    g,
)


ensemble_coefficient_error = mean_vector_error(
    ensemble_validation_coefficients,
    force_coefficients,
)


# Main selection score:
#
# U2 matters most.
# Generalized force is also important for coupling.
selection_score = (
    ensemble_U2
    +
    0.25
    *
    ensemble_g_error
)


summary = {

    "Segment":
        args.segment,

    "EnrichmentFraction":
        args.enrichment_fraction,

    "Members":
        args.members,

    "TrainCases":
        len(train_indices),

    "ValidationCases":
        len(validation_indices),

    "TestCasesNotUsed":
        len(test_indices),

    "ForceModesPerInterface":
        int(
            number_modes_1
        ),

    "Validation_Ensemble_U1_percent":
        ensemble_U1,

    "Validation_Ensemble_U2_percent":
        ensemble_U2,

    "Validation_Ensemble_U3_percent":
        ensemble_U3,

    "Validation_Ensemble_GeneralizedForce_percent":
        ensemble_g_error,

    "Validation_Ensemble_ForceCoefficient_percent":
        ensemble_coefficient_error,

    "ValidationSelectionScore":
        selection_score,
}


pd.DataFrame(
    member_rows
).to_csv(
    os.path.join(
        args.output_root,
        "validation_member_summary.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        args.output_root,
        "ensemble_validation_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        summary,
        file_object,
        indent=4,
    )


print("")
print(
    "========================================"
)

print(
    "STEP 53 TRAINING COMPLETE"
)

print(
    "========================================"
)


print(
    json.dumps(
        summary,
        indent=4,
    )
)