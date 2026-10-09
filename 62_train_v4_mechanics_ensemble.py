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

from src.hybrid_bulk_operator_v4 import (
    HybridBulkOperatorV4
)

from src.hard_interface_compatibility import (
    build_hard_boundary_context,
    apply_hard_compatibility,
)

from src.orthotropic_mechanics import (
    orthotropic_stiffness_matrix,
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
    "--initial_v4_root",
    default=None,
)


parser.add_argument(
    "--members",
    type=int,
    default=5,
)


parser.add_argument(
    "--epochs",
    type=int,
    default=1200,
)


parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=250,
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)


parser.add_argument(
    "--learning_rate",
    type=float,
    default=5.0e-5,
)


parser.add_argument(
    "--patience",
    type=int,
    default=250,
)


parser.add_argument(
    "--lambda_strain",
    type=float,
    default=0.50,
)


parser.add_argument(
    "--lambda_stress",
    type=float,
    default=0.50,
)


parser.add_argument(
    "--lambda_force_coeff",
    type=float,
    default=0.25,
)


parser.add_argument(
    "--lambda_g",
    type=float,
    default=1.0,
)


parser.add_argument(
    "--lambda_kin",
    type=float,
    default=0.0,
)


parser.add_argument(
    "--lambda_const",
    type=float,
    default=0.0,
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


ip_coordinates = np.load(
    os.path.join(
        args.data_dir,
        "ip_coordinates.npy",
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


LE = np.load(
    os.path.join(
        args.data_dir,
        "LE.npy",
    )
).astype(
    np.float32
)


S = np.load(
    os.path.join(
        args.data_dir,
        "S.npy",
    )
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


connectivity = np.load(
    os.path.join(
        args.data_dir,
        "element_node_indices.npy",
    )
).astype(
    np.int64
)


B_matrices = np.load(
    os.path.join(
        args.data_dir,
        "B_center.npy",
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
# ORIGINAL STEP-40 FORCE PCA
# ============================================================

force_coefficients = np.load(
    os.path.join(
        "data",
        "hybrid_force_pca",
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


number_force_modes_1 = (
    force_pca_1[
        "basis"
    ].shape[1]
)


number_force_modes_2 = (
    force_pca_2[
        "basis"
    ].shape[1]
)


# ============================================================
# DATA SPLIT
# ============================================================

split_values = (
    split_dataframe[
        "Split"
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

    return split_dataframe.loc[
        mask,
        "Index",
    ].to_numpy(
        dtype=int
    )


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
    "Test held out:",
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


ip_coordinate_normalized = (
    2.0
    *
    (
        ip_coordinates
        -
        coordinate_min
    )
    /
    coordinate_range
    -
    1.0
)


U_mean = U[
    train_indices
].reshape(
    -1,
    3,
).mean(
    axis=0
)


U_std = U[
    train_indices
].reshape(
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


LE_mean = LE[
    train_indices
].reshape(
    -1,
    6,
).mean(
    axis=0
)


LE_std = LE[
    train_indices
].reshape(
    -1,
    6,
).std(
    axis=0
)


LE_std = np.maximum(
    LE_std,
    1.0e-10,
)


LE_normalized = (
    LE
    -
    LE_mean.reshape(
        1,
        1,
        6,
    )
) / LE_std.reshape(
    1,
    1,
    6,
)


S_mean = S[
    train_indices
].reshape(
    -1,
    6,
).mean(
    axis=0
)


S_std = S[
    train_indices
].reshape(
    -1,
    6,
).std(
    axis=0
)


S_std = np.maximum(
    S_std,
    1.0e-8,
)


S_normalized = (
    S
    -
    S_mean.reshape(
        1,
        1,
        6,
    )
) / S_std.reshape(
    1,
    1,
    6,
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
# HARD BOUNDARY CONTEXT
# ============================================================

hard_context = (
    build_hard_boundary_context(

        coordinates=

            coordinates,

        left_interface_file=

            os.path.join(
                "data",
                "hybrid_interfaces_trainonly",
                "interface_{}.npz".format(
                    interface_1
                ),
            ),

        right_interface_file=

            os.path.join(
                "data",
                "hybrid_interfaces_trainonly",
                "interface_{}.npz".format(
                    interface_2
                ),
            ),
    )
)


# ============================================================
# DATASET
# ============================================================

class LocalDataset(
    Dataset
):

    def __init__(
        self,
        indices,
    ):

        self.indices = np.asarray(
            indices,
            dtype=int,
        )


    def __len__(
        self
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
                branch[
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
                LE_normalized[
                    index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                S_normalized[
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
# FIXED TENSORS
# ============================================================

node_coordinate_tensor = torch.tensor(
    coordinate_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


ip_coordinate_tensor = torch.tensor(
    ip_coordinate_normalized,
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


LE_mean_tensor = torch.tensor(
    LE_mean,
    dtype=torch.float32,
    device=DEVICE,
)


LE_std_tensor = torch.tensor(
    LE_std,
    dtype=torch.float32,
    device=DEVICE,
)


S_mean_tensor = torch.tensor(
    S_mean,
    dtype=torch.float32,
    device=DEVICE,
)


S_std_tensor = torch.tensor(
    S_std,
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


connectivity_tensor = torch.tensor(
    connectivity,
    dtype=torch.long,
    device=DEVICE,
)


B_tensor = torch.tensor(
    B_matrices,
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


mse = nn.MSELoss()


# ============================================================
# FORCE COEFFICIENTS -> GENERALIZED FORCE
# ============================================================

def coefficients_to_g_normalized(
    coefficient_prediction,
):

    coefficients = (
        coefficient_prediction
        *
        force_std_tensor
        +
        force_mean_tensor
    )


    c1 = coefficients[
        :,
        :number_force_modes_1
    ]


    c2 = coefficients[
        :,
        number_force_modes_1:
        number_force_modes_1
        +
        number_force_modes_2
    ]


    g1 = (
        g_mean_1.reshape(
            1,
            -1,
        )
        +
        c1
        @
        g_matrix_1.T
    )


    g2 = (
        g_mean_2.reshape(
            1,
            -1,
        )
        +
        c2
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
# HARD-CORRECT U
# ============================================================

def hard_correct_displacement(
    raw_U_normalized,
    branch_physical,
):

    raw_U_physical = (
        raw_U_normalized
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


    corrected_physical = (
        apply_hard_compatibility(
            raw_displacement=
                raw_U_physical,
            branch_physical=
                branch_physical,
            context=
                hard_context,
        )
    )


    corrected_normalized = (
        corrected_physical
        -
        U_mean_tensor.reshape(
            1,
            1,
            3,
        )
    ) / U_std_tensor.reshape(
        1,
        1,
        3,
    )


    return (
        corrected_normalized,
        corrected_physical,
    )


# ============================================================
# KINEMATIC CONSISTENCY
# ============================================================

def calculate_strain_from_displacement(
    displacement_physical,
):

    element_displacement = (
        displacement_physical[
            :,
            connectivity_tensor,
            :
        ]
    )


    element_displacement = (
        element_displacement.reshape(
            displacement_physical.shape[0],
            connectivity_tensor.shape[0],
            24,
        )
    )


    strain = torch.einsum(
        "eij,bej->bei",
        B_tensor,
        element_displacement,
    )


    return strain


# ============================================================
# CONSTITUTIVE CONSISTENCY
# ============================================================

def calculate_constitutive_stress(
    strain_physical,
    branch_physical,
):

    matrices = []


    for batch_index in range(
        branch_physical.shape[0]
    ):

        matrix = (
            orthotropic_stiffness_matrix(
                branch_physical[
                    batch_index,
                    :3
                ]
            )
        )

        matrices.append(
            matrix
        )


    stiffness = torch.stack(
        matrices,
        dim=0,
    )


    stress = torch.einsum(
        "bei,bji->bej",
        strain_physical,
        stiffness,
    )


    return stress


# ============================================================
# TOTAL LOSS
# ============================================================

def calculate_loss(
    model,
    branch_normalized_batch,
    branch_physical_batch,
    U_target,
    LE_target,
    S_target,
    coeff_target,
    g_target,
):

    (
        raw_U_prediction,
        LE_prediction,
        S_prediction,
        coeff_prediction,
    ) = model(
        branch_normalized_batch,
        node_coordinate_tensor,
        ip_coordinate_tensor,
    )


    (
        U_prediction,
        U_physical,
    ) = hard_correct_displacement(
        raw_U_prediction,
        branch_physical_batch,
    )


    g_prediction = (
        coefficients_to_g_normalized(
            coeff_prediction
        )
    )


    loss_U = mse(
        U_prediction,
        U_target,
    )


    loss_LE = mse(
        LE_prediction,
        LE_target,
    )


    loss_S = mse(
        S_prediction,
        S_target,
    )


    loss_coeff = mse(
        coeff_prediction,
        coeff_target,
    )


    loss_g = mse(
        g_prediction,
        g_target,
    )


    loss_kin = torch.zeros(
        (),
        device=DEVICE,
    )


    loss_const = torch.zeros(
        (),
        device=DEVICE,
    )


    if args.lambda_kin > 0.0:

        LE_physical = (
            LE_prediction
            *
            LE_std_tensor.reshape(
                1,
                1,
                6,
            )
            +
            LE_mean_tensor.reshape(
                1,
                1,
                6,
            )
        )


        LE_from_U = (
            calculate_strain_from_displacement(
                U_physical
            )
        )


        difference = (
            LE_from_U
            -
            LE_physical
        ) / LE_std_tensor.reshape(
            1,
            1,
            6,
        )


        loss_kin = torch.mean(
            difference ** 2
        )


    if args.lambda_const > 0.0:

        LE_physical = (
            LE_prediction
            *
            LE_std_tensor.reshape(
                1,
                1,
                6,
            )
            +
            LE_mean_tensor.reshape(
                1,
                1,
                6,
            )
        )


        S_physical = (
            S_prediction
            *
            S_std_tensor.reshape(
                1,
                1,
                6,
            )
            +
            S_mean_tensor.reshape(
                1,
                1,
                6,
            )
        )


        constitutive_stress = (
            calculate_constitutive_stress(
                LE_physical,
                branch_physical_batch,
            )
        )


        difference = (
            constitutive_stress
            -
            S_physical
        ) / S_std_tensor.reshape(
            1,
            1,
            6,
        )


        loss_const = torch.mean(
            difference ** 2
        )


    total = (
        loss_U
        +
        args.lambda_strain
        *
        loss_LE
        +
        args.lambda_stress
        *
        loss_S
        +
        args.lambda_force_coeff
        *
        loss_coeff
        +
        args.lambda_g
        *
        loss_g
        +
        args.lambda_kin
        *
        loss_kin
        +
        args.lambda_const
        *
        loss_const
    )


    return (
        total,
        loss_U,
        loss_LE,
        loss_S,
        loss_g,
        loss_kin,
        loss_const,
    )


# ============================================================
# MEMBER TRAINING
# ============================================================

SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


member_validation_predictions = []


for member_number in range(
    args.members
):

    seed = SEEDS[
        member_number
    ]


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    print("")
    print("==========================================")
    print(
        "MEMBER",
        member_number + 1
    )
    print("==========================================")


    model = HybridBulkOperatorV4(
        branch_dim=
            branch.shape[1],
        number_force_coefficients=
            force_coefficients.shape[1],
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    if args.initial_v4_root is not None:

        initial_file = os.path.join(
            args.initial_v4_root,
            "member_{:02d}".format(
                member_number + 1
            ),
            "best_v4.pt",
        )


        checkpoint = torch.load(
            initial_file,
            map_location=DEVICE,
            weights_only=False,
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        print(
            "Initialized from V4:",
            initial_file
        )


    else:

        step40_file = os.path.join(
            "results",
            "adaptive_ensemble",
            args.segment,
            "member_{:02d}".format(
                member_number + 1
            ),
            "best_v3_ensemble.pt",
        )


        checkpoint = torch.load(
            step40_file,
            map_location=DEVICE,
            weights_only=False,
        )


        load_result = model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=False,
        )


        print(
            "Initialized displacement/force from Step 40."
        )


        print(
            "Missing keys:",
            len(
                load_result.missing_keys
            )
        )


    # --------------------------------------------------------
    # OPTIONAL MECHANICS-ONLY WARMUP
    # --------------------------------------------------------

    if args.warmup_epochs > 0:

        for name, parameter in (
            model.named_parameters()
        ):

            parameter.requires_grad = (
                name.startswith(
                    "strain_operator"
                )
                or
                name.startswith(
                    "stress_operator"
                )
            )


        warmup_parameters = [
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ]


        warmup_optimizer = (
            torch.optim.Adam(
                warmup_parameters,
                lr=1.0e-4,
            )
        )


        for epoch in range(
            1,
            args.warmup_epochs + 1,
        ):

            model.train()


            for batch in train_loader:

                (
                    branch_n,
                    branch_p,
                    U_target,
                    LE_target,
                    S_target,
                    coeff_target,
                    g_target,
                ) = [
                    item.to(
                        DEVICE
                    )
                    for item in batch
                ]


                warmup_optimizer.zero_grad()


                (
                    _,
                    _,
                    LE_prediction,
                    S_prediction,
                    _,
                ) = (
                    None,
                    None,
                    *model(
                        branch_n,
                        node_coordinate_tensor,
                        ip_coordinate_tensor,
                    )[1:3],
                    None,
                )


                warmup_loss = (
                    mse(
                        LE_prediction,
                        LE_target,
                    )
                    +
                    mse(
                        S_prediction,
                        S_target,
                    )
                )


                warmup_loss.backward()


                warmup_optimizer.step()


            if (
                epoch == 1
                or
                epoch % 50 == 0
            ):

                print(
                    "Warmup epoch",
                    epoch,
                    "loss",
                    float(
                        warmup_loss.item()
                    )
                )


    # --------------------------------------------------------
    # UNFREEZE EVERYTHING
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=
            args.learning_rate,
        weight_decay=1.0e-6,
    )


    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=80,
            min_lr=1.0e-6,
        )
    )


    best_validation = np.inf
    best_epoch = -1
    no_improvement = 0


    member_directory = os.path.join(
        args.output_root,
        "member_{:02d}".format(
            member_number + 1
        ),
    )


    os.makedirs(
        member_directory,
        exist_ok=True,
    )


    checkpoint_file = os.path.join(
        member_directory,
        "best_v4.pt",
    )


    history = []


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_sum = 0.0
        train_count = 0


        for batch in train_loader:

            (
                branch_n,
                branch_p,
                U_target,
                LE_target,
                S_target,
                coeff_target,
                g_target,
            ) = [
                item.to(
                    DEVICE
                )
                for item in batch
            ]


            optimizer.zero_grad()


            losses = calculate_loss(
                model,
                branch_n,
                branch_p,
                U_target,
                LE_target,
                S_target,
                coeff_target,
                g_target,
            )


            total_loss = losses[0]


            total_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )


            optimizer.step()


            batch_size = (
                branch_n.shape[0]
            )


            train_sum += (
                float(
                    total_loss.item()
                )
                *
                batch_size
            )


            train_count += (
                batch_size
            )


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

            for batch in validation_loader:

                (
                    branch_n,
                    branch_p,
                    U_target,
                    LE_target,
                    S_target,
                    coeff_target,
                    g_target,
                ) = [
                    item.to(
                        DEVICE
                    )
                    for item in batch
                ]


                losses = calculate_loss(
                    model,
                    branch_n,
                    branch_p,
                    U_target,
                    LE_target,
                    S_target,
                    coeff_target,
                    g_target,
                )


                total_loss = losses[0]


                batch_size = (
                    branch_n.shape[0]
                )


                validation_sum += (
                    float(
                        total_loss.item()
                    )
                    *
                    batch_size
                )


                validation_count += (
                    batch_size
                )


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

            best_validation = (
                validation_loss
            )

            best_epoch = epoch

            no_improvement = 0


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "segment":
                        args.segment,

                    "branch_dim":
                        branch.shape[1],

                    "number_force_coefficients":
                        force_coefficients.shape[1],

                    "hidden_dim":
                        128,

                    "latent_dim":
                        128,

                    "depth":
                        4,

                    "best_epoch":
                        best_epoch,

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

                    "LE_mean":
                        LE_mean,

                    "LE_std":
                        LE_std,

                    "S_mean":
                        S_mean,

                    "S_std":
                        S_std,

                    "force_coeff_mean":
                        force_mean,

                    "force_coeff_std":
                        force_std,

                    "g_mean":
                        g_mean,

                    "g_std":
                        g_std,

                    "lambda_kin":
                        args.lambda_kin,

                    "lambda_const":
                        args.lambda_const,
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
                "Epoch {:4d} | Train {:.5e} | Val {:.5e}"
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


    pd.DataFrame(
        history
    ).to_csv(
        os.path.join(
            member_directory,
            "training_history.csv",
        ),
        index=False,
    )


# ============================================================
# VALIDATION ENSEMBLE
# ============================================================

def mean_case_relative_l2(
    prediction,
    truth,
):

    errors = []


    for case_index in range(
        prediction.shape[0]
    ):

        numerator = np.linalg.norm(
            prediction[
                case_index
            ]
            -
            truth[
                case_index
            ]
        )


        denominator = (
            np.linalg.norm(
                truth[
                    case_index
                ]
            )
            +
            1.0e-14
        )


        errors.append(
            100.0
            *
            numerator
            /
            denominator
        )


    return float(
        np.mean(
            errors
        )
    )


member_U = []
member_LE = []
member_S = []
member_g = []


for member_number in range(
    args.members
):

    checkpoint_file = os.path.join(
        args.output_root,
        "member_{:02d}".format(
            member_number + 1
        ),
        "best_v4.pt",
    )


    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=False,
    )


    model = HybridBulkOperatorV4(
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
    ).to(
        DEVICE
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    U_predictions = []
    LE_predictions = []
    S_predictions = []
    g_predictions = []


    with torch.no_grad():

        for case_index in (
            validation_indices
        ):

            branch_n = torch.tensor(
                branch_normalized[
                    case_index
                ],
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(
                1,
                -1,
            )


            branch_p = torch.tensor(
                branch[
                    case_index
                ],
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(
                1,
                -1,
            )


            (
                raw_U,
                LE_pred,
                S_pred,
                coeff_pred,
            ) = model(
                branch_n,
                node_coordinate_tensor,
                ip_coordinate_tensor,
            )


            (
                _,
                U_physical,
            ) = hard_correct_displacement(
                raw_U,
                branch_p,
            )


            LE_physical = (
                LE_pred
                *
                LE_std_tensor.reshape(
                    1,
                    1,
                    6,
                )
                +
                LE_mean_tensor.reshape(
                    1,
                    1,
                    6,
                )
            )


            S_physical = (
                S_pred
                *
                S_std_tensor.reshape(
                    1,
                    1,
                    6,
                )
                +
                S_mean_tensor.reshape(
                    1,
                    1,
                    6,
                )
            )


            g_pred_n = (
                coefficients_to_g_normalized(
                    coeff_pred
                )
            )


            g_physical = (
                g_pred_n
                *
                g_std_tensor
                +
                g_mean_tensor
            )


            U_predictions.append(
                U_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            LE_predictions.append(
                LE_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            S_predictions.append(
                S_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            g_predictions.append(
                g_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


    member_U.append(
        np.asarray(
            U_predictions
        )
    )


    member_LE.append(
        np.asarray(
            LE_predictions
        )
    )


    member_S.append(
        np.asarray(
            S_predictions
        )
    )


    member_g.append(
        np.asarray(
            g_predictions
        )
    )


ensemble_U = np.mean(
    np.asarray(
        member_U
    ),
    axis=0,
)


ensemble_LE = np.mean(
    np.asarray(
        member_LE
    ),
    axis=0,
)


ensemble_S = np.mean(
    np.asarray(
        member_S
    ),
    axis=0,
)


ensemble_g = np.mean(
    np.asarray(
        member_g
    ),
    axis=0,
)


validation_U = U[
    validation_indices
]


validation_LE = LE[
    validation_indices
]


validation_S = S[
    validation_indices
]


validation_g = g[
    validation_indices
]


component_names = [
    "11",
    "22",
    "33",
    "12",
    "13",
    "23",
]


summary = {
    "Segment":
        args.segment,

    "Validation_U1_percent":
        mean_case_relative_l2(
            ensemble_U[:, :, 0],
            validation_U[:, :, 0],
        ),

    "Validation_U2_percent":
        mean_case_relative_l2(
            ensemble_U[:, :, 1],
            validation_U[:, :, 1],
        ),

    "Validation_U3_percent":
        mean_case_relative_l2(
            ensemble_U[:, :, 2],
            validation_U[:, :, 2],
        ),

    "Validation_GeneralizedForce_percent":
        mean_case_relative_l2(
            ensemble_g,
            validation_g,
        ),
}


for component_index, name in enumerate(
    component_names
):

    summary[
        "Validation_LE{}_percent".format(
            name
        )
    ] = mean_case_relative_l2(
        ensemble_LE[
            :,
            :,
            component_index
        ],
        validation_LE[
            :,
            :,
            component_index
        ],
    )


    summary[
        "Validation_S{}_percent".format(
            name
        )
    ] = mean_case_relative_l2(
        ensemble_S[
            :,
            :,
            component_index
        ],
        validation_S[
            :,
            :,
            component_index
        ],
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
print("==========================================")
print("V4 ENSEMBLE COMPLETE")
print("==========================================")

print(
    json.dumps(
        summary,
        indent=4,
    )
)