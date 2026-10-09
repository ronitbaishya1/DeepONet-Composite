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

from src.hybrid_bulk_operator_v4_7region import (
    HybridBulkOperatorV4SevenRegion,
)

from src.hard_interface_compatibility_7region import (
    build_hard_boundary_context_7region,
    apply_hard_compatibility_7region,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "outer_left",
        "inner_left",
        "inner_right",
        "outer_right",
    ],
    required=True,
)


parser.add_argument(
    "--data_root",
    default=
        "data/hybrid_bulk_7region",
)


parser.add_argument(
    "--interface_dir",
    default=
        "data/candidate_partition_interfaces_7region",
)


parser.add_argument(
    "--force_pca_dir",
    default=
        "data/hybrid_force_pca_7region",
)


parser.add_argument(
    "--output_root",
    required=True,
)


parser.add_argument(
    "--initial_root",
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
    default=1500,
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
    "--lambda_u",
    type=float,
    default=1.0,
)


parser.add_argument(
    "--lambda_strain",
    type=float,
    default=0.5,
)


parser.add_argument(
    "--lambda_stress",
    type=float,
    default=0.5,
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


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    args.data_root,
    args.segment,
)


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


print("")
print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    os.path.join(
        DATA_DIR,
        "metadata.json",
    ),
    "r",
) as file_object:

    metadata = json.load(
        file_object
    )


left_interface_name = metadata[
    "left_interface"
]


right_interface_name = metadata[
    "right_interface"
]


# ============================================================
# LOAD ARRAYS
# ============================================================

branch = np.load(
    os.path.join(
        DATA_DIR,
        "branch_inputs.npy",
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


connectivity = np.load(
    os.path.join(
        DATA_DIR,
        "element_node_indices.npy",
    )
).astype(
    np.int64
)


B_center = np.load(
    os.path.join(
        DATA_DIR,
        "B_center.npy",
    )
).astype(
    np.float32
)


U = np.stack(
    [
        np.load(
            os.path.join(
                DATA_DIR,
                "U1.npy",
            )
        ),

        np.load(
            os.path.join(
                DATA_DIR,
                "U2.npy",
            )
        ),

        np.load(
            os.path.join(
                DATA_DIR,
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
        DATA_DIR,
        "LE.npy",
    )
).astype(
    np.float32
)


S = np.load(
    os.path.join(
        DATA_DIR,
        "S.npy",
    )
).astype(
    np.float32
)


g = np.load(
    os.path.join(
        DATA_DIR,
        "g_all.npy",
    )
).astype(
    np.float32
)


force_coefficients = np.load(
    os.path.join(
        DATA_DIR,
        "force_coefficients.npy",
    )
).astype(
    np.float32
)


split_dataframe = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)


# ============================================================
# SPLITS
# ============================================================

train_indices = split_dataframe.loc[
    split_dataframe[
        "Split"
    ] == "train",
    "Index",
].to_numpy(
    dtype=int
)


validation_indices = split_dataframe.loc[
    split_dataframe[
        "Split"
    ] == "validation",
    "Index",
].to_numpy(
    dtype=int
)


test_indices = split_dataframe.loc[
    split_dataframe[
        "Split"
    ] == "test",
    "Index",
].to_numpy(
    dtype=int
)


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

def safe_std(
    x,
    axis,
):

    value = np.std(
        x,
        axis=axis,
    )


    return np.maximum(
        value,
        1.0e-8,
    )


branch_mean = branch[
    train_indices
].mean(
    axis=0
)


branch_std = safe_std(
    branch[
        train_indices
    ],
    axis=0,
)


U_mean = U[
    train_indices
].mean(
    axis=(
        0,
        1,
    )
)


U_std = safe_std(
    U[
        train_indices
    ],
    axis=(
        0,
        1,
    ),
)


LE_mean = LE[
    train_indices
].mean(
    axis=(
        0,
        1,
    )
)


LE_std = safe_std(
    LE[
        train_indices
    ],
    axis=(
        0,
        1,
    ),
)


S_mean = S[
    train_indices
].mean(
    axis=(
        0,
        1,
    )
)


S_std = safe_std(
    S[
        train_indices
    ],
    axis=(
        0,
        1,
    ),
)


force_mean = force_coefficients[
    train_indices
].mean(
    axis=0
)


force_std = safe_std(
    force_coefficients[
        train_indices
    ],
    axis=0,
)


g_mean = g[
    train_indices
].mean(
    axis=0
)


g_std = safe_std(
    g[
        train_indices
    ],
    axis=0,
)


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


node_coordinates_normalized = (
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


ip_coordinates_normalized = (
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


# ============================================================
# NORMALIZED TARGET ARRAYS
# ============================================================

branch_n = (
    branch
    -
    branch_mean
) / branch_std


U_n = (
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


LE_n = (
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


S_n = (
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


force_n = (
    force_coefficients
    -
    force_mean
) / force_std


g_n = (
    g
    -
    g_mean
) / g_std


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
        index,
    ):

        case_index = self.indices[
            index
        ]


        return (

            torch.tensor(
                branch_n[
                    case_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                U_n[
                    case_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                LE_n[
                    case_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                S_n[
                    case_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                force_n[
                    case_index
                ],
                dtype=torch.float32,
            ),

            torch.tensor(
                g_n[
                    case_index
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

    shuffle=
        True,
)


validation_loader = DataLoader(

    LocalDataset(
        validation_indices
    ),

    batch_size=
        args.batch_size,

    shuffle=
        False,
)


# ============================================================
# CONSTANT TENSORS
# ============================================================

node_coordinate_tensor = torch.tensor(

    node_coordinates_normalized,

    dtype=torch.float32,

    device=DEVICE,
)


ip_coordinate_tensor = torch.tensor(

    ip_coordinates_normalized,

    dtype=torch.float32,

    device=DEVICE,
)


connectivity_tensor = torch.tensor(

    connectivity,

    dtype=torch.long,

    device=DEVICE,
)


B_tensor = torch.tensor(

    B_center,

    dtype=torch.float32,

    device=DEVICE,
)


branch_mean_tensor = torch.tensor(
    branch_mean,
    dtype=torch.float32,
    device=DEVICE,
)


branch_std_tensor = torch.tensor(
    branch_std,
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


# ============================================================
# HARD COMPATIBILITY
# ============================================================

hard_context = (
    build_hard_boundary_context_7region(

        coordinates=
            coordinates,

        interface_dir=
            args.interface_dir,

        left_interface_name=
            left_interface_name,

        right_interface_name=
            right_interface_name,

        device=
            DEVICE,
    )
)


if hard_context[
    "branch_dim"
] != branch.shape[
    1
]:

    raise RuntimeError(
        "Hard-boundary branch dimension mismatch."
    )


# ============================================================
# FORCE PCA -> GENERALIZED FORCE MAPPING
# ============================================================

force_pca_blocks = []


for interface_name in [
    left_interface_name,
    right_interface_name,
]:

    if interface_name is None:

        continue


    data = np.load(
        os.path.join(
            args.force_pca_dir,
            "force_pca_{}.npz".format(
                interface_name
            ),
        )
    )


    force_pca_blocks.append(
        {
            "interface":
                interface_name,

            "number_modes":
                int(
                    data[
                        "number_modes"
                    ][0]
                ),

            "g_mean":
                torch.tensor(
                    data[
                        "g_mean"
                    ],
                    dtype=torch.float32,
                    device=DEVICE,
                ),

            "g_matrix":
                torch.tensor(
                    data[
                        "g_matrix"
                    ],
                    dtype=torch.float32,
                    device=DEVICE,
                ),
        }
    )


def force_coefficients_to_g(
    coefficients_physical,
):

    outputs = []


    cursor = 0


    for block in force_pca_blocks:

        number_modes = block[
            "number_modes"
        ]


        current = coefficients_physical[
            :,
            cursor:
            cursor
            +
            number_modes
        ]


        current_g = (
            block[
                "g_mean"
            ].reshape(
                1,
                -1
            )
            +
            current
            @
            block[
                "g_matrix"
            ].T
        )


        outputs.append(
            current_g
        )


        cursor += number_modes


    if cursor != coefficients_physical.shape[
        1
    ]:

        raise RuntimeError(
            "Force coefficient dimension mismatch."
        )


    return torch.cat(
        outputs,
        dim=1,
    )


# ============================================================
# ORTHOTROPIC STIFFNESS
#
# Engineering strain vector:
#
# [e11,e22,e33,g12,g13,g23]
# ============================================================

def orthotropic_stiffness_batch(
    E1,
    E2,
    G12,
):

    batch_size = E1.shape[
        0
    ]


    E3 = torch.full_like(
        E1,
        12000.0,
    )


    nu12 = 0.28
    nu13 = 0.28
    nu23 = 0.40


    G13 = torch.full_like(
        E1,
        4500.0,
    )


    G23 = torch.full_like(
        E1,
        3500.0,
    )


    compliance = torch.zeros(
        (
            batch_size,
            6,
            6,
        ),
        dtype=torch.float32,
        device=E1.device,
    )


    compliance[
        :,
        0,
        0
    ] = 1.0 / E1


    compliance[
        :,
        1,
        1
    ] = 1.0 / E2


    compliance[
        :,
        2,
        2
    ] = 1.0 / E3


    compliance[
        :,
        0,
        1
    ] = -nu12 / E1


    compliance[
        :,
        1,
        0
    ] = -nu12 / E1


    compliance[
        :,
        0,
        2
    ] = -nu13 / E1


    compliance[
        :,
        2,
        0
    ] = -nu13 / E1


    compliance[
        :,
        1,
        2
    ] = -nu23 / E2


    compliance[
        :,
        2,
        1
    ] = -nu23 / E2


    compliance[
        :,
        3,
        3
    ] = 1.0 / G12


    compliance[
        :,
        4,
        4
    ] = 1.0 / G13


    compliance[
        :,
        5,
        5
    ] = 1.0 / G23


    stiffness = torch.linalg.inv(
        compliance
    )


    return stiffness


# ============================================================
# KINEMATIC STRAIN FROM U
# ============================================================

def kinematic_tensor_strain(
    displacement_physical,
):

    # [B,E,8,3]
    element_U = displacement_physical[
        :,
        connectivity_tensor,
        :
    ]


    # [B,E,24]
    element_U = element_U.reshape(
        displacement_physical.shape[
            0
        ],
        connectivity.shape[
            0
        ],
        24,
    )


    # B matrix produces:
    #
    # [e11,e22,e33,gamma12,gamma13,gamma23]
    engineering_strain = torch.einsum(

        "eij,bej->bei",

        B_tensor,

        element_U,
    )


    tensor_strain = engineering_strain.clone()


    tensor_strain[
        :,
        :,
        3:
    ] *= 0.5


    return tensor_strain


# ============================================================
# CONSTITUTIVE STRESS
# ============================================================

def constitutive_stress(
    strain_tensor,
    branch_physical,
):

    engineering_strain = (
        strain_tensor.clone()
    )


    engineering_strain[
        :,
        :,
        3:
    ] *= 2.0


    E1 = branch_physical[
        :,
        0
    ]


    E2 = branch_physical[
        :,
        1
    ]


    G12 = branch_physical[
        :,
        2
    ]


    stiffness = (
        orthotropic_stiffness_batch(
            E1,
            E2,
            G12,
        )
    )


    stress = torch.einsum(

        "bij,bej->bei",

        stiffness,

        engineering_strain,
    )


    return stress


# ============================================================
# MODEL FORWARD IN PHYSICAL + NORMALIZED FORM
# ============================================================

def model_forward(
    model,
    branch_normalized,
):

    (
        U_raw_n,
        LE_pred_n,
        S_pred_n,
        force_pred_n,
    ) = model(

        branch_normalized,

        node_coordinate_tensor,

        ip_coordinate_tensor,
    )


    branch_physical = (
        branch_normalized
        *
        branch_std_tensor
        +
        branch_mean_tensor
    )


    # ========================================================
    # DISPLACEMENT
    # ========================================================

    U_raw_physical = (
        U_raw_n
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


    U_corrected_physical = (
        apply_hard_compatibility_7region(

            raw_displacement=
                U_raw_physical,

            branch_physical=
                branch_physical,

            context=
                hard_context,
        )
    )


    U_corrected_n = (
        U_corrected_physical
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


    # ========================================================
    # MECHANICS PHYSICAL
    # ========================================================

    LE_pred_physical = (
        LE_pred_n
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


    S_pred_physical = (
        S_pred_n
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


    force_pred_physical = (
        force_pred_n
        *
        force_std_tensor.reshape(
            1,
            -1,
        )
        +
        force_mean_tensor.reshape(
            1,
            -1,
        )
    )


    g_pred_physical = (
        force_coefficients_to_g(
            force_pred_physical
        )
    )


    g_pred_n = (
        g_pred_physical
        -
        g_mean_tensor.reshape(
            1,
            -1,
        )
    ) / g_std_tensor.reshape(
        1,
        -1,
    )


    return (
        branch_physical,
        U_corrected_n,
        U_corrected_physical,
        LE_pred_n,
        LE_pred_physical,
        S_pred_n,
        S_pred_physical,
        force_pred_n,
        force_pred_physical,
        g_pred_n,
        g_pred_physical,
    )


# ============================================================
# LOSS
# ============================================================

mse = nn.MSELoss()


def calculate_loss(

    model,

    branch_batch,

    U_target,

    LE_target,

    S_target,

    force_target,

    g_target,
):

    (
        branch_physical,
        U_pred_n,
        U_pred_physical,
        LE_pred_n,
        LE_pred_physical,
        S_pred_n,
        S_pred_physical,
        force_pred_n,
        _,
        g_pred_n,
        _,
    ) = model_forward(
        model,
        branch_batch,
    )


    loss_u = mse(
        U_pred_n,
        U_target,
    )


    loss_strain = mse(
        LE_pred_n,
        LE_target,
    )


    loss_stress = mse(
        S_pred_n,
        S_target,
    )


    loss_force = mse(
        force_pred_n,
        force_target,
    )


    loss_g = mse(
        g_pred_n,
        g_target,
    )


    # ========================================================
    # OPTIONAL KINEMATIC CONSISTENCY
    # ========================================================

    if args.lambda_kin > 0.0:

        LE_kin = (
            kinematic_tensor_strain(
                U_pred_physical
            )
        )


        loss_kin = torch.mean(
            (
                (
                    LE_kin
                    -
                    LE_pred_physical
                )
                /
                LE_std_tensor.reshape(
                    1,
                    1,
                    6,
                )
            ) ** 2
        )

    else:

        loss_kin = torch.tensor(
            0.0,
            device=DEVICE,
        )


    # ========================================================
    # OPTIONAL CONSTITUTIVE CONSISTENCY
    #
    # Because Abaqus LE is logarithmic strain and NLGEOM
    # is active, this is intentionally only a SOFT penalty.
    # ========================================================

    if args.lambda_const > 0.0:

        S_const = (
            constitutive_stress(

                LE_pred_physical,

                branch_physical,
            )
        )


        loss_const = torch.mean(
            (
                (
                    S_const
                    -
                    S_pred_physical
                )
                /
                S_std_tensor.reshape(
                    1,
                    1,
                    6,
                )
            ) ** 2
        )

    else:

        loss_const = torch.tensor(
            0.0,
            device=DEVICE,
        )


    total = (

        args.lambda_u
        *
        loss_u

        +

        args.lambda_strain
        *
        loss_strain

        +

        args.lambda_stress
        *
        loss_stress

        +

        args.lambda_force_coeff
        *
        loss_force

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


    return total


# ============================================================
# VALIDATION METRIC HELPER
# ============================================================

def relative_case_error(
    prediction,
    truth,
):

    errors = []


    for index in range(
        prediction.shape[
            0
        ]
    ):

        errors.append(
            100.0
            *
            np.linalg.norm(
                prediction[
                    index
                ]
                -
                truth[
                    index
                ]
            )
            /
            (
                np.linalg.norm(
                    truth[
                        index
                    ]
                )
                +
                1.0e-14
            )
        )


    return float(
        np.mean(
            errors
        )
    )


# ============================================================
# SEEDS
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


# ============================================================
# ENSEMBLE STORAGE
# ============================================================

ensemble_validation_U = []

ensemble_validation_LE = []

ensemble_validation_S = []

ensemble_validation_g = []


member_rows = []


# ============================================================
# TRAIN
# ============================================================

for member_index in range(
    args.members
):

    seed = SEEDS[
        member_index
    ]


    print("")
    print(
        "=========================================="
    )

    print(
        "MEMBER {} / {} — SEED {}"
        .format(
            member_index + 1,
            args.members,
            seed,
        )
    )

    print(
        "=========================================="
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


    model = HybridBulkOperatorV4SevenRegion(

        branch_dim=
            branch.shape[1],

        number_force_coefficients=
            force_coefficients.shape[1],

        hidden_dim=
            128,

        latent_dim=
            128,

        depth=
            4,
    ).to(
        DEVICE
    )


    # ========================================================
    # OPTIONAL INITIALIZATION FROM DIRECT MODEL
    # ========================================================

    if args.initial_root is not None:

        initial_file = os.path.join(
            args.initial_root,
            "member_{:02d}".format(
                member_index + 1
            ),
            "best_v4_7region.pt",
        )


        if not os.path.isfile(
            initial_file
        ):

            raise FileNotFoundError(
                initial_file
            )


        initial_checkpoint = torch.load(
            initial_file,
            map_location=DEVICE,
            weights_only=False,
        )


        model.load_state_dict(
            initial_checkpoint[
                "model_state_dict"
            ]
        )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=
            args.learning_rate,

        weight_decay=
            1.0e-6,
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


    best_validation = np.inf

    best_epoch = -1

    no_improvement = 0

    history = []


    checkpoint_file = os.path.join(
        member_dir,
        "best_v4_7region.pt",
    )


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        train_total = 0.0

        train_count = 0


        for batch in train_loader:

            (
                branch_batch,
                U_target,
                LE_target,
                S_target,
                force_target,
                g_target,
            ) = [

                value.to(
                    DEVICE
                )

                for value in batch
            ]


            optimizer.zero_grad()


            loss = calculate_loss(

                model,

                branch_batch,

                U_target,

                LE_target,

                S_target,

                force_target,

                g_target,
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


            batch_size = branch_batch.shape[
                0
            ]


            train_total += (
                float(
                    loss.item()
                )
                *
                batch_size
            )


            train_count += (
                batch_size
            )


        train_total /= train_count


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        validation_total = 0.0

        validation_count = 0


        with torch.no_grad():

            for batch in validation_loader:

                (
                    branch_batch,
                    U_target,
                    LE_target,
                    S_target,
                    force_target,
                    g_target,
                ) = [

                    value.to(
                        DEVICE
                    )

                    for value in batch
                ]


                loss = calculate_loss(

                    model,

                    branch_batch,

                    U_target,

                    LE_target,

                    S_target,

                    force_target,

                    g_target,
                )


                batch_size = branch_batch.shape[
                    0
                ]


                validation_total += (
                    float(
                        loss.item()
                    )
                    *
                    batch_size
                )


                validation_count += (
                    batch_size
                )


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


        if validation_total < (
            best_validation
        ):

            best_validation = (
                validation_total
            )


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
                        force_coefficients.shape[1],

                    "hidden_dim":
                        128,

                    "latent_dim":
                        128,

                    "depth":
                        4,

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

                    "best_epoch":
                        epoch,
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
                    train_total,
                    validation_total,
                )
            )


        if no_improvement >= (
            args.patience
        ):

            print(
                "Early stopping at",
                epoch
            )

            break


    # ========================================================
    # HISTORY
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        os.path.join(
            member_dir,
            "training_history.csv",
        ),
        index=False,
    )


    # ========================================================
    # RELOAD BEST
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


    model.eval()


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_U = []

    validation_LE = []

    validation_S = []

    validation_g = []


    with torch.no_grad():

        for case_index in validation_indices:

            branch_case_n = torch.tensor(

                branch_n[
                    case_index
                ],

                dtype=torch.float32,

                device=DEVICE,

            ).reshape(
                1,
                -1,
            )


            (
                _,
                _,
                U_physical,
                _,
                LE_physical,
                _,
                S_physical,
                _,
                _,
                _,
                g_physical,
            ) = model_forward(
                model,
                branch_case_n,
            )


            validation_U.append(
                U_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            validation_LE.append(
                LE_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            validation_S.append(
                S_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


            validation_g.append(
                g_physical[
                    0
                ]
                .cpu()
                .numpy()
            )


    ensemble_validation_U.append(
        np.asarray(
            validation_U
        )
    )


    ensemble_validation_LE.append(
        np.asarray(
            validation_LE
        )
    )


    ensemble_validation_S.append(
        np.asarray(
            validation_S
        )
    )


    ensemble_validation_g.append(
        np.asarray(
            validation_g
        )
    )


    member_rows.append(
        {
            "Member":
                member_index + 1,

            "Seed":
                seed,

            "BestEpoch":
                best_epoch,

            "BestValidationLoss":
                best_validation,
        }
    )


# ============================================================
# ENSEMBLE VALIDATION MEAN
# ============================================================

ensemble_U = np.mean(
    np.asarray(
        ensemble_validation_U
    ),
    axis=0,
)


ensemble_LE = np.mean(
    np.asarray(
        ensemble_validation_LE
    ),
    axis=0,
)


ensemble_S = np.mean(
    np.asarray(
        ensemble_validation_S
    ),
    axis=0,
)


ensemble_g = np.mean(
    np.asarray(
        ensemble_validation_g
    ),
    axis=0,
)


truth_U = U[
    validation_indices
]


truth_LE = LE[
    validation_indices
]


truth_S = S[
    validation_indices
]


truth_g = g[
    validation_indices
]


summary = {

    "Segment":
        args.segment,

    "Members":
        args.members,

    "ValidationCases":
        int(
            len(
                validation_indices
            )
        ),

    "U1_percent":
        relative_case_error(
            ensemble_U[
                :,
                :,
                0
            ],
            truth_U[
                :,
                :,
                0
            ],
        ),

    "U2_percent":
        relative_case_error(
            ensemble_U[
                :,
                :,
                1
            ],
            truth_U[
                :,
                :,
                1
            ],
        ),

    "U3_percent":
        relative_case_error(
            ensemble_U[
                :,
                :,
                2
            ],
            truth_U[
                :,
                :,
                2
            ],
        ),

    "GeneralizedForce_percent":
        relative_case_error(
            ensemble_g,
            truth_g,
        ),
}


components = [
    "11",
    "22",
    "33",
    "12",
    "13",
    "23",
]


for component_index, component_name in enumerate(
    components
):

    summary[
        "LE{}_percent".format(
            component_name
        )
    ] = relative_case_error(

        ensemble_LE[
            :,
            :,
            component_index
        ],

        truth_LE[
            :,
            :,
            component_index
        ],
    )


    summary[
        "S{}_percent".format(
            component_name
        )
    ] = relative_case_error(

        ensemble_S[
            :,
            :,
            component_index
        ],

        truth_S[
            :,
            :,
            component_index
        ],
    )


# ============================================================
# SAVE
# ============================================================

pd.DataFrame(
    member_rows
).to_csv(
    os.path.join(
        args.output_root,
        "ensemble_member_summary.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        args.output_root,
        "validation_summary.json",
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
    "=========================================="
)

print(
    "STEP 64N COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    json.dumps(
        summary,
        indent=4,
    )
)