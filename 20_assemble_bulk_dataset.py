import os
import json
import argparse

import numpy as np
import pandas as pd


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
    "--design",
    required=True,
)


parser.add_argument(
    "--extracted_dir",
    required=True,
)


parser.add_argument(
    "--interface_dir",
    default="data/hybrid_interfaces_trainonly",
)


parser.add_argument(
    "--output_dir",
    required=True,
)


parser.add_argument(
    "--seed",
    type=int,
    default=12345,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# INTERFACE DEFINITIONS
# ============================================================

if args.segment == "left":

    LEFT_NAME = "m6"
    RIGHT_NAME = "m2"

else:

    LEFT_NAME = "p2"
    RIGHT_NAME = "p6"


# ============================================================
# LOAD INTERFACES
# ============================================================

left_interface_file = os.path.join(
    args.interface_dir,
    "interface_{}.npz".format(
        LEFT_NAME
    ),
)


right_interface_file = os.path.join(
    args.interface_dir,
    "interface_{}.npz".format(
        RIGHT_NAME
    ),
)


if not os.path.isfile(
    left_interface_file
):

    raise FileNotFoundError(
        left_interface_file
    )


if not os.path.isfile(
    right_interface_file
):

    raise FileNotFoundError(
        right_interface_file
    )


left_interface = np.load(
    left_interface_file
)


right_interface = np.load(
    right_interface_file
)


left_basis = left_interface[
    "basis"
]


right_basis = right_interface[
    "basis"
]


x_left = float(
    left_interface[
        "actual_x"
    ][
        0
    ]
)


x_right = float(
    right_interface[
        "actual_x"
    ][
        0
    ]
)


n_left_modes = (
    left_basis.shape[
        1
    ]
)


n_right_modes = (
    right_basis.shape[
        1
    ]
)


print("")
print(
    "=========================================="
)

print(
    "ASSEMBLING {} BULK DATASET".format(
        args.segment.upper()
    )
)

print(
    "=========================================="
)

print(
    "Left interface:",
    LEFT_NAME,
    "x =",
    x_left,
)

print(
    "Right interface:",
    RIGHT_NAME,
    "x =",
    x_right,
)

print(
    "Left modes:",
    n_left_modes,
)

print(
    "Right modes:",
    n_right_modes,
)


# ============================================================
# LOAD DESIGN
# ============================================================

design = pd.read_csv(
    args.design
)


number_cases = len(
    design
)


if number_cases == 0:

    raise RuntimeError(
        "Design file contains no cases."
    )


# ============================================================
# STORAGE
# ============================================================

all_U1 = []

all_U2 = []

all_U3 = []

all_g_left = []

all_g_right = []


reference_coordinates = None

reference_node_labels = None


force_balance_rows = []


# ============================================================
# CASE LOOP
# ============================================================

for case_number, row in design.iterrows():

    case_id = row[
        "CaseID"
    ]


    node_file = os.path.join(
        args.extracted_dir,
        case_id
        + "_nodes.csv",
    )


    if not os.path.isfile(
        node_file
    ):

        raise FileNotFoundError(
            node_file
        )


    nodes = pd.read_csv(
        node_file
    )


    # --------------------------------------------------------
    # Deterministic node ordering
    # --------------------------------------------------------

    nodes = nodes.sort_values(
        [
            "X",
            "Y",
            "Z",
        ]
    ).reset_index(
        drop=True
    )


    coordinates = nodes[
        [
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    labels = nodes[
        "NodeLabel"
    ].to_numpy(
        dtype=int
    )


    # --------------------------------------------------------
    # Check mesh consistency
    # --------------------------------------------------------

    if reference_coordinates is None:

        reference_coordinates = (
            coordinates.copy()
        )

        reference_node_labels = (
            labels.copy()
        )

    else:

        if coordinates.shape != reference_coordinates.shape:

            raise RuntimeError(
                "Coordinate shape mismatch for {}".format(
                    case_id
                )
            )


        if not np.allclose(
            coordinates,
            reference_coordinates,
            atol=1.0e-9,
        ):

            raise RuntimeError(
                "Coordinate mismatch for {}".format(
                    case_id
                )
            )


    # --------------------------------------------------------
    # Displacement fields
    # --------------------------------------------------------

    all_U1.append(
        nodes[
            "U1"
        ].to_numpy(
            dtype=np.float64
        )
    )


    all_U2.append(
        nodes[
            "U2"
        ].to_numpy(
            dtype=np.float64
        )
    )


    all_U3.append(
        nodes[
            "U3"
        ].to_numpy(
            dtype=np.float64
        )
    )


    # ========================================================
    # LEFT INTERFACE
    # ========================================================

    left_mask = np.isclose(
        nodes[
            "X"
        ].to_numpy(),
        x_left,
        atol=1.0e-7,
    )


    left_nodes = nodes[
        left_mask
    ].copy()


    left_nodes = left_nodes.sort_values(
        [
            "Y",
            "Z",
        ]
    )


    if len(
        left_nodes
    ) * 3 != left_basis.shape[
        0
    ]:

        raise RuntimeError(
            "Left interface dimension mismatch for {}".format(
                case_id
            )
        )


    left_rf = left_nodes[
        [
            "RF1",
            "RF2",
            "RF3",
        ]
    ].to_numpy(
        dtype=np.float64
    ).reshape(
        -1
    )


    # Generalized interface force
    g_left = (
        left_basis.T
        @ left_rf
    )


    # ========================================================
    # RIGHT INTERFACE
    # ========================================================

    right_mask = np.isclose(
        nodes[
            "X"
        ].to_numpy(),
        x_right,
        atol=1.0e-7,
    )


    right_nodes = nodes[
        right_mask
    ].copy()


    right_nodes = right_nodes.sort_values(
        [
            "Y",
            "Z",
        ]
    )


    if len(
        right_nodes
    ) * 3 != right_basis.shape[
        0
    ]:

        raise RuntimeError(
            "Right interface dimension mismatch for {}".format(
                case_id
            )
        )


    right_rf = right_nodes[
        [
            "RF1",
            "RF2",
            "RF3",
        ]
    ].to_numpy(
        dtype=np.float64
    ).reshape(
        -1
    )


    g_right = (
        right_basis.T
        @ right_rf
    )


    all_g_left.append(
        g_left
    )


    all_g_right.append(
        g_right
    )


    # ========================================================
    # GLOBAL FORCE BALANCE
    #
    # For the isolated Dirichlet block:
    #
    # sum(F_left) + sum(F_right) ~ 0
    # ========================================================

    total_left_force = left_nodes[
        [
            "RF1",
            "RF2",
            "RF3",
        ]
    ].sum().to_numpy(
        dtype=np.float64
    )


    total_right_force = right_nodes[
        [
            "RF1",
            "RF2",
            "RF3",
        ]
    ].sum().to_numpy(
        dtype=np.float64
    )


    imbalance = (
        total_left_force
        +
        total_right_force
    )


    force_scale = max(
        np.linalg.norm(
            total_left_force
        ),
        np.linalg.norm(
            total_right_force
        ),
        1.0e-12,
    )


    relative_imbalance = (
        np.linalg.norm(
            imbalance
        )
        /
        force_scale
    )


    force_balance_rows.append(
        {
            "CaseID":
                case_id,

            "Fx_left_N":
                total_left_force[
                    0
                ],

            "Fy_left_N":
                total_left_force[
                    1
                ],

            "Fz_left_N":
                total_left_force[
                    2
                ],

            "Fx_right_N":
                total_right_force[
                    0
                ],

            "Fy_right_N":
                total_right_force[
                    1
                ],

            "Fz_right_N":
                total_right_force[
                    2
                ],

            "Fx_imbalance_N":
                imbalance[
                    0
                ],

            "Fy_imbalance_N":
                imbalance[
                    1
                ],

            "Fz_imbalance_N":
                imbalance[
                    2
                ],

            "ImbalanceNorm_N":
                np.linalg.norm(
                    imbalance
                ),

            "RelativeImbalance":
                relative_imbalance,
        }
    )


    print(
        "[{}/{}] Assembled {}".format(
            case_number + 1,
            number_cases,
            case_id,
        )
    )


# ============================================================
# STACK ARRAYS
# ============================================================

U1 = np.stack(
    all_U1,
    axis=0,
).astype(
    np.float32
)


U2 = np.stack(
    all_U2,
    axis=0,
).astype(
    np.float32
)


U3 = np.stack(
    all_U3,
    axis=0,
).astype(
    np.float32
)


g_left = np.stack(
    all_g_left,
    axis=0,
).astype(
    np.float32
)


g_right = np.stack(
    all_g_right,
    axis=0,
).astype(
    np.float32
)


# ============================================================
# BRANCH INPUT
#
# [E1, E2, G12, c_left..., c_right...]
# ============================================================

branch_columns = [
    "E1_MPa",
    "E2_MPa",
    "G12_MPa",
]


branch_columns += [
    "cL_{:02d}".format(
        i + 1
    )
    for i in range(
        n_left_modes
    )
]


branch_columns += [
    "cR_{:02d}".format(
        i + 1
    )
    for i in range(
        n_right_modes
    )
]


missing_columns = [
    column

    for column in branch_columns

    if column not in design.columns
]


if len(
    missing_columns
) > 0:

    raise RuntimeError(
        "Missing branch columns: {}".format(
            missing_columns
        )
    )


branch_inputs = design[
    branch_columns
].to_numpy(
    dtype=np.float32
)


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    os.path.join(
        args.output_dir,
        "branch_inputs.npy",
    ),
    branch_inputs,
)


np.save(
    os.path.join(
        args.output_dir,
        "coordinates.npy",
    ),
    reference_coordinates.astype(
        np.float32
    ),
)


np.save(
    os.path.join(
        args.output_dir,
        "node_labels.npy",
    ),
    reference_node_labels,
)


np.save(
    os.path.join(
        args.output_dir,
        "U1.npy",
    ),
    U1,
)


np.save(
    os.path.join(
        args.output_dir,
        "U2.npy",
    ),
    U2,
)


np.save(
    os.path.join(
        args.output_dir,
        "U3.npy",
    ),
    U3,
)


np.save(
    os.path.join(
        args.output_dir,
        "g_left.npy",
    ),
    g_left,
)


np.save(
    os.path.join(
        args.output_dir,
        "g_right.npy",
    ),
    g_right,
)


# ============================================================
# SPLIT
#
# 70 / 15 / 15
#
# Same deterministic procedure for left/right.
# ============================================================

rng = np.random.default_rng(
    args.seed
)


indices = np.arange(
    number_cases
)


rng.shuffle(
    indices
)


n_train = int(
    round(
        0.70
        *
        number_cases
    )
)


n_validation = int(
    round(
        0.15
        *
        number_cases
    )
)


train_indices = indices[
    :n_train
]


validation_indices = indices[
    n_train:
    n_train
    +
    n_validation
]


test_indices = indices[
    n_train
    +
    n_validation:
]


split_labels = np.empty(
    number_cases,
    dtype=object
)


split_labels[
    train_indices
] = "train"


split_labels[
    validation_indices
] = "validation"


split_labels[
    test_indices
] = "test"


split_table = pd.DataFrame(
    {
        "Index":
            np.arange(
                number_cases
            ),

        "CaseID":
            design[
                "CaseID"
            ],

        "Source":
            design[
                "Source"
            ],

        "Split":
            split_labels,
    }
)


split_table.to_csv(
    os.path.join(
        args.output_dir,
        "split_assignment.csv",
    ),
    index=False,
)


# ============================================================
# FORCE BALANCE
# ============================================================

force_balance = pd.DataFrame(
    force_balance_rows
)


force_balance.to_csv(
    os.path.join(
        args.output_dir,
        "force_balance.csv",
    ),
    index=False,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "segment":
        args.segment,

    "cases":
        int(
            number_cases
        ),

    "nodes":
        int(
            reference_coordinates.shape[
                0
            ]
        ),

    "branch_dimension":
        int(
            branch_inputs.shape[
                1
            ]
        ),

    "left_modes":
        int(
            n_left_modes
        ),

    "right_modes":
        int(
            n_right_modes
        ),

    "force_dimension":
        int(
            n_left_modes
            +
            n_right_modes
        ),

    "train_cases":
        int(
            len(
                train_indices
            )
        ),

    "validation_cases":
        int(
            len(
                validation_indices
            )
        ),

    "test_cases":
        int(
            len(
                test_indices
            )
        ),

    "x_left":
        float(
            x_left
        ),

    "x_right":
        float(
            x_right
        ),

    "mean_relative_force_imbalance":
        float(
            force_balance[
                "RelativeImbalance"
            ].mean()
        ),

    "max_relative_force_imbalance":
        float(
            force_balance[
                "RelativeImbalance"
            ].max()
        ),
}


with open(
    os.path.join(
        args.output_dir,
        "dataset_metadata.json",
    ),
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "DATASET ASSEMBLY COMPLETE"
)

print(
    "=========================================="
)

print(
    "Segment:",
    args.segment,
)

print(
    "Cases:",
    number_cases,
)

print(
    "Nodes:",
    reference_coordinates.shape[
        0
    ],
)

print(
    "Branch shape:",
    branch_inputs.shape,
)

print(
    "U1 shape:",
    U1.shape,
)

print(
    "U2 shape:",
    U2.shape,
)

print(
    "U3 shape:",
    U3.shape,
)

print(
    "g_left shape:",
    g_left.shape,
)

print(
    "g_right shape:",
    g_right.shape,
)

print(
    "Mean relative force imbalance:",
    force_balance[
        "RelativeImbalance"
    ].mean(),
)

print(
    "Max relative force imbalance:",
    force_balance[
        "RelativeImbalance"
    ].max(),
)

print(
    "Output:",
    args.output_dir,
)