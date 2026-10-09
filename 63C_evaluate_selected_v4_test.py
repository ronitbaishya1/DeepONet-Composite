import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from src.hybrid_bulk_operator_v4 import HybridBulkOperatorV4

from src.hard_interface_compatibility import (
    build_hard_boundary_context,
    apply_hard_compatibility,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    choices=[
        "validation",
        "test",
    ],
    default="test",
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("")
print("Device:", DEVICE)


# ============================================================
# PATHS
# ============================================================

SELECTION_FILE = os.path.join(
    "results",
    "v4_selection",
    "selected_v4_models.json",
)


OUTPUT_DIR = os.path.join(
    "results",
    "v4_final_{}".format(
        args.split
    ),
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def mean_case_relative_l2(
    prediction,
    truth,
):

    values = []

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

        values.append(
            100.0
            *
            numerator
            /
            denominator
        )

    return float(
        np.mean(
            values
        )
    )


def force_coefficients_to_g(
    coefficients,
    force_pca_1,
    force_pca_2,
):

    n1 = force_pca_1[
        "basis"
    ].shape[1]

    n2 = force_pca_2[
        "basis"
    ].shape[1]


    c1 = coefficients[
        :n1
    ]

    c2 = coefficients[
        n1:
        n1 + n2
    ]


    g1 = (
        force_pca_1[
            "g_mean"
        ]
        +
        force_pca_1[
            "g_matrix"
        ]
        @
        c1
    )


    g2 = (
        force_pca_2[
            "g_mean"
        ]
        +
        force_pca_2[
            "g_matrix"
        ]
        @
        c2
    )


    return np.concatenate(
        [
            g1,
            g2,
        ]
    )


# ============================================================
# SELECTION
# ============================================================

with open(
    SELECTION_FILE,
    "r",
) as file_object:

    selection = json.load(
        file_object
    )


summary_rows = []
case_rows = []


# ============================================================
# SEGMENT LOOP
# ============================================================

for segment in [
    "left",
    "right",
]:

    print("")
    print(
        "=========================================="
    )

    print(
        "{} — {}".format(
            segment.upper(),
            args.split.upper(),
        )
    )

    print(
        "=========================================="
    )


    data_dir = os.path.join(
        "data",
        "hybrid_bulk_{}_mechanics".format(
            segment
        ),
    )


    selected_root = selection[
        segment
    ][
        "SelectedRoot"
    ]


    if segment == "left":

        interface_1 = "m6"
        interface_2 = "m2"

    else:

        interface_1 = "p2"
        interface_2 = "p6"


    # ========================================================
    # LOAD DATA
    # ========================================================

    branch = np.load(
        os.path.join(
            data_dir,
            "branch_inputs.npy",
        )
    ).astype(
        np.float32
    )


    coordinates = np.load(
        os.path.join(
            data_dir,
            "coordinates.npy",
        )
    ).astype(
        np.float32
    )


    ip_coordinates = np.load(
        os.path.join(
            data_dir,
            "ip_coordinates.npy",
        )
    ).astype(
        np.float32
    )


    connectivity = np.load(
        os.path.join(
            data_dir,
            "element_node_indices.npy",
        )
    ).astype(
        np.int64
    )


    B_matrices = np.load(
        os.path.join(
            data_dir,
            "B_center.npy",
        )
    ).astype(
        np.float32
    )


    U = np.stack(
        [
            np.load(
                os.path.join(
                    data_dir,
                    "U1.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
                    "U2.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
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
            data_dir,
            "LE.npy",
        )
    ).astype(
        np.float32
    )


    S = np.load(
        os.path.join(
            data_dir,
            "S.npy",
        )
    ).astype(
        np.float32
    )


    g_true = np.concatenate(
        [
            np.load(
                os.path.join(
                    data_dir,
                    "g_left.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
                    "g_right.npy",
                )
            ),
        ],
        axis=1,
    ).astype(
        np.float32
    )


    split_dataframe = pd.read_csv(
        os.path.join(
            data_dir,
            "split_assignment.csv",
        )
    )


    split_values = (
        split_dataframe[
            "Split"
        ]
        .astype(str)
        .str.lower()
    )


    if args.split == "validation":

        split_mask = split_values.isin(
            [
                "validation",
                "val",
            ]
        )

    else:

        split_mask = (
            split_values
            ==
            "test"
        )


    selected_indices = split_dataframe.loc[
        split_mask,
        "Index",
    ].to_numpy(
        dtype=int
    )


    selected_case_ids = split_dataframe.loc[
        split_mask,
        "CaseID",
    ].astype(
        str
    ).to_numpy()


    print(
        "Cases:",
        len(
            selected_indices
        )
    )


    # ========================================================
    # FORCE PCA
    # ========================================================

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


    # ========================================================
    # HARD INTERFACE CONTEXT
    # ========================================================

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


    # ========================================================
    # ENSEMBLE STORAGE
    # ========================================================

    member_U = []
    member_LE = []
    member_S = []
    member_g = []


    # ========================================================
    # FIVE MEMBERS
    # ========================================================

    for member_number in range(
        1,
        6,
    ):

        checkpoint_file = os.path.join(
            selected_root,
            "member_{:02d}".format(
                member_number
            ),
            "best_v4.pt",
        )


        if not os.path.isfile(
            checkpoint_file
        ):

            raise FileNotFoundError(
                checkpoint_file
            )


        print(
            "Member:",
            member_number
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


        branch_mean = np.asarray(
            checkpoint[
                "branch_mean"
            ],
            dtype=np.float32,
        )


        branch_std = np.asarray(
            checkpoint[
                "branch_std"
            ],
            dtype=np.float32,
        )


        coordinate_min = np.asarray(
            checkpoint[
                "coordinate_min"
            ],
            dtype=np.float32,
        )


        coordinate_max = np.asarray(
            checkpoint[
                "coordinate_max"
            ],
            dtype=np.float32,
        )


        coordinate_range = np.maximum(
            coordinate_max
            -
            coordinate_min,
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


        coordinate_tensor = torch.tensor(
            coordinates_normalized,
            dtype=torch.float32,
            device=DEVICE,
        )


        ip_coordinate_tensor = torch.tensor(
            ip_coordinates_normalized,
            dtype=torch.float32,
            device=DEVICE,
        )


        U_mean = np.asarray(
            checkpoint[
                "U_mean"
            ],
            dtype=np.float32,
        )


        U_std = np.asarray(
            checkpoint[
                "U_std"
            ],
            dtype=np.float32,
        )


        LE_mean = np.asarray(
            checkpoint[
                "LE_mean"
            ],
            dtype=np.float32,
        )


        LE_std = np.asarray(
            checkpoint[
                "LE_std"
            ],
            dtype=np.float32,
        )


        S_mean = np.asarray(
            checkpoint[
                "S_mean"
            ],
            dtype=np.float32,
        )


        S_std = np.asarray(
            checkpoint[
                "S_std"
            ],
            dtype=np.float32,
        )


        force_mean = np.asarray(
            checkpoint[
                "force_coeff_mean"
            ],
            dtype=np.float32,
        )


        force_std = np.asarray(
            checkpoint[
                "force_coeff_std"
            ],
            dtype=np.float32,
        )


        prediction_U = []
        prediction_LE = []
        prediction_S = []
        prediction_g = []


        with torch.no_grad():

            for case_index in selected_indices:

                branch_physical_numpy = branch[
                    case_index
                ]


                branch_normalized_numpy = (
                    branch_physical_numpy
                    -
                    branch_mean
                ) / branch_std


                branch_physical_tensor = torch.tensor(
                    branch_physical_numpy,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    -1,
                )


                branch_normalized_tensor = torch.tensor(
                    branch_normalized_numpy,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    -1,
                )


                (
                    U_raw_normalized,
                    LE_normalized,
                    S_normalized,
                    force_normalized,
                ) = model(
                    branch_normalized_tensor,
                    coordinate_tensor,
                    ip_coordinate_tensor,
                )


                U_raw_physical = (
                    U_raw_normalized
                    *
                    torch.tensor(
                        U_std,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).reshape(
                        1,
                        1,
                        3,
                    )
                    +
                    torch.tensor(
                        U_mean,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).reshape(
                        1,
                        1,
                        3,
                    )
                )


                U_corrected = apply_hard_compatibility(

                    raw_displacement=
                        U_raw_physical,

                    branch_physical=
                        branch_physical_tensor,

                    context=
                        hard_context,
                )


                LE_prediction = (
                    LE_normalized[
                        0
                    ]
                    .cpu()
                    .numpy()
                    *
                    LE_std.reshape(
                        1,
                        6,
                    )
                    +
                    LE_mean.reshape(
                        1,
                        6,
                    )
                )


                S_prediction = (
                    S_normalized[
                        0
                    ]
                    .cpu()
                    .numpy()
                    *
                    S_std.reshape(
                        1,
                        6,
                    )
                    +
                    S_mean.reshape(
                        1,
                        6,
                    )
                )


                force_coefficients = (
                    force_normalized[
                        0
                    ]
                    .cpu()
                    .numpy()
                    *
                    force_std
                    +
                    force_mean
                )


                g_prediction = (
                    force_coefficients_to_g(
                        force_coefficients,
                        force_pca_1,
                        force_pca_2,
                    )
                )


                prediction_U.append(
                    U_corrected[
                        0
                    ]
                    .cpu()
                    .numpy()
                )


                prediction_LE.append(
                    LE_prediction
                )


                prediction_S.append(
                    S_prediction
                )


                prediction_g.append(
                    g_prediction
                )


        member_U.append(
            np.asarray(
                prediction_U
            )
        )


        member_LE.append(
            np.asarray(
                prediction_LE
            )
        )


        member_S.append(
            np.asarray(
                prediction_S
            )
        )


        member_g.append(
            np.asarray(
                prediction_g
            )
        )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    member_U = np.asarray(
        member_U
    )

    member_LE = np.asarray(
        member_LE
    )

    member_S = np.asarray(
        member_S
    )

    member_g = np.asarray(
        member_g
    )


    ensemble_U = np.mean(
        member_U,
        axis=0,
    )


    ensemble_LE = np.mean(
        member_LE,
        axis=0,
    )


    ensemble_S = np.mean(
        member_S,
        axis=0,
    )


    ensemble_g = np.mean(
        member_g,
        axis=0,
    )


    std_U = np.std(
        member_U,
        axis=0,
        ddof=1,
    )


    std_LE = np.std(
        member_LE,
        axis=0,
        ddof=1,
    )


    std_S = np.std(
        member_S,
        axis=0,
        ddof=1,
    )


    truth_U = U[
        selected_indices
    ]

    truth_LE = LE[
        selected_indices
    ]

    truth_S = S[
        selected_indices
    ]

    truth_g = g_true[
        selected_indices
    ]


    # ========================================================
    # METRICS
    # ========================================================

    summary = {

        "Segment":
            segment,

        "Split":
            args.split,

        "U1_percent":
            mean_case_relative_l2(
                ensemble_U[:, :, 0],
                truth_U[:, :, 0],
            ),

        "U2_percent":
            mean_case_relative_l2(
                ensemble_U[:, :, 1],
                truth_U[:, :, 1],
            ),

        "U3_percent":
            mean_case_relative_l2(
                ensemble_U[:, :, 2],
                truth_U[:, :, 2],
            ),

        "GeneralizedForce_percent":
            mean_case_relative_l2(
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
        ] = mean_case_relative_l2(

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
        ] = mean_case_relative_l2(

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


    summary_rows.append(
        summary
    )


    # ========================================================
    # CASE-WISE U2
    # ========================================================

    for local_index, case_id in enumerate(
        selected_case_ids
    ):

        relative_error = (
            100.0
            *
            np.linalg.norm(
                ensemble_U[
                    local_index,
                    :,
                    1
                ]
                -
                truth_U[
                    local_index,
                    :,
                    1
                ]
            )
            /
            (
                np.linalg.norm(
                    truth_U[
                        local_index,
                        :,
                        1
                    ]
                )
                +
                1.0e-14
            )
        )


        case_rows.append(
            {
                "Segment":
                    segment,

                "CaseID":
                    case_id,

                "U2_RelL2_percent":
                    relative_error,
            }
        )


    # ========================================================
    # SAVE COMPLETE FIELDS
    # ========================================================

    np.savez_compressed(

        os.path.join(
            OUTPUT_DIR,
            "{}_{}_predictions.npz".format(
                segment,
                args.split,
            ),
        ),

        case_indices=
            selected_indices,

        case_ids=
            selected_case_ids,

        branch_physical=
            branch[
                selected_indices
            ],

        coordinates=
            coordinates,

        ip_coordinates=
            ip_coordinates,

        element_node_indices=
            connectivity,

        B_center=
            B_matrices,

        U_pred=
            ensemble_U,

        U_true=
            truth_U,

        U_ensemble_std=
            std_U,

        LE_pred=
            ensemble_LE,

        LE_true=
            truth_LE,

        LE_ensemble_std=
            std_LE,

        S_pred=
            ensemble_S,

        S_true=
            truth_S,

        S_ensemble_std=
            std_S,

        g_pred=
            ensemble_g,

        g_true=
            truth_g,
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)


case_dataframe = pd.DataFrame(
    case_rows
)


summary_dataframe.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_{}_summary.csv".format(
            args.split
        ),
    ),
    index=False,
)


case_dataframe.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_{}_case_u2.csv".format(
            args.split
        ),
    ),
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "v4_{}_summary.json".format(
            args.split
        ),
    ),
    "w",
) as file_object:

    json.dump(
        summary_rows,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 63C COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    summary_dataframe.to_string(
        index=False
    )
)