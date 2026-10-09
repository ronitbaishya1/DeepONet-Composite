import os
import json

import numpy as np
import pandas as pd

import torch

from src.hybrid_bulk_operator_v4_7region import (
    HybridBulkOperatorV4SevenRegion,
)

from src.hard_interface_compatibility_7region import (
    build_hard_boundary_context_7region,
    apply_hard_compatibility_7region,
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


SEGMENTS = [
    "outer_left",
    "inner_left",
    "inner_right",
    "outer_right",
]


DATA_ROOT = os.path.join(
    "data",
    "hybrid_bulk_7region",
)


INTERFACE_DIR = os.path.join(
    "data",
    "candidate_partition_interfaces_7region",
)


FORCE_PCA_DIR = os.path.join(
    "data",
    "hybrid_force_pca_7region",
)


SELECTION_FILE = os.path.join(
    "results",
    "v4_7region_selection",
    "selected_7region_models.json",
)


OUTPUT_DIR = os.path.join(
    "results",
    "v4_7region_final_test",
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


# ============================================================
# SEGMENT LOOP
# ============================================================

for segment in SEGMENTS:

    print("")
    print(
        "=========================================="
    )

    print(
        segment.upper()
    )

    print(
        "=========================================="
    )


    data_dir = os.path.join(
        DATA_ROOT,
        segment,
    )


    with open(
        os.path.join(
            data_dir,
            "metadata.json",
        ),
        "r",
    ) as file_object:

        metadata = json.load(
            file_object
        )


    left_interface = metadata[
        "left_interface"
    ]


    right_interface = metadata[
        "right_interface"
    ]


    # ========================================================
    # DATA
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


    g_true = np.load(
        os.path.join(
            data_dir,
            "g_all.npy",
        )
    ).astype(
        np.float32
    )


    split_dataframe = pd.read_csv(
        os.path.join(
            data_dir,
            "split_assignment.csv",
        )
    )


    test_indices = split_dataframe.loc[
        split_dataframe[
            "Split"
        ] == "test",
        "Index",
    ].to_numpy(
        dtype=int
    )


    test_case_ids = split_dataframe.loc[
        split_dataframe[
            "Split"
        ] == "test",
        "CaseID",
    ].astype(
        str
    ).to_numpy()


    # ========================================================
    # SELECTED ROOT
    # ========================================================

    selected_root = selection[
        segment
    ][
        "SelectedRoot"
    ]


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
            "best_v4_7region.pt",
        )


        checkpoint = torch.load(
            checkpoint_file,
            map_location=DEVICE,
            weights_only=False,
        )


        model = HybridBulkOperatorV4SevenRegion(

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


        # ====================================================
        # NORMALIZATION
        # ====================================================

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


        coordinate_range = np.maximum(
            coordinate_max
            -
            coordinate_min,
            1.0e-8,
        )


        node_coordinates_n = (
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


        ip_coordinates_n = (
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


        node_tensor = torch.tensor(
            node_coordinates_n,
            dtype=torch.float32,
            device=DEVICE,
        )


        ip_tensor = torch.tensor(
            ip_coordinates_n,
            dtype=torch.float32,
            device=DEVICE,
        )


        # ====================================================
        # HARD CONTEXT
        # ====================================================

        hard_context = (
            build_hard_boundary_context_7region(

                coordinates=
                    coordinates,

                interface_dir=
                    INTERFACE_DIR,

                left_interface_name=
                    left_interface,

                right_interface_name=
                    right_interface,

                device=
                    DEVICE,
            )
        )


        # ====================================================
        # FORCE MAPPINGS
        # ====================================================

        force_blocks = []


        for interface_name in [
            left_interface,
            right_interface,
        ]:

            if interface_name is None:

                continue


            force_data = np.load(
                os.path.join(
                    FORCE_PCA_DIR,
                    "force_pca_{}.npz".format(
                        interface_name
                    ),
                )
            )


            force_blocks.append(
                {
                    "number_modes":
                        int(
                            force_data[
                                "number_modes"
                            ][0]
                        ),

                    "g_mean":
                        force_data[
                            "g_mean"
                        ].astype(
                            np.float32
                        ),

                    "g_matrix":
                        force_data[
                            "g_matrix"
                        ].astype(
                            np.float32
                        ),
                }
            )


        # ====================================================
        # CASES
        # ====================================================

        predictions_U = []

        predictions_LE = []

        predictions_S = []

        predictions_g = []


        with torch.no_grad():

            for case_index in test_indices:

                branch_physical = branch[
                    case_index
                ]


                branch_n = (
                    branch_physical
                    -
                    branch_mean
                ) / branch_std


                branch_tensor = torch.tensor(
                    branch_n,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    -1,
                )


                (
                    U_raw_n,
                    LE_n,
                    S_n,
                    force_n,
                ) = model(

                    branch_tensor,

                    node_tensor,

                    ip_tensor,
                )


                U_raw = (
                    U_raw_n
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


                branch_physical_tensor = torch.tensor(
                    branch_physical,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    -1,
                )


                U_corrected = (
                    apply_hard_compatibility_7region(

                        raw_displacement=
                            U_raw,

                        branch_physical=
                            branch_physical_tensor,

                        context=
                            hard_context,
                    )
                )


                LE_prediction = (
                    LE_n[
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
                    S_n[
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
                    force_n[
                        0
                    ]
                    .cpu()
                    .numpy()
                    *
                    force_std
                    +
                    force_mean
                )


                # ============================================
                # FORCE COEFF -> g
                # ============================================

                g_parts = []

                cursor = 0


                for block in force_blocks:

                    number_modes = block[
                        "number_modes"
                    ]


                    coeff = force_coefficients[
                        cursor:
                        cursor
                        +
                        number_modes
                    ]


                    current_g = (
                        block[
                            "g_mean"
                        ]
                        +
                        block[
                            "g_matrix"
                        ]
                        @
                        coeff
                    )


                    g_parts.append(
                        current_g
                    )


                    cursor += (
                        number_modes
                    )


                g_prediction = np.concatenate(
                    g_parts
                )


                predictions_U.append(
                    U_corrected[
                        0
                    ]
                    .cpu()
                    .numpy()
                )


                predictions_LE.append(
                    LE_prediction
                )


                predictions_S.append(
                    S_prediction
                )


                predictions_g.append(
                    g_prediction
                )


        member_U.append(
            np.asarray(
                predictions_U
            )
        )


        member_LE.append(
            np.asarray(
                predictions_LE
            )
        )


        member_S.append(
            np.asarray(
                predictions_S
            )
        )


        member_g.append(
            np.asarray(
                predictions_g
            )
        )


    # ========================================================
    # ENSEMBLE
    # ========================================================

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


    truth_U = U[
        test_indices
    ]


    truth_LE = LE[
        test_indices
    ]


    truth_S = S[
        test_indices
    ]


    truth_g = g_true[
        test_indices
    ]


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "Segment":
            segment,

        "SelectedType":
            selection[
                segment
            ][
                "SelectedType"
            ],

        "U1_percent":
            mean_case_relative_l2(
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
            mean_case_relative_l2(
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
            mean_case_relative_l2(
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
    # SAVE COMPLETE TEST FIELDS
    # ========================================================

    np.savez_compressed(

        os.path.join(
            OUTPUT_DIR,
            "{}_test_predictions.npz".format(
                segment
            ),
        ),

        case_indices=
            test_indices,

        case_ids=
            test_case_ids,

        coordinates=
            coordinates,

        ip_coordinates=
            ip_coordinates,

        branch_physical=
            branch[
                test_indices
            ],

        U_pred=
            ensemble_U,

        U_true=
            truth_U,

        LE_pred=
            ensemble_LE,

        LE_true=
            truth_LE,

        S_pred=
            ensemble_S,

        S_true=
            truth_S,

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


summary_dataframe.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "seven_region_test_summary.csv",
    ),
    index=False,
)


print("")
print(
    "=========================================="
)

print(
    "STEP 64P COMPLETE"
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