import os
import math
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from src.hybrid_bulk_operator_v4_7region import (
    HybridBulkOperatorV4SevenRegion,
)


# ============================================================
# PATHS
# ============================================================

DIRECT_ROOT = os.path.join(
    "results",
    "v4_7region_direct",
)


DATA_ROOT = os.path.join(
    "data",
    "hybrid_bulk_7region",
)


FORCE_PCA_DIR = os.path.join(
    "data",
    "hybrid_force_pca_7region",
)


OUTPUT_DIR = os.path.join(
    "exports",
    "hybrid_force_models_7region",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


SEGMENTS = [
    "outer_left",
    "inner_left",
    "inner_right",
    "outer_right",
]


NUMBER_MEMBERS = 5


# ============================================================
# EXACT GELU USED BY PYTORCH nn.GELU()
# ============================================================

def gelu_exact(
    x,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )


    erf_values = np.vectorize(
        math.erf,
        otypes=[
            np.float64,
        ],
    )(
        x
        /
        np.sqrt(
            2.0
        )
    )


    return (
        0.5
        *
        x
        *
        (
            1.0
            +
            erf_values
        )
    )


# ============================================================
# NUMPY MEMBER FORWARD
# ============================================================

def numpy_member_forward(
    exported,
    member_index,
    branch_physical,
):

    branch_mean = exported[
        "branch_mean"
    ]


    branch_std = exported[
        "branch_std"
    ]


    force_mean = exported[
        "force_coeff_mean"
    ]


    force_std = exported[
        "force_coeff_std"
    ]


    x = (
        np.asarray(
            branch_physical,
            dtype=np.float64,
        )
        -
        branch_mean
    ) / branch_std


    number_layers = int(
        exported[
            "member_{}_number_layers".format(
                member_index
            )
        ][0]
    )


    for layer_index in range(
        number_layers
    ):

        W = exported[
            "member_{}_W{}".format(
                member_index,
                layer_index,
            )
        ]


        b = exported[
            "member_{}_b{}".format(
                member_index,
                layer_index,
            )
        ]


        x = (
            W
            @
            x
            +
            b
        )


        if layer_index < (
            number_layers
            -
            1
        ):

            x = gelu_exact(
                x
            )


    return (
        x
        *
        force_std
        +
        force_mean
    )


# ============================================================
# EXPORT ONE SEGMENT
# ============================================================

def export_segment(
    segment,
):

    print("")
    print(
        "=========================================="
    )

    print(
        "EXPORTING DIRECT ENSEMBLE:",
        segment.upper(),
    )

    print(
        "=========================================="
    )


    data_dir = os.path.join(
        DATA_ROOT,
        segment,
    )


    model_root = os.path.join(
        DIRECT_ROOT,
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


    branch_inputs = np.load(
        os.path.join(
            data_dir,
            "branch_inputs.npy",
        )
    ).astype(
        np.float64
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


    verification_index = (
        int(
            test_indices[
                0
            ]
        )
        if len(
            test_indices
        ) > 0
        else
        0
    )


    payload = {

        "number_members":
            np.asarray(
                [
                    NUMBER_MEMBERS
                ],
                dtype=np.int64,
            ),
    }


    torch_predictions = []


    # ========================================================
    # MEMBERS
    # ========================================================

    reference_normalization = None


    for member_number in range(
        1,
        NUMBER_MEMBERS + 1,
    ):

        checkpoint_file = os.path.join(
            model_root,
            "member_{:02d}".format(
                member_number
            ),
            "best_v4_7region.pt",
        )


        if not os.path.isfile(
            checkpoint_file
        ):

            raise FileNotFoundError(
                checkpoint_file
            )


        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
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
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.eval()


        normalization = {

            "branch_mean":
                np.asarray(
                    checkpoint[
                        "branch_mean"
                    ],
                    dtype=np.float64,
                ),

            "branch_std":
                np.asarray(
                    checkpoint[
                        "branch_std"
                    ],
                    dtype=np.float64,
                ),

            "force_coeff_mean":
                np.asarray(
                    checkpoint[
                        "force_coeff_mean"
                    ],
                    dtype=np.float64,
                ),

            "force_coeff_std":
                np.asarray(
                    checkpoint[
                        "force_coeff_std"
                    ],
                    dtype=np.float64,
                ),

            "g_mean":
                np.asarray(
                    checkpoint[
                        "g_mean"
                    ],
                    dtype=np.float64,
                ),

            "g_std":
                np.asarray(
                    checkpoint[
                        "g_std"
                    ],
                    dtype=np.float64,
                ),
        }


        if reference_normalization is None:

            reference_normalization = (
                normalization
            )


            for key, value in (
                normalization.items()
            ):

                payload[
                    key
                ] = value


            payload[
                "branch_dim"
            ] = np.asarray(
                [
                    checkpoint[
                        "branch_dim"
                    ]
                ],
                dtype=np.int64,
            )


            payload[
                "number_force_coefficients"
            ] = np.asarray(
                [
                    checkpoint[
                        "number_force_coefficients"
                    ]
                ],
                dtype=np.int64,
            )


        else:

            for key in (
                reference_normalization
            ):

                if not np.allclose(
                    reference_normalization[
                        key
                    ],
                    normalization[
                        key
                    ],
                    atol=1.0e-8,
                    rtol=1.0e-8,
                ):

                    raise RuntimeError(
                        (
                            "Normalization differs across "
                            "ensemble members: {}"
                        ).format(
                            key
                        )
                    )


        # ====================================================
        # EXTRACT FORCE HEAD
        # ====================================================

        linear_layers = [

            layer

            for layer
            in model.force_coefficient_head.network

            if isinstance(
                layer,
                nn.Linear,
            )
        ]


        member_index = (
            member_number
            -
            1
        )


        payload[
            "member_{}_number_layers".format(
                member_index
            )
        ] = np.asarray(
            [
                len(
                    linear_layers
                )
            ],
            dtype=np.int64,
        )


        for layer_index, layer in enumerate(
            linear_layers
        ):

            payload[
                "member_{}_W{}".format(
                    member_index,
                    layer_index,
                )
            ] = (
                layer
                .weight
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )


            payload[
                "member_{}_b{}".format(
                    member_index,
                    layer_index,
                )
            ] = (
                layer
                .bias
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )


        # ====================================================
        # TORCH VERIFICATION PREDICTION
        # ====================================================

        branch_physical = branch_inputs[
            verification_index
        ]


        branch_normalized = (
            branch_physical
            -
            normalization[
                "branch_mean"
            ]
        ) / normalization[
            "branch_std"
        ]


        with torch.no_grad():

            force_normalized = (
                model.force_coefficient_head(

                    torch.tensor(
                        branch_normalized,
                        dtype=torch.float32,
                    ).reshape(
                        1,
                        -1,
                    )

                )[
                    0
                ]
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )


        force_physical = (

            force_normalized
            *
            normalization[
                "force_coeff_std"
            ]

            +

            normalization[
                "force_coeff_mean"
            ]
        )


        torch_predictions.append(
            force_physical
        )


    # ========================================================
    # INTERFACE MODE INFORMATION
    # ========================================================

    for side_name, interface_name in [

        (
            "left",
            left_interface,
        ),

        (
            "right",
            right_interface,
        ),
    ]:

        if interface_name is None:

            payload[
                "{}_force_modes".format(
                    side_name
                )
            ] = np.asarray(
                [
                    0
                ],
                dtype=np.int64,
            )


            payload[
                "{}_g_modes".format(
                    side_name
                )
            ] = np.asarray(
                [
                    0
                ],
                dtype=np.int64,
            )


            continue


        force_pca = np.load(
            os.path.join(
                FORCE_PCA_DIR,
                "force_pca_{}.npz".format(
                    interface_name
                ),
            )
        )


        payload[
            "{}_force_modes".format(
                side_name
            )
        ] = np.asarray(
            [
                int(
                    force_pca[
                        "number_modes"
                    ][0]
                )
            ],
            dtype=np.int64,
        )


        payload[
            "{}_g_modes".format(
                side_name
            )
        ] = np.asarray(
            [
                int(
                    force_pca[
                        "g_mean"
                    ].shape[0]
                )
            ],
            dtype=np.int64,
        )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        "{}_direct_ensemble_force.npz".format(
            segment
        ),
    )


    np.savez(
        output_file,
        **payload
    )


    # ========================================================
    # VERIFY NUMPY ENSEMBLE
    # ========================================================

    exported = np.load(
        output_file
    )


    numpy_predictions = []


    for member_index in range(
        NUMBER_MEMBERS
    ):

        numpy_predictions.append(

            numpy_member_forward(

                exported,

                member_index,

                branch_inputs[
                    verification_index
                ],
            )
        )


    torch_ensemble = np.mean(
        np.asarray(
            torch_predictions
        ),
        axis=0,
    )


    numpy_ensemble = np.mean(
        np.asarray(
            numpy_predictions
        ),
        axis=0,
    )


    maximum_difference = float(
        np.max(
            np.abs(
                torch_ensemble
                -
                numpy_ensemble
            )
        )
    )


    print(
        "Verification case:",
        verification_index
    )

    print(
        "Max PyTorch/NumPy difference:",
        maximum_difference
    )


    if maximum_difference > 1.0e-4:

        raise RuntimeError(
            "Export verification failed."
        )


    print(
        "PASS"
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# MAIN
# ============================================================

for segment in SEGMENTS:

    export_segment(
        segment
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64Q COMPLETE"
)

print(
    "=========================================="
)