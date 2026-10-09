import os
import math

import numpy as np
import torch
import torch.nn as nn

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
)

from src.step40_ensemble_utils import (
    get_member_checkpoint_files,
    load_member,
)


# ============================================================
# PATHS
# ============================================================

EXPORT_DIR = os.path.join(
    "exports",
    "step40_ensemble_force_models",
)


os.makedirs(
    EXPORT_DIR,
    exist_ok=True,
)


FORCE_PCA_DIR = os.path.join(
    "data",
    "hybrid_force_pca",
)


# ============================================================
# INTERFACE CONFIGURATION
# ============================================================

CONFIG = {

    "left": {
        "interface_1":
            "m6",
        "interface_2":
            "m2",
    },

    "right": {
        "interface_1":
            "p2",
        "interface_2":
            "p6",
    },
}


# ============================================================
# EXACT GELU USED BY PYTORCH nn.GELU()
# ============================================================

def gelu_numpy(
    x,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )


    erf_values = np.vectorize(
        math.erf
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

def numpy_member_coefficients(
    payload,
    member_index,
    branch_physical,
):

    prefix = "member_{}".format(
        member_index
    )


    branch_mean = payload[
        "{}_branch_mean".format(
            prefix
        )
    ]


    branch_std = payload[
        "{}_branch_std".format(
            prefix
        )
    ]


    coefficient_mean = payload[
        "{}_force_coeff_mean".format(
            prefix
        )
    ]


    coefficient_std = payload[
        "{}_force_coeff_std".format(
            prefix
        )
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
        payload[
            "{}_number_layers".format(
                prefix
            )
        ][0]
    )


    for layer_index in range(
        number_layers
    ):

        W = payload[
            "{}_W{}".format(
                prefix,
                layer_index,
            )
        ]


        b = payload[
            "{}_b{}".format(
                prefix,
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

            x = gelu_numpy(
                x
            )


    return (
        x
        *
        coefficient_std
        +
        coefficient_mean
    )


# ============================================================
# COEFFICIENTS -> GENERALIZED FORCE
# ============================================================

def coefficients_to_g(
    payload,
    coefficients,
):

    number_force_pca_1 = int(
        payload[
            "number_force_pca_1"
        ][0]
    )


    number_force_pca_2 = int(
        payload[
            "number_force_pca_2"
        ][0]
    )


    coeff_1 = coefficients[
        :number_force_pca_1
    ]


    coeff_2 = coefficients[
        number_force_pca_1:
        number_force_pca_1
        +
        number_force_pca_2
    ]


    g1 = (

        payload[
            "g_mean_1"
        ]

        +

        payload[
            "g_matrix_1"
        ]
        @
        coeff_1
    )


    g2 = (

        payload[
            "g_mean_2"
        ]

        +

        payload[
            "g_matrix_2"
        ]
        @
        coeff_2
    )


    return np.concatenate(
        [
            g1,
            g2,
        ]
    )


# ============================================================
# EXPORT ONE SIDE
# ============================================================

def export_side(
    side,
):

    print("")
    print(
        "================================================"
    )

    print(
        "EXPORTING STEP-40 ENSEMBLE:",
        side.upper()
    )

    print(
        "================================================"
    )


    interface_1 = CONFIG[
        side
    ][
        "interface_1"
    ]


    interface_2 = CONFIG[
        side
    ][
        "interface_2"
    ]


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


    checkpoint_files = (
        get_member_checkpoint_files(
            side
        )
    )


    payload = {

        "number_members":
            np.array(
                [
                    len(
                        checkpoint_files
                    )
                ],
                dtype=np.int64,
            ),

        "number_force_pca_1":
            np.array(
                [
                    pca_1[
                        "basis"
                    ].shape[
                        1
                    ]
                ],
                dtype=np.int64,
            ),

        "number_force_pca_2":
            np.array(
                [
                    pca_2[
                        "basis"
                    ].shape[
                        1
                    ]
                ],
                dtype=np.int64,
            ),

        "number_g_1":
            np.array(
                [
                    pca_1[
                        "g_matrix"
                    ].shape[
                        0
                    ]
                ],
                dtype=np.int64,
            ),

        "number_g_2":
            np.array(
                [
                    pca_2[
                        "g_matrix"
                    ].shape[
                        0
                    ]
                ],
                dtype=np.int64,
            ),

        "g_mean_1":
            pca_1[
                "g_mean"
            ].astype(
                np.float64
            ),

        "g_matrix_1":
            pca_1[
                "g_matrix"
            ].astype(
                np.float64
            ),

        "g_mean_2":
            pca_2[
                "g_mean"
            ].astype(
                np.float64
            ),

        "g_matrix_2":
            pca_2[
                "g_matrix"
            ].astype(
                np.float64
            ),
    }


    pytorch_generalized_forces = []


    first_checkpoint = None


    for member_index, checkpoint_file in enumerate(
        checkpoint_files
    ):

        model, checkpoint = load_member(
            checkpoint_file
        )


        if first_checkpoint is None:

            first_checkpoint = checkpoint


            payload[
                "g_std"
            ] = np.asarray(
                checkpoint[
                    "g_std"
                ],
                dtype=np.float64,
            )


            payload[
                "g_mean_training"
            ] = np.asarray(
                checkpoint[
                    "g_mean"
                ],
                dtype=np.float64,
            )


        else:

            if not np.allclose(

                checkpoint[
                    "g_std"
                ],

                first_checkpoint[
                    "g_std"
                ],

                rtol=1.0e-6,

                atol=1.0e-8,
            ):

                raise RuntimeError(
                    "g_std differs between ensemble members."
                )


        prefix = "member_{}".format(
            member_index
        )


        payload[
            "{}_branch_mean".format(
                prefix
            )
        ] = np.asarray(
            checkpoint[
                "branch_mean"
            ],
            dtype=np.float64,
        )


        payload[
            "{}_branch_std".format(
                prefix
            )
        ] = np.asarray(
            checkpoint[
                "branch_std"
            ],
            dtype=np.float64,
        )


        payload[
            "{}_force_coeff_mean".format(
                prefix
            )
        ] = np.asarray(
            checkpoint[
                "force_coeff_mean"
            ],
            dtype=np.float64,
        )


        payload[
            "{}_force_coeff_std".format(
                prefix
            )
        ] = np.asarray(
            checkpoint[
                "force_coeff_std"
            ],
            dtype=np.float64,
        )


        linear_layers = [

            layer

            for layer in (
                model
                .force_coefficient_head
                .network
            )

            if isinstance(
                layer,
                nn.Linear,
            )
        ]


        payload[
            "{}_number_layers".format(
                prefix
            )
        ] = np.array(
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
                "{}_W{}".format(
                    prefix,
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
                "{}_b{}".format(
                    prefix,
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


    # ========================================================
    # SAVE FIRST
    # ========================================================

    output_file = os.path.join(
        EXPORT_DIR,
        "{}_ensemble_force_model.npz".format(
            side
        ),
    )


    np.savez(
        output_file,
        **payload
    )


    # ========================================================
    # VERIFICATION CASE
    # ========================================================

    data_dir = os.path.join(
        "data",
        "hybrid_bulk_{}".format(
            side
        ),
    )


    branch_inputs = np.load(
        os.path.join(
            data_dir,
            "branch_inputs.npy",
        )
    ).astype(
        np.float64
    )


    verification_branch = branch_inputs[
        0
    ]


    exported = np.load(
        output_file
    )


    numpy_g_members = []


    torch_g_members = []


    for member_index, checkpoint_file in enumerate(
        checkpoint_files
    ):

        # ----------------------------------------------------
        # NumPy
        # ----------------------------------------------------

        coeff_numpy = (
            numpy_member_coefficients(
                exported,
                member_index,
                verification_branch,
            )
        )


        g_numpy = coefficients_to_g(
            exported,
            coeff_numpy,
        )


        numpy_g_members.append(
            g_numpy
        )


        # ----------------------------------------------------
        # PyTorch
        # ----------------------------------------------------

        model, checkpoint = load_member(
            checkpoint_file
        )


        branch_normalized = (

            verification_branch

            -

            np.asarray(
                checkpoint[
                    "branch_mean"
                ],
                dtype=np.float64,
            )

        ) / np.asarray(
            checkpoint[
                "branch_std"
            ],
            dtype=np.float64,
        )


        with torch.no_grad():

            coeff_normalized = (

                model
                .force_coefficient_head(

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


        coeff_physical = (

            coeff_normalized

            *

            np.asarray(
                checkpoint[
                    "force_coeff_std"
                ],
                dtype=np.float64,
            )

            +

            np.asarray(
                checkpoint[
                    "force_coeff_mean"
                ],
                dtype=np.float64,
            )
        )


        g_torch = coefficients_to_g(
            exported,
            coeff_physical,
        )


        torch_g_members.append(
            g_torch
        )


    numpy_ensemble = np.mean(
        np.asarray(
            numpy_g_members
        ),
        axis=0,
    )


    torch_ensemble = np.mean(
        np.asarray(
            torch_g_members
        ),
        axis=0,
    )


    maximum_difference = float(
        np.max(
            np.abs(
                numpy_ensemble
                -
                torch_ensemble
            )
        )
    )


    print(
        "Verification max difference:",
        maximum_difference
    )


    if maximum_difference > 5.0e-4:

        raise RuntimeError(
            "NumPy ensemble export verification failed."
        )


    print(
        "Verification PASSED."
    )


    print(
        "Saved:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

export_side(
    "left"
)


export_side(
    "right"
)


print("")
print(
    "================================================"
)

print(
    "STEP 57 COMPLETE"
)

print(
    "================================================"
)