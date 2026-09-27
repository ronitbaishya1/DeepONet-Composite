import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from src.hybrid_bulk_operator import (
    HybridBulkOperator
)


# ============================================================
# PROJECT PATHS
# ============================================================

EXPORT_DIR = os.path.join(
    "exports",
    "hybrid_force_models",
)

os.makedirs(
    EXPORT_DIR,
    exist_ok=True,
)


MODELS = {

    "left": {
        "data_dir":
            "data/hybrid_bulk_left",

        "model_dir":
            "results/hybrid_bulk_left",
    },

    "right": {
        "data_dir":
            "data/hybrid_bulk_right",

        "model_dir":
            "results/hybrid_bulk_right",
    },
}


# ============================================================
# NUMPY FORWARD PASS
#
# Used only to verify exported weights.
# ============================================================

def numpy_force_forward(
    exported,
    branch_physical,
):

    branch_mean = exported[
        "branch_mean"
    ]

    branch_std = exported[
        "branch_std"
    ]

    force_mean = exported[
        "force_mean"
    ]

    force_std = exported[
        "force_std"
    ]


    x = (
        branch_physical
        - branch_mean
    ) / branch_std


    number_layers = int(
        exported[
            "number_layers"
        ][
            0
        ]
    )


    for layer_index in range(
        number_layers
    ):

        W = exported[
            "W{}".format(
                layer_index
            )
        ]


        b = exported[
            "b{}".format(
                layer_index
            )
        ]


        x = (
            W
            @ x
            + b
        )


        if layer_index < (
            number_layers
            - 1
        ):

            x = np.tanh(
                x
            )


    force_physical = (
        x
        * force_std
        + force_mean
    )


    return force_physical


# ============================================================
# EXPORT ONE MODEL
# ============================================================

def export_model(
    label,
    data_dir,
    model_dir,
):

    print("")
    print(
        "============================================"
    )

    print(
        "EXPORTING:",
        label,
    )

    print(
        "============================================"
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    normalization_file = os.path.join(
        model_dir,
        "normalization.json",
    )


    with open(
        normalization_file,
        "r",
    ) as f:

        normalization = json.load(
            f
        )


    branch_mean = np.asarray(
        normalization[
            "branch_mean"
        ],
        dtype=np.float64,
    )


    branch_std = np.asarray(
        normalization[
            "branch_std"
        ],
        dtype=np.float64,
    )


    force_mean = np.asarray(
        normalization[
            "force_mean"
        ],
        dtype=np.float64,
    )


    force_std = np.asarray(
        normalization[
            "force_std"
        ],
        dtype=np.float64,
    )


    n_left_force_modes = int(
        normalization[
            "n_left_force_modes"
        ]
    )


    n_right_force_modes = int(
        normalization[
            "n_right_force_modes"
        ]
    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_file = os.path.join(
        model_dir,
        "best_hybrid_bulk_operator.pt",
    )


    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
    )


    model = HybridBulkOperator(
        branch_dim=checkpoint[
            "branch_dim"
        ],
        number_force_outputs=checkpoint[
            "number_force_outputs"
        ],
        latent_dim=checkpoint[
            "latent_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    # --------------------------------------------------------
    # EXTRACT LINEAR LAYERS FROM FORCE HEAD
    #
    # Architecture:
    #
    # branch
    # -> Linear
    # -> tanh
    # -> Linear
    # -> tanh
    # -> Linear
    # -> tanh
    # -> Linear
    # -> generalized forces
    # --------------------------------------------------------

    linear_layers = [

        layer

        for layer in model.force_head.network

        if isinstance(
            layer,
            nn.Linear,
        )
    ]


    payload = {

        "number_layers":
            np.array(
                [
                    len(
                        linear_layers
                    )
                ],
                dtype=np.int64,
            ),

        "branch_mean":
            branch_mean,

        "branch_std":
            branch_std,

        "force_mean":
            force_mean,

        "force_std":
            force_std,

        "n_left_force_modes":
            np.array(
                [
                    n_left_force_modes
                ],
                dtype=np.int64,
            ),

        "n_right_force_modes":
            np.array(
                [
                    n_right_force_modes
                ],
                dtype=np.int64,
            ),
    }


    for layer_index, layer in enumerate(
        linear_layers
    ):

        payload[
            "W{}".format(
                layer_index
            )
        ] = (
            layer.weight
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        payload[
            "b{}".format(
                layer_index
            )
        ] = (
            layer.bias
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        EXPORT_DIR,
        "{}_force_model.npz".format(
            label
        ),
    )


    np.savez(
        output_file,
        **payload
    )


    # --------------------------------------------------------
    # VERIFY EXPORT AGAINST PYTORCH
    # --------------------------------------------------------

    branch_inputs = np.load(
        os.path.join(
            data_dir,
            "branch_inputs.npy",
        )
    ).astype(
        np.float64
    )


    split_table = pd.read_csv(
        os.path.join(
            data_dir,
            "split_assignment.csv",
        )
    )


    test_indices = split_table.loc[
        split_table[
            "Split"
        ] == "test",
        "Index",
    ].to_numpy(
        dtype=int
    )


    if len(
        test_indices
    ) == 0:

        verification_index = 0

    else:

        verification_index = int(
            test_indices[
                0
            ]
        )


    branch_physical = branch_inputs[
        verification_index
    ]


    branch_normalized = (
        branch_physical
        - branch_mean
    ) / branch_std


    with torch.no_grad():

        torch_force_normalized = (
            model.force_head(
                torch.tensor(
                    branch_normalized,
                    dtype=torch.float32,
                ).unsqueeze(
                    0
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


    torch_force = (
        torch_force_normalized
        * force_std
        + force_mean
    )


    exported = np.load(
        output_file
    )


    numpy_force = numpy_force_forward(
        exported,
        branch_physical,
    )


    maximum_difference = np.max(
        np.abs(
            torch_force
            - numpy_force
        )
    )


    print(
        "Verification case:",
        verification_index,
    )

    print(
        "Maximum PyTorch/NumPy difference:",
        maximum_difference,
    )


    if maximum_difference > 1.0e-4:

        raise RuntimeError(
            "Export verification failed."
        )


    print(
        "Export verified successfully."
    )

    print(
        "Saved:",
        output_file,
    )


# ============================================================
# MAIN
# ============================================================

for label, settings in MODELS.items():

    export_model(
        label=label,
        data_dir=settings[
            "data_dir"
        ],
        model_dir=settings[
            "model_dir"
        ],
    )


print("")
print(
    "============================================"
)

print(
    "FORCE MODEL EXPORT COMPLETE"
)

print(
    "============================================"
)

print(
    EXPORT_DIR
)