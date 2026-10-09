import os

import numpy as np
import torch

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(
            __file__
        )
    )
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# ENSEMBLE DIRECTORIES
# ============================================================

ENSEMBLE_DIRS = {

    "left":
        os.path.join(
            PROJECT_ROOT,
            "results",
            "adaptive_ensemble",
            "left",
        ),

    "right":
        os.path.join(
            PROJECT_ROOT,
            "results",
            "adaptive_ensemble",
            "right",
        ),
}


# ============================================================
# FIND FIVE CHECKPOINTS
# ============================================================

def get_member_checkpoint_files(
    side,
):

    if side not in [
        "left",
        "right",
    ]:

        raise ValueError(
            "side must be left or right"
        )


    ensemble_dir = ENSEMBLE_DIRS[
        side
    ]


    if not os.path.isdir(
        ensemble_dir
    ):

        raise FileNotFoundError(
            ensemble_dir
        )


    member_dirs = sorted(
        [
            name

            for name in os.listdir(
                ensemble_dir
            )

            if name.startswith(
                "member_"
            )
        ]
    )


    checkpoint_files = []


    for member_dir in member_dirs:

        checkpoint_file = os.path.join(
            ensemble_dir,
            member_dir,
            "best_v3_ensemble.pt",
        )


        if os.path.isfile(
            checkpoint_file
        ):

            checkpoint_files.append(
                checkpoint_file
            )


    if len(
        checkpoint_files
    ) != 5:

        raise RuntimeError(
            "{} ensemble should contain 5 checkpoints, "
            "but found {}.".format(
                side,
                len(
                    checkpoint_files
                ),
            )
        )


    return checkpoint_files


# ============================================================
# LOAD ONE MEMBER
# ============================================================

def load_member(
    checkpoint_file,
):

    checkpoint = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=False,
    )


    model = HybridBulkOperatorV3(

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


    return (
        model,
        checkpoint,
    )


# ============================================================
# PREDICT ONE MEMBER
# ============================================================

def predict_member_displacement(
    model,
    checkpoint,
    coordinates,
    branch_physical,
):

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


    branch_normalized = (

        np.asarray(
            branch_physical,
            dtype=np.float32,
        )

        -

        branch_mean

    ) / branch_std


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


    with torch.no_grad():

        U_normalized, _ = model(

            torch.tensor(
                branch_normalized,
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(
                1,
                -1,
            ),

            torch.tensor(
                coordinates_normalized,
                dtype=torch.float32,
                device=DEVICE,
            ),
        )


    U = (

        U_normalized[
            0
        ]
        .cpu()
        .numpy()

        *

        U_std.reshape(
            1,
            3,
        )

        +

        U_mean.reshape(
            1,
            3,
        )
    )


    return U


# ============================================================
# ENSEMBLE DISPLACEMENT
# ============================================================

def predict_ensemble_displacement(
    side,
    coordinates,
    branch_physical,
):

    checkpoint_files = (
        get_member_checkpoint_files(
            side
        )
    )


    predictions = []


    for checkpoint_file in checkpoint_files:

        model, checkpoint = load_member(
            checkpoint_file
        )


        prediction = (
            predict_member_displacement(
                model=
                    model,
                checkpoint=
                    checkpoint,
                coordinates=
                    coordinates,
                branch_physical=
                    branch_physical,
            )
        )


        predictions.append(
            prediction
        )


    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )


    ensemble_mean = np.mean(
        predictions,
        axis=0,
    )


    ensemble_std = np.std(
        predictions,
        axis=0,
        ddof=1,
    )


    return (
        ensemble_mean,
        ensemble_std,
        predictions,
    )