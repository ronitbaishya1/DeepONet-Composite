from __future__ import print_function

import os

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "rom_models_7region",
)


MODELS = {}


# ============================================================
# LOAD MODEL
# ============================================================

def load_rom(
    patch_name
):

    if patch_name not in MODELS:

        filename = os.path.join(
            MODEL_DIR,
            "{}_pod_krr_rom.npz".format(
                patch_name
            ),
        )


        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                filename
            )


        MODELS[
            patch_name
        ] = np.load(
            filename
        )


    return MODELS[
        patch_name
    ]


# ============================================================
# RBF KERNEL
# ============================================================

def rbf_kernel_vector(
    x,
    X_train,
    gamma,
):

    difference = (

        X_train

        -

        x.reshape(
            1,
            -1
        )
    )


    mean_squared_distance = np.mean(
        difference ** 2,
        axis=1,
    )


    return np.exp(
        -gamma
        *
        mean_squared_distance
    )


# ============================================================
# PREDICT COMPLETE REDUCED OUTPUT
# ============================================================

def predict_rom_output(
    patch_name,
    branch_physical,
):

    model = load_rom(
        patch_name
    )


    branch_physical = np.asarray(
        branch_physical,
        dtype=np.float64,
    )


    input_mean = model[
        "input_mean"
    ]


    input_std = model[
        "input_std"
    ]


    x = (
        branch_physical
        -
        input_mean
    ) / input_std


    training_X = model[
        "fit_X_normalized"
    ]


    gamma = float(
        model[
            "gamma"
        ][0]
    )


    kernel = rbf_kernel_vector(
        x,
        training_X,
        gamma,
    )


    output_normalized = (

        kernel

        @

        model[
            "alpha"
        ]
    )


    output = (

        output_normalized

        *

        model[
            "output_std"
        ]

        +

        model[
            "output_mean"
        ]
    )


    return output


# ============================================================
# GENERALIZED FORCE ONLY
# ============================================================

def predict_rom_generalized_force(
    patch_name,
    branch_physical,
):

    model = load_rom(
        patch_name
    )


    output = predict_rom_output(
        patch_name,
        branch_physical,
    )


    g_start = int(
        model[
            "g_start"
        ][0]
    )


    g_end = int(
        model[
            "g_end"
        ][0]
    )


    return output[
        g_start:
        g_end
    ]


# ============================================================
# OPTIONAL DISPLACEMENT RECONSTRUCTION
# ============================================================

def predict_rom_displacement(
    patch_name,
    branch_physical,
):

    model = load_rom(
        patch_name
    )


    output = predict_rom_output(
        patch_name,
        branch_physical,
    )


    number_modes = int(
        model[
            "number_pod_modes"
        ][0]
    )


    coefficients = output[
        :number_modes
    ]


    flattened = (

        model[
            "pod_mean"
        ]

        +

        model[
            "pod_basis"
        ]

        @

        coefficients
    )


    coordinates = model[
        "node_coordinates"
    ]


    U = flattened.reshape(
        coordinates.shape[
            0
        ],
        3,
    )


    return (
        coordinates,
        U,
    )


# ============================================================
# CENTER REACTION
# ============================================================

def predict_rom_center_reaction(
    branch_physical,
):

    model = load_rom(
        "center"
    )


    reaction_index = int(
        model[
            "reaction_index"
        ][0]
    )


    if reaction_index < 0:

        raise RuntimeError(
            "Center ROM has no reaction target."
        )


    output = predict_rom_output(
        "center",
        branch_physical,
    )


    return float(
        output[
            reaction_index
        ]
    )