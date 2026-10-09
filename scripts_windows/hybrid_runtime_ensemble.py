from __future__ import print_function

import os
import math
import numpy as np

from hybrid_runtime import (
    BASE_DIR,
    load_interface,
    reconstruct_interface_displacement,
    run_patch_job,
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "no_models_ensemble",
)

MODEL_CACHE = {}


def load_ensemble_model(side):

    if side not in MODEL_CACHE:

        path = os.path.join(
            MODEL_DIR,
            "{}_ensemble_force_model.npz".format(side),
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        MODEL_CACHE[side] = np.load(path)

    return MODEL_CACHE[side]


def gelu(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    result = np.empty_like(x)

    flat_x = x.reshape(-1)
    flat_result = result.reshape(-1)

    root_two = math.sqrt(2.0)

    for i in range(len(flat_x)):

        value = flat_x[i]

        flat_result[i] = (
            0.5
            * value
            * (
                1.0
                +
                math.erf(
                    value / root_two
                )
            )
        )

    return result


def predict_member_coefficients(
    model,
    member_index,
    branch_physical,
):

    prefix = "member_{}".format(
        member_index
    )

    branch_mean = model[
        "{}_branch_mean".format(prefix)
    ]

    branch_std = model[
        "{}_branch_std".format(prefix)
    ]

    coefficient_mean = model[
        "{}_force_coeff_mean".format(prefix)
    ]

    coefficient_std = model[
        "{}_force_coeff_std".format(prefix)
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
        model[
            "{}_number_layers".format(prefix)
        ][0]
    )

    for layer_index in range(
        number_layers
    ):

        W = model[
            "{}_W{}".format(
                prefix,
                layer_index,
            )
        ]

        b = model[
            "{}_b{}".format(
                prefix,
                layer_index,
            )
        ]

        x = W @ x + b

        if layer_index < number_layers - 1:
            x = gelu(x)

    return (
        x
        * coefficient_std
        +
        coefficient_mean
    )


def coefficients_to_generalized_force(
    model,
    coefficients,
):

    n1 = int(
        model[
            "number_force_pca_1"
        ][0]
    )

    n2 = int(
        model[
            "number_force_pca_2"
        ][0]
    )

    c1 = coefficients[
        :n1
    ]

    c2 = coefficients[
        n1:
        n1 + n2
    ]

    g1 = (
        model["g_mean_1"]
        +
        model["g_matrix_1"] @ c1
    )

    g2 = (
        model["g_mean_2"]
        +
        model["g_matrix_2"] @ c2
    )

    return np.concatenate(
        [
            g1,
            g2,
        ]
    )


def predict_ensemble_generalized_force(
    side,
    branch_physical,
):

    model = load_ensemble_model(
        side
    )

    number_members = int(
        model[
            "number_members"
        ][0]
    )

    predictions = []

    for member_index in range(
        number_members
    ):

        coefficients = (
            predict_member_coefficients(
                model,
                member_index,
                branch_physical,
            )
        )

        g = (
            coefficients_to_generalized_force(
                model,
                coefficients,
            )
        )

        predictions.append(g)

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    return np.mean(
        predictions,
        axis=0,
    )


def evaluate_neural_operators(
    E1,
    E2,
    G12,
    coefficient_dictionary,
):

    left_branch = np.concatenate(
        [
            np.asarray(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary["m6"],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary["m2"],
                dtype=np.float64,
            ),
        ]
    )

    left_force = (
        predict_ensemble_generalized_force(
            "left",
            left_branch,
        )
    )

    left_model = load_ensemble_model(
        "left"
    )

    left_count = int(
        left_model[
            "number_g_1"
        ][0]
    )

    right_branch = np.concatenate(
        [
            np.asarray(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary["p2"],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary["p6"],
                dtype=np.float64,
            ),
        ]
    )

    right_force = (
        predict_ensemble_generalized_force(
            "right",
            right_branch,
        )
    )

    right_model = load_ensemble_model(
        "right"
    )

    right_count = int(
        right_model[
            "number_g_1"
        ][0]
    )

    return {
        "m6":
            left_force[
                :left_count
            ],

        "m2":
            left_force[
                left_count:
            ],

        "p2":
            right_force[
                :right_count
            ],

        "p6":
            right_force[
                right_count:
            ],
    }


def get_residual_force_scales():

    left = load_ensemble_model(
        "left"
    )

    right = load_ensemble_model(
        "right"
    )

    left_std = np.abs(
        left[
            "g_std"
        ]
    )

    right_std = np.abs(
        right[
            "g_std"
        ]
    )

    left_count = int(
        left[
            "number_g_1"
        ][0]
    )

    right_count = int(
        right[
            "number_g_1"
        ][0]
    )

    scales = {
        "m6":
            left_std[
                :left_count
            ],

        "m2":
            left_std[
                left_count:
            ],

        "p2":
            right_std[
                :right_count
            ],

        "p6":
            right_std[
                right_count:
            ],
    }

    for name in scales:

        scales[name] = np.maximum(
            scales[name],
            1.0,
        )

    return scales