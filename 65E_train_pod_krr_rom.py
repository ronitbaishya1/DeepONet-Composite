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
    "--patch",
    choices=[
        "left",
        "center",
        "right",
        "all",
    ],
    default="all",
)


parser.add_argument(
    "--energy",
    type=float,
    default=0.99999,
)


parser.add_argument(
    "--max_modes",
    type=int,
    default=40,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DESIGN_DIR = os.path.join(
    "data",
    "fe_rom_designs_7region",
)


SNAPSHOT_ROOT = os.path.join(
    "data",
    "rom_snapshots_7region",
)


OUTPUT_ROOT = os.path.join(
    "results",
    "fe_pod_krr_rom_7region",
)


os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True,
)


PATCHES = (
    [
        "left",
        "center",
        "right",
    ]

    if args.patch == "all"

    else
    [
        args.patch
    ]
)


# ============================================================
# GRID
# ============================================================

GAMMA_VALUES = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
]


LAMBDA_VALUES = [
    1.0e-8,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
]


# ============================================================
# HELPERS
# ============================================================

def safe_std(
    values,
    axis=0,
):

    std = np.std(
        values,
        axis=axis,
        ddof=1,
    )


    return np.maximum(
        std,
        1.0e-10,
    )


def rbf_kernel(
    X1,
    X2,
    gamma,
):

    difference = (

        X1[
            :,
            None,
            :
        ]

        -

        X2[
            None,
            :,
            :
        ]
    )


    mean_squared_distance = np.mean(
        difference ** 2,
        axis=2,
    )


    return np.exp(
        -gamma
        *
        mean_squared_distance
    )


def mean_case_relative_l2(
    prediction,
    truth,
):

    values = []


    for index in range(
        prediction.shape[
            0
        ]
    ):

        values.append(

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
            values
        )
    )


def mean_absolute_percentage_scalar(
    prediction,
    truth,
):

    return float(
        np.mean(

            100.0

            *

            np.abs(
                prediction
                -
                truth
            )

            /

            (
                np.abs(
                    truth
                )
                +
                1.0e-14
            )
        )
    )


# ============================================================
# TRAIN ONE PATCH
# ============================================================

def train_patch(
    patch,
):

    print("")
    print(
        "=========================================="
    )

    print(
        "POD-KRR ROM:",
        patch.upper()
    )

    print(
        "=========================================="
    )


    output_dir = os.path.join(
        OUTPUT_ROOT,
        patch,
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    design = pd.read_csv(
        os.path.join(
            DESIGN_DIR,
            "{}_fe_rom_design.csv".format(
                patch
            ),
        )
    )


    snapshot_dir = os.path.join(
        SNAPSHOT_ROOT,
        patch,
    )


    U = np.load(
        os.path.join(
            snapshot_dir,
            "U_snapshots.npy",
        )
    ).astype(
        np.float64
    )


    g = np.load(
        os.path.join(
            snapshot_dir,
            "g_snapshots.npy",
        )
    ).astype(
        np.float64
    )


    nose_rf2 = np.load(
        os.path.join(
            snapshot_dir,
            "nose_rf2.npy",
        )
    ).astype(
        np.float64
    )


    node_coordinates = np.load(
        os.path.join(
            snapshot_dir,
            "node_coordinates.npy",
        )
    ).astype(
        np.float64
    )


    if len(
        design
    ) != U.shape[
        0
    ]:

        raise RuntimeError(
            "Design/snapshot count mismatch."
        )


    # ========================================================
    # INPUT COLUMNS
    # ========================================================

    coefficient_columns = [

        column

        for column
        in design.columns

        if column.startswith(
            "c_"
        )
    ]


    branch_columns = [

        "E1_MPa",
        "E2_MPa",
        "G12_MPa",

    ] + coefficient_columns


    X = design[
        branch_columns
    ].to_numpy(
        dtype=np.float64
    )


    split = (
        design[
            "Split"
        ]
        .astype(
            str
        )
        .str.lower()
        .to_numpy()
    )


    train_indices = np.where(
        split
        ==
        "train"
    )[0]


    validation_indices = np.where(
        split
        ==
        "validation"
    )[0]


    test_indices = np.where(
        split
        ==
        "test"
    )[0]


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
        "Test:",
        len(
            test_indices
        )
    )


    # ========================================================
    # POD — TRAIN ONLY
    # ========================================================

    U_flat = U.reshape(
        U.shape[
            0
        ],
        -1,
    )


    U_train = U_flat[
        train_indices
    ]


    U_mean = U_train.mean(
        axis=0
    )


    centered = (
        U_train
        -
        U_mean
    )


    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )


    energy = singular_values ** 2


    cumulative_energy = (
        np.cumsum(
            energy
        )
        /
        np.sum(
            energy
        )
    )


    number_modes = (
        np.searchsorted(
            cumulative_energy,
            args.energy,
        )
        +
        1
    )


    number_modes = max(
        3,
        number_modes,
    )


    number_modes = min(
        number_modes,
        args.max_modes,
        Vt.shape[
            0
        ],
    )


    pod_basis = Vt[
        :number_modes
    ].T


    retained_energy = float(
        cumulative_energy[
            number_modes - 1
        ]
    )


    all_pod_coefficients = (

        (
            U_flat
            -
            U_mean.reshape(
                1,
                -1
            )
        )

        @

        pod_basis
    )


    print(
        "POD modes:",
        number_modes
    )

    print(
        "Retained energy:",
        retained_energy
    )


    # ========================================================
    # OUTPUT VECTOR
    #
    # [POD coefficients, generalized forces, optional nose RF2]
    # ========================================================

    output_parts = [

        all_pod_coefficients,

        g,
    ]


    has_reaction = (
        patch
        ==
        "center"
    )


    if has_reaction:

        output_parts.append(
            nose_rf2.reshape(
                -1,
                1
            )
        )


    Y = np.concatenate(
        output_parts,
        axis=1,
    )


    g_start = number_modes

    g_end = (
        g_start
        +
        g.shape[
            1
        ]
    )


    reaction_index = (
        g_end

        if has_reaction

        else
        -1
    )


    # ========================================================
    # NORMALIZE TRAIN ONLY
    # ========================================================

    X_mean = X[
        train_indices
    ].mean(
        axis=0
    )


    X_std = safe_std(
        X[
            train_indices
        ],
        axis=0,
    )


    Y_mean = Y[
        train_indices
    ].mean(
        axis=0
    )


    Y_std = safe_std(
        Y[
            train_indices
        ],
        axis=0,
    )


    Xn = (
        X
        -
        X_mean
    ) / X_std


    Yn = (
        Y
        -
        Y_mean
    ) / Y_std


    # ========================================================
    # VALIDATION SEARCH
    # ========================================================

    grid_rows = []


    best = None


    X_train = Xn[
        train_indices
    ]


    Y_train = Yn[
        train_indices
    ]


    X_validation = Xn[
        validation_indices
    ]


    for gamma in GAMMA_VALUES:

        K_train = rbf_kernel(
            X_train,
            X_train,
            gamma,
        )


        K_validation = rbf_kernel(
            X_validation,
            X_train,
            gamma,
        )


        for regularization in LAMBDA_VALUES:

            matrix = (

                K_train

                +

                regularization

                *

                np.eye(
                    K_train.shape[
                        0
                    ]
                )
            )


            alpha = np.linalg.solve(
                matrix,
                Y_train,
            )


            prediction_n = (
                K_validation
                @
                alpha
            )


            prediction = (

                prediction_n
                *
                Y_std
                +
                Y_mean
            )


            predicted_coefficients = (
                prediction[
                    :,
                    :number_modes
                ]
            )


            predicted_U = (

                U_mean.reshape(
                    1,
                    -1
                )

                +

                predicted_coefficients

                @

                pod_basis.T
            )


            true_U = U_flat[
                validation_indices
            ]


            predicted_g = prediction[
                :,
                g_start:
                g_end
            ]


            true_g = g[
                validation_indices
            ]


            U_error = mean_case_relative_l2(
                predicted_U,
                true_U,
            )


            g_error = mean_case_relative_l2(
                predicted_g,
                true_g,
            )


            if has_reaction:

                predicted_reaction = prediction[
                    :,
                    reaction_index
                ]


                true_reaction = nose_rf2[
                    validation_indices
                ]


                reaction_error = (
                    mean_absolute_percentage_scalar(
                        predicted_reaction,
                        true_reaction,
                    )
                )


                score = (

                    0.65
                    *
                    g_error

                    +

                    0.25
                    *
                    U_error

                    +

                    0.10
                    *
                    reaction_error
                )


            else:

                reaction_error = np.nan


                score = (

                    0.75
                    *
                    g_error

                    +

                    0.25
                    *
                    U_error
                )


            row = {

                "Gamma":
                    gamma,

                "Lambda":
                    regularization,

                "Validation_U_percent":
                    U_error,

                "Validation_g_percent":
                    g_error,

                "Validation_Reaction_percent":
                    reaction_error,

                "Score":
                    score,
            }


            grid_rows.append(
                row
            )


            if (
                best is None
                or
                score
                <
                best[
                    "Score"
                ]
            ):

                best = row.copy()


    grid_table = pd.DataFrame(
        grid_rows
    )


    grid_table.to_csv(
        os.path.join(
            output_dir,
            "hyperparameter_grid.csv",
        ),
        index=False,
    )


    print("")
    print(
        "Selected:"
    )

    print(
        best
    )


    # ========================================================
    # REFIT USING TRAIN + VALIDATION
    #
    # Test remains untouched.
    # ========================================================

    fit_indices = np.concatenate(
        [
            train_indices,
            validation_indices,
        ]
    )


    X_fit_mean = X[
        fit_indices
    ].mean(
        axis=0
    )


    X_fit_std = safe_std(
        X[
            fit_indices
        ],
        axis=0,
    )


    Y_fit_mean = Y[
        fit_indices
    ].mean(
        axis=0
    )


    Y_fit_std = safe_std(
        Y[
            fit_indices
        ],
        axis=0,
    )


    X_fit_n = (
        X[
            fit_indices
        ]
        -
        X_fit_mean
    ) / X_fit_std


    Y_fit_n = (
        Y[
            fit_indices
        ]
        -
        Y_fit_mean
    ) / Y_fit_std


    X_test_n = (
        X[
            test_indices
        ]
        -
        X_fit_mean
    ) / X_fit_std


    K_fit = rbf_kernel(

        X_fit_n,

        X_fit_n,

        best[
            "Gamma"
        ],
    )


    alpha_final = np.linalg.solve(

        K_fit

        +

        best[
            "Lambda"
        ]

        *

        np.eye(
            K_fit.shape[
                0
            ]
        ),

        Y_fit_n,
    )


    K_test = rbf_kernel(

        X_test_n,

        X_fit_n,

        best[
            "Gamma"
        ],
    )


    test_prediction_n = (

        K_test
        @
        alpha_final
    )


    test_prediction = (

        test_prediction_n
        *
        Y_fit_std
        +
        Y_fit_mean
    )


    test_coefficients = test_prediction[
        :,
        :number_modes
    ]


    test_U_prediction = (

        U_mean.reshape(
            1,
            -1
        )

        +

        test_coefficients

        @

        pod_basis.T
    )


    test_U_true = U_flat[
        test_indices
    ]


    test_g_prediction = test_prediction[
        :,
        g_start:
        g_end
    ]


    test_g_true = g[
        test_indices
    ]


    test_U_error = mean_case_relative_l2(
        test_U_prediction,
        test_U_true,
    )


    test_g_error = mean_case_relative_l2(
        test_g_prediction,
        test_g_true,
    )


    if has_reaction:

        test_reaction_prediction = (
            test_prediction[
                :,
                reaction_index
            ]
        )


        test_reaction_true = nose_rf2[
            test_indices
        ]


        test_reaction_error = (
            mean_absolute_percentage_scalar(
                test_reaction_prediction,
                test_reaction_true,
            )
        )


    else:

        test_reaction_error = np.nan


    # ========================================================
    # SAVE MODEL — PURE NUMPY FORMAT
    # ========================================================

    model_file = os.path.join(
        output_dir,
        "{}_pod_krr_rom.npz".format(
            patch
        ),
    )


    np.savez_compressed(

        model_file,

        fit_X_normalized=
            X_fit_n.astype(
                np.float64
            ),

        alpha=
            alpha_final.astype(
                np.float64
            ),

        input_mean=
            X_fit_mean.astype(
                np.float64
            ),

        input_std=
            X_fit_std.astype(
                np.float64
            ),

        output_mean=
            Y_fit_mean.astype(
                np.float64
            ),

        output_std=
            Y_fit_std.astype(
                np.float64
            ),

        pod_mean=
            U_mean.astype(
                np.float64
            ),

        pod_basis=
            pod_basis.astype(
                np.float64
            ),

        node_coordinates=
            node_coordinates.astype(
                np.float64
            ),

        gamma=
            np.asarray(
                [
                    best[
                        "Gamma"
                    ]
                ],
                dtype=np.float64,
            ),

        regularization=
            np.asarray(
                [
                    best[
                        "Lambda"
                    ]
                ],
                dtype=np.float64,
            ),

        number_pod_modes=
            np.asarray(
                [
                    number_modes
                ],
                dtype=np.int64,
            ),

        g_start=
            np.asarray(
                [
                    g_start
                ],
                dtype=np.int64,
            ),

        g_end=
            np.asarray(
                [
                    g_end
                ],
                dtype=np.int64,
            ),

        reaction_index=
            np.asarray(
                [
                    reaction_index
                ],
                dtype=np.int64,
            ),

        retained_energy=
            np.asarray(
                [
                    retained_energy
                ],
                dtype=np.float64,
            ),

        branch_columns=
            np.asarray(
                branch_columns,
                dtype="U128",
            ),
    )


    summary = {

        "Patch":
            patch,

        "PODModes":
            int(
                number_modes
            ),

        "RetainedEnergy":
            retained_energy,

        "SelectedGamma":
            float(
                best[
                    "Gamma"
                ]
            ),

        "SelectedLambda":
            float(
                best[
                    "Lambda"
                ]
            ),

        "Validation_U_percent":
            float(
                best[
                    "Validation_U_percent"
                ]
            ),

        "Validation_g_percent":
            float(
                best[
                    "Validation_g_percent"
                ]
            ),

        "Test_U_percent":
            test_U_error,

        "Test_g_percent":
            test_g_error,

        "Test_Reaction_percent":
            (
                None

                if np.isnan(
                    test_reaction_error
                )

                else
                test_reaction_error
            ),

        "ModelFile":
            model_file,
    }


    with open(
        os.path.join(
            output_dir,
            "rom_validation_summary.json",
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
        json.dumps(
            summary,
            indent=4,
        )
    )


    return summary


# ============================================================
# RUN
# ============================================================

summaries = []


for patch in PATCHES:

    summaries.append(
        train_patch(
            patch
        )
    )


pd.DataFrame(
    summaries
).to_csv(
    os.path.join(
        OUTPUT_ROOT,
        "all_patch_rom_summary.csv",
    ),
    index=False,
)


print("")
print(
    "=========================================="
)

print(
    "STEP 65E COMPLETE"
)

print(
    "=========================================="
)