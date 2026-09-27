import os
import json

import numpy as np
import pandas as pd

import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.hybrid_bulk_operator import (
    HybridBulkOperator
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.getcwd()


STUDY_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "tolerance_comparison",
)


REFERENCE_FILE = os.path.join(
    STUDY_DIR,
    "full_reference_nodes.csv",
)


LEFT_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_bulk_left",
)


LEFT_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid_bulk_left",
)


RIGHT_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_bulk_right",
)


RIGHT_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid_bulk_right",
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "tolerance_study",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# TOLERANCE CONFIGURATION
# ============================================================

RUNS = [

    {
        "name":
            "tol_0200",

        "tolerance":
            0.020,
    },

    {
        "name":
            "tol_0100",

        "tolerance":
            0.010,
    },

    {
        "name":
            "tol_0050",

        "tolerance":
            0.005,
    },
]


# ============================================================
# COORDINATE MATCHING
# ============================================================

def coordinate_key(
    x,
    y,
    z,
):

    return (

        round(
            float(
                x
            ),
            5,
        ),

        round(
            float(
                y
            ),
            5,
        ),

        round(
            float(
                z
            ),
            5,
        ),
    )


# ============================================================
# LOAD LOCAL OPERATOR
# ============================================================

def load_local_operator(
    data_dir,
    model_dir,
):

    with open(
        os.path.join(
            model_dir,
            "normalization.json",
        ),
        "r",
    ) as file_object:

        normalization = json.load(
            file_object
        )


    checkpoint = torch.load(
        os.path.join(
            model_dir,
            "best_hybrid_bulk_operator.pt",
        ),
        map_location=DEVICE,
    )


    model = HybridBulkOperator(

        branch_dim=
            checkpoint[
                "branch_dim"
            ],

        number_force_outputs=
            checkpoint[
                "number_force_outputs"
            ],

        latent_dim=
            checkpoint[
                "latent_dim"
            ],

        hidden_dim=
            checkpoint[
                "hidden_dim"
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


    coordinates = np.load(
        os.path.join(
            data_dir,
            "coordinates.npy",
        )
    ).astype(
        np.float32
    )


    return (
        model,
        normalization,
        coordinates,
    )


# ============================================================
# LOCAL NO PREDICTION
# ============================================================

def predict_local_displacement(
    model,
    normalization,
    coordinates,
    branch_physical,
):

    branch_mean = np.asarray(
        normalization[
            "branch_mean"
        ],
        dtype=np.float32,
    )


    branch_std = np.asarray(
        normalization[
            "branch_std"
        ],
        dtype=np.float32,
    )


    coordinate_min = np.asarray(
        normalization[
            "coordinate_min"
        ],
        dtype=np.float32,
    )


    coordinate_max = np.asarray(
        normalization[
            "coordinate_max"
        ],
        dtype=np.float32,
    )


    output_mean = np.asarray(
        normalization[
            "output_mean"
        ],
        dtype=np.float32,
    )


    output_std = np.asarray(
        normalization[
            "output_std"
        ],
        dtype=np.float32,
    )


    branch_normalized = (
        branch_physical
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

        (
            U_normalized,
            _
        ) = model(

            torch.tensor(
                branch_normalized,
                dtype=torch.float32,
            ).unsqueeze(
                0
            ),

            torch.tensor(
                coordinates_normalized,
                dtype=torch.float32,
            ),
        )


    U = (

        U_normalized[
            0
        ]
        .cpu()
        .numpy()

        *
        output_std.reshape(
            1,
            3,
        )

        +
        output_mean.reshape(
            1,
            3,
        )
    )


    return U


# ============================================================
# LOAD MODELS ONCE
# ============================================================

(
    left_model,
    left_normalization,
    left_coordinates,
) = load_local_operator(

    LEFT_DATA_DIR,
    LEFT_MODEL_DIR,
)


(
    right_model,
    right_normalization,
    right_coordinates,
) = load_local_operator(

    RIGHT_DATA_DIR,
    RIGHT_MODEL_DIR,
)


# ============================================================
# LOAD REFERENCE
# ============================================================

reference = pd.read_csv(
    REFERENCE_FILE
)


reference_map = {}


for _, row in reference.iterrows():

    key = coordinate_key(

        row[
            "X"
        ],

        row[
            "Y"
        ],

        row[
            "Z"
        ],
    )


    reference_map[
        key
    ] = np.array(
        [
            row[
                "U1"
            ],
            row[
                "U2"
            ],
            row[
                "U3"
            ],
        ],
        dtype=np.float64,
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    prediction,
    truth,
):

    error = (
        prediction
        -
        truth
    )


    relative_l2 = (

        np.linalg.norm(
            error
        )

        /

        (
            np.linalg.norm(
                truth
            )
            +
            1.0e-14
        )
    )


    rmse = np.sqrt(
        np.mean(
            error
            ** 2
        )
    )


    mae = np.mean(
        np.abs(
            error
        )
    )


    return (
        relative_l2,
        rmse,
        mae,
    )


# ============================================================
# ASSEMBLE ONE TOLERANCE RUN
# ============================================================

def assemble_run(
    run_name,
):

    run_directory = os.path.join(
        STUDY_DIR,
        run_name,
    )


    solution = np.load(
        os.path.join(
            run_directory,
            "online_solution.npz",
        )
    )


    E1 = float(
        solution[
            "E1"
        ][
            0
        ]
    )


    E2 = float(
        solution[
            "E2"
        ][
            0
        ]
    )


    G12 = float(
        solution[
            "G12"
        ][
            0
        ]
    )


    c_m6 = solution[
        "c_m6"
    ].astype(
        np.float32
    )


    c_m2 = solution[
        "c_m2"
    ].astype(
        np.float32
    )


    c_p2 = solution[
        "c_p2"
    ].astype(
        np.float32
    )


    c_p6 = solution[
        "c_p6"
    ].astype(
        np.float32
    )


    # --------------------------------------------------------
    # LEFT NO
    # --------------------------------------------------------

    left_branch = np.concatenate(
        [

            np.array(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float32,
            ),

            c_m6,

            c_m2,
        ]
    )


    left_U = predict_local_displacement(

        left_model,
        left_normalization,
        left_coordinates,
        left_branch,
    )


    # --------------------------------------------------------
    # RIGHT NO
    # --------------------------------------------------------

    right_branch = np.concatenate(
        [

            np.array(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float32,
            ),

            c_p2,

            c_p6,
        ]
    )


    right_U = predict_local_displacement(

        right_model,
        right_normalization,
        right_coordinates,
        right_branch,
    )


    # --------------------------------------------------------
    # FE PATCH CSVs
    # --------------------------------------------------------

    left_fe = pd.read_csv(
        os.path.join(
            run_directory,
            "left_nodes.csv",
        )
    )


    center_fe = pd.read_csv(
        os.path.join(
            run_directory,
            "center_nodes.csv",
        )
    )


    right_fe = pd.read_csv(
        os.path.join(
            run_directory,
            "right_nodes.csv",
        )
    )


    hybrid = {}


    def add_fe_dataframe(
        dataframe,
        owner,
    ):

        for _, row in dataframe.iterrows():

            key = coordinate_key(

                row[
                    "X"
                ],

                row[
                    "Y"
                ],

                row[
                    "Z"
                ],
            )


            hybrid[
                key
            ] = {

                "U":
                    np.array(
                        [
                            row[
                                "U1"
                            ],
                            row[
                                "U2"
                            ],
                            row[
                                "U3"
                            ],
                        ],
                        dtype=np.float64,
                    ),

                "Owner":
                    owner,
            }


    add_fe_dataframe(
        left_fe,
        "FE_left",
    )


    add_fe_dataframe(
        center_fe,
        "FE_center",
    )


    add_fe_dataframe(
        right_fe,
        "FE_right",
    )


    # --------------------------------------------------------
    # LEFT NO STRICT INTERIOR
    # --------------------------------------------------------

    for index in range(
        len(
            left_coordinates
        )
    ):

        x = float(
            left_coordinates[
                index,
                0
            ]
        )


        if (
            x > -6.0 + 1.0e-5
            and
            x < -1.8 - 1.0e-5
        ):

            key = coordinate_key(

                left_coordinates[
                    index,
                    0
                ],

                left_coordinates[
                    index,
                    1
                ],

                left_coordinates[
                    index,
                    2
                ],
            )


            hybrid[
                key
            ] = {

                "U":
                    left_U[
                        index
                    ].astype(
                        np.float64
                    ),

                "Owner":
                    "NO_left",
            }


    # --------------------------------------------------------
    # RIGHT NO STRICT INTERIOR
    # --------------------------------------------------------

    for index in range(
        len(
            right_coordinates
        )
    ):

        x = float(
            right_coordinates[
                index,
                0
            ]
        )


        if (
            x > 1.8 + 1.0e-5
            and
            x < 6.0 - 1.0e-5
        ):

            key = coordinate_key(

                right_coordinates[
                    index,
                    0
                ],

                right_coordinates[
                    index,
                    1
                ],

                right_coordinates[
                    index,
                    2
                ],
            )


            hybrid[
                key
            ] = {

                "U":
                    right_U[
                        index
                    ].astype(
                        np.float64
                    ),

                "Owner":
                    "NO_right",
            }


    return (
        hybrid,
        solution,
    )


# ============================================================
# PROCESS ALL RUNS
# ============================================================

summary_rows = []

region_rows = []

solution_q = {}


for run in RUNS:

    run_name = run[
        "name"
    ]


    tolerance = run[
        "tolerance"
    ]


    print("")
    print(
        "============================================"
    )

    print(
        "PROCESSING:",
        run_name,
    )

    print(
        "============================================"
    )


    hybrid, solution = assemble_run(
        run_name
    )


    prediction = []

    truth = []

    owner_list = []

    x_list = []


    for _, row in reference.iterrows():

        key = coordinate_key(

            row[
                "X"
            ],

            row[
                "Y"
            ],

            row[
                "Z"
            ],
        )


        if key not in hybrid:

            raise RuntimeError(
                "Hybrid field missing node {}".format(
                    key
                )
            )


        prediction.append(
            hybrid[
                key
            ][
                "U"
            ]
        )


        truth.append(
            np.array(
                [
                    row[
                        "U1"
                    ],
                    row[
                        "U2"
                    ],
                    row[
                        "U3"
                    ],
                ],
                dtype=np.float64,
            )
        )


        owner_list.append(
            hybrid[
                key
            ][
                "Owner"
            ]
        )


        x_list.append(
            float(
                row[
                    "X"
                ]
            )
        )


    prediction = np.asarray(
        prediction
    )


    truth = np.asarray(
        truth
    )


    x_array = np.asarray(
        x_list
    )


    owner_array = np.asarray(
        owner_list
    )


    # --------------------------------------------------------
    # GLOBAL DISPLACEMENT METRICS
    # --------------------------------------------------------

    component_results = {}


    for component_index, component_name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        (
            relative_l2,
            rmse,
            mae,
        ) = calculate_metrics(

            prediction[
                :,
                component_index
            ],

            truth[
                :,
                component_index
            ],
        )


        component_results[
            component_name
        ] = {

            "RelativeL2":
                relative_l2,

            "RMSE":
                rmse,

            "MAE":
                mae,
        }


    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    with open(
        os.path.join(
            STUDY_DIR,
            run_name,
            "solution_manifest.json",
        ),
        "r",
    ) as file_object:

        manifest = json.load(
            file_object
        )


    # --------------------------------------------------------
    # REACTION
    # --------------------------------------------------------

    reaction_relative_error = np.nan

    hybrid_rf2 = np.nan


    reaction_file = os.path.join(
        STUDY_DIR,
        run_name,
        "reaction_summary.json",
    )


    if os.path.isfile(
        reaction_file
    ):

        with open(
            reaction_file,
            "r",
        ) as file_object:

            reaction = json.load(
                file_object
            )


        hybrid_rf2 = reaction.get(
            "hybrid_center_nose_RF2_N",
            np.nan,
        )


        if "relative_reaction_error" in reaction:

            reaction_relative_error = float(
                reaction[
                    "relative_reaction_error"
                ]
            )


    # --------------------------------------------------------
    # SAVE Q
    # --------------------------------------------------------

    q = solution[
        "q"
    ].astype(
        np.float64
    )


    solution_q[
        run_name
    ] = q


    # --------------------------------------------------------
    # SUMMARY ROW
    # --------------------------------------------------------

    summary_rows.append(
        {

            "Run":
                run_name,

            "Tolerance":
                tolerance,

            "Converged":
                manifest[
                    "converged"
                ],

            "ResidualEvaluations":
                manifest[
                    "number_residual_evaluations"
                ],

            "BestRMSResidual":
                manifest[
                    "best_rms_residual"
                ],

            "BestMaxResidual":
                manifest[
                    "best_max_residual"
                ],

            "MaxAbsQ":
                np.max(
                    np.abs(
                        q
                    )
                ),

            "U1_RelL2":
                component_results[
                    "U1"
                ][
                    "RelativeL2"
                ],

            "U2_RelL2":
                component_results[
                    "U2"
                ][
                    "RelativeL2"
                ],

            "U3_RelL2":
                component_results[
                    "U3"
                ][
                    "RelativeL2"
                ],

            "U1_RMSE_mm":
                component_results[
                    "U1"
                ][
                    "RMSE"
                ],

            "U2_RMSE_mm":
                component_results[
                    "U2"
                ][
                    "RMSE"
                ],

            "U3_RMSE_mm":
                component_results[
                    "U3"
                ][
                    "RMSE"
                ],

            "HybridNoseRF2_N":
                hybrid_rf2,

            "ReactionRelError":
                reaction_relative_error,
        }
    )


    # --------------------------------------------------------
    # REGIONWISE U2 ERRORS
    # --------------------------------------------------------

    regions = {

        "FE_left":
            (
                x_array
                <=
                -6.0
                +
                1.0e-6
            ),

        "NO_left":
            (
                (
                    x_array
                    >
                    -6.0
                    +
                    1.0e-6
                )
                &
                (
                    x_array
                    <
                    -1.8
                    -
                    1.0e-6
                )
            ),

        "FE_center":
            (
                (
                    x_array
                    >=
                    -1.8
                    -
                    1.0e-6
                )
                &
                (
                    x_array
                    <=
                    1.8
                    +
                    1.0e-6
                )
            ),

        "NO_right":
            (
                (
                    x_array
                    >
                    1.8
                    +
                    1.0e-6
                )
                &
                (
                    x_array
                    <
                    6.0
                    -
                    1.0e-6
                )
            ),

        "FE_right":
            (
                x_array
                >=
                6.0
                -
                1.0e-6
            ),
    }


    for region_name, mask in regions.items():

        (
            region_relative_l2,
            region_rmse,
            region_mae,
        ) = calculate_metrics(

            prediction[
                mask,
                1
            ],

            truth[
                mask,
                1
            ],
        )


        region_rows.append(
            {

                "Run":
                    run_name,

                "Tolerance":
                    tolerance,

                "Region":
                    region_name,

                "Nodes":
                    int(
                        np.sum(
                            mask
                        )
                    ),

                "U2_RelL2":
                    region_relative_l2,

                "U2_RMSE_mm":
                    region_rmse,

                "U2_MAE_mm":
                    region_mae,
            }
        )


# ============================================================
# Q DIFFERENCES
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)


reference_q = solution_q[
    "tol_0200"
]


for run_name in [

    "tol_0200",
    "tol_0100",
    "tol_0050",
]:

    q_difference = (
        solution_q[
            run_name
        ]
        -
        reference_q
    )


    summary_dataframe.loc[
        summary_dataframe[
            "Run"
        ]
        ==
        run_name,
        "Q_Diff_From_0200_L2",
    ] = np.linalg.norm(
        q_difference
    )


    summary_dataframe.loc[
        summary_dataframe[
            "Run"
        ]
        ==
        run_name,
        "Q_Diff_From_0200_MaxAbs",
    ] = np.max(
        np.abs(
            q_difference
        )
    )


# ============================================================
# SAVE TABLES
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "tolerance_summary.csv",
)


summary_dataframe.to_csv(
    summary_file,
    index=False,
)


region_dataframe = pd.DataFrame(
    region_rows
)


region_file = os.path.join(
    OUTPUT_DIR,
    "tolerance_region_u2.csv",
)


region_dataframe.to_csv(
    region_file,
    index=False,
)


# ============================================================
# PRINT MAIN TABLE
# ============================================================

print("")
print(
    "============================================================"
)

print(
    "TOLERANCE STUDY SUMMARY"
)

print(
    "============================================================"
)


display_table = summary_dataframe[
    [
        "Tolerance",
        "Converged",
        "ResidualEvaluations",
        "BestRMSResidual",
        "BestMaxResidual",
        "MaxAbsQ",
        "U1_RelL2",
        "U2_RelL2",
        "U3_RelL2",
        "ReactionRelError",
    ]
].copy()


display_table[
    "U1_RelL2"
] *= 100.0


display_table[
    "U2_RelL2"
] *= 100.0


display_table[
    "U3_RelL2"
] *= 100.0


display_table[
    "ReactionRelError"
] *= 100.0


display_table = display_table.rename(
    columns={
        "U1_RelL2":
            "U1_Error_percent",

        "U2_RelL2":
            "U2_Error_percent",

        "U3_RelL2":
            "U3_Error_percent",

        "ReactionRelError":
            "Reaction_Error_percent",
    }
)


print(
    display_table.to_string(
        index=False
    )
)


# ============================================================
# PRINT REGION TABLE
# ============================================================

print("")
print(
    "============================================================"
)

print(
    "REGIONWISE U2 ERROR"
)

print(
    "============================================================"
)


region_display = region_dataframe.copy()


region_display[
    "U2_RelL2"
] *= 100.0


region_display = region_display.rename(
    columns={
        "U2_RelL2":
            "U2_Error_percent",
    }
)


print(
    region_display.to_string(
        index=False
    )
)


# ============================================================
# PLOT 1:
# TOLERANCE vs GLOBAL U2 ERROR
# ============================================================

plot_dataframe = summary_dataframe.sort_values(
    "Tolerance",
    ascending=False,
)


figure, axis = plt.subplots(
    figsize=(
        8,
        5
    )
)


axis.plot(

    plot_dataframe[
        "Tolerance"
    ],

    100.0
    *
    plot_dataframe[
        "U2_RelL2"
    ],

    marker="o",
)


axis.set_xscale(
    "log"
)


axis.invert_xaxis()


axis.set_xlabel(
    "Broyden tolerance"
)


axis.set_ylabel(
    r"Global $U_2$ relative $L_2$ error (%)"
)


axis.set_title(
    "Does Tighter Interface Equilibrium Reduce Displacement Error?"
)


axis.grid(
    True,
    alpha=0.25,
)


figure.tight_layout()


figure.savefig(

    os.path.join(
        OUTPUT_DIR,
        "01_tolerance_vs_u2_error.png",
    ),

    dpi=300,
    bbox_inches="tight",
)


plt.close(
    figure
)


# ============================================================
# PLOT 2:
# COST vs U2 ERROR
# ============================================================

figure, axis = plt.subplots(
    figsize=(
        8,
        5
    )
)


axis.plot(

    summary_dataframe[
        "ResidualEvaluations"
    ],

    100.0
    *
    summary_dataframe[
        "U2_RelL2"
    ],

    marker="o",
)


for _, row in summary_dataframe.iterrows():

    axis.text(

        row[
            "ResidualEvaluations"
        ],

        100.0
        *
        row[
            "U2_RelL2"
        ],

        "tol={}".format(
            row[
                "Tolerance"
            ]
        ),

        fontsize=9,
    )


axis.set_xlabel(
    "Number of FE–NO residual evaluations"
)


axis.set_ylabel(
    r"Global $U_2$ relative $L_2$ error (%)"
)


axis.set_title(
    "Coupling Cost vs Hybrid Accuracy"
)


axis.grid(
    True,
    alpha=0.25,
)


figure.tight_layout()


figure.savefig(

    os.path.join(
        OUTPUT_DIR,
        "02_cost_vs_u2_error.png",
    ),

    dpi=300,
    bbox_inches="tight",
)


plt.close(
    figure
)


# ============================================================
# COMPLETE
# ============================================================

print("")
print(
    "Saved:"
)

print(
    summary_file
)

print(
    region_file
)


print("")
print(
    "Tolerance comparison complete."
)