import os
import json

import numpy as np
import pandas as pd

from src.step40_ensemble_utils import (
    predict_ensemble_displacement,
)


# ============================================================
# PATHS
# ============================================================

ONLINE_DIR = os.path.join(
    "data",
    "hybrid_online_ensemble",
)


VALIDATION_DIR = os.path.join(
    ONLINE_DIR,
    "validation_fields",
)


SOLUTION_FILE = os.path.join(
    ONLINE_DIR,
    "online_solution.npz",
)


OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_online_ensemble_validation",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD ENSEMBLE-COUPLED SOLUTION
# ============================================================

solution = np.load(
    SOLUTION_FILE
)


E1 = float(
    solution["E1"][0]
)

E2 = float(
    solution["E2"][0]
)

G12 = float(
    solution["G12"][0]
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


print("")
print(
    "Material:"
)

print(
    E1,
    E2,
    G12
)


# ============================================================
# LOCAL NO COORDINATES
# ============================================================

left_coordinates = np.load(
    "data/hybrid_bulk_left/coordinates.npy"
).astype(
    np.float32
)


right_coordinates = np.load(
    "data/hybrid_bulk_right/coordinates.npy"
).astype(
    np.float32
)


# ============================================================
# BRANCH INPUTS
# ============================================================

left_branch = np.concatenate(
    [

        np.asarray(
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


right_branch = np.concatenate(
    [

        np.asarray(
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


# ============================================================
# STEP-40 ENSEMBLE DISPLACEMENTS
# ============================================================

(
    left_U,
    left_U_std,
    _,
) = predict_ensemble_displacement(

    side="left",

    coordinates=
        left_coordinates,

    branch_physical=
        left_branch,
)


(
    right_U,
    right_U_std,
    _,
) = predict_ensemble_displacement(

    side="right",

    coordinates=
        right_coordinates,

    branch_physical=
        right_branch,
)


# ============================================================
# LOAD FE PATCHES + FULL FEM REFERENCE
# ============================================================

left_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "left_nodes.csv",
    )
)


center_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "center_nodes.csv",
    )
)


right_fe = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "right_nodes.csv",
    )
)


reference = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "full_reference_nodes.csv",
    )
)


# ============================================================
# COORDINATE KEY
# ============================================================

def coordinate_key(
    x,
    y,
    z,
):

    return (

        round(
            float(x),
            5,
        ),

        round(
            float(y),
            5,
        ),

        round(
            float(z),
            5,
        ),
    )


# ============================================================
# ASSEMBLE GLOBAL HYBRID FIELD
# ============================================================

hybrid = {}


def add_fe_dataframe(
    dataframe,
):

    for _, row in dataframe.iterrows():

        hybrid[
            coordinate_key(
                row["X"],
                row["Y"],
                row["Z"],
            )
        ] = np.asarray(
            [
                row["U1"],
                row["U2"],
                row["U3"],
            ],
            dtype=np.float64,
        )


add_fe_dataframe(
    left_fe
)

add_fe_dataframe(
    center_fe
)

add_fe_dataframe(
    right_fe
)


# ============================================================
# LEFT NO STRICT INTERIOR
# ============================================================

for index in range(
    len(
        left_coordinates
    )
):

    xyz = left_coordinates[
        index
    ]


    x = float(
        xyz[0]
    )


    if (
        x > -6.0 + 1.0e-5

        and

        x < -1.8 - 1.0e-5
    ):

        hybrid[
            coordinate_key(
                xyz[0],
                xyz[1],
                xyz[2],
            )
        ] = left_U[
            index
        ].astype(
            np.float64
        )


# ============================================================
# RIGHT NO STRICT INTERIOR
# ============================================================

for index in range(
    len(
        right_coordinates
    )
):

    xyz = right_coordinates[
        index
    ]


    x = float(
        xyz[0]
    )


    if (
        x > 1.8 + 1.0e-5

        and

        x < 6.0 - 1.0e-5
    ):

        hybrid[
            coordinate_key(
                xyz[0],
                xyz[1],
                xyz[2],
            )
        ] = right_U[
            index
        ].astype(
            np.float64
        )


# ============================================================
# MAP TO FULL FEM NODE ORDER
# ============================================================

hybrid_U = []

reference_U = []

coordinates_full = []

missing = []


for _, row in reference.iterrows():

    key = coordinate_key(
        row["X"],
        row["Y"],
        row["Z"],
    )


    if key not in hybrid:

        missing.append(
            key
        )

        continue


    hybrid_U.append(
        hybrid[
            key
        ]
    )


    reference_U.append(
        [
            row["U1"],
            row["U2"],
            row["U3"],
        ]
    )


    coordinates_full.append(
        [
            row["X"],
            row["Y"],
            row["Z"],
        ]
    )


if len(
    missing
) > 0:

    print(
        "Missing nodes:",
        len(
            missing
        )
    )


    print(
        missing[:10]
    )


    raise RuntimeError(
        "Complete hybrid field was not reconstructed."
    )


hybrid_U = np.asarray(
    hybrid_U,
    dtype=np.float64,
)


reference_U = np.asarray(
    reference_U,
    dtype=np.float64,
)


coordinates_full = np.asarray(
    coordinates_full,
    dtype=np.float64,
)


# ============================================================
# METRICS
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
            error ** 2
        )
    )


    mae = np.mean(
        np.abs(
            error
        )
    )


    ss_res = np.sum(
        error ** 2
    )


    ss_tot = np.sum(
        (
            truth
            -
            truth.mean()
        ) ** 2
    )


    if ss_tot > 1.0e-14:

        r2 = (
            1.0
            -
            ss_res
            /
            ss_tot
        )

    else:

        r2 = np.nan


    return (
        relative_l2,
        rmse,
        mae,
        r2,
    )


rows = []


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
        r2,
    ) = calculate_metrics(

        hybrid_U[
            :,
            component_index
        ],

        reference_U[
            :,
            component_index
        ],
    )


    rows.append(
        {

            "Component":
                component_name,

            "Relative_L2":
                relative_l2,

            "Relative_L2_percent":
                100.0
                *
                relative_l2,

            "RMSE_mm":
                rmse,

            "MAE_mm":
                mae,

            "R2":
                r2,
        }
    )


metrics = pd.DataFrame(
    rows
)


# ============================================================
# REACTION
# ============================================================

with open(
    os.path.join(
        VALIDATION_DIR,
        "reaction_summary.json",
    ),
    "r",
) as file_object:

    ensemble_reaction = json.load(
        file_object
    )


# ============================================================
# COMPARE AGAINST ORIGINAL SINGLE-NO HYBRID
# ============================================================

old_metrics_file = os.path.join(
    "results",
    "hybrid_online_validation",
    "hybrid_displacement_metrics.csv",
)


comparison_rows = []


if os.path.isfile(
    old_metrics_file
):

    old_metrics = pd.read_csv(
        old_metrics_file
    )


    for _, new_row in metrics.iterrows():

        component = new_row[
            "Component"
        ]


        old_row = old_metrics[
            old_metrics[
                "Component"
            ]
            ==
            component
        ].iloc[
            0
        ]


        old_error = (
            100.0
            *
            float(
                old_row[
                    "Relative_L2"
                ]
            )
        )


        new_error = float(
            new_row[
                "Relative_L2_percent"
            ]
        )


        comparison_rows.append(
            {

                "Component":
                    component,

                "OriginalSingleNO_percent":
                    old_error,

                "Step40EnsembleHybrid_percent":
                    new_error,

                "AbsoluteImprovement_percentage_points":
                    old_error
                    -
                    new_error,

                "RelativeImprovement_percent":
                    100.0
                    *
                    (
                        old_error
                        -
                        new_error
                    )
                    /
                    old_error,
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


# ============================================================
# OLD REACTION IF AVAILABLE
# ============================================================

old_reaction_file = os.path.join(
    "data",
    "hybrid_online_baseline",
    "validation_fields",
    "reaction_summary.json",
)


reaction_comparison = {}


if os.path.isfile(
    old_reaction_file
):

    with open(
        old_reaction_file,
        "r",
    ) as file_object:

        old_reaction = json.load(
            file_object
        )


    reaction_comparison = {

        "original_singleNO_relative_reaction_error_percent":

            100.0
            *
            float(
                old_reaction[
                    "relative_reaction_error"
                ]
            ),

        "ensemble_hybrid_relative_reaction_error_percent":

            100.0
            *
            float(
                ensemble_reaction[
                    "relative_reaction_error"
                ]
            ),
    }


# ============================================================
# SAVE
# ============================================================

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_hybrid_displacement_metrics.csv",
    ),
    index=False,
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "single_vs_ensemble_hybrid.csv",
    ),
    index=False,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_hybrid_U.npy",
    ),
    hybrid_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "reference_U.npy",
    ),
    reference_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "coordinates.npy",
    ),
    coordinates_full,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_reaction_summary.json",
    ),
    "w",
) as file_object:

    json.dump(
        ensemble_reaction,
        file_object,
        indent=4,
    )


if len(
    reaction_comparison
) > 0:

    with open(
        os.path.join(
            OUTPUT_DIR,
            "reaction_comparison.json",
        ),
        "w",
    ) as file_object:

        json.dump(
            reaction_comparison,
            file_object,
            indent=4,
        )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "================================================"
)

print(
    "STEP 60 — ENSEMBLE HYBRID vs FULL FEM"
)

print(
    "================================================"
)


print("")
print(
    metrics.to_string(
        index=False
    )
)


print("")
print(
    "ORIGINAL HYBRID vs ENSEMBLE HYBRID"
)


print(
    comparison.to_string(
        index=False
    )
)


print("")
print(
    "Ensemble nose reaction error = {:.6f}%"
    .format(

        100.0

        *

        float(
            ensemble_reaction[
                "relative_reaction_error"
            ]
        )
    )
)


print("")
print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)