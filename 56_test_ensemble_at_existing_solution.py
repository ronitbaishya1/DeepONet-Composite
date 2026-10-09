import os

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
    "hybrid_online_baseline",
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
    "step56_ensemble_existing_interfaces",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD EXISTING CONVERGED INTERFACE SOLUTION
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
    "E1 =",
    E1
)

print(
    "E2 =",
    E2
)

print(
    "G12 =",
    G12
)


# ============================================================
# LOAD LOCAL COORDINATES
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


# ============================================================
# STEP-40 ENSEMBLE PREDICTIONS
# ============================================================

print("")
print(
    "Predicting LEFT ensemble..."
)


(
    left_U,
    left_U_std,
    left_members,
) = predict_ensemble_displacement(

    side="left",

    coordinates=
        left_coordinates,

    branch_physical=
        left_branch,
)


print(
    "Predicting RIGHT ensemble..."
)


(
    right_U,
    right_U_std,
    right_members,
) = predict_ensemble_displacement(

    side="right",

    coordinates=
        right_coordinates,

    branch_physical=
        right_branch,
)


# ============================================================
# LOAD EXISTING FE PATCH RESULTS
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
# COORDINATE MATCHING
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
# ASSEMBLE HYBRID FIELD
#
# SAME RULE AS STEP 30:
#
# FEM OWNS THE INTERFACE PLANES.
# NO ONLY OWNS STRICT INTERIORS.
# ============================================================

hybrid = {}


def add_fe_dataframe(
    dataframe,
):

    for _, row in dataframe.iterrows():

        key = coordinate_key(
            row["X"],
            row["Y"],
            row["Z"],
        )


        hybrid[
            key
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
# MAP TO FULL REFERENCE
# ============================================================

hybrid_U = []

reference_U = []

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
        "First missing nodes:",
        missing[:10]
    )

    raise RuntimeError(
        "Hybrid field does not cover reference mesh."
    )


hybrid_U = np.asarray(
    hybrid_U,
    dtype=np.float64,
)


reference_U = np.asarray(
    reference_U,
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
# LOAD OLD SINGLE-NO RESULTS
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


        old_percent = (
            100.0
            *
            float(
                old_row[
                    "Relative_L2"
                ]
            )
        )


        new_percent = float(
            new_row[
                "Relative_L2_percent"
            ]
        )


        comparison_rows.append(
            {

                "Component":
                    component,

                "Old_SingleNO_percent":
                    old_percent,

                "Step40_Ensemble_percent":
                    new_percent,

                "AbsoluteImprovement_percentage_points":
                    old_percent
                    -
                    new_percent,

                "RelativeImprovement_percent":
                    100.0
                    *
                    (
                        old_percent
                        -
                        new_percent
                    )
                    /
                    old_percent,
            }
        )


# ============================================================
# SAVE
# ============================================================

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ensemble_existing_interface_metrics.csv",
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
        "left_ensemble_std.npy",
    ),
    left_U_std,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "right_ensemble_std.npy",
    ),
    right_U_std,
)


if len(
    comparison_rows
) > 0:

    comparison = pd.DataFrame(
        comparison_rows
    )


    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "single_vs_ensemble_existing_interfaces.csv",
        ),
        index=False,
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "================================================"
)

print(
    "STEP 56 COMPLETE"
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


if len(
    comparison_rows
) > 0:

    print("")
    print(
        "SINGLE NO vs STEP-40 ENSEMBLE"
    )

    print(
        comparison.to_string(
            index=False
        )
    )


print("")
print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)