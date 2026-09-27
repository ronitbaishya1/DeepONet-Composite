import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = os.path.join(
    "results",
    "mechanics_validation",
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "mechanics_metrics.csv",
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "mechanics_method_comparison.csv",
)


# ============================================================
# LOAD
# ============================================================

metrics = pd.read_csv(
    INPUT_FILE
)


print("")
print("==============================================")
print("MECHANICS VALIDATION REVIEW")
print("==============================================")
print("")


# ============================================================
# MEAN SUMMARY
# ============================================================

summary = (
    metrics
    .groupby(
        [
            "Quantity",
            "Method",
        ]
    )
    [
        [
            "Relative_L2",
            "RMSE",
            "MAE",
            "MaxAbsError",
        ]
    ]
    .agg(
        [
            "mean",
            "median",
        ]
    )
)


print(summary.to_string())


# ============================================================
# STRAIN COMPARISON:
# INFINITESIMAL VS LEFT HENCKY
# ============================================================

strain_metrics = metrics[
    metrics[
        "Quantity"
    ].str.startswith(
        "LE"
    )
].copy()


mean_strain = (
    strain_metrics
    .groupby(
        [
            "Quantity",
            "Method",
        ]
    )[
        "Relative_L2"
    ]
    .mean()
    .reset_index()
)


pivot = mean_strain.pivot(
    index="Quantity",
    columns="Method",
    values="Relative_L2",
)


if (
    "Infinitesimal"
    in pivot.columns
    and
    "LeftHencky"
    in pivot.columns
):

    pivot[
        "Hencky_to_Infinitesimal_Ratio"
    ] = (
        pivot[
            "LeftHencky"
        ]
        /
        (
            pivot[
                "Infinitesimal"
            ]
            + 1.0e-14
        )
    )


    pivot[
        "Lower_Error_Method"
    ] = np.where(

        pivot[
            "LeftHencky"
        ]
        <
        pivot[
            "Infinitesimal"
        ],

        "LeftHencky",

        "Infinitesimal",

    )


pivot.to_csv(
    OUTPUT_FILE
)


print("")
print("==============================================")
print("STRAIN METHOD COMPARISON")
print("==============================================")
print("")

print(
    pivot.to_string()
)


# ============================================================
# IMPORTANT COMPONENTS
# ============================================================

important_quantities = [
    "LE11",
    "LE12",
    "S11",
    "S12",
]


important = metrics[
    metrics[
        "Quantity"
    ].isin(
        important_quantities
    )
]


important_summary = (
    important
    .groupby(
        [
            "Quantity",
            "Method",
        ]
    )[
        [
            "Relative_L2",
            "RMSE",
            "MAE",
        ]
    ]
    .mean()
    .reset_index()
)


print("")
print("==============================================")
print("KEY COMPONENTS")
print("==============================================")
print("")

print(
    important_summary.to_string(
        index=False
    )
)


important_summary.to_csv(

    os.path.join(
        RESULTS_DIR,
        "key_component_summary.csv",
    ),

    index=False,

)


print("")
print("Saved:")
print(OUTPUT_FILE)

print("")
print(
    "Do not select the physics formulation "
    "using one component alone."
)

print(
    "Compare LE11, LE12 and the other "
    "strain components together, and inspect "
    "the spatial fields."
)