import os
import pandas as pd


# ============================================================
# MODEL EVALUATION DIRECTORIES
# ============================================================

MODELS = {

    "Baseline":
        "results/vector_deeponet_separate/"
        "evaluation_baseline",

    "MechanicsConsistent":
        "results/mechanics_consistent/"
        "eps_0.1_sig_0.1_frac_1/"
        "evaluation_mechanics",

    "PI_eq_0.01":
        "results/pi_mechanics_aware/"
        "eq_0.01/evaluation_pi",

    "PI_eq_0.1":
        "results/pi_mechanics_aware/"
        "eq_0.1/evaluation_pi",

    "PI_eq_1":
        "results/pi_mechanics_aware/"
        "eq_1/evaluation_pi",
}


OUTPUT_DIR = os.path.join(
    "results",
    "final_comparison",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_model(
    model_name,
    directory,
):

    displacement_file = os.path.join(
        directory,
        "displacement_summary.csv",
    )

    mechanics_file = os.path.join(
        directory,
        "mechanics_summary.csv",
    )

    equilibrium_file = os.path.join(
        directory,
        "equilibrium_summary.csv",
    )


    if not os.path.exists(
        displacement_file
    ):

        raise FileNotFoundError(
            displacement_file
        )


    if not os.path.exists(
        mechanics_file
    ):

        raise FileNotFoundError(
            mechanics_file
        )


    if not os.path.exists(
        equilibrium_file
    ):

        raise FileNotFoundError(
            equilibrium_file
        )


    displacement = pd.read_csv(
        displacement_file
    )


    mechanics = pd.read_csv(
        mechanics_file
    )


    equilibrium = pd.read_csv(
        equilibrium_file
    )


    row = {
        "Model":
            model_name,
    }


    # ========================================================
    # DISPLACEMENT
    # ========================================================

    for component in [
        "U1",
        "U2",
        "U3",
    ]:

        selected = displacement[
            displacement[
                "Component"
            ] == component
        ]


        if len(
            selected
        ) == 0:

            continue


        selected = selected.iloc[
            0
        ]


        row[
            component
            + "_RelL2"
        ] = selected[
            "Relative_L2"
        ]


        row[
            component
            + "_RMSE_mm"
        ] = selected[
            "RMSE_mm"
        ]


        row[
            component
            + "_R2"
        ] = selected[
            "R2"
        ]


    # ========================================================
    # KEY MECHANICS QUANTITIES
    # ========================================================

    for quantity in [
        "LE11",
        "LE12",
        "S11",
        "S12",
    ]:

        selected = mechanics[
            mechanics[
                "Quantity"
            ] == quantity
        ]


        if len(
            selected
        ) == 0:

            continue


        selected = selected.iloc[
            0
        ]


        row[
            quantity
            + "_RelL2"
        ] = selected[
            "Relative_L2"
        ]


        row[
            quantity
            + "_RMSE"
        ] = selected[
            "RMSE"
        ]


        row[
            quantity
            + "_R2"
        ] = selected[
            "R2"
        ]


    # ========================================================
    # EQUILIBRIUM
    # ========================================================

    row[
        "Equilibrium_Normalized_MSE"
    ] = equilibrium[
        "MeanNormalizedEquilibriumMSE"
    ].iloc[
        0
    ]


    row[
        "Equilibrium_RMS_MPa_per_mm"
    ] = equilibrium[
        "MeanEquilibriumRMS_MPa_per_mm"
    ].iloc[
        0
    ]


    return row


# ============================================================
# BUILD TABLE
# ============================================================

rows = []


for (
    model_name,
    directory,
) in MODELS.items():

    print(
        "Reading:",
        model_name
    )


    rows.append(
        load_model(
            model_name,
            directory,
        )
    )


comparison = pd.DataFrame(
    rows
)


# ============================================================
# SAVE FULL TABLE
# ============================================================

full_output = os.path.join(
    OUTPUT_DIR,
    "all_model_metrics.csv",
)


comparison.to_csv(
    full_output,
    index=False,
)


# ============================================================
# COMPACT PAPER-STYLE TABLE
# ============================================================

compact_columns = [

    "Model",

    "U1_RelL2",
    "U2_RelL2",
    "U3_RelL2",

    "LE11_RelL2",
    "LE12_RelL2",

    "S11_RelL2",
    "S12_RelL2",

    "Equilibrium_RMS_MPa_per_mm",

]


compact = comparison[
    compact_columns
].copy()


compact_output = os.path.join(
    OUTPUT_DIR,
    "model_comparison_compact.csv",
)


compact.to_csv(
    compact_output,
    index=False,
)


# ============================================================
# PERCENT FORMAT FOR DISPLAY
# ============================================================

display = compact.copy()


relative_columns = [

    "U1_RelL2",
    "U2_RelL2",
    "U3_RelL2",

    "LE11_RelL2",
    "LE12_RelL2",

    "S11_RelL2",
    "S12_RelL2",

]


for column in relative_columns:

    display[
        column
    ] = (
        100.0
        * display[
            column
        ]
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================================="
)

print(
    "MODEL COMPARISON"
)

print(
    "Relative L2 values below are shown as percentages."
)

print(
    "=========================================================="
)

print("")

print(
    display.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    full_output
)

print(
    compact_output
)