import os
import numpy as np
import pandas as pd


INPUT_FILE = os.path.join(
    "results",
    "final_comparison",
    "model_comparison_compact.csv",
)


OUTPUT_DIR = os.path.join(
    "results",
    "final_comparison",
)


comparison = pd.read_csv(
    INPUT_FILE
)


comparison = comparison.set_index(
    "Model"
)


# ============================================================
# MODELS
# ============================================================

BASELINE = (
    comparison.loc[
        "Baseline"
    ]
)


MECHANICS = (
    comparison.loc[
        "MechanicsConsistent"
    ]
)


PI = (
    comparison.loc[
        "PI_eq_0.01"
    ]
)


# ============================================================
# FUNCTION
#
# Positive value = improvement
# Negative value = degradation
# ============================================================

def percent_improvement(
    old,
    new,
):

    if abs(
        old
    ) < 1.0e-14:

        return np.nan


    return (
        (
            old
            - new
        )
        / abs(
            old
        )
        * 100.0
    )


# ============================================================
# QUANTITIES
# ============================================================

quantities = [

    "U1_RelL2",
    "U2_RelL2",
    "U3_RelL2",

    "LE11_RelL2",
    "LE12_RelL2",

    "S11_RelL2",
    "S12_RelL2",

    "Equilibrium_RMS_MPa_per_mm",

]


rows = []


for quantity in quantities:

    baseline_value = BASELINE[
        quantity
    ]


    mechanics_value = MECHANICS[
        quantity
    ]


    pi_value = PI[
        quantity
    ]


    rows.append(
        {

            "Quantity":
                quantity,

            "Baseline":
                baseline_value,

            "MechanicsConsistent":
                mechanics_value,

            "PI_eq_0.01":
                pi_value,

            "Baseline_to_Mechanics_pct":
                percent_improvement(
                    baseline_value,
                    mechanics_value,
                ),

            "Mechanics_to_PI_pct":
                percent_improvement(
                    mechanics_value,
                    pi_value,
                ),

            "Baseline_to_PI_pct":
                percent_improvement(
                    baseline_value,
                    pi_value,
                ),

        }
    )


result = pd.DataFrame(
    rows
)


output_file = os.path.join(
    OUTPUT_DIR,
    "stage_improvements.csv",
)


result.to_csv(
    output_file,
    index=False,
)


print("")
print(
    "=============================================="
)

print(
    "STAGE-BY-STAGE IMPROVEMENT"
)

print(
    "Positive % = improvement"
)

print(
    "Negative % = degradation"
)

print(
    "=============================================="
)

print("")

print(
    result.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    output_file
)