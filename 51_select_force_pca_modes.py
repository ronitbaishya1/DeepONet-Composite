import os
import json

import numpy as np
import pandas as pd


ROOT = os.getcwd()

BASE_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_force_pca_enriched",
)


# ============================================================
# TARGETS
# ============================================================

FORCE_MEAN_TARGET = 0.50
FORCE_MAX_TARGET = 1.50

G_MEAN_TARGET = 0.50
G_MAX_TARGET = 1.50


# ============================================================
# LOAD BOTH AUDITS
# ============================================================

left = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "left",
        "mode_audit.csv",
    )
)

right = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "right",
        "mode_audit.csv",
    )
)


audit = pd.concat(
    [
        left,
        right,
    ],
    ignore_index=True,
)


candidate_modes = sorted(
    audit[
        "Modes"
    ].unique()
)


# ============================================================
# CHECK EACH MODE COUNT
# ============================================================

rows = []

selected_modes = None


for modes in candidate_modes:

    subset = audit[
        audit[
            "Modes"
        ]
        ==
        modes
    ]


    worst_force_mean = float(
        subset[
            "ForceReconMean_percent"
        ].max()
    )

    worst_force_max = float(
        subset[
            "ForceReconMax_percent"
        ].max()
    )

    worst_g_mean = float(
        subset[
            "GeneralizedForceMean_percent"
        ].max()
    )

    worst_g_max = float(
        subset[
            "GeneralizedForceMax_percent"
        ].max()
    )


    passes = (
        worst_force_mean
        <=
        FORCE_MEAN_TARGET

        and

        worst_force_max
        <=
        FORCE_MAX_TARGET

        and

        worst_g_mean
        <=
        G_MEAN_TARGET

        and

        worst_g_max
        <=
        G_MAX_TARGET
    )


    rows.append(
        {
            "Modes":
                int(modes),

            "WorstForceMean_percent":
                worst_force_mean,

            "WorstForceMax_percent":
                worst_force_max,

            "WorstGeneralizedForceMean_percent":
                worst_g_mean,

            "WorstGeneralizedForceMax_percent":
                worst_g_max,

            "Pass":
                bool(
                    passes
                ),
        }
    )


    if (
        passes
        and
        selected_modes is None
    ):

        selected_modes = int(
            modes
        )


selection_df = pd.DataFrame(
    rows
)


# ============================================================
# IF NOTHING PASSES, STOP INSTEAD OF HIDING THE PROBLEM
# ============================================================

if selected_modes is None:

    print("")
    print(
        selection_df.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "\nNone of the candidate mode counts met the "
        "accuracy targets.\n"
        "Do not continue automatically. "
        "Inspect the table above first."
    )


# ============================================================
# SAVE
# ============================================================

selection = {

    "SelectedModes":
        selected_modes,

    "SelectionUsedTestData":
        False,

    "ForceMeanTarget_percent":
        FORCE_MEAN_TARGET,

    "ForceMaxTarget_percent":
        FORCE_MAX_TARGET,

    "GeneralizedForceMeanTarget_percent":
        G_MEAN_TARGET,

    "GeneralizedForceMaxTarget_percent":
        G_MAX_TARGET,
}


output_json = os.path.join(
    BASE_DIR,
    "selected_force_modes.json",
)


with open(
    output_json,
    "w",
) as file_object:

    json.dump(
        selection,
        file_object,
        indent=4,
    )


output_csv = os.path.join(
    BASE_DIR,
    "mode_selection_summary.csv",
)


selection_df.to_csv(
    output_csv,
    index=False,
)


print("")
print(
    "========================================"
)

print(
    "STEP 51 COMPLETE"
)

print(
    "========================================"
)


print("")
print(
    selection_df.to_string(
        index=False
    )
)


print("")
print(
    "Selected force modes:"
)

print(
    selected_modes
)


print("")
print(
    "IMPORTANT:"
)

print(
    "No test data were used to choose this number."
)