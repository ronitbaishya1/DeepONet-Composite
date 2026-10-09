import os
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

NEW_METRICS_FILE = os.path.join(
    "results",
    "hybrid_online_7region_direct_validation",
    "global_displacement_metrics.csv",
)


NEW_REACTION_FILE = os.path.join(
    "data",
    "hybrid_online_7region_direct",
    "validation_fields",
    "reaction_summary.json",
)


OUTPUT_DIR = os.path.join(
    "results",
    "final_old65_vs_new40_comparison",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOCKED OLD STEP-60 ENSEMBLE RESULT
# ============================================================

OLD_FINAL = {

    "U1":
        5.145127,

    "U2":
        0.898181,

    "U3":
        3.782752,
}


OLD_FULL_FEM_REACTION = (
    -1962.793701171875
)


OLD_HYBRID_REACTION = (
    -1963.48486328125
)


OLD_REACTION_ERROR_PERCENT = (

    100.0

    *

    abs(
        OLD_HYBRID_REACTION
        -
        OLD_FULL_FEM_REACTION
    )

    /

    abs(
        OLD_FULL_FEM_REACTION
    )
)


# ============================================================
# NEW RESULTS
# ============================================================

new_metrics = pd.read_csv(
    NEW_METRICS_FILE
)


with open(
    NEW_REACTION_FILE,
    "r",
) as file_object:

    new_reaction = json.load(
        file_object
    )


# ============================================================
# TABLE
# ============================================================

rows = []


for component in [
    "U1",
    "U2",
    "U3",
]:

    new_row = new_metrics[
        new_metrics[
            "Component"
        ]
        ==
        component
    ]


    if len(
        new_row
    ) != 1:

        raise RuntimeError(
            "Could not uniquely find {}."
            .format(
                component
            )
        )


    new_error = float(
        new_row.iloc[
            0
        ][
            "Relative_L2_percent"
        ]
    )


    old_error = OLD_FINAL[
        component
    ]


    rows.append(
        {
            "Component":
                component,

            "Old_65FE_Final_Ensemble_percent":
                old_error,

            "New_40FE_DirectV4_percent":
                new_error,

            "NewMinusOld_percentage_points":
                new_error
                -
                old_error,

            "ErrorRatio_NewOverOld":
                new_error
                /
                old_error,
        }
    )


comparison = pd.DataFrame(
    rows
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_old65_vs_new40_displacement.csv",
    ),
    index=False,
)


# ============================================================
# REACTION
# ============================================================

reaction = {

    "Full_FEM_RF2_N":
        float(
            new_reaction[
                "full_FEM_nose_RF2_N"
            ]
        ),

    "Old_65FE_Final_RF2_N":
        OLD_HYBRID_REACTION,

    "Old_65FE_Final_error_percent":
        OLD_REACTION_ERROR_PERCENT,

    "New_40FE_RF2_N":
        float(
            new_reaction[
                "hybrid_center_nose_RF2_N"
            ]
        ),

    "New_40FE_error_percent":
        float(
            new_reaction[
                "relative_reaction_error_percent"
            ]
        ),
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "final_reaction_comparison.json",
    ),
    "w",
) as file_object:

    json.dump(
        reaction,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64X COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    comparison.to_string(
        index=False
    )
)

print("")
print(
    json.dumps(
        reaction,
        indent=4,
    )
)