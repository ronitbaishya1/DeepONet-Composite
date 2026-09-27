import os
import json


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = os.getcwd()


RESULTS_DIR = os.path.join(
    ROOT,
    "results",
    "rebuilt_force_pca",
)


# ============================================================
# CANDIDATE CONFIGURATIONS
# ============================================================

CONFIG = {

    "left": {

        "natural":
            os.path.join(
                RESULTS_DIR,
                "left_natural",
                "ensemble_validation_summary.json",
            ),

        "weighted30":
            os.path.join(
                RESULTS_DIR,
                "left_weighted30",
                "ensemble_validation_summary.json",
            ),
    },


    "right": {

        "natural":
            os.path.join(
                RESULTS_DIR,
                "right_natural",
                "ensemble_validation_summary.json",
            ),

        "weighted30":
            os.path.join(
                RESULTS_DIR,
                "right_weighted30",
                "ensemble_validation_summary.json",
            ),
    },
}


# ============================================================
# SELECT USING VALIDATION ONLY
# ============================================================

final_selection = {}


for segment in [
    "left",
    "right",
]:

    results = {}


    for strategy_name, filename in CONFIG[
        segment
    ].items():

        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                filename
            )


        with open(
            filename,
            "r",
        ) as file_object:

            results[
                strategy_name
            ] = json.load(
                file_object
            )


    # --------------------------------------------------------
    # Lowest validation score wins.
    #
    # Score from Step 53:
    #
    # validation U2
    # +
    # 0.25 * generalized-force error
    #
    # No test result is involved here.
    # --------------------------------------------------------

    selected_strategy = min(

        results.keys(),

        key=lambda name:
            results[
                name
            ][
                "ValidationSelectionScore"
            ],
    )


    selected_result = results[
        selected_strategy
    ]


    rejected_strategy = (

        "weighted30"

        if selected_strategy
        ==
        "natural"

        else "natural"
    )


    rejected_result = results[
        rejected_strategy
    ]


    # ========================================================
    # DIRECTORY CONTAINING THE 5 CHECKPOINTS
    # ========================================================

    selected_directory = os.path.join(
        RESULTS_DIR,
        "{}_{}".format(
            segment,
            selected_strategy,
        ),
    )


    final_selection[
        segment
    ] = {

        "SelectedStrategy":
            selected_strategy,

        "SelectedDirectory":
            selected_directory,

        "RejectedStrategy":
            rejected_strategy,

        "ForceModesPerInterface":
            int(
                selected_result[
                    "ForceModesPerInterface"
                ]
            ),

        "SelectedValidation_U1_percent":
            float(
                selected_result[
                    "Validation_Ensemble_U1_percent"
                ]
            ),

        "SelectedValidation_U2_percent":
            float(
                selected_result[
                    "Validation_Ensemble_U2_percent"
                ]
            ),

        "SelectedValidation_U3_percent":
            float(
                selected_result[
                    "Validation_Ensemble_U3_percent"
                ]
            ),

        "SelectedValidation_GeneralizedForce_percent":
            float(
                selected_result[
                    "Validation_Ensemble_GeneralizedForce_percent"
                ]
            ),

        "SelectedValidation_ForceCoefficient_percent":
            float(
                selected_result[
                    "Validation_Ensemble_ForceCoefficient_percent"
                ]
            ),

        "SelectedValidationScore":
            float(
                selected_result[
                    "ValidationSelectionScore"
                ]
            ),

        "RejectedValidationScore":
            float(
                rejected_result[
                    "ValidationSelectionScore"
                ]
            ),

        "TestDataUsedForSelection":
            False,
    }


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "selected_strategies.json",
)


with open(
    output_file,
    "w",
) as file_object:

    json.dump(
        final_selection,
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
    "STEP 54 — STRATEGY SELECTION"
)

print(
    "================================================"
)


for segment in [
    "left",
    "right",
]:

    result = final_selection[
        segment
    ]


    print("")

    print(
        segment.upper()
    )


    print(
        "Selected:",
        result[
            "SelectedStrategy"
        ]
    )


    print(
        "Validation U2: {:.6f}%".format(
            result[
                "SelectedValidation_U2_percent"
            ]
        )
    )


    print(
        "Validation generalized force: {:.6f}%"
        .format(
            result[
                "SelectedValidation_GeneralizedForce_percent"
            ]
        )
    )


    print(
        "Selected score: {:.6f}".format(
            result[
                "SelectedValidationScore"
            ]
        )
    )


    print(
        "Other score:    {:.6f}".format(
            result[
                "RejectedValidationScore"
            ]
        )
    )


print("")
print(
    "IMPORTANT:"
)

print(
    "The test set was NOT used for this decision."
)


print("")
print(
    "Saved:"
)

print(
    output_file
)