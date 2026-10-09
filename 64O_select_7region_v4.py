import os
import json

import pandas as pd


SEGMENTS = [
    "outer_left",
    "inner_left",
    "inner_right",
    "outer_right",
]


DIRECT_ROOT = os.path.join(
    "results",
    "v4_7region_direct",
)


CONSISTENCY_ROOT = os.path.join(
    "results",
    "v4_7region_consistency",
)


OUTPUT_DIR = os.path.join(
    "results",
    "v4_7region_selection",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


selection = {}

rows = []


for segment in SEGMENTS:

    with open(
        os.path.join(
            DIRECT_ROOT,
            segment,
            "validation_summary.json",
        ),
        "r",
    ) as file_object:

        direct = json.load(
            file_object
        )


    with open(
        os.path.join(
            CONSISTENCY_ROOT,
            segment,
            "validation_summary.json",
        ),
        "r",
    ) as file_object:

        consistency = json.load(
            file_object
        )


    # ========================================================
    # MECHANICS GROUPS
    # ========================================================

    direct_LE = (
        0.5
        *
        (
            direct[
                "LE11_percent"
            ]
            +
            direct[
                "LE12_percent"
            ]
        )
    )


    consistency_LE = (
        0.5
        *
        (
            consistency[
                "LE11_percent"
            ]
            +
            consistency[
                "LE12_percent"
            ]
        )
    )


    direct_S = (
        0.5
        *
        (
            direct[
                "S11_percent"
            ]
            +
            direct[
                "S12_percent"
            ]
        )
    )


    consistency_S = (
        0.5
        *
        (
            consistency[
                "S11_percent"
            ]
            +
            consistency[
                "S12_percent"
            ]
        )
    )


    # ========================================================
    # CONSISTENCY / DIRECT SCORE
    #
    # < 1 means consistency is better overall.
    # ========================================================

    score = (

        0.45
        *
        (
            consistency[
                "U2_percent"
            ]
            /
            direct[
                "U2_percent"
            ]
        )

        +

        0.25
        *
        (
            consistency_LE
            /
            direct_LE
        )

        +

        0.20
        *
        (
            consistency_S
            /
            direct_S
        )

        +

        0.10
        *
        (
            consistency[
                "GeneralizedForce_percent"
            ]
            /
            direct[
                "GeneralizedForce_percent"
            ]
        )
    )


    # ========================================================
    # DISPLACEMENT SAFETY GATE
    #
    # Consistency may not worsen U2 by >5%.
    # ========================================================

    displacement_gate = (
        consistency[
            "U2_percent"
        ]
        <=
        1.05
        *
        direct[
            "U2_percent"
        ]
    )


    choose_consistency = (
        score < 1.0
        and
        displacement_gate
    )


    if choose_consistency:

        selected_type = (
            "consistency"
        )


        selected_root = os.path.join(
            CONSISTENCY_ROOT,
            segment,
        )


        selected_summary = (
            consistency
        )

    else:

        selected_type = (
            "direct"
        )


        selected_root = os.path.join(
            DIRECT_ROOT,
            segment,
        )


        selected_summary = (
            direct
        )


    selection[
        segment
    ] = {

        "SelectedType":
            selected_type,

        "SelectedRoot":
            selected_root,

        "ValidationScoreConsistencyOverDirect":
            float(
                score
            ),

        "DisplacementGatePassed":
            bool(
                displacement_gate
            ),

        "SelectedValidationSummary":
            selected_summary,
    }


    rows.append(
        {
            "Segment":
                segment,

            "Direct_U2_percent":
                direct[
                    "U2_percent"
                ],

            "Consistency_U2_percent":
                consistency[
                    "U2_percent"
                ],

            "Direct_LE11_percent":
                direct[
                    "LE11_percent"
                ],

            "Consistency_LE11_percent":
                consistency[
                    "LE11_percent"
                ],

            "Direct_LE12_percent":
                direct[
                    "LE12_percent"
                ],

            "Consistency_LE12_percent":
                consistency[
                    "LE12_percent"
                ],

            "Direct_S11_percent":
                direct[
                    "S11_percent"
                ],

            "Consistency_S11_percent":
                consistency[
                    "S11_percent"
                ],

            "Direct_S12_percent":
                direct[
                    "S12_percent"
                ],

            "Consistency_S12_percent":
                consistency[
                    "S12_percent"
                ],

            "Direct_GForce_percent":
                direct[
                    "GeneralizedForce_percent"
                ],

            "Consistency_GForce_percent":
                consistency[
                    "GeneralizedForce_percent"
                ],

            "ConsistencyOverDirectScore":
                score,

            "DisplacementGatePassed":
                displacement_gate,

            "Selected":
                selected_type,
        }
    )


with open(
    os.path.join(
        OUTPUT_DIR,
        "selected_7region_models.json",
    ),
    "w",
) as file_object:

    json.dump(
        selection,
        file_object,
        indent=4,
    )


comparison = pd.DataFrame(
    rows
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_comparison.csv",
    ),
    index=False,
)


print("")
print(
    "=========================================="
)

print(
    "STEP 64O COMPLETE"
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