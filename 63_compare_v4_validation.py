import os
import json

import pandas as pd


OUTPUT_DIR = os.path.join(
    "results",
    "v4_selection",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


rows = []

selection = {}


for segment in [
    "left",
    "right",
]:

    direct_file = os.path.join(
        "results",
        "v4_direct",
        segment,
        "ensemble_validation_summary.json",
    )


    consistency_file = os.path.join(
        "results",
        "v4_consistency",
        segment,
        "ensemble_validation_summary.json",
    )


    with open(
        direct_file,
        "r",
    ) as file_object:

        direct = json.load(
            file_object
        )


    with open(
        consistency_file,
        "r",
    ) as file_object:

        consistent = json.load(
            file_object
        )


    def mechanics_mean(
        result,
        prefix,
    ):

        return (
            result[
                "Validation_{}11_percent".format(
                    prefix
                )
            ]
            +
            result[
                "Validation_{}12_percent".format(
                    prefix
                )
            ]
        ) / 2.0


    direct_LE = mechanics_mean(
        direct,
        "LE",
    )


    consistency_LE = mechanics_mean(
        consistent,
        "LE",
    )


    direct_S = mechanics_mean(
        direct,
        "S",
    )


    consistency_S = mechanics_mean(
        consistent,
        "S",
    )


    # --------------------------------------------------------
    # Ratio score.
    #
    # 45% displacement U2
    # 25% strain
    # 20% stress
    # 10% interface generalized force
    # --------------------------------------------------------

    consistency_score = (
        0.45
        *
        consistent[
            "Validation_U2_percent"
        ]
        /
        direct[
            "Validation_U2_percent"
        ]

        +

        0.25
        *
        consistency_LE
        /
        direct_LE

        +

        0.20
        *
        consistency_S
        /
        direct_S

        +

        0.10
        *
        consistent[
            "Validation_GeneralizedForce_percent"
        ]
        /
        direct[
            "Validation_GeneralizedForce_percent"
        ]
    )


    # Don't accept a mechanics model that damages U2
    # by more than 5%.
    displacement_gate = (
        consistent[
            "Validation_U2_percent"
        ]
        <=
        1.05
        *
        direct[
            "Validation_U2_percent"
        ]
    )


    if (
        displacement_gate
        and
        consistency_score < 1.0
    ):

        selected = (
            "consistency"
        )

        selected_root = os.path.join(
            "results",
            "v4_consistency",
            segment,
        )

    else:

        selected = "direct"

        selected_root = os.path.join(
            "results",
            "v4_direct",
            segment,
        )


    selection[
        segment
    ] = {
        "Selected":
            selected,

        "SelectedRoot":
            selected_root,

        "ConsistencyScore":
            consistency_score,

        "DisplacementGatePassed":
            bool(
                displacement_gate
            ),

        "Direct_U2_percent":
            direct[
                "Validation_U2_percent"
            ],

        "Consistency_U2_percent":
            consistent[
                "Validation_U2_percent"
            ],

        "Direct_LE11_LE12_mean_percent":
            direct_LE,

        "Consistency_LE11_LE12_mean_percent":
            consistency_LE,

        "Direct_S11_S12_mean_percent":
            direct_S,

        "Consistency_S11_S12_mean_percent":
            consistency_S,
    }


    rows.append(
        {
            "Segment":
                segment,

            "Selected":
                selected,

            "ConsistencyScore":
                consistency_score,

            "Direct_U2":
                direct[
                    "Validation_U2_percent"
                ],

            "Consistency_U2":
                consistent[
                    "Validation_U2_percent"
                ],

            "Direct_LE":
                direct_LE,

            "Consistency_LE":
                consistency_LE,

            "Direct_S":
                direct_S,

            "Consistency_S":
                consistency_S,
        }
    )


pd.DataFrame(
    rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v4_validation_comparison.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "selected_v4_models.json",
    ),
    "w",
) as file_object:

    json.dump(
        selection,
        file_object,
        indent=4,
    )


print("")
print(
    pd.DataFrame(
        rows
    ).to_string(
        index=False
    )
)


print("")
print(
    json.dumps(
        selection,
        indent=4,
    )
)