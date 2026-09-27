import os
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = os.getcwd()


# ============================================================
# CONFIGURATION
# ============================================================

CONFIGURATIONS = [

    {
        "Segment":
            "left",

        "Before":
            os.path.join(
                PROJECT_ROOT,
                "results",
                "adaptive_ensemble",
                "left",
                "ensemble_summary.json",
            ),

        "After":
            os.path.join(
                PROJECT_ROOT,
                "results",
                "adaptive_ensemble_enriched",
                "left",
                "ensemble_summary.json",
            ),
    },

    {
        "Segment":
            "right",

        "Before":
            os.path.join(
                PROJECT_ROOT,
                "results",
                "adaptive_ensemble",
                "right",
                "ensemble_summary.json",
            ),

        "After":
            os.path.join(
                PROJECT_ROOT,
                "results",
                "adaptive_ensemble_enriched",
                "right",
                "ensemble_summary.json",
            ),
    },
]


# ============================================================
# HELPER
# ============================================================

def relative_improvement(
    before,
    after,
):

    if abs(
        before
    ) < 1.0e-14:

        return np.nan


    return (

        100.0

        *

        (
            before
            -
            after
        )

        /

        before
    )


# ============================================================
# COMPARE
# ============================================================

rows = []


for configuration in CONFIGURATIONS:

    with open(
        configuration[
            "Before"
        ],
        "r",
    ) as file_object:

        before = json.load(
            file_object
        )


    with open(
        configuration[
            "After"
        ],
        "r",
    ) as file_object:

        after = json.load(
            file_object
        )


    before_u1 = float(
        before[
            "EnsembleMean_U1_percent"
        ]
    )


    before_u2 = float(
        before[
            "EnsembleMean_U2_percent"
        ]
    )


    before_u3 = float(
        before[
            "EnsembleMean_U3_percent"
        ]
    )


    after_u1 = float(
        after[
            "EnsembleMean_U1_percent"
        ]
    )


    after_u2 = float(
        after[
            "EnsembleMean_U2_percent"
        ]
    )


    after_u3 = float(
        after[
            "EnsembleMean_U3_percent"
        ]
    )


    rows.append(
        {
            "Segment":
                configuration[
                    "Segment"
                ],

            "BeforeTotalCases":
                80,

            "AfterTotalCases":
                int(
                    after[
                        "TotalCases"
                    ]
                ),

            "BeforeTrainCases":
                int(
                    before[
                        "TrainCases"
                    ]
                ),

            "AfterTrainCases":
                int(
                    after[
                        "TrainCases"
                    ]
                ),

            "Before_U1_percent":
                before_u1,

            "After_U1_percent":
                after_u1,

            "U1_RelativeImprovement_percent":
                relative_improvement(
                    before_u1,
                    after_u1,
                ),

            "Before_U2_percent":
                before_u2,

            "After_U2_percent":
                after_u2,

            "U2_AbsoluteChange_percentage_points":
                before_u2
                -
                after_u2,

            "U2_RelativeImprovement_percent":
                relative_improvement(
                    before_u2,
                    after_u2,
                ),

            "Before_U3_percent":
                before_u3,

            "After_U3_percent":
                after_u3,

            "U3_RelativeImprovement_percent":
                relative_improvement(
                    before_u3,
                    after_u3,
                ),

            "After_GeneralizedForce_percent":
                float(
                    after[
                        "EnsembleMean_GeneralizedForce_percent"
                    ]
                ),

            "After_ForceCoefficient_percent":
                float(
                    after[
                        "EnsembleMean_ForceCoefficient_percent"
                    ]
                ),
        }
    )


dataframe = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    PROJECT_ROOT,
    "results",
    "adaptive_enrichment_comparison.csv",
)


dataframe.to_csv(
    output_file,
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
    "ADAPTIVE ENRICHMENT COMPARISON"
)

print(
    "================================================"
)


print("")
print(
    dataframe.to_string(
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