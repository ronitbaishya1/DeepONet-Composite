import os

import pandas as pd
import torch


# ============================================================
# PATH
# ============================================================

ROOT = os.path.join(
    "results",
    "pi_mechanics_aware",
)


OUTPUT_FILE = os.path.join(
    ROOT,
    "lambda_validation_comparison.csv",
)


# ============================================================
# FIND PI EXPERIMENT DIRECTORIES
# ============================================================

experiment_directories = []


for name in os.listdir(
    ROOT
):

    full_path = os.path.join(
        ROOT,
        name,
    )

    if (
        os.path.isdir(
            full_path
        )
        and name.startswith(
            "eq_"
        )
    ):

        experiment_directories.append(
            full_path
        )


experiment_directories = sorted(
    experiment_directories
)


# ============================================================
# READ CHECKPOINTS
# ============================================================

rows = []


for directory in experiment_directories:

    checkpoint_file = os.path.join(
        directory,
        "best_pi_deeponet.pt",
    )


    if not os.path.exists(
        checkpoint_file
    ):

        continue


    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
    )


    rows.append(
        {
            "Experiment":
                os.path.basename(
                    directory
                ),

            "TrainingLambdaEq":
                checkpoint[
                    "training_lambda_eq"
                ],

            "BestEpoch":
                checkpoint[
                    "epoch"
                ],

            "ValidationDataLoss":
                checkpoint[
                    "ValidationDataLoss"
                ],

            "ValidationStrainLoss":
                checkpoint[
                    "ValidationStrainLoss"
                ],

            "ValidationStressLoss":
                checkpoint[
                    "ValidationStressLoss"
                ],

            "ValidationEquilibriumLoss":
                checkpoint[
                    "ValidationEquilibriumLoss"
                ],

            "ValidationScore":
                checkpoint[
                    "ValidationScore"
                ],
        }
    )


# ============================================================
# TABLE
# ============================================================

comparison = pd.DataFrame(
    rows
)


comparison = comparison.sort_values(
    "ValidationScore",
    ascending=True,
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("")
print(
    "============================================"
)

print(
    "PI VALIDATION COMPARISON"
)

print(
    "============================================"
)

print("")

print(
    comparison.to_string(
        index=False
    )
)


if len(
    comparison
) > 0:

    best = comparison.iloc[
        0
    ]


    print("")
    print(
        "Selected by mechanics-aware validation:"
    )

    print(
        "lambda_eq =",
        best[
            "TrainingLambdaEq"
        ],
    )

    print(
        "Best epoch =",
        int(
            best[
                "BestEpoch"
            ]
        ),
    )

    print(
        "Validation score =",
        best[
            "ValidationScore"
        ],
    )


print("")
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)