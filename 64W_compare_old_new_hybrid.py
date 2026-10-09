import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

NEW_DIR = os.path.join(
    "results",
    "hybrid_online_7region_direct_validation",
)


NEW_VALIDATION_DIR = os.path.join(
    "data",
    "hybrid_online_7region_direct",
    "validation_fields",
)


OLD_METRICS_FILE = os.path.join(
    "results",
    "hybrid_online_validation",
    "hybrid_displacement_metrics.csv",
)


OLD_HYBRID_U_FILE = os.path.join(
    "results",
    "hybrid_online_validation",
    "hybrid_U.npy",
)


OLD_REACTION_FILE = os.path.join(
    "data",
    "hybrid_online_baseline",
    "validation_fields",
    "reaction_summary.json",
)


OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_40_vs_65_comparison",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# NEW RESULTS
# ============================================================

new_metrics = pd.read_csv(
    os.path.join(
        NEW_DIR,
        "global_displacement_metrics.csv",
    )
)


region_metrics = pd.read_csv(
    os.path.join(
        NEW_DIR,
        "region_displacement_metrics.csv",
    )
)


assembled = pd.read_csv(
    os.path.join(
        NEW_DIR,
        "assembled_hybrid_nodes.csv",
    )
)


with open(
    os.path.join(
        NEW_VALIDATION_DIR,
        "reaction_summary.json",
    ),
    "r",
) as file_object:

    new_reaction = json.load(
        file_object
    )


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison_rows = []


for _, row in new_metrics.iterrows():

    component = row[
        "Component"
    ]


    comparison_row = {

        "Component":
            component,

        "New_40FE_RelativeL2_percent":
            row[
                "Relative_L2_percent"
            ],
    }


    # ========================================================
    # OPTIONAL OLD RESULT
    # ========================================================

    if os.path.isfile(
        OLD_METRICS_FILE
    ):

        old_metrics = pd.read_csv(
            OLD_METRICS_FILE
        )


        old_match = old_metrics[
            old_metrics[
                "Component"
            ]
            ==
            component
        ]


        if len(
            old_match
        ) == 1:

            old_value = float(
                old_match.iloc[
                    0
                ][
                    "Relative_L2"
                ]
            )


            comparison_row[
                "Old_65FE_RelativeL2_percent"
            ] = (
                100.0
                *
                old_value
            )


            comparison_row[
                "NewMinusOld_percentage_points"
            ] = (

                comparison_row[
                    "New_40FE_RelativeL2_percent"
                ]

                -

                comparison_row[
                    "Old_65FE_RelativeL2_percent"
                ]
            )


    comparison_rows.append(
        comparison_row
    )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "old65_vs_new40_displacement.csv",
    ),

    index=False,
)


# ============================================================
# REACTION COMPARISON
# ============================================================

reaction_comparison = {

    "New_40FE_full_FEM_RF2_N":
        new_reaction[
            "full_FEM_nose_RF2_N"
        ],

    "New_40FE_hybrid_RF2_N":
        new_reaction[
            "hybrid_center_nose_RF2_N"
        ],

    "New_40FE_error_percent":
        new_reaction[
            "relative_reaction_error_percent"
        ],
}


if os.path.isfile(
    OLD_REACTION_FILE
):

    with open(
        OLD_REACTION_FILE,
        "r",
    ) as file_object:

        old_reaction = json.load(
            file_object
        )


    reaction_comparison[
        "Old_65FE_hybrid_RF2_N"
    ] = old_reaction[
        "hybrid_center_nose_RF2_N"
    ]


    reaction_comparison[
        "Old_65FE_error_percent"
    ] = (
        100.0
        *
        old_reaction[
            "relative_reaction_error"
        ]
    )


with open(
    os.path.join(
        OUTPUT_DIR,
        "reaction_comparison.json",
    ),
    "w",
) as file_object:

    json.dump(
        reaction_comparison,
        file_object,
        indent=4,
    )


# ============================================================
# CENTERLINE
# ============================================================

assembled[
    "X_round"
] = assembled[
    "X"
].round(
    5
)


assembled[
    "DistanceToCenter"
] = (

    (
        assembled[
            "Y"
        ]
        -
        2.0
    ) ** 2

    +

    (
        assembled[
            "Z"
        ]
        -
        4.0
    ) ** 2
)


centerline = (

    assembled

    .sort_values(
        [
            "X_round",
            "DistanceToCenter",
        ]
    )

    .groupby(
        "X_round",
        as_index=False,
    )

    .first()

    .sort_values(
        "X"
    )
)


# ============================================================
# OPTIONAL OLD FIELD
# ============================================================

old_available = (
    os.path.isfile(
        OLD_HYBRID_U_FILE
    )
)


if old_available:

    old_U = np.load(
        OLD_HYBRID_U_FILE
    )


    if old_U.shape[
        0
    ] == len(
        assembled
    ):

        assembled[
            "OldHybrid_U2"
        ] = old_U[
            :,
            1
        ]


        centerline = (

            assembled

            .sort_values(
                [
                    "X_round",
                    "DistanceToCenter",
                ]
            )

            .groupby(
                "X_round",
                as_index=False,
            )

            .first()

            .sort_values(
                "X"
            )
        )


    else:

        old_available = False


# ============================================================
# FIGURE 1 — CENTERLINE U2
# ============================================================

figure = plt.figure(
    figsize=(
        12,
        5,
    )
)


axis = figure.add_subplot(
    111
)


axis.plot(
    centerline[
        "X"
    ],
    centerline[
        "FEM_U2"
    ],
    linewidth=2.2,
    label="Full FEM",
)


axis.plot(
    centerline[
        "X"
    ],
    centerline[
        "Hybrid_U2"
    ],
    linestyle="--",
    linewidth=2.0,
    label="New hybrid: 40% FE",
)


if old_available:

    axis.plot(
        centerline[
            "X"
        ],
        centerline[
            "OldHybrid_U2"
        ],
        linestyle="-.",
        linewidth=1.8,
        label="Old hybrid: 65% FE",
    )


for interface_x in [
    -9.6,
    -6.6,
    -1.8,
    1.8,
    6.6,
    9.6,
]:

    axis.axvline(
        interface_x,
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
    )


axis.set_xlabel(
    "x (mm)"
)


axis.set_ylabel(
    "U2 (mm)"
)


axis.set_title(
    "Centerline vertical displacement"
)


axis.legend()


axis.grid(
    alpha=0.2
)


figure.tight_layout()


figure.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_centerline_U2_old_vs_new.png",
    ),
    dpi=300,
)


plt.close(
    figure
)


# ============================================================
# FIGURE 2 — ABSOLUTE U2 ERROR VS X
# ============================================================

centerline[
    "NewAbsError_U2"
] = np.abs(

    centerline[
        "Hybrid_U2"
    ]

    -

    centerline[
        "FEM_U2"
    ]
)


figure = plt.figure(
    figsize=(
        12,
        5,
    )
)


axis = figure.add_subplot(
    111
)


axis.plot(
    centerline[
        "X"
    ],
    centerline[
        "NewAbsError_U2"
    ],
    linewidth=2.0,
    label="New 40% FE hybrid",
)


if old_available:

    centerline[
        "OldAbsError_U2"
    ] = np.abs(

        centerline[
            "OldHybrid_U2"
        ]

        -

        centerline[
            "FEM_U2"
        ]
    )


    axis.plot(
        centerline[
            "X"
        ],
        centerline[
            "OldAbsError_U2"
        ],
        linestyle="--",
        linewidth=1.8,
        label="Old 65% FE hybrid",
    )


for interface_x in [
    -9.6,
    -6.6,
    -1.8,
    1.8,
    6.6,
    9.6,
]:

    axis.axvline(
        interface_x,
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
    )


axis.set_xlabel(
    "x (mm)"
)


axis.set_ylabel(
    "|U2 error| (mm)"
)


axis.set_title(
    "Centerline U2 absolute error"
)


axis.legend()


axis.grid(
    alpha=0.2
)


figure.tight_layout()


figure.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_centerline_U2_error.png",
    ),
    dpi=300,
)


plt.close(
    figure
)


# ============================================================
# FIGURE 3 — REGION-WISE U2 ERROR
# ============================================================

u2_regions = region_metrics[
    region_metrics[
        "Component"
    ]
    ==
    "U2"
].copy()


REGION_ORDER = [

    "NO_OL",
    "FE_L",
    "NO_L",
    "FE_C",
    "NO_R",
    "FE_R",
    "NO_OR",
]


u2_regions[
    "Region"
] = pd.Categorical(

    u2_regions[
        "Region"
    ],

    categories=
        REGION_ORDER,

    ordered=
        True,
)


u2_regions = u2_regions.sort_values(
    "Region"
)


figure = plt.figure(
    figsize=(
        10,
        5,
    )
)


axis = figure.add_subplot(
    111
)


axis.bar(
    u2_regions[
        "Region"
    ].astype(
        str
    ),

    u2_regions[
        "Relative_L2_percent"
    ],
)


axis.set_xlabel(
    "Hybrid region"
)


axis.set_ylabel(
    "U2 Relative L2 (%)"
)


axis.set_title(
    "Region-wise vertical-displacement error"
)


axis.grid(
    axis="y",
    alpha=0.2,
)


figure.tight_layout()


figure.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_regionwise_U2_error.png",
    ),
    dpi=300,
)


plt.close(
    figure
)


# ============================================================
# SUMMARY TXT
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "comparison_summary.txt",
)


with open(
    summary_file,
    "w",
) as file_object:

    file_object.write(
        "NEW 40% FE / 60% NO HYBRID\n"
    )


    file_object.write(
        "=========================\n\n"
    )


    file_object.write(
        comparison.to_string(
            index=False
        )
    )


    file_object.write(
        "\n\nREACTION\n"
    )


    file_object.write(
        json.dumps(
            reaction_comparison,
            indent=4,
        )
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64W COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    "DISPLACEMENT COMPARISON"
)

print(
    comparison.to_string(
        index=False
    )
)


print("")
print(
    "REACTION COMPARISON"
)

print(
    json.dumps(
        reaction_comparison,
        indent=4,
    )
)


print("")
print(
    "REGION-WISE U2"
)

print(
    u2_regions[
        [
            "Region",
            "Nodes",
            "Relative_L2_percent",
        ]
    ].to_string(
        index=False
    )
)