import os
import json
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--threshold",
    type=float,
    required=True,
)


parser.add_argument(
    "--input",
    default=
        "results/saint_venant_partition/"
        "partition_candidates_7region.csv",
)


parser.add_argument(
    "--output",
    default=
        "results/saint_venant_partition/"
        "selected_partition_7region.json",
)


parser.add_argument(
    "--allow_invalid",
    action="store_true",
)


args = parser.parse_args()


# ============================================================
# LOAD
# ============================================================

table = pd.read_csv(
    args.input
)


if "Threshold" not in table.columns:

    raise RuntimeError(
        "Threshold column not found."
    )


difference = np.abs(
    table[
        "Threshold"
    ].to_numpy(
        dtype=np.float64
    )
    -
    args.threshold
)


row_position = int(
    np.argmin(
        difference
    )
)


row = table.iloc[
    row_position
]


selected_threshold = float(
    row[
        "Threshold"
    ]
)


if abs(
    selected_threshold
    -
    args.threshold
) > 1.0e-8:

    raise RuntimeError(
        "Threshold {} was not found."
        .format(
            args.threshold
        )
    )


# ============================================================
# BOOLEAN HELPER
# ============================================================

def as_bool(
    value
):

    if isinstance(
        value,
        bool,
    ):

        return value


    value_string = str(
        value
    ).strip().lower()


    return value_string in [
        "true",
        "1",
        "yes",
    ]


candidate_valid = as_bool(
    row[
        "CandidateValid"
    ]
)


if (
    not candidate_valid
    and
    not args.allow_invalid
):

    raise RuntimeError(
        (
            "Selected partition is marked invalid. "
            "Do not continue unless you intentionally use "
            "--allow_invalid."
        )
    )


# ============================================================
# VALUES
# ============================================================

domain_x_min = float(
    row[
        "DomainXMin"
    ]
)


domain_x_max = float(
    row[
        "DomainXMax"
    ]
)


left_outer = float(
    row[
        "left_outer_x"
    ]
)


left_inner = float(
    row[
        "left_inner_x"
    ]
)


center_left = float(
    row[
        "center_left_x"
    ]
)


center_right = float(
    row[
        "center_right_x"
    ]
)


right_inner = float(
    row[
        "right_inner_x"
    ]
)


right_outer = float(
    row[
        "right_outer_x"
    ]
)


# ============================================================
# JSON
# ============================================================

selected = {

    "Threshold":
        selected_threshold,

    "CandidateValid":
        candidate_valid,

    "DomainXMin":
        domain_x_min,

    "DomainXMax":
        domain_x_max,

    "Current_FE_Percent":
        float(
            row[
                "Current_FE_Percent"
            ]
        ),

    "FE_Percent":
        float(
            row[
                "FE_Percent"
            ]
        ),

    "FE_Fraction":
        float(
            row[
                "FE_Fraction"
            ]
        ),

    "FE_PercentagePoint_Reduction":
        float(
            row[
                "FE_PercentagePoint_Reduction"
            ]
        ),

    # ========================================================
    # SIX FE-NO INTERFACES
    # ========================================================

    "interfaces": {

        "left_outer":
            left_outer,

        "left_inner":
            left_inner,

        "center_left":
            center_left,

        "center_right":
            center_right,

        "right_inner":
            right_inner,

        "right_outer":
            right_outer,
    },

    # ========================================================
    # SEVEN REGIONS
    # ========================================================

    "regions": {

        "NO_outer_left": {

            "solver":
                "NO",

            "x_min":
                domain_x_min,

            "x_max":
                left_outer,
        },

        "FE_left_support": {

            "solver":
                "FE",

            "x_min":
                left_outer,

            "x_max":
                left_inner,
        },

        "NO_inner_left": {

            "solver":
                "NO",

            "x_min":
                left_inner,

            "x_max":
                center_left,
        },

        "FE_center": {

            "solver":
                "FE",

            "x_min":
                center_left,

            "x_max":
                center_right,
        },

        "NO_inner_right": {

            "solver":
                "NO",

            "x_min":
                center_right,

            "x_max":
                right_inner,
        },

        "FE_right_support": {

            "solver":
                "FE",

            "x_min":
                right_inner,

            "x_max":
                right_outer,
        },

        "NO_outer_right": {

            "solver":
                "NO",

            "x_min":
                right_outer,

            "x_max":
                domain_x_max,
        },
    },
}


# ============================================================
# SAVE
# ============================================================

output_directory = os.path.dirname(
    args.output
)


if output_directory:

    os.makedirs(
        output_directory,
        exist_ok=True,
    )


with open(
    args.output,
    "w",
) as file_object:

    json.dump(
        selected,
        file_object,
        indent=4,
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64C COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    json.dumps(
        selected,
        indent=4,
    )
)

print("")
print(
    "Saved:"
)

print(
    args.output
)