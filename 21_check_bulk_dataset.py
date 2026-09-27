import os
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--data_dir",
    required=True,
)


args = parser.parse_args()


# ============================================================
# LOAD ARRAYS
# ============================================================

branch = np.load(
    os.path.join(
        args.data_dir,
        "branch_inputs.npy",
    )
)


coordinates = np.load(
    os.path.join(
        args.data_dir,
        "coordinates.npy",
    )
)


U1 = np.load(
    os.path.join(
        args.data_dir,
        "U1.npy",
    )
)


U2 = np.load(
    os.path.join(
        args.data_dir,
        "U2.npy",
    )
)


U3 = np.load(
    os.path.join(
        args.data_dir,
        "U3.npy",
    )
)


g_left = np.load(
    os.path.join(
        args.data_dir,
        "g_left.npy",
    )
)


g_right = np.load(
    os.path.join(
        args.data_dir,
        "g_right.npy",
    )
)


force_balance = pd.read_csv(
    os.path.join(
        args.data_dir,
        "force_balance.csv",
    )
)


split = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# BASIC CHECKS
# ============================================================

arrays = {
    "branch":
        branch,

    "coordinates":
        coordinates,

    "U1":
        U1,

    "U2":
        U2,

    "U3":
        U3,

    "g_left":
        g_left,

    "g_right":
        g_right,
}


print("")
print(
    "=============================================="
)

print(
    "HYBRID BULK DATASET CHECK"
)

print(
    "=============================================="
)


for name, array in arrays.items():

    print(
        "{:<12s} shape={} finite={}".format(
            name,
            array.shape,
            np.isfinite(
                array
            ).all(),
        )
    )


# ============================================================
# DISPLACEMENT RANGES
# ============================================================

print("")
print(
    "DISPLACEMENT RANGES"
)


for name, array in [
    (
        "U1",
        U1,
    ),
    (
        "U2",
        U2,
    ),
    (
        "U3",
        U3,
    ),
]:

    print(
        "{}: min={:.6e}, max={:.6e}, RMS={:.6e}".format(
            name,
            array.min(),
            array.max(),
            np.sqrt(
                np.mean(
                    array
                    ** 2
                )
            ),
        )
    )


# ============================================================
# GENERALIZED FORCE RANGES
# ============================================================

print("")
print(
    "GENERALIZED FORCE RANGES"
)


for name, array in [
    (
        "g_left",
        g_left,
    ),
    (
        "g_right",
        g_right,
    ),
]:

    print("")
    print(
        name
    )

    for component in range(
        array.shape[
            1
        ]
    ):

        values = array[
            :,
            component
        ]

        print(
            "  mode {:02d}: min={:.6e}, max={:.6e}, std={:.6e}".format(
                component + 1,
                values.min(),
                values.max(),
                values.std(),
            )
        )


# ============================================================
# FORCE BALANCE
# ============================================================

print("")
print(
    "FORCE BALANCE"
)

print(
    "Mean relative imbalance:",
    force_balance[
        "RelativeImbalance"
    ].mean(),
)

print(
    "Median relative imbalance:",
    force_balance[
        "RelativeImbalance"
    ].median(),
)

print(
    "Maximum relative imbalance:",
    force_balance[
        "RelativeImbalance"
    ].max(),
)


worst = force_balance.sort_values(
    "RelativeImbalance",
    ascending=False,
).head(
    5
)


print("")
print(
    "Worst five force-balance cases:"
)

print(
    worst[
        [
            "CaseID",
            "RelativeImbalance",
            "ImbalanceNorm_N",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SPLIT COUNTS
# ============================================================

print("")
print(
    "DATA SPLIT"
)

print(
    split[
        "Split"
    ].value_counts()
)


print("")
print(
    "SOURCE x SPLIT"
)

print(
    pd.crosstab(
        split[
            "Source"
        ],
        split[
            "Split"
        ],
    )
)


# ============================================================
# STATUS
# ============================================================

all_finite = all(
    np.isfinite(
        array
    ).all()

    for array in arrays.values()
)


if not all_finite:

    print("")
    print(
        "FAILED: non-finite values detected."
    )

else:

    print("")
    print(
        "All arrays contain finite values."
    )


print("")
print(
    "Dataset check complete."
)