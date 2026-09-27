import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--case_id",
    type=str,
    default=None,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"

RESULTS_DIR = os.path.join(
    "results",
    "mechanics_validation",
)

PLOT_DIR = os.path.join(
    RESULTS_DIR,
    "plots",
)


os.makedirs(
    PLOT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "ip_coordinates.npy",
    )
)


S_fem = np.load(
    os.path.join(
        DATA_DIR,
        "S_tensor.npy",
    )
)


LE_fem = np.load(
    os.path.join(
        DATA_DIR,
        "LE_tensor.npy",
    )
)


case_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "case_ids.csv",
    )
)


split_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)


test_indices = split_table.loc[
    split_table["Split"]
    == "test",
    "Index",
].to_numpy(
    dtype=int
)


# ============================================================
# CASE SELECTION
# ============================================================

if args.case_id is None:

    global_index = int(
        test_indices[0]
    )

    case_id = case_table.loc[
        case_table[
            "Index"
        ] == global_index,
        "CaseID",
    ].iloc[0]

else:

    case_id = args.case_id

    global_index = int(
        case_table.loc[
            case_table[
                "CaseID"
            ] == case_id,
            "Index",
        ].iloc[0]
    )


print("")
print(
    "Plotting:",
    case_id
)


# ============================================================
# LOAD DEEPONET-DERIVED FIELDS
# ============================================================

case_results = os.path.join(
    RESULTS_DIR,
    case_id,
)


small_strain = np.load(
    os.path.join(
        case_results,
        "small_strain.npy",
    )
)


log_strain = np.load(
    os.path.join(
        case_results,
        "log_strain.npy",
    )
)


predicted_stress = np.load(
    os.path.join(
        case_results,
        "predicted_stress.npy",
    )
)


# ============================================================
# SELECT MID-WIDTH INTEGRATION-POINT PLANE
# ============================================================

z_values = np.unique(
    coordinates[:, 2]
)


target_z = (
    coordinates[:, 2].min()
    +
    coordinates[:, 2].max()
) / 2.0


selected_z = z_values[
    np.argmin(
        np.abs(
            z_values
            - target_z
        )
    )
]


mask = np.isclose(
    coordinates[:, 2],
    selected_z,
)


x = coordinates[
    mask,
    0,
]


y = coordinates[
    mask,
    1,
]


print(
    "Selected Z plane:",
    selected_z,
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def make_comparison_plot(
    truth,
    prediction,
    quantity_name,
    unit,
):

    truth_plane = truth[
        mask
    ]

    prediction_plane = prediction[
        mask
    ]

    error_plane = (
        prediction_plane
        - truth_plane
    )


    field_min = min(
        truth_plane.min(),
        prediction_plane.min(),
    )


    field_max = max(
        truth_plane.max(),
        prediction_plane.max(),
    )


    # --------------------------------------------------------
    # FEM
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    scatter = plt.scatter(
        x,
        y,
        c=truth_plane,
        s=35,
        vmin=field_min,
        vmax=field_max,
    )

    plt.colorbar(
        scatter,
        label=unit,
    )

    plt.xlabel(
        "X [mm]"
    )

    plt.ylabel(
        "Y [mm]"
    )

    plt.title(
        "{} — Abaqus FEM — {}"
        .format(
            quantity_name,
            case_id,
        )
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "{}_{}_FEM.png"
            .format(
                case_id,
                quantity_name,
            ),
        ),
        dpi=300,
    )

    plt.close()


    # --------------------------------------------------------
    # DEEPONET
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    scatter = plt.scatter(
        x,
        y,
        c=prediction_plane,
        s=35,
        vmin=field_min,
        vmax=field_max,
    )

    plt.colorbar(
        scatter,
        label=unit,
    )

    plt.xlabel(
        "X [mm]"
    )

    plt.ylabel(
        "Y [mm]"
    )

    plt.title(
        "{} — DeepONet-derived — {}"
        .format(
            quantity_name,
            case_id,
        )
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "{}_{}_DeepONet.png"
            .format(
                case_id,
                quantity_name,
            ),
        ),
        dpi=300,
    )

    plt.close()


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    scatter = plt.scatter(
        x,
        y,
        c=np.abs(
            error_plane
        ),
        s=35,
    )

    plt.colorbar(
        scatter,
        label="Absolute error",
    )

    plt.xlabel(
        "X [mm]"
    )

    plt.ylabel(
        "Y [mm]"
    )

    plt.title(
        "{} — absolute error — {}"
        .format(
            quantity_name,
            case_id,
        )
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "{}_{}_Error.png"
            .format(
                case_id,
                quantity_name,
            ),
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# COMPONENT INDICES
#
# [11,22,33,12,13,23]
# ============================================================

INDEX_11 = 0
INDEX_12 = 3


# ============================================================
# LE11
# ============================================================

make_comparison_plot(

    LE_fem[
        global_index,
        :,
        INDEX_11,
    ],

    small_strain[
        :,
        INDEX_11,
    ],

    "LE11_small",

    "strain",

)


make_comparison_plot(

    LE_fem[
        global_index,
        :,
        INDEX_11,
    ],

    log_strain[
        :,
        INDEX_11,
    ],

    "LE11_Hencky",

    "strain",

)


# ============================================================
# LE12
# ============================================================

make_comparison_plot(

    LE_fem[
        global_index,
        :,
        INDEX_12,
    ],

    small_strain[
        :,
        INDEX_12,
    ],

    "LE12_small",

    "engineering strain",

)


make_comparison_plot(

    LE_fem[
        global_index,
        :,
        INDEX_12,
    ],

    log_strain[
        :,
        INDEX_12,
    ],

    "LE12_Hencky",

    "engineering strain",

)


# ============================================================
# S11
# ============================================================

make_comparison_plot(

    S_fem[
        global_index,
        :,
        INDEX_11,
    ],

    predicted_stress[
        :,
        INDEX_11,
    ],

    "S11",

    "MPa",

)


# ============================================================
# S12
# ============================================================

make_comparison_plot(

    S_fem[
        global_index,
        :,
        INDEX_12,
    ],

    predicted_stress[
        :,
        INDEX_12,
    ],

    "S12",

    "MPa",

)


print("")
print(
    "Plots saved in:",
    PLOT_DIR,
)