import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--segment",
    choices=[
        "left",
        "right",
    ],
    required=True,
)

parser.add_argument(
    "--split",
    choices=[
        "validation",
        "test",
    ],
    default="test",
)

parser.add_argument(
    "--case_index",
    type=int,
    default=-1,
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = os.path.join(
    "results",
    "v4_final_{}".format(
        args.split
    ),
)


INPUT_FILE = os.path.join(
    INPUT_DIR,
    "{}_{}_predictions.npz".format(
        args.segment,
        args.split,
    ),
)


OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "plots",
    args.segment,
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


data = np.load(
    INPUT_FILE,
    allow_pickle=True,
)


case_ids = data[
    "case_ids"
]


coordinates = data[
    "coordinates"
]


ip_coordinates = data[
    "ip_coordinates"
]


U_pred = data[
    "U_pred"
]


U_true = data[
    "U_true"
]


LE_pred = data[
    "LE_pred"
]


LE_true = data[
    "LE_true"
]


S_pred = data[
    "S_pred"
]


S_true = data[
    "S_true"
]


# ============================================================
# CHOOSE CASE
#
# Default:
# median U2-error case.
# ============================================================

case_errors = []


for case_index in range(
    U_pred.shape[0]
):

    value = (
        np.linalg.norm(
            U_pred[
                case_index,
                :,
                1
            ]
            -
            U_true[
                case_index,
                :,
                1
            ]
        )
        /
        (
            np.linalg.norm(
                U_true[
                    case_index,
                    :,
                    1
                ]
            )
            +
            1.0e-14
        )
    )


    case_errors.append(
        value
    )


case_errors = np.asarray(
    case_errors
)


if args.case_index >= 0:

    selected_case = (
        args.case_index
    )

else:

    order = np.argsort(
        case_errors
    )

    selected_case = order[
        len(
            order
        )
        //
        2
    ]


print(
    "Selected case:",
    case_ids[
        selected_case
    ]
)


# ============================================================
# MID-WIDTH SLICE
# ============================================================

z_mid_node = coordinates[
    np.argmin(
        np.abs(
            coordinates[:, 2]
            -
            np.median(
                coordinates[:, 2]
            )
        )
    ),
    2
]


z_mid_ip = ip_coordinates[
    np.argmin(
        np.abs(
            ip_coordinates[:, 2]
            -
            np.median(
                ip_coordinates[:, 2]
            )
        )
    ),
    2
]


node_mask = np.isclose(
    coordinates[:, 2],
    z_mid_node,
    atol=1.0e-5,
)


ip_mask = np.isclose(
    ip_coordinates[:, 2],
    z_mid_ip,
    atol=1.0e-5,
)


# ============================================================
# THREE-PANEL PLOT
# ============================================================

def make_plot(
    coordinates_3d,
    mask,
    truth,
    prediction,
    field_name,
    units,
):

    xy = coordinates_3d[
        mask
    ][
        :,
        :2
    ]


    truth = truth[
        mask
    ]


    prediction = prediction[
        mask
    ]


    error = np.abs(
        prediction
        -
        truth
    )


    triangulation = (
        mtri.Triangulation(
            xy[:, 0],
            xy[:, 1],
        )
    )


    field_min = min(
        float(
            np.min(
                truth
            )
        ),
        float(
            np.min(
                prediction
            )
        ),
    )


    field_max = max(
        float(
            np.max(
                truth
            )
        ),
        float(
            np.max(
                prediction
            )
        ),
    )


    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4,
        ),
    )


    plot_1 = axes[
        0
    ].tricontourf(
        triangulation,
        truth,
        levels=30,
        vmin=
            field_min,
        vmax=
            field_max,
    )


    axes[
        0
    ].set_title(
        "Full FEM"
    )


    plot_2 = axes[
        1
    ].tricontourf(
        triangulation,
        prediction,
        levels=30,
        vmin=
            field_min,
        vmax=
            field_max,
    )


    axes[
        1
    ].set_title(
        "Neural Operator"
    )


    plot_3 = axes[
        2
    ].tricontourf(
        triangulation,
        error,
        levels=30,
    )


    axes[
        2
    ].set_title(
        "Absolute error"
    )


    for axis in axes:

        axis.set_xlabel(
            "x (mm)"
        )

        axis.set_ylabel(
            "y (mm)"
        )

        axis.set_aspect(
            "equal"
        )


    figure.colorbar(
        plot_1,
        ax=axes[
            :2
        ],
        label=
            "{} ({})".format(
                field_name,
                units,
            ),
    )


    figure.colorbar(
        plot_3,
        ax=axes[
            2
        ],
        label=
            "Absolute error",
    )


    figure.suptitle(
        "{} — {} — Case {}".format(
            args.segment.upper(),
            field_name,
            case_ids[
                selected_case
            ],
        )
    )


    figure.tight_layout()


    output_file = os.path.join(
        OUTPUT_DIR,
        "{}_FEM_NO_error.png".format(
            field_name
        ),
    )


    figure.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# U2
# ============================================================

make_plot(

    coordinates,

    node_mask,

    U_true[
        selected_case,
        :,
        1
    ],

    U_pred[
        selected_case,
        :,
        1
    ],

    "U2",

    "mm",
)


# ============================================================
# IMPORTANT MECHANICS FIELDS
# ============================================================

mechanics_fields = [

    (
        "LE11",
        LE_true[
            selected_case,
            :,
            0
        ],
        LE_pred[
            selected_case,
            :,
            0
        ],
        "-"
    ),

    (
        "LE12",
        LE_true[
            selected_case,
            :,
            3
        ],
        LE_pred[
            selected_case,
            :,
            3
        ],
        "-"
    ),

    (
        "S11",
        S_true[
            selected_case,
            :,
            0
        ],
        S_pred[
            selected_case,
            :,
            0
        ],
        "MPa"
    ),

    (
        "S12",
        S_true[
            selected_case,
            :,
            3
        ],
        S_pred[
            selected_case,
            :,
            3
        ],
        "MPa"
    ),
]


for (
    name,
    truth,
    prediction,
    units,
) in mechanics_fields:

    make_plot(

        ip_coordinates,

        ip_mask,

        truth,

        prediction,

        name,

        units,
    )


# ============================================================
# X-WISE U2 ERROR
# ============================================================

x_values = np.sort(
    np.unique(
        np.round(
            coordinates[
                :,
                0
            ],
            8,
        )
    )
)


x_error = []


for x in x_values:

    mask = np.isclose(
        coordinates[
            :,
            0
        ],
        x,
        atol=1.0e-6,
    )


    error = (
        U_pred[
            selected_case,
            mask,
            1
        ]
        -
        U_true[
            selected_case,
            mask,
            1
        ]
    )


    rms = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    x_error.append(
        rms
    )


figure = plt.figure(
    figsize=(
        9,
        4,
    )
)


plt.plot(
    x_values,
    x_error,
    marker="o",
)


plt.xlabel(
    "x (mm)"
)


plt.ylabel(
    "U2 RMS error (mm)"
)


plt.title(
    "NO error along local block"
)


plt.grid(
    True,
    alpha=0.3,
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "U2_error_vs_x.png",
    ),
    dpi=250,
)


plt.close(
    figure
)


print("")
print(
    "Plots saved to:"
)

print(
    OUTPUT_DIR
)