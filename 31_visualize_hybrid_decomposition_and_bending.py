import os
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import (
    Rectangle,
    Circle,
    Polygon,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.getcwd()


REFERENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_online_baseline",
    "validation_fields",
    "full_reference_nodes.csv",
)


HYBRID_U_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid_online_validation",
    "hybrid_U.npy",
)


DEEPONET_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "vector_deeponet_separate",
    "baseline_deeponet_nodes.csv",
)


REACTION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_online_baseline",
    "validation_fields",
    "reaction_summary.json",
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "hybrid_visuals",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# GEOMETRY
# ============================================================

X_MIN = -12.0
X_MAX = 12.0

X_SUPPORT_LEFT = -8.0
X_SUPPORT_RIGHT = 8.0

X_LOAD = 0.0

X_M6 = -6.0
X_M2 = -1.8
X_P2 = 1.8
X_P6 = 6.0


# ============================================================
# VISUAL DEFORMATION SCALE
#
# Reduced from 20 to 8.
#
# The actual values remain unchanged.
# Only the displayed geometry is amplified.
# ============================================================

DEFORMATION_SCALE = 8.0


# ============================================================
# OUTLINE SCALE
#
# Smaller scale because the specimen thickness is only 4 mm.
# ============================================================

OUTLINE_DEFORMATION_SCALE = 5.0


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

for required_file in [

    REFERENCE_FILE,
    HYBRID_U_FILE,
    DEEPONET_FILE,
]:

    if not os.path.isfile(
        required_file
    ):

        raise FileNotFoundError(
            "\nRequired file missing:\n"
            "{}\n\n"
            "If baseline_deeponet_nodes.csv is missing, "
            "run:\n"
            "python 31A_export_baseline_deeponet_prediction.py"
            .format(
                required_file
            )
        )


# ============================================================
# COORDINATE KEY
# ============================================================

def coordinate_key(
    x,
    y,
    z,
):

    return (

        round(
            float(
                x
            ),
            5,
        ),

        round(
            float(
                y
            ),
            5,
        ),

        round(
            float(
                z
            ),
            5,
        ),
    )


# ============================================================
# DRAW SUPPORT
# ============================================================

def draw_support(
    axis,
    x,
    y,
):

    polygon = Polygon(
        [

            [
                x,
                y,
            ],

            [
                x - 0.45,
                y - 0.55,
            ],

            [
                x + 0.45,
                y - 0.55,
            ],
        ],
        fill=False,
        linewidth=1.4,
    )


    axis.add_patch(
        polygon
    )


# ============================================================
# REGION SHADING
# ============================================================

def add_region_shading(
    axis,
):

    regions = [

        (
            "FE left",
            -12.0,
            -6.0,
        ),

        (
            "NO left",
            -6.0,
            -1.8,
        ),

        (
            "FE center",
            -1.8,
            1.8,
        ),

        (
            "NO right",
            1.8,
            6.0,
        ),

        (
            "FE right",
            6.0,
            12.0,
        ),
    ]


    for (
        region_name,
        x_start,
        x_end,
    ) in regions:

        axis.axvspan(
            x_start,
            x_end,
            alpha=0.035,
        )


        axis.text(
            (
                x_start
                +
                x_end
            )
            /
            2.0,
            0.985,
            region_name,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )


    for x_interface in [

        X_M6,
        X_M2,
        X_P2,
        X_P6,
    ]:

        axis.axvline(
            x_interface,
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
        )


# ============================================================
# LOAD FULL FEM
# ============================================================

reference = pd.read_csv(
    REFERENCE_FILE
)


number_nodes = len(
    reference
)


print(
    "Full FEM nodes:",
    number_nodes,
)


# ============================================================
# LOAD HYBRID
# ============================================================

hybrid_U = np.load(
    HYBRID_U_FILE
)


if hybrid_U.shape != (
    number_nodes,
    3,
):

    raise RuntimeError(
        "Hybrid U shape mismatch: {}\n"
        "Expected ({},3)".format(
            hybrid_U.shape,
            number_nodes,
        )
    )


# ============================================================
# LOAD PURE DEEPONET
# ============================================================

deeponet = pd.read_csv(
    DEEPONET_FILE
)


required_columns = [

    "X",
    "Y",
    "Z",

    "U1",
    "U2",
    "U3",
]


for column in required_columns:

    if column not in deeponet.columns:

        raise RuntimeError(
            "DeepONet file missing column: {}".format(
                column
            )
        )


# ============================================================
# ALIGN DEEPONET TO FEM REFERENCE ORDER
# ============================================================

deeponet_map = {}


for _, row in deeponet.iterrows():

    key = coordinate_key(

        row[
            "X"
        ],

        row[
            "Y"
        ],

        row[
            "Z"
        ],
    )


    deeponet_map[
        key
    ] = np.array(
        [

            row[
                "U1"
            ],

            row[
                "U2"
            ],

            row[
                "U3"
            ],
        ],
        dtype=np.float64,
    )


deeponet_U = []


for _, row in reference.iterrows():

    key = coordinate_key(

        row[
            "X"
        ],

        row[
            "Y"
        ],

        row[
            "Z"
        ],
    )


    if key not in deeponet_map:

        raise RuntimeError(
            "DeepONet field does not contain "
            "reference coordinate {}".format(
                key
            )
        )


    deeponet_U.append(
        deeponet_map[
            key
        ]
    )


deeponet_U = np.asarray(
    deeponet_U,
    dtype=np.float64,
)


# ============================================================
# BUILD MASTER DATAFRAME
# ============================================================

data = reference[
    [
        "X",
        "Y",
        "Z",
    ]
].copy()


data[
    "FEM_U1"
] = reference[
    "U1"
].to_numpy()


data[
    "FEM_U2"
] = reference[
    "U2"
].to_numpy()


data[
    "FEM_U3"
] = reference[
    "U3"
].to_numpy()


data[
    "DeepONet_U1"
] = deeponet_U[
    :,
    0
]


data[
    "DeepONet_U2"
] = deeponet_U[
    :,
    1
]


data[
    "DeepONet_U3"
] = deeponet_U[
    :,
    2
]


data[
    "Hybrid_U1"
] = hybrid_U[
    :,
    0
]


data[
    "Hybrid_U2"
] = hybrid_U[
    :,
    1
]


data[
    "Hybrid_U3"
] = hybrid_U[
    :,
    2
]


# ============================================================
# ERROR COLUMNS
# ============================================================

for component in [

    "U1",
    "U2",
    "U3",
]:

    data[
        "DeepONet_AbsError_{}".format(
            component
        )
    ] = np.abs(

        data[
            "DeepONet_{}".format(
                component
            )
        ]

        -

        data[
            "FEM_{}".format(
                component
            )
        ]
    )


    data[
        "Hybrid_AbsError_{}".format(
            component
        )
    ] = np.abs(

        data[
            "Hybrid_{}".format(
                component
            )
        ]

        -

        data[
            "FEM_{}".format(
                component
            )
        ]
    )


    data[
        "DeepONet_SignedError_{}".format(
            component
        )
    ] = (

        data[
            "DeepONet_{}".format(
                component
            )
        ]

        -

        data[
            "FEM_{}".format(
                component
            )
        ]
    )


    data[
        "Hybrid_SignedError_{}".format(
            component
        )
    ] = (

        data[
            "Hybrid_{}".format(
                component
            )
        ]

        -

        data[
            "FEM_{}".format(
                component
            )
        ]
    )


# ============================================================
# EXTRACT SAME CENTERLINE FOR ALL THREE SOLUTIONS
#
# We choose, at each x-section, the FEM node closest to
#
# y = 2 mm
# z = 4 mm
#
# All three methods are then compared at exactly those nodes.
# ============================================================

data[
    "X_round"
] = data[
    "X"
].round(
    5
)


data[
    "DistanceToCenter"
] = (

    (
        data[
            "Y"
        ]
        -
        2.0
    )
    ** 2

    +

    (
        data[
            "Z"
        ]
        -
        4.0
    )
    ** 2
)


centerline = (

    data

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

    .reset_index(
        drop=True
    )
)


centerline.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "centerline_FEM_DeepONet_Hybrid.csv",
    ),

    index=False,
)


print("")
print(
    "Centerline points:",
    len(
        centerline
    ),
)


print(
    "Selected approximate centerline y:",
    centerline[
        "Y"
    ].median(),
)


print(
    "Selected approximate centerline z:",
    centerline[
        "Z"
    ].median(),
)


# ============================================================
# GLOBAL METRIC FUNCTION
# ============================================================

def relative_l2(
    prediction,
    truth,
):

    return (

        np.linalg.norm(
            prediction
            -
            truth
        )

        /

        (
            np.linalg.norm(
                truth
            )
            +
            1.0e-14
        )
    )


# ============================================================
# EXACT SAME-CASE GLOBAL METRICS
# ============================================================

metric_rows = []


for component in [

    "U1",
    "U2",
    "U3",
]:

    truth = data[
        "FEM_{}".format(
            component
        )
    ].to_numpy()


    deep_prediction = data[
        "DeepONet_{}".format(
            component
        )
    ].to_numpy()


    hybrid_prediction = data[
        "Hybrid_{}".format(
            component
        )
    ].to_numpy()


    deep_error = relative_l2(
        deep_prediction,
        truth,
    )


    hybrid_error = relative_l2(
        hybrid_prediction,
        truth,
    )


    metric_rows.append(
        {

            "Component":
                component,

            "DeepONet_RelativeL2":
                deep_error,

            "DeepONet_RelativeL2_percent":
                100.0
                *
                deep_error,

            "Hybrid_RelativeL2":
                hybrid_error,

            "Hybrid_RelativeL2_percent":
                100.0
                *
                hybrid_error,

            "ErrorReduction_percent":
                100.0
                *
                (
                    deep_error
                    -
                    hybrid_error
                )
                /
                deep_error,
        }
    )


metrics = pd.DataFrame(
    metric_rows
)


metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "baseline_same_case_metrics.csv",
    ),

    index=False,
)


print("")
print(
    "================================================"
)

print(
    "EXACT SAME BASELINE CASE"
)

print(
    "================================================"
)


print(
    metrics.to_string(
        index=False
    )
)


# ============================================================
# FIGURE 01
# METHOD SCHEMATIC
# ============================================================

def plot_method_schematic():

    figure, axis = plt.subplots(
        figsize=(
            13,
            4.5
        )
    )


    segments = [

        (
            -12,
            -6,
            "FEM\nsupport/contact",
        ),

        (
            -6,
            -1.8,
            "NO\nbulk",
        ),

        (
            -1.8,
            1.8,
            "FEM\nnose/contact",
        ),

        (
            1.8,
            6,
            "NO\nbulk",
        ),

        (
            6,
            12,
            "FEM\nsupport/contact",
        ),
    ]


    for (
        x_start,
        x_end,
        label,
    ) in segments:

        rectangle = Rectangle(

            (
                x_start,
                0
            ),

            x_end
            -
            x_start,

            1.0,

            alpha=0.30,
        )


        axis.add_patch(
            rectangle
        )


        axis.text(

            (
                x_start
                +
                x_end
            )
            /
            2,

            0.5,

            label,

            ha="center",
            va="center",

            fontsize=9,
        )


    draw_support(
        axis,
        -8,
        -0.05,
    )


    draw_support(
        axis,
        8,
        -0.05,
    )


    nose = Circle(

        (
            0,
            1.55
        ),

        0.40,

        fill=False,

        linewidth=1.5,
    )


    axis.add_patch(
        nose
    )


    axis.arrow(

        0,
        2.40,

        0,
        -0.55,

        head_width=0.22,

        head_length=0.15,

        length_includes_head=True,
    )


    for x_interface in [

        -6,
        -1.8,
        1.8,
        6,
    ]:

        axis.plot(

            [
                x_interface,
                x_interface
            ],

            [
                -0.1,
                1.15
            ],

            linestyle="--",

            linewidth=1.0,
        )


    axis.text(

        0,
        -1.15,

        r"$\mathbf{g}^{FE}_{\Gamma}"
        r"+"
        r"\mathbf{g}^{NO}_{\Gamma}"
        r"\approx0$"
        "     |     "
        "18 PCA interface DOFs",

        ha="center",

        fontsize=10,
    )


    axis.set_xlim(
        -13,
        13,
    )


    axis.set_ylim(
        -1.45,
        2.8,
    )


    axis.set_title(
        "Hybrid FE–NO Three-Point Bending Decomposition"
    )


    axis.axis(
        "off"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "01_hybrid_method_schematic.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 02
# ACTUAL U2 RESPONSE
#
# No visual deformation scaling.
# ============================================================

def plot_actual_u2():

    figure, axis = plt.subplots(
        figsize=(
            13,
            5.5
        )
    )


    add_region_shading(
        axis
    )


    x = centerline[
        "X"
    ]


    axis.axhline(
        0.0,
        linestyle=":",
        linewidth=1.5,
        label="Undeformed: U2 = 0",
    )


    axis.plot(

        x,

        centerline[
            "FEM_U2"
        ],

        linewidth=2.3,

        label="Full FEM",
    )


    axis.plot(

        x,

        centerline[
            "DeepONet_U2"
        ],

        linestyle="-.",

        linewidth=2.1,

        label="Full-domain DeepONet",
    )


    axis.plot(

        x,

        centerline[
            "Hybrid_U2"
        ],

        linestyle="--",

        linewidth=2.1,

        label="Hybrid FE–NO",
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(
        r"$U_2$ (mm)"
    )


    axis.set_title(
        "Actual Vertical Displacement Along the Beam"
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    axis.legend(
        loc="best"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "02_actual_U2_FEM_DeepONet_Hybrid.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 03
#
# THIS IS THE PLOT YOU ASKED FOR:
#
# 1. Undeformed
# 2. Full FEM
# 3. Full DeepONet
# 4. Hybrid FE-NO
#
# ALL ON TOP OF EACH OTHER.
# ============================================================

def plot_bending_shape():

    figure, axis = plt.subplots(
        figsize=(
            13,
            6
        )
    )


    add_region_shading(
        axis
    )


    x = centerline[
        "X"
    ].to_numpy()


    y0 = centerline[
        "Y"
    ].to_numpy()


    fem_y = (

        y0

        +
        DEFORMATION_SCALE
        *
        centerline[
            "FEM_U2"
        ].to_numpy()
    )


    deep_y = (

        y0

        +
        DEFORMATION_SCALE
        *
        centerline[
            "DeepONet_U2"
        ].to_numpy()
    )


    hybrid_y = (

        y0

        +
        DEFORMATION_SCALE
        *
        centerline[
            "Hybrid_U2"
        ].to_numpy()
    )


    axis.plot(

        x,
        y0,

        linestyle=":",

        linewidth=1.7,

        label="Undeformed beam",
    )


    axis.plot(

        x,
        fem_y,

        linewidth=2.5,

        label="Full FEM",
    )


    axis.plot(

        x,
        deep_y,

        linestyle="-.",

        linewidth=2.2,

        label="Full-domain DeepONet",
    )


    axis.plot(

        x,
        hybrid_y,

        linestyle="--",

        linewidth=2.2,

        label="Hybrid FE–NO",
    )


    axis.scatter(

        [
            -8,
            8
        ],

        [
            y0.min()
            -
            0.35,

            y0.min()
            -
            0.35
        ],

        marker="^",

        s=70,

        label="Support positions",
    )


    axis.scatter(

        [
            0
        ],

        [
            y0.max()
            +
            0.35
        ],

        marker="v",

        s=70,

        label="Loading nose",
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(
        "Visualized beam centerline position"
    )


    axis.set_title(

        "Three-Point Bending Shape Comparison\n"
        "Undeformed vs Full FEM vs Full DeepONet vs Hybrid FE–NO "
        "({}× displacement visualization)".format(
            DEFORMATION_SCALE
        )
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    axis.legend(
        loc="best",
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "03_bending_shape_comparison.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 04
#
# THIS IS THE ERROR VERSION YOU ASKED FOR:
#
# FEM is the reference.
#
# DeepONet error vs FEM
# Hybrid error vs FEM
#
# SAME AXIS.
# ============================================================

def plot_absolute_u2_error():

    figure, axis = plt.subplots(
        figsize=(
            13,
            5.5
        )
    )


    add_region_shading(
        axis
    )


    x = centerline[
        "X"
    ]


    axis.plot(

        x,

        centerline[
            "DeepONet_AbsError_U2"
        ],

        linewidth=2.3,

        label=
            r"Full DeepONet: "
            r"$|U_2^{DeepONet}-U_2^{FEM}|$",
    )


    axis.plot(

        x,

        centerline[
            "Hybrid_AbsError_U2"
        ],

        linestyle="--",

        linewidth=2.3,

        label=
            r"Hybrid FE–NO: "
            r"$|U_2^{Hybrid}-U_2^{FEM}|$",
    )


    axis.axhline(
        0.0,
        linestyle=":",
        linewidth=1.2,
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(
        r"Absolute $U_2$ error (mm)"
    )


    axis.set_title(

        "Vertical-Displacement Error Along the Beam\n"
        "Full-domain DeepONet vs Hybrid FE–NO"
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    axis.legend(
        loc="best"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "04_deeponet_vs_hybrid_u2_error_along_beam.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 05
# SIGNED ERROR
#
# Shows whether each model over- or under-predicts FEM.
# ============================================================

def plot_signed_u2_error():

    figure, axis = plt.subplots(
        figsize=(
            13,
            5.5
        )
    )


    add_region_shading(
        axis
    )


    axis.axhline(
        0.0,
        linewidth=1.3,
        linestyle=":",
    )


    axis.plot(

        centerline[
            "X"
        ],

        centerline[
            "DeepONet_SignedError_U2"
        ],

        linewidth=2.2,

        label=
            r"$U_2^{DeepONet}-U_2^{FEM}$",
    )


    axis.plot(

        centerline[
            "X"
        ],

        centerline[
            "Hybrid_SignedError_U2"
        ],

        linestyle="--",

        linewidth=2.2,

        label=
            r"$U_2^{Hybrid}-U_2^{FEM}$",
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(
        "Signed U2 error (mm)"
    )


    axis.set_title(

        "Signed Vertical-Displacement Error\n"
        "Positive/negative values show over- or under-prediction"
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    axis.legend(
        loc="best"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "05_signed_u2_error_comparison.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 06
# POINTWISE BENEFIT OF HYBRIDIZATION
#
# Positive value:
#
# Hybrid is closer to FEM than DeepONet.
# ============================================================

def plot_pointwise_improvement():

    improvement = (

        centerline[
            "DeepONet_AbsError_U2"
        ]

        -

        centerline[
            "Hybrid_AbsError_U2"
        ]
    )


    figure, axis = plt.subplots(
        figsize=(
            13,
            5.5
        )
    )


    add_region_shading(
        axis
    )


    axis.axhline(
        0.0,
        linewidth=1.2,
        linestyle=":",
    )


    axis.plot(

        centerline[
            "X"
        ],

        improvement,

        linewidth=2.2,
    )


    axis.fill_between(

        centerline[
            "X"
        ],

        improvement,

        0.0,

        where=
            improvement
            >=
            0,

        alpha=0.15,
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(

        r"$|e_{DeepONet}|-|e_{Hybrid}|$ (mm)"
    )


    axis.set_title(

        "Where Does Hybrid FE–NO Improve Over Full DeepONet?\n"
        "Positive values = Hybrid is closer to Full FEM"
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "06_pointwise_hybrid_improvement.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# MID-WIDTH SLICE
# ============================================================

unique_z = np.sort(
    data[
        "Z"
    ].unique()
)


selected_z = unique_z[
    np.argmin(
        np.abs(
            unique_z
            -
            4.0
        )
    )
]


midwidth = data[
    np.isclose(
        data[
            "Z"
        ],
        selected_z,
        atol=1.0e-5,
    )
].copy()


# ============================================================
# FIGURE 07
# WHOLE BEAM OUTLINE
#
# Corrected:
# much smaller deformation scale.
# ============================================================

def build_outline():

    rows = []


    for x in np.sort(
        midwidth[
            "X"
        ].unique()
    ):

        section = midwidth[
            np.isclose(
                midwidth[
                    "X"
                ],
                x,
                atol=1.0e-5,
            )
        ]


        bottom = section.loc[
            section[
                "Y"
            ].idxmin()
        ]


        top = section.loc[
            section[
                "Y"
            ].idxmax()
        ]


        rows.append(
            {

                "X":
                    x,

                "Bottom_Y":
                    bottom[
                        "Y"
                    ],

                "Top_Y":
                    top[
                        "Y"
                    ],

                "FEM_Bottom_U2":
                    bottom[
                        "FEM_U2"
                    ],

                "FEM_Top_U2":
                    top[
                        "FEM_U2"
                    ],

                "Deep_Bottom_U2":
                    bottom[
                        "DeepONet_U2"
                    ],

                "Deep_Top_U2":
                    top[
                        "DeepONet_U2"
                    ],

                "Hybrid_Bottom_U2":
                    bottom[
                        "Hybrid_U2"
                    ],

                "Hybrid_Top_U2":
                    top[
                        "Hybrid_U2"
                    ],
            }
        )


    return pd.DataFrame(
        rows
    )


outline = build_outline()


def plot_beam_outline():

    figure, axis = plt.subplots(
        figsize=(
            13,
            6
        )
    )


    x = outline[
        "X"
    ]


    # --------------------------------------------------------
    # UNDEFORMED
    # --------------------------------------------------------

    axis.plot(

        x,

        outline[
            "Top_Y"
        ],

        linestyle=":",

        linewidth=1.4,

        label="Undeformed beam",
    )


    axis.plot(

        x,

        outline[
            "Bottom_Y"
        ],

        linestyle=":",

        linewidth=1.4,
    )


    # --------------------------------------------------------
    # FEM
    # --------------------------------------------------------

    axis.plot(

        x,

        outline[
            "Top_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "FEM_Top_U2"
        ],

        linewidth=2.2,

        label="Full FEM",
    )


    axis.plot(

        x,

        outline[
            "Bottom_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "FEM_Bottom_U2"
        ],

        linewidth=2.2,
    )


    # --------------------------------------------------------
    # DEEPONET
    # --------------------------------------------------------

    axis.plot(

        x,

        outline[
            "Top_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "Deep_Top_U2"
        ],

        linestyle="-.",

        linewidth=2.0,

        label="Full-domain DeepONet",
    )


    axis.plot(

        x,

        outline[
            "Bottom_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "Deep_Bottom_U2"
        ],

        linestyle="-.",

        linewidth=2.0,
    )


    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    axis.plot(

        x,

        outline[
            "Top_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "Hybrid_Top_U2"
        ],

        linestyle="--",

        linewidth=2.0,

        label="Hybrid FE–NO",
    )


    axis.plot(

        x,

        outline[
            "Bottom_Y"
        ]
        +
        OUTLINE_DEFORMATION_SCALE
        *
        outline[
            "Hybrid_Bottom_U2"
        ],

        linestyle="--",

        linewidth=2.0,
    )


    axis.set_xlabel(
        "x (mm)"
    )


    axis.set_ylabel(
        "Visualized y position (mm)"
    )


    axis.set_title(

        "Deformed Beam Outline at Mid-Width "
        "(z ≈ {:.3f} mm)\n"
        "Undeformed vs FEM vs DeepONet vs Hybrid "
        "({}× deformation visualization)"
        .format(
            selected_z,
            OUTLINE_DEFORMATION_SCALE,
        )
    )


    axis.grid(
        True,
        alpha=0.25,
    )


    axis.legend(
        loc="best"
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "07_deformed_beam_outline_comparison.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 08
# REGION-BY-REGION U2 RMSE
# ============================================================

def calculate_region_rmse(
    x_min,
    x_max,
    error_column,
    include_right=False,
):

    if include_right:

        mask = (

            (
                data[
                    "X"
                ]
                >=
                x_min
            )

            &

            (
                data[
                    "X"
                ]
                <=
                x_max
            )
        )

    else:

        mask = (

            (
                data[
                    "X"
                ]
                >=
                x_min
            )

            &

            (
                data[
                    "X"
                ]
                <
                x_max
            )
        )


    values = data.loc[
        mask,
        error_column,
    ].to_numpy()


    return np.sqrt(
        np.mean(
            values
            ** 2
        )
    )


def plot_region_rmse():

    regions = [

        (
            "FE left",
            -12,
            -6,
        ),

        (
            "NO left",
            -6,
            -1.8,
        ),

        (
            "FE center",
            -1.8,
            1.8,
        ),

        (
            "NO right",
            1.8,
            6,
        ),

        (
            "FE right",
            6,
            12,
        ),
    ]


    deep_rmse = []

    hybrid_rmse = []


    for index, (
        name,
        x_start,
        x_end,
    ) in enumerate(
        regions
    ):

        include_right = (
            index
            ==
            len(
                regions
            )
            -
            1
        )


        deep_rmse.append(

            calculate_region_rmse(

                x_start,
                x_end,

                "DeepONet_AbsError_U2",

                include_right=
                    include_right,
            )
        )


        hybrid_rmse.append(

            calculate_region_rmse(

                x_start,
                x_end,

                "Hybrid_AbsError_U2",

                include_right=
                    include_right,
            )
        )


    x = np.arange(
        len(
            regions
        )
    )


    width = 0.35


    figure, axis = plt.subplots(
        figsize=(
            11,
            5.5
        )
    )


    axis.bar(

        x
        -
        width
        /
        2,

        deep_rmse,

        width,

        label="Full-domain DeepONet",
    )


    axis.bar(

        x
        +
        width
        /
        2,

        hybrid_rmse,

        width,

        label="Hybrid FE–NO",
    )


    axis.set_xticks(
        x
    )


    axis.set_xticklabels(
        [
            region[
                0
            ]
            for region in regions
        ]
    )


    axis.set_ylabel(
        r"$U_2$ RMSE (mm)"
    )


    axis.set_title(

        "Region-by-Region Error Comparison\n"
        "Showing the effect of replacing contact zones with FEM"
    )


    axis.grid(
        axis="y",
        alpha=0.25,
    )


    axis.legend()


    output_file = os.path.join(
        OUTPUT_DIR,
        "08_region_by_region_u2_error.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# FIGURE 09
# GLOBAL SAME-BASELINE ERRORS
# ============================================================

def plot_global_metrics():

    components = metrics[
        "Component"
    ].to_list()


    deep_values = metrics[
        "DeepONet_RelativeL2_percent"
    ].to_numpy()


    hybrid_values = metrics[
        "Hybrid_RelativeL2_percent"
    ].to_numpy()


    x = np.arange(
        len(
            components
        )
    )


    width = 0.35


    figure, axis = plt.subplots(
        figsize=(
            9,
            5.5
        )
    )


    deep_bars = axis.bar(

        x
        -
        width
        /
        2,

        deep_values,

        width,

        label="Full-domain DeepONet",
    )


    hybrid_bars = axis.bar(

        x
        +
        width
        /
        2,

        hybrid_values,

        width,

        label="Hybrid FE–NO",
    )


    axis.set_xticks(
        x
    )


    axis.set_xticklabels(
        components
    )


    axis.set_ylabel(
        r"Relative $L_2$ error (%)"
    )


    axis.set_title(

        "Exact Baseline-Case Full-Field Error\n"
        "Same FEM realization used for both comparisons"
    )


    axis.grid(
        axis="y",
        alpha=0.25,
    )


    axis.legend()


    for bars in [

        deep_bars,
        hybrid_bars,
    ]:

        for bar in bars:

            height = bar.get_height()


            axis.text(

                bar.get_x()
                +
                bar.get_width()
                /
                2,

                height
                +
                0.08,

                "{:.2f}%".format(
                    height
                ),

                ha="center",

                fontsize=9,
            )


    output_file = os.path.join(
        OUTPUT_DIR,
        "09_exact_baseline_global_error_comparison.png",
    )


    figure.tight_layout()


    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        figure
    )


# ============================================================
# RUN PLOTS
# ============================================================

print("")
print(
    "Generating corrected visualizations..."
)


plot_method_schematic()

plot_actual_u2()

plot_bending_shape()

plot_absolute_u2_error()

plot_signed_u2_error()

plot_pointwise_improvement()

plot_beam_outline()

plot_region_rmse()

plot_global_metrics()


print("")
print(
    "================================================"
)

print(
    "VISUALIZATION COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    "Main comparison files:"
)


for filename in [

    "02_actual_U2_FEM_DeepONet_Hybrid.png",

    "03_bending_shape_comparison.png",

    "04_deeponet_vs_hybrid_u2_error_along_beam.png",

    "05_signed_u2_error_comparison.png",

    "06_pointwise_hybrid_improvement.png",

    "07_deformed_beam_outline_comparison.png",

    "08_region_by_region_u2_error.png",

    "09_exact_baseline_global_error_comparison.png",
]:

    print(
        "  ",
        filename,
    )


print("")
print(
    "Saved under:"
)

print(
    OUTPUT_DIR
)