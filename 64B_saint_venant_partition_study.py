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
    "--mechanics_csv",
    default=
        "data/full_reference_mechanics.csv",
)


parser.add_argument(
    "--full_coordinates",
    default=
        "data/coordinates.npy",
)


parser.add_argument(
    "--output_dir",
    default=
        "results/saint_venant_partition",
)


parser.add_argument(
    "--left_contact_x",
    type=float,
    default=
        -8.0,
)


parser.add_argument(
    "--center_contact_x",
    type=float,
    default=
        0.0,
)


parser.add_argument(
    "--right_contact_x",
    type=float,
    default=
        8.0,
)


parser.add_argument(
    "--background_distance",
    type=float,
    default=
        1.8,
)


parser.add_argument(
    "--minimum_no_width",
    type=float,
    default=
        1.2,
)


parser.add_argument(
    "--current_fe_percent",
    type=float,
    default=
        65.0,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# THRESHOLDS
#
# 0.30 = aggressive
# 0.20 = balanced starting candidate
# 0.10 = more conservative FEM region
# ============================================================

THRESHOLDS = [
    0.30,
    0.20,
    0.10,
]


# ============================================================
# LOAD MECHANICS DATA
# ============================================================

mechanics = pd.read_csv(
    args.mechanics_csv
)


required_columns = [
    "X",
    "S11",
    "S12",
    "LE11",
    "LE12",
]


for column in required_columns:

    if column not in mechanics.columns:

        raise RuntimeError(
            "Missing required column: {}".format(
                column
            )
        )


# ============================================================
# LOAD FULL MESH NODE COORDINATES
#
# We need actual x-node planes because all candidate
# interfaces must lie exactly on a real FEM mesh plane.
# ============================================================

coordinates = np.load(
    args.full_coordinates
).astype(
    np.float64
)


if (
    coordinates.ndim != 2
    or
    coordinates.shape[1] != 3
):

    raise RuntimeError(
        "coordinates.npy must have shape [N,3]."
    )


node_x_planes = np.unique(
    np.round(
        coordinates[
            :,
            0
        ],
        decimals=8,
    )
)


node_x_planes.sort()


domain_x_min = float(
    np.min(
        node_x_planes
    )
)


domain_x_max = float(
    np.max(
        node_x_planes
    )
)


# ============================================================
# CONTACT LOCATIONS
# ============================================================

left_contact = float(
    args.left_contact_x
)


center_contact = float(
    args.center_contact_x
)


right_contact = float(
    args.right_contact_x
)


contacts = np.asarray(
    [
        left_contact,
        center_contact,
        right_contact,
    ],
    dtype=np.float64,
)


# Midpoints are only used to decide which contact a
# high-disturbance point belongs to.
left_center_midpoint = (
    0.5
    *
    (
        left_contact
        +
        center_contact
    )
)


center_right_midpoint = (
    0.5
    *
    (
        center_contact
        +
        right_contact
    )
)


print("")
print(
    "=========================================="
)

print(
    "STEP 64B"
)

print(
    "BACKGROUND-SUBTRACTED CONTACT DISTURBANCE"
)

print(
    "=========================================="
)

print(
    "Domain:",
    domain_x_min,
    "to",
    domain_x_max,
)

print(
    "Contacts:",
    contacts,
)

print(
    "Background distance:",
    args.background_distance,
    "mm",
)

print("")


# ============================================================
# BUILD CROSS-SECTION PROFILE
#
# Abaqus C3D8R integration-point centroids occur halfway
# between x-node planes, so we group by element centroid X.
# ============================================================

mechanics = mechanics.copy()


mechanics[
    "X_Group"
] = np.round(
    mechanics[
        "X"
    ].to_numpy(
        dtype=np.float64
    ),
    decimals=6,
)


x_sections = np.sort(
    mechanics[
        "X_Group"
    ].unique()
)


profile_rows = []


for x_value in x_sections:

    section = mechanics[
        mechanics[
            "X_Group"
        ]
        ==
        x_value
    ]


    row = {
        "X":
            float(
                x_value
            )
    }


    for field_name in [
        "S11",
        "S12",
        "LE11",
        "LE12",
    ]:

        values = section[
            field_name
        ].to_numpy(
            dtype=np.float64
        )


        row[
            "{}_RMS".format(
                field_name
            )
        ] = float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        )


        row[
            "{}_STD".format(
                field_name
            )
        ] = float(
            np.std(
                values
            )
        )


    profile_rows.append(
        row
    )


profile = pd.DataFrame(
    profile_rows
)


profile = profile.sort_values(
    "X"
).reset_index(
    drop=True
)


x_profile = profile[
    "X"
].to_numpy(
    dtype=np.float64
)


# ============================================================
# DISTANCE FROM NEAREST CONTACT
# ============================================================

distance_from_contact = np.min(
    np.abs(
        x_profile.reshape(
            -1,
            1
        )
        -
        contacts.reshape(
            1,
            -1
        )
    ),
    axis=1,
)


profile[
    "DistanceFromNearestContact_mm"
] = distance_from_contact


# ============================================================
# STRUCTURAL REGIONS
#
# We do NOT use one global background slope.
#
# Ordinary beam response has different smooth slopes in:
#
#   1. left overhang
#   2. left internal span
#   3. right internal span
#   4. right overhang
#
# Therefore a separate background derivative is estimated
# in each region.
# ============================================================

region_id = np.full(
    len(
        x_profile
    ),
    -1,
    dtype=int,
)


region_id[
    x_profile
    <
    left_contact
] = 0


region_id[
    (
        x_profile
        >=
        left_contact
    )
    &
    (
        x_profile
        <
        center_contact
    )
] = 1


region_id[
    (
        x_profile
        >=
        center_contact
    )
    &
    (
        x_profile
        <
        right_contact
    )
] = 2


region_id[
    x_profile
    >=
    right_contact
] = 3


profile[
    "StructuralRegion"
] = region_id


REGION_NAMES = {
    0:
        "left_overhang",

    1:
        "left_span",

    2:
        "right_span",

    3:
        "right_overhang",
}


# ============================================================
# BACKGROUND-SUBTRACTED GRADIENT
#
# Old method:
#
#       |dq/dx|
#
# Problem:
# ordinary bending gives a large dq/dx.
#
# New method:
#
#       |dq/dx - background_slope|
#
# The background slope is the median gradient in points
# sufficiently far from all three contacts.
#
# This detects LOCAL disturbances rather than normal bending.
# ============================================================

field_names = [
    "S11",
    "S12",
    "LE11",
    "LE12",
]


normalized_disturbance_columns = []


background_rows = []


for field_name in field_names:

    rms_column = (
        "{}_RMS".format(
            field_name
        )
    )


    values = profile[
        rms_column
    ].to_numpy(
        dtype=np.float64
    )


    raw_gradient = np.gradient(
        values,
        x_profile,
    )


    profile[
        "{}_RawGradient".format(
            field_name
        )
    ] = raw_gradient


    disturbance_gradient = np.zeros_like(
        raw_gradient
    )


    background_gradient = np.zeros_like(
        raw_gradient
    )


    for current_region in [
        0,
        1,
        2,
        3,
    ]:

        region_mask = (
            region_id
            ==
            current_region
        )


        background_mask = (
            region_mask
            &
            (
                distance_from_contact
                >=
                args.background_distance
            )
        )


        # ----------------------------------------------------
        # Prefer points away from contacts.
        #
        # If there are too few, use all points in that
        # structural region.
        # ----------------------------------------------------

        if np.sum(
            background_mask
        ) < 2:

            background_mask = (
                region_mask
            )


        if np.sum(
            background_mask
        ) == 0:

            raise RuntimeError(
                "No points found in structural region {}."
                .format(
                    current_region
                )
            )


        background_slope = float(
            np.median(
                raw_gradient[
                    background_mask
                ]
            )
        )


        background_gradient[
            region_mask
        ] = background_slope


        disturbance_gradient[
            region_mask
        ] = np.abs(
            raw_gradient[
                region_mask
            ]
            -
            background_slope
        )


        background_rows.append(
            {
                "Field":
                    field_name,

                "RegionID":
                    current_region,

                "RegionName":
                    REGION_NAMES[
                        current_region
                    ],

                "BackgroundSlope":
                    background_slope,

                "NumberBackgroundPoints":
                    int(
                        np.sum(
                            background_mask
                        )
                    ),
            }
        )


    profile[
        "{}_BackgroundGradient".format(
            field_name
        )
    ] = background_gradient


    profile[
        "{}_DisturbanceGradient".format(
            field_name
        )
    ] = disturbance_gradient


    maximum = float(
        np.max(
            disturbance_gradient
        )
    )


    if maximum <= 1.0e-30:

        normalized = np.zeros_like(
            disturbance_gradient
        )

    else:

        normalized = (
            disturbance_gradient
            /
            maximum
        )


    normalized_column = (
        "{}_DisturbanceNormalized"
        .format(
            field_name
        )
    )


    profile[
        normalized_column
    ] = normalized


    normalized_disturbance_columns.append(
        normalized_column
    )


# ============================================================
# COMBINED CONTACT-DISTURBANCE INDICATOR
#
# Maximum is conservative:
#
# if ANY important mechanics quantity detects localization,
# that position is considered disturbed.
# ============================================================

profile[
    "CombinedContactDisturbance"
] = profile[
    normalized_disturbance_columns
].max(
    axis=1
)


# ============================================================
# NODE-PLANE HELPERS
# ============================================================

def node_plane_at_or_below(
    x_value
):

    available = node_x_planes[
        node_x_planes
        <=
        x_value
        +
        1.0e-8
    ]


    if len(
        available
    ) == 0:

        raise RuntimeError(
            "No node plane <= {}".format(
                x_value
            )
        )


    return float(
        available[
            -1
        ]
    )


def node_plane_at_or_above(
    x_value
):

    available = node_x_planes[
        node_x_planes
        >=
        x_value
        -
        1.0e-8
    ]


    if len(
        available
    ) == 0:

        raise RuntimeError(
            "No node plane >= {}".format(
                x_value
            )
        )


    return float(
        available[
            0
        ]
    )


# ============================================================
# CONTACT-ZONE HELPER
# ============================================================

def disturbed_span(
    contact_name,
    contact_x,
    zone_mask,
    threshold,
):

    selected_mask = (
        zone_mask
        &
        (
            profile[
                "CombinedContactDisturbance"
            ].to_numpy(
                dtype=np.float64
            )
            >=
            threshold
        )
    )


    selected_x = x_profile[
        selected_mask
    ]


    # --------------------------------------------------------
    # Fallback:
    #
    # if a very high threshold leaves no point, retain the
    # integration-point section nearest to that contact.
    # --------------------------------------------------------

    if len(
        selected_x
    ) == 0:

        possible_indices = np.where(
            zone_mask
        )[0]


        if len(
            possible_indices
        ) == 0:

            raise RuntimeError(
                "No profile points for contact {}."
                .format(
                    contact_name
                )
            )


        nearest_local = possible_indices[
            np.argmin(
                np.abs(
                    x_profile[
                        possible_indices
                    ]
                    -
                    contact_x
                )
            )
        ]


        selected_x = np.asarray(
            [
                x_profile[
                    nearest_local
                ]
            ],
            dtype=np.float64,
        )


    minimum_x = float(
        np.min(
            selected_x
        )
    )


    maximum_x = float(
        np.max(
            selected_x
        )
    )


    # --------------------------------------------------------
    # Always make sure the physical contact itself lies
    # inside the retained FE region.
    # --------------------------------------------------------

    minimum_x = min(
        minimum_x,
        contact_x,
    )


    maximum_x = max(
        maximum_x,
        contact_x,
    )


    # --------------------------------------------------------
    # Integration-point centroids are enclosed by the
    # nearest real node planes.
    # --------------------------------------------------------

    fe_left = node_plane_at_or_below(
        minimum_x
    )


    fe_right = node_plane_at_or_above(
        maximum_x
    )


    return {
        "raw_min":
            minimum_x,

        "raw_max":
            maximum_x,

        "fe_left":
            fe_left,

        "fe_right":
            fe_right,
    }


# ============================================================
# CONTACT OWNERSHIP ZONES
# ============================================================

left_support_zone = (
    x_profile
    <
    left_center_midpoint
)


center_zone = (
    (
        x_profile
        >=
        left_center_midpoint
    )
    &
    (
        x_profile
        <=
        center_right_midpoint
    )
)


right_support_zone = (
    x_profile
    >
    center_right_midpoint
)


# ============================================================
# BUILD 7-REGION CANDIDATES
# ============================================================

candidate_rows = []


for threshold in THRESHOLDS:

    left_support = disturbed_span(

        contact_name=
            "left_support",

        contact_x=
            left_contact,

        zone_mask=
            left_support_zone,

        threshold=
            threshold,
    )


    center_support = disturbed_span(

        contact_name=
            "center_nose",

        contact_x=
            center_contact,

        zone_mask=
            center_zone,

        threshold=
            threshold,
    )


    right_support = disturbed_span(

        contact_name=
            "right_support",

        contact_x=
            right_contact,

        zone_mask=
            right_support_zone,

        threshold=
            threshold,
    )


    # ========================================================
    # SIX FE-NO INTERFACES
    #
    # domain
    # |
    # NO outer-left
    # |
    # left_outer
    # |
    # FE left support
    # |
    # left_inner
    # |
    # NO inner-left
    # |
    # center_left
    # |
    # FE center
    # |
    # center_right
    # |
    # NO inner-right
    # |
    # right_inner
    # |
    # FE right support
    # |
    # right_outer
    # |
    # NO outer-right
    # |
    # domain
    # ========================================================

    left_outer = (
        left_support[
            "fe_left"
        ]
    )


    left_inner = (
        left_support[
            "fe_right"
        ]
    )


    center_left = (
        center_support[
            "fe_left"
        ]
    )


    center_right = (
        center_support[
            "fe_right"
        ]
    )


    right_inner = (
        right_support[
            "fe_left"
        ]
    )


    right_outer = (
        right_support[
            "fe_right"
        ]
    )


    # ========================================================
    # NO WIDTHS
    # ========================================================

    no_outer_left_width = (
        left_outer
        -
        domain_x_min
    )


    no_inner_left_width = (
        center_left
        -
        left_inner
    )


    no_inner_right_width = (
        right_inner
        -
        center_right
    )


    no_outer_right_width = (
        domain_x_max
        -
        right_outer
    )


    # ========================================================
    # FE PATCH LENGTHS
    # ========================================================

    fe_left_length = (
        left_inner
        -
        left_outer
    )


    fe_center_length = (
        center_right
        -
        center_left
    )


    fe_right_length = (
        right_outer
        -
        right_inner
    )


    total_fe_length = (
        fe_left_length
        +
        fe_center_length
        +
        fe_right_length
    )


    total_domain_length = (
        domain_x_max
        -
        domain_x_min
    )


    fe_fraction = (
        total_fe_length
        /
        total_domain_length
    )


    fe_percent = (
        100.0
        *
        fe_fraction
    )


    # ========================================================
    # VALIDITY
    # ========================================================

    ordering_valid = (

        domain_x_min
        <
        left_outer
        <
        left_inner
        <
        center_left
        <
        center_right
        <
        right_inner
        <
        right_outer
        <
        domain_x_max
    )


    no_width_valid = (

        no_outer_left_width
        >=
        args.minimum_no_width

        and

        no_inner_left_width
        >=
        args.minimum_no_width

        and

        no_inner_right_width
        >=
        args.minimum_no_width

        and

        no_outer_right_width
        >=
        args.minimum_no_width
    )


    candidate_valid = (
        ordering_valid
        and
        no_width_valid
    )


    candidate_rows.append(
        {
            "Threshold":
                threshold,

            "DomainXMin":
                domain_x_min,

            "DomainXMax":
                domain_x_max,

            # -----------------------------------------------
            # Raw detected disturbance spans
            # -----------------------------------------------

            "LeftSupportRawMin":
                left_support[
                    "raw_min"
                ],

            "LeftSupportRawMax":
                left_support[
                    "raw_max"
                ],

            "CenterRawMin":
                center_support[
                    "raw_min"
                ],

            "CenterRawMax":
                center_support[
                    "raw_max"
                ],

            "RightSupportRawMin":
                right_support[
                    "raw_min"
                ],

            "RightSupportRawMax":
                right_support[
                    "raw_max"
                ],

            # -----------------------------------------------
            # Six actual interfaces
            # -----------------------------------------------

            "left_outer_x":
                left_outer,

            "left_inner_x":
                left_inner,

            "center_left_x":
                center_left,

            "center_right_x":
                center_right,

            "right_inner_x":
                right_inner,

            "right_outer_x":
                right_outer,

            # -----------------------------------------------
            # NO lengths
            # -----------------------------------------------

            "NO_OuterLeft_mm":
                no_outer_left_width,

            "NO_InnerLeft_mm":
                no_inner_left_width,

            "NO_InnerRight_mm":
                no_inner_right_width,

            "NO_OuterRight_mm":
                no_outer_right_width,

            # -----------------------------------------------
            # FEM lengths
            # -----------------------------------------------

            "FE_LeftSupport_mm":
                fe_left_length,

            "FE_Center_mm":
                fe_center_length,

            "FE_RightSupport_mm":
                fe_right_length,

            "Total_FE_Length_mm":
                total_fe_length,

            "FE_Fraction":
                fe_fraction,

            "FE_Percent":
                fe_percent,

            "Current_FE_Percent":
                args.current_fe_percent,

            "FE_PercentagePoint_Reduction":
                (
                    args.current_fe_percent
                    -
                    fe_percent
                ),

            "OrderingValid":
                ordering_valid,

            "NOWidthValid":
                no_width_valid,

            "CandidateValid":
                candidate_valid,
        }
    )


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# SAVE PROFILE
# ============================================================

profile_file = os.path.join(
    args.output_dir,
    "contact_disturbance_profile.csv",
)


profile.to_csv(
    profile_file,
    index=False,
)


# ============================================================
# SAVE BACKGROUND SLOPES
# ============================================================

background_dataframe = pd.DataFrame(
    background_rows
)


background_file = os.path.join(
    args.output_dir,
    "background_gradient_slopes.csv",
)


background_dataframe.to_csv(
    background_file,
    index=False,
)


# ============================================================
# SAVE CANDIDATES
# ============================================================

candidate_file = os.path.join(
    args.output_dir,
    "partition_candidates_7region.csv",
)


candidates.to_csv(
    candidate_file,
    index=False,
)


# ============================================================
# FIGURE 1 — CONTACT DISTURBANCE
# ============================================================

figure = plt.figure(
    figsize=(
        11,
        5,
    )
)


plt.plot(
    profile[
        "X"
    ],
    profile[
        "CombinedContactDisturbance"
    ],
    marker="o",
)


for contact_x in contacts:

    plt.axvline(
        contact_x,
        linestyle="--",
        alpha=0.5,
    )


for threshold in THRESHOLDS:

    plt.axhline(
        threshold,
        linestyle=":",
        alpha=0.6,
    )


plt.xlabel(
    "Beam x coordinate (mm)"
)


plt.ylabel(
    "Normalized contact-disturbance indicator"
)


plt.title(
    "Background-Subtracted Contact Disturbance Along Beam"
)


plt.grid(
    True,
    alpha=0.25,
)


plt.tight_layout()


disturbance_plot = os.path.join(
    args.output_dir,
    "contact_disturbance_indicator.png",
)


plt.savefig(
    disturbance_plot,
    dpi=250,
    bbox_inches="tight",
)


plt.close(
    figure
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64B COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    candidates.to_string(
        index=False
    )
)

print("")
print(
    "Saved:"
)

print(
    profile_file
)

print(
    background_file
)

print(
    candidate_file
)

print(
    disturbance_plot
)