import os
import json

import numpy as np
import pandas as pd

import torch


from src.hybrid_bulk_operator_v4_7region import (
    HybridBulkOperatorV4SevenRegion,
)


from src.hard_interface_compatibility_7region import (
    build_hard_boundary_context_7region,
    apply_hard_compatibility_7region,
)


# ============================================================
# PATHS
# ============================================================

ONLINE_DIR = os.path.join(
    "data",
    "hybrid_online_7region_direct",
)


VALIDATION_DIR = os.path.join(
    ONLINE_DIR,
    "validation_fields",
)


SOLUTION_FILE = os.path.join(
    ONLINE_DIR,
    "online_solution_7region.npz",
)


DATA_ROOT = os.path.join(
    "data",
    "hybrid_bulk_7region",
)


DIRECT_ROOT = os.path.join(
    "results",
    "v4_7region_direct",
)


INTERFACE_DIR = os.path.join(
    "data",
    "candidate_partition_interfaces_7region",
)


OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_online_7region_direct_validation",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("")
print(
    "Device:",
    DEVICE
)


# ============================================================
# CONFIGURATION
# ============================================================

SEGMENTS = {

    "outer_left": {

        "left_interface":
            None,

        "right_interface":
            "left_outer",
    },

    "inner_left": {

        "left_interface":
            "left_inner",

        "right_interface":
            "center_left",
    },

    "inner_right": {

        "left_interface":
            "center_right",

        "right_interface":
            "right_inner",
    },

    "outer_right": {

        "left_interface":
            "right_outer",

        "right_interface":
            None,
    },
}


REGION_ORDER = [

    "NO_OL",
    "FE_L",
    "NO_L",
    "FE_C",
    "NO_R",
    "FE_R",
    "NO_OR",
]


COMPONENTS = [

    "11",
    "22",
    "33",
    "12",
    "13",
    "23",
]


# ============================================================
# INTEGRATION-POINT MATCHING TOLERANCE
#
# Local NO FEM data and the original full FEM use the same
# physical mesh, but coordinates passed through different
# floating-point representations.
#
# 1e-4 mm is tiny relative to the ~0.6 mm mesh spacing.
# ============================================================

IP_MATCH_TOLERANCE_MM = 1.0e-4


# ============================================================
# LOAD BROYDEN SOLUTION
# ============================================================

if not os.path.isfile(
    SOLUTION_FILE
):

    raise FileNotFoundError(
        SOLUTION_FILE
    )


solution = np.load(
    SOLUTION_FILE
)


E1 = float(
    solution[
        "E1"
    ][0]
)


E2 = float(
    solution[
        "E2"
    ][0]
)


G12 = float(
    solution[
        "G12"
    ][0]
)


coefficients = {}


for interface_name in [

    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",

]:

    coefficients[
        interface_name
    ] = solution[
        "c_{}".format(
            interface_name
        )
    ].astype(
        np.float32
    )


print("")
print(
    "=========================================="
)

print(
    "BRODYEN SOLUTION"
)

print(
    "=========================================="
)

print(
    "E1:",
    E1
)

print(
    "E2:",
    E2
)

print(
    "G12:",
    G12
)


# ============================================================
# NODE COORDINATE KEY
#
# Nodal matching already worked correctly. Keep this method
# for the nodal displacement assembly.
# ============================================================

def coordinate_key(
    xyz,
):

    values = np.asarray(
        xyz,
        dtype=np.float64,
    )


    return tuple(
        [
            round(
                float(
                    value
                ),
                5,
            )

            for value
            in values
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    truth,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )


    truth = np.asarray(
        truth,
        dtype=np.float64,
    )


    error = (
        prediction
        -
        truth
    )


    denominator = np.linalg.norm(
        truth
    )


    relative_l2 = (
        np.linalg.norm(
            error
        )
        /
        (
            denominator
            +
            1.0e-14
        )
    )


    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    mae = np.mean(
        np.abs(
            error
        )
    )


    ss_res = np.sum(
        error ** 2
    )


    ss_tot = np.sum(
        (
            truth
            -
            truth.mean()
        ) ** 2
    )


    if ss_tot > 1.0e-14:

        r2 = (
            1.0
            -
            ss_res
            /
            ss_tot
        )

    else:

        r2 = np.nan


    return {

        "Relative_L2":
            float(
                relative_l2
            ),

        "Relative_L2_percent":
            float(
                100.0
                *
                relative_l2
            ),

        "RMSE":
            float(
                rmse
            ),

        "MAE":
            float(
                mae
            ),

        "R2":
            float(
                r2
            ),
    }


# ============================================================
# BUILD PHYSICAL BRANCH
# ============================================================

def build_branch(
    segment,
):

    material = np.asarray(
        [
            E1,
            E2,
            G12,
        ],
        dtype=np.float32,
    )


    if segment == "outer_left":

        return np.concatenate(
            [
                material,

                coefficients[
                    "left_outer"
                ],
            ]
        )


    if segment == "inner_left":

        return np.concatenate(
            [
                material,

                coefficients[
                    "left_inner"
                ],

                coefficients[
                    "center_left"
                ],
            ]
        )


    if segment == "inner_right":

        return np.concatenate(
            [
                material,

                coefficients[
                    "center_right"
                ],

                coefficients[
                    "right_inner"
                ],
            ]
        )


    if segment == "outer_right":

        return np.concatenate(
            [
                material,

                coefficients[
                    "right_outer"
                ],
            ]
        )


    raise RuntimeError(
        "Unknown segment: {}".format(
            segment
        )
    )


# ============================================================
# PREDICT ONE SEGMENT USING FIVE DIRECT V4 MEMBERS
# ============================================================

def predict_segment(
    segment,
):

    print("")
    print(
        "=========================================="
    )

    print(
        "PREDICTING:",
        segment.upper()
    )

    print(
        "=========================================="
    )


    data_dir = os.path.join(
        DATA_ROOT,
        segment,
    )


    model_root = os.path.join(
        DIRECT_ROOT,
        segment,
    )


    coordinates = np.load(
        os.path.join(
            data_dir,
            "coordinates.npy",
        )
    ).astype(
        np.float32
    )


    ip_coordinates = np.load(
        os.path.join(
            data_dir,
            "ip_coordinates.npy",
        )
    ).astype(
        np.float32
    )


    branch_physical = build_branch(
        segment
    )


    config = SEGMENTS[
        segment
    ]


    # ========================================================
    # HARD INTERFACE COMPATIBILITY CONTEXT
    # ========================================================

    hard_context = (
        build_hard_boundary_context_7region(

            coordinates=
                coordinates,

            interface_dir=
                INTERFACE_DIR,

            left_interface_name=
                config[
                    "left_interface"
                ],

            right_interface_name=
                config[
                    "right_interface"
                ],

            device=
                DEVICE,
        )
    )


    member_U = []

    member_LE = []

    member_S = []


    # ========================================================
    # FIVE DIRECT V4 MEMBERS
    # ========================================================

    for member_number in range(
        1,
        6,
    ):

        checkpoint_file = os.path.join(

            model_root,

            "member_{:02d}".format(
                member_number
            ),

            "best_v4_7region.pt",
        )


        if not os.path.isfile(
            checkpoint_file
        ):

            raise FileNotFoundError(
                checkpoint_file
            )


        checkpoint = torch.load(
            checkpoint_file,
            map_location=DEVICE,
            weights_only=False,
        )


        model = HybridBulkOperatorV4SevenRegion(

            branch_dim=
                checkpoint[
                    "branch_dim"
                ],

            number_force_coefficients=
                checkpoint[
                    "number_force_coefficients"
                ],

            hidden_dim=
                checkpoint[
                    "hidden_dim"
                ],

            latent_dim=
                checkpoint[
                    "latent_dim"
                ],

            depth=
                checkpoint[
                    "depth"
                ],
        ).to(
            DEVICE
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.eval()


        # ====================================================
        # NORMALIZATION
        # ====================================================

        branch_mean = np.asarray(
            checkpoint[
                "branch_mean"
            ],
            dtype=np.float32,
        )


        branch_std = np.asarray(
            checkpoint[
                "branch_std"
            ],
            dtype=np.float32,
        )


        coordinate_min = np.asarray(
            checkpoint[
                "coordinate_min"
            ],
            dtype=np.float32,
        )


        coordinate_max = np.asarray(
            checkpoint[
                "coordinate_max"
            ],
            dtype=np.float32,
        )


        U_mean = np.asarray(
            checkpoint[
                "U_mean"
            ],
            dtype=np.float32,
        )


        U_std = np.asarray(
            checkpoint[
                "U_std"
            ],
            dtype=np.float32,
        )


        LE_mean = np.asarray(
            checkpoint[
                "LE_mean"
            ],
            dtype=np.float32,
        )


        LE_std = np.asarray(
            checkpoint[
                "LE_std"
            ],
            dtype=np.float32,
        )


        S_mean = np.asarray(
            checkpoint[
                "S_mean"
            ],
            dtype=np.float32,
        )


        S_std = np.asarray(
            checkpoint[
                "S_std"
            ],
            dtype=np.float32,
        )


        # ====================================================
        # NORMALIZE INPUT
        # ====================================================

        branch_normalized = (
            branch_physical
            -
            branch_mean
        ) / branch_std


        coordinate_range = np.maximum(
            coordinate_max
            -
            coordinate_min,
            1.0e-8,
        )


        coordinates_normalized = (
            2.0
            *
            (
                coordinates
                -
                coordinate_min
            )
            /
            coordinate_range
            -
            1.0
        )


        ip_coordinates_normalized = (
            2.0
            *
            (
                ip_coordinates
                -
                coordinate_min
            )
            /
            coordinate_range
            -
            1.0
        )


        branch_tensor = torch.tensor(

            branch_normalized,

            dtype=torch.float32,

            device=DEVICE,

        ).reshape(
            1,
            -1,
        )


        branch_physical_tensor = torch.tensor(

            branch_physical,

            dtype=torch.float32,

            device=DEVICE,

        ).reshape(
            1,
            -1,
        )


        coordinate_tensor = torch.tensor(

            coordinates_normalized,

            dtype=torch.float32,

            device=DEVICE,
        )


        ip_tensor = torch.tensor(

            ip_coordinates_normalized,

            dtype=torch.float32,

            device=DEVICE,
        )


        # ====================================================
        # DIRECT V4 FORWARD
        # ====================================================

        with torch.no_grad():

            (
                U_raw_n,
                LE_n,
                S_n,
                _,
            ) = model(

                branch_tensor,

                coordinate_tensor,

                ip_tensor,
            )


            # =================================================
            # DISPLACEMENT → PHYSICAL
            # =================================================

            U_raw = (

                U_raw_n

                *

                torch.tensor(
                    U_std,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    1,
                    3,
                )

                +

                torch.tensor(
                    U_mean,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(
                    1,
                    1,
                    3,
                )
            )


            # =================================================
            # HARD INTERFACE COMPATIBILITY
            # =================================================

            U_corrected = (
                apply_hard_compatibility_7region(

                    raw_displacement=
                        U_raw,

                    branch_physical=
                        branch_physical_tensor,

                    context=
                        hard_context,
                )
            )


        # ====================================================
        # PHYSICAL OUTPUTS
        # ====================================================

        U_prediction = (

            U_corrected[
                0
            ]

            .cpu()

            .numpy()

            .astype(
                np.float64
            )
        )


        LE_prediction = (

            LE_n[
                0
            ]

            .cpu()

            .numpy()

            *

            LE_std.reshape(
                1,
                6,
            )

            +

            LE_mean.reshape(
                1,
                6,
            )
        ).astype(
            np.float64
        )


        S_prediction = (

            S_n[
                0
            ]

            .cpu()

            .numpy()

            *

            S_std.reshape(
                1,
                6,
            )

            +

            S_mean.reshape(
                1,
                6,
            )
        ).astype(
            np.float64
        )


        member_U.append(
            U_prediction
        )


        member_LE.append(
            LE_prediction
        )


        member_S.append(
            S_prediction
        )


    # ========================================================
    # ENSEMBLE AVERAGE
    # ========================================================

    U_ensemble = np.mean(
        np.asarray(
            member_U
        ),
        axis=0,
    )


    LE_ensemble = np.mean(
        np.asarray(
            member_LE
        ),
        axis=0,
    )


    S_ensemble = np.mean(
        np.asarray(
            member_S
        ),
        axis=0,
    )


    # ========================================================
    # SAVE LOCAL PREDICTION
    # ========================================================

    np.savez_compressed(

        os.path.join(
            OUTPUT_DIR,
            "{}_local_prediction.npz".format(
                segment
            ),
        ),

        coordinates=
            coordinates,

        ip_coordinates=
            ip_coordinates,

        branch=
            branch_physical,

        U=
            U_ensemble,

        LE=
            LE_ensemble,

        S=
            S_ensemble,
    )


    print(
        "Nodes:",
        len(
            coordinates
        )
    )


    print(
        "Integration points/elements:",
        len(
            ip_coordinates
        )
    )


    return {

        "coordinates":
            coordinates.astype(
                np.float64
            ),

        "ip_coordinates":
            ip_coordinates.astype(
                np.float64
            ),

        "U":
            U_ensemble,

        "LE":
            LE_ensemble,

        "S":
            S_ensemble,
    }


# ============================================================
# PREDICT ALL FOUR NO REGIONS
# ============================================================

NO_PREDICTIONS = {}


for segment in SEGMENTS:

    NO_PREDICTIONS[
        segment
    ] = predict_segment(
        segment
    )


# ============================================================
# LOAD FULL FEM REFERENCE
# ============================================================

reference_nodes = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "full_reference_nodes.csv",
    )
)


reference_ip = pd.read_csv(
    os.path.join(
        VALIDATION_DIR,
        "full_reference_ip.csv",
    )
)


print("")
print(
    "Full FEM reference nodes:",
    len(
        reference_nodes
    )
)


print(
    "Full FEM reference elements:",
    len(
        reference_ip
    )
)


# ============================================================
# FE FILES
# ============================================================

FE_NODE_FILES = {

    "FE_L":
        "left_nodes.csv",

    "FE_C":
        "center_nodes.csv",

    "FE_R":
        "right_nodes.csv",
}


FE_IP_FILES = {

    "FE_L":
        "left_ip.csv",

    "FE_C":
        "center_ip.csv",

    "FE_R":
        "right_ip.csv",
}


# ============================================================
# ASSEMBLE NODAL DISPLACEMENT
#
# FE owns every shared interface node.
# ============================================================

hybrid_node_map = {}

node_owner_map = {}


for region_name, filename in (
    FE_NODE_FILES.items()
):

    dataframe = pd.read_csv(
        os.path.join(
            VALIDATION_DIR,
            filename,
        )
    )


    for _, row in dataframe.iterrows():

        xyz = [
            row[
                "X"
            ],
            row[
                "Y"
            ],
            row[
                "Z"
            ],
        ]


        key = coordinate_key(
            xyz
        )


        hybrid_node_map[
            key
        ] = np.asarray(
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


        node_owner_map[
            key
        ] = region_name


# ============================================================
# ADD NO NODAL FIELDS
#
# Do NOT overwrite interface nodes.
# FE owns all six shared interfaces.
# ============================================================

TOL = 1.0e-5


def add_no_nodes(
    segment,
    x_condition,
    owner_name,
):

    prediction = NO_PREDICTIONS[
        segment
    ]


    coordinates = prediction[
        "coordinates"
    ]


    U = prediction[
        "U"
    ]


    added = 0


    for index in range(
        len(
            coordinates
        )
    ):

        xyz = coordinates[
            index
        ]


        x_value = float(
            xyz[
                0
            ]
        )


        if not x_condition(
            x_value
        ):

            continue


        key = coordinate_key(
            xyz
        )


        if key in hybrid_node_map:

            raise RuntimeError(
                (
                    "NO tried to overwrite FE-owned node: {}"
                ).format(
                    key
                )
            )


        hybrid_node_map[
            key
        ] = U[
            index
        ]


        node_owner_map[
            key
        ] = owner_name


        added += 1


    print(
        "{} added nodes: {}".format(
            owner_name,
            added,
        )
    )


add_no_nodes(

    "outer_left",

    lambda x:
        x
        <
        -9.6
        -
        TOL,

    "NO_OL",
)


add_no_nodes(

    "inner_left",

    lambda x:
        (
            x
            >
            -6.6
            +
            TOL

            and

            x
            <
            -1.8
            -
            TOL
        ),

    "NO_L",
)


add_no_nodes(

    "inner_right",

    lambda x:
        (
            x
            >
            1.8
            +
            TOL

            and

            x
            <
            6.6
            -
            TOL
        ),

    "NO_R",
)


add_no_nodes(

    "outer_right",

    lambda x:
        x
        >
        9.6
        +
        TOL,

    "NO_OR",
)


# ============================================================
# ALIGN NODAL FIELD TO FULL FEM
# ============================================================

hybrid_U = []

reference_U = []

owners = []

missing_nodes = []


for _, row in reference_nodes.iterrows():

    xyz = [
        row[
            "X"
        ],
        row[
            "Y"
        ],
        row[
            "Z"
        ],
    ]


    key = coordinate_key(
        xyz
    )


    if key not in hybrid_node_map:

        missing_nodes.append(
            key
        )

        continue


    hybrid_U.append(
        hybrid_node_map[
            key
        ]
    )


    reference_U.append(
        np.asarray(
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
    )


    owners.append(
        node_owner_map[
            key
        ]
    )


if len(
    missing_nodes
) > 0:

    print("")
    print(
        "Missing nodal coordinates:",
        len(
            missing_nodes
        )
    )


    print(
        missing_nodes[
            :20
        ]
    )


    raise RuntimeError(
        "Hybrid displacement field does not cover full mesh."
    )


hybrid_U = np.asarray(
    hybrid_U,
    dtype=np.float64,
)


reference_U = np.asarray(
    reference_U,
    dtype=np.float64,
)


owners = np.asarray(
    owners,
    dtype=object,
)


print("")
print(
    "NODAL ASSEMBLY PASS"
)

print(
    "Hybrid nodes:",
    hybrid_U.shape[
        0
    ]
)


# ============================================================
# GLOBAL DISPLACEMENT METRICS
# ============================================================

displacement_rows = []


for component_index, component_name in enumerate(
    [
        "U1",
        "U2",
        "U3",
    ]
):

    metrics = calculate_metrics(

        hybrid_U[
            :,
            component_index
        ],

        reference_U[
            :,
            component_index
        ],
    )


    displacement_rows.append(
        {
            "Component":
                component_name,

            **metrics,
        }
    )


displacement_metrics = pd.DataFrame(
    displacement_rows
)


# ============================================================
# REGION-WISE DISPLACEMENT
# ============================================================

region_displacement_rows = []


for region_name in REGION_ORDER:

    mask = (
        owners
        ==
        region_name
    )


    number_nodes = int(
        np.sum(
            mask
        )
    )


    if number_nodes == 0:

        raise RuntimeError(
            "Region {} has zero nodes."
            .format(
                region_name
            )
        )


    for component_index, component_name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        metrics = calculate_metrics(

            hybrid_U[
                mask,
                component_index
            ],

            reference_U[
                mask,
                component_index
            ],
        )


        region_displacement_rows.append(
            {
                "Region":
                    region_name,

                "Nodes":
                    number_nodes,

                "Component":
                    component_name,

                **metrics,
            }
        )


region_displacement_metrics = pd.DataFrame(
    region_displacement_rows
)


# ============================================================
# BUILD RAW HYBRID INTEGRATION-POINT ARRAYS
#
# IMPORTANT CHANGE:
#
# We no longer use rounded dictionary coordinates here.
#
# We construct one complete hybrid coordinate array and then
# perform a nearest-coordinate one-to-one mapping to the
# full-FEM centroids.
# ============================================================

hybrid_ip_coordinates_parts = []

hybrid_LE_parts = []

hybrid_S_parts = []

hybrid_ip_owner_parts = []


# ============================================================
# ADD FE MECHANICS
# ============================================================

for region_name, filename in (
    FE_IP_FILES.items()
):

    dataframe = pd.read_csv(
        os.path.join(
            VALIDATION_DIR,
            filename,
        )
    )


    coordinates = dataframe[
        [
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    LE_values = dataframe[
        [
            "LE11",
            "LE22",
            "LE33",
            "LE12",
            "LE13",
            "LE23",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    S_values = dataframe[
        [
            "S11",
            "S22",
            "S33",
            "S12",
            "S13",
            "S23",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    hybrid_ip_coordinates_parts.append(
        coordinates
    )


    hybrid_LE_parts.append(
        LE_values
    )


    hybrid_S_parts.append(
        S_values
    )


    hybrid_ip_owner_parts.append(
        np.asarray(
            [
                region_name
            ]
            *
            len(
                dataframe
            ),
            dtype=object,
        )
    )


    print(
        "{} mechanics elements: {}".format(
            region_name,
            len(
                dataframe
            ),
        )
    )


# ============================================================
# ADD NO MECHANICS
# ============================================================

NO_OWNER_NAMES = {

    "outer_left":
        "NO_OL",

    "inner_left":
        "NO_L",

    "inner_right":
        "NO_R",

    "outer_right":
        "NO_OR",
}


for segment, owner_name in (
    NO_OWNER_NAMES.items()
):

    prediction = NO_PREDICTIONS[
        segment
    ]


    coordinates = np.asarray(
        prediction[
            "ip_coordinates"
        ],
        dtype=np.float64,
    )


    LE_values = np.asarray(
        prediction[
            "LE"
        ],
        dtype=np.float64,
    )


    S_values = np.asarray(
        prediction[
            "S"
        ],
        dtype=np.float64,
    )


    hybrid_ip_coordinates_parts.append(
        coordinates
    )


    hybrid_LE_parts.append(
        LE_values
    )


    hybrid_S_parts.append(
        S_values
    )


    hybrid_ip_owner_parts.append(
        np.asarray(
            [
                owner_name
            ]
            *
            len(
                coordinates
            ),
            dtype=object,
        )
    )


    print(
        "{} mechanics elements: {}".format(
            owner_name,
            len(
                coordinates
            ),
        )
    )


# ============================================================
# CONCATENATE COMPLETE HYBRID MECHANICS FIELD
# ============================================================

hybrid_ip_coordinates_raw = np.concatenate(
    hybrid_ip_coordinates_parts,
    axis=0,
)


hybrid_LE_raw = np.concatenate(
    hybrid_LE_parts,
    axis=0,
)


hybrid_S_raw = np.concatenate(
    hybrid_S_parts,
    axis=0,
)


hybrid_ip_owners_raw = np.concatenate(
    hybrid_ip_owner_parts,
    axis=0,
)


reference_ip_coordinates = reference_ip[
    [
        "X",
        "Y",
        "Z",
    ]
].to_numpy(
    dtype=np.float64
)


print("")
print(
    "=========================================="
)

print(
    "MECHANICS FIELD COVERAGE"
)

print(
    "=========================================="
)


print(
    "Hybrid mechanics points:",
    len(
        hybrid_ip_coordinates_raw
    )
)


print(
    "Reference FEM mechanics points:",
    len(
        reference_ip_coordinates
    )
)


if len(
    hybrid_ip_coordinates_raw
) != len(
    reference_ip_coordinates
):

    raise RuntimeError(
        (
            "Hybrid/reference element count mismatch: "
            "{} vs {}"
        ).format(
            len(
                hybrid_ip_coordinates_raw
            ),
            len(
                reference_ip_coordinates
            ),
        )
    )


# ============================================================
# NEAREST COORDINATE MATCHING
#
# Dependency-free chunked nearest-neighbor search.
#
# For each full-FEM centroid:
#
#     find nearest hybrid centroid
#
# We then enforce:
#
# 1. maximum distance <= 1e-4 mm
# 2. one-to-one mapping
#
# Therefore this cannot silently map two FEM elements to the
# same hybrid element.
# ============================================================

def nearest_coordinate_mapping(

    query_coordinates,

    source_coordinates,

    tolerance,

    chunk_size=250,
):

    query_coordinates = np.asarray(
        query_coordinates,
        dtype=np.float64,
    )


    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64,
    )


    number_query = query_coordinates.shape[
        0
    ]


    matched_indices = np.empty(
        number_query,
        dtype=np.int64,
    )


    matched_distances = np.empty(
        number_query,
        dtype=np.float64,
    )


    for start in range(
        0,
        number_query,
        chunk_size,
    ):

        end = min(
            start
            +
            chunk_size,

            number_query,
        )


        query_chunk = query_coordinates[
            start:
            end
        ]


        difference = (

            query_chunk[
                :,
                None,
                :
            ]

            -

            source_coordinates[
                None,
                :,
                :
            ]
        )


        distance_squared = np.sum(
            difference ** 2,
            axis=2,
        )


        local_indices = np.argmin(
            distance_squared,
            axis=1,
        )


        local_distances = np.sqrt(

            distance_squared[
                np.arange(
                    end
                    -
                    start
                ),

                local_indices,
            ]
        )


        matched_indices[
            start:
            end
        ] = local_indices


        matched_distances[
            start:
            end
        ] = local_distances


    maximum_distance = float(
        np.max(
            matched_distances
        )
    )


    mean_distance = float(
        np.mean(
            matched_distances
        )
    )


    print("")
    print(
        "IP nearest-coordinate matching:"
    )


    print(
        "Mean distance (mm):",
        mean_distance
    )


    print(
        "Maximum distance (mm):",
        maximum_distance
    )


    print(
        "Tolerance (mm):",
        tolerance
    )


    if maximum_distance > tolerance:

        worst_index = int(
            np.argmax(
                matched_distances
            )
        )


        print("")
        print(
            "Worst reference coordinate:"
        )


        print(
            query_coordinates[
                worst_index
            ]
        )


        print(
            "Closest hybrid coordinate:"
        )


        print(
            source_coordinates[
                matched_indices[
                    worst_index
                ]
            ]
        )


        print(
            "Distance:"
        )


        print(
            matched_distances[
                worst_index
            ]
        )


        raise RuntimeError(
            (
                "Integration-point coordinate mismatch "
                "exceeds tolerance."
            )
        )


    # ========================================================
    # ONE-TO-ONE CHECK
    # ========================================================

    unique_indices = np.unique(
        matched_indices
    )


    print(
        "Unique matched hybrid points:",
        len(
            unique_indices
        )
    )


    if len(
        unique_indices
    ) != number_query:

        raise RuntimeError(
            (
                "Integration-point matching is not "
                "one-to-one."
            )
        )


    return (
        matched_indices,
        matched_distances,
    )


(
    ip_mapping,
    ip_mapping_distances,
) = nearest_coordinate_mapping(

    query_coordinates=
        reference_ip_coordinates,

    source_coordinates=
        hybrid_ip_coordinates_raw,

    tolerance=
        IP_MATCH_TOLERANCE_MM,
)


# ============================================================
# REORDER HYBRID FIELDS TO FULL FEM ORDER
# ============================================================

hybrid_LE = hybrid_LE_raw[
    ip_mapping
]


hybrid_S = hybrid_S_raw[
    ip_mapping
]


ip_owners = hybrid_ip_owners_raw[
    ip_mapping
]


matched_hybrid_ip_coordinates = (
    hybrid_ip_coordinates_raw[
        ip_mapping
    ]
)


# ============================================================
# FULL FEM MECHANICS ARRAYS
# ============================================================

reference_LE = reference_ip[
    [
        "LE11",
        "LE22",
        "LE33",
        "LE12",
        "LE13",
        "LE23",
    ]
].to_numpy(
    dtype=np.float64
)


reference_S = reference_ip[
    [
        "S11",
        "S22",
        "S33",
        "S12",
        "S13",
        "S23",
    ]
].to_numpy(
    dtype=np.float64
)


print("")
print(
    "MECHANICS ASSEMBLY PASS"
)


# ============================================================
# GLOBAL MECHANICS METRICS
# ============================================================

mechanics_rows = []


for component_index, component_name in enumerate(
    COMPONENTS
):

    LE_metrics = calculate_metrics(

        hybrid_LE[
            :,
            component_index
        ],

        reference_LE[
            :,
            component_index
        ],
    )


    mechanics_rows.append(
        {
            "Field":
                "LE",

            "Component":
                "LE{}".format(
                    component_name
                ),

            **LE_metrics,
        }
    )


    S_metrics = calculate_metrics(

        hybrid_S[
            :,
            component_index
        ],

        reference_S[
            :,
            component_index
        ],
    )


    mechanics_rows.append(
        {
            "Field":
                "S",

            "Component":
                "S{}".format(
                    component_name
                ),

            **S_metrics,
        }
    )


mechanics_metrics = pd.DataFrame(
    mechanics_rows
)


# ============================================================
# REGION-WISE MECHANICS
# ============================================================

region_mechanics_rows = []


for region_name in REGION_ORDER:

    mask = (
        ip_owners
        ==
        region_name
    )


    number_elements = int(
        np.sum(
            mask
        )
    )


    if number_elements == 0:

        raise RuntimeError(
            (
                "Region {} has zero mechanics elements."
            ).format(
                region_name
            )
        )


    for component_index, component_name in enumerate(
        COMPONENTS
    ):

        LE_metrics = calculate_metrics(

            hybrid_LE[
                mask,
                component_index
            ],

            reference_LE[
                mask,
                component_index
            ],
        )


        region_mechanics_rows.append(
            {
                "Region":
                    region_name,

                "Elements":
                    number_elements,

                "Field":
                    "LE",

                "Component":
                    "LE{}".format(
                        component_name
                    ),

                **LE_metrics,
            }
        )


        S_metrics = calculate_metrics(

            hybrid_S[
                mask,
                component_index
            ],

            reference_S[
                mask,
                component_index
            ],
        )


        region_mechanics_rows.append(
            {
                "Region":
                    region_name,

                "Elements":
                    number_elements,

                "Field":
                    "S",

                "Component":
                    "S{}".format(
                        component_name
                    ),

                **S_metrics,
            }
        )


region_mechanics_metrics = pd.DataFrame(
    region_mechanics_rows
)


# ============================================================
# ASSEMBLED NODE CSV
# ============================================================

assembled_nodes = reference_nodes[
    [
        "NodeLabel",
        "X",
        "Y",
        "Z",
    ]
].copy()


assembled_nodes[
    "Owner"
] = owners


for component_index, component_name in enumerate(
    [
        "U1",
        "U2",
        "U3",
    ]
):

    assembled_nodes[
        "FEM_{}".format(
            component_name
        )
    ] = reference_U[
        :,
        component_index
    ]


    assembled_nodes[
        "Hybrid_{}".format(
            component_name
        )
    ] = hybrid_U[
        :,
        component_index
    ]


    assembled_nodes[
        "AbsError_{}".format(
            component_name
        )
    ] = np.abs(

        hybrid_U[
            :,
            component_index
        ]

        -

        reference_U[
            :,
            component_index
        ]
    )


# ============================================================
# ASSEMBLED MECHANICS CSV
# ============================================================

assembled_ip = reference_ip[
    [
        "ElementLabel",
        "X",
        "Y",
        "Z",
    ]
].copy()


assembled_ip[
    "MatchedHybridX"
] = matched_hybrid_ip_coordinates[
    :,
    0
]


assembled_ip[
    "MatchedHybridY"
] = matched_hybrid_ip_coordinates[
    :,
    1
]


assembled_ip[
    "MatchedHybridZ"
] = matched_hybrid_ip_coordinates[
    :,
    2
]


assembled_ip[
    "CoordinateMatchDistance_mm"
] = (
    ip_mapping_distances
)


assembled_ip[
    "Owner"
] = ip_owners


for component_index, component_name in enumerate(
    COMPONENTS
):

    assembled_ip[
        "FEM_LE{}".format(
            component_name
        )
    ] = reference_LE[
        :,
        component_index
    ]


    assembled_ip[
        "Hybrid_LE{}".format(
            component_name
        )
    ] = hybrid_LE[
        :,
        component_index
    ]


    assembled_ip[
        "AbsError_LE{}".format(
            component_name
        )
    ] = np.abs(

        hybrid_LE[
            :,
            component_index
        ]

        -

        reference_LE[
            :,
            component_index
        ]
    )


    assembled_ip[
        "FEM_S{}".format(
            component_name
        )
    ] = reference_S[
        :,
        component_index
    ]


    assembled_ip[
        "Hybrid_S{}".format(
            component_name
        )
    ] = hybrid_S[
        :,
        component_index
    ]


    assembled_ip[
        "AbsError_S{}".format(
            component_name
        )
    ] = np.abs(

        hybrid_S[
            :,
            component_index
        ]

        -

        reference_S[
            :,
            component_index
        ]
    )


# ============================================================
# IP MATCHING AUDIT
# ============================================================

ip_matching_audit = {

    "number_reference_points":
        int(
            len(
                reference_ip_coordinates
            )
        ),

    "number_hybrid_points":
        int(
            len(
                hybrid_ip_coordinates_raw
            )
        ),

    "unique_matched_points":
        int(
            len(
                np.unique(
                    ip_mapping
                )
            )
        ),

    "mean_coordinate_distance_mm":
        float(
            np.mean(
                ip_mapping_distances
            )
        ),

    "maximum_coordinate_distance_mm":
        float(
            np.max(
                ip_mapping_distances
            )
        ),

    "matching_tolerance_mm":
        float(
            IP_MATCH_TOLERANCE_MM
        ),

    "one_to_one":
        bool(
            len(
                np.unique(
                    ip_mapping
                )
            )
            ==
            len(
                reference_ip_coordinates
            )
        ),
}


# ============================================================
# SAVE OUTPUTS
# ============================================================

displacement_metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "global_displacement_metrics.csv",
    ),

    index=False,
)


region_displacement_metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "region_displacement_metrics.csv",
    ),

    index=False,
)


mechanics_metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "global_mechanics_metrics.csv",
    ),

    index=False,
)


region_mechanics_metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "region_mechanics_metrics.csv",
    ),

    index=False,
)


assembled_nodes.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "assembled_hybrid_nodes.csv",
    ),

    index=False,
)


assembled_ip.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "assembled_hybrid_ip.csv",
    ),

    index=False,
)


with open(

    os.path.join(
        OUTPUT_DIR,
        "ip_matching_audit.json",
    ),

    "w",

) as file_object:

    json.dump(
        ip_matching_audit,
        file_object,
        indent=4,
    )


np.save(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_U.npy",
    ),
    hybrid_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "reference_U.npy",
    ),
    reference_U,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_LE.npy",
    ),
    hybrid_LE,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "reference_LE.npy",
    ),
    reference_LE,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "hybrid_S.npy",
    ),
    hybrid_S,
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "reference_S.npy",
    ),
    reference_S,
)


# ============================================================
# REACTION SUMMARY
# ============================================================

reaction_file = os.path.join(
    VALIDATION_DIR,
    "reaction_summary.json",
)


with open(
    reaction_file,
    "r",
) as file_object:

    reaction_summary = json.load(
        file_object
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64V COMPLETE"
)

print(
    "=========================================="
)


print("")
print(
    "GLOBAL DISPLACEMENT"
)

print(
    displacement_metrics.to_string(
        index=False
    )
)


print("")
print(
    "REGION-WISE U2"
)

print(

    region_displacement_metrics[
        region_displacement_metrics[
            "Component"
        ]
        ==
        "U2"
    ].to_string(
        index=False
    )
)


print("")
print(
    "IMPORTANT MECHANICS"
)

important_mechanics = mechanics_metrics[
    mechanics_metrics[
        "Component"
    ].isin(
        [
            "LE11",
            "LE12",
            "S11",
            "S12",
        ]
    )
]


print(
    important_mechanics.to_string(
        index=False
    )
)


print("")
print(
    "INTEGRATION-POINT MATCHING AUDIT"
)

print(
    json.dumps(
        ip_matching_audit,
        indent=4,
    )
)


print("")
print(
    "REACTION"
)

print(
    json.dumps(
        reaction_summary,
        indent=4,
    )
)


print("")
print(
    "Saved results to:"
)

print(
    OUTPUT_DIR
)