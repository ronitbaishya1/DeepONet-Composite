import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DESIGN_DIR = (
    "data/candidate_partition_designs_7region"
)


INTERFACE_DIR = (
    "data/candidate_partition_interfaces_7region"
)


EXTRACTED_ROOT = (
    "data/windows_candidate_7region_extracted"
)


FULL_DATA_DIR = "data"


OUTPUT_DIR = (
    "results/seven_region_smoke"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


SEGMENTS = [
    "outer_left",
    "inner_left",
    "inner_right",
    "outer_right",
]


# ============================================================
# FULL-DOMAIN REFERENCE
# ============================================================

full_coordinates = np.load(
    os.path.join(
        FULL_DATA_DIR,
        "coordinates.npy",
    )
).astype(
    np.float64
)


full_U1 = np.load(
    os.path.join(
        FULL_DATA_DIR,
        "U1.npy",
    )
).astype(
    np.float64
)


full_U2 = np.load(
    os.path.join(
        FULL_DATA_DIR,
        "U2.npy",
    )
).astype(
    np.float64
)


full_U3 = np.load(
    os.path.join(
        FULL_DATA_DIR,
        "U3.npy",
    )
).astype(
    np.float64
)


full_U = np.stack(
    [
        full_U1,
        full_U2,
        full_U3,
    ],
    axis=-1,
)


# ============================================================
# COORDINATE LOOKUP
# ============================================================

def coordinate_key(
    xyz
):

    return tuple(
        np.round(
            np.asarray(
                xyz,
                dtype=np.float64,
            ),
            decimals=7,
        )
    )


full_coordinate_lookup = {}


for node_index, xyz in enumerate(
    full_coordinates
):

    full_coordinate_lookup[
        coordinate_key(
            xyz
        )
    ] = node_index


# ============================================================
# ERROR
# ============================================================

def relative_l2_percent(
    prediction,
    truth,
):

    denominator = np.linalg.norm(
        truth
    )


    if denominator < 1.0e-14:

        return np.nan


    return (
        100.0
        *
        np.linalg.norm(
            prediction
            -
            truth
        )
        /
        denominator
    )


# ============================================================
# INTERFACE RECONSTRUCTION
# ============================================================

def check_interface(
    nodes,
    row,
    interface_name,
    side,
):

    interface_file = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    interface = np.load(
        interface_file
    )


    mean = interface[
        "mean"
    ].astype(
        np.float64
    )


    basis = interface[
        "basis"
    ].astype(
        np.float64
    )


    x_interface = float(
        interface[
            "actual_x"
        ][0]
    )


    number_modes = (
        basis.shape[
            1
        ]
    )


    prefix = (
        "cL"
        if side == "left"
        else
        "cR"
    )


    coefficients = np.asarray(
        [
            float(
                row[
                    "{}_{:02d}".format(
                        prefix,
                        mode_index + 1,
                    )
                ]
            )

            for mode_index
            in range(
                number_modes
            )
        ],
        dtype=np.float64,
    )


    target = (
        mean
        +
        basis
        @
        coefficients
    ).reshape(
        -1,
        3
    )


    face = nodes[
        np.isclose(
            nodes[
                "X"
            ].to_numpy(
                dtype=np.float64
            ),
            x_interface,
            atol=1.0e-7,
        )
    ].copy()


    face = face.sort_values(
        [
            "Y",
            "Z",
        ]
    )


    actual = face[
        [
            "U1",
            "U2",
            "U3",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    if actual.shape != (
        target.shape
    ):

        raise RuntimeError(
            "{} interface shape mismatch."
            .format(
                interface_name
            )
        )


    maximum_absolute_error = float(
        np.max(
            np.abs(
                actual
                -
                target
            )
        )
    )


    relative_error = (
        relative_l2_percent(
            actual,
            target,
        )
    )


    return (
        maximum_absolute_error,
        relative_error,
    )


# ============================================================
# AUDIT
# ============================================================

rows = []


for segment in SEGMENTS:

    design_file = os.path.join(
        DESIGN_DIR,
        "{}_candidate_design_80.csv".format(
            segment
        ),
    )


    design = pd.read_csv(
        design_file
    )


    # The smoke job is the first generated case.
    design_row = design.iloc[
        0
    ]


    case_id = str(
        design_row[
            "CaseID"
        ]
    )


    source = str(
        design_row[
            "Source"
        ]
    ).lower()


    parent_index = int(
        design_row[
            "ParentGlobalIndex"
        ]
    )


    if source != "anchor":

        raise RuntimeError(
            "{} smoke case is not an anchor."
            .format(
                segment
            )
        )


    if parent_index < 0:

        raise RuntimeError(
            "{} has invalid ParentGlobalIndex."
            .format(
                segment
            )
        )


    node_file = os.path.join(
        EXTRACTED_ROOT,
        segment,
        case_id
        +
        "_nodes.csv",
    )


    if not os.path.isfile(
        node_file
    ):

        raise FileNotFoundError(
            node_file
        )


    nodes = pd.read_csv(
        node_file
    )


    nodes = nodes.sort_values(
        [
            "X",
            "Y",
            "Z",
        ]
    ).reset_index(
        drop=True
    )


    local_coordinates = nodes[
        [
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    local_U = nodes[
        [
            "U1",
            "U2",
            "U3",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    # ========================================================
    # MATCH LOCAL NODES TO FULL FEM
    # ========================================================

    full_indices = []


    for xyz in local_coordinates:

        key = coordinate_key(
            xyz
        )


        if key not in (
            full_coordinate_lookup
        ):

            raise RuntimeError(
                "Coordinate not found in full FEM: {}"
                .format(
                    xyz
                )
            )


        full_indices.append(
            full_coordinate_lookup[
                key
            ]
        )


    full_indices = np.asarray(
        full_indices,
        dtype=int,
    )


    reference_U = full_U[
        parent_index,
        full_indices,
        :
    ]


    # ========================================================
    # FIELD ERRORS
    # ========================================================

    U_vector_error = (
        relative_l2_percent(
            local_U,
            reference_U,
        )
    )


    U1_error = (
        relative_l2_percent(
            local_U[
                :,
                0
            ],
            reference_U[
                :,
                0
            ],
        )
    )


    U2_error = (
        relative_l2_percent(
            local_U[
                :,
                1
            ],
            reference_U[
                :,
                1
            ],
        )
    )


    U3_error = (
        relative_l2_percent(
            local_U[
                :,
                2
            ],
            reference_U[
                :,
                2
            ],
        )
    )


    # ========================================================
    # INTERFACE BC CHECKS
    # ========================================================

    left_interface_name = str(
        design_row[
            "LeftInterface"
        ]
    )


    right_interface_name = str(
        design_row[
            "RightInterface"
        ]
    )


    left_bc_max = np.nan
    left_bc_rel = np.nan

    right_bc_max = np.nan
    right_bc_rel = np.nan


    if left_interface_name.upper() not in [
        "NONE",
        "NAN",
    ]:

        (
            left_bc_max,
            left_bc_rel,
        ) = check_interface(
            nodes,
            design_row,
            left_interface_name,
            "left",
        )


    if right_interface_name.upper() not in [
        "NONE",
        "NAN",
    ]:

        (
            right_bc_max,
            right_bc_rel,
        ) = check_interface(
            nodes,
            design_row,
            right_interface_name,
            "right",
        )


    # ========================================================
    # FREE FACE RF CHECK
    # ========================================================

    x_min = float(
        local_coordinates[
            :,
            0
        ].min()
    )


    x_max = float(
        local_coordinates[
            :,
            0
        ].max()
    )


    free_face_max_rf = np.nan


    left_type = str(
        design_row[
            "LeftBoundaryType"
        ]
    ).lower()


    right_type = str(
        design_row[
            "RightBoundaryType"
        ]
    ).lower()


    if left_type == "free":

        free_face = nodes[
            np.isclose(
                nodes[
                    "X"
                ],
                x_min,
                atol=1.0e-7,
            )
        ]


        free_face_max_rf = float(
            np.max(
                np.abs(
                    free_face[
                        [
                            "RF1",
                            "RF2",
                            "RF3",
                        ]
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            )
        )


    if right_type == "free":

        free_face = nodes[
            np.isclose(
                nodes[
                    "X"
                ],
                x_max,
                atol=1.0e-7,
            )
        ]


        free_face_max_rf = float(
            np.max(
                np.abs(
                    free_face[
                        [
                            "RF1",
                            "RF2",
                            "RF3",
                        ]
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            )
        )


    rows.append(
        {
            "Segment":
                segment,

            "CaseID":
                case_id,

            "ParentGlobalIndex":
                parent_index,

            "Nodes":
                len(
                    nodes
                ),

            "U_Vector_RelL2_percent":
                U_vector_error,

            "U1_RelL2_percent":
                U1_error,

            "U2_RelL2_percent":
                U2_error,

            "U3_RelL2_percent":
                U3_error,

            "LeftInterface_MaxAbsBCError_mm":
                left_bc_max,

            "LeftInterface_RelL2_percent":
                left_bc_rel,

            "RightInterface_MaxAbsBCError_mm":
                right_bc_max,

            "RightInterface_RelL2_percent":
                right_bc_rel,

            "FreeFace_MaxAbsRF_N":
                free_face_max_rf,
        }
    )


results = pd.DataFrame(
    rows
)


output_file = os.path.join(
    OUTPUT_DIR,
    "seven_region_smoke_audit.csv",
)


results.to_csv(
    output_file,
    index=False,
)


print("")
print(
    "=========================================="
)

print(
    "STEP 64I — SMOKE AUDIT"
)

print(
    "=========================================="
)

print("")
print(
    results.to_string(
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