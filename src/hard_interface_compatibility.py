import numpy as np
import torch


# ============================================================
# NEAREST Y-Z MATCH
# ============================================================

def nearest_yz_indices(
    query_coordinates,
    interface_coordinates,
    tolerance=1.0e-4,
):

    query_coordinates = np.asarray(
        query_coordinates,
        dtype=np.float64,
    )


    interface_coordinates = np.asarray(
        interface_coordinates,
        dtype=np.float64,
    )


    query_yz = query_coordinates[
        :,
        1:3
    ]


    interface_yz = interface_coordinates[
        :,
        1:3
    ]


    nearest_indices = []

    nearest_distances = []


    for yz in query_yz:

        difference = (
            interface_yz
            -
            yz.reshape(
                1,
                2,
            )
        )


        distances = np.sqrt(
            np.sum(
                difference ** 2,
                axis=1,
            )
        )


        nearest_index = int(
            np.argmin(
                distances
            )
        )


        nearest_distance = float(
            distances[
                nearest_index
            ]
        )


        nearest_indices.append(
            nearest_index
        )


        nearest_distances.append(
            nearest_distance
        )


    nearest_indices = np.asarray(
        nearest_indices,
        dtype=np.int64,
    )


    nearest_distances = np.asarray(
        nearest_distances,
        dtype=np.float64,
    )


    maximum_distance = float(
        np.max(
            nearest_distances
        )
    )


    if maximum_distance > tolerance:

        worst_index = int(
            np.argmax(
                nearest_distances
            )
        )


        raise RuntimeError(

            "Interface Y-Z matching failed.\n"
            "Maximum nearest distance = {:.8e} mm\n"
            "Tolerance                = {:.8e} mm\n"
            "Problem bulk coordinate  = {}\n"
            "Nearest interface coord  = {}"
            .format(

                maximum_distance,

                tolerance,

                query_coordinates[
                    worst_index
                ],

                interface_coordinates[
                    nearest_indices[
                        worst_index
                    ]
                ],
            )
        )


    return (
        nearest_indices,
        nearest_distances,
    )


# ============================================================
# BUILD HARD-BOUNDARY CONTEXT
# ============================================================

def build_hard_boundary_context(
    coordinates,
    left_interface_file,
    right_interface_file,
    yz_tolerance=1.0e-4,
    x_tolerance=1.0e-4,
):

    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )


    # ========================================================
    # LOAD PCA INTERFACES
    # ========================================================

    left = np.load(
        left_interface_file
    )


    right = np.load(
        right_interface_file
    )


    left_coordinates = np.asarray(
        left[
            "coordinates"
        ],
        dtype=np.float64,
    )


    right_coordinates = np.asarray(
        right[
            "coordinates"
        ],
        dtype=np.float64,
    )


    left_x = float(
        left[
            "actual_x"
        ][0]
    )


    right_x = float(
        right[
            "actual_x"
        ][0]
    )


    # ========================================================
    # BASIC CHECKS
    # ========================================================

    if coordinates.ndim != 2:

        raise RuntimeError(
            "Bulk coordinates must be a 2-D array."
        )


    if coordinates.shape[1] != 3:

        raise RuntimeError(
            "Bulk coordinates must have shape [N,3]."
        )


    if left_coordinates.shape[1] != 3:

        raise RuntimeError(
            "Left interface coordinates must have shape [N,3]."
        )


    if right_coordinates.shape[1] != 3:

        raise RuntimeError(
            "Right interface coordinates must have shape [N,3]."
        )


    if left_coordinates.shape[0] != right_coordinates.shape[0]:

        raise RuntimeError(
            "Left and right interfaces have different node counts."
        )


    if left[
        "mean"
    ].shape[0] != (
        3
        *
        left_coordinates.shape[0]
    ):

        raise RuntimeError(
            "Left PCA mean dimension is inconsistent."
        )


    if right[
        "mean"
    ].shape[0] != (
        3
        *
        right_coordinates.shape[0]
    ):

        raise RuntimeError(
            "Right PCA mean dimension is inconsistent."
        )


    # ========================================================
    # FIND BULK BOUNDARY NODES
    # ========================================================

    left_boundary_mask = np.isclose(

        coordinates[
            :,
            0
        ],

        left_x,

        atol=
            x_tolerance,
    )


    right_boundary_mask = np.isclose(

        coordinates[
            :,
            0
        ],

        right_x,

        atol=
            x_tolerance,
    )


    left_boundary_bulk_indices = np.where(
        left_boundary_mask
    )[0]


    right_boundary_bulk_indices = np.where(
        right_boundary_mask
    )[0]


    expected_interface_nodes = (
        left_coordinates.shape[
            0
        ]
    )


    if len(
        left_boundary_bulk_indices
    ) != expected_interface_nodes:

        raise RuntimeError(

            "Wrong number of left boundary nodes.\n"
            "Expected: {}\n"
            "Found: {}\n"
            "left_x = {}"
            .format(

                expected_interface_nodes,

                len(
                    left_boundary_bulk_indices
                ),

                left_x,
            )
        )


    if len(
        right_boundary_bulk_indices
    ) != expected_interface_nodes:

        raise RuntimeError(

            "Wrong number of right boundary nodes.\n"
            "Expected: {}\n"
            "Found: {}\n"
            "right_x = {}"
            .format(

                expected_interface_nodes,

                len(
                    right_boundary_bulk_indices
                ),

                right_x,
            )
        )


    # ========================================================
    # MATCH EVERY BULK NODE TO LEFT/RIGHT Y-Z CROSS SECTION
    #
    # Because the local block is an extrusion, every interior
    # point should have the same Y-Z coordinate as one of the
    # interface nodes.
    # ========================================================

    (
        node_to_left_interface,
        left_distances,
    ) = nearest_yz_indices(

        query_coordinates=
            coordinates,

        interface_coordinates=
            left_coordinates,

        tolerance=
            yz_tolerance,
    )


    (
        node_to_right_interface,
        right_distances,
    ) = nearest_yz_indices(

        query_coordinates=
            coordinates,

        interface_coordinates=
            right_coordinates,

        tolerance=
            yz_tolerance,
    )


    # ========================================================
    # MATCH ACTUAL BULK LEFT BOUNDARY TO PCA LEFT INTERFACE
    # ========================================================

    (
        left_boundary_to_interface,
        left_boundary_distances,
    ) = nearest_yz_indices(

        query_coordinates=
            coordinates[
                left_boundary_bulk_indices
            ],

        interface_coordinates=
            left_coordinates,

        tolerance=
            yz_tolerance,
    )


    # ========================================================
    # MATCH ACTUAL BULK RIGHT BOUNDARY TO PCA RIGHT INTERFACE
    # ========================================================

    (
        right_boundary_to_interface,
        right_boundary_distances,
    ) = nearest_yz_indices(

        query_coordinates=
            coordinates[
                right_boundary_bulk_indices
            ],

        interface_coordinates=
            right_coordinates,

        tolerance=
            yz_tolerance,
    )


    # ========================================================
    # BUILD:
    #
    # interface index -> bulk boundary node index
    # ========================================================

    left_boundary_nodes = np.full(
        expected_interface_nodes,
        -1,
        dtype=np.int64,
    )


    right_boundary_nodes = np.full(
        expected_interface_nodes,
        -1,
        dtype=np.int64,
    )


    for local_bulk_index, interface_index in enumerate(
        left_boundary_to_interface
    ):

        bulk_node_index = (
            left_boundary_bulk_indices[
                local_bulk_index
            ]
        )


        if left_boundary_nodes[
            interface_index
        ] != -1:

            raise RuntimeError(
                "Duplicate left interface Y-Z match."
            )


        left_boundary_nodes[
            interface_index
        ] = bulk_node_index


    for local_bulk_index, interface_index in enumerate(
        right_boundary_to_interface
    ):

        bulk_node_index = (
            right_boundary_bulk_indices[
                local_bulk_index
            ]
        )


        if right_boundary_nodes[
            interface_index
        ] != -1:

            raise RuntimeError(
                "Duplicate right interface Y-Z match."
            )


        right_boundary_nodes[
            interface_index
        ] = bulk_node_index


    # ========================================================
    # VERIFY EVERY INTERFACE NODE WAS FOUND
    # ========================================================

    if np.any(
        left_boundary_nodes
        <
        0
    ):

        missing = np.where(
            left_boundary_nodes
            <
            0
        )[0]


        raise RuntimeError(
            "Missing left boundary interface nodes: {}".format(
                missing
            )
        )


    if np.any(
        right_boundary_nodes
        <
        0
    ):

        missing = np.where(
            right_boundary_nodes
            <
            0
        )[0]


        raise RuntimeError(
            "Missing right boundary interface nodes: {}".format(
                missing
            )
        )


    # ========================================================
    # HARD-BOUNDARY BLENDING
    #
    # s = 0 at left
    # s = 1 at right
    #
    # Cubic Hermite functions:
    #
    # H_L = 1 - 3s^2 + 2s^3
    # H_R =     3s^2 - 2s^3
    #
    # They satisfy:
    #
    # H_L(0)=1
    # H_L(1)=0
    #
    # H_R(0)=0
    # H_R(1)=1
    #
    # and zero derivative at both ends.
    # ========================================================

    segment_length = (
        right_x
        -
        left_x
    )


    if segment_length <= 0.0:

        raise RuntimeError(
            "Invalid interface x ordering."
        )


    s = (
        coordinates[
            :,
            0
        ]
        -
        left_x
    ) / segment_length


    h_right = (
        3.0
        *
        s ** 2

        -

        2.0
        *
        s ** 3
    )


    h_left = (
        1.0
        -
        h_right
    )


    # ========================================================
    # DIAGNOSTIC PRINT
    # ========================================================

    print("")
    print(
        "=========================================="
    )

    print(
        "HARD INTERFACE CONTEXT"
    )

    print(
        "=========================================="
    )


    print(
        "Bulk nodes:",
        coordinates.shape[
            0
        ],
    )


    print(
        "Interface nodes:",
        expected_interface_nodes,
    )


    print(
        "Left x:",
        left_x,
    )


    print(
        "Right x:",
        right_x,
    )


    print(
        "Left boundary bulk nodes:",
        len(
            left_boundary_bulk_indices
        ),
    )


    print(
        "Right boundary bulk nodes:",
        len(
            right_boundary_bulk_indices
        ),
    )


    print(
        "Max bulk->left YZ distance:",
        float(
            np.max(
                left_distances
            )
        ),
        "mm",
    )


    print(
        "Max bulk->right YZ distance:",
        float(
            np.max(
                right_distances
            )
        ),
        "mm",
    )


    print(
        "Max left-boundary YZ distance:",
        float(
            np.max(
                left_boundary_distances
            )
        ),
        "mm",
    )


    print(
        "Max right-boundary YZ distance:",
        float(
            np.max(
                right_boundary_distances
            )
        ),
        "mm",
    )


    print(
        "Hard interface mapping PASSED."
    )


    # ========================================================
    # RETURN CONTEXT
    # ========================================================

    return {

        "left_mean":

            left[
                "mean"
            ].astype(
                np.float32
            ),

        "left_basis":

            left[
                "basis"
            ].astype(
                np.float32
            ),

        "right_mean":

            right[
                "mean"
            ].astype(
                np.float32
            ),

        "right_basis":

            right[
                "basis"
            ].astype(
                np.float32
            ),

        "left_modes":

            int(
                left[
                    "basis"
                ].shape[
                    1
                ]
            ),

        "right_modes":

            int(
                right[
                    "basis"
                ].shape[
                    1
                ]
            ),

        "left_boundary_nodes":

            left_boundary_nodes,

        "right_boundary_nodes":

            right_boundary_nodes,

        "node_to_left_interface":

            node_to_left_interface,

        "node_to_right_interface":

            node_to_right_interface,

        "h_left":

            h_left.astype(
                np.float32
            ),

        "h_right":

            h_right.astype(
                np.float32
            ),

        "left_x":

            np.asarray(
                [
                    left_x
                ],
                dtype=np.float32,
            ),

        "right_x":

            np.asarray(
                [
                    right_x
                ],
                dtype=np.float32,
            ),
    }


# ============================================================
# APPLY HARD DISPLACEMENT COMPATIBILITY
# ============================================================

def apply_hard_compatibility(
    raw_displacement,
    branch_physical,
    context,
):

    device = (
        raw_displacement.device
    )


    dtype = (
        raw_displacement.dtype
    )


    # ========================================================
    # NUMBER OF PCA MODES
    # ========================================================

    left_modes = context[
        "left_modes"
    ]


    right_modes = context[
        "right_modes"
    ]


    # ========================================================
    # PHYSICAL BRANCH FORMAT:
    #
    # [E1,E2,G12,c_left...,c_right...]
    # ========================================================

    c_left = branch_physical[
        :,
        3:
        3 + left_modes
    ]


    c_right = branch_physical[
        :,
        3 + left_modes:
        3 + left_modes
        + right_modes
    ]


    # ========================================================
    # PCA ARRAYS
    # ========================================================

    left_mean = torch.as_tensor(
        context[
            "left_mean"
        ],
        dtype=dtype,
        device=device,
    )


    left_basis = torch.as_tensor(
        context[
            "left_basis"
        ],
        dtype=dtype,
        device=device,
    )


    right_mean = torch.as_tensor(
        context[
            "right_mean"
        ],
        dtype=dtype,
        device=device,
    )


    right_basis = torch.as_tensor(
        context[
            "right_basis"
        ],
        dtype=dtype,
        device=device,
    )


    # ========================================================
    # RECONSTRUCT TARGET INTERFACE DISPLACEMENTS
    #
    # flattened:
    #
    # [u1_1,u2_1,u3_1,u1_2,u2_2,u3_2,...]
    # ========================================================

    target_left_flat = (
        left_mean.reshape(
            1,
            -1,
        )
        +
        c_left
        @
        left_basis.T
    )


    target_right_flat = (
        right_mean.reshape(
            1,
            -1,
        )
        +
        c_right
        @
        right_basis.T
    )


    target_left = (
        target_left_flat.reshape(
            branch_physical.shape[
                0
            ],
            -1,
            3,
        )
    )


    target_right = (
        target_right_flat.reshape(
            branch_physical.shape[
                0
            ],
            -1,
            3,
        )
    )


    # ========================================================
    # BULK NODE INDICES ON ACTUAL LEFT/RIGHT BOUNDARIES
    # ========================================================

    left_boundary_nodes = torch.as_tensor(
        context[
            "left_boundary_nodes"
        ],
        dtype=torch.long,
        device=device,
    )


    right_boundary_nodes = torch.as_tensor(
        context[
            "right_boundary_nodes"
        ],
        dtype=torch.long,
        device=device,
    )


    # ========================================================
    # RAW NETWORK DISPLACEMENT ON BOTH BOUNDARIES
    # ========================================================

    raw_left = raw_displacement[
        :,
        left_boundary_nodes,
        :
    ]


    raw_right = raw_displacement[
        :,
        right_boundary_nodes,
        :
    ]


    # ========================================================
    # DIFFERENCE REQUIRED FOR EXACT BOUNDARY SATISFACTION
    # ========================================================

    left_correction_interface = (
        target_left
        -
        raw_left
    )


    right_correction_interface = (
        target_right
        -
        raw_right
    )


    # ========================================================
    # MAP EACH BULK NODE TO SAME Y-Z POINT ON INTERFACE
    # ========================================================

    left_lookup = torch.as_tensor(
        context[
            "node_to_left_interface"
        ],
        dtype=torch.long,
        device=device,
    )


    right_lookup = torch.as_tensor(
        context[
            "node_to_right_interface"
        ],
        dtype=torch.long,
        device=device,
    )


    left_correction_all = (
        left_correction_interface[
            :,
            left_lookup,
            :
        ]
    )


    right_correction_all = (
        right_correction_interface[
            :,
            right_lookup,
            :
        ]
    )


    # ========================================================
    # BLENDING FUNCTIONS
    # ========================================================

    h_left = torch.as_tensor(
        context[
            "h_left"
        ],
        dtype=dtype,
        device=device,
    ).reshape(
        1,
        -1,
        1,
    )


    h_right = torch.as_tensor(
        context[
            "h_right"
        ],
        dtype=dtype,
        device=device,
    ).reshape(
        1,
        -1,
        1,
    )


    # ========================================================
    # HARD-CORRECTED DISPLACEMENT
    # ========================================================

    corrected_displacement = (

        raw_displacement

        +

        h_left
        *
        left_correction_all

        +

        h_right
        *
        right_correction_all
    )


    return (
        corrected_displacement
    )