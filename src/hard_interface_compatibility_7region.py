import os

import numpy as np
import torch


# ============================================================
# HELPERS
# ============================================================

def yz_key(
    y,
    z,
):

    return (
        round(
            float(
                y
            ),
            7,
        ),

        round(
            float(
                z
            ),
            7,
        ),
    )


def load_interface(
    interface_dir,
    interface_name,
):

    if interface_name is None:

        return None


    filename = os.path.join(
        interface_dir,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    if not os.path.isfile(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


    data = np.load(
        filename
    )


    return {
        "name":
            interface_name,

        "x":
            float(
                data[
                    "actual_x"
                ][0]
            ),

        "coordinates":
            data[
                "coordinates"
            ].astype(
                np.float32
            ),

        "mean":
            data[
                "mean"
            ].astype(
                np.float32
            ),

        "basis":
            data[
                "basis"
            ].astype(
                np.float32
            ),
    }


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_hard_boundary_context_7region(

    coordinates,

    interface_dir,

    left_interface_name=None,

    right_interface_name=None,

    device="cpu",
):

    coordinates = np.asarray(
        coordinates,
        dtype=np.float32,
    )


    left = load_interface(
        interface_dir,
        left_interface_name,
    )


    right = load_interface(
        interface_dir,
        right_interface_name,
    )


    if (
        left is None
        and
        right is None
    ):

        raise RuntimeError(
            "At least one interface is required."
        )


    x_min = float(
        np.min(
            coordinates[
                :,
                0
            ]
        )
    )


    x_max = float(
        np.max(
            coordinates[
                :,
                0
            ]
        )
    )


    length = (
        x_max
        -
        x_min
    )


    if length <= 0.0:

        raise RuntimeError(
            "Invalid local domain length."
        )


    # ========================================================
    # REFERENCE CROSS-SECTION
    # ========================================================

    reference = (
        left
        if left is not None
        else
        right
    )


    interface_coordinates = reference[
        "coordinates"
    ]


    interface_yz = interface_coordinates[
        :,
        1:3
    ]


    yz_to_interface_index = {}


    for interface_index, yz in enumerate(
        interface_yz
    ):

        key = yz_key(
            yz[
                0
            ],
            yz[
                1
            ],
        )


        yz_to_interface_index[
            key
        ] = interface_index


    # ========================================================
    # EACH LOCAL NODE -> YZ INDEX
    # ========================================================

    yz_indices = []


    for xyz in coordinates:

        key = yz_key(
            xyz[
                1
            ],
            xyz[
                2
            ],
        )


        if key not in (
            yz_to_interface_index
        ):

            raise RuntimeError(
                "Y-Z coordinate missing from interface: {}"
                .format(
                    key
                )
            )


        yz_indices.append(
            yz_to_interface_index[
                key
            ]
        )


    yz_indices = np.asarray(
        yz_indices,
        dtype=np.int64,
    )


    # ========================================================
    # BOUNDARY LOCAL INDICES
    # ========================================================

    left_local_indices = None

    right_local_indices = None


    if left is not None:

        left_local_indices = []


        for interface_xyz in left[
            "coordinates"
        ]:

            distance = np.linalg.norm(

                coordinates
                -
                interface_xyz.reshape(
                    1,
                    3
                ),

                axis=1,
            )


            index = int(
                np.argmin(
                    distance
                )
            )


            if distance[
                index
            ] > 1.0e-5:

                raise RuntimeError(
                    "Could not map left interface node."
                )


            left_local_indices.append(
                index
            )


        left_local_indices = np.asarray(
            left_local_indices,
            dtype=np.int64,
        )


    if right is not None:

        right_local_indices = []


        for interface_xyz in right[
            "coordinates"
        ]:

            distance = np.linalg.norm(

                coordinates
                -
                interface_xyz.reshape(
                    1,
                    3
                ),

                axis=1,
            )


            index = int(
                np.argmin(
                    distance
                )
            )


            if distance[
                index
            ] > 1.0e-5:

                raise RuntimeError(
                    "Could not map right interface node."
                )


            right_local_indices.append(
                index
            )


        right_local_indices = np.asarray(
            right_local_indices,
            dtype=np.int64,
        )


    # ========================================================
    # BRANCH COEFFICIENT SLICES
    #
    # branch:
    #
    # [E1,E2,G12,
    #  left coefficients if present,
    #  right coefficients if present]
    # ========================================================

    cursor = 3


    left_slice = None

    right_slice = None


    if left is not None:

        number_modes = left[
            "basis"
        ].shape[1]


        left_slice = slice(
            cursor,
            cursor
            +
            number_modes,
        )


        cursor += number_modes


    if right is not None:

        number_modes = right[
            "basis"
        ].shape[1]


        right_slice = slice(
            cursor,
            cursor
            +
            number_modes,
        )


        cursor += number_modes


    # ========================================================
    # HERMITE BLENDING WEIGHTS
    #
    # hR = 3 s^2 - 2 s^3
    # hL = 1 - hR
    #
    # Both have zero derivative at each end.
    # ========================================================

    s = (
        (
            coordinates[
                :,
                0
            ]
            -
            x_min
        )
        /
        length
    )


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


    context = {

        "x_min":
            x_min,

        "x_max":
            x_max,

        "left":
            left,

        "right":
            right,

        "left_slice":
            left_slice,

        "right_slice":
            right_slice,

        "left_local_indices":
            (
                None
                if left_local_indices is None
                else
                torch.tensor(
                    left_local_indices,
                    dtype=torch.long,
                    device=device,
                )
            ),

        "right_local_indices":
            (
                None
                if right_local_indices is None
                else
                torch.tensor(
                    right_local_indices,
                    dtype=torch.long,
                    device=device,
                )
            ),

        "yz_indices":
            torch.tensor(
                yz_indices,
                dtype=torch.long,
                device=device,
            ),

        "h_left":
            torch.tensor(
                h_left,
                dtype=torch.float32,
                device=device,
            ).reshape(
                1,
                -1,
                1,
            ),

        "h_right":
            torch.tensor(
                h_right,
                dtype=torch.float32,
                device=device,
            ).reshape(
                1,
                -1,
                1,
            ),

        "branch_dim":
            cursor,
    }


    # Convert interface PCA data to torch.
    for side_name in [
        "left",
        "right",
    ]:

        side = context[
            side_name
        ]


        if side is None:

            continue


        side[
            "mean_tensor"
        ] = torch.tensor(

            side[
                "mean"
            ],

            dtype=torch.float32,

            device=device,

        )


        side[
            "basis_tensor"
        ] = torch.tensor(

            side[
                "basis"
            ],

            dtype=torch.float32,

            device=device,

        )


    return context


# ============================================================
# RECONSTRUCT INTERFACE TARGET
# ============================================================

def reconstruct_interface_target(
    branch_physical,
    side,
    coefficient_slice,
):

    coefficients = branch_physical[
        :,
        coefficient_slice
    ]


    mean = side[
        "mean_tensor"
    ]


    basis = side[
        "basis_tensor"
    ]


    target_flat = (
        mean.reshape(
            1,
            -1
        )
        +
        coefficients
        @
        basis.T
    )


    number_nodes = (
        side[
            "coordinates"
        ].shape[0]
    )


    target = target_flat.reshape(
        branch_physical.shape[
            0
        ],
        number_nodes,
        3,
    )


    return target


# ============================================================
# APPLY HARD COMPATIBILITY
# ============================================================

def apply_hard_compatibility_7region(

    raw_displacement,

    branch_physical,

    context,
):

    corrected = raw_displacement


    yz_indices = context[
        "yz_indices"
    ]


    # ========================================================
    # LEFT INTERFACE
    # ========================================================

    if context[
        "left"
    ] is not None:

        target_left = (
            reconstruct_interface_target(

                branch_physical,

                context[
                    "left"
                ],

                context[
                    "left_slice"
                ],
            )
        )


        raw_left = raw_displacement[
            :,
            context[
                "left_local_indices"
            ],
            :,
        ]


        mismatch_left = (
            target_left
            -
            raw_left
        )


        node_correction_left = (
            mismatch_left[
                :,
                yz_indices,
                :
            ]
        )


        corrected = (
            corrected
            +
            context[
                "h_left"
            ]
            *
            node_correction_left
        )


    # ========================================================
    # RIGHT INTERFACE
    # ========================================================

    if context[
        "right"
    ] is not None:

        target_right = (
            reconstruct_interface_target(

                branch_physical,

                context[
                    "right"
                ],

                context[
                    "right_slice"
                ],
            )
        )


        raw_right = raw_displacement[
            :,
            context[
                "right_local_indices"
            ],
            :,
        ]


        mismatch_right = (
            target_right
            -
            raw_right
        )


        node_correction_right = (
            mismatch_right[
                :,
                yz_indices,
                :
            ]
        )


        corrected = (
            corrected
            +
            context[
                "h_right"
            ]
            *
            node_correction_right
        )


    return corrected