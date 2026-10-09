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
    "--partition",
    default=
        "results/saint_venant_partition/"
        "selected_partition_7region.json",
)


parser.add_argument(
    "--data_dir",
    default=
        "data",
)


parser.add_argument(
    "--output_dir",
    default=
        "data/candidate_partition_interfaces_7region",
)


parser.add_argument(
    "--energy",
    type=float,
    default=
        0.9999,
)


parser.add_argument(
    "--min_modes",
    type=int,
    default=
        3,
)


parser.add_argument(
    "--max_modes",
    type=int,
    default=
        12,
)


parser.add_argument(
    "--x_tolerance",
    type=float,
    default=
        1.0e-5,
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# LOAD PARTITION
# ============================================================

with open(
    args.partition,
    "r",
) as file_object:

    partition = json.load(
        file_object
    )


TARGET_PLANES = {

    interface_name:
        float(
            x_value
        )

    for interface_name, x_value
    in partition[
        "interfaces"
    ].items()
}


EXPECTED_NAMES = [
    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
]


for interface_name in EXPECTED_NAMES:

    if interface_name not in TARGET_PLANES:

        raise RuntimeError(
            "Partition is missing interface {}."
            .format(
                interface_name
            )
        )


# ============================================================
# LOAD FULL-DOMAIN DATA
# ============================================================

coordinates = np.load(
    os.path.join(
        args.data_dir,
        "coordinates.npy",
    )
).astype(
    np.float64
)


node_labels = np.load(
    os.path.join(
        args.data_dir,
        "node_labels.npy",
    )
)


U1 = np.load(
    os.path.join(
        args.data_dir,
        "U1.npy",
    )
).astype(
    np.float64
)


U2 = np.load(
    os.path.join(
        args.data_dir,
        "U2.npy",
    )
).astype(
    np.float64
)


U3 = np.load(
    os.path.join(
        args.data_dir,
        "U3.npy",
    )
).astype(
    np.float64
)


split_dataframe = pd.read_csv(
    os.path.join(
        args.data_dir,
        "split_assignment.csv",
    )
)


# ============================================================
# BASIC CHECKS
# ============================================================

if (
    coordinates.ndim != 2
    or
    coordinates.shape[1] != 3
):

    raise RuntimeError(
        "coordinates.npy must have shape [Nnodes,3]."
    )


number_nodes = coordinates.shape[
    0
]


number_cases = U1.shape[
    0
]


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

    if array.shape != (
        number_cases,
        number_nodes,
    ):

        raise RuntimeError(
            "{} has unexpected shape {}."
            .format(
                name,
                array.shape,
            )
        )


U = np.stack(
    [
        U1,
        U2,
        U3,
    ],
    axis=-1,
)


# ============================================================
# TRAIN-ONLY CASES
# ============================================================

if "Split" in split_dataframe.columns:

    split_column = "Split"

elif "split" in split_dataframe.columns:

    split_column = "split"

else:

    raise RuntimeError(
        "Could not find Split/split column."
    )


split_values = (
    split_dataframe[
        split_column
    ]
    .astype(str)
    .str.lower()
)


train_mask = (
    split_values
    ==
    "train"
)


if "Index" in split_dataframe.columns:

    train_indices = split_dataframe.loc[
        train_mask,
        "Index",
    ].to_numpy(
        dtype=int
    )

else:

    train_indices = np.where(
        train_mask.to_numpy()
    )[0]


if len(
    train_indices
) == 0:

    raise RuntimeError(
        "No training cases found."
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64D"
)

print(
    "BUILD SIX TRAIN-ONLY PCA INTERFACES"
)

print(
    "=========================================="
)

print(
    "Training cases:",
    len(
        train_indices
    ),
)

print("")


# ============================================================
# PCA FUNCTION
# ============================================================

def fit_pca(
    matrix,
):

    mean = np.mean(
        matrix,
        axis=0,
    )


    centered = (
        matrix
        -
        mean.reshape(
            1,
            -1,
        )
    )


    _, singular_values, Vt = (
        np.linalg.svd(
            centered,
            full_matrices=False,
        )
    )


    variance = (
        singular_values ** 2
    )


    total_variance = float(
        np.sum(
            variance
        )
    )


    if total_variance <= 1.0e-30:

        number_modes = (
            args.min_modes
        )

    else:

        cumulative = (
            np.cumsum(
                variance
            )
            /
            total_variance
        )


        number_modes = (
            int(
                np.searchsorted(
                    cumulative,
                    args.energy,
                )
            )
            +
            1
        )


    number_modes = max(
        number_modes,
        args.min_modes,
    )


    number_modes = min(
        number_modes,
        args.max_modes,
        Vt.shape[
            0
        ],
    )


    basis = (
        Vt[
            :number_modes
        ].T
    )


    scores = (
        centered
        @
        basis
    )


    reconstruction = (
        mean.reshape(
            1,
            -1,
        )
        +
        scores
        @
        basis.T
    )


    reconstruction_error = (
        np.linalg.norm(
            reconstruction
            -
            matrix
        )
        /
        (
            np.linalg.norm(
                matrix
            )
            +
            1.0e-30
        )
    )


    if total_variance <= 1.0e-30:

        retained_energy = 1.0

    else:

        retained_energy = float(
            np.sum(
                variance[
                    :number_modes
                ]
            )
            /
            total_variance
        )


    return (
        mean,
        basis,
        scores,
        singular_values,
        retained_energy,
        float(
            reconstruction_error
        ),
    )


# ============================================================
# BUILD INTERFACES
# ============================================================

metadata = {}


reference_yz = None


for interface_name in EXPECTED_NAMES:

    requested_x = TARGET_PLANES[
        interface_name
    ]


    x_distance = np.abs(
        coordinates[
            :,
            0
        ]
        -
        requested_x
    )


    minimum_distance = float(
        np.min(
            x_distance
        )
    )


    if minimum_distance > (
        args.x_tolerance
    ):

        raise RuntimeError(
            (
                "Interface {} requested at x={} but no "
                "matching node plane exists. "
                "Nearest distance = {}"
            ).format(
                interface_name,
                requested_x,
                minimum_distance,
            )
        )


    node_mask = np.isclose(
        coordinates[
            :,
            0
        ],
        requested_x,
        atol=
            args.x_tolerance,
    )


    node_indices = np.where(
        node_mask
    )[0]


    if len(
        node_indices
    ) == 0:

        raise RuntimeError(
            "No nodes found for {}."
            .format(
                interface_name
            )
        )


    # ========================================================
    # ORDER BY Y THEN Z
    # ========================================================

    interface_coordinates = coordinates[
        node_indices
    ]


    order = np.lexsort(
        (
            interface_coordinates[
                :,
                2
            ],

            interface_coordinates[
                :,
                1
            ],
        )
    )


    node_indices = node_indices[
        order
    ]


    interface_coordinates = coordinates[
        node_indices
    ]


    interface_node_labels = node_labels[
        node_indices
    ]


    # ========================================================
    # STRUCTURED CROSS-SECTION CHECK
    # ========================================================

    y_values = np.unique(
        np.round(
            interface_coordinates[
                :,
                1
            ],
            decimals=8,
        )
    )


    z_values = np.unique(
        np.round(
            interface_coordinates[
                :,
                2
            ],
            decimals=8,
        )
    )


    expected_count = (
        len(
            y_values
        )
        *
        len(
            z_values
        )
    )


    if expected_count != len(
        node_indices
    ):

        raise RuntimeError(
            (
                "{} is not a complete structured Y-Z grid. "
                "Found {}, expected {}."
            ).format(
                interface_name,
                len(
                    node_indices
                ),
                expected_count,
            )
        )


    # ========================================================
    # ALL SIX INTERFACES MUST HAVE SAME Y-Z GRID
    # ========================================================

    current_yz = interface_coordinates[
        :,
        1:3
    ]


    if reference_yz is None:

        reference_yz = current_yz.copy()

    else:

        if reference_yz.shape != (
            current_yz.shape
        ):

            raise RuntimeError(
                (
                    "{} has a different Y-Z grid size."
                ).format(
                    interface_name
                )
            )


        if not np.allclose(
            reference_yz,
            current_yz,
            atol=1.0e-6,
        ):

            raise RuntimeError(
                (
                    "{} does not match the reference "
                    "Y-Z cross-section."
                ).format(
                    interface_name
                )
            )


    # ========================================================
    # TRAIN MATRIX
    # ========================================================

    training_matrix = U[
        train_indices
    ][
        :,
        node_indices,
        :
    ].reshape(
        len(
            train_indices
        ),
        -1,
    )


    (
        mean,
        basis,
        training_scores,
        singular_values,
        retained_energy,
        reconstruction_error,
    ) = fit_pca(
        training_matrix
    )


    # ========================================================
    # PROJECT ALL FULL-DOMAIN CASES
    # ========================================================

    all_matrix = U[
        :,
        node_indices,
        :
    ].reshape(
        number_cases,
        -1,
    )


    all_scores = (
        (
            all_matrix
            -
            mean.reshape(
                1,
                -1,
            )
        )
        @
        basis
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        args.output_dir,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    np.savez_compressed(

        output_file,

        actual_x=
            np.asarray(
                [
                    requested_x
                ],
                dtype=np.float64,
            ),

        node_indices=
            node_indices,

        node_labels=
            interface_node_labels,

        coordinates=
            interface_coordinates,

        mean=
            mean,

        basis=
            basis,

        scores_train=
            training_scores,

        scores_all=
            all_scores,

        train_case_indices=
            train_indices,

        singular_values=
            singular_values,

        retained_energy=
            np.asarray(
                [
                    retained_energy
                ],
                dtype=np.float64,
            ),

        reconstruction_error=
            np.asarray(
                [
                    reconstruction_error
                ],
                dtype=np.float64,
            ),
    )


    metadata[
        interface_name
    ] = {

        "x":
            requested_x,

        "number_nodes":
            int(
                len(
                    node_indices
                )
            ),

        "number_modes":
            int(
                basis.shape[
                    1
                ]
            ),

        "retained_energy":
            retained_energy,

        "training_reconstruction_error":
            reconstruction_error,

        "file":
            output_file,
    }


    print(
        "{}: x={:.6f}, nodes={}, modes={}, "
        "energy={:.8f}, reconstruction={:.6e}"
        .format(

            interface_name,

            requested_x,

            len(
                node_indices
            ),

            basis.shape[
                1
            ],

            retained_energy,

            reconstruction_error,
        )
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_file = os.path.join(
    args.output_dir,
    "interface_metadata.json",
)


with open(
    metadata_file,
    "w",
) as file_object:

    json.dump(
        metadata,
        file_object,
        indent=4,
    )


# Copy partition information as well.
partition_copy_file = os.path.join(
    args.output_dir,
    "partition_used.json",
)


with open(
    partition_copy_file,
    "w",
) as file_object:

    json.dump(
        partition,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64D COMPLETE"
)

print(
    "=========================================="
)

print("")
print(
    "All six interface Y-Z grids match."
)

print(
    "Saved to:"
)

print(
    args.output_dir
)