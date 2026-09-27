import os
import json

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data"

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "hybrid_interfaces_trainonly",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


TARGET_PLANES = {
    "m6": -6.0,
    "m2": -1.8,
    "p2":  1.8,
    "p6":  6.0,
}


ENERGY_THRESHOLD = 0.9999

MIN_MODES = 3

MAX_MODES = 12


# ============================================================
# LOAD DATA
# ============================================================

coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "coordinates.npy",
    )
).astype(
    np.float64
)


node_labels = np.load(
    os.path.join(
        DATA_DIR,
        "node_labels.npy",
    )
)


U1 = np.load(
    os.path.join(
        DATA_DIR,
        "U1.npy",
    )
).astype(
    np.float64
)


U2 = np.load(
    os.path.join(
        DATA_DIR,
        "U2.npy",
    )
).astype(
    np.float64
)


U3 = np.load(
    os.path.join(
        DATA_DIR,
        "U3.npy",
    )
).astype(
    np.float64
)


U = np.stack(
    [
        U1,
        U2,
        U3,
    ],
    axis=-1,
)


split_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)


train_indices = split_table.loc[
    split_table["Split"] == "train",
    "Index",
].to_numpy(
    dtype=int
)


print(
    "Training cases used for PCA:",
    len(train_indices),
)


# ============================================================
# PCA FUNCTION
# ============================================================

def fit_pca(
    matrix,
    energy_threshold,
    min_modes,
    max_modes,
):

    mean = matrix.mean(
        axis=0
    )


    centered = (
        matrix
        - mean
    )


    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )


    variance = (
        singular_values
        ** 2
    )


    total_variance = (
        variance.sum()
    )


    cumulative = (
        np.cumsum(
            variance
        )
        /
        total_variance
    )


    number_modes = (
        np.searchsorted(
            cumulative,
            energy_threshold,
        )
        + 1
    )


    number_modes = max(
        min_modes,
        number_modes,
    )


    number_modes = min(
        max_modes,
        number_modes,
        Vt.shape[0],
    )


    basis = (
        Vt[
            :number_modes
        ].T
    )


    training_scores = (
        centered
        @ basis
    )


    retained_energy = (
        variance[
            :number_modes
        ].sum()
        /
        total_variance
    )


    return (
        mean,
        basis,
        training_scores,
        singular_values,
        retained_energy,
    )


# ============================================================
# PROCESS INTERFACES
# ============================================================

metadata = {}


for (
    interface_name,
    x_target,
) in TARGET_PLANES.items():

    mask = np.isclose(
        coordinates[
            :,
            0
        ],
        x_target,
        atol=1.0e-7,
    )


    node_indices = np.where(
        mask
    )[0]


    if len(
        node_indices
    ) == 0:

        raise RuntimeError(
            "No nodes found at x = {}".format(
                x_target
            )
        )


    face_coordinates = coordinates[
        node_indices
    ]


    # y primary, z secondary
    order = np.lexsort(
        (
            face_coordinates[
                :,
                2
            ],
            face_coordinates[
                :,
                1
            ],
        )
    )


    node_indices = node_indices[
        order
    ]


    face_coordinates = coordinates[
        node_indices
    ]


    face_labels = node_labels[
        node_indices
    ]


    # --------------------------------------------------------
    # TRAINING CASES ONLY
    # --------------------------------------------------------

    training_displacements = U[
        train_indices
    ][
        :,
        node_indices,
        :
    ]


    training_flat = (
        training_displacements.reshape(
            len(
                train_indices
            ),
            -1,
        )
    )


    (
        mean,
        basis,
        training_scores,
        singular_values,
        retained_energy,
    ) = fit_pca(
        training_flat,
        ENERGY_THRESHOLD,
        MIN_MODES,
        MAX_MODES,
    )


    # --------------------------------------------------------
    # PROJECT ALL CASES USING TRAIN-ONLY BASIS
    #
    # This does NOT refit PCA using validation/test data.
    # --------------------------------------------------------

    all_flat = U[
        :,
        node_indices,
        :
    ].reshape(
        U.shape[0],
        -1,
    )


    all_scores = (
        (
            all_flat
            - mean
        )
        @ basis
    )


    # Reconstruction errors
    reconstructed_training = (
        mean.reshape(
            1,
            -1
        )
        +
        training_scores
        @ basis.T
    )


    reconstruction_error = (
        np.linalg.norm(
            reconstructed_training
            - training_flat
        )
        /
        np.linalg.norm(
            training_flat
        )
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    np.savez(
        output_file,

        actual_x=np.array(
            [
                x_target
            ],
            dtype=np.float64,
        ),

        node_indices=node_indices,

        node_labels=face_labels,

        coordinates=face_coordinates,

        mean=mean,

        basis=basis,

        scores_train=training_scores,

        scores_all=all_scores,

        train_case_indices=train_indices,

        singular_values=singular_values,

        retained_energy=np.array(
            [
                retained_energy
            ]
        ),

        reconstruction_error=np.array(
            [
                reconstruction_error
            ]
        ),
    )


    metadata[
        interface_name
    ] = {

        "actual_x":
            float(
                x_target
            ),

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
            float(
                retained_energy
            ),

        "training_reconstruction_error":
            float(
                reconstruction_error
            ),
    }


    print("")
    print(
        interface_name
    )

    print(
        "x =",
        x_target
    )

    print(
        "nodes =",
        len(
            node_indices
        )
    )

    print(
        "modes =",
        basis.shape[
            1
        ]
    )

    print(
        "retained energy =",
        retained_energy
    )

    print(
        "reconstruction error =",
        reconstruction_error
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_file = os.path.join(
    OUTPUT_DIR,
    "interface_metadata.json",
)


with open(
    metadata_file,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print("")
print(
    "Saved:"
)

print(
    metadata_file
)