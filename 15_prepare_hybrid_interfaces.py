import os
import json

import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data"

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "hybrid_interfaces",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


TARGET_PLANES = {
    "m6": -6.0,
    "m2": -2.0,
    "p2":  2.0,
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


number_cases = U.shape[0]


# ============================================================
# HELPERS
# ============================================================

def fit_pca(
    matrix,
    energy_threshold,
    min_modes,
    max_modes,
):
    """
    matrix:
        [number_cases, number_features]
    """

    mean = matrix.mean(
        axis=0
    )

    centered = (
        matrix
        - mean
    )

    U_svd, singular_values, Vt = (
        np.linalg.svd(
            centered,
            full_matrices=False,
        )
    )

    variance = (
        singular_values
        ** 2
    )

    total_variance = (
        variance.sum()
    )

    if total_variance <= 0.0:

        number_modes = min_modes

    else:

        cumulative = (
            np.cumsum(
                variance
            )
            / total_variance
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

    scores = (
        centered
        @ basis
    )

    if total_variance > 0.0:

        retained_energy = (
            variance[
                :number_modes
            ].sum()
            / total_variance
        )

    else:

        retained_energy = 1.0

    return (
        mean,
        basis,
        scores,
        retained_energy,
        singular_values,
    )


# ============================================================
# UNIQUE X PLANES
# ============================================================

unique_x = np.unique(
    np.round(
        coordinates[
            :,
            0
        ],
        decimals=8,
    )
)


print("")
print(
    "Available x planes:"
)

print(
    unique_x
)


# ============================================================
# PROCESS EACH INTERFACE
# ============================================================

metadata = {}


for (
    interface_name,
    requested_x,
) in TARGET_PLANES.items():

    nearest_x = unique_x[
        np.argmin(
            np.abs(
                unique_x
                - requested_x
            )
        )
    ]


    print("")
    print(
        "============================================"
    )

    print(
        "Interface:",
        interface_name
    )

    print(
        "Requested x:",
        requested_x
    )

    print(
        "Using actual mesh plane:",
        nearest_x
    )


    mask = np.isclose(
        coordinates[
            :,
            0
        ],
        nearest_x,
        atol=1.0e-7,
    )


    indices = np.where(
        mask
    )[0]


    if len(
        indices
    ) == 0:

        raise RuntimeError(
            "No nodes found on interface "
            + interface_name
        )


    face_coordinates = coordinates[
        indices
    ]


    # Sort by y, then z so the interface ordering
    # is deterministic.
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


    indices = indices[
        order
    ]


    face_coordinates = coordinates[
        indices
    ]


    labels = node_labels[
        indices
    ]


    displacement = U[
        :,
        indices,
        :
    ]


    # Flatten:
    #
    # [case, node, component]
    #
    # ->
    #
    # [case, node*3]
    #
    flattened = displacement.reshape(
        number_cases,
        -1,
    )


    (
        mean,
        basis,
        scores,
        retained_energy,
        singular_values,
    ) = fit_pca(
        matrix=flattened,
        energy_threshold=(
            ENERGY_THRESHOLD
        ),
        min_modes=MIN_MODES,
        max_modes=MAX_MODES,
    )


    output_file = os.path.join(
        OUTPUT_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    np.savez(
        output_file,

        requested_x=np.array(
            [
                requested_x
            ]
        ),

        actual_x=np.array(
            [
                nearest_x
            ]
        ),

        node_indices=indices,

        node_labels=labels,

        coordinates=face_coordinates,

        mean=mean,

        basis=basis,

        scores=scores,

        singular_values=singular_values,

        retained_energy=np.array(
            [
                retained_energy
            ]
        ),
    )


    metadata[
        interface_name
    ] = {

        "requested_x":
            float(
                requested_x
            ),

        "actual_x":
            float(
                nearest_x
            ),

        "number_nodes":
            int(
                len(
                    indices
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
    }


    print(
        "Nodes:",
        len(
            indices
        )
    )

    print(
        "PCA modes:",
        basis.shape[
            1
        ]
    )

    print(
        "Retained energy:",
        retained_energy
    )

    print(
        "Saved:",
        output_file
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
    "============================================"
)

print(
    "INTERFACE PREPARATION COMPLETE"
)

print(
    "============================================"
)

print(
    metadata_file
)