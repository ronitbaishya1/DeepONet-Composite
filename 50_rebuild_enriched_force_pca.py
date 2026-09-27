import os
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--segment",
    choices=["left", "right"],
    required=True,
)

parser.add_argument(
    "--coordinate_tolerance",
    type=float,
    default=1.0e-4,
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

ROOT = os.getcwd()

DATA_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_bulk_{}_adaptive120".format(args.segment),
)

INTERFACE_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_interfaces_trainonly",
)

ORIGINAL_EXTRACTED_DIR = os.path.join(
    ROOT,
    "data",
    "windows_hybrid_bulk",
    "{}_extracted".format(args.segment),
)

ENRICHMENT_EXTRACTED_DIR = os.path.join(
    ROOT,
    "data",
    "adaptive_enrichment",
    "windows",
    "{}_extracted".format(args.segment),
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_force_pca_enriched",
    args.segment,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# INTERFACES
# ============================================================

if args.segment == "left":
    INTERFACES = ["m6", "m2"]
else:
    INTERFACES = ["p2", "p6"]


CANDIDATE_MODES = [
    8,
    10,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
]


# ============================================================
# LOAD SPLIT
# ============================================================

split_df = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)

if "CaseID" not in split_df.columns:
    raise RuntimeError(
        "split_assignment.csv must contain CaseID."
    )

if "Split" in split_df.columns:
    split_column = "Split"
elif "split" in split_df.columns:
    split_column = "split"
else:
    raise RuntimeError(
        "Could not find Split/split column."
    )

case_ids = split_df["CaseID"].astype(str).to_numpy()

split_values = (
    split_df[split_column]
    .astype(str)
    .str.lower()
    .to_numpy()
)

train_indices = np.where(
    split_values == "train"
)[0]

validation_indices = np.where(
    np.isin(
        split_values,
        ["validation", "val"],
    )
)[0]

test_indices = np.where(
    split_values == "test"
)[0]


print("")
print("Segment:", args.segment)
print("Total:", len(case_ids))
print("Train:", len(train_indices))
print("Validation:", len(validation_indices))
print("Test:", len(test_indices))


if len(train_indices) != 96:
    print(
        "WARNING: expected 96 training cases, found",
        len(train_indices),
    )


# ============================================================
# HELPERS
# ============================================================

def find_node_file(case_id):

    original_file = os.path.join(
        ORIGINAL_EXTRACTED_DIR,
        "{}_nodes.csv".format(case_id),
    )

    enrichment_file = os.path.join(
        ENRICHMENT_EXTRACTED_DIR,
        "{}_nodes.csv".format(case_id),
    )

    if os.path.isfile(original_file):
        return original_file

    if os.path.isfile(enrichment_file):
        return enrichment_file

    raise FileNotFoundError(
        "Could not find nodal CSV for case {}\n"
        "Checked:\n{}\n{}".format(
            case_id,
            original_file,
            enrichment_file,
        )
    )


def nearest_mapping(
    reference_coordinates,
    source_coordinates,
    tolerance,
):

    reference_coordinates = np.asarray(
        reference_coordinates,
        dtype=np.float64,
    )

    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64,
    )

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(
            source_coordinates
        )

        distances, indices = tree.query(
            reference_coordinates,
            k=1,
        )

    except ImportError:

        distances = []
        indices = []

        for reference in reference_coordinates:

            difference = (
                source_coordinates
                -
                reference.reshape(1, 3)
            )

            distance = np.sqrt(
                np.sum(
                    difference ** 2,
                    axis=1,
                )
            )

            index = int(
                np.argmin(distance)
            )

            distances.append(
                distance[index]
            )

            indices.append(index)

        distances = np.asarray(distances)

        indices = np.asarray(
            indices,
            dtype=int,
        )

    maximum_distance = float(
        np.max(distances)
    )

    if maximum_distance > tolerance:

        worst = int(
            np.argmax(distances)
        )

        raise RuntimeError(
            "Coordinate matching failed.\n"
            "Maximum distance = {:.8e}\n"
            "Tolerance        = {:.8e}\n"
            "Reference        = {}\n"
            "Nearest          = {}".format(
                maximum_distance,
                tolerance,
                reference_coordinates[worst],
                source_coordinates[
                    indices[worst]
                ],
            )
        )

    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(
            "Coordinate mapping is not one-to-one."
        )

    return indices


def row_relative_l2(
    prediction,
    truth,
):

    numerator = np.linalg.norm(
        prediction - truth,
        axis=1,
    )

    denominator = np.linalg.norm(
        truth,
        axis=1,
    ) + 1.0e-14

    return numerator / denominator


# ============================================================
# BUILD AUDIT
# ============================================================

audit_rows = []


for interface_name in INTERFACES:

    print("")
    print(
        "========================================"
    )
    print(
        "INTERFACE:",
        interface_name
    )
    print(
        "========================================"
    )

    interface_file = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )

    interface = np.load(
        interface_file
    )

    interface_coordinates = interface[
        "coordinates"
    ].astype(
        np.float64
    )

    displacement_basis = interface[
        "basis"
    ].astype(
        np.float64
    )

    # --------------------------------------------------------
    # Read nodal RF for all 120 cases
    # --------------------------------------------------------

    force_vectors = []

    max_coordinate_distance = 0.0

    for counter, case_id in enumerate(case_ids):

        node_file = find_node_file(
            case_id
        )

        nodes = pd.read_csv(
            node_file
        )

        source_coordinates = nodes[
            ["X", "Y", "Z"]
        ].to_numpy(
            dtype=np.float64
        )

        source_force = nodes[
            ["RF1", "RF2", "RF3"]
        ].to_numpy(
            dtype=np.float64
        )

        mapping = nearest_mapping(
            interface_coordinates,
            source_coordinates,
            args.coordinate_tolerance,
        )

        force_vector = source_force[
            mapping
        ].reshape(
            -1
        )

        force_vectors.append(
            force_vector
        )

    force_vectors = np.asarray(
        force_vectors,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Save raw force vectors
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "{}_force_vectors.npy".format(
                interface_name
            ),
        ),
        force_vectors.astype(
            np.float32
        ),
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "{}_case_ids.npy".format(
                interface_name
            ),
        ),
        case_ids,
    )

    # --------------------------------------------------------
    # Fit PCA using ONLY the 96 training cases
    # --------------------------------------------------------

    training_force = force_vectors[
        train_indices
    ]

    force_mean = training_force.mean(
        axis=0
    )

    centered_training = (
        training_force
        -
        force_mean
    )

    _, singular_values, Vt = np.linalg.svd(
        centered_training,
        full_matrices=False,
    )

    full_basis = Vt.T

    energy = singular_values ** 2

    cumulative_energy = (
        np.cumsum(energy)
        /
        np.sum(energy)
    )

    # --------------------------------------------------------
    # Test candidate mode counts
    # --------------------------------------------------------

    for number_modes in CANDIDATE_MODES:

        if number_modes > full_basis.shape[1]:
            continue

        force_basis = full_basis[
            :,
            :number_modes
        ]

        coefficients = (
            force_vectors
            -
            force_mean
        ) @ force_basis

        reconstructed_force = (
            force_mean.reshape(1, -1)
            +
            coefficients
            @ force_basis.T
        )

        # ----------------------------------------------------
        # Mapping to generalized force
        # ----------------------------------------------------

        g_mean = (
            displacement_basis.T
            @
            force_mean
        )

        g_matrix = (
            displacement_basis.T
            @
            force_basis
        )

        g_direct = (
            force_vectors
            @
            displacement_basis
        )

        g_from_pca = (
            g_mean.reshape(1, -1)
            +
            coefficients
            @
            g_matrix.T
        )

        # ----------------------------------------------------
        # Save this candidate basis
        # ----------------------------------------------------

        candidate_file = os.path.join(
            OUTPUT_DIR,
            "force_pca_{}_m{:02d}.npz".format(
                interface_name,
                number_modes,
            ),
        )

        np.savez(
            candidate_file,

            mean=
                force_mean.astype(
                    np.float32
                ),

            basis=
                force_basis.astype(
                    np.float32
                ),

            coordinates=
                interface_coordinates.astype(
                    np.float32
                ),

            displacement_basis=
                displacement_basis.astype(
                    np.float32
                ),

            g_mean=
                g_mean.astype(
                    np.float32
                ),

            g_matrix=
                g_matrix.astype(
                    np.float32
                ),

            number_modes=
                np.array(
                    [number_modes],
                    dtype=np.int64,
                ),

            retained_energy=
                np.array(
                    [
                        cumulative_energy[
                            number_modes - 1
                        ]
                    ],
                    dtype=np.float32,
                ),
        )

        # ----------------------------------------------------
        # Audit TRAIN and VALIDATION only
        #
        # Test is intentionally NOT used for mode selection.
        # ----------------------------------------------------

        for split_name, indices in [
            ("train", train_indices),
            ("validation", validation_indices),
        ]:

            force_errors = (
                row_relative_l2(
                    reconstructed_force[
                        indices
                    ],
                    force_vectors[
                        indices
                    ],
                )
                *
                100.0
            )

            g_errors = (
                row_relative_l2(
                    g_from_pca[
                        indices
                    ],
                    g_direct[
                        indices
                    ],
                )
                *
                100.0
            )

            audit_rows.append(
                {
                    "Segment":
                        args.segment,

                    "Interface":
                        interface_name,

                    "Modes":
                        number_modes,

                    "Split":
                        split_name,

                    "RetainedEnergy_percent":
                        100.0
                        *
                        cumulative_energy[
                            number_modes - 1
                        ],

                    "ForceReconMean_percent":
                        float(
                            np.mean(
                                force_errors
                            )
                        ),

                    "ForceReconMax_percent":
                        float(
                            np.max(
                                force_errors
                            )
                        ),

                    "GeneralizedForceMean_percent":
                        float(
                            np.mean(
                                g_errors
                            )
                        ),

                    "GeneralizedForceMax_percent":
                        float(
                            np.max(
                                g_errors
                            )
                        ),
                }
            )

        print(
            "Modes {:2d} | energy {:.6f}%"
            .format(
                number_modes,
                100.0
                *
                cumulative_energy[
                    number_modes - 1
                ],
            )
        )


# ============================================================
# SAVE AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_file = os.path.join(
    OUTPUT_DIR,
    "mode_audit.csv",
)

audit_df.to_csv(
    audit_file,
    index=False,
)


print("")
print(
    "========================================"
)
print(
    "STEP 50 COMPLETE"
)
print(
    "========================================"
)

print(
    "Saved:"
)

print(
    audit_file
)

print("")
print(
    audit_df.to_string(
        index=False
    )
)