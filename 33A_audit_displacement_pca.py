import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)

INTERFACE_DIR = os.path.join(
    DATA_DIR,
    "hybrid_interfaces_trainonly",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "interface_information_audit",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


COORDINATES_FILE = os.path.join(
    DATA_DIR,
    "coordinates.npy",
)

U1_FILE = os.path.join(
    DATA_DIR,
    "U1.npy",
)

U2_FILE = os.path.join(
    DATA_DIR,
    "U2.npy",
)

U3_FILE = os.path.join(
    DATA_DIR,
    "U3.npy",
)

SPLIT_FILE = os.path.join(
    DATA_DIR,
    "split_assignment.csv",
)


INTERFACES = [
    "m6",
    "m2",
    "p2",
    "p6",
]


# ============================================================
# HELPERS
# ============================================================

def coordinate_key(x, y, z):

    return (
        round(float(x), 6),
        round(float(y), 6),
        round(float(z), 6),
    )


def identify_split_column(dataframe):

    for candidate in [
        "Split",
        "split",
        "Set",
        "set",
        "Partition",
        "partition",
    ]:

        if candidate in dataframe.columns:

            return candidate

    raise RuntimeError(
        "Could not identify split column."
    )


def split_indices(
    dataframe,
    split_name,
):

    split_column = identify_split_column(
        dataframe
    )

    values = (
        dataframe[
            split_column
        ]
        .astype(str)
        .str.lower()
    )


    if split_name == "validation":

        valid_names = [
            "validation",
            "val",
        ]

    else:

        valid_names = [
            split_name.lower()
        ]


    mask = values.isin(
        valid_names
    )


    if "Index" in dataframe.columns:

        return dataframe.loc[
            mask,
            "Index",
        ].to_numpy(
            dtype=int
        )


    return np.where(
        mask.to_numpy()
    )[0]


def relative_l2(
    prediction,
    truth,
):

    denominator = np.linalg.norm(
        truth
    )


    if denominator < 1.0e-14:

        return np.linalg.norm(
            prediction
            -
            truth
        )


    return (
        np.linalg.norm(
            prediction
            -
            truth
        )
        /
        denominator
    )


def fit_pca(
    matrix,
    training_indices,
):

    training = matrix[
        training_indices
    ]


    mean = training.mean(
        axis=0
    )


    centered = (
        training
        -
        mean
    )


    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )


    basis = Vt.T


    energy = (
        singular_values
        ** 2
    )


    cumulative_energy = (
        np.cumsum(
            energy
        )
        /
        np.sum(
            energy
        )
    )


    return (
        mean,
        basis,
        cumulative_energy,
    )


def reconstruction_errors(
    matrix,
    indices,
    mean,
    basis,
):

    errors = []


    for case_index in indices:

        vector = matrix[
            case_index
        ]


        coefficients = (
            basis.T
            @
            (
                vector
                -
                mean
            )
        )


        reconstructed = (
            mean
            +
            basis
            @
            coefficients
        )


        errors.append(
            relative_l2(
                reconstructed,
                vector,
            )
        )


    return np.asarray(
        errors,
        dtype=np.float64,
    )


# ============================================================
# LOAD FULL DATA
# ============================================================

coordinates = np.load(
    COORDINATES_FILE
).astype(
    np.float64
)


U1 = np.load(
    U1_FILE
).astype(
    np.float64
)


U2 = np.load(
    U2_FILE
).astype(
    np.float64
)


U3 = np.load(
    U3_FILE
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


split_dataframe = pd.read_csv(
    SPLIT_FILE
)


train_indices = split_indices(
    split_dataframe,
    "train",
)


validation_indices = split_indices(
    split_dataframe,
    "validation",
)


test_indices = split_indices(
    split_dataframe,
    "test",
)


coordinate_map = {

    coordinate_key(
        xyz[0],
        xyz[1],
        xyz[2],
    ):
        index

    for index, xyz in enumerate(
        coordinates
    )
}


print("")
print(
    "Cases:"
)

print(
    "Train      =",
    len(train_indices),
)

print(
    "Validation =",
    len(validation_indices),
)

print(
    "Test       =",
    len(test_indices),
)


# ============================================================
# AUDIT
# ============================================================

summary_rows = []

case_rows = []


for interface_name in INTERFACES:

    interface_file = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    if not os.path.isfile(
        interface_file
    ):

        raise FileNotFoundError(
            interface_file
        )


    interface = np.load(
        interface_file
    )


    interface_coordinates = interface[
        "coordinates"
    ].astype(
        np.float64
    )


    current_mean = interface[
        "mean"
    ].astype(
        np.float64
    )


    current_basis = interface[
        "basis"
    ].astype(
        np.float64
    )


    current_modes = current_basis.shape[
        1
    ]


    full_node_indices = []


    for xyz in interface_coordinates:

        key = coordinate_key(
            xyz[0],
            xyz[1],
            xyz[2],
        )


        if key not in coordinate_map:

            raise RuntimeError(
                "Interface coordinate not found "
                "in full dataset: {}".format(
                    key
                )
            )


        full_node_indices.append(
            coordinate_map[
                key
            ]
        )


    full_node_indices = np.asarray(
        full_node_indices,
        dtype=int,
    )


    interface_matrix = (
        U[
            :,
            full_node_indices,
            :
        ]
        .reshape(
            U.shape[0],
            -1,
        )
    )


    (
        recomputed_mean,
        recomputed_basis,
        cumulative_energy,
    ) = fit_pca(
        interface_matrix,
        train_indices,
    )


    candidate_modes = sorted(
        set(
            [
                current_modes,
                current_modes + 1,
                current_modes + 2,
                current_modes + 4,
                6,
                8,
                10,
                12,
                16,
                20,
            ]
        )
    )


    candidate_modes = [

        modes

        for modes in candidate_modes

        if modes <= recomputed_basis.shape[1]
    ]


    # --------------------------------------------------------
    # CURRENT SAVED BASIS
    # --------------------------------------------------------

    configurations = [

        (
            "current_saved",
            current_modes,
            current_mean,
            current_basis,
            np.nan,
        )
    ]


    # --------------------------------------------------------
    # RECOMPUTED TRAIN-ONLY BASES
    # --------------------------------------------------------

    for modes in candidate_modes:

        configurations.append(
            (
                "recomputed_train_only",
                modes,
                recomputed_mean,
                recomputed_basis[
                    :,
                    :modes
                ],
                cumulative_energy[
                    modes
                    -
                    1
                ],
            )
        )


    for (
        basis_type,
        modes,
        mean,
        basis,
        energy,
    ) in configurations:

        for split_name, indices in [

            (
                "train",
                train_indices,
            ),

            (
                "validation",
                validation_indices,
            ),

            (
                "test",
                test_indices,
            ),
        ]:

            errors = reconstruction_errors(
                interface_matrix,
                indices,
                mean,
                basis,
            )


            summary_rows.append(
                {

                    "Interface":
                        interface_name,

                    "BasisType":
                        basis_type,

                    "Modes":
                        modes,

                    "Split":
                        split_name,

                    "CumulativeEnergy":
                        energy,

                    "MeanRelativeL2":
                        float(
                            np.mean(
                                errors
                            )
                        ),

                    "MedianRelativeL2":
                        float(
                            np.median(
                                errors
                            )
                        ),

                    "MaxRelativeL2":
                        float(
                            np.max(
                                errors
                            )
                        ),

                    "MeanRelativeL2_percent":
                        float(
                            100.0
                            *
                            np.mean(
                                errors
                            )
                        ),

                    "MaxRelativeL2_percent":
                        float(
                            100.0
                            *
                            np.max(
                                errors
                            )
                        ),
                }
            )


            for local_index, case_index in enumerate(
                indices
            ):

                case_rows.append(
                    {

                        "Interface":
                            interface_name,

                        "BasisType":
                            basis_type,

                        "Modes":
                            modes,

                        "Split":
                            split_name,

                        "CaseIndex":
                            int(
                                case_index
                            ),

                        "RelativeL2":
                            float(
                                errors[
                                    local_index
                                ]
                            ),

                        "RelativeL2_percent":
                            float(
                                100.0
                                *
                                errors[
                                    local_index
                                ]
                            ),
                    }
                )


    print("")
    print(
        interface_name,
        "| current modes =",
        current_modes,
    )


# ============================================================
# SAVE
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)


case_dataframe = pd.DataFrame(
    case_rows
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "displacement_pca_audit.csv",
)


case_file = os.path.join(
    OUTPUT_DIR,
    "displacement_pca_case_errors.csv",
)


summary_dataframe.to_csv(
    summary_file,
    index=False,
)


case_dataframe.to_csv(
    case_file,
    index=False,
)


# ============================================================
# PRINT CURRENT BASIS TEST PERFORMANCE
# ============================================================

current_test = summary_dataframe[
    (
        summary_dataframe[
            "BasisType"
        ]
        ==
        "current_saved"
    )
    &
    (
        summary_dataframe[
            "Split"
        ]
        ==
        "test"
    )
].copy()


print("")
print(
    "================================================"
)

print(
    "CURRENT SAVED PCA — TEST RECONSTRUCTION"
)

print(
    "================================================"
)


print(
    current_test[
        [
            "Interface",
            "Modes",
            "MeanRelativeL2_percent",
            "MaxRelativeL2_percent",
        ]
    ].to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    summary_file
)

print(
    case_file
)