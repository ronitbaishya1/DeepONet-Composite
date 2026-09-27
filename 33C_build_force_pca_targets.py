import os
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.getcwd()

INTERFACE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_interfaces_trainonly",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_force_pca",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


SEGMENTS = {

    "left": {

        "dataset_dir":
            os.path.join(
                PROJECT_ROOT,
                "data",
                "hybrid_bulk_left",
            ),

        "extracted_dir":
            os.path.join(
                PROJECT_ROOT,
                "data",
                "windows_hybrid_bulk",
                "left_extracted",
            ),

        "interfaces":
            [
                "m6",
                "m2",
            ],
    },

    "right": {

        "dataset_dir":
            os.path.join(
                PROJECT_ROOT,
                "data",
                "hybrid_bulk_right",
            ),

        "extracted_dir":
            os.path.join(
                PROJECT_ROOT,
                "data",
                "windows_hybrid_bulk",
                "right_extracted",
            ),

        "interfaces":
            [
                "p2",
                "p6",
            ],
    },
}


NUMBER_FORCE_MODES = 8


# ============================================================
# HELPERS
# ============================================================

def coordinate_key(x, y, z):

    return (
        round(float(x), 6),
        round(float(y), 6),
        round(float(z), 6),
    )


def get_split_indices(
    dataframe,
    split_name,
):

    split_column = None

    for candidate in [
        "Split",
        "split",
    ]:

        if candidate in dataframe.columns:

            split_column = candidate

            break


    if split_column is None:

        raise RuntimeError(
            "Could not find Split column."
        )


    values = (
        dataframe[
            split_column
        ]
        .astype(str)
        .str.lower()
    )


    if split_name == "validation":

        mask = values.isin(
            [
                "validation",
                "val",
            ]
        )

    else:

        mask = (
            values
            ==
            split_name.lower()
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

    return (
        np.linalg.norm(
            prediction
            -
            truth
        )
        /
        (
            np.linalg.norm(
                truth
            )
            +
            1.0e-14
        )
    )


# ============================================================
# MAIN LOOP
# ============================================================

summary_rows = []

mapping_rows = []


for segment_name, configuration in SEGMENTS.items():

    print("")
    print(
        "================================================"
    )

    print(
        "SEGMENT:",
        segment_name
    )

    print(
        "================================================"
    )


    dataset_dir = configuration[
        "dataset_dir"
    ]


    extracted_dir = configuration[
        "extracted_dir"
    ]


    split_dataframe = pd.read_csv(
        os.path.join(
            dataset_dir,
            "split_assignment.csv",
        )
    )


    case_ids = (
        split_dataframe[
            "CaseID"
        ]
        .astype(str)
        .to_list()
    )


    train_indices = get_split_indices(
        split_dataframe,
        "train",
    )


    validation_indices = get_split_indices(
        split_dataframe,
        "validation",
    )


    test_indices = get_split_indices(
        split_dataframe,
        "test",
    )


    stored_generalized_forces = [

        np.load(
            os.path.join(
                dataset_dir,
                "g_left.npy",
            )
        ),

        np.load(
            os.path.join(
                dataset_dir,
                "g_right.npy",
            )
        ),
    ]


    segment_force_coefficients = []


    for local_interface_index, interface_name in enumerate(
        configuration[
            "interfaces"
        ]
    ):

        print("")
        print(
            "Interface:",
            interface_name
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


        # ====================================================
        # READ FULL NODAL REACTION/FORCE VECTOR
        # ====================================================

        force_vectors = []


        for case_id in case_ids:

            node_file = os.path.join(
                extracted_dir,
                "{}_nodes.csv".format(
                    case_id
                ),
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


            force_map = {}


            for _, row in nodes.iterrows():

                key = coordinate_key(

                    row[
                        "X"
                    ],

                    row[
                        "Y"
                    ],

                    row[
                        "Z"
                    ],
                )


                force_map[
                    key
                ] = np.array(
                    [
                        row[
                            "RF1"
                        ],

                        row[
                            "RF2"
                        ],

                        row[
                            "RF3"
                        ],
                    ],
                    dtype=np.float64,
                )


            interface_force = []


            for xyz in interface_coordinates:

                key = coordinate_key(
                    xyz[
                        0
                    ],
                    xyz[
                        1
                    ],
                    xyz[
                        2
                    ],
                )


                if key not in force_map:

                    raise RuntimeError(
                        "Could not find interface "
                        "coordinate {} in {}".format(
                            key,
                            node_file,
                        )
                    )


                interface_force.append(
                    force_map[
                        key
                    ]
                )


            interface_force = np.asarray(
                interface_force,
                dtype=np.float64,
            ).reshape(
                -1
            )


            force_vectors.append(
                interface_force
            )


        force_vectors = np.stack(
            force_vectors,
            axis=0,
        )


        # ====================================================
        # FIT PCA USING TRAINING CASES ONLY
        # ====================================================

        training_forces = force_vectors[
            train_indices
        ]


        force_mean = training_forces.mean(
            axis=0
        )


        centered_training = (
            training_forces
            -
            force_mean
        )


        _, singular_values, Vt = np.linalg.svd(
            centered_training,
            full_matrices=False,
        )


        full_force_basis = Vt.T


        force_basis = full_force_basis[
            :,
            :NUMBER_FORCE_MODES
        ]


        energy = singular_values ** 2


        cumulative_energy = (
            np.cumsum(
                energy
            )
            /
            np.sum(
                energy
            )
        )


        retained_energy = cumulative_energy[
            NUMBER_FORCE_MODES
            -
            1
        ]


        # ====================================================
        # PCA COEFFICIENTS FOR ALL 80 CASES
        # ====================================================

        force_coefficients = (

            force_vectors
            -
            force_mean

        ) @ force_basis


        reconstructed_force = (

            force_mean.reshape(
                1,
                -1
            )

            +

            force_coefficients
            @
            force_basis.T
        )


        segment_force_coefficients.append(
            force_coefficients.astype(
                np.float32
            )
        )


        # ====================================================
        # MAP FORCE PCA -> ORIGINAL GENERALIZED FORCE
        #
        # Original coupling uses:
        #
        # g = B_u^T f
        #
        # With force PCA:
        #
        # f = mean + B_f a
        #
        # therefore:
        #
        # g = g_mean + M a
        # ====================================================

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


        g_from_force_pca = (

            g_mean.reshape(
                1,
                -1
            )

            +

            force_coefficients
            @
            g_matrix.T
        )


        stored_g = stored_generalized_forces[
            local_interface_index
        ]


        # ====================================================
        # CHECK MAPPING
        # ====================================================

        mapping_error = (

            np.linalg.norm(
                g_from_force_pca
                -
                stored_g
            )

            /

            (
                np.linalg.norm(
                    stored_g
                )
                +
                1.0e-14
            )
        )


        mapping_rows.append(
            {

                "Segment":
                    segment_name,

                "Interface":
                    interface_name,

                "MappingRelativeL2":
                    mapping_error,

                "MappingRelativeL2_percent":
                    100.0
                    *
                    mapping_error,
            }
        )


        # ====================================================
        # RECONSTRUCTION ERRORS
        # ====================================================

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

            errors = []


            for case_index in indices:

                errors.append(
                    relative_l2(

                        reconstructed_force[
                            case_index
                        ],

                        force_vectors[
                            case_index
                        ],
                    )
                )


            errors = np.asarray(
                errors
            )


            summary_rows.append(
                {

                    "Segment":
                        segment_name,

                    "Interface":
                        interface_name,

                    "Modes":
                        NUMBER_FORCE_MODES,

                    "Split":
                        split_name,

                    "RetainedEnergy":
                        retained_energy,

                    "MeanRelativeL2_percent":
                        100.0
                        *
                        np.mean(
                            errors
                        ),

                    "MaxRelativeL2_percent":
                        100.0
                        *
                        np.max(
                            errors
                        ),
                }
            )


        # ====================================================
        # SAVE THIS INTERFACE
        # ====================================================

        output_file = os.path.join(
            OUTPUT_DIR,
            "force_pca_{}.npz".format(
                interface_name
            ),
        )


        np.savez(

            output_file,

            mean=
                force_mean.astype(
                    np.float32
                ),

            basis=
                force_basis.astype(
                    np.float32
                ),

            coefficients=
                force_coefficients.astype(
                    np.float32
                ),

            force_vectors=
                force_vectors.astype(
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

            retained_energy=
                np.array(
                    [
                        retained_energy
                    ],
                    dtype=np.float32,
                ),

            number_modes=
                np.array(
                    [
                        NUMBER_FORCE_MODES
                    ],
                    dtype=np.int64,
                ),
        )


        print(
            "Saved:",
            output_file
        )


    # ========================================================
    # COMBINE LEFT + RIGHT FORCE COEFFICIENTS FOR THIS NO BLOCK
    # ========================================================

    combined_force_coefficients = np.concatenate(

        segment_force_coefficients,

        axis=1,
    )


    combined_file = os.path.join(
        OUTPUT_DIR,
        "{}_force_coefficients.npy".format(
            segment_name
        ),
    )


    np.save(
        combined_file,
        combined_force_coefficients,
    )


    print("")
    print(
        "Combined coefficient shape:",
        combined_force_coefficients.shape
    )

    print(
        "Saved:",
        combined_file
    )


# ============================================================
# SAVE SUMMARIES
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)


mapping_dataframe = pd.DataFrame(
    mapping_rows
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "force_pca_summary.csv",
)


mapping_file = os.path.join(
    OUTPUT_DIR,
    "generalized_force_mapping_check.csv",
)


summary_dataframe.to_csv(
    summary_file,
    index=False,
)


mapping_dataframe.to_csv(
    mapping_file,
    index=False,
)


print("")
print(
    "================================================"
)

print(
    "FORCE PCA COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    summary_dataframe[
        summary_dataframe[
            "Split"
        ]
        ==
        "test"
    ].to_string(
        index=False
    )
)


print("")
print(
    "Generalized-force mapping:"
)

print(
    mapping_dataframe.to_string(
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
    mapping_file
)