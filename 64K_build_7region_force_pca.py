import os
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = os.path.join(
    "data",
    "hybrid_bulk_7region",
)


INTERFACE_DIR = os.path.join(
    "data",
    "candidate_partition_interfaces_7region",
)


OUTPUT_DIR = os.path.join(
    "data",
    "hybrid_force_pca_7region",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

NUMBER_FORCE_MODES = 8


SEGMENTS = {

    "outer_left": {
        "left_interface": None,
        "right_interface": "left_outer",
    },

    "inner_left": {
        "left_interface": "left_inner",
        "right_interface": "center_left",
    },

    "inner_right": {
        "left_interface": "center_right",
        "right_interface": "right_inner",
    },

    "outer_right": {
        "left_interface": "right_outer",
        "right_interface": None,
    },
}


# ============================================================
# HELPERS
# ============================================================

def get_split_indices(
    dataframe,
    split_name,
):

    values = (
        dataframe[
            "Split"
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
            split_name
        )


    return dataframe.loc[
        mask,
        "Index",
    ].to_numpy(
        dtype=int
    )


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
# RESULTS
# ============================================================

summary_rows = []

mapping_rows = []


# ============================================================
# SEGMENT LOOP
# ============================================================

for segment_name, config in SEGMENTS.items():

    print("")
    print(
        "=========================================="
    )

    print(
        "SEGMENT:",
        segment_name.upper()
    )

    print(
        "=========================================="
    )


    data_dir = os.path.join(
        DATA_ROOT,
        segment_name,
    )


    split_dataframe = pd.read_csv(
        os.path.join(
            data_dir,
            "split_assignment.csv",
        )
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


    combined_force_coefficients = []


    # ========================================================
    # LEFT AND RIGHT SIDE
    # ========================================================

    for side_name in [
        "left",
        "right",
    ]:

        interface_name = config[
            "{}_interface".format(
                side_name
            )
        ]


        if interface_name is None:

            continue


        print("")
        print(
            "Interface:",
            interface_name
        )


        # ====================================================
        # RAW RF FIELD
        # ====================================================

        raw_force_file = os.path.join(
            data_dir,
            "rf_{}.npy".format(
                side_name
            ),
        )


        force_vectors = np.load(
            raw_force_file
        ).astype(
            np.float64
        )


        if (
            force_vectors.ndim != 2
            or
            force_vectors.shape[1] == 0
        ):

            raise RuntimeError(
                "Invalid RF field for {}."
                .format(
                    interface_name
                )
            )


        # ====================================================
        # STORED GENERALIZED FORCE
        # ====================================================

        stored_g = np.load(
            os.path.join(
                data_dir,
                "g_{}.npy".format(
                    side_name
                ),
            )
        ).astype(
            np.float64
        )


        # ====================================================
        # DISPLACEMENT PCA BASIS
        # ====================================================

        interface = np.load(
            os.path.join(
                INTERFACE_DIR,
                "interface_{}.npz".format(
                    interface_name
                ),
            )
        )


        displacement_basis = interface[
            "basis"
        ].astype(
            np.float64
        )


        interface_coordinates = interface[
            "coordinates"
        ].astype(
            np.float64
        )


        if force_vectors.shape[1] != (
            displacement_basis.shape[0]
        ):

            raise RuntimeError(
                "RF/displacement basis dimension mismatch "
                "at {}.".format(
                    interface_name
                )
            )


        # ====================================================
        # TRAIN-ONLY FORCE PCA
        # ====================================================

        training_force = force_vectors[
            train_indices
        ]


        force_mean = training_force.mean(
            axis=0
        )


        centered = (
            training_force
            -
            force_mean
        )


        _, singular_values, Vt = np.linalg.svd(
            centered,
            full_matrices=False,
        )


        number_modes = min(
            NUMBER_FORCE_MODES,
            Vt.shape[0],
        )


        force_basis = (
            Vt[
                :number_modes
            ].T
        )


        # ====================================================
        # RETAINED ENERGY
        # ====================================================

        energy = (
            singular_values ** 2
        )


        if np.sum(
            energy
        ) <= 1.0e-30:

            retained_energy = 1.0

        else:

            retained_energy = float(
                np.sum(
                    energy[
                        :number_modes
                    ]
                )
                /
                np.sum(
                    energy
                )
            )


        # ====================================================
        # ALL-CASE COEFFICIENTS
        # ====================================================

        coefficients = (
            (
                force_vectors
                -
                force_mean.reshape(
                    1,
                    -1
                )
            )
            @
            force_basis
        )


        reconstructed_force = (
            force_mean.reshape(
                1,
                -1
            )
            +
            coefficients
            @
            force_basis.T
        )


        combined_force_coefficients.append(
            coefficients.astype(
                np.float32
            )
        )


        # ====================================================
        # MAP FORCE PCA -> DISPLACEMENT GENERALIZED FORCE
        #
        # f = f_mean + Bf a
        #
        # g = Bu^T f
        #
        # therefore:
        #
        # g = g_mean + g_matrix a
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
            coefficients
            @
            g_matrix.T
        )


        mapping_error = relative_l2(
            g_from_force_pca,
            stored_g,
        )


        mapping_rows.append(
            {
                "Segment":
                    segment_name,

                "Interface":
                    interface_name,

                "Modes":
                    number_modes,

                "MappingRelativeL2_percent":
                    100.0
                    *
                    mapping_error,
            }
        )


        # ====================================================
        # SPLIT RECONSTRUCTION ERRORS
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

                error = relative_l2(

                    reconstructed_force[
                        case_index
                    ],

                    force_vectors[
                        case_index
                    ],
                )


                errors.append(
                    100.0
                    *
                    error
                )


            summary_rows.append(
                {
                    "Segment":
                        segment_name,

                    "Interface":
                        interface_name,

                    "Split":
                        split_name,

                    "Modes":
                        number_modes,

                    "RetainedEnergy":
                        retained_energy,

                    "MeanRelativeL2_percent":
                        float(
                            np.mean(
                                errors
                            )
                        ),

                    "MaxRelativeL2_percent":
                        float(
                            np.max(
                                errors
                            )
                        ),
                }
            )


        # ====================================================
        # SAVE INTERFACE PCA
        # ====================================================

        output_file = os.path.join(
            OUTPUT_DIR,
            "force_pca_{}.npz".format(
                interface_name
            ),
        )


        np.savez_compressed(

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
                coefficients.astype(
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
                np.asarray(
                    [
                        retained_energy
                    ],
                    dtype=np.float32,
                ),

            number_modes=
                np.asarray(
                    [
                        number_modes
                    ],
                    dtype=np.int64,
                ),
        )


        print(
            "Modes:",
            number_modes,
        )

        print(
            "Retained energy:",
            retained_energy,
        )

        print(
            "g mapping error (%):",
            100.0
            *
            mapping_error,
        )


    # ========================================================
    # SAVE COEFFICIENT TARGETS FOR THIS NO REGION
    # ========================================================

    force_coefficients = np.concatenate(
        combined_force_coefficients,
        axis=1,
    )


    np.save(
        os.path.join(
            data_dir,
            "force_coefficients.npy",
        ),
        force_coefficients,
    )


    print("")
    print(
        "Force coefficient target shape:",
        force_coefficients.shape
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
    "=========================================="
)

print(
    "STEP 64K COMPLETE"
)

print(
    "=========================================="
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
    "GENERALIZED FORCE MAPPING"
)

print(
    mapping_dataframe.to_string(
        index=False
    )
)