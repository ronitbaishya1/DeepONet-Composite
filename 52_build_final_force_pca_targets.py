import os
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = os.getcwd()


BASE_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_force_pca_enriched",
)


FINAL_DIR = os.path.join(
    BASE_DIR,
    "final",
)


os.makedirs(
    FINAL_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD SELECTED NUMBER OF MODES
# ============================================================

SELECTION_FILE = os.path.join(
    BASE_DIR,
    "selected_force_modes.json",
)


if not os.path.isfile(
    SELECTION_FILE
):

    raise FileNotFoundError(
        "Could not find:\n{}".format(
            SELECTION_FILE
        )
    )


with open(
    SELECTION_FILE,
    "r",
) as file_object:

    selection = json.load(
        file_object
    )


MODES = int(
    selection[
        "SelectedModes"
    ]
)


print("")
print(
    "========================================"
)

print(
    "STEP 52"
)

print(
    "========================================"
)


print("")
print(
    "Selected force modes per interface:",
    MODES
)


# ============================================================
# IMPORTANT CHECK
# ============================================================

if MODES != 8:

    print("")
    print(
        "NOTE:"
    )

    print(
        "Step 51 selected {} modes instead of 8."
        .format(
            MODES
        )
    )


else:

    print(
        "Confirmed: using 8 force modes per interface."
    )


# ============================================================
# DATA CONFIGURATION
# ============================================================

CONFIG = {

    "left": {

        "interfaces":
            [
                "m6",
                "m2",
            ],

        "data_dir":
            os.path.join(
                ROOT,
                "data",
                "hybrid_bulk_left_adaptive120",
            ),
    },


    "right": {

        "interfaces":
            [
                "p2",
                "p6",
            ],

        "data_dir":
            os.path.join(
                ROOT,
                "data",
                "hybrid_bulk_right_adaptive120",
            ),
    },
}


# ============================================================
# ERROR FUNCTION
# ============================================================

def relative_l2(
    prediction,
    truth,
):

    numerator = np.linalg.norm(
        prediction
        -
        truth
    )


    denominator = (
        np.linalg.norm(
            truth
        )
        +
        1.0e-14
    )


    return numerator / denominator


# ============================================================
# FINAL AUDIT
# ============================================================

audit_rows = []


# ============================================================
# PROCESS LEFT AND RIGHT
# ============================================================

for segment, configuration in CONFIG.items():

    print("")
    print(
        "========================================"
    )

    print(
        "SEGMENT:",
        segment.upper()
    )

    print(
        "========================================"
    )


    segment_pca_dir = os.path.join(
        BASE_DIR,
        segment,
    )


    dataset_dir = configuration[
        "data_dir"
    ]


    # ========================================================
    # LOAD SPLIT INFORMATION
    # ========================================================

    split_file = os.path.join(
        dataset_dir,
        "split_assignment.csv",
    )


    split_dataframe = pd.read_csv(
        split_file
    )


    if "CaseID" not in split_dataframe.columns:

        raise RuntimeError(
            "split_assignment.csv does not contain CaseID."
        )


    case_ids = (
        split_dataframe[
            "CaseID"
        ]
        .astype(str)
        .to_numpy()
    )


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
        .to_numpy()
    )


    train_indices = np.where(
        split_values
        ==
        "train"
    )[0]


    validation_indices = np.where(
        np.isin(
            split_values,
            [
                "validation",
                "val",
            ],
        )
    )[0]


    test_indices = np.where(
        split_values
        ==
        "test"
    )[0]


    print("")
    print(
        "Total cases:",
        len(
            case_ids
        )
    )


    print(
        "Train:",
        len(
            train_indices
        )
    )


    print(
        "Validation:",
        len(
            validation_indices
        )
    )


    print(
        "Test:",
        len(
            test_indices
        )
    )


    if len(
        case_ids
    ) != 120:

        raise RuntimeError(
            "{} dataset should contain 120 cases, found {}."
            .format(
                segment,
                len(
                    case_ids
                ),
            )
        )


    if len(
        train_indices
    ) != 96:

        raise RuntimeError(
            "{} dataset should contain 96 training cases."
            .format(
                segment
            )
        )


    # ========================================================
    # FORCE COEFFICIENTS FROM BOTH INTERFACES
    # ========================================================

    segment_coefficients = []


    for interface_name in configuration[
        "interfaces"
    ]:

        print("")
        print(
            "Interface:",
            interface_name
        )


        # ====================================================
        # LOAD RAW FORCE VECTORS GENERATED IN STEP 50
        # ====================================================

        force_vector_file = os.path.join(
            segment_pca_dir,
            "{}_force_vectors.npy".format(
                interface_name
            ),
        )


        case_id_file = os.path.join(
            segment_pca_dir,
            "{}_case_ids.npy".format(
                interface_name
            ),
        )


        if not os.path.isfile(
            force_vector_file
        ):

            raise FileNotFoundError(
                force_vector_file
            )


        if not os.path.isfile(
            case_id_file
        ):

            raise FileNotFoundError(
                case_id_file
            )


        force_vectors = np.load(
            force_vector_file
        ).astype(
            np.float64
        )


        saved_case_ids = np.load(
            case_id_file,
            allow_pickle=True,
        ).astype(str)


        # ====================================================
        # VERIFY CASE ORDER
        # ====================================================

        if len(
            saved_case_ids
        ) != len(
            case_ids
        ):

            raise RuntimeError(
                "{} case-count mismatch."
                .format(
                    interface_name
                )
            )


        if not np.array_equal(
            saved_case_ids,
            case_ids,
        ):

            raise RuntimeError(

                "{} case ordering does not match "
                "the enriched dataset."
                .format(
                    interface_name
                )
            )


        # ====================================================
        # LOAD THE NEW PCA BASIS CHOSEN IN STEP 51
        # ====================================================

        candidate_file = os.path.join(
            segment_pca_dir,
            "force_pca_{}_m{:02d}.npz".format(
                interface_name,
                MODES,
            ),
        )


        if not os.path.isfile(
            candidate_file
        ):

            raise FileNotFoundError(
                candidate_file
            )


        pca = np.load(
            candidate_file
        )


        mean = pca[
            "mean"
        ].astype(
            np.float64
        )


        basis = pca[
            "basis"
        ].astype(
            np.float64
        )


        coordinates = pca[
            "coordinates"
        ].astype(
            np.float64
        )


        displacement_basis = pca[
            "displacement_basis"
        ].astype(
            np.float64
        )


        g_mean = pca[
            "g_mean"
        ].astype(
            np.float64
        )


        g_matrix = pca[
            "g_matrix"
        ].astype(
            np.float64
        )


        # ====================================================
        # CHECK MODE COUNT
        # ====================================================

        if basis.shape[
            1
        ] != MODES:

            raise RuntimeError(

                "{} PCA basis has {} modes, expected {}."
                .format(

                    interface_name,

                    basis.shape[
                        1
                    ],

                    MODES,
                )
            )


        # ====================================================
        # PROJECT ALL 120 FORCE FIELDS INTO NEW PCA
        # ====================================================

        coefficients = (

            force_vectors

            -

            mean.reshape(
                1,
                -1,
            )

        ) @ basis


        print(
            "Coefficient shape:",
            coefficients.shape
        )


        segment_coefficients.append(
            coefficients.astype(
                np.float32
            )
        )


        # ====================================================
        # RECONSTRUCT FORCE FIELD
        # ====================================================

        reconstructed_force = (

            mean.reshape(
                1,
                -1,
            )

            +

            coefficients
            @
            basis.T
        )


        # ====================================================
        # DIRECT GENERALIZED FORCE
        #
        # g = B_u^T f
        # ====================================================

        g_direct = (

            force_vectors

            @

            displacement_basis
        )


        # ====================================================
        # GENERALIZED FORCE FROM FORCE PCA
        #
        # f = mean + B_f a
        #
        # therefore:
        #
        # g = g_mean + g_matrix a
        # ====================================================

        g_from_pca = (

            g_mean.reshape(
                1,
                -1,
            )

            +

            coefficients

            @

            g_matrix.T
        )


        # ====================================================
        # FINAL AUDIT
        #
        # Test set can be inspected now because Step 51
        # already selected the number of modes without it.
        # ====================================================

        split_groups = [

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
        ]


        for split_name, indices in split_groups:

            force_errors = []

            generalized_force_errors = []


            for case_index in indices:

                force_error = relative_l2(

                    reconstructed_force[
                        case_index
                    ],

                    force_vectors[
                        case_index
                    ],
                )


                generalized_force_error = relative_l2(

                    g_from_pca[
                        case_index
                    ],

                    g_direct[
                        case_index
                    ],
                )


                force_errors.append(
                    100.0
                    *
                    force_error
                )


                generalized_force_errors.append(
                    100.0
                    *
                    generalized_force_error
                )


            force_errors = np.asarray(
                force_errors
            )


            generalized_force_errors = np.asarray(
                generalized_force_errors
            )


            audit_rows.append(
                {

                    "Segment":
                        segment,

                    "Interface":
                        interface_name,

                    "Modes":
                        MODES,

                    "Split":
                        split_name,

                    "ForceMean_percent":
                        float(
                            np.mean(
                                force_errors
                            )
                        ),

                    "ForceMax_percent":
                        float(
                            np.max(
                                force_errors
                            )
                        ),

                    "GeneralizedForceMean_percent":
                        float(
                            np.mean(
                                generalized_force_errors
                            )
                        ),

                    "GeneralizedForceMax_percent":
                        float(
                            np.max(
                                generalized_force_errors
                            )
                        ),
                }
            )


        # ====================================================
        # SAVE FINAL PCA BASIS
        #
        # These are the files later used by the new operators.
        # ====================================================

        final_pca_file = os.path.join(
            FINAL_DIR,
            "force_pca_{}.npz".format(
                interface_name
            ),
        )


        np.savez(

            final_pca_file,

            mean=
                mean.astype(
                    np.float32
                ),

            basis=
                basis.astype(
                    np.float32
                ),

            coordinates=
                coordinates.astype(
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
                    [
                        MODES
                    ],
                    dtype=np.int64,
                ),

            retained_energy=
                pca[
                    "retained_energy"
                ],
        )


        print(
            "Saved final PCA:"
        )

        print(
            final_pca_file
        )


    # ========================================================
    # COMBINE BOTH INTERFACE FORCE COEFFICIENTS
    #
    # 8 + 8 = 16 outputs per NO block
    # ========================================================

    combined_coefficients = np.concatenate(

        segment_coefficients,

        axis=1,
    )


    expected_dimension = (
        MODES
        *
        2
    )


    if combined_coefficients.shape[
        1
    ] != expected_dimension:

        raise RuntimeError(

            "{} combined force dimension should be {}, "
            "but found {}."
            .format(

                segment,

                expected_dimension,

                combined_coefficients.shape[
                    1
                ],
            )
        )


    coefficient_output_file = os.path.join(
        FINAL_DIR,
        "{}_force_coefficients.npy".format(
            segment
        ),
    )


    np.save(
        coefficient_output_file,
        combined_coefficients.astype(
            np.float32
        ),
    )


    print("")
    print(
        "{} combined target shape:"
        .format(
            segment
        ),
        combined_coefficients.shape,
    )


    print(
        "Saved:"
    )

    print(
        coefficient_output_file
    )


# ============================================================
# SAVE FINAL AUDIT
# ============================================================

audit_dataframe = pd.DataFrame(
    audit_rows
)


audit_file = os.path.join(
    FINAL_DIR,
    "final_force_pca_audit.csv",
)


audit_dataframe.to_csv(
    audit_file,
    index=False,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "SelectedModesPerInterface":
        MODES,

    "ForceOutputsPerLocalOperator":
        MODES
        *
        2,

    "PCAFitCases":
        96,

    "ValidationCases":
        12,

    "TestCases":
        12,

    "ModeSelectionUsedTestData":
        False,

    "Description":
        (
            "Final force PCA rebuilt using the enlarged "
            "96-case training set."
        ),
}


metadata_file = os.path.join(
    FINAL_DIR,
    "final_force_pca_metadata.json",
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


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("")
print(
    "========================================"
)

print(
    "STEP 52 COMPLETE"
)

print(
    "========================================"
)


print("")
print(
    "Final force PCA:"
)


print(
    "{} modes/interface".format(
        MODES
    )
)


print(
    "{} force outputs/local NO".format(
        MODES
        *
        2
    )
)


print("")
print(
    audit_dataframe.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)


print(
    audit_file
)


print(
    metadata_file
)