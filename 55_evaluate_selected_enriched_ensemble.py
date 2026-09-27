import os
import json

import numpy as np
import pandas as pd

import torch

from src.hybrid_bulk_operator_v3 import (
    HybridBulkOperatorV3
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = os.getcwd()


RESULTS_ROOT = os.path.join(
    ROOT,
    "results",
    "rebuilt_force_pca",
)


PCA_DIR = os.path.join(
    ROOT,
    "data",
    "hybrid_force_pca_enriched",
    "final",
)


SELECTION_FILE = os.path.join(
    RESULTS_ROOT,
    "selected_strategies.json",
)


with open(
    SELECTION_FILE,
    "r",
) as file_object:

    selected_strategies = json.load(
        file_object
    )


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# RELATIVE L2
# ============================================================

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
# OUTPUT CONTAINERS
# ============================================================

summary_rows = []

all_case_rows = []

final_model_selection = {}


# ============================================================
# LEFT + RIGHT
# ============================================================

for segment in [
    "left",
    "right",
]:

    print("")
    print(
        "================================================"
    )

    print(
        "FINAL TEST:",
        segment.upper()
    )

    print(
        "================================================"
    )


    # ========================================================
    # DATASET CONFIGURATION
    # ========================================================

    if segment == "left":

        data_dir = os.path.join(
            ROOT,
            "data",
            "hybrid_bulk_left_adaptive120",
        )


        interface_1 = "m6"

        interface_2 = "m2"


    else:

        data_dir = os.path.join(
            ROOT,
            "data",
            "hybrid_bulk_right_adaptive120",
        )


        interface_1 = "p2"

        interface_2 = "p6"


    selected_strategy = (
        selected_strategies[
            segment
        ][
            "SelectedStrategy"
        ]
    )


    ensemble_directory = (
        selected_strategies[
            segment
        ][
            "SelectedDirectory"
        ]
    )


    print(
        "Selected strategy:",
        selected_strategy
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    branch = np.load(
        os.path.join(
            data_dir,
            "branch_inputs.npy",
        )
    ).astype(
        np.float32
    )


    coordinates = np.load(
        os.path.join(
            data_dir,
            "coordinates.npy",
        )
    ).astype(
        np.float32
    )


    U = np.stack(
        [

            np.load(
                os.path.join(
                    data_dir,
                    "U1.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
                    "U2.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
                    "U3.npy",
                )
            ),
        ],

        axis=-1,

    ).astype(
        np.float32
    )


    g = np.concatenate(
        [

            np.load(
                os.path.join(
                    data_dir,
                    "g_left.npy",
                )
            ),

            np.load(
                os.path.join(
                    data_dir,
                    "g_right.npy",
                )
            ),
        ],

        axis=1,

    ).astype(
        np.float32
    )


    force_coefficients = np.load(
        os.path.join(
            PCA_DIR,
            "{}_force_coefficients.npy".format(
                segment
            ),
        )
    ).astype(
        np.float32
    )


    split_dataframe = pd.read_csv(
        os.path.join(
            data_dir,
            "split_assignment.csv",
        )
    )


    # ========================================================
    # TEST SET
    # ========================================================

    if "Split" in split_dataframe.columns:

        split_column = "Split"

    elif "split" in split_dataframe.columns:

        split_column = "split"

    else:

        raise RuntimeError(
            "Could not find split column."
        )


    test_mask = (

        split_dataframe[
            split_column
        ]
        .astype(str)
        .str.lower()

        ==

        "test"
    )


    if "Index" in split_dataframe.columns:

        test_indices = split_dataframe.loc[
            test_mask,
            "Index",
        ].to_numpy(
            dtype=int
        )


    else:

        test_indices = np.where(
            test_mask.to_numpy()
        )[0]


    if "CaseID" in split_dataframe.columns:

        case_ids = (
            split_dataframe[
                "CaseID"
            ]
            .astype(str)
            .to_numpy()
        )


    else:

        case_ids = np.asarray(
            [
                "Case_{:03d}".format(
                    index
                )
                for index in range(
                    len(
                        branch
                    )
                )
            ]
        )


    print(
        "Test cases:",
        len(
            test_indices
        )
    )


    if len(
        test_indices
    ) != 12:

        raise RuntimeError(
            "Expected 12 untouched test cases."
        )


    # ========================================================
    # FINAL FORCE PCA
    # ========================================================

    pca_1 = np.load(
        os.path.join(
            PCA_DIR,
            "force_pca_{}.npz".format(
                interface_1
            ),
        )
    )


    pca_2 = np.load(
        os.path.join(
            PCA_DIR,
            "force_pca_{}.npz".format(
                interface_2
            ),
        )
    )


    modes_1 = pca_1[
        "basis"
    ].shape[
        1
    ]


    modes_2 = pca_2[
        "basis"
    ].shape[
        1
    ]


    if modes_1 != modes_2:

        raise RuntimeError(
            "Interface force-mode counts do not match."
        )


    print(
        "Force modes/interface:",
        modes_1
    )


    # ========================================================
    # FIND FIVE MEMBER CHECKPOINTS
    # ========================================================

    member_directories = sorted(
        [

            name

            for name in os.listdir(
                ensemble_directory
            )

            if name.startswith(
                "member_"
            )
        ]
    )


    if len(
        member_directories
    ) != 5:

        raise RuntimeError(
            "Expected 5 ensemble members, found {}."
            .format(
                len(
                    member_directories
                )
            )
        )


    # ========================================================
    # MEMBER PREDICTIONS
    # ========================================================

    all_member_U = []

    all_member_g = []

    all_member_coefficients = []


    for member_directory in member_directories:

        checkpoint_file = os.path.join(
            ensemble_directory,
            member_directory,
            "best_enriched_forcepca.pt",
        )


        if not os.path.isfile(
            checkpoint_file
        ):

            raise FileNotFoundError(
                checkpoint_file
            )


        checkpoint = torch.load(

            checkpoint_file,

            map_location=DEVICE,

            weights_only=False,
        )


        # ====================================================
        # MODEL
        # ====================================================

        model = HybridBulkOperatorV3(

            branch_dim=
                checkpoint[
                    "branch_dim"
                ],

            number_force_coefficients=
                checkpoint[
                    "number_force_coefficients"
                ],

            hidden_dim=
                checkpoint[
                    "hidden_dim"
                ],

            latent_dim=
                checkpoint[
                    "latent_dim"
                ],

            depth=
                checkpoint[
                    "depth"
                ],
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.eval()


        # ====================================================
        # NORMALIZATION
        # ====================================================

        branch_mean = np.asarray(
            checkpoint[
                "branch_mean"
            ],
            dtype=np.float32,
        )


        branch_std = np.asarray(
            checkpoint[
                "branch_std"
            ],
            dtype=np.float32,
        )


        coordinate_min = np.asarray(
            checkpoint[
                "coordinate_min"
            ],
            dtype=np.float32,
        )


        coordinate_max = np.asarray(
            checkpoint[
                "coordinate_max"
            ],
            dtype=np.float32,
        )


        U_mean = np.asarray(
            checkpoint[
                "U_mean"
            ],
            dtype=np.float32,
        )


        U_std = np.asarray(
            checkpoint[
                "U_std"
            ],
            dtype=np.float32,
        )


        force_mean = np.asarray(
            checkpoint[
                "force_coeff_mean"
            ],
            dtype=np.float32,
        )


        force_std = np.asarray(
            checkpoint[
                "force_coeff_std"
            ],
            dtype=np.float32,
        )


        # ====================================================
        # COORDINATE NORMALIZATION
        # ====================================================

        coordinate_range = np.maximum(

            coordinate_max
            -
            coordinate_min,

            1.0e-8,
        )


        coordinate_normalized = (

            2.0

            *

            (
                coordinates
                -
                coordinate_min
            )

            /

            coordinate_range

            -

            1.0
        )


        coordinate_tensor = torch.tensor(

            coordinate_normalized,

            dtype=torch.float32,
        )


        member_U = []

        member_g = []

        member_coefficients = []


        # ====================================================
        # PREDICT 12 TEST CASES
        # ====================================================

        with torch.no_grad():

            for case_index in test_indices:

                normalized_branch = (

                    branch[
                        case_index
                    ]

                    -

                    branch_mean

                ) / branch_std


                branch_tensor = torch.tensor(

                    normalized_branch,

                    dtype=torch.float32,

                ).reshape(
                    1,
                    -1,
                )


                (
                    U_prediction_normalized,
                    coefficient_prediction_normalized,
                ) = model(

                    branch_tensor,

                    coordinate_tensor,
                )


                # --------------------------------------------
                # DISPLACEMENT BACK TO PHYSICAL UNITS
                # --------------------------------------------

                U_prediction = (

                    U_prediction_normalized[
                        0
                    ]
                    .cpu()
                    .numpy()

                    *

                    U_std.reshape(
                        1,
                        3,
                    )

                    +

                    U_mean.reshape(
                        1,
                        3,
                    )
                )


                # --------------------------------------------
                # FORCE PCA COEFFICIENTS BACK TO PHYSICAL
                # --------------------------------------------

                coefficients = (

                    coefficient_prediction_normalized[
                        0
                    ]
                    .cpu()
                    .numpy()

                    *

                    force_std

                    +

                    force_mean
                )


                c1 = coefficients[
                    :modes_1
                ]


                c2 = coefficients[
                    modes_1:
                    modes_1
                    +
                    modes_2
                ]


                # --------------------------------------------
                # GENERALIZED FORCE
                # --------------------------------------------

                g1 = (

                    pca_1[
                        "g_mean"
                    ]

                    +

                    pca_1[
                        "g_matrix"
                    ]

                    @

                    c1
                )


                g2 = (

                    pca_2[
                        "g_mean"
                    ]

                    +

                    pca_2[
                        "g_matrix"
                    ]

                    @

                    c2
                )


                g_prediction = np.concatenate(
                    [
                        g1,
                        g2,
                    ]
                )


                member_U.append(
                    U_prediction
                )


                member_g.append(
                    g_prediction
                )


                member_coefficients.append(
                    coefficients
                )


        all_member_U.append(
            np.asarray(
                member_U
            )
        )


        all_member_g.append(
            np.asarray(
                member_g
            )
        )


        all_member_coefficients.append(
            np.asarray(
                member_coefficients
            )
        )


    # ========================================================
    # ENSEMBLE MEAN
    # ========================================================

    all_member_U = np.asarray(
        all_member_U
    )


    all_member_g = np.asarray(
        all_member_g
    )


    all_member_coefficients = np.asarray(
        all_member_coefficients
    )


    ensemble_U = np.mean(
        all_member_U,
        axis=0,
    )


    ensemble_g = np.mean(
        all_member_g,
        axis=0,
    )


    ensemble_coefficients = np.mean(
        all_member_coefficients,
        axis=0,
    )


    # ========================================================
    # CASE-BY-CASE ERRORS
    # ========================================================

    U1_errors = []

    U2_errors = []

    U3_errors = []

    g_errors = []

    coefficient_errors = []


    for local_index, case_index in enumerate(
        test_indices
    ):

        case_U1 = (
            100.0
            *
            relative_l2(

                ensemble_U[
                    local_index,
                    :,
                    0
                ],

                U[
                    case_index,
                    :,
                    0
                ],
            )
        )


        case_U2 = (
            100.0
            *
            relative_l2(

                ensemble_U[
                    local_index,
                    :,
                    1
                ],

                U[
                    case_index,
                    :,
                    1
                ],
            )
        )


        case_U3 = (
            100.0
            *
            relative_l2(

                ensemble_U[
                    local_index,
                    :,
                    2
                ],

                U[
                    case_index,
                    :,
                    2
                ],
            )
        )


        case_g = (
            100.0
            *
            relative_l2(

                ensemble_g[
                    local_index
                ],

                g[
                    case_index
                ],
            )
        )


        case_coeff = (
            100.0
            *
            relative_l2(

                ensemble_coefficients[
                    local_index
                ],

                force_coefficients[
                    case_index
                ],
            )
        )


        U1_errors.append(
            case_U1
        )


        U2_errors.append(
            case_U2
        )


        U3_errors.append(
            case_U3
        )


        g_errors.append(
            case_g
        )


        coefficient_errors.append(
            case_coeff
        )


        all_case_rows.append(
            {

                "Segment":
                    segment,

                "SelectedStrategy":
                    selected_strategy,

                "CaseIndex":
                    int(
                        case_index
                    ),

                "CaseID":
                    str(
                        case_ids[
                            case_index
                        ]
                    ),

                "U1_RelL2_percent":
                    case_U1,

                "U2_RelL2_percent":
                    case_U2,

                "U3_RelL2_percent":
                    case_U3,

                "GeneralizedForce_RelL2_percent":
                    case_g,

                "ForceCoefficient_RelL2_percent":
                    case_coeff,
            }
        )


    # ========================================================
    # NEW ENRICHED TEST RESULTS
    # ========================================================

    new_U1 = float(
        np.mean(
            U1_errors
        )
    )


    new_U2 = float(
        np.mean(
            U2_errors
        )
    )


    new_U3 = float(
        np.mean(
            U3_errors
        )
    )


    new_g = float(
        np.mean(
            g_errors
        )
    )


    new_coefficients = float(
        np.mean(
            coefficient_errors
        )
    )


    # ========================================================
    # ORIGINAL STEP-40 ENSEMBLE RESULT
    # ========================================================

    old_summary_file = os.path.join(
        ROOT,
        "results",
        "adaptive_ensemble",
        segment,
        "ensemble_summary.json",
    )


    with open(
        old_summary_file,
        "r",
    ) as file_object:

        old_summary = json.load(
            file_object
        )


    old_U1 = float(
        old_summary[
            "EnsembleMean_U1_percent"
        ]
    )


    old_U2 = float(
        old_summary[
            "EnsembleMean_U2_percent"
        ]
    )


    old_U3 = float(
        old_summary[
            "EnsembleMean_U3_percent"
        ]
    )


    # ========================================================
    # IMPROVEMENT
    # ========================================================

    U2_absolute_change = (
        old_U2
        -
        new_U2
    )


    U2_relative_improvement = (

        100.0

        *

        U2_absolute_change

        /

        old_U2
    )


    # ========================================================
    # MODEL DECISION
    #
    # The enriched model must improve U2.
    #
    # We do not automatically use it merely because it has
    # more FEM cases.
    # ========================================================

    enriched_wins = (
        new_U2
        <
        old_U2
    )


    if enriched_wins:

        final_choice = (
            "enriched_{}".format(
                selected_strategy
            )
        )

    else:

        final_choice = "original_step40"


    # ========================================================
    # SAVE SUMMARY ROW
    # ========================================================

    summary_rows.append(
        {

            "Segment":
                segment,

            "SelectedEnrichmentStrategy":
                selected_strategy,

            "ForceModesPerInterface":
                int(
                    modes_1
                ),

            "Old_U1_percent":
                old_U1,

            "New_U1_percent":
                new_U1,

            "Old_U2_percent":
                old_U2,

            "New_U2_percent":
                new_U2,

            "U2_AbsoluteImprovement_percentage_points":
                U2_absolute_change,

            "U2_RelativeImprovement_percent":
                U2_relative_improvement,

            "Old_U3_percent":
                old_U3,

            "New_U3_percent":
                new_U3,

            "New_GeneralizedForce_percent":
                new_g,

            "New_ForceCoefficient_percent":
                new_coefficients,

            "EnrichedBeatsOriginal_U2":
                bool(
                    enriched_wins
                ),

            "FinalChoice":
                final_choice,
        }
    )


    # ========================================================
    # JSON MODEL SELECTION
    # ========================================================

    final_model_selection[
        segment
    ] = {

        "SelectedEnrichmentStrategy":
            selected_strategy,

        "OldStep40_U2_percent":
            old_U2,

        "Enriched_U2_percent":
            new_U2,

        "Enriched_GeneralizedForce_percent":
            new_g,

        "U2_RelativeImprovement_percent":
            U2_relative_improvement,

        "UseEnrichedModel":
            bool(
                enriched_wins
            ),

        "FinalChoice":
            final_choice,

        "SelectedEnrichedDirectory":
            ensemble_directory,
    }


    # ========================================================
    # PRINT SIDE RESULT
    # ========================================================

    print("")
    print(
        "Old Step-40 U2: {:.6f}%".format(
            old_U2
        )
    )


    print(
        "New enriched U2: {:.6f}%".format(
            new_U2
        )
    )


    print(
        "New generalized-force error: {:.6f}%"
        .format(
            new_g
        )
    )


    print(
        "U2 relative improvement: {:.2f}%"
        .format(
            U2_relative_improvement
        )
    )


    print(
        "Final choice:",
        final_choice
    )


# ============================================================
# SAVE CASE-BY-CASE RESULTS
# ============================================================

case_dataframe = pd.DataFrame(
    all_case_rows
)


case_file = os.path.join(
    RESULTS_ROOT,
    "final_test_case_metrics.csv",
)


case_dataframe.to_csv(
    case_file,
    index=False,
)


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary_dataframe = pd.DataFrame(
    summary_rows
)


summary_file = os.path.join(
    RESULTS_ROOT,
    "final_test_comparison.csv",
)


summary_dataframe.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# SAVE FINAL MODEL DECISIONS
# ============================================================

selection_file = os.path.join(
    RESULTS_ROOT,
    "final_selected_models.json",
)


with open(
    selection_file,
    "w",
) as file_object:

    json.dump(
        final_model_selection,
        file_object,
        indent=4,
    )


# ============================================================
# FINAL PRINT
# ============================================================

print("")
print(
    "================================================"
)

print(
    "STEP 55 COMPLETE"
)

print(
    "================================================"
)


print("")
print(
    summary_dataframe.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    case_file
)


print(
    summary_file
)


print(
    selection_file
)