import os
import json

import numpy as np
import pandas as pd

import torch


from src.separate_vector_deeponet import (
    SeparateVectorDeepONet
)


from src.orthotropic_mechanics import (
    displacement_and_gradient,
    infinitesimal_strain,
    left_hencky_strain,
    strain_tensor_to_abaqus_voigt,
    stress_from_small_strain,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"


MODEL_DIR = os.path.join(
    "results",
    "vector_deeponet_separate",
)


RESULTS_DIR = os.path.join(
    "results",
    "mechanics_validation",
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# CPU FOR DERIVATIVE VALIDATION
# ============================================================

DEVICE = torch.device(
    "cpu"
)


print("")
print(
    "Mechanics validation device:",
    DEVICE,
)


# ============================================================
# LOAD DATA
# ============================================================

parameters = np.load(
    os.path.join(
        DATA_DIR,
        "parameters.npy",
    )
).astype(
    np.float32
)


integration_coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "ip_coordinates.npy",
    )
).astype(
    np.float32
)


stress_fem = np.load(
    os.path.join(
        DATA_DIR,
        "S_tensor.npy",
    )
).astype(
    np.float32
)


strain_fem = np.load(
    os.path.join(
        DATA_DIR,
        "LE_tensor.npy",
    )
).astype(
    np.float32
)


split_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv",
    )
)


case_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "case_ids.csv",
    )
)


test_indices = split_table.loc[
    split_table["Split"] == "test",
    "Index",
].to_numpy(
    dtype=int
)


print("")
print(
    "Parameter shape:",
    parameters.shape,
)


print(
    "Integration coordinates:",
    integration_coordinates.shape,
)


print(
    "Abaqus stress tensor:",
    stress_fem.shape,
)


print(
    "Abaqus LE tensor:",
    strain_fem.shape,
)


print(
    "Test cases:",
    len(
        test_indices
    ),
)


# ============================================================
# CHECK TENSOR SHAPES
# ============================================================

if stress_fem.shape[0] != parameters.shape[0]:

    raise RuntimeError(
        "Stress case count does not "
        "match parameter case count."
    )


if strain_fem.shape[0] != parameters.shape[0]:

    raise RuntimeError(
        "Strain case count does not "
        "match parameter case count."
    )


if stress_fem.shape[2] != 6:

    raise RuntimeError(
        "S_tensor must contain "
        "6 components."
    )


if strain_fem.shape[2] != 6:

    raise RuntimeError(
        "LE_tensor must contain "
        "6 components."
    )


# ============================================================
# NORMALIZATION
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "normalization.json",
    ),
    "r",
) as f:

    normalization = json.load(
        f
    )


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "best_separate_vector_deeponet.pt",
    ),
    map_location=DEVICE,
)


model = SeparateVectorDeepONet(
    branch_dim=3,
    trunk_dim=3,
    latent_dim=checkpoint.get(
        "latent_dim",
        128,
    ),
).to(
    DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print("")
print(
    "Loaded model from epoch:",
    checkpoint["epoch"],
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def field_metrics(
    prediction,
    truth,
):

    error = (
        prediction
        - truth
    )

    truth_norm = np.linalg.norm(
        truth
    )

    error_norm = np.linalg.norm(
        error
    )

    relative_l2 = (
        error_norm
        / (
            truth_norm
            + 1.0e-14
        )
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    mae = np.mean(
        np.abs(
            error
        )
    )

    max_abs_error = np.max(
        np.abs(
            error
        )
    )

    return (
        relative_l2,
        rmse,
        mae,
        max_abs_error,
    )


# ============================================================
# COMPONENT ORDER
#
# Matches the extraction script:
#
# 0 = 11
# 1 = 22
# 2 = 33
# 3 = 12
# 4 = 13
# 5 = 23
# ============================================================

component_names = [
    "11",
    "22",
    "33",
    "12",
    "13",
    "23",
]


rows = []


coordinates_tensor = torch.tensor(
    integration_coordinates,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# LOOP OVER UNSEEN TEST CASES
# ============================================================

for global_index in test_indices:

    case_id = case_table.loc[
        case_table["Index"]
        == global_index,
        "CaseID",
    ].iloc[0]


    parameter_tensor = torch.tensor(
        parameters[
            global_index
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


    print("")
    print(
        "Processing:",
        case_id,
    )


    # ========================================================
    # DEEPONET DISPLACEMENT + GRADIENT
    # ========================================================

    (
        displacement_prediction,
        displacement_gradient,
    ) = displacement_and_gradient(
        model,
        parameter_tensor,
        coordinates_tensor,
        normalization,
        create_graph=False,
    )


    # ========================================================
    # INFINITESIMAL STRAIN
    # ========================================================

    small_strain_tensor = (
        infinitesimal_strain(
            displacement_gradient
        )
    )


    # --------------------------------------------------------
    # Convert shear components to Abaqus engineering-shear
    # convention before comparing with LE.
    # --------------------------------------------------------

    small_strain_abaqus = (
        strain_tensor_to_abaqus_voigt(
            small_strain_tensor
        )
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # LEFT HENCKY / LOG STRAIN
    # ========================================================

    logarithmic_strain_tensor = (
        left_hencky_strain(
            displacement_gradient
        )
    )


    logarithmic_strain_abaqus = (
        strain_tensor_to_abaqus_voigt(
            logarithmic_strain_tensor
        )
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # STRESS FROM SMALL-STRAIN ORTHOTROPIC CONSTITUTIVE MODEL
    # ========================================================

    predicted_stress = (
        stress_from_small_strain(
            small_strain_tensor,
            parameter_tensor,
        )
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # SAVE DERIVED FIELDS FOR THIS TEST CASE
    # ========================================================

    case_result_directory = os.path.join(
        RESULTS_DIR,
        case_id,
    )


    os.makedirs(
        case_result_directory,
        exist_ok=True,
    )


    np.save(
        os.path.join(
            case_result_directory,
            "small_strain.npy",
        ),
        small_strain_abaqus,
    )


    np.save(
        os.path.join(
            case_result_directory,
            "log_strain.npy",
        ),
        logarithmic_strain_abaqus,
    )


    np.save(
        os.path.join(
            case_result_directory,
            "predicted_stress.npy",
        ),
        predicted_stress,
    )


    # ========================================================
    # STRAIN COMPARISON
    # ========================================================

    for (
        component_index,
        component_name,
    ) in enumerate(
        component_names
    ):

        # ----------------------------------------------------
        # Small strain vs Abaqus LE
        # ----------------------------------------------------

        (
            relative_l2,
            rmse,
            mae,
            max_abs_error,
        ) = field_metrics(

            small_strain_abaqus[
                :,
                component_index,
            ],

            strain_fem[
                global_index,
                :,
                component_index,
            ],

        )


        rows.append(
            {
                "CaseID":
                    case_id,

                "Quantity":
                    "LE"
                    + component_name,

                "Method":
                    "Infinitesimal",

                "Relative_L2":
                    relative_l2,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "MaxAbsError":
                    max_abs_error,
            }
        )


        # ----------------------------------------------------
        # Log strain vs Abaqus LE
        # ----------------------------------------------------

        (
            relative_l2,
            rmse,
            mae,
            max_abs_error,
        ) = field_metrics(

            logarithmic_strain_abaqus[
                :,
                component_index,
            ],

            strain_fem[
                global_index,
                :,
                component_index,
            ],

        )


        rows.append(
            {
                "CaseID":
                    case_id,

                "Quantity":
                    "LE"
                    + component_name,

                "Method":
                    "LeftHencky",

                "Relative_L2":
                    relative_l2,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "MaxAbsError":
                    max_abs_error,
            }
        )


    # ========================================================
    # STRESS COMPARISON
    # ========================================================

    for (
        component_index,
        component_name,
    ) in enumerate(
        component_names
    ):

        (
            relative_l2,
            rmse,
            mae,
            max_abs_error,
        ) = field_metrics(

            predicted_stress[
                :,
                component_index,
            ],

            stress_fem[
                global_index,
                :,
                component_index,
            ],

        )


        rows.append(
            {
                "CaseID":
                    case_id,

                "Quantity":
                    "S"
                    + component_name,

                "Method":
                    "OrthotropicSmallStrain",

                "Relative_L2":
                    relative_l2,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "MaxAbsError":
                    max_abs_error,
            }
        )


    print(
        "Finished:",
        case_id,
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    rows
)


metrics_file = os.path.join(
    RESULTS_DIR,
    "mechanics_metrics.csv",
)


metrics.to_csv(
    metrics_file,
    index=False,
)


# ============================================================
# MEAN SUMMARY
# ============================================================

summary = (
    metrics.groupby(
        [
            "Quantity",
            "Method",
        ]
    )[
        [
            "Relative_L2",
            "RMSE",
            "MAE",
            "MaxAbsError",
        ]
    ]
    .mean()
    .reset_index()
)


summary_file = os.path.join(
    RESULTS_DIR,
    "mechanics_summary.csv",
)


summary.to_csv(
    summary_file,
    index=False,
)


print("")
print(
    "============================================"
)

print(
    "MECHANICS VALIDATION COMPLETE"
)

print(
    "============================================"
)


print("")
print(
    summary.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    metrics_file
)

print(
    summary_file
)


print("")
print(
    "DO NOT START PI TRAINING YET."
)


print(
    "First compare Infinitesimal and LeftHencky "
    "against Abaqus LE."
)