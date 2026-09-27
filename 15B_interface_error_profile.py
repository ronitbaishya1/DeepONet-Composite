import os
import json

import numpy as np
import pandas as pd

import torch

from src.separate_vector_deeponet import (
    SeparateVectorDeepONet
)

from src.mechanics_losses import (
    derived_mechanics_single,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data"

MODEL_DIR = os.path.join(
    "results",
    "mechanics_consistent",
    "eps_0.1_sig_0.1_frac_1",
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_mechanics_consistent.pt",
)

OUTPUT_DIR = os.path.join(
    "results",
    "hybrid_interface_diagnostic",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


DEVICE = torch.device(
    "cpu"
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


ip_coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "ip_coordinates.npy",
    )
).astype(
    np.float32
)


LE_tensor = np.load(
    os.path.join(
        DATA_DIR,
        "LE_tensor.npy",
    )
).astype(
    np.float32
)


S_tensor = np.load(
    os.path.join(
        DATA_DIR,
        "S_tensor.npy",
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


# ============================================================
# USE VALIDATION CASES ONLY FOR INTERFACE SELECTION
# ============================================================

validation_indices = split_table.loc[
    split_table[
        "Split"
    ] == "validation",
    "Index",
].to_numpy(
    dtype=int
)


print(
    "Validation cases:",
    validation_indices
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
# MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
)


model = SeparateVectorDeepONet(
    branch_dim=checkpoint.get(
        "branch_dim",
        3,
    ),
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


# ============================================================
# IP COORDINATES
# ============================================================

ip_coordinates_tensor = torch.tensor(
    ip_coordinates,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# GLOBAL NORMALIZATION SCALES
#
# We normalize by the overall RMS magnitude rather than
# section-by-section magnitude. This avoids artificial huge
# relative errors in sections where the true field is near zero.
# ============================================================

validation_LE = LE_tensor[
    validation_indices
]


validation_S = S_tensor[
    validation_indices
]


scale_LE11 = np.sqrt(
    np.mean(
        validation_LE[
            :,
            :,
            0
        ] ** 2
    )
)


scale_LE12 = np.sqrt(
    np.mean(
        validation_LE[
            :,
            :,
            3
        ] ** 2
    )
)


scale_S11 = np.sqrt(
    np.mean(
        validation_S[
            :,
            :,
            0
        ] ** 2
    )
)


scale_S12 = np.sqrt(
    np.mean(
        validation_S[
            :,
            :,
            3
        ] ** 2
    )
)


print("")
print(
    "Global validation scales:"
)

print(
    "LE11:",
    scale_LE11
)

print(
    "LE12:",
    scale_LE12
)

print(
    "S11:",
    scale_S11
)

print(
    "S12:",
    scale_S12
)


# ============================================================
# STORAGE FOR PREDICTIONS
# ============================================================

number_validation = len(
    validation_indices
)


number_ips = len(
    ip_coordinates
)


predicted_LE = np.zeros(
    (
        number_validation,
        number_ips,
        6,
    ),
    dtype=np.float32,
)


predicted_S = np.zeros(
    (
        number_validation,
        number_ips,
        6,
    ),
    dtype=np.float32,
)


# ============================================================
# PREDICT VALIDATION CASES
# ============================================================

for local_index, global_index in enumerate(
    validation_indices
):

    print(
        "Predicting validation case:",
        global_index
    )


    parameter_physical = torch.tensor(
        parameters[
            global_index
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


    with torch.enable_grad():

        (
            _,
            strain_prediction,
            stress_prediction,
        ) = derived_mechanics_single(
            model=model,

            parameter_physical=(
                parameter_physical
            ),

            coordinates_physical=(
                ip_coordinates_tensor
            ),

            normalization=(
                normalization
            ),

            create_graph=False,
        )


    predicted_LE[
        local_index
    ] = (
        strain_prediction
        .detach()
        .cpu()
        .numpy()
    )


    predicted_S[
        local_index
    ] = (
        stress_prediction
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# UNIQUE X SECTIONS
# ============================================================

x_coordinates = ip_coordinates[
    :,
    0
]


unique_x = np.unique(
    np.round(
        x_coordinates,
        decimals=8,
    )
)


rows = []


# ============================================================
# SECTION-BY-SECTION ERRORS
# ============================================================

for x_value in unique_x:

    section_mask = np.isclose(
        x_coordinates,
        x_value,
        atol=1.0e-7,
    )


    # --------------------------------------------------------
    # TRUE
    # --------------------------------------------------------

    true_LE11 = validation_LE[
        :,
        section_mask,
        0
    ]


    true_LE12 = validation_LE[
        :,
        section_mask,
        3
    ]


    true_S11 = validation_S[
        :,
        section_mask,
        0
    ]


    true_S12 = validation_S[
        :,
        section_mask,
        3
    ]


    # --------------------------------------------------------
    # PREDICTED
    # --------------------------------------------------------

    pred_LE11 = predicted_LE[
        :,
        section_mask,
        0
    ]


    pred_LE12 = predicted_LE[
        :,
        section_mask,
        3
    ]


    pred_S11 = predicted_S[
        :,
        section_mask,
        0
    ]


    pred_S12 = predicted_S[
        :,
        section_mask,
        3
    ]


    # --------------------------------------------------------
    # NORMALIZED RMSE
    # --------------------------------------------------------

    error_LE11 = (
        np.sqrt(
            np.mean(
                (
                    pred_LE11
                    - true_LE11
                ) ** 2
            )
        )
        /
        (
            scale_LE11
            + 1.0e-14
        )
    )


    error_LE12 = (
        np.sqrt(
            np.mean(
                (
                    pred_LE12
                    - true_LE12
                ) ** 2
            )
        )
        /
        (
            scale_LE12
            + 1.0e-14
        )
    )


    error_S11 = (
        np.sqrt(
            np.mean(
                (
                    pred_S11
                    - true_S11
                ) ** 2
            )
        )
        /
        (
            scale_S11
            + 1.0e-14
        )
    )


    error_S12 = (
        np.sqrt(
            np.mean(
                (
                    pred_S12
                    - true_S12
                ) ** 2
            )
        )
        /
        (
            scale_S12
            + 1.0e-14
        )
    )


    combined = np.mean(
        [
            error_LE11,
            error_LE12,
            error_S11,
            error_S12,
        ]
    )


    rows.append(
        {
            "X_mm":
                float(
                    x_value
                ),

            "LE11_NRMSE":
                error_LE11,

            "LE12_NRMSE":
                error_LE12,

            "S11_NRMSE":
                error_S11,

            "S12_NRMSE":
                error_S12,

            "CombinedMechanicsError":
                combined,
        }
    )


# ============================================================
# SAVE
# ============================================================

profile = pd.DataFrame(
    rows
)


profile = profile.sort_values(
    "X_mm"
)


output_file = os.path.join(
    OUTPUT_DIR,
    "mechanics_error_vs_x.csv",
)


profile.to_csv(
    output_file,
    index=False,
)


print("")
print(
    "============================================"
)

print(
    "MECHANICS ERROR BY X SECTION"
)

print(
    "============================================"
)

print(
    profile.to_string(
        index=False
    )
)


print("")
print(
    "Values near proposed interfaces:"
)


for requested in [
    -6.0,
    -1.8,
    1.8,
    6.0,
]:

    index = (
        profile[
            "X_mm"
        ]
        .sub(
            requested
        )
        .abs()
        .idxmin()
    )


    print("")
    print(
        profile.loc[
            index
        ].to_string()
    )


print("")
print(
    "Saved:"
)

print(
    output_file
)