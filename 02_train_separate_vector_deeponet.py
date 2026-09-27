import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.separate_vector_deeponet import (
    SeparateVectorDeepONet
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = "data"

RESULTS_DIR = os.path.join(
    "results",
    "vector_deeponet_separate",
)

LATENT_DIM = 128

BATCH_SIZE = 5

LEARNING_RATE = 1.0e-3

WEIGHT_DECAY = 1.0e-6

MAX_EPOCHS = 6000

PATIENCE = 600


random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("")
print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

parameters = np.load(
    os.path.join(
        DATA_DIR,
        "parameters.npy",
    )
).astype(np.float32)


coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "coordinates.npy",
    )
).astype(np.float32)


U1 = np.load(
    os.path.join(
        DATA_DIR,
        "U1.npy",
    )
).astype(np.float32)


U2 = np.load(
    os.path.join(
        DATA_DIR,
        "U2.npy",
    )
).astype(np.float32)


U3 = np.load(
    os.path.join(
        DATA_DIR,
        "U3.npy",
    )
).astype(np.float32)


U = np.stack(
    [U1, U2, U3],
    axis=-1,
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


train_indices = split_table.loc[
    split_table["Split"] == "train",
    "Index",
].to_numpy(
    dtype=int
)


val_indices = split_table.loc[
    split_table["Split"] == "validation",
    "Index",
].to_numpy(
    dtype=int
)


test_indices = split_table.loc[
    split_table["Split"] == "test",
    "Index",
].to_numpy(
    dtype=int
)


print(
    "parameters:",
    parameters.shape,
)

print(
    "coordinates:",
    coordinates.shape,
)

print(
    "U:",
    U.shape,
)


# ============================================================
# NORMALIZE BRANCH INPUTS
# TRAINING CASES ONLY
# ============================================================

parameter_mean = parameters[
    train_indices
].mean(
    axis=0,
    keepdims=True,
)


parameter_std = parameters[
    train_indices
].std(
    axis=0,
    keepdims=True,
)


parameter_std[
    parameter_std < 1.0e-12
] = 1.0


parameters_normalized = (
    parameters
    - parameter_mean
) / parameter_std


# ============================================================
# NORMALIZE COORDINATES TO [-1,1]
# ============================================================

coordinate_min = coordinates.min(
    axis=0,
    keepdims=True,
)


coordinate_max = coordinates.max(
    axis=0,
    keepdims=True,
)


coordinate_range = (
    coordinate_max
    - coordinate_min
)


coordinate_range[
    coordinate_range < 1.0e-12
] = 1.0


coordinates_normalized = (
    2.0
    * (
        coordinates
        - coordinate_min
    )
    / coordinate_range
    - 1.0
)


# ============================================================
# NORMALIZE EACH DISPLACEMENT COMPONENT SEPARATELY
# TRAINING DATA ONLY
# ============================================================

train_fields = U[
    train_indices
].reshape(
    -1,
    3,
)


output_mean = train_fields.mean(
    axis=0
)


output_std = train_fields.std(
    axis=0
)


output_std[
    output_std < 1.0e-12
] = 1.0


U_normalized = (
    U
    - output_mean.reshape(
        1,
        1,
        3,
    )
) / output_std.reshape(
    1,
    1,
    3,
)


print("")
print(
    "Output mean:",
    output_mean,
)

print(
    "Output std:",
    output_std,
)


# ============================================================
# SAVE NORMALIZATION
# ============================================================

normalization = {

    "parameter_mean":
        parameter_mean.reshape(-1).tolist(),

    "parameter_std":
        parameter_std.reshape(-1).tolist(),

    "coordinate_min":
        coordinate_min.reshape(-1).tolist(),

    "coordinate_max":
        coordinate_max.reshape(-1).tolist(),

    "output_mean":
        output_mean.tolist(),

    "output_std":
        output_std.tolist(),

    "output_order":
        ["U1", "U2", "U3"],
}


with open(
    os.path.join(
        RESULTS_DIR,
        "normalization.json",
    ),
    "w",
) as f:

    json.dump(
        normalization,
        f,
        indent=4,
    )


# ============================================================
# DATASET
# ============================================================

class CaseDataset(Dataset):

    def __init__(
        self,
        parameters_array,
        output_array,
        indices,
    ):

        self.parameters = torch.tensor(
            parameters_array[
                indices
            ],
            dtype=torch.float32,
        )

        self.outputs = torch.tensor(
            output_array[
                indices
            ],
            dtype=torch.float32,
        )


    def __len__(self):

        return len(
            self.parameters
        )


    def __getitem__(
        self,
        index,
    ):

        return (
            self.parameters[
                index
            ],
            self.outputs[
                index
            ],
        )


train_loader = DataLoader(
    CaseDataset(
        parameters_normalized,
        U_normalized,
        train_indices,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


val_loader = DataLoader(
    CaseDataset(
        parameters_normalized,
        U_normalized,
        val_indices,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

model = SeparateVectorDeepONet(
    branch_dim=3,
    trunk_dim=3,
    latent_dim=LATENT_DIM,
).to(
    DEVICE
)


trunk_coordinates = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=120,
        min_lr=1.0e-6,
    )
)


best_model_path = os.path.join(
    RESULTS_DIR,
    "best_separate_vector_deeponet.pt",
)


best_validation_loss = np.inf

epochs_without_improvement = 0

train_history = []

validation_history = []


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()

    train_total = 0.0

    train_count = 0


    for (
        branch_batch,
        target_batch,
    ) in train_loader:

        branch_batch = branch_batch.to(
            DEVICE
        )

        target_batch = target_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        prediction = model(
            branch_batch,
            trunk_coordinates,
        )


        loss = criterion(
            prediction,
            target_batch,
        )


        loss.backward()

        optimizer.step()


        batch_size = len(
            branch_batch
        )


        train_total += (
            loss.item()
            * batch_size
        )

        train_count += batch_size


    train_loss = (
        train_total
        / train_count
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_total = 0.0

    validation_count = 0


    with torch.no_grad():

        for (
            branch_batch,
            target_batch,
        ) in val_loader:

            branch_batch = branch_batch.to(
                DEVICE
            )

            target_batch = target_batch.to(
                DEVICE
            )


            prediction = model(
                branch_batch,
                trunk_coordinates,
            )


            validation_loss_batch = (
                criterion(
                    prediction,
                    target_batch,
                )
            )


            batch_size = len(
                branch_batch
            )


            validation_total += (
                validation_loss_batch.item()
                * batch_size
            )

            validation_count += (
                batch_size
            )


    validation_loss = (
        validation_total
        / validation_count
    )


    train_history.append(
        train_loss
    )

    validation_history.append(
        validation_loss
    )


    scheduler.step(
        validation_loss
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        validation_loss
        < best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )

        epochs_without_improvement = 0


        torch.save(
            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "validation_loss":
                    validation_loss,

                "branch_dim":
                    3,

                "latent_dim":
                    LATENT_DIM,

            },
            best_model_path,
        )

    else:

        epochs_without_improvement += 1


    if (
        epoch == 1
        or epoch % 50 == 0
    ):

        print(
            "Epoch {:5d} | "
            "Train {:.6e} | "
            "Val {:.6e} | "
            "LR {:.2e}".format(
                epoch,
                train_loss,
                validation_loss,
                optimizer.param_groups[
                    0
                ]["lr"],
            )
        )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "Early stopping at epoch",
            epoch,
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history = pd.DataFrame(
    {

        "Epoch":
            np.arange(
                1,
                len(train_history) + 1,
            ),

        "TrainLoss":
            train_history,

        "ValidationLoss":
            validation_history,

    }
)


history.to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv",
    ),
    index=False,
)


plt.figure(
    figsize=(8, 5)
)


plt.semilogy(
    history["Epoch"],
    history["TrainLoss"],
    label="Training",
)


plt.semilogy(
    history["Epoch"],
    history["ValidationLoss"],
    label="Validation",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Normalized MSE"
)

plt.title(
    "Separate displacement DeepONets"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_history.png",
    ),
    dpi=300,
)


plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# ============================================================
# TEST
# ============================================================

test_parameters = torch.tensor(
    parameters_normalized[
        test_indices
    ],
    dtype=torch.float32,
    device=DEVICE,
)


with torch.no_grad():

    prediction_normalized = model(
        test_parameters,
        trunk_coordinates,
    )


prediction_normalized = (
    prediction_normalized
    .cpu()
    .numpy()
)


prediction = (
    prediction_normalized
    * output_std.reshape(
        1,
        1,
        3,
    )
    + output_mean.reshape(
        1,
        1,
        3,
    )
)


truth = U[
    test_indices
]


np.save(
    os.path.join(
        RESULTS_DIR,
        "test_predictions.npy",
    ),
    prediction,
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "test_truth.npy",
    ),
    truth,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predicted,
    actual,
):

    error = (
        predicted
        - actual
    )


    relative_l2 = (
        np.linalg.norm(
            error
        )
        /
        (
            np.linalg.norm(
                actual
            )
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


    ss_res = np.sum(
        error ** 2
    )


    ss_tot = np.sum(
        (
            actual
            - actual.mean()
        ) ** 2
    )


    if (
        ss_tot < 1.0e-14
    ):

        r2 = np.nan

    else:

        r2 = (
            1.0
            - ss_res
            / ss_tot
        )


    return (
        relative_l2,
        rmse,
        mae,
        r2,
    )


component_names = [
    "U1",
    "U2",
    "U3",
]


rows = []


for (
    local_index,
    global_index,
) in enumerate(
    test_indices
):

    case_id = case_table.loc[
        case_table["Index"]
        == global_index,
        "CaseID",
    ].iloc[0]


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
            r2,
        ) = calculate_metrics(

            prediction[
                local_index,
                :,
                component_index,
            ],

            truth[
                local_index,
                :,
                component_index,
            ],

        )


        rows.append(
            {

                "CaseID":
                    case_id,

                "Component":
                    component_name,

                "Relative_L2":
                    relative_l2,

                "RMSE_mm":
                    rmse,

                "MAE_mm":
                    mae,

                "R2":
                    r2,

            }
        )


metrics = pd.DataFrame(
    rows
)


metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_metrics.csv",
    ),
    index=False,
)


print("")
print(
    "Best epoch:",
    checkpoint["epoch"],
)


print(
    "Best validation loss:",
    checkpoint[
        "validation_loss"
    ],
)


print("")
print(
    "Mean test metrics:"
)


print(
    metrics.groupby(
        "Component"
    )[
        [
            "Relative_L2",
            "RMSE_mm",
            "MAE_mm",
            "R2",
        ]
    ].mean()
)