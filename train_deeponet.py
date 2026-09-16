# ================================================================
# DeepONet for 3D composite three-point bending
#
# Branch input:
#       E1, E2, G12
#
# Trunk input:
#       x, y, z
#
# Output:
#       U2(x,y,z)
#
# Designed for:
#       Apple Silicon M1 Max
#       PyTorch MPS
# ================================================================

import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

DATA_DIR = "data"
RESULTS_DIR = "results"

LATENT_DIM = 128

BRANCH_HIDDEN = [128, 128, 128]
TRUNK_HIDDEN = [128, 128, 128]

BATCH_SIZE = 5

LEARNING_RATE = 1.0e-3

MAX_EPOCHS = 3000

PATIENCE = 250

WEIGHT_DECAY = 1.0e-6


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ================================================================
# DEVICE
# ================================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("\n============================================")
print("DeepONet Composite")
print("============================================")
print("Device:", device)
print("")


# ================================================================
# CREATE RESULTS DIRECTORY
# ================================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ================================================================
# LOAD DATA
# ================================================================

parameters = np.load(
    os.path.join(
        DATA_DIR,
        "parameters.npy"
    )
).astype(np.float32)

coordinates = np.load(
    os.path.join(
        DATA_DIR,
        "coordinates.npy"
    )
).astype(np.float32)

U2 = np.load(
    os.path.join(
        DATA_DIR,
        "U2.npy"
    )
).astype(np.float32)


case_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "case_ids.csv"
    )
)

split_table = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "split_assignment.csv"
    )
)


print("Parameters :", parameters.shape)
print("Coordinates:", coordinates.shape)
print("U2         :", U2.shape)


# ================================================================
# GET TRAIN / VALIDATION / TEST INDICES
# ================================================================

train_indices = split_table[
    split_table["Split"] == "train"
]["Index"].to_numpy(dtype=int)

val_indices = split_table[
    split_table["Split"] == "validation"
]["Index"].to_numpy(dtype=int)

test_indices = split_table[
    split_table["Split"] == "test"
]["Index"].to_numpy(dtype=int)


print("")
print("Training cases  :", len(train_indices))
print("Validation cases:", len(val_indices))
print("Test cases      :", len(test_indices))


# ================================================================
# NORMALIZATION
# ================================================================

# ------------------------------------------------
# Branch parameters
# Training data ONLY
# ------------------------------------------------

parameter_mean = parameters[
    train_indices
].mean(
    axis=0,
    keepdims=True
)

parameter_std = parameters[
    train_indices
].std(
    axis=0,
    keepdims=True
)

parameter_std[
    parameter_std < 1.0e-12
] = 1.0


parameters_normalized = (
    parameters
    - parameter_mean
) / parameter_std


# ------------------------------------------------
# Coordinates
#
# Geometry is fixed for every FEM realization.
# Scale each spatial coordinate to [-1,1].
# ------------------------------------------------

coord_min = coordinates.min(
    axis=0,
    keepdims=True
)

coord_max = coordinates.max(
    axis=0,
    keepdims=True
)

coordinates_normalized = (
    2.0
    * (
        coordinates
        - coord_min
    )
    /
    (
        coord_max
        - coord_min
    )
    - 1.0
)


# ------------------------------------------------
# Output U2
# Training data ONLY
# ------------------------------------------------

U2_train_values = U2[
    train_indices
]

U2_mean = U2_train_values.mean()
U2_std = U2_train_values.std()

if U2_std < 1.0e-12:
    raise RuntimeError(
        "U2 standard deviation is zero."
    )


U2_normalized = (
    U2
    - U2_mean
) / U2_std


print("")
print("Normalization:")
print("Parameter mean:", parameter_mean)
print("Parameter std :", parameter_std)

print("U2 mean:", U2_mean)
print("U2 std :", U2_std)


# ================================================================
# SAVE NORMALIZATION INFORMATION
# ================================================================

normalization_data = {

    "parameter_mean":
        parameter_mean.reshape(-1).tolist(),

    "parameter_std":
        parameter_std.reshape(-1).tolist(),

    "coordinate_min":
        coord_min.reshape(-1).tolist(),

    "coordinate_max":
        coord_max.reshape(-1).tolist(),

    "U2_mean":
        float(U2_mean),

    "U2_std":
        float(U2_std)
}


with open(
    os.path.join(
        RESULTS_DIR,
        "normalization.json"
    ),
    "w"
) as f:

    json.dump(
        normalization_data,
        f,
        indent=4
    )


# ================================================================
# DATASET CLASS
# ================================================================

class FEMCaseDataset(Dataset):

    def __init__(
        self,
        parameters_array,
        output_array,
        indices
    ):

        self.parameters = torch.tensor(
            parameters_array[indices],
            dtype=torch.float32
        )

        self.outputs = torch.tensor(
            output_array[indices],
            dtype=torch.float32
        )

    def __len__(self):

        return len(
            self.parameters
        )

    def __getitem__(
        self,
        index
    ):

        return (
            self.parameters[index],
            self.outputs[index]
        )


train_dataset = FEMCaseDataset(
    parameters_normalized,
    U2_normalized,
    train_indices
)

val_dataset = FEMCaseDataset(
    parameters_normalized,
    U2_normalized,
    val_indices
)

test_dataset = FEMCaseDataset(
    parameters_normalized,
    U2_normalized,
    test_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# MLP
# ================================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim
    ):

        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.Tanh()
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                output_dim
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


# ================================================================
# DEEPONET
# ================================================================

class DeepONet(nn.Module):

    def __init__(
        self,
        branch_input_dim=3,
        trunk_input_dim=3,
        latent_dim=128
    ):

        super().__init__()

        self.branch = MLP(
            branch_input_dim,
            BRANCH_HIDDEN,
            latent_dim
        )

        self.trunk = MLP(
            trunk_input_dim,
            TRUNK_HIDDEN,
            latent_dim
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        branch_input,
        trunk_input
    ):

        # branch_output:
        # [batch_size, latent_dim]

        branch_output = self.branch(
            branch_input
        )

        # trunk_output:
        # [num_nodes, latent_dim]

        trunk_output = self.trunk(
            trunk_input
        )

        # Output:
        # [batch_size, num_nodes]

        output = torch.matmul(
            branch_output,
            trunk_output.T
        )

        output = output + self.bias

        return output


# ================================================================
# MODEL
# ================================================================

model = DeepONet(
    branch_input_dim=3,
    trunk_input_dim=3,
    latent_dim=LATENT_DIM
).to(device)


print("")
print(model)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("")
print(
    "Trainable parameters:",
    total_parameters
)


# ================================================================
# TRUNK COORDINATES
# ================================================================

trunk_coordinates = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32
).to(device)


# ================================================================
# LOSS / OPTIMIZER / SCHEDULER
# ================================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=75,
    min_lr=1.0e-6
)


# ================================================================
# TRAINING
# ================================================================

train_history = []
val_history = []

best_val_loss = np.inf

epochs_without_improvement = 0


BEST_MODEL_FILE = os.path.join(
    RESULTS_DIR,
    "best_deeponet.pt"
)


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    number_train_samples = 0


    for branch_batch, target_batch in train_loader:

        branch_batch = branch_batch.to(
            device
        )

        target_batch = target_batch.to(
            device
        )

        optimizer.zero_grad()

        predictions = model(
            branch_batch,
            trunk_coordinates
        )

        loss = criterion(
            predictions,
            target_batch
        )

        loss.backward()

        optimizer.step()

        batch_size = branch_batch.shape[0]

        running_train_loss += (
            loss.item()
            * batch_size
        )

        number_train_samples += (
            batch_size
        )


    train_loss = (
        running_train_loss
        / number_train_samples
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    number_val_samples = 0


    with torch.no_grad():

        for branch_batch, target_batch in val_loader:

            branch_batch = branch_batch.to(
                device
            )

            target_batch = target_batch.to(
                device
            )

            predictions = model(
                branch_batch,
                trunk_coordinates
            )

            loss = criterion(
                predictions,
                target_batch
            )

            batch_size = branch_batch.shape[0]

            running_val_loss += (
                loss.item()
                * batch_size
            )

            number_val_samples += (
                batch_size
            )


    val_loss = (
        running_val_loss
        / number_val_samples
    )


    train_history.append(
        train_loss
    )

    val_history.append(
        val_loss
    )


    scheduler.step(
        val_loss
    )


    # ------------------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_loss":
                    val_loss
            },
            BEST_MODEL_FILE
        )

    else:

        epochs_without_improvement += 1


    # ------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------

    if (
        epoch == 1
        or
        epoch % 25 == 0
    ):

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            "Epoch {:5d} | "
            "Train {:.6e} | "
            "Val {:.6e} | "
            "LR {:.2e}".format(
                epoch,
                train_loss,
                val_loss,
                current_lr
            )
        )


    # ------------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print("")
        print(
            "Early stopping at epoch",
            epoch
        )

        break


# ================================================================
# SAVE LOSS HISTORY
# ================================================================

history_dataframe = pd.DataFrame(
    {
        "Epoch":
            np.arange(
                1,
                len(train_history) + 1
            ),

        "TrainLoss":
            train_history,

        "ValidationLoss":
            val_history
    }
)


history_dataframe.to_csv(
    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    ),
    index=False
)


# ================================================================
# PLOT TRAINING HISTORY
# ================================================================

plt.figure(
    figsize=(8, 5)
)

plt.semilogy(
    history_dataframe["Epoch"],
    history_dataframe["TrainLoss"],
    label="Training"
)

plt.semilogy(
    history_dataframe["Epoch"],
    history_dataframe["ValidationLoss"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("DeepONet training history")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_history.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# LOAD BEST MODEL
# ================================================================

checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print("")
print(
    "Best validation epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation loss:",
    checkpoint[
        "validation_loss"
    ]
)


# ================================================================
# TEST PREDICTIONS
# ================================================================

test_parameters_tensor = torch.tensor(
    parameters_normalized[
        test_indices
    ],
    dtype=torch.float32
).to(device)


with torch.no_grad():

    predictions_normalized = model(
        test_parameters_tensor,
        trunk_coordinates
    )


predictions_normalized = (
    predictions_normalized
    .cpu()
    .numpy()
)


# ================================================================
# DENORMALIZE
# ================================================================

predictions_test = (
    predictions_normalized
    * U2_std
    + U2_mean
)

truth_test = U2[
    test_indices
]


# ================================================================
# METRICS
# ================================================================

def relative_l2(
    prediction,
    truth
):

    numerator = np.linalg.norm(
        prediction - truth
    )

    denominator = np.linalg.norm(
        truth
    )

    return (
        numerator
        / denominator
    )


metrics_rows = []


for local_index, global_index in enumerate(
    test_indices
):

    prediction = predictions_test[
        local_index
    ]

    truth = truth_test[
        local_index
    ]

    error = prediction - truth


    rel_l2 = relative_l2(
        prediction,
        truth
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
        (
            truth
            - prediction
        ) ** 2
    )

    ss_tot = np.sum(
        (
            truth
            - truth.mean()
        ) ** 2
    )

    r2 = 1.0 - (
        ss_res
        / ss_tot
    )


    case_id = case_table.iloc[
        global_index
    ]["CaseID"]


    metrics_rows.append(
        {
            "CaseID": case_id,

            "Relative_L2":
                rel_l2,

            "RMSE_mm":
                rmse,

            "MAE_mm":
                mae,

            "R2":
                r2
        }
    )


metrics_dataframe = pd.DataFrame(
    metrics_rows
)


metrics_dataframe.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_metrics.csv"
    ),
    index=False
)


print("")
print("============================================")
print("TEST RESULTS")
print("============================================")

print(metrics_dataframe)

print("")
print(
    "Mean Relative L2:",
    metrics_dataframe[
        "Relative_L2"
    ].mean()
)

print(
    "Mean RMSE [mm]:",
    metrics_dataframe[
        "RMSE_mm"
    ].mean()
)

print(
    "Mean MAE [mm]:",
    metrics_dataframe[
        "MAE_mm"
    ].mean()
)

print(
    "Mean R2:",
    metrics_dataframe[
        "R2"
    ].mean()
)


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "test_predictions_U2.npy"
    ),
    predictions_test
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "test_truth_U2.npy"
    ),
    truth_test
)


# ================================================================
# PARITY PLOT - ALL TEST NODES
# ================================================================

truth_flat = truth_test.reshape(-1)

prediction_flat = predictions_test.reshape(-1)


plt.figure(
    figsize=(6, 6)
)

plt.scatter(
    truth_flat,
    prediction_flat,
    s=4,
    alpha=0.25
)

minimum_value = min(
    truth_flat.min(),
    prediction_flat.min()
)

maximum_value = max(
    truth_flat.max(),
    prediction_flat.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    "--"
)

plt.xlabel(
    "FEM U2 [mm]"
)

plt.ylabel(
    "DeepONet U2 [mm]"
)

plt.title(
    "FEM vs DeepONet — unseen test cases"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "test_parity_U2.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# MID-WIDTH FIELD COMPARISON
# First test case
# ================================================================

first_prediction = predictions_test[0]

first_truth = truth_test[0]

first_error = np.abs(
    first_prediction
    - first_truth
)


# Find plane closest to z = 4 mm
target_z = (
    coordinates[:, 2].min()
    +
    coordinates[:, 2].max()
) / 2.0


unique_z = np.unique(
    coordinates[:, 2]
)

selected_z = unique_z[
    np.argmin(
        np.abs(
            unique_z
            - target_z
        )
    )
]


plane_mask = np.isclose(
    coordinates[:, 2],
    selected_z
)


plane_coordinates = coordinates[
    plane_mask
]

plane_truth = first_truth[
    plane_mask
]

plane_prediction = first_prediction[
    plane_mask
]

plane_error = first_error[
    plane_mask
]


# ------------------------------------------------
# FEM field
# ------------------------------------------------

plt.figure(
    figsize=(8, 4)
)

scatter = plt.scatter(
    plane_coordinates[:, 0],
    plane_coordinates[:, 1],
    c=plane_truth,
    s=25
)

plt.colorbar(
    scatter,
    label="U2 [mm]"
)

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")

plt.title(
    "FEM U2 — mid-width plane"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "test_case_FEM_U2.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------
# DeepONet field
# ------------------------------------------------

plt.figure(
    figsize=(8, 4)
)

scatter = plt.scatter(
    plane_coordinates[:, 0],
    plane_coordinates[:, 1],
    c=plane_prediction,
    s=25
)

plt.colorbar(
    scatter,
    label="U2 [mm]"
)

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")

plt.title(
    "DeepONet U2 — mid-width plane"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "test_case_DeepONet_U2.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------
# Absolute error field
# ------------------------------------------------

plt.figure(
    figsize=(8, 4)
)

scatter = plt.scatter(
    plane_coordinates[:, 0],
    plane_coordinates[:, 1],
    c=plane_error,
    s=25
)

plt.colorbar(
    scatter,
    label="Absolute error [mm]"
)

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")

plt.title(
    "Absolute DeepONet error — mid-width plane"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "test_case_absolute_error.png"
    ),
    dpi=300
)

plt.close()


print("")
print("============================================")
print("TRAINING COMPLETE")
print("============================================")

print("")
print(
    "Results saved in:",
    RESULTS_DIR
)

print("")
print(
    "Best model:",
    BEST_MODEL_FILE
)