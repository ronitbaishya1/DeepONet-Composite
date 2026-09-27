import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.vector_deeponet import VectorDeepONet


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DATA_DIR = "data"
RESULTS_DIR = os.path.join("results", "vector_deeponet")

LATENT_DIM = 128
BATCH_SIZE = 5
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-6
MAX_EPOCHS = 4000
PATIENCE = 300

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

parameters = np.load(
    os.path.join(DATA_DIR, "parameters.npy")
).astype(np.float32)

coordinates = np.load(
    os.path.join(DATA_DIR, "coordinates.npy")
).astype(np.float32)

U1 = np.load(os.path.join(DATA_DIR, "U1.npy")).astype(np.float32)
U2 = np.load(os.path.join(DATA_DIR, "U2.npy")).astype(np.float32)
U3 = np.load(os.path.join(DATA_DIR, "U3.npy")).astype(np.float32)

# [cases, nodes, components]
U = np.stack([U1, U2, U3], axis=-1)

split_table = pd.read_csv(
    os.path.join(DATA_DIR, "split_assignment.csv")
)

case_table = pd.read_csv(
    os.path.join(DATA_DIR, "case_ids.csv")
)

train_idx = split_table.loc[
    split_table["Split"] == "train", "Index"
].to_numpy(dtype=int)

val_idx = split_table.loc[
    split_table["Split"] == "validation", "Index"
].to_numpy(dtype=int)

test_idx = split_table.loc[
    split_table["Split"] == "test", "Index"
].to_numpy(dtype=int)

print("parameters:", parameters.shape)
print("coordinates:", coordinates.shape)
print("U:", U.shape)
print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))


# ============================================================
# NORMALIZATION
# IMPORTANT: material and output statistics use TRAIN only.
# ============================================================

p_mean = parameters[train_idx].mean(axis=0, keepdims=True)
p_std = parameters[train_idx].std(axis=0, keepdims=True)
p_std[p_std < 1.0e-12] = 1.0

p_norm = (parameters - p_mean) / p_std

x_min = coordinates.min(axis=0, keepdims=True)
x_max = coordinates.max(axis=0, keepdims=True)
x_range = x_max - x_min
x_range[x_range < 1.0e-12] = 1.0

x_norm = 2.0 * (coordinates - x_min) / x_range - 1.0

train_fields = U[train_idx].reshape(-1, 3)
u_mean = train_fields.mean(axis=0, keepdims=True)
u_std = train_fields.std(axis=0, keepdims=True)
u_std[u_std < 1.0e-12] = 1.0

U_norm = (U - u_mean.reshape(1, 1, 3)) / u_std.reshape(1, 1, 3)

print("U mean:", u_mean.reshape(-1))
print("U std :", u_std.reshape(-1))

normalization = {
    "parameter_mean": p_mean.reshape(-1).tolist(),
    "parameter_std": p_std.reshape(-1).tolist(),
    "coordinate_min": x_min.reshape(-1).tolist(),
    "coordinate_max": x_max.reshape(-1).tolist(),
    "output_mean": u_mean.reshape(-1).tolist(),
    "output_std": u_std.reshape(-1).tolist(),
    "output_order": ["U1", "U2", "U3"],
}

with open(
    os.path.join(RESULTS_DIR, "normalization.json"),
    "w",
) as f:
    json.dump(normalization, f, indent=2)


# ============================================================
# DATASET
# ============================================================

class CaseDataset(Dataset):
    def __init__(self, p, u, indices):
        self.p = torch.tensor(p[indices], dtype=torch.float32)
        self.u = torch.tensor(u[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i], self.u[i]


train_loader = DataLoader(
    CaseDataset(p_norm, U_norm, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    CaseDataset(p_norm, U_norm, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

model = VectorDeepONet(
    branch_dim=3,
    trunk_dim=3,
    latent_dim=LATENT_DIM,
).to(DEVICE)

x_tensor = torch.tensor(
    x_norm,
    dtype=torch.float32,
    device=DEVICE,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=100,
    min_lr=1.0e-6,
)

criterion = nn.MSELoss()

best_path = os.path.join(
    RESULTS_DIR,
    "best_vector_deeponet.pt",
)

best_val = np.inf
wait = 0
train_hist = []
val_hist = []


# ============================================================
# TRAIN
# ============================================================

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()
    total = 0.0
    count = 0

    for p_batch, u_batch in train_loader:

        p_batch = p_batch.to(DEVICE)
        u_batch = u_batch.to(DEVICE)

        optimizer.zero_grad()

        pred = model(p_batch, x_tensor)

        loss = criterion(pred, u_batch)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(p_batch)
        count += len(p_batch)

    train_loss = total / count

    model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for p_batch, u_batch in val_loader:

            p_batch = p_batch.to(DEVICE)
            u_batch = u_batch.to(DEVICE)

            pred = model(p_batch, x_tensor)
            loss = criterion(pred, u_batch)

            total += loss.item() * len(p_batch)
            count += len(p_batch)

    val_loss = total / count

    train_hist.append(train_loss)
    val_hist.append(val_loss)

    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        wait = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_loss": val_loss,
                "latent_dim": LATENT_DIM,
            },
            best_path,
        )
    else:
        wait += 1

    if epoch == 1 or epoch % 50 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Train {train_loss:.6e} | "
            f"Val {val_loss:.6e} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

    if wait >= PATIENCE:
        print("Early stopping at:", epoch)
        break


# ============================================================
# SAVE HISTORY
# ============================================================

history = pd.DataFrame(
    {
        "Epoch": np.arange(1, len(train_hist) + 1),
        "TrainLoss": train_hist,
        "ValidationLoss": val_hist,
    }
)

history.to_csv(
    os.path.join(RESULTS_DIR, "training_history.csv"),
    index=False,
)

plt.figure(figsize=(8, 5))
plt.semilogy(history["Epoch"], history["TrainLoss"], label="Train")
plt.semilogy(history["Epoch"], history["ValidationLoss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Normalized MSE")
plt.title("Vector DeepONet training")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "training_history.png"),
    dpi=300,
)
plt.close()


# ============================================================
# TEST
# ============================================================

checkpoint = torch.load(
    best_path,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

p_test = torch.tensor(
    p_norm[test_idx],
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():
    pred_norm = model(p_test, x_tensor).cpu().numpy()

pred = (
    pred_norm * u_std.reshape(1, 1, 3)
    + u_mean.reshape(1, 1, 3)
)

truth = U[test_idx]

np.save(
    os.path.join(RESULTS_DIR, "test_predictions.npy"),
    pred,
)

np.save(
    os.path.join(RESULTS_DIR, "test_truth.npy"),
    truth,
)


def metrics(predicted, actual):
    err = predicted - actual

    rel_l2 = (
        np.linalg.norm(err)
        / (np.linalg.norm(actual) + 1.0e-14)
    )

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    if ss_tot < 1.0e-14:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return rel_l2, rmse, mae, r2


rows = []
names = ["U1", "U2", "U3"]

for local_i, global_i in enumerate(test_idx):
    case_id = case_table.loc[
        case_table["Index"] == global_i,
        "CaseID",
    ].iloc[0]

    for c, name in enumerate(names):
        rel_l2, rmse, mae, r2 = metrics(
            pred[local_i, :, c],
            truth[local_i, :, c],
        )

        rows.append(
            {
                "Index": int(global_i),
                "CaseID": case_id,
                "Component": name,
                "Relative_L2": rel_l2,
                "RMSE_mm": rmse,
                "MAE_mm": mae,
                "R2": r2,
            }
        )

metrics_df = pd.DataFrame(rows)

metrics_df.to_csv(
    os.path.join(RESULTS_DIR, "test_metrics.csv"),
    index=False,
)

print("\nBest epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["validation_loss"])
print("\nMean test metrics by component:")
print(
    metrics_df.groupby("Component")[
        ["Relative_L2", "RMSE_mm", "MAE_mm", "R2"]
    ].mean()
)
