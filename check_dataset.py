import os
import numpy as np
import pandas as pd
import torch


DATA_DIR = "data"


# ============================================================
# LOAD DATA
# ============================================================

parameters = np.load(
    os.path.join(DATA_DIR, "parameters.npy")
)

coordinates = np.load(
    os.path.join(DATA_DIR, "coordinates.npy")
)

node_labels = np.load(
    os.path.join(DATA_DIR, "node_labels.npy")
)

U1 = np.load(
    os.path.join(DATA_DIR, "U1.npy")
)

U2 = np.load(
    os.path.join(DATA_DIR, "U2.npy")
)

U3 = np.load(
    os.path.join(DATA_DIR, "U3.npy")
)

splits = pd.read_csv(
    os.path.join(DATA_DIR, "split_assignment.csv")
)

case_ids = pd.read_csv(
    os.path.join(DATA_DIR, "case_ids.csv")
)


# ============================================================
# PRINT SHAPES
# ============================================================

print("\n================ DATASET SHAPES ================\n")

print("parameters :", parameters.shape)
print("coordinates:", coordinates.shape)
print("node_labels:", node_labels.shape)

print("U1:", U1.shape)
print("U2:", U2.shape)
print("U3:", U3.shape)


# ============================================================
# BASIC CHECKS
# ============================================================

assert parameters.shape == (50, 3)
assert coordinates.shape == (4592, 3)

assert U1.shape == (50, 4592)
assert U2.shape == (50, 4592)
assert U3.shape == (50, 4592)

assert np.all(np.isfinite(parameters))
assert np.all(np.isfinite(coordinates))
assert np.all(np.isfinite(U2))

print("\nAll arrays have expected dimensions.")
print("No NaN or Inf values detected.")


# ============================================================
# PARAMETER RANGES
# ============================================================

print("\n================ PARAMETER RANGES ================\n")

names = ["E1", "E2", "G12"]

for i, name in enumerate(names):

    print(
        "{}: {:.6f} to {:.6f} MPa".format(
            name,
            parameters[:, i].min(),
            parameters[:, i].max()
        )
    )


# ============================================================
# COORDINATE RANGES
# ============================================================

print("\n================ COORDINATE RANGES ================\n")

coordinate_names = ["X", "Y", "Z"]

for i, name in enumerate(coordinate_names):

    print(
        "{}: {:.6f} to {:.6f} mm".format(
            name,
            coordinates[:, i].min(),
            coordinates[:, i].max()
        )
    )


# ============================================================
# DISPLACEMENT RANGE
# ============================================================

print("\n================ U2 RANGE ================\n")

print("Minimum U2:", U2.min())
print("Maximum U2:", U2.max())
print("Mean U2   :", U2.mean())
print("Std U2    :", U2.std())


# ============================================================
# SPLITS
# ============================================================

print("\n================ SPLITS ================\n")

print(splits["Split"].value_counts())


# ============================================================
# APPLE GPU CHECK
# ============================================================

print("\n================ DEVICE ================\n")

if torch.backends.mps.is_available():

    print("Apple MPS GPU is available.")

else:

    print("MPS unavailable. Training will use CPU.")


print("\nDataset check completed successfully.\n")