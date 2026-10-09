
# Mac: numpy, pandas, matplotlib

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

OLD = (
    ROOT / "results"
    / "hybrid_online_7region_direct_validation"
)

NEW = (
    ROOT / "results"
    / "hybrid_online_7region_rom_validation"
)

OLD_ONLINE = (
    ROOT / "data"
    / "hybrid_online_7region_direct"
)

NEW_ONLINE = (
    ROOT / "data"
    / "hybrid_online_7region_rom"
)

OUT = (
    ROOT / "results"
    / "65J_direct_vs_rom_comparison"
)

OUT.mkdir(parents=True, exist_ok=True)


INTERFACES = [
    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
]

REGIONS = [
    "NO_OL",
    "FE_L",
    "NO_L",
    "FE_C",
    "NO_R",
    "FE_R",
    "NO_OR",
]

COMPONENTS = [
    "11", "22", "33", "12", "13", "23"
]


# ============================================================
# HELPERS
# ============================================================

def read_csv(folder, name):
    path = folder / name

    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def read_json(path):
    with path.open() as f:
        return json.load(f)


def relative_l2(pred, truth):
    return float(
        100.0
        * np.linalg.norm(pred - truth)
        / max(np.linalg.norm(truth), 1e-14)
    )


# ============================================================
# MATCH PHYSICAL LOCATIONS BY LABEL
# ============================================================

def join_field(file_name, key, comps):
    a = read_csv(OLD, file_name)
    b = read_csv(NEW, file_name)

    if (
        a[key].duplicated().any()
        or b[key].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate {} in {}"
            .format(key, file_name)
        )

    joined = a.merge(
        b,
        on=key,
        suffixes=("_direct", "_rom"),
        validate="one_to_one",
    )

    if len(joined) != len(a) or len(joined) != len(b):
        raise RuntimeError(
            "Mismatch in {} count".format(file_name)
        )

    for axis in ("X", "Y", "Z"):
        delta = np.max(
            np.abs(
                joined[axis + "_direct"]
                - joined[axis + "_rom"]
            )
        )

        if delta > 1e-4:
            raise RuntimeError(
                "Physical locations differ in "
                + file_name
            )

    # Check both runs use the identical full-FEM data.
    for comp in comps:
        direct = joined[
            "FEM_" + comp + "_direct"
        ].to_numpy(dtype=float)

        rom = joined[
            "FEM_" + comp + "_rom"
        ].to_numpy(dtype=float)

        if not np.allclose(
            direct,
            rom,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Different full-FEM reference for "
                + comp
            )

    if not (
        joined["Owner_direct"]
        == joined["Owner_rom"]
    ).all():
        raise RuntimeError(
            "Region ownership changed."
        )

    return (
        joined.sort_values(key)
        .reset_index(drop=True)
    )


# ============================================================
# ERROR COMPARISON
# ============================================================

def compare_components(df, comps, source):
    rows = []

    for comp in comps:
        truth = df[
            "FEM_" + comp + "_direct"
        ].to_numpy(dtype=float)

        a = df[
            "Hybrid_" + comp + "_direct"
        ].to_numpy(dtype=float)

        b = df[
            "Hybrid_" + comp + "_rom"
        ].to_numpy(dtype=float)

        old_error = relative_l2(a, truth)
        new_error = relative_l2(b, truth)

        rms_difference = float(
            np.sqrt(np.mean((a - b) ** 2))
        )

        direct_rom_difference_percent = (
            100.0
            * np.linalg.norm(b - a)
            / max(np.linalg.norm(truth), 1e-14)
        )

        rows.append({
            "Field": source,
            "Component": comp,
            "OriginalDirectRelL2_percent": old_error,
            "ROMCorrectedRelL2_percent": new_error,
            "ChangePercentagePoints":
                new_error - old_error,
            "DirectVsROM_RelL2_to_FEM_percent":
                direct_rom_difference_percent,
            "DirectVsROM_RMSE":
                rms_difference,
        })

    return rows


# ============================================================
# LOAD MATCHED NODE AND ELEMENT FIELDS
# ============================================================

nodes = join_field(
    "assembled_hybrid_nodes.csv",
    "NodeLabel",
    ["U1", "U2", "U3"],
)

ip = join_field(
    "assembled_hybrid_ip.csv",
    "ElementLabel",
    (
        ["LE" + x for x in COMPONENTS]
        + ["S" + x for x in COMPONENTS]
    ),
)


# ============================================================
# GLOBAL COMPARISON
# ============================================================

metrics = pd.DataFrame(
    compare_components(
        nodes,
        ["U1", "U2", "U3"],
        "Displacement",
    )
    + compare_components(
        ip,
        ["LE" + x for x in COMPONENTS],
        "Logarithmic strain",
    )
    + compare_components(
        ip,
        ["S" + x for x in COMPONENTS],
        "Stress",
    )
)

metrics.to_csv(
    OUT / "01_global_direct_vs_rom_metrics.csv",
    index=False,
)


# ============================================================
# REGION-WISE COMPARISON
# ============================================================

regions = []

for region in REGIONS:
    n = nodes.loc[
        nodes.Owner_direct == region
    ]

    e = ip.loc[
        ip.Owner_direct == region
    ]

    comparisons = [
        (
            "Displacement",
            n,
            ["U1", "U2", "U3"],
        ),
        (
            "Logarithmic strain",
            e,
            ["LE11", "LE12"],
        ),
        (
            "Stress",
            e,
            ["S11", "S12"],
        ),
    ]

    for field, part, comps in comparisons:
        if part.empty:
            raise RuntimeError(
                "Region has no {} samples: {}"
                .format(field, region)
            )

        for item in compare_components(
            part, comps, field
        ):
            item["Region"] = region
            item["Count"] = len(part)
            regions.append(item)

region_table = pd.DataFrame(regions)

region_table.to_csv(
    OUT / "02_region_direct_vs_rom_metrics.csv",
    index=False,
)


# ============================================================
# COMPARE ALL 26 INTERFACE PCA COEFFICIENTS
# ============================================================

old_s = np.load(
    OLD_ONLINE / "online_solution_7region.npz"
)

new_s = np.load(
    NEW_ONLINE / "rom_solution_7region.npz"
)

interface_rows = []

for name in INTERFACES:
    a = np.asarray(
        old_s["c_" + name]
    ).ravel()

    b = np.asarray(
        new_s["c_" + name]
    ).ravel()

    if a.shape != b.shape:
        raise RuntimeError(
            "Interface dimension changed: " + name
        )

    pca_file = (
        ROOT / "data"
        / "candidate_partition_interfaces_7region"
        / ("interface_" + name + ".npz")
    )

    data = np.load(pca_file)

    scale = np.maximum(
        np.std(
            data["scores_train"],
            axis=0,
            ddof=1,
        ),
        1e-8,
    )

    for j in range(a.size):
        interface_rows.append({
            "Interface": name,
            "Mode": j + 1,
            "DirectCoefficient": a[j],
            "ROMCorrectedCoefficient": b[j],
            "Difference": b[j] - a[j],
            "DifferenceInTrainingStd":
                (b[j] - a[j]) / scale[j],
        })

interface_df = pd.DataFrame(interface_rows)

if len(interface_df) != 26:
    raise RuntimeError(
        "Expected 26 interface variables."
    )

interface_df.to_csv(
    OUT / "03_interface_coefficient_comparison.csv",
    index=False,
)


# ============================================================
# REACTION FORCE COMPARISON
# ============================================================

r_old = read_json(
    OLD_ONLINE
    / "validation_fields"
    / "reaction_summary.json"
)

r_new = read_json(
    NEW_ONLINE
    / "validation_fields"
    / "reaction_summary.json"
)

manifest_file = (
    NEW_ONLINE / "rom_solution_manifest.json"
)

manifest = (
    read_json(manifest_file)
    if manifest_file.exists()
    else None
)

summary = {
    "FullFEMNoseReaction_N":
        float(r_old["full_FEM_nose_RF2_N"]),
    "DirectHybridNoseReaction_N":
        float(
            r_old["hybrid_center_nose_RF2_N"]
        ),
    "ROMCorrectedNoseReaction_N":
        float(
            r_new["hybrid_center_nose_RF2_N"]
        ),
    "DirectNoseReactionError_percent":
        100.0
        * float(r_old["relative_reaction_error"]),
    "ROMCorrectedNoseReactionError_percent":
        100.0
        * float(r_new["relative_reaction_error"]),
    "MaxAbsInterfaceCoefficientDeltaInTrainingStd":
        float(
            np.max(
                np.abs(
                    interface_df[
                        "DifferenceInTrainingStd"
                    ]
                )
            )
        ),
}


if abs(
    float(r_old["full_FEM_nose_RF2_N"])
    - float(r_new["full_FEM_nose_RF2_N"])
) > 1e-5:
    raise RuntimeError(
        "Reaction full-FEM references differ."
    )


if manifest is not None:
    for key in (
        "converged",
        "rom_best_rms",
        "final_high_fidelity_rms",
        "high_fidelity_abaqus_jobs",
        "total_seconds",
    ):
        summary[key] = manifest[key]


with (
    OUT / "04_reaction_and_solver_summary.json"
).open("w") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# CENTERLINE U2 PLOT
#
# Select the same near-centerline node at every X plane.
# ============================================================

plot = nodes.copy()

plot["DistanceFromMidWidthThickness"] = (
    (plot["Y_direct"] - 2.0) ** 2
    + (plot["Z_direct"] - 4.0) ** 2
)

plot["PlaneX"] = (
    plot["X_direct"].round(5)
)

plot = (
    plot.sort_values(
        "DistanceFromMidWidthThickness"
    )
    .drop_duplicates("PlaneX")
    .sort_values("PlaneX")
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

x = plot["X_direct"]

ax.plot(
    x,
    plot["FEM_U2_direct"],
    marker=".",
    label="Full FEM",
)

ax.plot(
    x,
    plot["Hybrid_U2_direct"],
    linestyle="--",
    label="Original direct hybrid (64T)",
)

ax.plot(
    x,
    plot["Hybrid_U2_rom"],
    linestyle=":",
    label="ROM-corrected hybrid (65H)",
)

ax.set(
    xlabel="Beam coordinate X (mm)",
    ylabel="Vertical displacement U2 (mm)",
    title="65J: Matched-node vertical displacement",
)

ax.legend()
ax.grid(True, alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "05_centerline_U2_comparison.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# CENTERLINE U2 ABSOLUTE ERROR
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    x,
    np.abs(
        plot["Hybrid_U2_direct"]
        - plot["FEM_U2_direct"]
    ),
    marker=".",
    label="Direct absolute error",
)

ax.plot(
    x,
    np.abs(
        plot["Hybrid_U2_rom"]
        - plot["FEM_U2_rom"]
    ),
    marker=".",
    label="ROM-corrected absolute error",
)

ax.set(
    xlabel="Beam coordinate X (mm)",
    ylabel="Absolute U2 error (mm)",
    title="65J: Vertical displacement error",
)

ax.legend()
ax.grid(True, alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "06_centerline_U2_absolute_error.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# PRINT SUMMARY
# ============================================================

important = metrics[
    metrics.Component.isin([
        "U1", "U2", "U3",
        "LE11", "LE12",
        "S11", "S12",
    ])
]

print("")
print("===================================")
print("STEP 65J COMPLETE")
print("===================================")

print("")
print(important.to_string(index=False))

print("")
print(json.dumps(summary, indent=2))

print("")
print("Saved:", OUT)
