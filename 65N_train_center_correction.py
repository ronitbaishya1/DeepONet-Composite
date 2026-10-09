
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rbf(x, y, gamma):
    d2 = np.maximum(
        0,
        ((x[:, None, :] - y[None, :, :]) ** 2).sum(2),
    )
    return np.exp(-gamma * d2)


def fit(x, y, gamma, lam):
    mean = y.mean(axis=0)

    alpha = np.linalg.solve(
        rbf(x, x, gamma) + lam * np.eye(len(x)),
        y - mean,
    )
    return mean, alpha


def predict(x, train_x, mean, alpha, gamma):
    return mean + rbf(x, train_x, gamma).dot(alpha)


def rms(y):
    return float(
        np.sqrt(np.mean(np.asarray(y) ** 2))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="65M_E_center_enrichment_candidates.csv",
    )
    p.add_argument(
        "--sample",
        default="sample_011.npz",
    )
    p.add_argument(
        "--output",
        default="65N_center_correction.npz",
    )
    args = p.parse_args()

    rows = list(csv.DictReader(
        open(args.csv, newline="")
    ))

    rows.sort(
        key=lambda row: int(row["sample_id"])
    )

    if len(rows) < 10:
        raise RuntimeError(
            "At least ten snapshots are required."
        )

    IDs = np.array([
        int(row["sample_id"]) for row in rows
    ])

    if len(set(IDs.tolist())) != len(IDs):
        raise RuntimeError(
            "Repeated sample_id found."
        )

    materials = np.array([
        [
            float(row["input_%02d" % j])
            for j in range(3)
        ]
        for row in rows
    ])

    c = np.array([
        [
            float(row["input_%02d" % j])
            for j in range(3, 13)
        ]
        for row in rows
    ])

    rom_g = np.array([
        [
            float(row["ROM_g_%02d" % j])
            for j in range(10)
        ]
        for row in rows
    ])

    fe_g = np.array([
        [
            float(row["Abaqus_g_%02d" % j])
            for j in range(10)
        ]
        for row in rows
    ])

    # Recover the existing 26-variable solver's
    # force normalization from the uploaded sample.
    with np.load(args.sample) as s:
        force_difference = np.r_[
            s["Abaqus_g_center_left"]
            - s["ROM_g_center_left"],

            s["Abaqus_g_center_right"]
            - s["ROM_g_center_right"],
        ]

        normalized_difference = s["delta"][8:18]

        if np.any(
            abs(normalized_difference) < 1e-10
        ):
            raise RuntimeError(
                "Cannot infer force normalization."
            )

        force_scale = (
            force_difference / normalized_difference
        )

    if (
        np.any(force_scale <= 0)
        or not np.all(np.isfinite(force_scale))
    ):
        raise RuntimeError(
            "Invalid force scaling."
        )

    if not np.allclose(
        force_difference / force_scale,
        normalized_difference,
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            "Force scaling audit failed."
        )

    # Last four trajectory states are holdout.
    split = max(6, len(rows) - 4)
    train = np.arange(split)
    test = np.arange(split, len(rows))

    material_ref = materials[0]

    if not np.allclose(
        materials,
        material_ref,
        atol=1e-8,
        rtol=0,
    ):
        raise RuntimeError(
            "Multi-material validation is not "
            "implemented in this local model."
        )

    # Use training data ONLY for scaling.
    mean_x = c[train].mean(0)

    std_x = np.maximum(
        c[train].std(0),
        1e-7,
    )

    x = (c - mean_x) / std_x

    # Normalized Abaqus-minus-ROM force error.
    y = (fe_g - rom_g) / force_scale

    # Training-only leave-one-out tuning.
    candidates = []

    for gamma in (0.02, 0.1, 0.5, 2.0):
        for lam in (0.01, 0.1, 1.0):
            prediction = np.zeros_like(y[train])

            for j, held in enumerate(train):
                fit_ids = np.delete(train, j)

                mu, alpha = fit(
                    x[fit_ids],
                    y[fit_ids],
                    gamma,
                    lam,
                )

                prediction[j] = predict(
                    x[held:held + 1],
                    x[fit_ids],
                    mu,
                    alpha,
                    gamma,
                )[0]

            candidates.append((
                rms(prediction - y[train]),
                gamma,
                lam,
            ))

    _, gamma, lam = min(candidates)

    mean, alpha = fit(
        x[train],
        y[train],
        gamma,
        lam,
    )

    yhat = predict(
        x[test],
        x[train],
        mean,
        alpha,
        gamma,
    )

    raw = rms(y[test])
    corrected = rms(y[test] - yhat)

    # Require at least 10% improvement.
    approved = corrected < 0.9 * raw

    np.savez(
        args.output,
        x_train=x[train],
        mean_x=mean_x,
        std_x=std_x,
        delta_mean=mean,
        alpha=alpha,
        gamma=gamma,
        ridge=lam,
        force_scale_center=force_scale,
        material_MPa=material_ref,
        max_train_distance=8.0,
        approved=np.array(approved),
        train_ids=IDs[train],
        test_ids=IDs[test],
    )

    summary = {
        "train_sample_ids": IDs[train].tolist(),
        "holdout_sample_ids": IDs[test].tolist(),
        "gamma": gamma,
        "ridge": lam,
        "center_normalized_gap_rms_before": raw,
        "center_normalized_gap_rms_after": corrected,
        "improvement_pct":
            100 * (1 - corrected / max(raw, 1e-12)),
        "approved_for_same_material_only":
            bool(approved),
        "limitation":
            "Chronological holdout from one "
            "short solver trajectory; not "
            "cross-material validation.",
    }

    with open(
        Path(args.output).with_suffix(".json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
