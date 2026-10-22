
#!/usr/bin/env python3
"""B9A11R: training-only D-optimal sensors and conditional B5 inversion.

Does not modify any checkpoint. No Abaqus calls.
Known FEM interface q; NOT full coupled FE-NO inversion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REGIONS = ("inner_left", "inner_right")
PROPS = ("E1", "E2", "G12")
ROOT_OUT = "65AM_B9A11R_sensor_inverse"


def mesh_hash(xyz):
    x = np.ascontiguousarray(xyz, dtype=np.float64)
    return hashlib.sha256(x.tobytes()).hexdigest()


def setup(args):
    import torch

    root = Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    torch.set_num_threads(min(torch.get_num_threads(), 2))

    audit = importlib.import_module(
        "65AM_B9A11P_A_readiness"
    )
    audit.audit(root, args.variant, args.loss, None)

    Q = importlib.import_module(
        "65AM_B9A11Q_A_inverse_diagnostics"
    )

    configs = {
        region: Q.load_region(
            root, region, args.variant,
            args.loss, args.sensors
        )
        for region in REGIONS
    }

    return root, Q, configs


def free_nodes(c):
    d = c["d"]
    return np.setdiff1d(
        np.arange(len(c["ds"]["coordinates"])),
        np.asarray(d["fixed"], dtype=int)
    )


def with_nodes(c, ids):
    import torch

    ids = np.asarray(ids, dtype=int)
    new = dict(c)
    new["ids"] = ids

    basis = np.asarray(
        c["d"]["basis"][:, ids, :],
        dtype=np.float32
    ).reshape(c["d"]["rank"], -1)

    new["B"] = torch.as_tensor(basis)
    return new


def valid_tune_cases(c, limit):
    ids = []
    for cid in c["d"]["tune"]:
        p = c["phys"][int(cid), :3]
        if np.all(p >= c["lo"]) and np.all(p <= c["hi"]):
            ids.append(int(cid))

    if not ids:
        raise RuntimeError(
            "No tuning cases within material training bounds"
        )

    return ids[:limit]


def logdet(A):
    sign, value = np.linalg.slogdet(A)
    if sign <= 0:
        raise RuntimeError("Invalid information matrix")
    return float(value)


def greedy_select(A, coords, count, min_distance):
    """Greedy D-optimal selection for three material parameters."""
    n = len(A)
    chosen = []
    allowed = np.ones(n, dtype=bool)
    M = np.eye(3)

    for _ in range(count):
        candidates = np.flatnonzero(allowed)

        if len(candidates) == 0:
            raise RuntimeError(
                "Not enough separated candidate sensor nodes. "
                "Reduce --min-distance."
            )

        matrices = M[None, :, :] + A[candidates]
        signs, scores = np.linalg.slogdet(matrices)
        scores[signs <= 0] = -np.inf

        j = int(candidates[np.argmax(scores)])
        chosen.append(j)
        M += A[j]

        allowed[j] = False

        if min_distance > 0:
            distances = np.linalg.norm(
                coords - coords[j], axis=1
            )
            allowed[distances < min_distance] = False

    return np.asarray(chosen, dtype=int)


def selftest():
    rng = np.random.default_rng(42)

    J = rng.normal(size=(100, 3, 3))
    A = np.einsum("nik,nil->nkl", J, J)

    coords = np.column_stack([
        np.arange(100) * 0.1,
        np.zeros(100),
        np.zeros(100)
    ])

    selected = greedy_select(A, coords, 20, 0.15)

    assert len(selected) == 20
    assert len(np.unique(selected)) == 20
    assert np.min(np.diff(np.sort(selected))) >= 2
    assert np.all(np.linalg.eigvalsh(
        np.eye(3) + A[selected].sum(axis=0)
    ) > 0)

    print("B9A11R SELFTEST PASS")


def check(args):
    _, Q, configs = setup(args)

    for region, c in configs.items():
        tune = valid_tune_cases(c, args.cases)
        op = Q.ConditionalOperator(c, tune[0])

        _, J = op.pred_and_jac(
            op.normalized_true(), True
        )

        expected = (args.sensors * 3, 3)

        if J.shape != expected or not np.isfinite(J).all():
            raise RuntimeError(
                "Invalid B5 Jacobian: " + region
            )

        print(
            "CHECK PASS", region,
            "available free nodes", len(free_nodes(c)),
            "tune cases", len(tune),
            "Jacobian", J.shape
        )


def design(args):
    import pandas as pd

    root, Q, configs = setup(args)
    out = root / "results" / ROOT_OUT / args.tag

    if out.exists():
        raise FileExistsError(
            "Output already exists. Use a new --tag: " + str(out)
        )

    selected_by_region = {}
    rows = []
    diagnostics = {}

    for ri, (region, c) in enumerate(configs.items()):
        free = free_nodes(c)
        xyz = np.asarray(c["ds"]["coordinates"], float)

        baseline = np.asarray(c["ids"], dtype=int)

        rng = np.random.default_rng(args.seed + ri)
        count = min(args.candidate_count, len(free))

        sampled = rng.choice(
            free, size=count, replace=False
        )

        pool = np.unique(np.r_[sampled, baseline])
        candidate = with_nodes(c, pool)

        fit = np.asarray(c["d"]["fit"], dtype=int)
        pick = np.linspace(
            0, len(fit) - 1,
            min(args.fit_cases, len(fit))
        ).astype(int)

        fit_ids = fit[pick]
        A = np.zeros((len(pool), 3, 3), dtype=float)

        # A +5% physical material change expressed in B5's
        # normalized parameter coordinates.
        zscale = (
            0.05 * c["mid"] / c["half"]
        ) / args.sigma_u

        for cid in fit_ids:
            op = Q.ConditionalOperator(candidate, int(cid))

            # Jacobian at training-range center, with that
            # FIT case's prescribed interface state held fixed.
            _, J = op.pred_and_jac(
                np.zeros(3), True
            )

            J = J.reshape(len(pool), 3, 3)
            J = J * zscale[None, None, :]

            A += np.einsum(
                "nik,nil->nkl", J, J
            ) / len(fit_ids)

        chosen_indices = greedy_select(
            A, xyz[pool], args.sensors,
            args.min_distance
        )

        optimized = pool[chosen_indices]

        baseline_index = np.searchsorted(
            pool, baseline
        )

        score_old = logdet(
            np.eye(3) + A[baseline_index].sum(axis=0)
        )
        score_new = logdet(
            np.eye(3) + A[chosen_indices].sum(axis=0)
        )

        if len(set(optimized.tolist())) != args.sensors:
            raise RuntimeError("Duplicate optimized sensor")

        selected_by_region[region] = {
            "baseline": baseline.tolist(),
            "optimized": optimized.tolist(),
            "mesh_sha256": mesh_hash(xyz)
        }

        diagnostics[region] = {
            "fit_cases_used": fit_ids.tolist(),
            "candidate_nodes": len(pool),
            "baseline_train_logdet": score_old,
            "optimized_train_logdet": score_new,
            "improvement_in_train_logdet": score_new - score_old
        }

        for layout, ids in (
            ("baseline", baseline),
            ("optimized", optimized)
        ):
            for index, node in enumerate(ids):
                x, y, z = xyz[node]

                rows.append({
                    "region": region,
                    "layout": layout,
                    "sensor_index": index,
                    "local_node_index": int(node),
                    "x_mm": float(x),
                    "y_mm": float(y),
                    "z_mm": float(z)
                })

        print(
            "DESIGN", region,
            "old logdet", round(score_old, 3),
            "new logdet", round(score_new, 3)
        )

    out.mkdir(parents=True)

    payload = {
        "stage": "B9A11R-design",
        "variant": args.variant,
        "loss": args.loss,
        "sensors": args.sensors,
        "sigma_u_mm": args.sigma_u,
        "seed": args.seed,
        "selection_uses_fit_only": True,
        "not_independent_validation": True,
        "prescribed_interface_q_from_FEM": True,
        "regions": selected_by_region,
        "diagnostics": diagnostics
    }

    (out / "B9A11R_locked_design.json").write_text(
        json.dumps(payload, indent=2)
    )

    pd.DataFrame(rows).to_csv(
        out / "B9A11R_sensor_locations.csv",
        index=False
    )

    print("DESIGN COMPLETE:", out)


def evaluate(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, Q, configs = setup(args)
    out = root / "results" / ROOT_OUT / args.tag

    design_file = out / "B9A11R_locked_design.json"

    if not design_file.is_file():
        raise FileNotFoundError(
            "Run --stage design first: " + str(design_file)
        )

    design_info = json.loads(design_file.read_text())

    for key, value in (
        ("variant", args.variant),
        ("loss", args.loss),
        ("sensors", args.sensors)
    ):
        if design_info[key] != value:
            raise RuntimeError(
                "Locked design configuration mismatch: " + key
            )

    result_file = out / "B9A11R_inverse_cases.csv"

    if result_file.exists():
        raise FileExistsError(
            "Inverse results already exist. "
            "Use a fresh design tag rather than overwrite."
        )

    rows = []
    sensitivity = []

    for ri, (region, c) in enumerate(configs.items()):
        ds = c["ds"]
        xyz = np.asarray(ds["coordinates"], float)

        stored = design_info["regions"][region]

        if stored["mesh_sha256"] != mesh_hash(xyz):
            raise RuntimeError(
                "FEM/B5 mesh changed since sensor design"
            )

        layouts = {
            "baseline": np.asarray(stored["baseline"], int),
            "optimized": np.asarray(stored["optimized"], int)
        }

        free = free_nodes(c)

        for name, ids in layouts.items():
            if (
                len(ids) != args.sensors
                or len(np.unique(ids)) != len(ids)
                or not np.isin(ids, free).all()
            ):
                raise RuntimeError(
                    "Invalid locked sensor nodes: " + name
                )

        cases = valid_tune_cases(c, args.cases)

        for cid in cases:
            # Generate one noise field per case/repetition,
            # shared across the two layouts at matching nodes.
            noises = []

            for rep in range(args.noise_reps):
                rng = np.random.default_rng(
                    args.seed + 100000 * ri + 100 * cid + rep
                )
                noises.append(
                    rng.normal(
                        0, args.sigma_u,
                        size=(len(xyz), 3)
                    )
                )

            for name, ids in layouts.items():
                op = Q.ConditionalOperator(
                    with_nodes(c, ids), cid
                )

                true_z = op.normalized_true()

                y_b5, J = op.pred_and_jac(
                    true_z, True
                )
                y_fem = op.base_FEM

                fwd_rmse = float(np.sqrt(
                    np.mean((y_b5 - y_fem) ** 2)
                ))

                singular = np.linalg.svd(
                    J / args.sigma_u,
                    compute_uv=False
                )

                condition = float(
                    singular[0] / max(singular[-1], 1e-12)
                )

                for j, prop in enumerate(PROPS):
                    shift = (
                        0.05 * op.true[j] / c["half"][j]
                    )

                    score = float(np.sqrt(np.mean(
                        (J[:, j] * shift / args.sigma_u) ** 2
                    )))

                    sensitivity.append({
                        "region": region,
                        "case_id": cid,
                        "layout": name,
                        "property": prop,
                        "B5_local_sensitivity": score,
                        "conditional_known_q": True
                    })

                jobs = [
                    ("B5_to_B5_zero_noise", -1, y_b5),
                    ("FEM_to_B5_zero_noise", -1, y_fem)
                ]

                for rep, noise in enumerate(noises):
                    observed = y_fem + noise[ids, :].reshape(-1)
                    jobs.append(
                        ("FEM_to_B5_noisy", rep, observed)
                    )

                for mode, rep, observed in jobs:
                    fit, rms = Q.inversion(
                        op, observed, args.sigma_u
                    )

                    estimate = c["mid"] + c["half"] * fit.x

                    row = {
                        "region": region,
                        "case_id": cid,
                        "layout": name,
                        "mode": mode,
                        "repeat": rep,
                        "success": bool(fit.success),
                        "forward_RMSE_over_sigma":
                            fwd_rmse / args.sigma_u,
                        "inverse_fit_whitened_RMS": rms,
                        "jacobian_condition": condition,
                        "any_parameter_at_bound":
                            bool(np.any(np.abs(fit.x) > 0.99))
                    }

                    for j, prop in enumerate(PROPS):
                        row[prop + "_true_MPa"] = float(op.true[j])
                        row[prop + "_estimated_MPa"] = float(
                            estimate[j]
                        )
                        row[prop + "_abs_rel_pct"] = float(
                            100 * abs(estimate[j] - op.true[j])
                            / op.true[j]
                        )
                        row[prop + "_hit_bound"] = bool(
                            abs(fit.x[j]) > 0.99
                        )

                    rows.append(row)

            print(
                "EVALUATED", region,
                "case", cid,
                flush=True
            )

    df = pd.DataFrame(rows)
    sf = pd.DataFrame(sensitivity)

    if len(df) == 0:
        raise RuntimeError("No inverse results produced")

    df.to_csv(result_file, index=False)

    sf.to_csv(
        out / "B9A11R_B5_sensitivities.csv",
        index=False
    )

    med = df.groupby(
        ["mode", "layout"]
    )[[p + "_abs_rel_pct" for p in PROPS]].median()

    modes = [
        "B5_to_B5_zero_noise",
        "FEM_to_B5_zero_noise",
        "FEM_to_B5_noisy"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for j, prop in enumerate(PROPS):
        ax = axes[j]
        xx = np.arange(len(modes))

        for offset, layout in (
            (-0.19, "baseline"),
            (0.19, "optimized")
        ):
            vals = [
                float(med.loc[(mode, layout),
                              prop + "_abs_rel_pct"])
                for mode in modes
            ]

            ax.bar(
                xx + offset, vals, width=0.38,
                label=layout
            )

        ax.set_xticks(xx)
        ax.set_xticklabels(
            ["B5 oracle", "FEM clean", "FEM noisy"],
            rotation=18
        )
        ax.set_title(prop)
        ax.set_ylabel("Median inverse error (%)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "B9A11R: baseline versus optimized sensors | known q"
    )
    fig.tight_layout()
    fig.savefig(
        out / "B9A11R_inverse_error_comparison.png",
        dpi=190
    )
    plt.close(fig)

    grouped = sf.groupby(
        ["layout", "property"]
    )["B5_local_sensitivity"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(3)

    for offset, layout in (
        (-0.19, "baseline"),
        (0.19, "optimized")
    ):
        vals = [
            grouped.loc[(layout, prop)]
            for prop in PROPS
        ]

        ax.bar(
            x + offset, vals, width=0.38,
            label=layout
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PROPS)
    ax.set_ylabel(
        "RMS B5 response shift / displacement noise"
    )
    ax.set_title(
        "B5 local material sensitivity: old vs new sensors"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out / "B9A11R_B5_sensitivity_comparison.png",
        dpi=190
    )
    plt.close(fig)

    summary = {
        "stage": "B9A11R-evaluate",
        "scope": "Known-interface B5 conditional inverse",
        "n_results": len(df),
        "tune_cases_per_region": args.cases,
        "noise_reps": args.noise_reps,
        "sensor_count_per_region": args.sensors,
        "median_inverse_errors": {
            mode: {
                layout: {
                    p: float(med.loc[
                        (mode, layout), p + "_abs_rel_pct"
                    ])
                    for p in PROPS
                }
                for layout in ("baseline", "optimized")
            }
            for mode in modes
        },
        "boundary_hit_counts": {
            str(k): int(g["any_parameter_at_bound"].sum())
            for k, g in df.groupby(["mode", "layout"])
        },
        "sensor_design_uses_fit_only": True,
        "evaluation_uses_internal_tune": True,
        "not_external_validation": True,
        "FEM_interface_q_is_known": True,
        "not_full_coupled_digital_twin": True,
        "B5_checkpoints_unchanged": True
    }

    (out / "B9A11R_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("B9A11R COMPLETE:", out)
    print(med.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True,
        choices=("selftest", "check", "design", "evaluate")
    )
    p.add_argument("--project", default=".")
    p.add_argument("--tag", default="pilot01")
    p.add_argument("--variant", default="centered")
    p.add_argument("--loss", default="grad")
    p.add_argument("--sensors", type=int, default=20)
    p.add_argument("--fit-cases", type=int, default=12)
    p.add_argument("--candidate-count", type=int, default=500)
    p.add_argument("--min-distance", type=float, default=0.20)
    p.add_argument("--cases", type=int, default=6)
    p.add_argument("--noise-reps", type=int, default=2)
    p.add_argument("--sigma-u", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=651022)
    args = p.parse_args()

    if args.stage == "selftest":
        selftest()
        return

    if not (
        3 <= args.sensors <= 50
        and args.fit_cases >= 3
        and args.candidate_count >= args.sensors
        and args.cases >= 1
        and args.noise_reps >= 0
        and args.sigma_u > 0
    ):
        raise ValueError("Invalid experiment settings")

    if args.stage == "check":
        check(args)
    elif args.stage == "design":
        design(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
