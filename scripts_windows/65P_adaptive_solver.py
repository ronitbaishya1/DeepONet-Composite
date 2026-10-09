
"""
65P — Adaptive interface solver.

Methods:
    Broyden
    Levenberg-Marquardt
    ROM-preconditioned Newton-Krylov

Actual equilibrium convergence requires all three Abaqus
patches to be verified at the same interface state.
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


CASES = {
    "baseline": (45000., 12000., 4500.),
    "soft": (42000., 11700., 4250.),
    "middle": (46500., 12500., 4750.),
    "stiff": (49000., 10900., 4500.),
}


NAMES = (
    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
)


def load_module(name, filename):
    path = os.path.join(HERE, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def rmsp(v):
    return float(
        np.sqrt(
            np.mean(np.asarray(v) ** 2)
        )
    )


def phi(r):
    return 0.5 * float(
        np.dot(r, r)
    )


# ============================================================
# CORRECTED-ROM JACOBIAN
# ============================================================

def numerical_jacobian(fun, q, h=2e-3):
    f0 = fun(q)

    J = np.empty(
        (26, 26), dtype=float
    )

    for j in range(26):
        x = q.copy()

        dx = h * max(
            1.0, abs(q[j])
        )

        x[j] = np.clip(
            q[j] + dx,
            -3,
            3,
        )

        if x[j] == q[j]:
            x[j] = np.clip(
                q[j] - dx,
                -3,
                3,
            )

        if x[j] == q[j]:
            raise RuntimeError(
                "Cannot perturb interface coordinate "
                + str(j)
            )

        J[:, j] = (
            fun(x) - f0
        ) / (
            x[j] - q[j]
        )

    return J


def trunc(p, radius):
    n = float(
        np.linalg.norm(p)
    )

    if n > radius:
        p = p * radius / n

    return np.asarray(
        p, dtype=float
    )


# ============================================================
# LEVENBERG-MARQUARDT
# ============================================================

def lm(J, r, reg, radius):
    G = J.T @ J

    A = (
        G
        + reg * np.diag(
            np.maximum(
                np.diag(G),
                1e-5,
            )
        )
    )

    return trunc(
        -np.linalg.solve(
            A,
            J.T @ r,
        ),
        radius,
    )


# ============================================================
# GMRES: CHEAP CORRECTED-ROM KRYLOV SOLVER
# ============================================================

def gmres(
    Aop,
    b,
    maximum=16,
    tol=1e-3,
):
    b = np.asarray(
        b, dtype=float
    )

    beta = float(
        np.linalg.norm(b)
    )

    if beta < 1e-14:
        return np.zeros_like(b)

    n = len(b)
    kmax = min(maximum, n)

    V = np.zeros(
        (n, kmax + 1)
    )

    H = np.zeros(
        (kmax + 1, kmax)
    )

    V[:, 0] = b / beta

    result = np.zeros(n)

    for j in range(kmax):
        w = Aop(V[:, j])

        for i in range(j + 1):
            H[i, j] = np.dot(
                V[:, i], w
            )

            w -= H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] > 1e-13:
            V[:, j + 1] = (
                w / H[j + 1, j]
            )

        e = np.zeros(j + 2)
        e[0] = beta

        coeff = np.linalg.lstsq(
            H[:j + 2, :j + 1],
            e,
            rcond=None,
        )[0]

        result = (
            V[:, :j + 1] @ coeff
        )

        error = np.linalg.norm(
            e - H[:j + 2, :j + 1] @ coeff
        )

        if (
            error < tol * beta
            or H[j + 1, j] <= 1e-13
        ):
            break

    return result


# ============================================================
# ROM-PRECONDITIONED NEWTON-KRYLOV
# ============================================================

def nk(
    corrected,
    romfun,
    q,
    r,
    Jrom,
    radius,
):
    # Left preconditioner from cheap ROM Jacobian.
    P = np.linalg.pinv(
        Jrom,
        rcond=1e-3,
    )

    e = 1e-4

    def Aop(v):
        jv = (
            corrected(
                np.clip(
                    q + e * v,
                    -3,
                    3,
                )
            ) - r
        ) / e

        return P @ jv

    p = gmres(
        Aop,
        -P @ r,
        maximum=18,
        tol=1e-3,
    )

    return trunc(p, radius)


# ============================================================
# PREVIOUS VERIFIED STARTING STATE
# ============================================================

def start_state(case):
    if case == "baseline":
        roots = [
            (
                "online_results_7region_rom",
                "65H",
            )
        ]

    elif case == "middle":
        roots = [
            (
                "online_results_7region_65M_middle",
                "65M",
            ),
            (
                "online_results_7region_rom_65K_case_middle",
                "65K",
            ),
        ]

    else:
        roots = [
            (
                "online_results_7region_rom_65K_case_"
                + case,
                "65K",
            )
        ]

    candidates = []

    for dirname, source in roots:
        path = os.path.join(
            ROOT,
            dirname,
            "rom_solution_7region.npz",
        )

        if not os.path.isfile(path):
            continue

        with np.load(path) as d:
            q = np.asarray(
                d["q"], dtype=float
            ).ravel().copy()

            mat = np.array([
                float(d[k].ravel()[0])
                for k in (
                    "E1", "E2", "G12"
                )
            ])

            residual = np.asarray(
                d["normalized_residual"]
            ).ravel()

            scale = d["c_scale"].copy()

        if not np.allclose(
            mat,
            CASES[case],
            atol=1e-8,
            rtol=0,
        ):
            raise RuntimeError(
                "Material mismatch: " + path
            )

        if (
            q.shape != (26,)
            or residual.shape != (26,)
            or scale.shape != (26,)
        ):
            raise RuntimeError(
                "Invalid saved solution dimensions."
            )

        if not np.all(
            np.abs(q) <= 3
        ):
            raise RuntimeError(
                "Starting interface coordinates "
                "outside [-3,3]."
            )

        candidates.append((
            rmsp(residual),
            q,
            source,
        ))

    if not candidates:
        raise FileNotFoundError(
            "No previously verified solution "
            "available for " + case
        )

    return min(
        candidates,
        key=lambda item: item[0],
    )


def dump_csv(path, rows):
    if not rows:
        return

    with open(
        path, "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# COMPLETE 65P SOLVER
# ============================================================

def solve(args):
    old = load_module(
        "previous65L",
        "65L_A_to_D_complete.py",
    )

    adapt = load_module(
        "step65O",
        "65O_adaptive_coupling.py",
    )

    mat = np.array(
        CASES[args.case],
        dtype=float,
    )

    suffix = (
        "_"
        + args.policy
        + (
            "_full"
            if args.full_patches
            else "_selective"
        )
        + (
            "_no_correction"
            if args.no_online_correction
            else ""
        )
    )

    out = os.path.join(
        ROOT,
        "online_results_7region_65P_"
        + args.case
        + suffix,
    )

    os.makedirs(
        out, exist_ok=True
    )

    manifest_path = os.path.join(
        out,
        "rom_solution_manifest.json",
    )

    if os.path.isfile(manifest_path):
        raise RuntimeError(
            "65P results already exist. "
            "Archive the folder before rerunning."
        )

    saved_rms, q, source = start_state(
        args.case
    )

    ctx = old.Coupling(
        mat, out
    )

    prior_file = os.path.join(
        ROOT,
        "rom_models_7region",
        "65N_center_correction.npz",
    )

    oracle = adapt.Oracle(
        ctx,
        out,

        prior_path=(
            None
            if args.no_center_prior
            else prior_file
        ),

        job_budget=args.jobs,
        cpus=args.cpus,

        adaptive_fidelity=(
            not args.full_patches
        ),

        online_correction=(
            not args.no_online_correction
        ),
    )

    t0 = time.perf_counter()

    # ========================================================
    # VERIFY THE INITIAL STATE
    # ========================================================

    initial = oracle.evaluate(q)

    if not initial["verified"]:
        raise RuntimeError(
            "Initial state must be fully verified."
        )

    if abs(
        initial["rms"] - saved_rms
    ) > args.repro_tol:
        raise RuntimeError(
            "Starting residual not reproducible. "
            "Expected {}, observed {}".format(
                saved_rms,
                initial["rms"],
            )
        )

    current = initial
    best = initial

    radius = float(args.radius)
    damping = 0.1

    stats = {
        k: {
            "attempts": 0,
            "successful": 0,
            "reward": 0.0,
        }

        for k in (
            "Broyden",
            "LM",
            "Newton-Krylov",
        )
    }

    Jb = None
    history = []
    it = 0

    # ========================================================
    # ADAPTIVE NONLINEAR ITERATION
    # ========================================================

    while (
        oracle.jobs + 3 <= args.jobs
        and best["rms"] > args.tol
        and it < args.max_iter
    ):
        it += 1

        q = current["q"].copy()

        fun = oracle.corrected
        r = fun(q)

        J = numerical_jacobian(
            fun, q
        )

        if Jb is None:
            Jb = J.copy()

        candidates = []

        if args.policy == "adaptive":
            methods = tuple(stats)
        else:
            methods = (args.policy,)

        # ====================================================
        # THREE SOLVER PROPOSALS
        # ====================================================

        for method in methods:
            try:
                if method == "Broyden":
                    p = -np.linalg.lstsq(
                        Jb + 1e-4 * np.eye(26),
                        current["r"],
                        rcond=1e-8,
                    )[0]

                    p = trunc(
                        p, radius
                    )

                elif method == "LM":
                    p = lm(
                        J,
                        current["r"],
                        damping,
                        radius,
                    )

                else:
                    Jrom = numerical_jacobian(
                        ctx.rom, q
                    )

                    p = nk(
                        fun,
                        ctx.rom,
                        q,
                        r,
                        Jrom,
                        radius,
                    )

                trial = np.clip(
                    q + p,
                    -3,
                    3,
                )

                if np.linalg.norm(
                    trial - q
                ) < 1e-8:
                    continue

                predicted = max(
                    0,
                    phi(r) - phi(fun(trial)),
                )

                if predicted <= 1e-10:
                    continue

                # Increase preference for methods that
                # previously improved verified residuals.
                skill = (
                    1.0
                    + min(
                        0.5,
                        stats[method]["reward"],
                    )
                )

                candidates.append((
                    predicted * skill,
                    method,
                    trial,
                    predicted,
                ))

            except (
                np.linalg.LinAlgError,
                ValueError,
                FloatingPointError,
            ) as exc:
                print(
                    "Skipping", method, exc
                )

        # ====================================================
        # NO PROMISING CHEAP DIRECTION
        # ====================================================

        if not candidates:
            radius *= 0.5

            damping = min(
                1e7,
                damping * 3,
            )

            if radius < 0.01:
                break

            continue

        _, method, trial, predicted = max(
            candidates,
            key=lambda item: item[0],
        )

        # ====================================================
        # SELECTIVE ABAQUS VERIFICATION
        # ====================================================

        previous_jobs = oracle.jobs

        result = oracle.evaluate(
            trial,
            incumbent_r=current["r"],
        )

        used = (
            oracle.jobs - previous_jobs
        )

        stats[method]["attempts"] += 1
        accepted = False

        if result["verified"]:
            actual = (
                phi(current["r"])
                - phi(result["r"])
            )

            accepted = (
                actual > 1e-11
            )

            if accepted:
                # Broyden secant update using only
                # fully verified real-FEM residuals.
                s = (
                    result["q"]
                    - current["q"]
                )

                y = (
                    result["r"]
                    - current["r"]
                )

                Jb += (
                    np.outer(
                        y - Jb @ s,
                        s,
                    )
                    / max(
                        float(s @ s),
                        1e-14,
                    )
                )

                current = result

                if (
                    result["rms"]
                    < best["rms"]
                ):
                    best = result

                radius = min(
                    1.2,
                    radius * 1.25,
                )

                damping = max(
                    1e-6,
                    damping * 0.7,
                )

                stats[method]["successful"] += 1

                stats[method]["reward"] += min(
                    0.5,
                    actual / max(used, 1),
                )

            else:
                radius = max(
                    0.01,
                    radius * 0.6,
                )

                damping = min(
                    1e7,
                    damping * 3,
                )

        else:
            # Partially evaluated trial rejected
            # by exact nonnegative SSE bound.
            radius = max(
                0.01,
                radius * 0.7,
            )

            damping = min(
                1e7,
                damping * 2,
            )

        history.append({
            "iteration": it,
            "method": method,

            "fully_verified":
                result["verified"],

            "accepted":
                accepted,

            "candidate_rms":
                result.get("rms", ""),

            "best_rms":
                best["rms"],

            "partial_lower_bound":
                result["partial_sse_lower_bound"],

            "abaqus_jobs_this_trial":
                used,

            "abaqus_jobs_total":
                oracle.jobs,

            "radius": radius,
            "damping": damping,

            "rom_calls":
                ctx.rom_calls,
        })

        print(
            "65P {:02d} {:13s} {} "
            "best={:.6f} jobs={}"
            .format(
                it,
                method,
                (
                    "ACCEPT"
                    if accepted
                    else "REJECT"
                ),
                best["rms"],
                oracle.jobs,
            )
        )

        if (
            radius <= 0.01
            and not accepted
            and it >= 6
        ):
            break

    # ========================================================
    # FINAL FULLY VERIFIED RESULT
    # ========================================================

    dump_csv(
        os.path.join(
            out,
            "65P_adaptive_history.csv",
        ),
        history,
    )

    arrays = {
        "q": best["q"],
        "c_scale": ctx.scale,
        "normalized_residual": best["r"],
        "rms_residual":
            np.array([best["rms"]]),
        "E1": np.array([mat[0]]),
        "E2": np.array([mat[1]]),
        "G12": np.array([mat[2]]),
    }

    coeff = ctx.coeffs(
        best["q"]
    )

    for name in NAMES:
        arrays["c_" + name] = coeff[name]

    np.savez(
        os.path.join(
            out,
            "rom_solution_7region.npz",
        ),
        **arrays
    )

    final_dir = os.path.join(
        out, "final"
    )

    os.makedirs(
        final_dir,
        exist_ok=True,
    )

    # Copy only the ODBs corresponding
    # to the BEST fully verified state.
    for patch in (
        "left", "center", "right"
    ):
        record = best["files"][patch]

        for key, ext in (
            ("odb_file", ".odb"),
            ("input_file", ".inp"),
        ):
            src = record[key]

            if not os.path.isfile(src):
                raise FileNotFoundError(src)

            shutil.copy2(
                src,
                os.path.join(
                    final_dir,
                    patch + ext,
                ),
            )

    manifest = {
        "method":
            "Adaptive Broyden/LM/Newton-Krylov "
            "with progressive Abaqus verification",

        "case": args.case,
        "policy": args.policy,

        "adaptive_fidelity":
            not args.full_patches,

        "online_correction":
            not args.no_online_correction,

        "material_MPa": mat.tolist(),
        "start_source": source,

        "initial_verified_rms":
            initial["rms"],

        "final_high_fidelity_rms":
            best["rms"],

        "converged":
            bool(
                best["rms"] <= args.tol
            ),

        "high_fidelity_abaqus_jobs":
            oracle.jobs,

        "full_verified_evaluations":
            sum(
                int(x["verified"])
                for x in oracle.log
            ),

        "partial_rejections":
            sum(
                not x["verified"]
                for x in oracle.log
            ),

        "rom_residual_evaluations":
            ctx.rom_calls,

        "solver_statistics": stats,

        "total_seconds":
            time.perf_counter() - t0,

        "note":
            "Only full three-patch Abaqus "
            "evaluations count for convergence.",
    }

    with open(
        manifest_path,
        "w",
    ) as f:
        json.dump(
            manifest, f, indent=2
        )

    print(
        json.dumps(manifest, indent=2)
    )

    if not manifest["converged"]:
        raise SystemExit(2)


# ============================================================
# COMMAND-LINE OPTIONS
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--case",
        choices=list(CASES),
        default="middle",
    )

    p.add_argument(
        "--cpus",
        type=int,
        default=4,
    )

    p.add_argument(
        "--policy",
        choices=[
            "adaptive",
            "Broyden",
            "LM",
            "Newton-Krylov",
        ],
        default="adaptive",
    )

    p.add_argument(
        "--full-patches",
        action="store_true",
    )

    p.add_argument(
        "--no-center-prior",
        action="store_true",
    )

    p.add_argument(
        "--no-online-correction",
        action="store_true",
    )

    p.add_argument(
        "--jobs",
        type=int,
        default=36,
    )

    p.add_argument(
        "--radius",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--tol",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--max_iter",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repro_tol",
        type=float,
        default=0.015,
    )

    args = p.parse_args()

    if args.jobs < 3:
        raise ValueError(
            "At least three Abaqus jobs required."
        )

    solve(args)


if __name__ == "__main__":
    main()
