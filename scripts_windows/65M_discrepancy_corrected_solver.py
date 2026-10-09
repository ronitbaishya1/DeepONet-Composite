
"""
STEP 65M — DISCREPANCY-CORRECTED FE–NO–ROM SOLVER

65M-A: Compare Broyden and LM interface states.
65M-B: Verify Broyden starting state with Abaqus.
65M-C: Build local ROM discrepancy correction.
65M-D: Safeguarded, discrepancy-aware LM optimization.
65M-E: Export center FEM enrichment candidates.

Requires existing:
    scripts/65L_A_to_D_complete.py

Run:
    abaqus python scripts/65M_discrepancy_corrected_solver.py --stage audit
    abaqus python scripts/65M_discrepancy_corrected_solver.py --stage solve
    abaqus python scripts/65M_discrepancy_corrected_solver.py --stage enrich

Original 64T, 65H, 65K and 65L results are not overwritten.
"""

from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np


# ============================================================
# PATHS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PREVIOUS = os.path.join(
    HERE,
    "65L_A_to_D_complete.py",
)

BROYDEN = os.path.join(
    ROOT,
    "online_results_7region_rom_65K_case_middle",
    "rom_solution_7region.npz",
)

OLD_LM = os.path.join(
    ROOT,
    "online_results_7region_65L_middle_test",
    "rom_solution_7region.npz",
)

OUT = os.path.join(
    ROOT,
    "online_results_7region_65M_middle",
)


NAMES = (
    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
)

COUNTS = (4, 4, 5, 5, 4, 4)


# ============================================================
# FILE HELPERS
# ============================================================

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def table(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def load_65l():
    if not os.path.isfile(PREVIOUS):
        raise FileNotFoundError(
            "The complete Step 65L script is required: "
            + PREVIOUS
        )

    spec = importlib.util.spec_from_file_location(
        "prior_step_65L",
        PREVIOUS,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD AND VALIDATE THE TWO OLD SOLUTIONS
# ============================================================

def load_pair():

    for path in (BROYDEN, OLD_LM):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Missing previous solution: " + path
            )

    a = np.load(BROYDEN)
    b = np.load(OLD_LM)

    try:
        material_a = np.array([
            float(a[k][0])
            for k in ("E1", "E2", "G12")
        ])

        material_b = np.array([
            float(b[k][0])
            for k in ("E1", "E2", "G12")
        ])

        if not np.allclose(
            material_a,
            material_b,
            rtol=0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Material properties differ between solutions."
            )

        if not np.allclose(
            a["c_scale"],
            b["c_scale"],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise RuntimeError(
                "PCA scaling differs between solutions."
            )

        if (
            len(a["q"]) != 26
            or len(b["q"]) != 26
        ):
            raise RuntimeError(
                "Expected 26 interface coordinates."
            )

        if (
            len(a["normalized_residual"]) != 26
            or len(b["normalized_residual"]) != 26
        ):
            raise RuntimeError(
                "Expected 26-component residual vectors."
            )

        for solution in (a, b):

            coefficients = np.concatenate([
                solution["c_" + name].ravel()
                for name in NAMES
            ])

            reconstructed = (
                solution["q"]
                * solution["c_scale"]
            )

            if not np.allclose(
                coefficients,
                reconstructed,
                rtol=1e-7,
                atol=1e-10,
            ):
                raise RuntimeError(
                    "Saved PCA coefficients do not match q."
                )

        return (
            material_a,
            np.array(a["q"]),
            np.array(b["q"]),
            np.array(a["normalized_residual"]),
            np.array(b["normalized_residual"]),
            np.array(a["c_scale"]),
        )

    finally:
        a.close()
        b.close()


def rms(x):
    x = np.asarray(x, dtype=float)

    return float(
        np.sqrt(np.mean(x * x))
    )


def objective(r):
    return 0.5 * float(np.dot(r, r))


# ============================================================
# STEP 65M-A — AUDIT PREVIOUS SOLUTIONS
# ============================================================

def audit():

    mkdir(OUT)

    (
        material,
        q_broyden,
        q_lm,
        r_broyden,
        r_lm,
        scale,
    ) = load_pair()

    rows = []
    offset = 0

    for interface, count in zip(NAMES, COUNTS):

        for j in range(count):

            k = offset + j

            rows.append({
                "interface": interface,
                "mode": j + 1,

                "Broyden_q": q_broyden[k],
                "Old_LM_q": q_lm[k],

                "delta_q":
                    q_lm[k] - q_broyden[k],

                "Broyden_c":
                    q_broyden[k] * scale[k],

                "Old_LM_c":
                    q_lm[k] * scale[k],

                "Broyden_normalized_force_residual":
                    r_broyden[k],

                "Old_LM_normalized_force_residual":
                    r_lm[k],
            })

        offset += count

    table(
        os.path.join(
            OUT,
            "65M_A_interface_state_comparison.csv",
        ),
        rows,
    )

    summary = {
        "material_MPa": material.tolist(),

        "Broyden_hf_rms":
            rms(r_broyden),

        "Old_LM_hf_rms":
            rms(r_lm),

        "q_difference_l2":
            float(
                np.linalg.norm(
                    q_lm - q_broyden
                )
            ),

        "initialization":
            "verified_65K_Broyden",

        "note":
            "Uses saved results; no extra Abaqus jobs.",
    }

    dump(
        os.path.join(
            OUT, "65M_A_audit.json"
        ),
        summary,
    )

    print(json.dumps(summary, indent=2))


# ============================================================
# STEP 65M-C — LOCAL ROM DISCREPANCY MODEL
# ============================================================

class AffineCorrection:

    """
    Fit:

        delta(q) = delta0 + B @ (q - q0)

    Every evaluated high-fidelity point contributes,
    including previously rejected trial states.

    Regularization prevents an unstable 26x26 fit
    from a small number of Abaqus observations.
    """

    def __init__(
        self,
        samples,
        ridge=0.05,
        length=1.0,
    ):

        anchor = min(
            samples,
            key=lambda item: item["hf_rms"],
        )

        x0 = anchor["q"]
        y0 = anchor["delta"]

        points = [
            p for p in samples
            if p is not anchor
        ]

        B = np.zeros((26, 26), dtype=float)

        if points:

            D = np.vstack([
                p["q"] - x0
                for p in points
            ])

            Y = np.vstack([
                p["delta"] - y0
                for p in points
            ])

            distances = np.linalg.norm(
                D,
                axis=1,
            )

            weights = np.maximum(
                0.08,
                np.exp(
                    -0.5
                    * (distances / length) ** 2
                ),
            )

            DW = (
                D * weights[:, None] ** 0.5
            )

            YW = (
                Y * weights[:, None] ** 0.5
            )

            alpha = np.linalg.solve(
                DW.dot(DW.T)
                + ridge * np.eye(len(DW)),
                YW,
            )

            B = (
                DW.T.dot(alpha)
            ).T

            # Limit excessive discrepancy slopes.
            norm = np.linalg.norm(B, 2)

            if norm > 3.0:
                B *= 3.0 / norm

        self.x0 = x0.copy()
        self.y0 = y0.copy()
        self.B = B

    def correction(self, q):

        return (
            self.y0
            + self.B.dot(
                np.asarray(q) - self.x0
            )
        )

    def model(self, q, rom_function):

        return (
            rom_function(q)
            + self.correction(q)
        )


# ============================================================
# ROM JACOBIAN — NO ABAQUS REQUIRED
# ============================================================

def jacobian(fun, q, step=1e-3):

    base = fun(q)

    J = np.empty(
        (26, 26),
        dtype=float,
    )

    for j in range(26):

        d = max(
            step,
            step * abs(q[j]),
        )

        x = q.copy()

        x[j] = np.clip(
            x[j] + d,
            -3.0,
            3.0,
        )

        if x[j] == q[j]:

            x[j] = np.clip(
                q[j] - d,
                -3.0,
                3.0,
            )

        if x[j] == q[j]:
            raise RuntimeError(
                "Cannot perturb q component {}".format(j)
            )

        J[:, j] = (
            fun(x) - base
        ) / (
            x[j] - q[j]
        )

    return J


# ============================================================
# TRUST-REGION LM STEP
# ============================================================

def lm_direction(
    J,
    r,
    damping,
    radius,
):

    JTJ = J.T.dot(J)

    diagonal = np.maximum(
        np.diag(JTJ),
        1e-5,
    )

    A = (
        JTJ
        + damping * np.diag(diagonal)
    )

    p = -np.linalg.solve(
        A,
        J.T.dot(r),
    )

    norm = np.linalg.norm(p)

    if norm > radius:
        p = p * radius / norm

    return p


# ============================================================
# STORE EVERY HIGH-FIDELITY OBSERVATION
# ============================================================

def save_observation(out, point):

    idx = point["id"]

    destination = os.path.join(
        out, "observations"
    )

    mkdir(destination)

    fields = {
        "q": point["q"],
        "hf_r": point["hf_r"],
        "rom_r": point["rom_r"],
        "delta": point["delta"],
    }

    for name in NAMES:

        fields[
            "Abaqus_g_" + name
        ] = point["hf_forces"][name]

        fields[
            "ROM_g_" + name
        ] = point["rom_forces"][name]

    np.savez(
        os.path.join(
            destination,
            "sample_{:03d}.npz".format(idx),
        ),
        **fields
    )

    dump(
        os.path.join(
            destination,
            "sample_{:03d}.json".format(idx),
        ),
        {
            "sample_id": idx,
            "hf_rms": point["hf_rms"],
            "rom_rms": rms(point["rom_r"]),
            "delta_rms": rms(point["delta"]),
            "center_odb": point["center_odb"],
            "accepted": point["accepted"],
        },
    )


# ============================================================
# STEP 65M-B — ROM AND ABAQUS AT THE SAME STATE
# ============================================================

def observe(
    ctx,
    q,
    accepted,
    points,
    cpus,
):

    rom_r, _, rom_forces, _ = ctx.rom(
        q,
        detailed=True,
    )

    verified = ctx.hf(q, cpus)

    point = {
        "id": verified["eval_id"],
        "q": q.copy(),

        "hf_r": verified["r"].copy(),
        "rom_r": rom_r.copy(),

        "delta":
            verified["r"].copy() - rom_r,

        "hf_rms": verified["rms"],

        "hf_forces":
            verified["forces"],

        "rom_forces":
            rom_forces,

        "accepted": accepted,

        "center_odb":
            verified["files"]["center"]["odb_file"],
    }

    points.append(point)

    save_observation(
        ctx.out_dir,
        point,
    )

    return point, verified


# ============================================================
# STEPS 65M-B/C/D — COMPLETE COUPLED SOLVER
# ============================================================

def solve(
    max_hf,
    cpus,
    tol,
    radius0,
    ridge,
    repeat_limit,
    max_rom_calls,
):

    mkdir(OUT)

    audit()

    (
        material,
        start,
        _,
        saved_r,
        _,
        scales,
    ) = load_pair()

    previous = load_65l()

    ctx = previous.Coupling(
        material,
        OUT,
    )

    if not np.allclose(
        ctx.scale,
        scales,
        atol=1e-15,
        rtol=1e-12,
    ):
        raise RuntimeError(
            "ROM runtime uses different PCA scaling."
        )

    start_time = time.perf_counter()

    points = []

    print("")
    print(
        "VERIFYING ORIGINAL 65K BROYDEN STATE"
    )

    first, first_hf = observe(
        ctx,
        start,
        True,
        points,
        cpus,
    )

    repeat_gap = rms(
        first["hf_r"] - saved_r
    )

    if repeat_gap > repeat_limit:

        raise RuntimeError(
            "65K residual not reproducible: "
            "RMS gap={:.6f} > {:.6f}"
            .format(
                repeat_gap,
                repeat_limit,
            )
        )

    current = first_hf
    best = first_hf

    accepted = 0
    rejected = 0

    radius = radius0
    damping = 0.08

    history = [{
        "evaluation": 0,
        "accepted": True,

        "true_rms":
            current["rms"],

        "best_true_rms":
            best["rms"],

        "predicted_rms":
            rms(first["rom_r"]),

        "discrepancy_rms":
            rms(first["delta"]),

        "radius": radius,
        "damping": damping,
        "ratio": "",
    }]

    # ========================================================
    # ITERATIVE DISCREPANCY-AWARE CORRECTION
    # ========================================================

    while (
        ctx.hf_calls < max_hf
        and best["rms"] > tol
    ):

        correction = AffineCorrection(
            points,
            ridge=ridge,
        )

        def corrected_model(x):
            return correction.model(
                x,
                ctx.rom,
            )

        q = current["q"].copy()

        rhat = corrected_model(q)

        J = jacobian(
            corrected_model,
            q,
        )

        trial = None
        predicted_improvement = 0.0

        # Cheap trial search before launching Abaqus.
        for cheap_try in range(9):

            p = lm_direction(
                J,
                rhat,
                damping,
                radius,
            )

            q_try = np.clip(
                q + p,
                -3.0,
                3.0,
            )

            if np.linalg.norm(
                q_try - q
            ) < 1e-7:

                radius *= 0.5
                damping *= 3.0

                continue

            predicted = corrected_model(
                q_try
            )

            predicted_improvement = (
                objective(rhat)
                - objective(predicted)
            )

            if predicted_improvement > 1e-10:

                trial = (
                    q_try,
                    predicted,
                )

                break

            radius *= 0.5
            damping *= 3.0

        if trial is None:

            print(
                "No predicted descent: "
                "terminating safely."
            )

            break

        q_try, predicted = trial

        # Real Abaqus verification.
        candidate, verified = observe(
            ctx,
            q_try,
            False,
            points,
            cpus,
        )

        true_improvement = (
            objective(current["r"])
            - objective(verified["r"])
        )

        ratio = (
            true_improvement
            / max(
                predicted_improvement,
                1e-12,
            )
        )

        improvement = (
            true_improvement > 1e-9
            and ratio > 0.01
        )

        if improvement:

            # Accept trial.
            current = verified

            candidate["accepted"] = True

            save_observation(
                ctx.out_dir,
                candidate,
            )

            accepted += 1

            radius = min(
                1.5,
                radius * 1.25,
            )

            damping = max(
                1e-5,
                damping * 0.75,
            )

        else:

            # Reject the new state, but retain its
            # Abaqus force information for correction.
            rejected += 1

            radius = max(
                0.015,
                radius * 0.6,
            )

            damping = min(
                1e6,
                damping * 3.0,
            )

        if verified["rms"] < best["rms"]:
            best = verified

        history.append({
            "evaluation":
                verified["eval_id"],

            "accepted":
                improvement,

            "true_rms":
                verified["rms"],

            "best_true_rms":
                best["rms"],

            "predicted_rms":
                rms(predicted),

            "discrepancy_rms":
                rms(candidate["delta"]),

            "radius": radius,
            "damping": damping,
            "ratio": ratio,
        })

        print(
            "HF {:02d}: true RMS={:.6f}, "
            "best={:.6f}, {} | radius={:.4f}"
            .format(
                verified["eval_id"],
                verified["rms"],
                best["rms"],
                (
                    "ACCEPT"
                    if improvement
                    else "REJECT"
                ),
                radius,
            )
        )

        if ctx.rom_calls >= max_rom_calls:

            print(
                "ROM evaluation budget exhausted."
            )

            break

        if (
            radius <= 0.015
            and rejected >= 3
        ):

            print(
                "Trust region too small. "
                "Stopping safely."
            )

            break

    # ========================================================
    # SAVE HISTORY AND BEST VERIFIED SOLUTION
    # ========================================================

    table(
        os.path.join(
            OUT,
            "65M_CD_high_fidelity_history.csv",
        ),
        history,
    )

    final_coeffs = ctx.coeffs(
        best["q"]
    )

    archive = {
        "q": best["q"],
        "c_scale": scales,

        "normalized_residual":
            best["r"],

        "physical_residual":
            best["raw"],

        "rms_residual":
            np.array([best["rms"]]),

        "E1":
            np.array([material[0]]),

        "E2":
            np.array([material[1]]),

        "G12":
            np.array([material[2]]),
    }

    for name in NAMES:
        archive[
            "c_" + name
        ] = final_coeffs[name]

    np.savez(
        os.path.join(
            OUT,
            "rom_solution_7region.npz",
        ),
        **archive
    )

    final_dir = os.path.join(
        OUT, "final"
    )

    mkdir(final_dir)

    # Save actual Abaqus patch files corresponding
    # to the BEST verified interface state.
    for patch in (
        "left", "center", "right"
    ):

        result = best["files"][patch]

        for name, ext in (
            ("odb_file", ".odb"),
            ("input_file", ".inp"),
        ):

            source = result[name]

            if not os.path.isfile(source):
                raise FileNotFoundError(
                    "Missing final FEM file: "
                    + source
                )

            shutil.copy2(
                source,
                os.path.join(
                    final_dir,
                    patch + ext,
                ),
            )

    manifest = {
        "method":
            "ROM + low-rank discrepancy "
            "+ verified trust-region LM",

        "material_MPa":
            material.tolist(),

        "initialization":
            "original_65K_middle_Broyden_solution",

        "initial_HF_RMS":
            first["hf_rms"],

        "initial_Broyden_reproduction_gap_RMS":
            repeat_gap,

        "final_high_fidelity_rms":
            best["rms"],

        "final_high_fidelity_max":
            best["maximum"],

        "converged":
            bool(best["rms"] <= tol),

        "high_fidelity_residual_evaluations":
            ctx.hf_calls,

        "high_fidelity_abaqus_jobs":
            3 * ctx.hf_calls,

        "rom_residual_evaluations":
            ctx.rom_calls,

        "accepted_trial_steps":
            accepted,

        "rejected_trial_steps":
            rejected,

        "best_high_fidelity_evaluation":
            best["eval_id"],

        "total_seconds":
            time.perf_counter() - start_time,

        "tolerance":
            tol,

        "center_enrichment_candidate_directory":
            os.path.join(
                OUT,
                "observations",
            ),
    }

    dump(
        os.path.join(
            OUT,
            "rom_solution_manifest.json",
        ),
        manifest,
    )

    print("")
    print("STEP 65M RESULT")
    print(json.dumps(manifest, indent=2))

    # Failure is not silently reported as success.
    if not manifest["converged"]:
        sys.exit(2)


# ============================================================
# STEP 65M-E — EXPORT CENTER ROM ENRICHMENT CANDIDATES
# ============================================================

def enrich():

    observations = os.path.join(
        OUT,
        "observations",
    )

    if not os.path.isdir(observations):
        raise FileNotFoundError(
            "Run --stage solve first."
        )

    (
        material,
        _,
        _,
        _,
        _,
        scale,
    ) = load_pair()

    entries = []

    for filename in sorted(
        os.listdir(observations)
    ):

        if not filename.endswith(".npz"):
            continue

        ident = filename.rsplit(
            ".", 1
        )[0]

        data_path = os.path.join(
            observations,
            filename,
        )

        note = os.path.join(
            observations,
            ident + ".json",
        )

        with open(note) as f:
            meta = json.load(f)

        with np.load(data_path) as data:

            q = np.array(data["q"])
            c = q * scale

            # Center branch has:
            # 3 material properties
            # 5 center-left PCA coefficients
            # 5 center-right PCA coefficients
            # = 13 inputs.

            x = np.concatenate([
                material,
                c[8:13],
                c[13:18],
            ])

            rom_g = np.concatenate([
                data["ROM_g_center_left"],
                data["ROM_g_center_right"],
            ])

            fe_g = np.concatenate([
                data["Abaqus_g_center_left"],
                data["Abaqus_g_center_right"],
            ])

            row = {
                "sample_id":
                    meta["sample_id"],

                "accepted":
                    meta["accepted"],

                "HF_RMS":
                    meta["hf_rms"],

                "center_force_gap_l2":
                    float(
                        np.linalg.norm(
                            fe_g - rom_g
                        )
                    ),

                "center_odb_path":
                    meta["center_odb"],
            }

            for i, value in enumerate(x):
                row[
                    "input_{:02d}".format(i)
                ] = float(value)

            for i, value in enumerate(fe_g):
                row[
                    "Abaqus_g_{:02d}".format(i)
                ] = float(value)

            for i, value in enumerate(rom_g):
                row[
                    "ROM_g_{:02d}".format(i)
                ] = float(value)

            entries.append(row)

    entries.sort(
        key=lambda row:
            row["center_force_gap_l2"],
        reverse=True,
    )

    table(
        os.path.join(
            OUT,
            "65M_E_center_enrichment_candidates.csv",
        ),
        entries,
    )

    dump(
        os.path.join(
            OUT,
            "65M_E_enrichment_notes.json",
        ),
        {
            "number_candidates":
                len(entries),

            "input_dimension": 13,

            "target_generalized_force_dimension": 10,

            "status":
                "Candidate-only; original 65E "
                "center model unchanged.",

            "next_action":
                "Use the existing 65D and 65E "
                "snapshot pipeline for a controlled "
                "center-ROM retraining experiment.",
        },
    )

    print("")
    print(
        "Center enrichment candidates exported:",
        len(entries),
    )

    print(
        "The original 65E center ROM "
        "has NOT been retrained."
    )


# ============================================================
# SYNTHETIC SELF-TEST
# ============================================================

def selftest():

    r0 = np.arange(26) / 400.0
    x0 = np.zeros(26)

    Btrue = 0.14 * np.eye(26)

    points = [{
        "q": x0,
        "delta": r0,
        "hf_rms": 0.1,
    }]

    for i in range(5):

        x = np.zeros(26)
        x[i] = 0.2

        points.append({
            "q": x,
            "delta":
                r0 + Btrue.dot(x),
            "hf_rms": 0.2,
        })

    fit = AffineCorrection(
        points,
        ridge=1e-7,
    )

    test = np.zeros(26)
    test[1] = 0.12

    error = np.linalg.norm(
        fit.correction(test)
        - (
            r0
            + Btrue.dot(test)
        )
    )

    if error >= 1e-5:
        raise RuntimeError(
            "Discrepancy model test failed."
        )

    J = np.eye(26)
    r = np.ones(26) * 0.1

    p = lm_direction(
        J,
        r,
        0.1,
        0.8,
    )

    if not (
        objective(r + p)
        < objective(r)
    ):
        raise RuntimeError(
            "LM test failed."
        )

    print(
        "65M synthetic algebra self-test passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "solve",
            "enrich",
            "selftest",
        ),
        required=True,
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_hf",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--reproduction_tol",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--max_rom_calls",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    if args.stage == "audit":

        audit()

    elif args.stage == "solve":

        # Prevent accidental mixing with a previous run.
        manifest_path = os.path.join(
            OUT,
            "rom_solution_manifest.json",
        )

        if os.path.isfile(manifest_path):
            raise RuntimeError(
                "Step 65M result already exists. "
                "Archive or rename the output folder "
                "before running again."
            )

        solve(
            args.max_hf,
            args.cpus,
            args.tol,
            args.radius,
            args.ridge,
            args.reproduction_tol,
            args.max_rom_calls,
        )

    elif args.stage == "enrich":

        enrich()

    elif args.stage == "selftest":

        selftest()


if __name__ == "__main__":
    main()
