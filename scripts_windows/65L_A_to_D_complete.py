"""Step 65L: ROM/FE diagnostic, warm start, guarded LM solve, benchmarks.

Run with: abaqus python scripts\65L_A_to_D_complete.py --action <...>
Uses existing hybrid_runtime_7region.py + rom_runtime_7region.py
and does NOT change any original 64T/65H outputs.
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

NAMES = ["left_outer", "left_inner", "center_left", "center_right",
         "right_inner", "right_outer"]
PATCH_NAMES = ["left", "center", "right"]
PATCH_INTERFACES = {
    "left": ["left_outer", "left_inner"],
    "center": ["center_left", "center_right"],
    "right": ["right_inner", "right_outer"],
}
MATERIAL_CASES = [
    ("soft", 42000.0, 11700.0, 4250.0),
    ("middle", 46500.0, 12500.0, 4750.0),
    ("stiff", 49000.0, 10900.0, 4500.0),
]
BASELINE_SOLUTION = os.path.join(
    BASE_DIR, "online_results_7region_rom", "rom_solution_7region.npz")


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def write_rows(path, rows, columns=None):
    if not rows:
        return
    if columns is None:
        columns = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_configuration():
    if not os.path.isfile(BASELINE_SOLUTION):
        raise FileNotFoundError("Need existing 65H baseline: " + BASELINE_SOLUTION)
    data = np.load(BASELINE_SOLUTION)
    scale = np.asarray(data["c_scale"], dtype=float).ravel()
    counts = [len(np.asarray(data["c_" + name]).ravel()) for name in NAMES]
    if len(scale) != 26 or sum(counts) != 26:
        raise RuntimeError("65H baseline must contain 26 coefficients")
    slices = {}
    offset = 0
    for name, count in zip(NAMES, counts):
        slices[name] = slice(offset, offset + count)
        offset += count
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise RuntimeError("Invalid saved 65H interface scaling")
    return scale, counts, slices


def read_design_training_anchors():
    """Join real global TRAIN-only design rows across all three patches."""
    design_dir = os.path.join(BASE_DIR, "rom_designs_7region")
    per_patch = {}
    for patch in PATCH_NAMES:
        filename = os.path.join(design_dir, patch + "_fe_rom_design.csv")
        if not os.path.isfile(filename):
            raise FileNotFoundError("65A design CSV required: " + filename)
        with open(filename, "r", newline="") as handle:
            table = list(csv.DictReader(handle))
        per_patch[patch] = {
            int(row["GlobalIndex"]): row for row in table
            if row["Source"].strip().lower() == "real"
            and row["Split"].strip().lower() == "train"
        }
    common = set(per_patch["left"])
    common.intersection_update(per_patch["center"])
    common.intersection_update(per_patch["right"])
    if len(common) != 35:
        raise RuntimeError("Expected 35 common real training anchors, got {}".format(len(common)))
    _, counts, _ = load_configuration()
    mode_counts = dict(zip(NAMES, counts))
    results = []
    for global_index in sorted(common):
        left_row = per_patch["left"][global_index]
        material = np.array([float(left_row[k]) for k in
                             ("E1_MPa", "E2_MPa", "G12_MPa")], dtype=float)
        cdict = {}
        for patch in PATCH_NAMES:
            row = per_patch[patch][global_index]
            other = np.array([float(row[k]) for k in
                              ("E1_MPa", "E2_MPa", "G12_MPa")], dtype=float)
            if not np.allclose(material, other, rtol=0, atol=1e-5):
                raise RuntimeError("Inconsistent material at global index {}".format(global_index))
            for name in PATCH_INTERFACES[patch]:
                vals = np.array([float(row["c_{}_{:02d}".format(name, j+1)])
                                 for j in range(mode_counts[name])], dtype=float)
                if name in cdict and not np.allclose(vals, cdict[name]):
                    raise RuntimeError("Inconsistent interface {}".format(name))
                cdict[name] = vals
        c = np.concatenate([cdict[name] for name in NAMES])
        results.append((global_index, material, c))
    return results


def nearest_anchor(material):
    anchors = read_design_training_anchors()
    # Normalize by full material variation ranges, not by unit of MPa.
    width = np.array([9000.0, 2400.0, 900.0])
    ranked = sorted(anchors, key=lambda a:
                    float(np.linalg.norm((a[1] - material) / width)))
    index, nearest_mat, c = ranked[0]
    scales, _, _ = load_configuration()
    return {
        "global_training_index": int(index),
        "anchor_material_MPa": nearest_mat.tolist(),
        "material_distance_in_range_units": float(
            np.linalg.norm((nearest_mat - material) / width)),
        "q": (c / scales).tolist(),
    }


class Coupling:
    """Exact 65H interface contract; fails explicitly if runtime differs."""
    def __init__(self, material, out_dir):
        # Import only here so --action init and --action selftest do not
        # require a license or neural-operator model load.
        from hybrid_runtime_7region import (
            run_patch_job, evaluate_neural_operators,
            get_residual_force_scales)
        from rom_runtime_7region import predict_rom_generalized_force
        self.run_patch_job = run_patch_job
        self.eval_no = evaluate_neural_operators
        self.eval_rom_patch = predict_rom_generalized_force
        self.material = np.asarray(material, dtype=float)
        self.out_dir = out_dir
        self.scale, self.counts, self.slices = load_configuration()
        scales = get_residual_force_scales()
        self.force_scale = np.concatenate([
            np.asarray(scales[name], dtype=float).ravel() for name in NAMES])
        if len(self.force_scale) != 26 or np.any(self.force_scale <= 0):
            raise RuntimeError("Incorrect 26-component force normalization")
        self.rom_calls = 0
        self.hf_calls = 0
        self.hf_cache = {}
        self.last_ood = {}

    def coeffs(self, q):
        c = np.asarray(q, dtype=float) * self.scale
        return {name: c[self.slices[name]].copy() for name in NAMES}

    def branch(self, patch, coeffs):
        return np.concatenate([self.material] +
                              [coeffs[name] for name in PATCH_INTERFACES[patch]])

    def split_patch(self, patch, g):
        g = np.asarray(g, dtype=float).ravel()
        expected = sum(self.counts[NAMES.index(name)]
                       for name in PATCH_INTERFACES[patch])
        if len(g) != expected:
            raise RuntimeError("{} force dimension {} != {}".format(
                patch, len(g), expected))
        result = {}
        offset = 0
        for name in PATCH_INTERFACES[patch]:
            count = self.counts[NAMES.index(name)]
            result[name] = g[offset:offset+count].copy()
            offset += count
        return result

    def rom_patch_forces(self, q):
        coeff = self.coeffs(q)
        result = {}
        for patch in PATCH_NAMES:
            forces = self.eval_rom_patch(patch, self.branch(patch, coeff))
            result.update(self.split_patch(patch, forces))
        return result

    def normalized_residual(self, patch_g, no_g):
        raw = np.concatenate([
            np.asarray(patch_g[name], dtype=float)
            + np.asarray(no_g[name], dtype=float)
            for name in NAMES])
        r = raw / self.force_scale
        if r.shape != (26,) or not np.all(np.isfinite(r)):
            raise RuntimeError("Nonfinite coupling residual")
        return r, raw

    def rom(self, q, detailed=False):
        self.rom_calls += 1
        q = np.asarray(q, dtype=float)
        coeff = self.coeffs(q)
        pg = self.rom_patch_forces(q)
        ng = self.eval_no(float(self.material[0]), float(self.material[1]),
                          float(self.material[2]), coeff)
        r, raw = self.normalized_residual(pg, ng)
        if detailed:
            return r, raw, pg, ng
        return r

    def hf(self, q, cpus=4):
        q = np.asarray(q, dtype=float)
        key = tuple(np.round(q, decimals=12))
        if key in self.hf_cache:
            return self.hf_cache[key]
        eval_id = self.hf_calls
        self.hf_calls += 1
        eval_dir = os.path.join(self.out_dir, "hf_eval_{:03d}".format(eval_id))
        mkdir(eval_dir)
        coeff = self.coeffs(q)
        pg = {}
        files = {}
        for patch in PATCH_NAMES:
            # EXACT signature used by the working original Step 65H.
            result = self.run_patch_job(
                patch_name=patch,
                E1=float(self.material[0]),
                E2=float(self.material[1]),
                G12=float(self.material[2]),
                coefficient_dictionary=coeff,
                job_directory=eval_dir,
                job_name="l65_{:03d}_{}".format(eval_id, patch),
                cpus=cpus,
            )
            pg.update(result["generalized_forces"])
            files[patch] = result
        no_g = self.eval_no(float(self.material[0]), float(self.material[1]),
                            float(self.material[2]), coeff)
        r, raw = self.normalized_residual(pg, no_g)
        result = {
            "q": q.copy(), "r": r, "raw": raw,
            "rms": float(np.sqrt(np.mean(r*r))),
            "maximum": float(np.max(np.abs(r))),
            "forces": pg, "no_forces": no_g, "files": files,
            "eval_id": eval_id,
        }
        self.hf_cache[key] = result
        save_json(os.path.join(eval_dir, "force_audit.json"), {
            "q": q.tolist(), "rms": result["rms"],
            "maximum": result["maximum"],
            "r_normalized": r.tolist(),
        })
        print("REAL FEM {:03d} RMS={:.8f} max={:.8f}".format(
            eval_id, result["rms"], result["maximum"]))
        return result

    def finite_difference_rom_jacobian(self, q, step=1e-3):
        q = np.asarray(q, dtype=float)
        r0 = self.rom(q)
        jac = np.empty((26, 26), dtype=float)
        for i in range(26):
            delta = step * max(1.0, abs(q[i]))
            trial = q.copy()
            trial[i] = min(3.0, max(-3.0, trial[i] + delta))
            if trial[i] == q[i]:
                trial[i] = max(-3.0, q[i] - delta)
            dx = trial[i] - q[i]
            if abs(dx) < 1e-12:
                raise RuntimeError("Cannot perturb q component {}".format(i))
            jac[:, i] = (self.rom(trial) - r0) / dx
        return jac


def rms(r):
    return float(np.sqrt(np.mean(np.asarray(r, dtype=float)**2)))


def lm_step(J, residual, damping, step_radius=0.7):
    matrix = J.T.dot(J)
    diagonal = np.diag(matrix)
    regularizer = damping * np.diag(np.maximum(diagonal, 1e-4))
    direction = -np.linalg.solve(matrix + regularizer,
                                  J.T.dot(residual))
    length = float(np.linalg.norm(direction))
    if length > step_radius:
        direction *= step_radius / length
    return np.clip(direction, -0.45, 0.45)


def phi(r):
    return 0.5 * float(np.dot(r, r))


def run_rom_lm(ctx, seeds, max_iter, target, label):
    ranked = []
    for source, q in seeds:
        q = np.clip(np.asarray(q, dtype=float).ravel(), -3.0, 3.0)
        if q.shape != (26,):
            raise RuntimeError("Wrong initial q size from " + source)
        r = ctx.rom(q)
        ranked.append((rms(r), source, q))
    ranked.sort(key=lambda x: x[0])
    print("ROM seed ranking:", [(a, b) for a, b, _ in ranked])
    current_rms, chosen, q = ranked[0]
    best_q = q.copy()
    best_r = ctx.rom(q)
    current_r = best_r.copy()
    damping = 1e-2
    radius = 0.8
    history = [{"iter": 0, "seed": chosen, "rms": current_rms,
                "accepted": True, "lambda": damping, "radius": radius}]
    for k in range(1, max_iter+1):
        if rms(current_r) <= target:
            break
        J = ctx.finite_difference_rom_jacobian(q)
        step = lm_step(J, current_r, damping, radius)
        trial = np.clip(q + step, -3.0, 3.0)
        trial_r = ctx.rom(trial)
        accepted = phi(trial_r) < phi(current_r) - 1e-12
        if accepted:
            q = trial
            current_r = trial_r
            damping = max(1e-8, damping*0.65)
            radius = min(2.0, radius*1.2)
            if phi(current_r) < phi(best_r):
                best_q, best_r = q.copy(), current_r.copy()
        else:
            damping = min(1e8, damping*5.0)
            radius = max(0.005, radius*0.5)
        history.append({"iter": k, "seed": chosen,
                        "rms": rms(current_r), "trial_rms": rms(trial_r),
                        "accepted": accepted,
                        "lambda": damping, "radius": radius})
        if k % 5 == 0 or accepted and rms(current_r) <= target:
            print("ROM {:02d}: RMS={:.6f} accepted={}".format(
                k, rms(current_r), accepted))
        if radius <= 0.005 and damping >= 1e6:
            break
    write_rows(os.path.join(ctx.out_dir, "rom_lm_history.csv"), history)
    return best_q, best_r, chosen


def run_guarded_hf_lm(ctx, initial_q, max_eval, tol, cpus):
    current = ctx.hf(initial_q, cpus)
    best = current
    discrepancy_jac = np.zeros((26, 26), dtype=float)
    damping = 0.1
    radius = 0.65
    history = [{"evaluation": current["eval_id"], "accepted": True,
                "rms": current["rms"], "best_rms": best["rms"],
                "lambda": damping, "radius": radius, "rho": ""}]
    while ctx.hf_calls < max_eval and best["rms"] > tol:
        J_rom = ctx.finite_difference_rom_jacobian(current["q"])
        J = J_rom + discrepancy_jac
        step = lm_step(J, current["r"], damping, radius)
        candidate_q = np.clip(current["q"] + step, -3.0, 3.0)
        s = candidate_q - current["q"]
        if np.linalg.norm(s) < 1e-9:
            print("No acceptable step at current damping/radius.")
            break
        pred_r = current["r"] + J.dot(s)
        pred_improvement = max(1e-14, phi(current["r"]) - phi(pred_r))
        trial = ctx.hf(candidate_q, cpus)
        actual_improvement = phi(current["r"]) - phi(trial["r"])
        rho = actual_improvement / pred_improvement
        accepted = actual_improvement > 1e-12 and rho > 0.01
        if accepted:
            # Correct the cheap ROM Jacobian with high-fidelity
            # secant information: (rHF-rROM) change across the step.
            old_discrepancy = current["r"] - ctx.rom(current["q"])
            new_discrepancy = trial["r"] - ctx.rom(candidate_q)
            secant = new_discrepancy - old_discrepancy
            denom = float(np.dot(s, s))
            discrepancy_jac += np.outer(
                secant - discrepancy_jac.dot(s), s) / max(denom, 1e-14)
            current = trial
            damping = max(1e-6, damping * (0.5 if rho > 0.75 else 0.9))
            radius = min(2.0, radius * (1.3 if rho > 0.75 else 1.05))
            if trial["rms"] < best["rms"]:
                best = trial
        else:
            # Crucial safeguard: do NOT overwrite current with bad trial.
            damping = min(1e9, damping * 6.0)
            radius = max(0.003, radius * 0.45)
        history.append({"evaluation": trial["eval_id"],
                        "accepted": accepted, "rms": trial["rms"],
                        "best_rms": best["rms"], "lambda": damping,
                        "radius": radius, "rho": float(rho)})
        if radius <= 0.003 and not accepted:
            print("Trust region collapsed; stop rather than invent convergence")
            break
    write_rows(os.path.join(ctx.out_dir, "hf_lm_history.csv"), history)
    return best


def seed_candidates(material):
    scales, _, _ = load_configuration()
    seeds = [("zero", np.zeros(26, dtype=float))]
    if os.path.isfile(BASELINE_SOLUTION):
        with np.load(BASELINE_SOLUTION) as data:
            seeds.append(("converged_baseline",
                          np.asarray(data["q"], dtype=float)))
    try:
        anchor = nearest_anchor(np.asarray(material, dtype=float))
        seeds.append(("nearest_training_anchor",
                      np.asarray(anchor["q"], dtype=float)))
    except FileNotFoundError as exc:
        print("No training designs for warm start: {}".format(exc))
    return seeds


def step_init(args):
    out = os.path.join(BASE_DIR, "65L_initializations")
    mkdir(out)
    for case, E1, E2, G12 in MATERIAL_CASES:
        material = [E1, E2, G12]
        anchor = nearest_anchor(np.array(material))
        output = {"case": case, "material_MPa": material,
                  "nearest_training_anchor": anchor,
                  "additional_seeds": ["zero", "converged_baseline"]}
        save_json(os.path.join(out, case + "_initialization.json"), output)
        print(case, "nearest training global index:",
              anchor["global_training_index"],
              "distance:", anchor["material_distance_in_range_units"])
    print("65L-C done:", out)


def step_diagnose(args):
    out = os.path.join(BASE_DIR, "65L_force_diagnosis")
    mkdir(out)
    cases = [("baseline", (45000., 12000., 4500.),
              "online_results_7region_rom")]
    cases += [(n, (e1, e2, g),
               "online_results_7region_rom_65K_case_" + n)
              for n, e1, e2, g in MATERIAL_CASES]
    if args.case != "all":
        cases = [x for x in cases if x[0] == args.case]
    summary = []
    for name, material, directory in cases:
        file = os.path.join(BASE_DIR, directory, "rom_solution_7region.npz")
        if not os.path.isfile(file):
            print("SKIP (no 65H/65K solution):", file)
            continue
        with np.load(file) as data:
            q = np.asarray(data["q"], dtype=float)
        case_out = os.path.join(out, name)
        mkdir(case_out)
        ctx = Coupling(material, case_out)
        rom_r, _, rom_forces, _ = ctx.rom(q, detailed=True)
        true = ctx.hf(q, args.cpus)
        rows = []
        mismatch = []
        for interface in NAMES:
            g_rom = np.asarray(rom_forces[interface]).ravel()
            g_true = np.asarray(true["forces"][interface]).ravel()
            scale = ctx.force_scale[ctx.slices[interface]]
            mismatch.extend(((g_rom - g_true)/scale).tolist())
            for j, (a, b) in enumerate(zip(g_rom, g_true)):
                rows.append({"Interface": interface, "Mode": j+1,
                             "ROM_g": float(a), "Abaqus_g": float(b),
                             "AbsoluteForceError": float(abs(a-b)),
                             "RelativeForceError_percent": float(
                                 100*abs(a-b)/max(abs(b), 1e-12)),
                             "ForceErrorInResidualUnits": float((a-b)/scale[j])})
        write_rows(os.path.join(case_out, "same_state_forces.csv"), rows)
        entry = {"case": name, "material_MPa": list(material),
                 "rms_rom": rms(rom_r), "rms_true": true["rms"],
                 "rom_vs_true_force_gap_normalized_rms": rms(mismatch),
                 "rom_vs_true_force_gap_normalized_max": float(
                     np.max(np.abs(mismatch))),
                 "extra_abaqus_jobs_for_diagnosis": 3}
        summary.append(entry)
        save_json(os.path.join(case_out, "summary.json"), entry)
        print("DIAG", name, "ROM", entry["rms_rom"],
              "True", entry["rms_true"],
              "ROM/FE force gap", entry["rom_vs_true_force_gap_normalized_rms"])
    write_rows(os.path.join(out, "all_cases_diagnosis.csv"), summary)
    print("65L-A done:", out)


def step_solve(args):
    material = np.array([args.E1, args.E2, args.G12], dtype=float)
    if not np.all(np.isfinite(material)):
        raise ValueError("Nonfinite material properties")
    name = args.label or "single"
    out_dir = os.path.join(
        BASE_DIR, "online_results_7region_65L_" + name)
    mkdir(out_dir)
    ctx = Coupling(material, out_dir)
    t0 = time.perf_counter()
    seeds = seed_candidates(material)
    initial_q, rom_r, source = run_rom_lm(
        ctx, seeds, args.rom_max_iter, args.tol, name)
    t_rom = time.perf_counter() - t0
    print("ROM best RMS={:.8f}, seed={}".format(rms(rom_r), source))
    best = run_guarded_hf_lm(
        ctx, initial_q, args.hf_max_eval, args.tol, args.cpus)
    elapsed = time.perf_counter()-t0
    final_coeff = ctx.coeffs(best["q"])
    arrays = {"q": best["q"], "c_scale": ctx.scale,
              "normalized_residual": best["r"], "physical_residual": best["raw"],
              "rms_residual": np.array([best["rms"]]),
              "E1": np.array([material[0]]),
              "E2": np.array([material[1]]),
              "G12": np.array([material[2]])}
    for k in NAMES:
        arrays["c_"+k] = final_coeff[k]
    np.savez(os.path.join(out_dir, "rom_solution_7region.npz"), **arrays)
    final_dir = os.path.join(out_dir, "final")
    mkdir(final_dir)
    for patch in PATCH_NAMES:
        record = best["files"][patch]
        for key, ext in (("odb_file", ".odb"), ("input_file", ".inp")):
            src = record[key]
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            shutil.copy2(src, os.path.join(final_dir, patch + ext))
    manifest = {
        "case": name, "material_MPa": material.tolist(),
        "method": "ROM-initialized discrepancy-corrected trust-region LM",
        "converged": bool(best["rms"] <= args.tol),
        "rom_best_rms": rms(rom_r),
        "final_high_fidelity_rms": best["rms"],
        "final_high_fidelity_max": best["maximum"],
        "rom_residual_evaluations": ctx.rom_calls,
        "high_fidelity_residual_evaluations": ctx.hf_calls,
        "high_fidelity_abaqus_jobs": 3 * ctx.hf_calls,
        "rom_phase_seconds": t_rom,
        "high_fidelity_phase_seconds": elapsed - t_rom,
        "total_seconds": elapsed,
        "best_high_fidelity_evaluation": best["eval_id"],
        "initialization_selected": source,
        "tolerance": args.tol,
        "output_directory": out_dir,
    }
    save_json(os.path.join(out_dir, "rom_solution_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    # A nonconverged result remains saved for diagnosis but is not success.
    if not manifest["converged"]:
        raise SystemExit(2)


def step_benchmark(args):
    out_dir = os.path.join(BASE_DIR, "65L_robustness_benchmark")
    mkdir(out_dir)
    rows = []
    for name, E1, E2, G12 in MATERIAL_CASES:
        manifest_file = os.path.join(
            BASE_DIR, "online_results_7region_65L_" + name,
            "rom_solution_manifest.json")
        log_file = os.path.join(out_dir, name + ".log")
        if os.path.isfile(manifest_file) and not args.rerun:
            print("Already has output:", name, "(use --rerun to rerun)")
            return_code = None
            elapsed = None
        else:
            # On an explicit rerun, preserve the complete old case folder.
            # Abaqus job names and result paths would otherwise collide.
            case_folder = os.path.dirname(manifest_file)
            if args.rerun and os.path.isdir(case_folder):
                backup = case_folder + "_backup_" + time.strftime("%Y%m%d_%H%M%S")
                if os.path.exists(backup):
                    raise RuntimeError("Backup destination already exists: " + backup)
                os.rename(case_folder, backup)
                print("Preserved previous case at:", backup)
            command = 'abaqus python "{}" --action solve --label {} --E1 {} --E2 {} --G12 {} --cpus {} --hf_max_eval {} --rom_max_iter {} --tol {}'.format(
                os.path.abspath(__file__), name, E1, E2, G12,
                args.cpus, args.hf_max_eval, args.rom_max_iter, args.tol)
            print("Running", name, "with", command)
            t0 = time.perf_counter()
            with open(log_file, "w") as handle:
                return_code = subprocess.call(
                    command, shell=True, cwd=BASE_DIR,
                    stdout=handle, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - t0
        data = {}
        if os.path.isfile(manifest_file):
            with open(manifest_file, "r") as handle:
                data = json.load(handle)
        rows.append({
            "case": name, "E1_MPa": E1, "E2_MPa": E2, "G12_MPa": G12,
            "converged": data.get("converged", False) and (return_code in (None, 0)),
            "final_high_fidelity_rms": data.get("final_high_fidelity_rms", ""),
            "hf_evaluations": data.get("high_fidelity_residual_evaluations", ""),
            "abaqus_jobs": data.get("high_fidelity_abaqus_jobs", ""),
            "rom_evaluations": data.get("rom_residual_evaluations", ""),
            "total_seconds": data.get("total_seconds", ""),
            "process_seconds": elapsed if elapsed is not None else "",
            "exit_code": return_code if return_code is not None else "already_exists",
            "manifest": manifest_file,
        })
    write_rows(os.path.join(out_dir, "65L_benchmark.csv"), rows)
    passed = sum(bool(row["converged"]) for row in rows)
    summary = {"total_cases": len(rows), "passed": passed,
               "all_passed": passed == 3 and len(rows) == 3,
               "note": "Convergence only; full-field accuracy for new materials is NOT validated."}
    save_json(os.path.join(out_dir, "65L_benchmark_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    print("65L-D output:", out_dir)


def step_selftest():
    # Algebra-only test; no Abaqus required.
    rng = np.random.RandomState(9)
    A = np.eye(26) + 0.03*rng.randn(26, 26)
    target = 0.07*np.arange(26)/26
    q = np.zeros(26)
    initial = rms(A.dot(q-target))
    for k in range(6):
        r = A.dot(q-target)
        q = q + lm_step(A, r, 0.02, 0.8)
    final = rms(A.dot(q-target))
    print("SELFTEST initial RMS", initial, "final RMS", final)
    if not final < min(1e-4, initial/10):
        raise RuntimeError("Trust-region LM algebra test failed")
    print("SELFTEST PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True,
                        choices=["diagnose", "init", "solve", "benchmark", "selftest"])
    parser.add_argument("--case", default="all",
                        choices=["all", "baseline", "soft", "middle", "stiff"])
    parser.add_argument("--label", default="")
    parser.add_argument("--E1", type=float, default=45000.)
    parser.add_argument("--E2", type=float, default=12000.)
    parser.add_argument("--G12", type=float, default=4500.)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--rom_max_iter", type=int, default=40)
    parser.add_argument("--hf_max_eval", type=int, default=15)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.action == "selftest":
        step_selftest()
    elif args.action == "diagnose":
        step_diagnose(args)
    elif args.action == "init":
        step_init(args)
    elif args.action == "solve":
        step_solve(args)
    elif args.action == "benchmark":
        step_benchmark(args)


if __name__ == "__main__":
    main()
