#!/usr/bin/env python3
"""65AM-B9A9: pre-registered, independent multi-case Abaqus force-head validation.

Mac stages:
  python3 65AM_B9A9_mac_multicase.py --stage selftest
  python3 65AM_B9A9_mac_multicase.py --stage check --project .
  python3 65AM_B9A9_mac_multicase.py --stage plan --project .
  python3 65AM_B9A9_mac_multicase.py --stage evaluate --project .

Uses existing B9A7 + B9A8 scripts/checkpoints and original 65V exported force head.
NEVER trains/tunes a predictor. B9A3 saved-state reference is used only to authenticate
legacy anchors; new B9A9 Abaqus reactions are held out until evaluation.
Requires user-created B9A9_transfer_to_windows/ and results/.../B9A9_multicase_force/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

REGIONS = ("inner_left", "inner_right")
FACES = {"inner_left": ("left_inner", "center_left"),
         "inner_right": ("center_right", "right_inner")}
TRANSFER = Path("B9A9_transfer_to_windows")
WINDOWS = Path("B9A9_exports")
RESULT = Path("results/65AM_B9_coupled_feasibility/B9A9_multicase_force")
N = 12
SEED = 659009


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def import_script(root, filename, name):
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_free(paths):
    for p in paths:
        if not p.parent.is_dir():
            raise RuntimeError("Create this folder manually in Finder first: " + str(p.parent))
        if p.exists():
            raise RuntimeError("Protected existing output (do not overwrite): " + str(p))


def save_csv(path, rows):
    if not rows:
        raise ValueError("Empty CSV")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_models(root):
    a7 = import_script(root, "65AM_B9A7_force_head.py", "B9A7_existing")
    a8 = import_script(root, "65AM_B9A8_residual_force_operator.py", "B9A8_existing")
    prev, anchor, reference, ahash, parts, olddir, pcadir = a8.load_everything(root, None, None, verbose=True)
    models = {}
    for region in REGIONS:
        part = parts[region]
        part["region"] = region
        ck = root / a8.OUT / ("B9A8_" + region + "_correction.npz")
        corrected = a8.load_chk(ck, part, ahash)
        models[region] = corrected
    return a7, a8, anchor, ahash, parts, models


def generate_plan(anchor, parts, n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    materials = np.asarray(anchor["material"], dtype=float).ravel()
    if materials.shape != (3,):
        raise RuntimeError("Expected three material properties")
    # Only the ORIGINAL training set determines envelopes and perturbation sizes.
    trains = {r: parts[r]["ds"]["x"][parts[r]["ds"]["split"]["train"]] for r in REGIONS}
    low = np.max(np.stack([x[:, :3].min(axis=0) for x in trains.values()]), axis=0)
    high = np.min(np.stack([x[:, :3].max(axis=0) for x in trains.values()]), axis=0)
    stdmat = np.min(np.stack([np.maximum(x[:, :3].std(axis=0), 1e-9) for x in trains.values()]), axis=0)
    if np.any(low >= high) or np.any(materials < low) or np.any(materials > high):
        raise RuntimeError("Saved material is outside overlapping training ranges; stop and inspect")
    # Alternating bounded train-scale changes. Not tuned using any new labels.
    mat_lhs = np.zeros((n, 3), float)
    for j in range(3):
        mat_lhs[:, j] = rng.permutation(np.linspace(-0.65, 0.65, n))
    mat_candidates = materials[None, :] + mat_lhs * (0.45 * stdmat[None, :])
    mat_candidates = np.clip(mat_candidates, low + .01 * (high - low), high - .01 * (high - low))
    allx, rows = {}, []
    groups = ["material"] * (n // 3) + ["interface"] * (n // 3) + ["combined"] * (n - 2 * (n // 3))
    ids = ["B9A9_%02d" % (i + 1) for i in range(n)]
    for region in REGIONS:
        ds = parts[region]["ds"]
        train = trains[region]
        faces = FACES[region]
        q0 = np.concatenate([np.asarray(anchor["c_" + f], float).ravel() for f in faces])
        base = np.r_[materials, q0]
        if train.shape[1] != base.size or base.size != 12:
            raise RuntimeError("Unexpected 12-feature material/PCA ordering: " + region)
        q_std = np.maximum(train[:, 3:].std(axis=0), 1e-12)
        # Independent, deterministic Latin-hypercube-style coefficient perturbations.
        perturb = np.column_stack([rng.permutation(np.linspace(-1., 1., n)) for _ in range(len(q0))])
        qlo = train[:, 3:].min(axis=0)
        qhi = train[:, 3:].max(axis=0)
        xcases = []
        for i, group in enumerate(groups):
            mat = materials if group == "interface" else mat_candidates[i]
            delta = np.zeros(len(q0)) if group == "material" else (0.32 * q_std * perturb[i])
            q = q0.copy() if group == "material" else np.clip(q0 + delta, qlo + .01 * (qhi-qlo), qhi - .01*(qhi-qlo))
            x = np.r_[mat, q]
            if not np.all(np.isfinite(x)) or np.any(x[:3] <= 0):
                raise RuntimeError("Invalid B9A9 material or coefficients")
            if np.linalg.norm((x - base) / ds["x_scale"]) < .015:
                raise RuntimeError("Generated case too close to saved anchor: " + region + " " + ids[i])
            if np.min(np.linalg.norm((ds["x"] - x) / ds["x_scale"], axis=1)) < .015:
                raise RuntimeError("Case overlaps 64J data within normalized 0.015: " + region + " " + ids[i])
            xcases.append(x)
            rows.append(dict(region=region,case_id=ids[i],class_name=group,
                E1_MPa=x[0],E2_MPa=x[1],G12_MPa=x[2],
                nearest_64J_case_normalized=float(np.min(np.linalg.norm((ds["x"] - x) / ds["x_scale"], axis=1))),
                distance_from_anchor_normalized=float(np.linalg.norm((x-base)/ds["x_scale"]))))
        if np.unique(np.round(np.asarray(xcases), 8), axis=0).shape[0] != n:
            raise RuntimeError("Duplicate cases produced for " + region)
        allx[region] = np.asarray(xcases, dtype=float)
    # These are separate Dirichlet subdomain problems, NOT full seven-region coupled states.
    return ids, groups, allx, rows


def check_files(root):
    _, _, anchor, ahash, parts, _ = load_models(root)
    print("B9A9 CHECK PASS: original 65V models, B9A7/B9A8 checkpoints, PCA bases and anchor authenticated", ahash[:16])
    return anchor, parts


def plan(root):
    a7, a8, anchor, ahash, parts, _ = load_models(root)
    ids, groups, xx, rows = generate_plan(anchor, parts)
    target = root / TRANSFER / "B9A9_case_plan.npz"
    manifest = root / TRANSFER / "B9A9_plan_manifest.json"
    table = root / TRANSFER / "B9A9_case_plan.csv"
    ensure_free([target, manifest, table])
    payload = {"anchor_sha256": np.asarray(ahash), "case_ids": np.asarray(ids),
               "class_names": np.asarray(groups), "x_inner_left": xx["inner_left"],
               "x_inner_right": xx["inner_right"]}
    np.savez_compressed(target, **payload)
    save_csv(table, rows)
    info = dict(step="B9A9", count_per_region=N, seed=SEED, plan_sha256=sha(target),
                anchor_sha256=ahash, region_names=list(REGIONS), case_ids=ids,
                experiment_types=groups,
                design_rule="Train-only range/std, deterministic; original validation/test and B9A3 labels excluded from planning",
                training_data_hashes={r:dict(branch_sha256=parts[r]["ds"]["x_sha256"],
                                                   split_sha256=parts[r]["ds"]["split_sha256"]) for r in REGIONS},
                not_a_coupled_solution=True)
    manifest.write_text(json.dumps(info, indent=2))
    print("PLAN DONE:", target)
    print("Cases per inner region", N, "(24 independent Abaqus subdomain solves total)")


def evaluate(root):
    a7, a8, anchor, ahash, parts, checkpoints = load_models(root)
    inp = root / TRANSFER
    planfile = inp / "B9A9_case_plan.npz"
    meta = json.loads((inp / "B9A9_plan_manifest.json").read_text())
    if meta["plan_sha256"] != sha(planfile) or meta["anchor_sha256"] != ahash:
        raise RuntimeError("Plan or saved anchor has changed after generation")
    plan_data = load_npz(planfile)
    ref_file = root / WINDOWS / "B9A9_reference_forces.npz"
    ws = json.loads((root / WINDOWS / "B9A9_windows_summary.json").read_text())
    if ws.get("anchor_sha256") != ahash or ws.get("plan_sha256") != sha(planfile):
        raise RuntimeError("Windows results are not for this exact B9A9 plan")
    if ws.get("reference_sha256") != sha(ref_file):
        raise RuntimeError("Windows force reference file checksum mismatch")
    if ws.get("all_jobs_pass") is not True or ws.get("case_count") != 24:
        raise RuntimeError("Not all 24 Abaqus cases passed the ODB/completion audits")
    ref = load_npz(ref_file)
    if str(ref["anchor_sha256"].item()) != ahash or str(ref["plan_sha256"].item()) != sha(planfile):
        raise RuntimeError("NPZ metadata does not match plan")
    for name in ("case_ids", "class_names"):
        if not np.array_equal(plan_data[name], ref[name]):
            raise RuntimeError("Case order differs between Mac and Windows: " + name)
    if np.max([float(k["max_boundary_U_error_mm"]) for k in ws["jobs"]]) > 1e-5:
        raise RuntimeError("Abaqus prescribed U mismatch > 1e-5 mm")
    if np.max([float(k["max_coordinate_error_mm"]) for k in ws["jobs"]]) > 2e-5:
        raise RuntimeError("Abaqus node mapping error > 2e-5 mm")
    output = root / RESULT
    casepath = output / "B9A9_case_metrics.csv"
    modepath = output / "B9A9_mode_metrics.csv"
    summarypath = output / "B9A9_summary.json"
    plotpath = output / "B9A9_force_comparison.png"
    ensure_free([casepath, modepath, summarypath, plotpath])
    cases, modes, report = [], [], {}
    for region in REGIONS:
        x = plan_data["x_" + region]
        y = np.asarray(ref["g_" + region], dtype=float)
        if not np.array_equal(x, ref["x_" + region]):
            raise RuntimeError("Windows returned different material/interface coefficients: " + region)
        part = parts[region]
        scale = np.asarray(part["force_scale"], dtype=float)
        if x.shape != (N, 12) or y.shape != (N, 9) or scale.shape != (9,):
            raise RuntimeError("Force or input dimensional mismatch for " + region)
        if not np.isfinite(y).all():
            raise RuntimeError("Nonfinite Abaqus reaction forces in " + region)
        old = a8.original_g(region, x, part["model"], part["pca"])
        a7path = root / a7.OUT_REL / ("B9A7_" + region + "_force_head.npz")
        ck7 = a7.load_npz(a7path)
        for field, value in (("data_x_sha256",part["ds"]["x_sha256"]),
                             ("data_y_sha256",part["ds"]["y_sha256"]),
                             ("split_sha256",part["ds"]["split_sha256"])):
            if str(np.asarray(ck7[field]).item()) != value:
                raise RuntimeError("B9A7 checkpoint no longer matches training data")
        p7 = a7.infer(ck7, x)
        p8 = a8.predict_final(part, checkpoints[region], X=x, old=old)
        preds = {"Original65V": old, "B9A7": p7, "B9A8": p8}
        for model, pred in preds.items():
            for i in range(N):
                diff = (pred[i] - y[i]) / scale
                cases.append(dict(region=region,case_id=str(plan_data["case_ids"][i]),
                    class_name=str(plan_data["class_names"][i]),model=model,
                    scaled_mode_RMS=float(np.sqrt(np.mean(diff**2))),
                    abs_force_L2_N=float(np.linalg.norm(pred[i]-y[i])),
                    force_relative_L2_pct=float(100 * np.linalg.norm(pred[i]-y[i]) / max(np.linalg.norm(y[i]),1e-12))))
            start = 0
            for face in FACES[region]:
                k = a7.SL[face].stop - a7.SL[face].start
                for m in range(k):
                    d = (pred[:,start+m]-y[:,start+m])/scale[start+m]
                    modes.append(dict(region=region,interface=face,mode=m+1,model=model,
                        scaled_mode_RMSE=float(np.sqrt(np.mean(d**2))),
                        scaled_mode_MAE=float(np.mean(np.abs(d)))))
                start += k
        per = {model:float(np.sqrt(np.mean(((pred-y)/scale)**2))) for model,pred in preds.items()}
        winners = np.argmin(np.stack([np.sqrt(np.mean(((pred-y)/scale)**2,axis=1)) for pred in preds.values()]),axis=0)
        report[region] = dict(cases=N, force_error_scaled_RMS=per,
            case_wins=dict(zip(preds.keys(),[int(np.sum(winners==j)) for j in range(3)])),
            B9A7_better_than_original=bool(per["B9A7"] < per["Original65V"]),
            B9A8_better_than_original=bool(per["B9A8"] < per["Original65V"]))
        print("B9A9",region, *("%s=%.6f"%(k,v) for k,v in per.items()))
    save_csv(casepath,cases)
    save_csv(modepath,modes)
    overall = dict(step="65AM-B9A9",stage="independent_multicase_validation",anchor_sha256=ahash,
       plan_sha256=sha(planfile),reference_sha256=sha(ref_file),cases_total=24,regions=report,
       new_case_labels_used_in_fitting=False,original_models_untouched=True,
       caveats=["Each Abaqus job is an independently prescribed two-interface subdomain, NOT a coupled seven-region solve",
                "Force-head accuracy is distinct from normalized FE+NO residual cancellation",
                "No model has been approved for live Broyden coupling or 0.02 convergence based on this test alone"])
    summarypath.write_text(json.dumps(overall,indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["left_inner","center_left","center_right","right_inner"]
    mdl = ("Original65V","B9A7","B9A8")
    fig, ax = plt.subplots(figsize=(12,5))
    xpos = np.arange(4)
    for j, model in enumerate(mdl):
        ys = []
        for name in labels:
            sub = [m["scaled_mode_RMSE"]**2 for m in modes if m["interface"] == name and m["model"] == model]
            ys.append(float(np.sqrt(np.mean(sub))))
        ax.bar(xpos+(j-1)*.25,ys,.24,label=model)
    ax.set_xticks(xpos);ax.set_xticklabels(labels)
    ax.set_ylabel("RMS generalized-force error / frozen 65V force scales")
    ax.set_title("B9A9: genuinely new Abaqus boundary/material conditions (12 cases per region)")
    ax.legend();fig.tight_layout();fig.savefig(plotpath,dpi=180);plt.close(fig)
    print("EVALUATION COMPLETE:",output)
    print("NOT a new 65V/Broyden coupled equilibrium solution.")


def selftest():
    rng = np.random.default_rng(16)
    z=rng.normal(size=(12,9));y=rng.normal(size=(12,9));s=np.linspace(.1,2.,9)
    assert np.isclose(np.sqrt(np.mean(((z-y)/s)**2)),np.sqrt(np.mean(((z-y)/s)**2,axis=1).mean()))
    assert np.unique(np.round(rng.normal(size=(12,12)),8),axis=0).shape[0] == 12
    print("B9A9 MAC SELFTEST PASS: scaled modes, case and mode aggregation")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("selftest","check","plan","evaluate"),required=True)
    p.add_argument("--project",default=".")
    args=p.parse_args()
    if args.stage=="selftest":return selftest()
    root=Path(args.project).expanduser().resolve()
    if args.stage=="check":return check_files(root)
    if args.stage=="plan":return plan(root)
    if args.stage=="evaluate":return evaluate(root)

if __name__=="__main__":main()
