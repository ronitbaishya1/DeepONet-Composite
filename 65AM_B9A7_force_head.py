#!/usr/bin/env python3
"""65AM-B9A7: frozen-B5 companion force head, on Mac (no Abaqus jobs).

Predict *actual Abaqus NO-side generalized interface reaction forces* g from
(material E1,E2,G12 + displacement PCA coefficients). No stress/one-point proxy.
B5 displacement model is NEVER changed or retrained. Old 65V is unchanged.

Stages:
 python3 65AM_B9A7_force_head.py --stage selftest
 python3 65AM_B9A7_force_head.py --stage check --project .
 python3 65AM_B9A7_force_head.py --stage train --project .
 python3 65AM_B9A7_force_head.py --stage anchor --project .

Required (Mac): data/hybrid_bulk_7region/{inner_left,inner_right}/
  branch_inputs.npy, g_all.npy, split_assignment.csv, metadata.json
 data/candidate_partition_interfaces_7region/interface_*.npz
 B9_coupled_anchor/B9_65V_anchor.npz + manifest
 B9A3_exports/B9A3_NO_reference.npz + Windows summary

New output: results/65AM_B9_coupled_feasibility/B9A7_force_head/
Create empty output folder yourself using Finder. Never overwrites outputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import copy
import numpy as np

REGIONS = {"inner_left": ("left_inner", "center_left"),
           "inner_right": ("center_right", "right_inner")}
INTERFACES = ("left_outer", "left_inner", "center_left", "center_right", "right_inner", "right_outer")
COUNTS = (4, 4, 5, 5, 4, 4)
SL = {}
_cursor = 0
for _name, _count in zip(INTERFACES, COUNTS):
    SL[_name] = slice(_cursor, _cursor + _count)
    _cursor += _count
OUT_REL = Path("results/65AM_B9_coupled_feasibility/B9A7_force_head")
RIDGES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SEEDS = (7, 23, 41)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def read_json(path):
    return json.loads(Path(path).read_text())


def check_free(path):
    if Path(path).exists():
        raise RuntimeError("Output protected. Do not overwrite: " + str(path))
    if not Path(path).parent.is_dir():
        raise RuntimeError("Create this output folder with Finder first: " + str(Path(path).parent))


def rows_csv(path, rows):
    check_free(path)
    if not rows:
        raise ValueError("Cannot save an empty CSV")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_json(path, x):
    check_free(path)
    Path(path).write_text(json.dumps(x, indent=2))


def mode_rms(yhat, y, scale):
    return float(np.sqrt(np.mean(((np.asarray(yhat) - np.asarray(y)) / scale) ** 2)))


def basis_check(root, d, names):
    base = root / "data/candidate_partition_interfaces_7region"
    for name in names:
        fname = base / ("interface_" + name + ".npz")
        src = load_npz(fname)
        for key in ("basis", "mean", "coordinates"):
            old = np.asarray(d["pca_" + key + "_" + name], float)
            local = np.asarray(src[key], float)
            if old.shape != local.shape:
                raise RuntimeError("PCA shape mismatch: " + name + " " + key)
            tolerance = 1e-4 if key == "coordinates" else 3e-5
            if np.max(np.abs(old - local)) > tolerance:
                raise RuntimeError("STOP: training PCA differs from 65V PCA at " + name + "/" + key)


def read_case_split(path, n):
    records = list(csv.DictReader(Path(path).open(newline="")))
    if len(records) != n:
        raise ValueError("Split length inconsistent: " + str(path))
    found = {}
    indices = []
    for row in records:
        idx = int(row["Index"])
        sp = str(row["Split"]).lower().strip()
        if sp in ("val", "valid"):
            sp = "validation"
        if sp not in ("train", "validation", "test"):
            raise ValueError("Invalid split: " + sp)
        if idx in found or idx < 0 or idx >= n:
            raise ValueError("Invalid or duplicate Index in split file")
        found[idx] = sp
        indices.append(idx)
    if set(indices) != set(range(n)):
        raise ValueError("Missing indices in split file")
    ans = {s: np.asarray(sorted(k for k, v in found.items() if v == s), int)
           for s in ("train", "validation", "test")}
    if min(len(t) for t in ans.values()) < 5:
        raise RuntimeError("At least five train/validation/test cases per split required")
    return ans


def load_anchor(root):
    folder = root / "B9_coupled_anchor"
    p = folder / "B9_65V_anchor.npz"
    m = read_json(folder / "B9_65V_anchor_manifest.json")
    if sha(p) != m["anchor_sha256"]:
        raise RuntimeError("Saved 65V anchor checksum mismatch")
    d = load_npz(p)
    refp = root / "B9A3_exports/B9A3_NO_reference.npz"
    refm = read_json(root / "B9A3_exports/B9A3_windows_summary.json")
    if refm.get("anchor_sha256") != sha(p) or sha(refp) != refm.get("export_sha256"):
        raise RuntimeError("B9A3 independent Abaqus force reference checksum mismatch")
    ref = load_npz(refp)
    if str(np.asarray(ref["anchor_sha256"]).item()) != sha(p):
        raise RuntimeError("B9A3 anchor metadata mismatch")
    if refm.get("geometry_and_displacement_checks") is not True:
        raise RuntimeError("B9A3 Abaqus geometry/displacement verification did not pass")
    for key in ("old_NO_g", "force_scale", "normalized_residual", "FE_g_inferred"):
        if np.asarray(d[key]).shape != (26,):
            raise RuntimeError("Anchor 26-mode array missing: " + key)
    if np.any(np.asarray(d["force_scale"]) <= 0):
        raise RuntimeError("Invalid original 65V force normalization")
    if mode_rms((np.asarray(d["old_NO_g"]) + np.asarray(d["FE_g_inferred"])) /
                np.asarray(d["force_scale"]), np.asarray(d["normalized_residual"]), 1.) > 1e-6:
        raise RuntimeError("Anchor does not reproduce saved 65V residual")
    if not np.allclose(np.asarray(d["material"]), np.asarray(ref["material"]), atol=1e-8):
        raise RuntimeError("Material mismatch with B9A3")
    mask = np.zeros(26, dtype=bool)
    for sides in REGIONS.values():
        for name in sides:
            mask[SL[name]] = True
    if not np.array_equal(np.asarray(ref["reference_mask"]).astype(bool), mask):
        raise RuntimeError("B9A3 reference mode mask mismatch")
    return d, ref, str(sha(p))


def load_region(root, anchor, region):
    folder = root / "data/hybrid_bulk_7region" / region
    xfile = folder / "branch_inputs.npy"
    yfile = folder / "g_all.npy"
    x = np.asarray(np.load(xfile, allow_pickle=False), float)
    y = np.asarray(np.load(yfile, allow_pickle=False), float)
    meta = read_json(folder / "metadata.json")
    split = read_case_split(folder / "split_assignment.csv", len(x))
    left, right = REGIONS[region]
    if meta.get("left_interface") != left or meta.get("right_interface") != right:
        raise RuntimeError("Region-to-interface mapping changed for " + region)
    counts = (SL[left].stop - SL[left].start, SL[right].stop - SL[right].start)
    if x.ndim != 2 or x.shape[1] != 3 + sum(counts) or y.shape != (len(x), sum(counts)):
        raise RuntimeError("Wrong branch/force dimensions for " + region + ": " + str((x.shape, y.shape)))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Nonfinite training data at " + region)
    if len(x) < 40:
        raise RuntimeError("Too few candidate NO-side force cases for " + region)
    if np.linalg.matrix_rank(np.cov(x[split["train"]].T)) < 4:
        raise RuntimeError("Training branch values have too little variation: " + region)
    # Exact duplicate branch inputs across splits are a leakage risk.
    for sa in ("train", "validation", "test"):
        for sb in ("train", "validation", "test"):
            if sa >= sb:
                continue
            xa = np.round(x[split[sa]], 7)
            xb = np.round(x[split[sb]], 7)
            if set(map(tuple, xa)).intersection(set(map(tuple, xb))):
                raise RuntimeError("Duplicate branch cases across %s/%s in %s" % (sa, sb, region))
    basis_check(root, anchor, (left, right))
    scale = np.concatenate((anchor["force_scale"][SL[left]], anchor["force_scale"][SL[right]])).astype(float)
    # Train-only normalization. The force target is normalized by 65V's ACTUAL scales.
    mu_x = x[split["train"]].mean(axis=0)
    sig_x = np.maximum(x[split["train"]].std(axis=0), 1e-6)
    mu_y = y[split["train"]].mean(axis=0)
    xx = (x - mu_x) / sig_x
    yy = (y - mu_y) / scale
    info = dict(x=x, y=y, xx=xx, yy=yy, split=split, x_mean=mu_x,
                x_scale=sig_x, g_mean=mu_y, g_scale=scale,
                x_sha256=sha(xfile), y_sha256=sha(yfile),
                split_sha256=sha(folder / "split_assignment.csv"), left=left, right=right)
    print("CHECK PASS", region, "cases", len(x), "branch", x.shape[1], "modes", y.shape[1],
          "split", {k: len(v) for k, v in split.items()})
    return info


def fit_ridge(x, y, alpha):
    # intercept unregularized, all other weights penalized
    D = np.column_stack([x, np.ones(len(x))])
    pen = np.diag(np.r_[np.ones(x.shape[1]), 0.]) * alpha
    return np.linalg.solve(D.T @ D + pen + np.eye(D.shape[1]) * 1e-10, D.T @ y)


def linpred(x, W):
    return np.column_stack([x, np.ones(len(x))]) @ W


def nn_pred(x, W):
    z = np.tanh(x @ W["W1"].T + W["b1"])
    z = np.tanh(z @ W["W2"].T + W["b2"])
    return z @ W["W3"].T + W["b3"]


def train_nn(info, W, seed, epochs, patience, device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    x = info["xx"].astype(np.float32)
    y = info["yy"].astype(np.float32)
    baseline = linpred(info["xx"], W).astype(np.float32)
    ix, iv = info["split"]["train"], info["split"]["validation"]
    torch_device = torch.device(device)
    xt = torch.tensor(x[ix], device=torch_device)
    yt = torch.tensor(y[ix] - baseline[ix], device=torch_device)
    xv = torch.tensor(x[iv], device=torch_device)
    yv = torch.tensor(y[iv], device=torch_device)
    base_val = torch.tensor(baseline[iv], device=torch_device)
    d = x.shape[1]
    model = nn.Sequential(nn.Linear(d, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, y.shape[1])).to(torch_device)
    with torch.no_grad():
        nn.init.zeros_(model[-1].weight)
        nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    best = float("inf")
    best_state = None
    best_epoch = 0
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        r = model(xt)
        loss = ((r - yt) ** 2).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            lv = float(torch.sqrt(torch.mean((base_val + model(xv) - yv) ** 2)).cpu().item())
        if lv < best - 1e-7:
            best = lv
            best_epoch = ep
            best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        if ep - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("No NN training checkpoint")
    weights = {}
    for i, label in zip((0, 2, 4), ("1", "2", "3")):
        weights["W" + label] = best_state[str(i) + ".weight"]
        weights["b" + label] = best_state[str(i) + ".bias"]
    return weights, best, best_epoch


def predict_model(info, W, weights=None):
    v = linpred(info["xx"], W)
    if weights is not None:
        v = v + nn_pred(info["xx"], weights)
    return info["g_mean"] + info["g_scale"] * v


def fit_one(info, region, epochs, patience, device):
    xx, yy, spl = info["xx"], info["yy"], info["split"]
    train, val = spl["train"], spl["validation"]
    options = []
    for a in RIDGES:
        W = fit_ridge(xx[train], yy[train], a)
        ev = mode_rms(linpred(xx[val], W), yy[val], 1.)
        options.append((ev, a, W))
    options.sort(key=lambda z: z[0])
    base_val, alpha, Wbest = options[0]
    print("RIDGE", region, "selected alpha", alpha, "validation RMS", round(base_val, 6))
    winner_type = "ridge"
    winner_weights = None
    winner_val = base_val
    trials = []
    for seed in SEEDS:
        w, score, ep = train_nn(info, Wbest, seed, epochs, patience, device)
        trials.append(dict(seed=seed, best_epoch=ep, validation_scaled_RMS=score))
        print("NN", region, "seed", seed, "epoch", ep, "validation RMS", round(score, 6))
        # Avoid selecting an effectively tied, more complex model.
        if score <= 0.98 * winner_val:
            winner_val = score
            winner_weights = w
            winner_type = "ridge_plus_nn"
    pred_base = predict_model(info, Wbest)
    pred_selected = predict_model(info, Wbest, winner_weights)
    summaries, modes = [], []
    for label, p in (("ridge", pred_base), ("selected", pred_selected)):
        for split_name, ix in spl.items():
            summaries.append(dict(region=region, model=label, split=split_name,
                  samples=len(ix), scaled_force_RMS=mode_rms(p[ix], info["y"][ix], info["g_scale"]),
                  force_relative_L2_pct=100 * np.linalg.norm((p[ix] - info["y"][ix]).ravel()) / max(np.linalg.norm(info["y"][ix].ravel()), 1e-12)))
            if split_name == "test":
                for m in range(info["y"].shape[1]):
                    name = info["left"] if m < SL[info["left"]].stop - SL[info["left"]].start else info["right"]
                    local = m + 1 if name == info["left"] else m + 1 - (SL[info["left"]].stop - SL[info["left"]].start)
                    modes.append(dict(region=region, model=label, interface=name, mode=local,
                              test_mode_scaled_RMSE=float(np.sqrt(np.mean(((p[ix, m] - info["y"][ix, m]) / info["g_scale"][m]) ** 2)))))
    state = dict(model_type=np.array(winner_type),x_mean=info["x_mean"],x_scale=info["x_scale"],
                 g_mean=info["g_mean"],g_scale=info["g_scale"],ridge_weights=Wbest,
                 data_x_sha256=np.array(info["x_sha256"]),data_y_sha256=np.array(info["y_sha256"]),
                 split_sha256=np.array(info["split_sha256"]),validation_scaled_RMS=np.array(winner_val))
    if winner_weights:
        state.update(winner_weights)
    return state, summaries, modes, dict(region=region,selected=winner_type,alpha=alpha,
                ridge_validation_RMS=base_val,chosen_validation_RMS=winner_val,nn_trials=trials,
                note="Validation selected; test untouched during selection")


def infer(state, x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] != len(state["x_mean"]):
        raise ValueError("Wrong feature count for force head")
    xx = (x - state["x_mean"]) / state["x_scale"]
    normed = linpred(xx, state["ridge_weights"])
    if str(np.asarray(state["model_type"]).item()) == "ridge_plus_nn":
        normed += nn_pred(xx, state)
    return state["g_mean"] + state["g_scale"] * normed


def stage_check(root):
    anchor, reference, anchor_hash = load_anchor(root)
    ds = {r: load_region(root, anchor, r) for r in REGIONS}
    print("65V reference RMS", mode_rms((anchor["old_NO_g"] + anchor["FE_g_inferred"]) / anchor["force_scale"], np.zeros(26), 1.))
    print("B9A3 real NO reference present and checksum verified", anchor_hash[:12])
    return anchor, reference, anchor_hash, ds


def stage_train(root, epochs, patience, device):
    anchor, ref, anchor_hash, datasets = stage_check(root)
    out = root / OUT_REL
    if not out.is_dir():
        raise RuntimeError("Create output folder with Finder first: " + str(out))
    targets = [out / ("B9A7_" + r + "_force_head.npz") for r in REGIONS]
    targets += [out / n for n in ("B9A7_split_metrics.csv", "B9A7_test_modes.csv", "B9A7_training_summary.json")]
    for t in targets:
        check_free(t)
    all_splits, all_modes, reports = [], [], []
    states = []
    for region, info in datasets.items():
        st, metrics, modes, rep = fit_one(info, region, epochs, patience, device)
        all_splits.extend(metrics)
        all_modes.extend(modes)
        reports.append(rep)
        states.append(st)
    for target, st in zip(targets[:2], states):
        np.savez_compressed(target, **st)
    rows_csv(out / "B9A7_split_metrics.csv", all_splits)
    rows_csv(out / "B9A7_test_modes.csv", all_modes)
    save_json(out / "B9A7_training_summary.json", dict(step="65AM-B9A7",phase="independent_force_head_train",
              original_65V_anchor_SHA256=anchor_hash, B5_displacement_frozen=True,
              no_65V_solver_changes=True, no_B9A3_anchor_labels_used_in_training=True,
              original_64J_train_validation_test_split_preserved=True,
              trained_on="data/hybrid_bulk_7region/{region}/g_all.npy (actual Abaqus RF PCA generalized forces)",
              models=reports, note="No live Broyden authorization. Candidate NO-reaction force head only."))
    print("TRAIN COMPLETE. Check the original held-out test metrics BEFORE anchor review. Outputs:", out)


def stage_anchor(root):
    anchor, ref, anchor_hash, ds = stage_check(root)
    out = root / OUT_REL
    models = {r: load_npz(out / ("B9A7_" + r + "_force_head.npz")) for r in REGIONS}
    for region, data in ds.items():
        st = models[region]
        if str(np.asarray(st["data_x_sha256"]).item()) != data["x_sha256"] or str(np.asarray(st["data_y_sha256"]).item()) != data["y_sha256"] or str(np.asarray(st["split_sha256"]).item()) != data["split_sha256"]:
            raise RuntimeError("Training data changed after checkpoint saved: " + region)
    targets = (out / "B9A7_anchor_modes.csv", out / "B9A7_anchor_summary.json", out / "B9A7_anchor_force_modes.png")
    for target in targets:
        check_free(target)
    old = np.asarray(anchor["old_NO_g"], float).copy()
    FE = np.asarray(anchor["FE_g_inferred"], float)
    true = np.asarray(ref["reference_NO_g"], float)
    mask = np.asarray(ref["reference_mask"], bool)
    scale = np.asarray(anchor["force_scale"], float)
    new = old.copy()
    cases = {}
    for region, names in REGIONS.items():
        feature = np.concatenate([anchor["material"], anchor["c_"+names[0]], anchor["c_"+names[1]]]).astype(float)
        if len(feature) != ds[region]["x"].shape[1]:
            raise RuntimeError("Saved anchor branch structure differs from force training data")
        got = infer(models[region], feature)[0]
        j = 0
        for name in names:
            count = SL[name].stop - SL[name].start
            new[SL[name]] = got[j:j+count]
            j += count
        d = (feature - ds[region]["x_mean"]) / ds[region]["x_scale"]
        training_x = ds[region]["xx"][ds[region]["split"]["train"]]
        nearest = float(np.min(np.linalg.norm(training_x - d[None, :], axis=1)))
        cases[region] = dict(nearest_training_normalized_feature_distance=nearest,
                             possible_exact_training_case_overlap=bool(nearest < 1e-4),
                             branch_feature_count=len(feature))
    oracle = old.copy(); oracle[mask] = true[mask]
    rows = []
    for name in INTERFACES:
        sl = SL[name]
        for i in range(sl.stop-sl.start):
            idx=sl.start+i
            rows.append(dict(interface=name, mode=i+1, replaced=bool(mask[idx]), FE_N=float(FE[idx]),
                original_NO_N=float(old[idx]), new_head_N=float(new[idx]),
                Abaqus_true_NO_N=float(true[idx]) if mask[idx] else None,
                force_scale_N=float(scale[idx]),
                old_error_scaled=float((old[idx]-true[idx])/scale[idx]) if mask[idx] else None,
                new_error_scaled=float((new[idx]-true[idx])/scale[idx]) if mask[idx] else None,
                old_coupling_normalized=float((FE[idx]+old[idx])/scale[idx]),
                new_coupling_normalized=float((FE[idx]+new[idx])/scale[idx])))
    def res(v):return mode_rms((FE + v) / scale, np.zeros(26), 1.)
    metrics = dict(old_solver_saved_q_RMS=res(old), actual_NO_Abaqus_saved_q_RMS=res(oracle),
         new_force_head_saved_q_RMS_ESTIMATE=res(new),
         old_vs_Abaqus_true_inner_scaled_RMS=mode_rms(old[mask], true[mask], scale[mask]),
         new_vs_Abaqus_true_inner_scaled_RMS=mode_rms(new[mask], true[mask], scale[mask]),
)
    report = dict(step="65AM-B9A7",phase="saved-65V-anchor-DIAGNOSTIC-ONLY", metrics=metrics,
          regions=cases, anchor_sha256=anchor_hash,
          limitations=["B9A3 anchor was NOT used to fit or select either force head",
                       "These are saved-state estimates, NOT newly converged coupled solutions",
                       "The B5 displacement model is unchanged; this is a separate learned force head",
                       "Do NOT replace the 65V interface force head without multiple-case Abaqus validation and live solver checks"],
          live_Broyden_approved=False)
    rows_csv(targets[0], rows)
    save_json(targets[1], report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = [r for r in rows if r["replaced"]]
    olderr = [abs(r["old_error_scaled"]) for r in ordered]
    newerr = [abs(r["new_error_scaled"]) for r in ordered]
    labels = [r["interface"] + "/m" + str(r["mode"]) for r in ordered]
    xplot = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(xplot - .19, olderr, width=.38, label="Original NO vs true Abaqus")
    ax.bar(xplot + .19, newerr, width=.38, label="B9A7 head vs true Abaqus")
    ax.set_xticks(xplot);ax.set_xticklabels(labels, rotation=50, ha="right")
    ax.set_ylabel("Absolute generalized-force error / original 65V force scale")
    ax.set_title("B9A7: saved-state mode errors (diagnostic only)")
    ax.legend(); fig.tight_layout()
    fig.savefig(targets[2], dpi=180);plt.close(fig)
    print("B9A7 ANCHOR REPLAY: 65V=%.6f true_Abaqus=%.6f proposed_head=%.6f" %(
         metrics["old_solver_saved_q_RMS"], metrics["actual_NO_Abaqus_saved_q_RMS"], metrics["new_force_head_saved_q_RMS_ESTIMATE"]))
    print("Old vs true scaled RMS %.6f | New vs true scaled RMS %.6f" %(
         metrics["old_vs_Abaqus_true_inner_scaled_RMS"],metrics["new_vs_Abaqus_true_inner_scaled_RMS"]))
    print("NOT a converged B5 coupling result; no changes to original solver.")


def selftest():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((50, 5))
    y = 2 + x @ rng.standard_normal((5, 4))
    W = fit_ridge(x, y, 1e-5)
    assert mode_rms(linpred(x, W), y, np.ones(4)) < 1e-5
    zeros = {"W1":np.zeros((48, 5)), "b1":np.zeros(48),
             "W2":np.zeros((48, 48)), "b2":np.zeros(48),
             "W3":np.zeros((4, 48)), "b3":np.zeros(4)}
    assert nn_pred(x, zeros).shape == (50, 4)
    assert np.allclose(nn_pred(x, zeros), 0.)
    st = dict(x_mean=np.zeros(5),x_scale=np.ones(5),g_mean=np.zeros(4),g_scale=np.ones(4),
              ridge_weights=W,model_type=np.array("ridge"))
    assert infer(st, x).shape == y.shape
    B,_ = np.linalg.qr(rng.standard_normal((336, 5)))
    q, f = rng.standard_normal(5), rng.standard_normal(336)
    assert np.isclose((B @ q) @ f, q @ (B.T @ f), atol=1e-10)
    print("B9A7 SELFTEST PASS: normalized regression, checkpoint inference, output dimensions")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default=".")
    p.add_argument("--stage", required=True, choices=("selftest", "check", "train", "anchor"))
    p.add_argument("--epochs", type=int, default=750)
    p.add_argument("--patience", type=int, default=160)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    a = p.parse_args()
    if a.stage == "selftest":return selftest()
    root = Path(a.project).expanduser().resolve()
    if a.stage == "check":return stage_check(root)
    device = a.device
    if a.stage == "train":
        import torch
        if device == "auto":device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device not available. Use --device cpu")
        if a.epochs < 50 or a.patience < 20:raise ValueError("Insufficient epochs or patience")
        print("Training small force head on:",device)
        return stage_train(root, a.epochs, a.patience, device)
    if a.stage == "anchor":return stage_anchor(root)

if __name__ == "__main__":main()
