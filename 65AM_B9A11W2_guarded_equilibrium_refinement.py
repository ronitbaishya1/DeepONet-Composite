#!/usr/bin/env python3
"""B9A11W2: guarded, read-only fixed-q reduced-basis equilibrium experiment.

Requirements: existing B9A11W0_weak_form_gate.py and original B9A9/H exports,
B9A10L/K approximate force law, B5 + B9A11U pilot checkpoints.

W2-A: screen Abaqus FEM reference using W0's independent reference reactions.
      Passing reference fields are *not optimized* (an explicit safety gate).
W2-B: optional reduced-basis refinement of original and trained B5 on the SAME q.
W2-C: report unmodified/refined free residual, displacement error, and PCA forces.

No checkpoints altered. Does NOT imply Abaqus-exact physics, FE-NO re-equilibration,
strain-gauge measurements, or readiness for production inverse identification.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REGIONS = (("inner_left", "L"), ("inner_right", "R"))
OUT = "65AM_B9A11W2_guarded_refinement"
MODELS = ("original_B5", "trained_B5")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Refusing to write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False))


def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a * a)))


def normalized_free_norm(f, boundary):
    f = np.asarray(f, float)
    base = float(np.linalg.norm(f[boundary]))
    if not np.isfinite(base) or base < 1e-10:
        raise RuntimeError("Predicted interface force norm not finite/positive")
    return float(np.linalg.norm(f[~boundary]) / base)


def make_basis(basis, node_map, boundary, nmode):
    """Normalize existing B5 free-field modes to RMS 1 mm (unit coefficients: mm)."""
    V = np.asarray(basis[:nmode], float)[:, node_map, :].copy()
    if V.ndim != 3 or V.shape[2] != 3:
        raise RuntimeError("Unexpected B5 basis shape")
    if np.max(np.abs(V[:, boundary, :])) > 1e-7:
        raise RuntimeError("B5 basis changes prescribed q; refusing refinement")
    norms = np.sqrt(np.mean(V[:, ~boundary, :] ** 2, axis=(1, 2)))
    keep = np.isfinite(norms) & (norms > 1e-12)
    if not keep.any():
        raise RuntimeError("No usable B5 displacement modes")
    corrected = V[keep] / norms[keep, None, None]
    corrected[:,boundary,:] = 0.0  # Exact Dirichlet preservation, independent of float32 noise.
    return corrected


def numerical_force_jacobian(force_fun, u, V, boundary, step_mm):
    """Columns d(f_free)/d(alpha_i) for correction alpha_i*V_i, alpha in mm."""
    cols = []
    for v in V:
        fp = force_fun(u + step_mm * v)
        fm = force_fun(u - step_mm * v)
        col = ((fp - fm) / (2 * step_mm))[~boundary].reshape(-1)
        if not np.isfinite(col).all():
            raise RuntimeError("Nonfinite force Jacobian")
        cols.append(col)
    return np.stack(cols, axis=1)


def refinement(u, V, boundary, force_fun, *, cap_mm, fd_mm, ridge,
               steps, min_rel_decrease):
    """Optimize candidate free-node equilibrium only. No FEM truth enters solver."""
    u = np.asarray(u, float)
    u0 = u.copy()
    starting = force_fun(u)
    old_score = normalized_free_norm(starting, boundary)
    fixed_scale = max(float(np.linalg.norm(starting[boundary])), 1e-10)
    current = float(np.linalg.norm(starting[~boundary]) / fixed_scale)
    trace = []
    remaining = cap_mm
    for iteration in range(steps):
        if remaining <= 1e-12:
            break
        force = force_fun(u)
        scale = fixed_scale  # Prevent a false gain by increasing boundary-force denominator.
        jac = numerical_force_jacobian(force_fun, u, V, boundary, fd_mm)
        # t = alpha/remaining is dimensionless; ridge penalizes ||t||^2.
        A = (jac / scale) * remaining
        b = -force[~boundary].reshape(-1) / scale
        augmented = np.vstack((A, math.sqrt(ridge) * np.eye(len(V))))
        rhs = np.r_[b, np.zeros(len(V))]
        t = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        correction = remaining * np.einsum("k,kij->ij", t, V)
        magnitude = rms(correction)
        if magnitude > remaining:
            correction *= remaining / magnitude
        if np.max(np.abs(correction[boundary])) > 1e-9:
            raise RuntimeError("Correction alters prescribed interface")
        chosen = None
        for factor in (1.0, 0.5, 0.25, 0.125):
            trial = u + factor * correction
            try:
                f = force_fun(trial)
                score = float(np.linalg.norm(f[~boundary]) / fixed_scale)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if np.isfinite(score) and score < current * (1 - min_rel_decrease):
                chosen = (trial, score, factor)
                break
        if chosen is None:
            trace.append({"iteration": iteration, "accepted": False,
                          "free_force_ratio": current})
            break
        u, current, factor = chosen
        remaining = max(0.0, cap_mm - rms(u - u0))
        trace.append({"iteration": iteration, "accepted": True,
                      "free_force_ratio": current, "line_search": factor,
                      "correction_rms_mm": rms(u - u0)})
    delta = u - u0
    if rms(delta) > cap_mm * (1 + 1e-5):
        # Multiple correction steps could in principle align and exceed cap;
        # rescale globally and recheck instead of silently violating it.
        delta *= cap_mm / rms(delta)
        u = u0 + delta
        current = float(np.linalg.norm(force_fun(u)[~boundary]) / fixed_scale)
    return u, dict(before_free_force_ratio=old_score,
                   after_free_force_ratio=current,
                   free_force_change_pct=100 * (current / old_score - 1) if old_score else 0.,
                   correction_rms_mm=rms(u - u0),
                   accepted_steps=sum(int(z["accepted"]) for z in trace),
                   trace=trace)


def force_model_fem_and_b5(ctx, data, material):
    import torch
    L = ctx["L"]
    model = L.ConstitutiveC3D8R(
        data["node_labels"], data["X"], data["connectivity"],
        ctx["beta"], ctx["root"] / "65AM_B9A10K_hourglass_operator.py",
        device="cpu", dtype=torch.float64
    )
    mat = torch.as_tensor(np.asarray(material, float), dtype=torch.float64)

    def force(u):
        arr = np.asarray(u, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise RuntimeError("Nonfinite U")
        with torch.no_grad():
            ret = model(torch.as_tensor(arr, dtype=torch.float64), mat)
        result = ret.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite candidate internal force")
        return result

    return force


def selftest():
    # A synthetic equilibrium law with 2 free modes and fixed boundary nodes.
    n = 8
    b = np.zeros(n, bool)
    b[[0, 7]] = True
    x = np.linspace(0, 1, n)
    v1 = x * (1 - x)
    v2 = (x * (1 - x)) * (x - 0.5)
    V = np.stack((np.column_stack((v1, v1 * 0, v1 * 0)),
                  np.column_stack((v2, v2 * 0, v2 * 0))))
    V = make_basis(V, np.arange(n), b, 2)
    target = (0.4 * V[0] + 0.22 * V[1]) * 1e-5
    def force(u):
        f = 1e6 * (u - target)
        f[b, 1] = [10., -10.]
        return f
    u0 = np.zeros((n, 3))
    corrected, audit = refinement(u0, V, b, force, cap_mm=1e-5,
                                  fd_mm=1e-6, ridge=1e-4,
                                  steps=2, min_rel_decrease=1e-4)
    assert audit["after_free_force_ratio"] < audit["before_free_force_ratio"]
    assert np.max(abs(corrected[b] - u0[b])) == 0
    assert rms(corrected-u0) <= 1e-5 * 1.00001
    # Reject mode that violates Dirichlet boundary.
    V2 = V.copy(); V2[0,0,0] = 1e-4
    try:
        make_basis(V2, np.arange(n), b, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected prescribed-boundary protection")
    print("B9A11W2 SELFTEST PASS | force FD | guarded update | fixed q | correction cap")


def run(args, write):
    root = Path(args.project).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not (root / "65AM_B9A11W0_weak_form_gate.py").is_file():
        raise FileNotFoundError("Keep W0 script in the Mac project root")
    W0 = importlib.import_module("65AM_B9A11W0_weak_form_gate")
    ctx = W0.context(root, args)
    ctx["root"] = root
    H = ctx["H"]
    rows = []
    modes = []
    traces = []
    gate_records = []
    with np.load(root / W0.H_PATH, allow_pickle=False) as z:
        for region, letter in REGIONS:
            d = ctx["data"][region]
            kdtree = d["cKDTree"](d["coords"])
            for i in range(args.cases):
                job = f"B9A9_{letter}{i+1:02d}"
                a = H.make_case(z, job)
                distances, match = kdtree.query(a["X"], k=1)
                if np.max(distances) > 2e-5 or len(np.unique(match)) != len(match):
                    raise RuntimeError("B5/FEM node mapping mismatch " + job)
                bc = H.boundary(a["X"])
                Uref = np.asarray(a["U"], float)
                if np.max(np.abs(d["lift"][i][match][bc] - Uref[bc])) > args.bc_tolerance_mm:
                    raise RuntimeError("FEM q incompatible with B5 lift: " + job)
                f = force_model_fem_and_b5(ctx, a, d["branch"][i,:3])
                f_ref = f(Uref)
                ref_metrics = W0.metrics(f_ref, a["RF"], bc)
                passed = (ref_metrics["force_gap_pct"] <= args.max_reference_force_pct
                          and ref_metrics["force_free_gap_pct"] <= args.max_reference_free_pct)
                gate_records.append({"job": job, "region": region,
                                     "FEM_reference_force_gap_pct": ref_metrics["force_gap_pct"],
                                     "FEM_reference_free_gap_pct": ref_metrics["force_free_gap_pct"],
                                     "FEM_reference_gate_pass": bool(passed)})
                if not passed:
                    raise RuntimeError("FEM control did not pass; REFUSE B5 refinement: " + job)
                # FEM already approximately equilibrated: explicitly preserve it.
                # This is a threshold safeguard, not evidence the force law is exact.
                rows.append({"job":job,"region":region,"model":"FEM",
                             "phase":"unchanged_reference", "candidate_free_norm_pct":
                             100 * normalized_free_norm(f_ref, bc),
                             "gap_free_pct":ref_metrics["force_free_gap_pct"],
                             "gap_all_pct":ref_metrics["force_gap_pct"],
                             "u_rmse_mm":0.,"u_rel_pct":0.,"interface_RMS_65V":0.,
                             "interface_truth_error_RMS_65V":None,
                             "q_change_max_mm":0.,"correction_rms_mm":0.,
                             "accepted_steps":0})
                for name, pred in (("original_B5",d["u0"]), ("trained_B5",d["u1"])):
                    u0 = np.asarray(pred[i][match],float)
                    if np.max(abs(u0[bc] - Uref[bc])) > args.bc_tolerance_mm:
                        raise RuntimeError("B5 prescribed q mismatch: " + job + " " + name)
                    V = make_basis(d["d"]["basis"], match, bc, args.modes)
                    before = f(u0)
                    norm_before = normalized_free_norm(before,bc)
                    # No use of FEM U or RF in selecting correction, except audit/metrics.
                    if norm_before * 100 <= args.screen_free_pct:
                        u1 = u0.copy()
                        info = {"trace":[],"accepted_steps":0,"correction_rms_mm":0.,
                                "before_free_force_ratio":norm_before,
                                "after_free_force_ratio":norm_before}
                    else:
                        u1,info = refinement(u0,V,bc,f,cap_mm=args.cap_rms_mm,
                                              fd_mm=args.fd_mm,ridge=args.ridge,
                                              steps=args.steps,min_rel_decrease=args.min_rel_decrease)
                    after=f(u1)
                    if np.max(abs(u1[bc] - u0[bc]))>args.q_tolerance_mm:
                        raise RuntimeError("Known-q interface altered: " + job)
                    if rms(u1-u0)>args.cap_rms_mm*(1+1e-5):
                        raise RuntimeError("Maximum correction exceeded")
                    for phase,uf,ff in (("before",u0,before),("after",u1,after)):
                        met = W0.metrics(ff,a["RF"],bc)
                        errors=[]
                        for interface in W0.FACES[letter]:
                            v,scale=H.pca_force(a,ff,ctx["anchor"],interface)
                            t,_=H.pca_force(a,a["RF"],ctx["anchor"],interface)
                            for j, err in enumerate((v-t)/scale,1):
                                value={"job":job,"region":region,"model":name,
                                       "phase":phase,"interface":interface,"mode":j,
                                       "normalized_error":float(err)}
                                modes.append(value)
                                errors.append(float(err))
                        rows.append({"job":job,"region":region,"model":name,"phase":phase,
                                     "candidate_free_norm_pct":100*normalized_free_norm(ff,bc),
                                     "gap_free_pct":met["force_free_gap_pct"],
                                     "gap_all_pct":met["force_gap_pct"],
                                     "u_rmse_mm":rms(uf-Uref),
                                     "u_rel_pct":100*float(np.linalg.norm(uf-Uref))/max(float(np.linalg.norm(Uref)),1e-12),
                                     "interface_RMS_65V":rms(np.asarray(errors)),
                                     "interface_truth_error_RMS_65V":rms(np.asarray(errors)),
                                     "q_change_max_mm":float(np.max(abs(uf[bc]-u0[bc]))),
                                     "correction_rms_mm":rms(uf-u0),
                                     "accepted_steps":info["accepted_steps"]})
                    for entry in info["trace"]:
                        traces.append({"job":job,"region":region,"model":name,
                                       "iteration":entry["iteration"],
                                       "accepted":entry["accepted"],
                                       "free_force_ratio":entry["free_force_ratio"],
                                       "line_search":entry.get("line_search"),
                                       "correction_rms_mm":entry.get("correction_rms_mm")})
                    print("W2",job,name,"free %.3f%% -> %.3f%%" %
                          (100*norm_before,100*normalized_free_norm(after,bc)),
                          "U rmse %.3g -> %.3g"%(rms(u0-Uref),rms(u1-Uref)),flush=True)
    if not write:
        print("W2 CHECK PASS | fixed q | FEM reference gate | forward predictions and basis mapped")
        return
    out=root/"results"/OUT/args.tag
    if out.exists():
        raise FileExistsError("Output already exists; use a new --tag: "+str(out))
    out.mkdir(parents=True)
    write_csv(out/"B9A11W2_FEM_control.csv",gate_records)
    write_csv(out/"B9A11W2_case_before_after.csv",rows)
    write_csv(out/"B9A11W2_interface_modes.csv",modes)
    if traces:write_csv(out/"B9A11W2_optimizer_trace.csv",traces)
    summary = {"stage":"B9A11W2","scope":"Fixed-q diagnostic only",
        "cases_per_region":args.cases,
        "approximate_force_law":"B9A10L + fitted B9A10K hourglass",
        "FEM_reference_gate_pass":all(t["FEM_reference_gate_pass"] for t in gate_records),
        "FEM_reference_preserved_by_explicit_skip":True,
        "B5_checkpoints_unchanged":True,
        "not_approved_for_training_or_inverse_or_coupling":True,
        "settings":{"modes":args.modes,"cap_rms_mm":args.cap_rms_mm,
                    "fd_mm":args.fd_mm,"ridge":args.ridge,"steps":args.steps},
        "regions":{}}
    for region,_ in REGIONS:
        summary["regions"][region]={}
        for model in MODELS:
            rr=[r for r in rows if r["region"]==region and r["model"]==model]
            pre=[r for r in rr if r["phase"]=="before"]
            post=[r for r in rr if r["phase"]=="after"]
            summary["regions"][region][model]={
                "before_free_residual_rms_pct":rms([r["candidate_free_norm_pct"] for r in pre]),
                "after_free_residual_rms_pct":rms([r["candidate_free_norm_pct"] for r in post]),
                "before_FEM_displacement_RMSE_mm":float(np.mean([r["u_rmse_mm"] for r in pre])),
                "after_FEM_displacement_RMSE_mm":float(np.mean([r["u_rmse_mm"] for r in post])),
                "before_interface_PCA_RMS":rms([r["interface_RMS_65V"] for r in pre]),
                "after_interface_PCA_RMS":rms([r["interface_RMS_65V"] for r in post]),
                "n_improved_displacement":int(sum(q["u_rmse_mm"]<p["u_rmse_mm"] for p,q in zip(pre,post))),
                "n_improved_interface":int(sum(q["interface_RMS_65V"]<p["interface_RMS_65V"] for p,q in zip(pre,post))),
                "n_corrected":int(sum(q["accepted_steps"]>0 for q in post))}
    json_write(out/"B9A11W2_summary.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for field,filename,ylabel in (
        ("candidate_free_norm_pct","B9A11W2_free_residual.png","Free-node residual / candidate boundary force (%)"),
        ("u_rmse_mm","B9A11W2_FEM_displacement.png","FEM displacement RMSE (mm)"),
        ("interface_RMS_65V","B9A11W2_interface_force.png","Interface error (65V normalized units)")
    ):
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
        for ax,(region,_) in zip(axes,REGIONS):
            x=np.arange(2)
            for k,phase in enumerate(("before","after")):
                ss=[r for r in rows if r["region"]==region and r["phase"]==phase and r["model"] in MODELS]
                v=[float(np.mean([r[field] for r in ss if r["model"]==name])) for name in MODELS]
                ax.bar(x + (k-.5)*.35,v,width=.35,label=phase)
            ax.set_xticks(x,MODELS,rotation=12)
            ax.set_title(region);ax.set_ylabel(ylabel);ax.legend()
        fig.suptitle("B9A11W2 guarded read-only refinement | NOT validated Abaqus physics")
        fig.tight_layout();fig.savefig(out/filename,dpi=165);plt.close(fig)
    print("W2 DONE",out)
    for r in REGIONS:print(r[0],summary["regions"][r[0]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=["selftest","check","run"])
    parser.add_argument("--project",default=".")
    parser.add_argument("--candidate-tag",default="pilot01")
    parser.add_argument("--tag",default="pilot01")
    parser.add_argument("--cases",type=int,default=2)
    parser.add_argument("--modes",type=int,default=8)
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--fd-mm",type=float,default=1e-6)
    parser.add_argument("--cap-rms-mm",type=float,default=1e-5)
    parser.add_argument("--ridge",type=float,default=1e-4)
    parser.add_argument("--min-rel-decrease",type=float,default=1e-3)
    parser.add_argument("--screen-free-pct",type=float,default=.15)
    parser.add_argument("--max-reference-force-pct",type=float,default=.5)
    parser.add_argument("--max-reference-free-pct",type=float,default=.5)
    parser.add_argument("--bc-tolerance-mm",type=float,default=2e-4)
    parser.add_argument("--q-tolerance-mm",type=float,default=1e-8)
    args=parser.parse_args()
    if args.stage=="selftest":
        selftest();return
    if (not 1<=args.cases<=12 or not 1<=args.modes<=48 or not 1<=args.steps<=5
        or min(args.fd_mm,args.cap_rms_mm,args.ridge,args.min_rel_decrease,args.q_tolerance_mm)<=0):
        parser.error("Invalid experiment parameters")
    run(args,write=(args.stage=="run"))

if __name__=="__main__":main()