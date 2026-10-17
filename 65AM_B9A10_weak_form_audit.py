#!/usr/bin/env python3
"""65AM-B9A10-A: verify C3D8R weak-form nodal-force assembly (MAC ONLY).

Reads ONLY existing exported arrays; never runs Abaqus, modifies model checkpoints,
changes saved 65V, or creates/clears project folders.

Usage (from DeepONet_Composite):
  python3 65AM_B9A10_weak_form_audit.py --selftest
  python3 65AM_B9A10_weak_form_audit.py --project . --check-only
  python3 65AM_B9A10_weak_form_audit.py --project .

This diagnostic compares (1) FE one-point weak-form internal nodal forces from
true Abaqus displacement and Cauchy stress, and (2) the full nodal reaction
forces from B9A6 all-node-displacement Abaqus replay. The latter includes
solver contributions absent from the simplified one-point assembly.

Free-DOF residual and constrained reactions are reported separately.
The code DOES NOT distinguish which portion of the force gap is caused by
hourglass stabilization; it is NOT a verified differentiable physics loss.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ("inner_left", "inner_right")
INTERFACES = {
    "inner_left": ("left_inner", "center_left"),
    "inner_right": ("center_right", "right_inner"),
}
SLICES = {
    "left_inner": slice(4, 8),
    "center_left": slice(8, 13),
    "center_right": slice(13, 18),
    "right_inner": slice(18, 22),
}
OUTDIR = Path("results/65AM_B9_coupled_feasibility/B9A10_verified_weak_form")

# Abaqus C3D8R standard node ordering in the natural reference cube.
DN_DR = (np.asarray([
    [-1., -1., -1.], [1., -1., -1.],
    [1., 1., -1.], [-1., 1., -1.],
    [-1., -1., 1.], [1., -1., 1.],
    [1., 1., 1.], [-1., 1., 1.],
], dtype=np.float64) / 8.)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def read_npz(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def safe_norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=float).ravel()))


def relative_pct(pred, truth):
    return 100. * safe_norm(np.asarray(pred) - np.asarray(truth)) / max(safe_norm(truth), 1.e-12)


def scaled_rms(delta, scale):
    delta = np.asarray(delta, float).ravel()
    scale = np.asarray(scale, float).ravel()
    if delta.shape != scale.shape or not np.isfinite(delta).all() or (scale <= 0).any():
        raise RuntimeError("Invalid signed force difference or original 65V scales")
    return float(np.sqrt(np.mean((delta / scale)**2)))


def require_match(path, manifest, field):
    s = digest(path)
    if manifest.get(field) != s:
        raise RuntimeError("SHA256 mismatch: %s | %s" % (str(path), field))
    return s


def reorder_by_labels(values, current_labels, desired_labels):
    current_labels = np.asarray(current_labels, np.int64)
    desired_labels = np.asarray(desired_labels, np.int64)
    if len(set(current_labels.tolist())) != len(current_labels):
        raise RuntimeError("Duplicate source node labels")
    lookup = {int(k): i for i, k in enumerate(current_labels)}
    try:
        index = [lookup[int(k)] for k in desired_labels]
    except KeyError as exc:
        raise RuntimeError("Unexpected node label: %s" % str(exc)) from exc
    if len(set(index)) != len(index) or len(index) != len(current_labels):
        raise RuntimeError("Incomplete node matching")
    return np.asarray(values)[index]


def assemble_c3d8r_weak(x, u, element_indices, stress):
    """Return nodal one-point force, element volumes, J(deformation), and detF.

    Reference shape gradients grad_X N_a = dN/dr * inverse(dX/dr).
    F = I + sum_a (u_a outer grad_X N_a).
    P = det(F) * sigma * F^{-T}; f_int^a = 8det(dX/dr) * P grad_X N_a.
    Assumes S is Cauchy stress in global (x,y,z) axes, MPa; X,U are mm;
    force units N. NO hourglass, contact, external force, or geometric follower
    terms are added by this simplified element-level formula.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    conn = np.asarray(element_indices, dtype=np.int64)
    stress = np.asarray(stress, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or u.shape != x.shape:
        raise RuntimeError("Invalid nodal coordinate/displacement dimensions")
    if conn.ndim != 2 or conn.shape[1] != 8 or stress.shape != (len(conn), 6):
        raise RuntimeError("C3D8R elements require connectivity (E,8) and S (E,6)")
    if np.any(conn < 0) or np.any(conn >= len(x)):
        raise RuntimeError("Bad FE connectivity")
    if not all(np.isfinite(t).all() for t in (x, u, stress)):
        raise RuntimeError("Nonfinite FE values")
    forces = np.zeros_like(x)
    vols, jacobians = np.empty(len(conn)), np.empty(len(conn))
    for e, row in enumerate(conn):
        X = x[row]
        du = u[row]
        J0 = DN_DR.T @ X
        detJ0 = float(np.linalg.det(J0))
        if detJ0 <= 0:
            raise RuntimeError("Negative/zero reference Jacobian, element index %d: %g" % (e, detJ0))
        grad = DN_DR @ np.linalg.inv(J0)
        F = np.eye(3) + du.T @ grad
        j = float(np.linalg.det(F))
        if j <= 0 or not np.isfinite(j):
            raise RuntimeError("Nonphysical deformation determinant at element %d" % e)
        a, b, c, d, h, k = stress[e]
        sigma = np.asarray([[a, d, h], [d, b, k], [h, k, c]])
        P = j * sigma @ np.linalg.inv(F).T
        vol = 8. * detJ0
        elem_force = vol * (P @ grad.T).T  # (8,3)
        np.add.at(forces, row, elem_force)
        vols[e], jacobians[e] = vol, j
    return forces, vols, jacobians


def face_indices(x, iface_coordinates, tol=2.e-5):
    x = np.asarray(x, float)
    coords = np.asarray(iface_coordinates, float)
    if coords.shape != (112, 3):
        raise RuntimeError("Expected 112 PCA interface nodes")
    idx = []
    for q in coords:
        dist = np.linalg.norm(x - q[None], axis=1)
        k = int(np.argmin(dist))
        if dist[k] > tol:
            raise RuntimeError("Interface coordinates differ from Abaqus mesh (gap %.5g mm)" % dist[k])
        idx.append(k)
    if len(set(idx)) != 112:
        raise RuntimeError("Interface PCA node correspondence is not unique")
    return np.asarray(idx, dtype=int)


def input_files(root):
    return {
        "anchor": root/"B9_coupled_anchor/B9_65V_anchor.npz",
        "anchor_manifest": root/"B9_coupled_anchor/B9_65V_anchor_manifest.json",
        "reference": root/"B9A4_exports/B9A4_reference_fields.npz",
        "reference_manifest": root/"B9A4_exports/B9A4_windows_summary.json",
        "transfer": root/"B9A6_transfer_to_windows/B9A6_kinematic_fields.npz",
        "transfer_manifest": root/"B9A6_transfer_to_windows/B9A6_mac_manifest.json",
        "replay": root/"B9A6_exports/B9A6_kinematic_RF.npz",
        "replay_manifest": root/"B9A6_exports/B9A6_windows_summary.json",
    }


def load_verified(root):
    p = input_files(root)
    missing = [str(v) for v in p.values() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Required B9A4/B9A6 files missing:\n" + "\n".join(missing))
    am = read_json(p["anchor_manifest"])
    rm = read_json(p["reference_manifest"])
    tm = read_json(p["transfer_manifest"])
    wm = read_json(p["replay_manifest"])
    ah = require_match(p["anchor"], am, "anchor_sha256")
    require_match(p["reference"], rm, "reference_fields_sha256")
    require_match(p["transfer"], tm, "transfer_sha256")
    require_match(p["replay"], wm, "output_sha256")
    if any(m.get("anchor_sha256") != ah for m in (rm, tm, wm)):
        raise RuntimeError("B9A4 or B9A6 exported from a different saved 65V anchor")
    if wm.get("transfer_sha256") != digest(p["transfer"]):
        raise RuntimeError("B9A6 Abaqus replay does not match the B5/FEM displacement export")
    anchor, ref = read_npz(p["anchor"]), read_npz(p["reference"])
    transfer, replay = read_npz(p["transfer"]), read_npz(p["replay"])
    if anchor["force_scale"].shape != (26,) or (anchor["force_scale"] <= 0).any():
        raise RuntimeError("Original 65V force scales are incorrect")
    for reg in REGIONS:
        for k in ("node_labels", "node_xyz", "U", "RF", "connectivity_labels", "S"):
            if reg + "_" + k not in ref:
                raise RuntimeError("B9A4 missing %s_%s" % (reg, k))
        for tag in ("FEM", "B5"):
            if reg+"_"+tag+"_RF_nodal" not in replay or reg+"_U_"+tag not in transfer:
                raise RuntimeError("Missing B9A6 full-nodal RF/U for %s %s" % (reg, tag))
        if str(transfer["anchor_sha256"].item()) != ah or str(replay["anchor_sha256"].item()) != ah:
            raise RuntimeError("Embedded NPZ anchor mismatch")
    return anchor, ref, transfer, replay, ah, wm


def region_audit(anchor, ref, transfer, replay, region):
    nlab = np.asarray(ref[region+"_node_labels"], int)
    xyz = np.asarray(ref[region+"_node_xyz"], float)
    uref = np.asarray(ref[region+"_U"], float)
    stress = np.asarray(ref[region+"_S"], float)
    con_labels = np.asarray(ref[region+"_connectivity_labels"], int)
    tlabels = np.asarray(transfer[region+"_node_labels"], int)
    tx = reorder_by_labels(transfer[region+"_node_xyz"], tlabels, nlab)
    uf = reorder_by_labels(transfer[region+"_U_FEM"], tlabels, nlab)
    ub = reorder_by_labels(transfer[region+"_U_B5"], tlabels, nlab)
    f_fem = reorder_by_labels(replay[region+"_FEM_RF_nodal"], tlabels, nlab)
    f_b5 = reorder_by_labels(replay[region+"_B5_RF_nodal"], tlabels, nlab)
    if xyz.shape != (1008,3) or stress.shape != (728,6) or con_labels.shape != (728,8):
        raise RuntimeError("Unexpected B9A3 region nodal/element count: " + region)
    if np.max(np.abs(tx-xyz)) > 2.e-5 or np.max(np.abs(uf-uref)) > 2.e-5:
        raise RuntimeError("B9A4 FEM reference does not match B9A6 kinematic replay: " + region)
    if not all(np.isfinite(t).all() for t in (ub, f_fem, f_b5)):
        raise RuntimeError("Invalid replay values")
    lookup = {int(k): i for i,k in enumerate(nlab)}
    if len(lookup) != len(nlab):
        raise RuntimeError("Repeated FE node labels")
    try:
        con = np.asarray([[lookup[int(k)] for k in elem] for elem in con_labels], int)
    except KeyError as exc:
        raise RuntimeError("Unknown Abaqus element node label %s" % str(exc)) from exc
    weak, vol, detf = assemble_c3d8r_weak(xyz, uref, con, stress)
    bc = np.zeros(1008, dtype=bool)
    interfaces = {}
    for interface in INTERFACES[region]:
        coords = anchor["pca_coordinates_"+interface]
        Q = np.asarray(anchor["pca_basis_"+interface], float)
        sl = SLICES[interface]
        ids = face_indices(xyz, coords)
        bc[ids] = True
        if Q.shape != (112*3,sl.stop-sl.start):
            raise RuntimeError("PCA basis shape mismatch " + interface)
        interfaces[interface] = (ids, Q, sl)
    if int(bc.sum()) != 224:
        raise RuntimeError("Constrained-interface mask must cover 224 distinct nodes")
    free = ~bc
    RF_original = np.asarray(ref[region+"_RF"], float)
    face_control = relative_pct(f_fem[bc], RF_original[bc])
    if face_control > .1:
        raise RuntimeError("FEM replay control failed (%.6f%%) in %s; STOP" % (face_control, region))
    norm_boundary = max(safe_norm(f_fem[bc]), 1.e-12)
    rx_fem_free = 100.*safe_norm(f_fem[free])/norm_boundary
    rf_b5_free = 100.*safe_norm(f_b5[free])/norm_boundary
    if rx_fem_free > 1.0:
        raise RuntimeError("FEM all-node replay not near free-node equilibrium in %s: %.4f%% of interface reaction norm. Stop before attributing B5 error." % (region,rx_fem_free))
    weak_free = 100.*safe_norm(weak[free])/norm_boundary
    weak_vs_replay_all = relative_pct(weak, f_fem)
    weak_vs_replay_bound = relative_pct(weak[bc], f_fem[bc])
    # This is a numerical check, not a claim that all terms are hourglass forces.
    remainder = f_fem - weak
    region_result = {
        "region": region,
        "node_count": 1008,
        "element_count": 728,
        "interface_node_count": int(bc.sum()),
        "free_node_count": int(free.sum()),
        "max_geometry_difference_mm": float(np.max(np.abs(tx-xyz))),
        "max_reference_displacement_difference_mm": float(np.max(np.abs(uf-uref))),
        "FE_replay_to_original_Abaqus_reaction_pct": face_control,
        "FEM_replay_free_nodal_RF_norm_pct_of_boundary_RF": rx_fem_free,
        "B5_replay_free_nodal_RF_norm_pct_of_boundary_RF": rf_b5_free,
        "weak_FE_free_nodal_force_norm_pct_of_boundary_RF": weak_free,
        "weak_FE_vs_exact_replay_all_nodes_pct": weak_vs_replay_all,
        "weak_FE_vs_exact_replay_boundary_nodes_pct": weak_vs_replay_bound,
        "unmodelled_force_gap_norm_N": safe_norm(remainder),
        "unmodelled_force_gap_free_norm_N": safe_norm(remainder[free]),
        "reference_volume_mm3": float(vol.sum()),
        "min_deformation_J": float(detf.min()),
        "max_deformation_J": float(detf.max()),
        "B5_vs_FEM_disp_pct": relative_pct(ub, uref),
    }
    mode_rows, iface_rows = [], []
    for name, (ids, Q, sl) in interfaces.items():
        g_weak = Q.T @ weak[ids].ravel()
        g_fem = Q.T @ f_fem[ids].ravel()
        g_b5 = Q.T @ f_b5[ids].ravel()
        g_original = Q.T @ RF_original[ids].ravel()
        scl = anchor["force_scale"][sl]
        if relative_pct(g_fem, g_original) > .1:
            raise RuntimeError("Projected FEM replay fails original reference control for " + name)
        iface_rows.append({
            "region": region,
            "interface": name,
            "modes": int(sl.stop-sl.start),
            "FEM_weak_vs_Abaqus_replay_relative_pct": relative_pct(g_weak,g_fem),
            "FEM_weak_vs_Abaqus_replay_65V_scaled_RMS": scaled_rms(g_weak-g_fem,scl),
            "B5_Abaqus_replay_vs_FEM_replay_65V_scaled_RMS": scaled_rms(g_b5-g_fem,scl),
            "FEM_weak_sign_flipped_relative_pct": relative_pct(-g_weak,g_fem),
        })
        for j in range(len(scl)):
            mode_rows.append({
                "region": region,
                "interface": name,
                "mode": j+1,
                "65V_force_scale_N": float(scl[j]),
                "true_FEM_Abaqus_RF_N": float(g_fem[j]),
                "FEM_weak_onepoint_N": float(g_weak[j]),
                "B5_Abaqus_kinematic_RF_N": float(g_b5[j]),
                "weak_reconstruction_gap_N": float(g_fem[j]-g_weak[j]),
                "B5_field_force_gap_N": float(g_b5[j]-g_fem[j]),
                "weak_reconstruction_gap_scaled": float((g_fem[j]-g_weak[j])/scl[j]),
                "B5_field_force_gap_scaled": float((g_b5[j]-g_fem[j])/scl[j]),
            })
    return region_result, iface_rows, mode_rows


def selftest():
    # Constant (reference-config) stress in an undeformed single C3D8R element.
    x=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    u=np.zeros_like(x)
    s=np.asarray([[3.,2.,1.,0.,0.,0.]])
    f,vol,j=assemble_c3d8r_weak(x,u,np.arange(8)[None,:],s)
    assert np.allclose(vol,[1.]),np.allclose(j,[1.])
    assert np.allclose(f.sum(axis=0),np.zeros(3),atol=1e-12)
    # Virtual work for virtual displacement v=X: integral P:I dV = tr(P)*V = 6.
    assert abs(float(np.sum(f*x))-6.)<1e-12
    # Two equivalent constant stress configurations of opposite sign invert f.
    opposite,*_=assemble_c3d8r_weak(x,u,np.arange(8)[None,:],-s)
    assert np.allclose(opposite,-f)
    labs=np.array([100+i for i in range(8)])
    shuffled=np.array([3,6,4,2,5,0,7,1]);result=reorder_by_labels(f[shuffled],labs[shuffled],labs)
    assert np.allclose(result,f)
    assert abs(scaled_rms([.2,-.4],[2.,4.])-.1)<1e-12
    print("B9A10 SELFTEST PASS: C3D8R weak form, Piola virtual work, signs, node alignment")


def run(root, check_only):
    a,ref,transfer,replay,hash_anchor,meta=load_verified(root)
    region_rows, interface_rows, mode_rows = [], [], []
    for r in REGIONS:
        reg, ifs, mods = region_audit(a,ref,transfer,replay,r)
        region_rows.append(reg)
        interface_rows.extend(ifs)
        mode_rows.extend(mods)
        print("%-11s | replay control %.6f%% | FE weak vs replay %.4f%% | free weak %.4f%% | B5 replay free %.4f%%" % (
            r, reg["FE_replay_to_original_Abaqus_reaction_pct"],
            reg["weak_FE_vs_exact_replay_boundary_nodes_pct"],
            reg["weak_FE_free_nodal_force_norm_pct_of_boundary_RF"],
            reg["B5_replay_free_nodal_RF_norm_pct_of_boundary_RF"]))
    if check_only:
        print("B9A10 CHECK ONLY PASS; NO FILES WRITTEN")
        return
    out = root/OUTDIR
    if not out.is_dir():
        raise RuntimeError("In Finder, create this output folder first: " + str(out))
    names=["B9A10_regions.csv", "B9A10_interfaces.csv", "B9A10_modes.csv", "B9A10_summary.json", "B9A10_force_modes.png"]
    for name in names:
        if (out/name).exists():
            raise RuntimeError("Existing B9A10 result protected; do not overwrite " + str(out/name))
    rdf=pd.DataFrame(region_rows)
    idf=pd.DataFrame(interface_rows)
    mdf=pd.DataFrame(mode_rows)
    # Report original 65V scale modes, but not an online Broyden residual.
    inner_scale=np.concatenate([a["force_scale"][SLICES[i]] for r in REGIONS for i in INTERFACES[r]])
    weak_delta=mdf["weak_reconstruction_gap_N"].to_numpy(float)
    b5_delta=mdf["B5_field_force_gap_N"].to_numpy(float)
    all_weak_rms=scaled_rms(weak_delta,inner_scale)
    all_b5_rms=scaled_rms(b5_delta,inner_scale)
    summary={
        "step":"65AM-B9A10-A",
        "method":"C3D8R weak-form Piola stress integral vs actual all-node Abaqus RF",
        "anchor_sha256":hash_anchor,
        "reference_replay_control_pass":bool(all(r["FE_replay_to_original_Abaqus_reaction_pct"]<.1 and r["FEM_replay_free_nodal_RF_norm_pct_of_boundary_RF"]<1.0 for r in region_rows)),
        "FEM_weak_minus_exact_replay_65V_scaled_RMS_18_modes":all_weak_rms,
        "B5_Abaqus_replay_minus_FEM_replay_65V_scaled_RMS_18_modes":all_b5_rms,
        "region_audit":region_rows,
        "top_modes_by_absolute_weak_gap":mdf.assign(abs_scaled_gap=mdf.weak_reconstruction_gap_scaled.abs()).sort_values("abs_scaled_gap",ascending=False).head(8)[["interface","mode","weak_reconstruction_gap_scaled"]].to_dict(orient="records"),
        "warning":"Difference FEM replay minus weak one-point force includes ALL missing FE/constitutive/stabilization terms, not proven solely hourglass.",
        "physics_loss_ready":False,
        "live_Broyden_approved":False,
        "models_unchanged":True,
        "no_new_Abaqus_jobs":True,
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labs=["%s/%d"%(n,m) for n,m in zip(mdf.interface,mdf['mode'])]
    fig,ax=plt.subplots(figsize=(max(12,.7*len(labs)),5.4))
    xx=np.arange(len(labs));w=.39
    ax.bar(xx-w/2,mdf.weak_reconstruction_gap_scaled.abs(),w,label='One-point FE gap vs Abaqus RF')
    ax.bar(xx+w/2,mdf.B5_field_force_gap_scaled.abs(),w,label='B5 displacement-induced force gap')
    ax.set_xticks(xx,labs,rotation=55,ha='right');ax.set_ylabel('Absolute difference / original 65V force scale')
    ax.set_title('B9A10-A | FE weak-form force gap vs B5 field-induced gap')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    # Save only after all source/control checks pass.
    rdf.to_csv(out/names[0],index=False)
    idf.to_csv(out/names[1],index=False)
    mdf.to_csv(out/names[2],index=False)
    (out/names[3]).write_text(json.dumps(summary,indent=2))
    fig.savefig(out/names[4],dpi=180);plt.close(fig)
    print("B9A10-A COMPLETE | 18 PCA modes | outputs in",out)
    print("Weak gap scaled RMS %.6f | B5 field gap scaled RMS %.6f"%(all_weak_rms,all_b5_rms))
    print("No force correction was applied. NOT APPROVED for online Broyden.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--project',type=Path,default=Path('.'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.selftest:
        selftest()
        return
    run(args.project.expanduser().resolve(),args.check_only)


if __name__=='__main__':
    main()
