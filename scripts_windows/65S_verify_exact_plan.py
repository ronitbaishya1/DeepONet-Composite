"""65S-C verification: evaluate the EXACT planned five-mode step with 3 real Abaqus patches.

Place next to 65S_A_to_E.py in ODBS/scripts.
Run from ODBS: abaqus python scripts\65S_verify_exact_plan.py --cpus 4
Does not overwrite prior experiments. This does NOT perform optimization.
"""
import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, '65S_A_to_E.py')


def load_65s():
    if not os.path.isfile(SOURCE):
        raise FileNotFoundError('Missing: ' + SOURCE)
    spec = importlib.util.spec_from_file_location('step_65S_for_verification', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpus', type=int, default=4)
    args = p.parse_args()
    s = load_65s()
    if not os.path.isfile(s.PLAN):
        raise FileNotFoundError('Run Step 65S-C --stage plan before verification')
    with open(s.PLAN) as f:
        plan = json.load(f)
    q0, saved_r, coefficient_scales, J, force_scales = s.load_calibration()
    indexes = np.asarray(plan['selected_q_indices'], dtype=int)
    delta = np.asarray(plan['proposed_dq'], dtype=float)
    if not np.array_equal(indexes, s.IDX) or delta.shape != (5,):
        raise RuntimeError('Plan indexes or five-mode step do not match calibration')
    if not np.all(np.isfinite(delta)):
        raise RuntimeError('Plan contains invalid step values')
    q_trial = q0.copy()
    q_trial[indexes] += delta
    if np.any(np.abs(q_trial) > 3.0):
        raise RuntimeError('Planned trial exceeds allowed q bounds; do not clip silently')
    work = os.path.join(s.OUT, '65S_C_exact_step_verification')
    report_path = os.path.join(work, 'verification.json')
    if os.path.exists(report_path):
        raise RuntimeError('Existing 65S-C verification detected. No overwriting.')
    if os.path.isdir(work) and os.listdir(work):
        raise RuntimeError('Existing files in verification work directory; inspect before rerunning.')
    ctx, oracle = s.context('65S_C_exact_step_verification', 3, args.cpus)
    s.scales_ok(ctx, coefficient_scales, force_scales)
    # No incumbent_r: this FORCES all three Abaqus patches to run.
    trial = oracle.evaluate(q_trial, incumbent_r=None)
    if not trial['verified'] or oracle.jobs != 3:
        raise RuntimeError('Exact planned trial was not fully verified using three patch jobs')
    actual_r = np.asarray(trial['r'], dtype=float)
    original_rms = s.rms(saved_r)
    result = {
        'material_MPa': s.CASE.tolist(),
        'selected_q_indices': indexes.tolist(),
        'dq': delta.tolist(),
        'starting_verified_rms_from_65R': original_rms,
        '65S_C_linearized_predicted_full_rms': float(plan['full_rms_if_unchanged_other_patch_residuals']),
        'actual_three_patch_full_rms': s.rms(actual_r),
        'starting_center_rms': s.rms(saved_r[s.CENTER]),
        'linearized_predicted_center_rms': float(plan['linearized_center_rms']),
        'actual_center_rms': s.rms(actual_r[s.CENTER]),
        'actual_improvement_percent': 100.0 * (1.0 - s.rms(actual_r) / original_rms),
        'actual_improvement': bool(s.rms(actual_r) < original_rms),
        'true_converged': bool(s.rms(actual_r) <= 0.02),
        'abaqus_patch_jobs': oracle.jobs,
        'patch_odb_paths': {k: v.get('odb_file') for k, v in trial['files'].items()},
        'note': 'No partial residual or ROM-only prediction was accepted as a true result.',
    }
    np.savez(os.path.join(work, 'verified_trial.npz'), q=q_trial,
             normalized_residual=actual_r, q0=q0, r0=saved_r)
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
