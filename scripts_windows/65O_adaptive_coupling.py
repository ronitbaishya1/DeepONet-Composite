
"""
65O — Adaptive, selectively verified FE–NO coupling.

Uses the existing Step 65L Coupling class.

A partially evaluated residual is NEVER labeled converged.
"""

import json
import os

import numpy as np


NAMES = (
    "left_outer",
    "left_inner",
    "center_left",
    "center_right",
    "right_inner",
    "right_outer",
)

PATCHES = ("left", "center", "right")

PATCH_KEYS = {
    "left": ("left_outer", "left_inner"),
    "center": ("center_left", "center_right"),
    "right": ("right_inner", "right_outer"),
}


def rms(v):
    return float(
        np.sqrt(
            np.mean(np.asarray(v, dtype=float) ** 2)
        )
    )


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class CenterPrior:
    """
    Load the experimentally validated 65N center correction.

    It is only permitted at the material used to train it.
    """

    def __init__(self, path, material, c_scale):
        self.available = False

        if not path or not os.path.isfile(path):
            return

        with np.load(path) as m:
            approved = bool(m["approved"])

            same_material = np.allclose(
                material,
                m["material_MPa"],
                rtol=0,
                atol=1e-8,
            )

            if not approved or not same_material:
                print(
                    "65O: Center correction disabled "
                    "(unapproved or different material)."
                )
                return

            self.x_train = m["x_train"].copy()
            self.mean_x = m["mean_x"].copy()
            self.std_x = m["std_x"].copy()
            self.delta_mean = m["delta_mean"].copy()
            self.alpha = m["alpha"].copy()

            self.gamma = float(m["gamma"])

            self.c_scale = np.asarray(
                c_scale, dtype=float
            )[8:18].copy()

            self.available = True

            print(
                "65O: Same-material center correction loaded."
            )

    def predict(self, q):
        if not self.available:
            return np.zeros(10)

        c_center = (
            np.asarray(q)[8:18] * self.c_scale
        )

        x = (
            c_center - self.mean_x
        ) / self.std_x

        distances = (
            (self.x_train - x) ** 2
        ).sum(axis=1) ** 0.5

        # Applicability guard, not a calibrated
        # statistical uncertainty bound.
        if float(np.min(distances)) > 8.0:
            return np.zeros(10)

        k = np.exp(
            -self.gamma
            * ((self.x_train - x) ** 2).sum(axis=1)
        )

        return (
            self.delta_mean
            + k.dot(self.alpha)
        )


class Oracle:
    """
    Adaptive high-fidelity coupling oracle.

    All residuals use the existing 26-dimensional
    normalized generalized-force convention.

    The three FE patches are independent given
    prescribed interface displacement coefficients.

    Partially evaluated states may be rejected,
    but ONLY full evaluations can be accepted
    as verified solutions.
    """

    def __init__(
        self,
        ctx,
        workdir,
        prior_path=None,
        job_budget=36,
        cpus=4,
        adaptive_fidelity=True,
        online_correction=True,
    ):
        self.ctx = ctx
        self.workdir = workdir
        self.cpus = int(cpus)
        self.budget = int(job_budget)

        self.adaptive_fidelity = bool(
            adaptive_fidelity
        )

        self.online_correction = bool(
            online_correction
        )

        self.jobs = 0
        self.evaluation = 0
        self.log = []

        self.samples = {
            patch: []
            for patch in PATCHES
        }

        self.prior = CenterPrior(
            prior_path,
            ctx.material,
            ctx.scale,
        )

        self.ids = {
            name: ctx.slices[name]
            for name in NAMES
        }

        self.patch_idx = {
            patch: np.concatenate([
                np.arange(
                    self.ids[name].start,
                    self.ids[name].stop,
                )
                for name in PATCH_KEYS[patch]
            ])
            for patch in PATCHES
        }

        os.makedirs(
            workdir, exist_ok=True
        )

    # ========================================================
    # OFFLINE CENTER CORRECTION
    # ========================================================

    def prior_patch(self, patch, q):
        if patch == "center":
            return self.prior.predict(q)

        return np.zeros(
            len(self.patch_idx[patch])
        )

    # ========================================================
    # ONLINE CORRECTION FROM ABAQUS OBSERVATIONS
    # ========================================================

    def _online_patch(self, patch, q):
        pts = self.samples[patch]
        prior = self.prior_patch(patch, q)

        if not pts:
            return prior

        q = np.asarray(q)

        anchor = min(
            pts,
            key=lambda item:
                np.linalg.norm(
                    item["q"] - q
                ),
        )

        # Fit changes in discrepancy relative to
        # the closest observed state.
        dq = [
            point["q"] - anchor["q"]
            for point in pts
            if point is not anchor
        ]

        dy = [
            (
                point["delta"]
                - self.prior_patch(
                    patch, point["q"]
                )
            )
            -
            (
                anchor["delta"]
                - self.prior_patch(
                    patch, anchor["q"]
                )
            )
            for point in pts
            if point is not anchor
        ]

        correction = (
            anchor["delta"]
            - self.prior_patch(
                patch, anchor["q"]
            )
        ).copy()

        if dq:
            X = np.asarray(dq)
            Y = np.asarray(dy)

            # Low-rank, regularized local correction.
            B = (
                X.T
                @ np.linalg.solve(
                    X @ X.T
                    + 0.08 * np.eye(len(X)),
                    Y,
                )
            ).T

            operator_norm = float(
                np.linalg.norm(B, 2)
            )

            if operator_norm > 2.0:
                B *= 2.0 / operator_norm

            correction += B @ (
                q - anchor["q"]
            )

        return prior + correction

    # ========================================================
    # CHEAP CORRECTED RESIDUAL
    # ========================================================

    def corrected(self, q):
        r = np.asarray(
            self.ctx.rom(q),
            dtype=float,
        ).copy()

        if self.online_correction:
            for patch in PATCHES:
                idx = self.patch_idx[patch]

                r[idx] += self._online_patch(
                    patch, q
                )

        return r

    # ========================================================
    # RUN ONE ACTUAL ABAQUS PATCH
    # ========================================================

    def _patch(self, patch, q, serial, coeff):
        if self.jobs >= self.budget:
            raise RuntimeError(
                "65O Abaqus job budget exhausted."
            )

        directory = os.path.join(
            self.workdir,
            "evaluation_%03d" % serial,
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        result = self.ctx.run_patch_job(
            patch_name=patch,

            E1=float(self.ctx.material[0]),
            E2=float(self.ctx.material[1]),
            G12=float(self.ctx.material[2]),

            coefficient_dictionary=coeff,
            job_directory=directory,

            job_name="n65_%03d_%s" % (
                serial, patch
            ),

            cpus=self.cpus,
        )

        if "generalized_forces" not in result:
            raise RuntimeError(
                "Expected generalized_forces "
                "from Abaqus patch runtime."
            )

        forces = {
            name: np.asarray(
                result["generalized_forces"][name],
                dtype=float,
            ).ravel()

            for name in PATCH_KEYS[patch]
        }

        self.jobs += 1

        return result, forces

    # ========================================================
    # ADAPTIVE HIGH-FIDELITY VERIFICATION
    # ========================================================

    def evaluate(self, q, incumbent_r=None):
        q = np.asarray(
            q, dtype=float
        ).copy().ravel()

        if q.shape != (26,):
            raise ValueError(
                "Expected 26 interface coordinates."
            )

        if self.budget - self.jobs < 3:
            raise RuntimeError(
                "Not enough Abaqus budget for "
                "a fully verified trial."
            )

        serial = self.evaluation
        self.evaluation += 1

        _, _, rom_g, no_g = self.ctx.rom(
            q, detailed=True
        )

        corrected_prediction = (
            self.corrected(q)
        )

        coeff = self.ctx.coeffs(q)

        if incumbent_r is None:
            order = list(PATCHES)

        else:
            # Check the most influential FE block first.
            order = sorted(
                PATCHES,
                reverse=True,
                key=lambda patch: max(
                    float(
                        np.dot(
                            incumbent_r[
                                self.patch_idx[patch]
                            ],
                            incumbent_r[
                                self.patch_idx[patch]
                            ],
                        )
                    ),
                    float(
                        np.dot(
                            corrected_prediction[
                                self.patch_idx[patch]
                            ],
                            corrected_prediction[
                                self.patch_idx[patch]
                            ],
                        )
                    ),
                ),
            )

        r = np.empty(26)
        partial_sse = 0.0

        files = {}
        checked = []

        for patch in order:
            record, forces = self._patch(
                patch,
                q,
                serial,
                coeff,
            )

            files[patch] = record
            checked.append(patch)

            idx = self.patch_idx[patch]

            joined = np.concatenate([
                forces[name]
                + np.asarray(no_g[name])

                for name in PATCH_KEYS[patch]
            ])

            rp = (
                joined
                / self.ctx.force_scale[idx]
            )

            if (
                len(rp) != len(idx)
                or not np.all(np.isfinite(rp))
            ):
                raise RuntimeError(
                    "Invalid residual for patch "
                    + patch
                )

            r[idx] = rp

            partial_sse += float(
                np.dot(rp, rp)
            )

            rom_block = np.concatenate([
                np.asarray(rom_g[name])
                for name in PATCH_KEYS[patch]
            ])

            actual_block = np.concatenate([
                forces[name]
                for name in PATCH_KEYS[patch]
            ])

            delta = (
                actual_block - rom_block
            ) / self.ctx.force_scale[idx]

            # Store the Abaqus information even
            # if this trial is later rejected.
            self.samples[patch].append({
                "q": q.copy(),
                "delta": delta.copy(),
            })

            # Safe early rejection from partial
            # exact sum-of-squared residuals.
            if (
                self.adaptive_fidelity
                and incumbent_r is not None
                and len(checked) < 3
            ):
                incumbent_sse = float(
                    np.dot(
                        incumbent_r,
                        incumbent_r,
                    )
                )

                if (
                    partial_sse
                    >= incumbent_sse - 1e-11
                ):
                    output = {
                        "serial": serial,
                        "q": q,
                        "verified": False,
                        "status": "safely_pruned",
                        "patches": checked,

                        "partial_sse_lower_bound":
                            partial_sse,

                        "predicted_rms":
                            rms(corrected_prediction),

                        "files": files,
                    }

                    self._record(output)
                    return output

        # All 3 Abaqus patches completed:
        # only now is this a full residual.
        result = {
            "serial": serial,
            "q": q,
            "verified": True,
            "status": "fully_verified",

            "r": r,
            "rms": rms(r),
            "max": float(
                np.max(np.abs(r))
            ),

            "patches": checked,
            "files": files,

            "partial_sse_lower_bound":
                partial_sse,

            "predicted_rms":
                rms(corrected_prediction),
        }

        self._record(result)
        return result

    # ========================================================
    # AUDIT LOG
    # ========================================================

    def _record(self, result):
        item = {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in result.items()
            if key != "files"
        }

        item["jobs_total"] = self.jobs

        self.log.append(item)

        write_json(
            os.path.join(
                self.workdir,
                "progress.json",
            ),
            self.log,
        )
