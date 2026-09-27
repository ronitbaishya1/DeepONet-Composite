from __future__ import print_function

import os
import csv

import numpy as np

from hybrid_runtime import (
    BASE_DIR,
    load_interface,
    run_patch_job,
    evaluate_neural_operators,
    get_residual_force_scales,
)


# ============================================================
# BASELINE MATERIAL
# ============================================================

E1 = 45000.0

E2 = 12000.0

G12 = 4500.0


# ============================================================
# INITIAL COEFFICIENTS
#
# Zero PCA coefficients = training-set mean interface field.
# ============================================================

coefficients = {}


for name in [
    "m6",
    "m2",
    "p2",
    "p6",
]:

    basis = load_interface(
        name
    )[
        "basis"
    ]


    coefficients[
        name
    ] = np.zeros(
        basis.shape[
            1
        ],
        dtype=np.float64,
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

job_directory = os.path.join(
    BASE_DIR,
    "online_smoke",
)


if not os.path.isdir(
    job_directory
):

    os.makedirs(
        job_directory
    )


# ============================================================
# FE PATCHES
# ============================================================

left = run_patch_job(
    patch_name="left",
    E1=E1,
    E2=E2,
    G12=G12,
    coefficient_dictionary=coefficients,
    job_directory=job_directory,
    job_name="smoke_left",
    cpus=4,
)


center = run_patch_job(
    patch_name="center",
    E1=E1,
    E2=E2,
    G12=G12,
    coefficient_dictionary=coefficients,
    job_directory=job_directory,
    job_name="smoke_center",
    cpus=4,
)


right = run_patch_job(
    patch_name="right",
    E1=E1,
    E2=E2,
    G12=G12,
    coefficient_dictionary=coefficients,
    job_directory=job_directory,
    job_name="smoke_right",
    cpus=4,
)


# ============================================================
# NEURAL OPERATOR FORCES
# ============================================================

no_force = evaluate_neural_operators(
    E1,
    E2,
    G12,
    coefficients,
)


# ============================================================
# INTERFACE RESIDUALS
#
# FE reaction + NO reaction = 0 at equilibrium
# ============================================================

residuals = {

    "m6":
        left[
            "generalized_forces"
        ][
            "m6"
        ]
        +
        no_force[
            "m6"
        ],

    "m2":
        center[
            "generalized_forces"
        ][
            "m2"
        ]
        +
        no_force[
            "m2"
        ],

    "p2":
        center[
            "generalized_forces"
        ][
            "p2"
        ]
        +
        no_force[
            "p2"
        ],

    "p6":
        right[
            "generalized_forces"
        ][
            "p6"
        ]
        +
        no_force[
            "p6"
        ],
}


scales = get_residual_force_scales()


normalized = []


print("")
print(
    "============================================"
)

print(
    "SMOKE TEST RESULTS"
)

print(
    "============================================"
)


rows = []


for interface_name in [
    "m6",
    "m2",
    "p2",
    "p6",
]:

    r = residuals[
        interface_name
    ]


    rn = (
        r
        /
        scales[
            interface_name
        ]
    )


    normalized.extend(
        rn.tolist()
    )


    print("")
    print(
        interface_name
    )

    print(
        "Raw residual:",
        r
    )

    print(
        "Normalized residual:",
        rn
    )


    for index in range(
        len(
            r
        )
    ):

        rows.append(
            {
                "Interface":
                    interface_name,

                "Mode":
                    index + 1,

                "Residual_N":
                    r[
                        index
                    ],

                "NormalizedResidual":
                    rn[
                        index
                    ],
            }
        )


normalized = np.asarray(
    normalized,
    dtype=np.float64,
)


print("")
print(
    "Normalized RMS residual:",
    np.sqrt(
        np.mean(
            normalized
            ** 2
        )
    ),
)

print(
    "Normalized max residual:",
    np.max(
        np.abs(
            normalized
        )
    ),
)


csv_file = os.path.join(
    job_directory,
    "smoke_residual.csv",
)


with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Interface",
            "Mode",
            "Residual_N",
            "NormalizedResidual",
        ],
    )


    writer.writeheader()

    writer.writerows(
        rows
    )


print("")
print(
    "Smoke test complete."
)

print(
    csv_file
)