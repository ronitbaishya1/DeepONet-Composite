from __future__ import print_function

import os
import csv
import json
import shutil
import argparse

import numpy as np

from hybrid_runtime import (
    BASE_DIR,
    load_interface,
    load_force_model,
    run_patch_job,
    evaluate_neural_operators,
    get_residual_force_scales,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--E1",
    type=float,
    default=45000.0,
)


parser.add_argument(
    "--E2",
    type=float,
    default=12000.0,
)


parser.add_argument(
    "--G12",
    type=float,
    default=4500.0,
)


parser.add_argument(
    "--cpus",
    type=int,
    default=4,
)


parser.add_argument(
    "--max_iter",
    type=int,
    default=15,
)


parser.add_argument(
    "--tol",
    type=float,
    default=0.02,
)


parser.add_argument(
    "--initial_inverse",
    type=float,
    default=0.35,
)


parser.add_argument(
    "--damping",
    type=float,
    default=0.6,
)


parser.add_argument(
    "--max_step",
    type=float,
    default=0.75,
)


parser.add_argument(
    "--q_limit",
    type=float,
    default=3.0,
)


args = parser.parse_args()


E1 = args.E1

E2 = args.E2

G12 = args.G12


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results",
)


EVALUATION_ROOT = os.path.join(
    RESULTS_DIR,
    "evaluations",
)


FINAL_DIR = os.path.join(
    RESULTS_DIR,
    "final",
)


for directory in [
    RESULTS_DIR,
    EVALUATION_ROOT,
    FINAL_DIR,
]:

    if not os.path.isdir(
        directory
    ):

        os.makedirs(
            directory
        )


# ============================================================
# INTERFACE ORDER
# ============================================================

INTERFACE_ORDER = [
    "m6",
    "m2",
    "p2",
    "p6",
]


MODE_COUNTS = {}


COEFFICIENT_SCALES = {}


for interface_name in INTERFACE_ORDER:

    data = load_interface(
        interface_name
    )


    basis = data[
        "basis"
    ]


    MODE_COUNTS[
        interface_name
    ] = basis.shape[
        1
    ]


    if "scores_train" in data.files:

        scores = data[
            "scores_train"
        ]

    else:

        raise RuntimeError(
            "scores_train missing from interface {}".format(
                interface_name
            )
        )


    scale = scores.std(
        axis=0,
        ddof=1,
    )


    scale = np.maximum(
        scale,
        1.0e-8,
    )


    COEFFICIENT_SCALES[
        interface_name
    ] = scale


# ============================================================
# BUILD CONCATENATED COEFFICIENT SCALE
# ============================================================

c_scale_vector = np.concatenate(
    [

        COEFFICIENT_SCALES[
            name
        ]

        for name in INTERFACE_ORDER
    ]
)


number_unknowns = len(
    c_scale_vector
)


if number_unknowns != 18:

    print(
        "Warning: expected 18 interface modes, found",
        number_unknowns,
    )


# ============================================================
# RESIDUAL NORMALIZATION SCALE
# ============================================================

force_scales = get_residual_force_scales()


force_scale_vector = np.concatenate(
    [

        force_scales[
            name
        ]

        for name in INTERFACE_ORDER
    ]
)


# ============================================================
# HELPERS
# ============================================================

def vector_to_coefficients(
    q
):

    physical = (
        q
        *
        c_scale_vector
    )


    result = {}


    start = 0


    for name in INTERFACE_ORDER:

        count = MODE_COUNTS[
            name
        ]


        result[
            name
        ] = physical[
            start:
            start
            +
            count
        ].copy()


        start += count


    return result


def coefficient_dictionary_to_vector(
    coefficient_dictionary
):

    return np.concatenate(
        [

            coefficient_dictionary[
                name
            ]

            for name in INTERFACE_ORDER
        ]
    )


# ============================================================
# RESIDUAL EVALUATION COUNTER
# ============================================================

evaluation_counter = 0


def evaluate_residual(
    q
):

    global evaluation_counter


    evaluation_id = evaluation_counter


    evaluation_counter += 1


    coefficients = vector_to_coefficients(
        q
    )


    evaluation_directory = os.path.join(
        EVALUATION_ROOT,
        "eval_{:03d}".format(
            evaluation_id
        ),
    )


    if not os.path.isdir(
        evaluation_directory
    ):

        os.makedirs(
            evaluation_directory
        )


    # --------------------------------------------------------
    # FE PATCHES
    # --------------------------------------------------------

    left = run_patch_job(
        patch_name="left",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=coefficients,
        job_directory=evaluation_directory,
        job_name="hyb_{:03d}_left".format(
            evaluation_id
        ),
        cpus=args.cpus,
    )


    center = run_patch_job(
        patch_name="center",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=coefficients,
        job_directory=evaluation_directory,
        job_name="hyb_{:03d}_center".format(
            evaluation_id
        ),
        cpus=args.cpus,
    )


    right = run_patch_job(
        patch_name="right",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=coefficients,
        job_directory=evaluation_directory,
        job_name="hyb_{:03d}_right".format(
            evaluation_id
        ),
        cpus=args.cpus,
    )


    # --------------------------------------------------------
    # NEURAL OPERATOR FORCES
    # --------------------------------------------------------

    no_force = evaluate_neural_operators(
        E1,
        E2,
        G12,
        coefficients,
    )


    # --------------------------------------------------------
    # PHYSICAL RESIDUAL
    # --------------------------------------------------------

    residual_dictionary = {

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


    physical_residual = np.concatenate(
        [

            residual_dictionary[
                name
            ]

            for name in INTERFACE_ORDER
        ]
    )


    normalized_residual = (
        physical_residual
        /
        force_scale_vector
    )


    rms = np.sqrt(
        np.mean(
            normalized_residual
            ** 2
        )
    )


    maximum = np.max(
        np.abs(
            normalized_residual
        )
    )


    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            evaluation_directory,
            "evaluation_state.npz",
        ),

        q=np.asarray(
            q,
            dtype=np.float64,
        ),

        physical_coefficients=(
            coefficient_dictionary_to_vector(
                coefficients
            )
        ),

        physical_residual=physical_residual,

        normalized_residual=normalized_residual,

        rms=np.array(
            [
                rms
            ]
        ),

        maximum=np.array(
            [
                maximum
            ]
        ),
    )


    print("")
    print(
        "--------------------------------------------"
    )

    print(
        "Residual evaluation:",
        evaluation_id,
    )

    print(
        "RMS normalized residual:",
        rms,
    )

    print(
        "Max normalized residual:",
        maximum,
    )

    print(
        "--------------------------------------------"
    )


    details = {

        "evaluation_id":
            evaluation_id,

        "evaluation_directory":
            evaluation_directory,

        "left":
            left,

        "center":
            center,

        "right":
            right,

        "coefficients":
            coefficients,

        "physical_residual":
            physical_residual,

        "normalized_residual":
            normalized_residual,

        "rms":
            rms,

        "maximum":
            maximum,
    }


    return (
        normalized_residual,
        details,
    )


# ============================================================
# INITIAL STATE
#
# q = c / training_score_std
#
# q=0 corresponds to PCA mean interface fields.
# ============================================================

q = np.zeros(
    number_unknowns,
    dtype=np.float64,
)


residual, details = evaluate_residual(
    q
)


current_norm = np.sqrt(
    np.mean(
        residual
        ** 2
    )
)


# ============================================================
# INVERSE JACOBIAN INITIALIZATION
# ============================================================

H = (
    args.initial_inverse
    *
    np.eye(
        number_unknowns,
        dtype=np.float64,
    )
)


damping = args.damping


history = []


best_norm = current_norm

best_q = q.copy()

best_details = details


history.append(
    {

        "Iteration":
            0,

        "ResidualEvaluation":
            details[
                "evaluation_id"
            ],

        "RMSResidual":
            current_norm,

        "MaxResidual":
            details[
                "maximum"
            ],

        "Damping":
            damping,
    }
)


# ============================================================
# BROYDEN ITERATIONS
# ============================================================

converged = (
    current_norm
    <= args.tol
)


iteration = 0


while (
    not converged

    and

    iteration
    <
    args.max_iter
):

    iteration += 1


    # --------------------------------------------------------
    # PROPOSE STEP
    #
    # q_{k+1} = q_k - H r
    # --------------------------------------------------------

    step = (
        -H
        @ residual
    )


    maximum_step = np.max(
        np.abs(
            step
        )
    )


    if maximum_step > args.max_step:

        step *= (
            args.max_step
            /
            maximum_step
        )


    q_trial = (
        q
        +
        damping
        *
        step
    )


    q_trial = np.clip(
        q_trial,
        -args.q_limit,
        args.q_limit,
    )


    residual_trial, details_trial = (
        evaluate_residual(
            q_trial
        )
    )


    trial_norm = np.sqrt(
        np.mean(
            residual_trial
            ** 2
        )
    )


    # --------------------------------------------------------
    # IF STEP IS BADLY DIVERGENT:
    # RESET APPROXIMATE INVERSE AND REDUCE DAMPING.
    # --------------------------------------------------------

    if trial_norm > (
        1.50
        *
        current_norm
    ):

        print(
            "Residual increased strongly. "
            "Resetting Broyden matrix."
        )


        H = (
            args.initial_inverse
            *
            np.eye(
                number_unknowns,
                dtype=np.float64,
            )
        )


        damping = max(
            0.20,
            0.5
            *
            damping,
        )


    else:

        # ----------------------------------------------------
        # INVERSE BROYDEN SECANT UPDATE
        #
        # H_new =
        # H + ((s - H y) y^T)/(y^T y)
        #
        # ----------------------------------------------------

        s = (
            q_trial
            -
            q
        )


        y = (
            residual_trial
            -
            residual
        )


        denominator = (
            y
            @ y
        )


        if denominator > 1.0e-12:

            H_y = (
                H
                @ y
            )


            H += (
                np.outer(
                    s
                    -
                    H_y,
                    y,
                )
                /
                denominator
            )


        damping = min(
            1.0,
            1.05
            *
            damping,
        )


    # --------------------------------------------------------
    # ACCEPT TRIAL
    # --------------------------------------------------------

    q = q_trial

    residual = residual_trial

    details = details_trial

    current_norm = trial_norm


    if current_norm < best_norm:

        best_norm = current_norm

        best_q = q.copy()

        best_details = details


    history.append(
        {

            "Iteration":
                iteration,

            "ResidualEvaluation":
                details[
                    "evaluation_id"
                ],

            "RMSResidual":
                current_norm,

            "MaxResidual":
                details[
                    "maximum"
                ],

            "Damping":
                damping,
        }
    )


    print("")
    print(
        "============================================"
    )

    print(
        "BROYDEN ITERATION:",
        iteration,
    )

    print(
        "RMS residual:",
        current_norm,
    )

    print(
        "Best RMS residual:",
        best_norm,
    )

    print(
        "Tolerance:",
        args.tol,
    )

    print(
        "============================================"
    )


    converged = (
        current_norm
        <= args.tol
    )


# ============================================================
# USE BEST STATE FOUND
# ============================================================

final_q = best_q

final_details = best_details

final_coefficients = vector_to_coefficients(
    final_q
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_file = os.path.join(
    RESULTS_DIR,
    "broyden_history.csv",
)


with open(
    history_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Iteration",
            "ResidualEvaluation",
            "RMSResidual",
            "MaxResidual",
            "Damping",
        ],
    )


    writer.writeheader()

    writer.writerows(
        history
    )


# ============================================================
# SAVE SOLUTION
# ============================================================

solution_file = os.path.join(
    RESULTS_DIR,
    "online_solution.npz",
)


np.savez(
    solution_file,

    E1=np.array(
        [
            E1
        ]
    ),

    E2=np.array(
        [
            E2
        ]
    ),

    G12=np.array(
        [
            G12
        ]
    ),

    q=final_q,

    c_scale=c_scale_vector,

    c_m6=final_coefficients[
        "m6"
    ],

    c_m2=final_coefficients[
        "m2"
    ],

    c_p2=final_coefficients[
        "p2"
    ],

    c_p6=final_coefficients[
        "p6"
    ],

    normalized_residual=final_details[
        "normalized_residual"
    ],

    physical_residual=final_details[
        "physical_residual"
    ],

    rms_residual=np.array(
        [
            final_details[
                "rms"
            ]
        ]
    ),
)


# ============================================================
# COPY BEST FE PATCH FILES TO FINAL DIRECTORY
# ============================================================

for patch_name in [
    "left",
    "center",
    "right",
]:

    patch_result = final_details[
        patch_name
    ]


    source_odb = patch_result[
        "odb_file"
    ]


    source_inp = patch_result[
        "input_file"
    ]


    destination_odb = os.path.join(
        FINAL_DIR,
        "{}.odb".format(
            patch_name
        ),
    )


    destination_inp = os.path.join(
        FINAL_DIR,
        "{}.inp".format(
            patch_name
        ),
    )


    shutil.copyfile(
        source_odb,
        destination_odb,
    )


    shutil.copyfile(
        source_inp,
        destination_inp,
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "converged":
        bool(
            final_details[
                "rms"
            ]
            <= args.tol
        ),

    "E1":
        E1,

    "E2":
        E2,

    "G12":
        G12,

    "best_rms_residual":
        float(
            final_details[
                "rms"
            ]
        ),

    "best_max_residual":
        float(
            final_details[
                "maximum"
            ]
        ),

    "best_evaluation":
        int(
            final_details[
                "evaluation_id"
            ]
        ),

    "number_residual_evaluations":
        int(
            evaluation_counter
        ),

    "solution_file":
        solution_file,

    "final_directory":
        FINAL_DIR,
}


manifest_file = os.path.join(
    RESULTS_DIR,
    "solution_manifest.json",
)


with open(
    manifest_file,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=4,
    )


print("")
print(
    "============================================"
)

print(
    "ONLINE HYBRID SOLUTION COMPLETE"
)

print(
    "============================================"
)

print(
    "Converged:",
    manifest[
        "converged"
    ],
)

print(
    "Best RMS residual:",
    manifest[
        "best_rms_residual"
    ],
)

print(
    "Best max residual:",
    manifest[
        "best_max_residual"
    ],
)

print(
    "Residual evaluations:",
    evaluation_counter,
)

print(
    "Solution:",
    solution_file,
)

print(
    "Final patch ODBs:",
    FINAL_DIR,
)