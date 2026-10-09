from __future__ import print_function

import os
import csv
import json
import shutil
import argparse

import numpy as np

from hybrid_runtime_7region import (
    BASE_DIR,
    load_interface,
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
    default=20,
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
    default=0.60,
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
# OUTPUT
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results_7region_direct",
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
# SIX INTERFACES
# ============================================================

INTERFACE_ORDER = [

    "left_outer",

    "left_inner",

    "center_left",

    "center_right",

    "right_inner",

    "right_outer",
]


MODE_COUNTS = {}

COEFFICIENT_SCALES = {}


for interface_name in INTERFACE_ORDER:

    interface = load_interface(
        interface_name
    )


    basis = interface[
        "basis"
    ]


    MODE_COUNTS[
        interface_name
    ] = basis.shape[
        1
    ]


    scores = interface[
        "scores_train"
    ]


    scale = scores.std(
        axis=0,
        ddof=1,
    )


    COEFFICIENT_SCALES[
        interface_name
    ] = np.maximum(
        scale,
        1.0e-8,
    )


c_scale_vector = np.concatenate(
    [

        COEFFICIENT_SCALES[
            name
        ]

        for name
        in INTERFACE_ORDER
    ]
)


number_unknowns = len(
    c_scale_vector
)


print("")
print(
    "Number of Broyden variables:",
    number_unknowns
)


if number_unknowns != 26:

    raise RuntimeError(
        "Expected 26 unknowns, found {}."
        .format(
            number_unknowns
        )
    )


# ============================================================
# FORCE NORMALIZATION
# ============================================================

force_scales = get_residual_force_scales()


force_scale_vector = np.concatenate(
    [

        force_scales[
            name
        ]

        for name
        in INTERFACE_ORDER
    ]
)


if force_scale_vector.shape[
    0
] != 26:

    raise RuntimeError(
        "Expected 26 residual components."
    )


# ============================================================
# VECTOR -> INTERFACE COEFFICIENTS
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


    cursor = 0


    for interface_name in (
        INTERFACE_ORDER
    ):

        count = MODE_COUNTS[
            interface_name
        ]


        result[
            interface_name
        ] = physical[
            cursor:
            cursor
            +
            count
        ].copy()


        cursor += (
            count
        )


    return result


def coefficient_dictionary_to_vector(
    coefficient_dictionary
):

    return np.concatenate(
        [

            coefficient_dictionary[
                interface_name
            ]

            for interface_name
            in INTERFACE_ORDER
        ]
    )


# ============================================================
# RESIDUAL
# ============================================================

evaluation_counter = 0


def evaluate_residual(
    q
):

    global evaluation_counter


    evaluation_id = (
        evaluation_counter
    )


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


    # ========================================================
    # THREE FE CONTACT PATCHES
    # ========================================================

    left_patch = run_patch_job(

        patch_name=
            "left",

        E1=
            E1,

        E2=
            E2,

        G12=
            G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "hyb7_{:03d}_left".format(
                evaluation_id
            ),

        cpus=
            args.cpus,
    )


    center_patch = run_patch_job(

        patch_name=
            "center",

        E1=
            E1,

        E2=
            E2,

        G12=
            G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "hyb7_{:03d}_center".format(
                evaluation_id
            ),

        cpus=
            args.cpus,
    )


    right_patch = run_patch_job(

        patch_name=
            "right",

        E1=
            E1,

        E2=
            E2,

        G12=
            G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "hyb7_{:03d}_right".format(
                evaluation_id
            ),

        cpus=
            args.cpus,
    )


    # ========================================================
    # FOUR DIRECT V4 NO FORCE PREDICTIONS
    # ========================================================

    no_force = evaluate_neural_operators(

        E1,

        E2,

        G12,

        coefficients,
    )


    # ========================================================
    # SIX ACTION-REACTION RESIDUALS
    # ========================================================

    residual_dictionary = {

        "left_outer":

            left_patch[
                "generalized_forces"
            ][
                "left_outer"
            ]
            +
            no_force[
                "left_outer"
            ],

        "left_inner":

            left_patch[
                "generalized_forces"
            ][
                "left_inner"
            ]
            +
            no_force[
                "left_inner"
            ],

        "center_left":

            center_patch[
                "generalized_forces"
            ][
                "center_left"
            ]
            +
            no_force[
                "center_left"
            ],

        "center_right":

            center_patch[
                "generalized_forces"
            ][
                "center_right"
            ]
            +
            no_force[
                "center_right"
            ],

        "right_inner":

            right_patch[
                "generalized_forces"
            ][
                "right_inner"
            ]
            +
            no_force[
                "right_inner"
            ],

        "right_outer":

            right_patch[
                "generalized_forces"
            ][
                "right_outer"
            ]
            +
            no_force[
                "right_outer"
            ],
    }


    physical_residual = np.concatenate(
        [

            residual_dictionary[
                name
            ]

            for name
            in INTERFACE_ORDER
        ]
    )


    normalized_residual = (
        physical_residual
        /
        force_scale_vector
    )


    rms = float(
        np.sqrt(
            np.mean(
                normalized_residual
                ** 2
            )
        )
    )


    maximum = float(
        np.max(
            np.abs(
                normalized_residual
            )
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    np.savez(

        os.path.join(
            evaluation_directory,
            "evaluation_state.npz",
        ),

        q=
            np.asarray(
                q,
                dtype=np.float64,
            ),

        physical_coefficients=
            coefficient_dictionary_to_vector(
                coefficients
            ),

        physical_residual=
            physical_residual,

        normalized_residual=
            normalized_residual,

        rms=
            np.asarray(
                [
                    rms
                ]
            ),

        maximum=
            np.asarray(
                [
                    maximum
                ]
            ),
    )


    print("")
    print(
        "------------------------------------------"
    )

    print(
        "Residual evaluation:",
        evaluation_id
    )

    print(
        "RMS normalized residual:",
        rms
    )

    print(
        "Maximum normalized residual:",
        maximum
    )

    print(
        "------------------------------------------"
    )


    details = {

        "evaluation_id":
            evaluation_id,

        "evaluation_directory":
            evaluation_directory,

        "left":
            left_patch,

        "center":
            center_patch,

        "right":
            right_patch,

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
# INITIAL CONDITION
#
# q = c / training-score std
#
# q = 0 corresponds to the train-PCA mean displacement
# at every interface.
# ============================================================

q = np.zeros(
    number_unknowns,
    dtype=np.float64,
)


residual, details = evaluate_residual(
    q
)


current_norm = float(
    np.sqrt(
        np.mean(
            residual
            ** 2
        )
    )
)


# ============================================================
# INITIAL INVERSE JACOBIAN
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


converged = (
    current_norm
    <=
    args.tol
)


iteration = 0


# ============================================================
# BROYDEN
# ============================================================

while (
    not converged

    and

    iteration
    <
    args.max_iter
):

    iteration += 1


    # ========================================================
    # QUASI-NEWTON STEP
    # ========================================================

    step = (
        -H
        @
        residual
    )


    maximum_step = float(
        np.max(
            np.abs(
                step
            )
        )
    )


    if maximum_step > (
        args.max_step
    ):

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


    trial_norm = float(
        np.sqrt(
            np.mean(
                residual_trial
                ** 2
            )
        )
    )


    # ========================================================
    # BAD STEP
    # ========================================================

    if trial_norm > (
        1.50
        *
        current_norm
    ):

        print(
            "Residual increased strongly."
        )

        print(
            "Resetting inverse Broyden matrix."
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

        # ====================================================
        # INVERSE BROYDEN UPDATE
        #
        # H_{k+1}
        # =
        # H_k
        # +
        # ((s - H y) y^T)/(y^T y)
        # ====================================================

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


        denominator = float(
            y
            @
            y
        )


        if denominator > 1.0e-12:

            H_y = (
                H
                @
                y
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


    # ========================================================
    # ACCEPT
    # ========================================================

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
        "=========================================="
    )

    print(
        "BROYDEN ITERATION:",
        iteration
    )

    print(
        "RMS residual:",
        current_norm
    )

    print(
        "Best RMS residual:",
        best_norm
    )

    print(
        "Tolerance:",
        args.tol
    )

    print(
        "=========================================="
    )


    converged = (
        current_norm
        <=
        args.tol
    )


# ============================================================
# BEST STATE
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
) as file_object:

    writer = csv.DictWriter(

        file_object,

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

solution_payload = {

    "E1":
        np.asarray(
            [
                E1
            ]
        ),

    "E2":
        np.asarray(
            [
                E2
            ]
        ),

    "G12":
        np.asarray(
            [
                G12
            ]
        ),

    "q":
        final_q,

    "c_scale":
        c_scale_vector,

    "normalized_residual":
        final_details[
            "normalized_residual"
        ],

    "physical_residual":
        final_details[
            "physical_residual"
        ],

    "rms_residual":
        np.asarray(
            [
                final_details[
                    "rms"
                ]
            ]
        ),
}


for interface_name in INTERFACE_ORDER:

    solution_payload[
        "c_{}".format(
            interface_name
        )
    ] = final_coefficients[
        interface_name
    ]


solution_file = os.path.join(
    RESULTS_DIR,
    "online_solution_7region.npz",
)


np.savez(
    solution_file,
    **solution_payload
)


# ============================================================
# COPY BEST PATCH FILES
# ============================================================

for patch_name in [

    "left",
    "center",
    "right",

]:

    patch_result = final_details[
        patch_name
    ]


    shutil.copyfile(

        patch_result[
            "odb_file"
        ],

        os.path.join(
            FINAL_DIR,
            "{}.odb".format(
                patch_name
            ),
        ),
    )


    shutil.copyfile(

        patch_result[
            "input_file"
        ],

        os.path.join(
            FINAL_DIR,
            "{}.inp".format(
                patch_name
            ),
        ),
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
            <=
            args.tol
        ),

    "number_unknowns":
        int(
            number_unknowns
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
) as file_object:

    json.dump(
        manifest,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64T COMPLETE"
)

print(
    "=========================================="
)

print(
    "Converged:",
    manifest[
        "converged"
    ]
)

print(
    "Unknowns:",
    number_unknowns
)

print(
    "Best RMS residual:",
    manifest[
        "best_rms_residual"
    ]
)

print(
    "Best max residual:",
    manifest[
        "best_max_residual"
    ]
)

print(
    "Residual evaluations:",
    evaluation_counter
)

print(
    "Solution:",
    solution_file
)