from __future__ import print_function

import os
import csv
import json
import shutil
import argparse

import numpy as np

from hybrid_runtime_ensemble import (
    BASE_DIR,
    load_interface,
    run_patch_job,
    evaluate_neural_operators,
    get_residual_force_scales,
)


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


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results_ensemble",
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

    os.makedirs(
        directory,
        exist_ok=True,
    )


INTERFACE_ORDER = [
    "m6",
    "m2",
    "p2",
    "p6",
]


MODE_COUNTS = {}

COEFFICIENT_SCALES = {}


for interface_name in INTERFACE_ORDER:

    interface = load_interface(
        interface_name
    )

    MODE_COUNTS[
        interface_name
    ] = interface[
        "basis"
    ].shape[1]

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
        COEFFICIENT_SCALES[name]
        for name in INTERFACE_ORDER
    ]
)


force_scales = (
    get_residual_force_scales()
)


force_scale_vector = np.concatenate(
    [
        force_scales[name]
        for name in INTERFACE_ORDER
    ]
)


number_unknowns = len(
    c_scale_vector
)


def q_to_coefficients(q):

    physical = (
        q
        *
        c_scale_vector
    )

    output = {}

    start = 0

    for name in INTERFACE_ORDER:

        count = MODE_COUNTS[name]

        output[name] = physical[
            start:
            start + count
        ].copy()

        start += count

    return output


def coefficients_to_vector(
    coefficients
):

    return np.concatenate(
        [
            coefficients[name]
            for name in INTERFACE_ORDER
        ]
    )


evaluation_counter = 0


def evaluate_residual(q):

    global evaluation_counter

    evaluation_id = (
        evaluation_counter
    )

    evaluation_counter += 1

    coefficients = (
        q_to_coefficients(q)
    )

    evaluation_directory = os.path.join(
        EVALUATION_ROOT,
        "eval_{:03d}".format(
            evaluation_id
        ),
    )

    os.makedirs(
        evaluation_directory,
        exist_ok=True,
    )

    left = run_patch_job(
        patch_name="left",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=
            coefficients,
        job_directory=
            evaluation_directory,
        job_name=
            "ens_{:03d}_left".format(
                evaluation_id
            ),
        cpus=args.cpus,
    )

    center = run_patch_job(
        patch_name="center",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=
            coefficients,
        job_directory=
            evaluation_directory,
        job_name=
            "ens_{:03d}_center".format(
                evaluation_id
            ),
        cpus=args.cpus,
    )

    right = run_patch_job(
        patch_name="right",
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=
            coefficients,
        job_directory=
            evaluation_directory,
        job_name=
            "ens_{:03d}_right".format(
                evaluation_id
            ),
        cpus=args.cpus,
    )

    no_force = (
        evaluate_neural_operators(
            E1,
            E2,
            G12,
            coefficients,
        )
    )

    residual_dictionary = {

        "m6":
            left[
                "generalized_forces"
            ]["m6"]
            +
            no_force["m6"],

        "m2":
            center[
                "generalized_forces"
            ]["m2"]
            +
            no_force["m2"],

        "p2":
            center[
                "generalized_forces"
            ]["p2"]
            +
            no_force["p2"],

        "p6":
            right[
                "generalized_forces"
            ]["p6"]
            +
            no_force["p6"],
    }

    physical_residual = np.concatenate(
        [
            residual_dictionary[name]
            for name in INTERFACE_ORDER
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
                normalized_residual ** 2
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

    print("")
    print(
        "Residual evaluation:",
        evaluation_id
    )

    print(
        "RMS:",
        rms
    )

    print(
        "Max:",
        maximum
    )

    return (
        normalized_residual,

        {
            "evaluation_id":
                evaluation_id,

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
    )


q = np.zeros(
    number_unknowns,
    dtype=np.float64,
)


residual, details = (
    evaluate_residual(q)
)


current_norm = float(
    np.sqrt(
        np.mean(
            residual ** 2
        )
    )
)


H = (
    args.initial_inverse
    *
    np.eye(
        number_unknowns
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


while (
    not converged
    and
    iteration < args.max_iter
):

    iteration += 1

    step = (
        -H
        @
        residual
    )

    maximum_step = float(
        np.max(
            np.abs(step)
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

    trial_norm = float(
        np.sqrt(
            np.mean(
                residual_trial ** 2
            )
        )
    )

    if trial_norm > (
        1.5
        *
        current_norm
    ):

        H = (
            args.initial_inverse
            *
            np.eye(
                number_unknowns
            )
        )

        damping = max(
            0.20,
            0.5
            *
            damping,
        )

    else:

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
            y @ y
        )

        if denominator > 1.0e-12:

            H_y = H @ y

            H += (
                np.outer(
                    s - H_y,
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
        "Iteration:",
        iteration
    )

    print(
        "Current RMS:",
        current_norm
    )

    print(
        "Best RMS:",
        best_norm
    )

    converged = (
        current_norm
        <=
        args.tol
    )


final_q = best_q

final_details = (
    best_details
)

final_coefficients = (
    q_to_coefficients(
        final_q
    )
)


with open(
    os.path.join(
        RESULTS_DIR,
        "broyden_history.csv",
    ),
    "w",
    newline="",
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
    writer.writerows(history)


np.savez(
    os.path.join(
        RESULTS_DIR,
        "online_solution.npz",
    ),

    E1=np.asarray([E1]),
    E2=np.asarray([E2]),
    G12=np.asarray([G12]),

    q=final_q,

    c_scale=c_scale_vector,

    c_m6=
        final_coefficients["m6"],

    c_m2=
        final_coefficients["m2"],

    c_p2=
        final_coefficients["p2"],

    c_p6=
        final_coefficients["p6"],

    normalized_residual=
        final_details[
            "normalized_residual"
        ],

    physical_residual=
        final_details[
            "physical_residual"
        ],

    rms_residual=
        np.asarray(
            [
                final_details[
                    "rms"
                ]
            ]
        ),
)


for patch_name in [
    "left",
    "center",
    "right",
]:

    patch_result = (
        final_details[
            patch_name
        ]
    )

    shutil.copyfile(
        patch_result["odb_file"],
        os.path.join(
            FINAL_DIR,
            "{}.odb".format(
                patch_name
            ),
        ),
    )

    shutil.copyfile(
        patch_result["input_file"],
        os.path.join(
            FINAL_DIR,
            "{}.inp".format(
                patch_name
            ),
        ),
    )


manifest = {

    "model":
        "Step40_ensemble",

    "converged":
        bool(
            final_details["rms"]
            <=
            args.tol
        ),

    "E1":
        E1,

    "E2":
        E2,

    "G12":
        G12,

    "best_rms_residual":
        float(
            final_details["rms"]
        ),

    "best_max_residual":
        float(
            final_details["maximum"]
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
}


with open(
    os.path.join(
        RESULTS_DIR,
        "solution_manifest.json",
    ),
    "w",
) as file_object:

    json.dump(
        manifest,
        file_object,
        indent=4,
    )


print("")
print(
    "========================================"
)

print(
    "STEP 59 COMPLETE"
)

print(
    "========================================"
)

print(
    json.dumps(
        manifest,
        indent=4,
    )
)