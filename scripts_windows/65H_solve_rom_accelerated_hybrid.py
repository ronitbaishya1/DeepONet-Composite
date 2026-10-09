from __future__ import print_function

import os
import csv
import json
import time
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


from rom_runtime_7region import (
    predict_rom_generalized_force,
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
    "--tol",
    type=float,
    default=0.02,
)


parser.add_argument(
    "--rom_max_iter",
    type=int,
    default=25,
)


parser.add_argument(
    "--hf_max_iter",
    type=int,
    default=8,
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
# PATHS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "online_results_7region_rom",
)


ROM_HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "rom_broyden_history.csv",
)


HF_HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "high_fidelity_correction_history.csv",
)


FINAL_DIR = os.path.join(
    RESULTS_DIR,
    "final",
)


HF_JOB_ROOT = os.path.join(
    RESULTS_DIR,
    "high_fidelity_evaluations",
)


for directory in [

    RESULTS_DIR,
    FINAL_DIR,
    HF_JOB_ROOT,

]:

    if not os.path.isdir(
        directory
    ):

        os.makedirs(
            directory
        )


# ============================================================
# INTERFACES
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


    MODE_COUNTS[
        interface_name
    ] = interface[
        "basis"
    ].shape[1]


    scores_train = interface[
        "scores_train"
    ]


    COEFFICIENT_SCALES[
        interface_name
    ] = np.maximum(

        scores_train.std(
            axis=0,
            ddof=1,
        ),

        1.0e-8,
    )


c_scale = np.concatenate(
    [
        COEFFICIENT_SCALES[
            name
        ]

        for name
        in INTERFACE_ORDER
    ]
)


if c_scale.shape[
    0
] != 26:

    raise RuntimeError(
        "Expected 26 Broyden variables."
    )


# ============================================================
# FORCE SCALES
# ============================================================

force_scales_dictionary = (
    get_residual_force_scales()
)


force_scale = np.concatenate(
    [
        force_scales_dictionary[
            name
        ]

        for name
        in INTERFACE_ORDER
    ]
)


# ============================================================
# VECTOR CONVERSION
# ============================================================

def q_to_coefficients(
    q,
):

    physical = (
        q
        *
        c_scale
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


        cursor += count


    return result


# ============================================================
# BUILD ROM PATCH BRANCHES
# ============================================================

def build_patch_branches(
    coefficient_dictionary,
):

    material = np.asarray(
        [
            E1,
            E2,
            G12,
        ],
        dtype=np.float64,
    )


    left = np.concatenate(
        [
            material,

            coefficient_dictionary[
                "left_outer"
            ],

            coefficient_dictionary[
                "left_inner"
            ],
        ]
    )


    center = np.concatenate(
        [
            material,

            coefficient_dictionary[
                "center_left"
            ],

            coefficient_dictionary[
                "center_right"
            ],
        ]
    )


    right = np.concatenate(
        [
            material,

            coefficient_dictionary[
                "right_inner"
            ],

            coefficient_dictionary[
                "right_outer"
            ],
        ]
    )


    return (
        left,
        center,
        right,
    )


# ============================================================
# ASSEMBLE RESIDUAL VECTOR
# ============================================================

def assemble_residual(
    patch_g,
    no_g,
):

    residual_dictionary = {

        "left_outer":

            patch_g[
                "left_outer"
            ]
            +
            no_g[
                "left_outer"
            ],

        "left_inner":

            patch_g[
                "left_inner"
            ]
            +
            no_g[
                "left_inner"
            ],

        "center_left":

            patch_g[
                "center_left"
            ]
            +
            no_g[
                "center_left"
            ],

        "center_right":

            patch_g[
                "center_right"
            ]
            +
            no_g[
                "center_right"
            ],

        "right_inner":

            patch_g[
                "right_inner"
            ]
            +
            no_g[
                "right_inner"
            ],

        "right_outer":

            patch_g[
                "right_outer"
            ]
            +
            no_g[
                "right_outer"
            ],
    }


    physical = np.concatenate(
        [
            residual_dictionary[
                name
            ]

            for name
            in INTERFACE_ORDER
        ]
    )


    normalized = (
        physical
        /
        force_scale
    )


    rms = float(
        np.sqrt(
            np.mean(
                normalized ** 2
            )
        )
    )


    maximum = float(
        np.max(
            np.abs(
                normalized
            )
        )
    )


    return (
        physical,
        normalized,
        rms,
        maximum,
    )


# ============================================================
# ROM RESIDUAL
# ============================================================

rom_evaluation_counter = 0


def evaluate_rom_residual(
    q,
):

    global rom_evaluation_counter


    rom_evaluation_counter += 1


    coefficients = q_to_coefficients(
        q
    )


    (
        left_branch,
        center_branch,
        right_branch,
    ) = build_patch_branches(
        coefficients
    )


    left_g = predict_rom_generalized_force(
        "left",
        left_branch,
    )


    center_g = predict_rom_generalized_force(
        "center",
        center_branch,
    )


    right_g = predict_rom_generalized_force(
        "right",
        right_branch,
    )


    patch_g = {

        "left_outer":
            left_g[
                :4
            ],

        "left_inner":
            left_g[
                4:
                8
            ],

        "center_left":
            center_g[
                :5
            ],

        "center_right":
            center_g[
                5:
                10
            ],

        "right_inner":
            right_g[
                :4
            ],

        "right_outer":
            right_g[
                4:
                8
            ],
    }


    no_g = evaluate_neural_operators(

        E1,
        E2,
        G12,
        coefficients,
    )


    return assemble_residual(
        patch_g,
        no_g,
    )


# ============================================================
# GENERIC BROYDEN ROM PHASE
# ============================================================

q = np.zeros(
    26,
    dtype=np.float64,
)


H = (
    args.initial_inverse

    *

    np.eye(
        26,
        dtype=np.float64,
    )
)


damping = args.damping


rom_history = []


rom_start_time = time.time()


(
    physical_residual,
    residual,
    current_norm,
    current_max,
) = evaluate_rom_residual(
    q
)


best_q_rom = q.copy()

best_norm_rom = current_norm


rom_history.append(
    {
        "Iteration":
            0,

        "RMSResidual":
            current_norm,

        "MaxResidual":
            current_max,

        "Damping":
            damping,
    }
)


print("")
print(
    "ROM initial RMS:",
    current_norm
)


for iteration in range(
    1,
    args.rom_max_iter + 1,
):

    if current_norm <= (
        args.tol
    ):

        break


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


    q_trial = np.clip(

        q

        +

        damping
        *
        step,

        -args.q_limit,

        args.q_limit,
    )


    (
        physical_trial,
        residual_trial,
        trial_norm,
        trial_max,
    ) = evaluate_rom_residual(
        q_trial
    )


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


    if trial_norm > (
        1.50
        *
        current_norm
    ):

        H = (
            args.initial_inverse

            *

            np.eye(
                26,
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

        if denominator > (
            1.0e-12
        ):

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


    q = q_trial

    residual = residual_trial

    physical_residual = physical_trial

    current_norm = trial_norm

    current_max = trial_max


    if current_norm < best_norm_rom:

        best_norm_rom = current_norm

        best_q_rom = q.copy()


    rom_history.append(
        {
            "Iteration":
                iteration,

            "RMSResidual":
                current_norm,

            "MaxResidual":
                current_max,

            "Damping":
                damping,
        }
    )


    print(
        "ROM iteration {} | RMS = {:.8f}"
        .format(
            iteration,
            current_norm,
        )
    )


rom_elapsed = (
    time.time()
    -
    rom_start_time
)


# ============================================================
# HIGH-FIDELITY RESIDUAL
# ============================================================

hf_evaluation_counter = 0


def evaluate_high_fidelity(
    q,
):

    global hf_evaluation_counter


    evaluation_id = (
        hf_evaluation_counter
    )


    hf_evaluation_counter += 1


    coefficients = q_to_coefficients(
        q
    )


    evaluation_directory = os.path.join(
        HF_JOB_ROOT,
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


    left_result = run_patch_job(

        patch_name="left",

        E1=E1,

        E2=E2,

        G12=G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "romhf_{:03d}_left".format(
                evaluation_id
            ),

        cpus=args.cpus,
    )


    center_result = run_patch_job(

        patch_name="center",

        E1=E1,

        E2=E2,

        G12=G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "romhf_{:03d}_center".format(
                evaluation_id
            ),

        cpus=args.cpus,
    )


    right_result = run_patch_job(

        patch_name="right",

        E1=E1,

        E2=E2,

        G12=G12,

        coefficient_dictionary=
            coefficients,

        job_directory=
            evaluation_directory,

        job_name=
            "romhf_{:03d}_right".format(
                evaluation_id
            ),

        cpus=args.cpus,
    )


    patch_g = {}


    patch_g.update(
        left_result[
            "generalized_forces"
        ]
    )


    patch_g.update(
        center_result[
            "generalized_forces"
        ]
    )


    patch_g.update(
        right_result[
            "generalized_forces"
        ]
    )


    no_g = evaluate_neural_operators(

        E1,
        E2,
        G12,
        coefficients,
    )


    (
        physical,
        normalized,
        rms,
        maximum,
    ) = assemble_residual(
        patch_g,
        no_g,
    )


    details = {

        "left":
            left_result,

        "center":
            center_result,

        "right":
            right_result,

        "physical":
            physical,

        "normalized":
            normalized,

        "rms":
            rms,

        "maximum":
            maximum,

        "evaluation_id":
            evaluation_id,
    }


    print("")
    print(
        "HF evaluation {} | RMS = {:.8f}"
        .format(
            evaluation_id,
            rms,
        )
    )


    return (
        normalized,
        details,
    )


# ============================================================
# START HIGH-FIDELITY CORRECTION FROM ROM SOLUTION
# ============================================================

hf_start_time = time.time()


q = best_q_rom.copy()


residual, hf_details = (
    evaluate_high_fidelity(
        q
    )
)


current_norm = hf_details[
    "rms"
]


best_hf_q = q.copy()

best_hf_details = hf_details


hf_history = [

    {
        "Iteration":
            0,

        "RMSResidual":
            current_norm,

        "MaxResidual":
            hf_details[
                "maximum"
            ],
    }
]


# ============================================================
# FULL-FIDELITY CORRECTION
#
# Reuse the inverse Jacobian learned during cheap ROM solve.
# ============================================================

for iteration in range(
    1,
    args.hf_max_iter + 1,
):

    if current_norm <= (
        args.tol
    ):

        break


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


    q_trial = np.clip(

        q

        +

        damping
        *
        step,

        -args.q_limit,

        args.q_limit,
    )


    residual_trial, details_trial = (
        evaluate_high_fidelity(
            q_trial
        )
    )


    trial_norm = details_trial[
        "rms"
    ]


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


    if denominator > (
        1.0e-12
    ):

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


    q = q_trial

    residual = residual_trial

    hf_details = details_trial

    current_norm = trial_norm


    if current_norm < (
        best_hf_details[
            "rms"
        ]
    ):

        best_hf_q = q.copy()

        best_hf_details = (
            hf_details
        )


    hf_history.append(
        {
            "Iteration":
                iteration,

            "RMSResidual":
                current_norm,

            "MaxResidual":
                hf_details[
                    "maximum"
                ],
        }
    )


hf_elapsed = (
    time.time()
    -
    hf_start_time
)


# ============================================================
# SAVE HISTORIES
# ============================================================

with open(
    ROM_HISTORY_FILE,
    "w",
    newline=""
) as file_object:

    writer = csv.DictWriter(
        file_object,
        fieldnames=[
            "Iteration",
            "RMSResidual",
            "MaxResidual",
            "Damping",
        ],
    )


    writer.writeheader()

    writer.writerows(
        rom_history
    )


with open(
    HF_HISTORY_FILE,
    "w",
    newline=""
) as file_object:

    writer = csv.DictWriter(
        file_object,
        fieldnames=[
            "Iteration",
            "RMSResidual",
            "MaxResidual",
        ],
    )


    writer.writeheader()

    writer.writerows(
        hf_history
    )


# ============================================================
# FINAL SOLUTION
# ============================================================

final_coefficients = q_to_coefficients(
    best_hf_q
)


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
        best_hf_q,

    "c_scale":
        c_scale,

    "normalized_residual":
        best_hf_details[
            "normalized"
        ],

    "physical_residual":
        best_hf_details[
            "physical"
        ],

    "rms_residual":
        np.asarray(
            [
                best_hf_details[
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
    "rom_solution_7region.npz",
)


np.savez(
    solution_file,
    **solution_payload
)


# ============================================================
# COPY BEST HIGH-FIDELITY PATCH FILES
# ============================================================

for patch_name in [
    "left",
    "center",
    "right",
]:

    result = best_hf_details[
        patch_name
    ]


    shutil.copyfile(

        result[
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

        result[
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
            best_hf_details[
                "rms"
            ]
            <=
            args.tol
        ),

    "rom_best_rms":
        float(
            best_norm_rom
        ),

    "final_high_fidelity_rms":
        float(
            best_hf_details[
                "rms"
            ]
        ),

    "final_high_fidelity_max":
        float(
            best_hf_details[
                "maximum"
            ]
        ),

    "rom_residual_evaluations":
        int(
            rom_evaluation_counter
        ),

    "high_fidelity_residual_evaluations":
        int(
            hf_evaluation_counter
        ),

    "high_fidelity_abaqus_jobs":
        int(
            3
            *
            hf_evaluation_counter
        ),

    "old_direct_solver_abaqus_jobs":
        54,

    "abaqus_job_reduction_percent":
        float(

            100.0

            *

            (
                54
                -
                3
                *
                hf_evaluation_counter
            )

            /

            54.0
        ),

    "rom_phase_seconds":
        float(
            rom_elapsed
        ),

    "high_fidelity_phase_seconds":
        float(
            hf_elapsed
        ),

    "total_seconds":
        float(
            rom_elapsed
            +
            hf_elapsed
        ),

    "solution_file":
        solution_file,

    "final_directory":
        FINAL_DIR,
}


with open(
    os.path.join(
        RESULTS_DIR,
        "rom_solution_manifest.json",
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
    "=========================================="
)

print(
    "STEP 65H COMPLETE"
)

print(
    "=========================================="
)

print(
    json.dumps(
        manifest,
        indent=4,
    )
)