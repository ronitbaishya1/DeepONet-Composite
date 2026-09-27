from __future__ import print_function

import os
import json
import glob
import shutil
import subprocess

import numpy as np

from odbAccess import openOdb


# ============================================================
# PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


INTERFACE_DIR = os.path.join(
    BASE_DIR,
    "interfaces",
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "no_models",
)


TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "patch_templates",
)


PATCH_METADATA_FILE = os.path.join(
    TEMPLATE_DIR,
    "patch_metadata.json",
)


# ============================================================
# LOAD PATCH METADATA
# ============================================================

with open(
    PATCH_METADATA_FILE,
    "r"
) as f:

    PATCH_METADATA = json.load(
        f
    )[
        "patches"
    ]


# ============================================================
# INTERFACE CACHE
# ============================================================

INTERFACES = {}


def load_interface(
    name
):

    if name not in INTERFACES:

        path = os.path.join(
            INTERFACE_DIR,
            "interface_{}.npz".format(
                name
            ),
        )


        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                path
            )


        INTERFACES[
            name
        ] = np.load(
            path
        )


    return INTERFACES[
        name
    ]


# ============================================================
# FORCE MODEL CACHE
# ============================================================

FORCE_MODELS = {}


def load_force_model(
    side
):

    if side not in FORCE_MODELS:

        path = os.path.join(
            MODEL_DIR,
            "{}_force_model.npz".format(
                side
            ),
        )


        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                path
            )


        FORCE_MODELS[
            side
        ] = np.load(
            path
        )


    return FORCE_MODELS[
        side
    ]


# ============================================================
# NUMPY FORCE NETWORK
# ============================================================

def predict_force(
    side,
    branch_physical,
):

    model = load_force_model(
        side
    )


    branch_mean = model[
        "branch_mean"
    ]


    branch_std = model[
        "branch_std"
    ]


    force_mean = model[
        "force_mean"
    ]


    force_std = model[
        "force_std"
    ]


    x = (
        np.asarray(
            branch_physical,
            dtype=np.float64,
        )
        -
        branch_mean
    ) / branch_std


    number_layers = int(
        model[
            "number_layers"
        ][
            0
        ]
    )


    for layer_index in range(
        number_layers
    ):

        W = model[
            "W{}".format(
                layer_index
            )
        ]


        b = model[
            "b{}".format(
                layer_index
            )
        ]


        x = (
            W
            @ x
            + b
        )


        if layer_index < (
            number_layers
            - 1
        ):

            x = np.tanh(
                x
            )


    return (
        x
        * force_std
        + force_mean
    )


# ============================================================
# INTERFACE DISPLACEMENT RECONSTRUCTION
# ============================================================

def reconstruct_interface_displacement(
    interface_name,
    coefficients,
):

    data = load_interface(
        interface_name
    )


    mean = data[
        "mean"
    ]


    basis = data[
        "basis"
    ]


    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )


    if coefficients.shape[
        0
    ] != basis.shape[
        1
    ]:

        raise RuntimeError(
            "Coefficient dimension mismatch for {}".format(
                interface_name
            )
        )


    flattened = (
        mean
        +
        basis
        @ coefficients
    )


    return flattened.reshape(
        -1,
        3,
    )


# ============================================================
# FIND INSTANCE IN ODB CASE-INSENSITIVELY
# ============================================================

def find_odb_instance(
    odb,
    requested_name,
):

    requested_lower = requested_name.lower()


    for key, value in odb.rootAssembly.instances.items():

        if key.lower() == requested_lower:

            return value


    raise RuntimeError(
        "ODB instance not found: {}".format(
            requested_name
        )
    )


# ============================================================
# MATERIAL TEXT
# ============================================================

def glass_material_lines(
    E1,
    E2,
    G12,
):

    # Preserve all fixed values from the baseline model.
    E3 = 12000.0

    NU12 = 0.28

    NU13 = 0.28

    NU23 = 0.40

    G13 = 4500.0

    G23 = 3500.0


    line1 = (
        "{:.12g}, {:.12g}, {:.12g}, "
        "{:.12g}, {:.12g}, {:.12g}, "
        "{:.12g}, {:.12g}"
    ).format(
        E1,
        E2,
        E3,
        NU12,
        NU13,
        NU23,
        G12,
        G13,
    )


    line2 = "{:.12g},".format(
        G23
    )


    return (
        line1,
        line2,
    )


# ============================================================
# CREATE INTERFACE BC TEXT
# ============================================================

def create_interface_bc_text(
    patch_name,
    coefficient_dictionary,
):

    patch = PATCH_METADATA[
        patch_name
    ]


    lines = []


    for interface_name, info in patch[
        "interfaces"
    ].items():

        coefficients = coefficient_dictionary[
            interface_name
        ]


        displacement = (
            reconstruct_interface_displacement(
                interface_name,
                coefficients,
            )
        )


        node_labels = info[
            "node_labels"
        ]


        if len(
            node_labels
        ) != displacement.shape[
            0
        ]:

            raise RuntimeError(
                "Interface node count mismatch: {}".format(
                    interface_name
                )
            )


        for local_index, node_label in enumerate(
            node_labels
        ):

            values = displacement[
                local_index
            ]


            lines.append(
                "Composite-1.{}, 1, 1, {:.12e}".format(
                    node_label,
                    values[
                        0
                    ],
                )
            )


            lines.append(
                "Composite-1.{}, 2, 2, {:.12e}".format(
                    node_label,
                    values[
                        1
                    ],
                )
            )


            lines.append(
                "Composite-1.{}, 3, 3, {:.12e}".format(
                    node_label,
                    values[
                        2
                    ],
                )
            )


    return "\n".join(
        lines
    )


# ============================================================
# RENDER ONE PATCH INPUT
# ============================================================

def render_patch_input(
    patch_name,
    E1,
    E2,
    G12,
    coefficient_dictionary,
    output_file,
):

    template_file = os.path.join(
        TEMPLATE_DIR,
        "{}_template.inp".format(
            patch_name
        ),
    )


    with open(
        template_file,
        "r"
    ) as f:

        text = f.read()


    line1, line2 = glass_material_lines(
        E1,
        E2,
        G12,
    )


    interface_bc = create_interface_bc_text(
        patch_name,
        coefficient_dictionary,
    )


    text = text.replace(
        "@@GLASS_LINE1@@",
        line1,
    )


    text = text.replace(
        "@@GLASS_LINE2@@",
        line2,
    )


    text = text.replace(
        "@@INTERFACE_BC@@",
        interface_bc,
    )


    if "@@" in text:

        raise RuntimeError(
            "Unreplaced template marker remains."
        )


    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            text
        )


# ============================================================
# REMOVE OLD JOB FILES
# ============================================================

def cleanup_job_files(
    directory,
    job_name,
):

    patterns = [

        "{}.*".format(
            job_name
        ),
    ]


    for pattern in patterns:

        for path in glob.glob(
            os.path.join(
                directory,
                pattern
            )
        ):

            try:

                os.remove(
                    path
                )

            except Exception:

                pass


# ============================================================
# RUN ABAQUS PATCH
# ============================================================

def run_patch_job(
    patch_name,
    E1,
    E2,
    G12,
    coefficient_dictionary,
    job_directory,
    job_name,
    cpus=4,
):

    if not os.path.isdir(
        job_directory
    ):

        os.makedirs(
            job_directory
        )


    cleanup_job_files(
        job_directory,
        job_name,
    )


    input_file = os.path.join(
        job_directory,
        job_name
        +
        ".inp",
    )


    render_patch_input(
        patch_name=patch_name,
        E1=E1,
        E2=E2,
        G12=G12,
        coefficient_dictionary=coefficient_dictionary,
        output_file=input_file,
    )


    command = (
        'abaqus job="{}" '
        'input="{}" '
        'cpus={} '
        'interactive'
    ).format(
        job_name,
        os.path.basename(
            input_file
        ),
        cpus,
    )


    print(
        "Running patch:",
        patch_name,
        "| job:",
        job_name,
    )


    return_code = subprocess.call(
        command,
        cwd=job_directory,
        shell=True,
    )


    odb_file = os.path.join(
        job_directory,
        job_name
        +
        ".odb",
    )


    if (
        return_code != 0
        or not os.path.isfile(
            odb_file
        )
    ):

        raise RuntimeError(
            "Abaqus patch failed: {}".format(
                patch_name
            )
        )


    # --------------------------------------------------------
    # EXTRACT INTERFACE REACTIONS
    # --------------------------------------------------------

    odb = openOdb(
        odb_file,
        readOnly=True,
    )


    instance = find_odb_instance(
        odb,
        "Composite-1",
    )


    step = odb.steps[
        "Loading"
    ]


    frame = step.frames[
        -1
    ]


    RF_field = frame.fieldOutputs[
        "RF"
    ].getSubset(
        region=instance
    )


    RF_map = {}


    for value in RF_field.values:

        RF_map[
            value.nodeLabel
        ] = np.asarray(
            value.data,
            dtype=np.float64,
        )


    generalized_forces = {}


    patch_info = PATCH_METADATA[
        patch_name
    ]


    for interface_name, info in patch_info[
        "interfaces"
    ].items():

        node_labels = info[
            "node_labels"
        ]


        reaction_matrix = []


        for node_label in node_labels:

            if node_label not in RF_map:

                raise RuntimeError(
                    "RF missing at interface node {}".format(
                        node_label
                    )
                )


            reaction_matrix.append(
                RF_map[
                    node_label
                ]
            )


        reaction_matrix = np.asarray(
            reaction_matrix,
            dtype=np.float64,
        )


        flattened_reaction = reaction_matrix.reshape(
            -1
        )


        basis = load_interface(
            interface_name
        )[
            "basis"
        ]


        generalized_forces[
            interface_name
        ] = (
            basis.T
            @ flattened_reaction
        )


    odb.close()


    return {

        "generalized_forces":
            generalized_forces,

        "odb_file":
            odb_file,

        "input_file":
            input_file,

        "job_name":
            job_name,
    }


# ============================================================
# NO GENERALIZED FORCES
# ============================================================

def evaluate_neural_operators(
    E1,
    E2,
    G12,
    coefficient_dictionary,
):

    # --------------------------------------------------------
    # LEFT NO
    #
    # branch =
    # E1,E2,G12,m6(4),m2(5)
    # --------------------------------------------------------

    left_branch = np.concatenate(
        [

            np.array(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary[
                    "m6"
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary[
                    "m2"
                ],
                dtype=np.float64,
            ),
        ]
    )


    left_force = predict_force(
        "left",
        left_branch,
    )


    left_model = load_force_model(
        "left"
    )


    left_count = int(
        left_model[
            "n_left_force_modes"
        ][
            0
        ]
    )


    # --------------------------------------------------------
    # RIGHT NO
    #
    # branch =
    # E1,E2,G12,p2(5),p6(4)
    # --------------------------------------------------------

    right_branch = np.concatenate(
        [

            np.array(
                [
                    E1,
                    E2,
                    G12,
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary[
                    "p2"
                ],
                dtype=np.float64,
            ),

            np.asarray(
                coefficient_dictionary[
                    "p6"
                ],
                dtype=np.float64,
            ),
        ]
    )


    right_force = predict_force(
        "right",
        right_branch,
    )


    right_model = load_force_model(
        "right"
    )


    right_count = int(
        right_model[
            "n_left_force_modes"
        ][
            0
        ]
    )


    return {

        "m6":
            left_force[
                :left_count
            ],

        "m2":
            left_force[
                left_count:
            ],

        "p2":
            right_force[
                :right_count
            ],

        "p6":
            right_force[
                right_count:
            ],
    }


# ============================================================
# FORCE NORMALIZATION SCALES
# ============================================================

def get_residual_force_scales():

    left = load_force_model(
        "left"
    )


    right = load_force_model(
        "right"
    )


    left_std = np.abs(
        left[
            "force_std"
        ]
    )


    right_std = np.abs(
        right[
            "force_std"
        ]
    )


    left_count = int(
        left[
            "n_left_force_modes"
        ][
            0
        ]
    )


    right_count = int(
        right[
            "n_left_force_modes"
        ][
            0
        ]
    )


    scales = {

        "m6":
            left_std[
                :left_count
            ],

        "m2":
            left_std[
                left_count:
            ],

        "p2":
            right_std[
                :right_count
            ],

        "p6":
            right_std[
                right_count:
            ],
    }


    for key in scales:

        scales[
            key
        ] = np.maximum(
            scales[
                key
            ],
            1.0,
        )


    return scales