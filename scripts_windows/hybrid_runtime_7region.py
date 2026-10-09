from __future__ import print_function

import os
import math
import json
import glob
import subprocess

import numpy as np

from odbAccess import openOdb


# ============================================================
# ROOT
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
    "interfaces_7region"
)


FORCE_PCA_DIR = os.path.join(
    BASE_DIR,
    "force_pca_7region"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "no_models_7region"
)


TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "patch_templates_7region"
)


PATCH_METADATA_FILE = os.path.join(
    TEMPLATE_DIR,
    "patch_metadata.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEGMENT_INTERFACES = {

    "outer_left": [
        None,
        "left_outer",
    ],

    "inner_left": [
        "left_inner",
        "center_left",
    ],

    "inner_right": [
        "center_right",
        "right_inner",
    ],

    "outer_right": [
        "right_outer",
        None,
    ],
}


# ============================================================
# PATCH METADATA
# ============================================================

with open(
    PATCH_METADATA_FILE,
    "r"
) as file_object:

    PATCH_METADATA = json.load(
        file_object
    )[
        "patches"
    ]


# ============================================================
# CACHES
# ============================================================

INTERFACES = {}

MODELS = {}

FORCE_PCAS = {}


# ============================================================
# INTERFACES
# ============================================================

def load_interface(
    interface_name
):

    if interface_name not in INTERFACES:

        filename = os.path.join(
            INTERFACE_DIR,
            "interface_{}.npz".format(
                interface_name
            ),
        )


        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                filename
            )


        INTERFACES[
            interface_name
        ] = np.load(
            filename
        )


    return INTERFACES[
        interface_name
    ]


# ============================================================
# FORCE PCA
# ============================================================

def load_force_pca(
    interface_name
):

    if interface_name not in FORCE_PCAS:

        filename = os.path.join(
            FORCE_PCA_DIR,
            "force_pca_{}.npz".format(
                interface_name
            ),
        )


        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                filename
            )


        FORCE_PCAS[
            interface_name
        ] = np.load(
            filename
        )


    return FORCE_PCAS[
        interface_name
    ]


# ============================================================
# ENSEMBLE MODELS
# ============================================================

def load_force_model(
    segment
):

    if segment not in MODELS:

        filename = os.path.join(
            MODEL_DIR,
            "{}_direct_ensemble_force.npz".format(
                segment
            ),
        )


        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                filename
            )


        MODELS[
            segment
        ] = np.load(
            filename
        )


    return MODELS[
        segment
    ]


# ============================================================
# EXACT GELU
# ============================================================

def gelu_exact(
    x
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )


    flattened = x.reshape(
        -1
    )


    output = np.empty_like(
        flattened
    )


    sqrt_two = math.sqrt(
        2.0
    )


    for index in range(
        flattened.shape[
            0
        ]
    ):

        value = flattened[
            index
        ]


        output[
            index
        ] = (
            0.5
            *
            value
            *
            (
                1.0
                +
                math.erf(
                    value
                    /
                    sqrt_two
                )
            )
        )


    return output.reshape(
        x.shape
    )


# ============================================================
# ONE ENSEMBLE FORCE PREDICTION
# ============================================================

def predict_force_coefficients(

    segment,

    branch_physical,
):

    model = load_force_model(
        segment
    )


    branch_mean = model[
        "branch_mean"
    ]


    branch_std = model[
        "branch_std"
    ]


    force_mean = model[
        "force_coeff_mean"
    ]


    force_std = model[
        "force_coeff_std"
    ]


    branch_physical = np.asarray(
        branch_physical,
        dtype=np.float64,
    )


    x0 = (
        branch_physical
        -
        branch_mean
    ) / branch_std


    number_members = int(
        model[
            "number_members"
        ][0]
    )


    predictions = []


    for member_index in range(
        number_members
    ):

        x = x0.copy()


        number_layers = int(
            model[
                "member_{}_number_layers".format(
                    member_index
                )
            ][0]
        )


        for layer_index in range(
            number_layers
        ):

            W = model[
                "member_{}_W{}".format(
                    member_index,
                    layer_index,
                )
            ]


            b = model[
                "member_{}_b{}".format(
                    member_index,
                    layer_index,
                )
            ]


            x = (
                W
                @
                x
                +
                b
            )


            if layer_index < (
                number_layers
                -
                1
            ):

                x = gelu_exact(
                    x
                )


        physical = (
            x
            *
            force_std
            +
            force_mean
        )


        predictions.append(
            physical
        )


    return np.mean(
        np.asarray(
            predictions,
            dtype=np.float64,
        ),
        axis=0,
    )


# ============================================================
# INTERFACE DISPLACEMENT
# ============================================================

def reconstruct_interface_displacement(

    interface_name,

    coefficients,
):

    interface = load_interface(
        interface_name
    )


    mean = interface[
        "mean"
    ]


    basis = interface[
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
            "Coefficient mismatch at {}."
            .format(
                interface_name
            )
        )


    flattened = (
        mean
        +
        basis
        @
        coefficients
    )


    return flattened.reshape(
        -1,
        3,
    )


# ============================================================
# FIND ODB INSTANCE
# ============================================================

def find_odb_instance(

    odb,

    requested_name,
):

    requested_lower = (
        requested_name.lower()
    )


    for key, value in (
        odb.rootAssembly.instances.items()
    ):

        if key.lower() == (
            requested_lower
        ):

            return value


    raise RuntimeError(
        "ODB instance not found: {}"
        .format(
            requested_name
        )
    )


# ============================================================
# MATERIAL
# ============================================================

def glass_material_lines(

    E1,

    E2,

    G12,
):

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


    line2 = (
        "{:.12g},"
    ).format(
        G23
    )


    return (
        line1,
        line2,
    )


# ============================================================
# INTERFACE BC
# ============================================================

def create_interface_bc_text(

    patch_name,

    coefficient_dictionary,
):

    patch = PATCH_METADATA[
        patch_name
    ]


    lines = []


    for interface_name, info in (
        patch[
            "interfaces"
        ].items()
    ):

        displacement = (
            reconstruct_interface_displacement(

                interface_name,

                coefficient_dictionary[
                    interface_name
                ],
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
                "Interface node count mismatch: {}"
                .format(
                    interface_name
                )
            )


        for local_index, node_label in enumerate(
            node_labels
        ):

            values = displacement[
                local_index
            ]


            for dof in range(
                3
            ):

                lines.append(
                    (
                        "Composite-1.{}, {}, {}, {:.12e}"
                    ).format(
                        node_label,
                        dof + 1,
                        dof + 1,
                        values[
                            dof
                        ],
                    )
                )


    return "\n".join(
        lines
    )


# ============================================================
# RENDER PATCH
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
    ) as file_object:

        text = file_object.read()


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
    ) as file_object:

        file_object.write(
            text
        )


# ============================================================
# CLEAN JOB
# ============================================================

def cleanup_job_files(

    directory,

    job_name,
):

    for path in glob.glob(
        os.path.join(
            directory,
            "{}.*".format(
                job_name
            ),
        )
    ):

        try:

            os.remove(
                path
            )

        except Exception:

            pass


# ============================================================
# RUN FE PATCH
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

        patch_name=

            patch_name,

        E1=

            E1,

        E2=

            E2,

        G12=

            G12,

        coefficient_dictionary=

            coefficient_dictionary,

        output_file=

            input_file,
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
        "Running FE patch:",
        patch_name,
        "|",
        job_name
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
        or
        not os.path.isfile(
            odb_file
        )
    ):

        raise RuntimeError(
            "Abaqus patch failed: {}"
            .format(
                patch_name
            )
        )


    # ========================================================
    # EXTRACT INTERFACE RF
    # ========================================================

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
        region=
            instance
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


    for interface_name, info in (
        PATCH_METADATA[
            patch_name
        ][
            "interfaces"
        ].items()
    ):

        node_labels = info[
            "node_labels"
        ]


        reaction_matrix = []


        for node_label in node_labels:

            if node_label not in RF_map:

                odb.close()


                raise RuntimeError(
                    (
                        "RF missing at interface node {}"
                    ).format(
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


        flattened = reaction_matrix.reshape(
            -1
        )


        displacement_basis = load_interface(
            interface_name
        )[
            "basis"
        ]


        generalized_forces[
            interface_name
        ] = (
            displacement_basis.T
            @
            flattened
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
# NO FORCE COEFFICIENT -> GENERALIZED FORCE
# ============================================================

def force_coefficients_to_generalized(

    segment,

    force_coefficients,
):

    interfaces = SEGMENT_INTERFACES[
        segment
    ]


    result = {}


    cursor = 0


    for interface_name in interfaces:

        if interface_name is None:

            continue


        force_pca = load_force_pca(
            interface_name
        )


        number_modes = int(
            force_pca[
                "number_modes"
            ][0]
        )


        current_coefficients = (
            force_coefficients[
                cursor:
                cursor
                +
                number_modes
            ]
        )


        g = (
            force_pca[
                "g_mean"
            ]
            +
            force_pca[
                "g_matrix"
            ]
            @
            current_coefficients
        )


        result[
            interface_name
        ] = np.asarray(
            g,
            dtype=np.float64,
        )


        cursor += (
            number_modes
        )


    if cursor != force_coefficients.shape[
        0
    ]:

        raise RuntimeError(
            (
                "Unused force coefficients for {}: "
                "{} of {}"
            ).format(
                segment,
                cursor,
                force_coefficients.shape[
                    0
                ],
            )
        )


    return result


# ============================================================
# FOUR DIRECT NOs
# ============================================================

def evaluate_neural_operators(

    E1,

    E2,

    G12,

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


    segment_branches = {

        "outer_left":

            np.concatenate(
                [
                    material,

                    coefficient_dictionary[
                        "left_outer"
                    ],
                ]
            ),

        "inner_left":

            np.concatenate(
                [
                    material,

                    coefficient_dictionary[
                        "left_inner"
                    ],

                    coefficient_dictionary[
                        "center_left"
                    ],
                ]
            ),

        "inner_right":

            np.concatenate(
                [
                    material,

                    coefficient_dictionary[
                        "center_right"
                    ],

                    coefficient_dictionary[
                        "right_inner"
                    ],
                ]
            ),

        "outer_right":

            np.concatenate(
                [
                    material,

                    coefficient_dictionary[
                        "right_outer"
                    ],
                ]
            ),
    }


    all_generalized_forces = {}


    for segment, branch in (
        segment_branches.items()
    ):

        force_coefficients = (
            predict_force_coefficients(

                segment,

                branch,
            )
        )


        segment_forces = (
            force_coefficients_to_generalized(

                segment,

                force_coefficients,
            )
        )


        all_generalized_forces.update(
            segment_forces
        )


    return all_generalized_forces


# ============================================================
# RESIDUAL FORCE SCALES
# ============================================================

def get_residual_force_scales():

    scales = {}


    for segment, interfaces in (
        SEGMENT_INTERFACES.items()
    ):

        model = load_force_model(
            segment
        )


        g_std = np.abs(
            model[
                "g_std"
            ]
        )


        cursor = 0


        for interface_name in interfaces:

            if interface_name is None:

                continue


            number_modes = load_interface(
                interface_name
            )[
                "basis"
            ].shape[1]


            current = g_std[
                cursor:
                cursor
                +
                number_modes
            ]


            scales[
                interface_name
            ] = np.maximum(
                current,
                1.0,
            )


            cursor += (
                number_modes
            )


        if cursor != g_std.shape[
            0
        ]:

            raise RuntimeError(
                "g_std dimension mismatch for {}."
                .format(
                    segment
                )
            )


    return scales