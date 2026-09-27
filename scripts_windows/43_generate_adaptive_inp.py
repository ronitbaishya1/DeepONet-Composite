from __future__ import print_function

import os
import csv
import argparse

import numpy as np


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "left",
        "right",
    ],
    required=True,
)


args = parser.parse_args()


# ============================================================
# PROJECT PATHS
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


ENRICHMENT_DIR = os.path.join(
    BASE_DIR,
    "adaptive_enrichment",
)


OUTPUT_DIR = os.path.join(
    ENRICHMENT_DIR,
    "{}_inp".format(
        args.segment
    ),
)


if not os.path.isdir(
    OUTPUT_DIR
):

    os.makedirs(
        OUTPUT_DIR
    )


# ============================================================
# SEGMENT CONFIGURATION
# ============================================================

if args.segment == "left":

    interface_left_name = "m6"

    interface_right_name = "m2"

    left_modes = 4

    right_modes = 5


else:

    interface_left_name = "p2"

    interface_right_name = "p6"

    left_modes = 5

    right_modes = 4


DESIGN_FILE = os.path.join(
    ENRICHMENT_DIR,
    "{}_enrichment_design_40.csv".format(
        args.segment
    ),
)


# ============================================================
# CHECK DESIGN FILE
# ============================================================

if not os.path.isfile(
    DESIGN_FILE
):

    raise RuntimeError(
        "Design file not found:\n{}".format(
            DESIGN_FILE
        )
    )


# ============================================================
# LOAD INTERFACE PCA DATA
# ============================================================

left_interface_file = os.path.join(
    INTERFACE_DIR,
    "interface_{}.npz".format(
        interface_left_name
    ),
)


right_interface_file = os.path.join(
    INTERFACE_DIR,
    "interface_{}.npz".format(
        interface_right_name
    ),
)


if not os.path.isfile(
    left_interface_file
):

    raise RuntimeError(
        "Missing interface file:\n{}".format(
            left_interface_file
        )
    )


if not os.path.isfile(
    right_interface_file
):

    raise RuntimeError(
        "Missing interface file:\n{}".format(
            right_interface_file
        )
    )


interface_left = np.load(
    left_interface_file
)


interface_right = np.load(
    right_interface_file
)


left_coordinates = interface_left[
    "coordinates"
].astype(
    np.float64
)


right_coordinates = interface_right[
    "coordinates"
].astype(
    np.float64
)


left_mean = interface_left[
    "mean"
].astype(
    np.float64
)


right_mean = interface_right[
    "mean"
].astype(
    np.float64
)


left_basis = interface_left[
    "basis"
].astype(
    np.float64
)


right_basis = interface_right[
    "basis"
].astype(
    np.float64
)


# ============================================================
# BASIC CHECKS
# ============================================================

if left_basis.shape[1] != left_modes:

    raise RuntimeError(
        "Unexpected number of modes for {}. "
        "Expected {}, found {}.".format(
            interface_left_name,
            left_modes,
            left_basis.shape[1],
        )
    )


if right_basis.shape[1] != right_modes:

    raise RuntimeError(
        "Unexpected number of modes for {}. "
        "Expected {}, found {}.".format(
            interface_right_name,
            right_modes,
            right_basis.shape[1],
        )
    )


# ============================================================
# LOCAL BLOCK X COORDINATES
# ============================================================

x_left = float(
    np.mean(
        left_coordinates[
            :,
            0
        ]
    )
)


x_right = float(
    np.mean(
        right_coordinates[
            :,
            0
        ]
    )
)


# ============================================================
# ORIGINAL LOCAL BULK GRID
#
# 8 x-nodes -> 7 elements through the block length
# ============================================================

x_values = np.linspace(
    x_left,
    x_right,
    8,
)


y_values = np.sort(
    np.unique(
        np.round(
            left_coordinates[
                :,
                1
            ],
            8,
        )
    )
)


z_values = np.sort(
    np.unique(
        np.round(
            left_coordinates[
                :,
                2
            ],
            8,
        )
    )
)


# ============================================================
# CHECK INTERFACE GRID
# ============================================================

expected_interface_nodes = (
    len(
        y_values
    )
    *
    len(
        z_values
    )
)


if expected_interface_nodes != len(
    left_coordinates
):

    raise RuntimeError(
        "Left interface does not form expected Y-Z grid."
    )


if expected_interface_nodes != len(
    right_coordinates
):

    raise RuntimeError(
        "Right interface does not form expected Y-Z grid."
    )


print("")
print(
    "Segment:",
    args.segment
)


print(
    "Left interface:",
    interface_left_name,
    "x =",
    x_left,
)


print(
    "Right interface:",
    interface_right_name,
    "x =",
    x_right,
)


print("")
print(
    "Grid:"
)


print(
    "X nodes:",
    len(
        x_values
    )
)


print(
    "Y nodes:",
    len(
        y_values
    )
)


print(
    "Z nodes:",
    len(
        z_values
    )
)


print(
    "Total nodes:",
    len(
        x_values
    )
    *
    len(
        y_values
    )
    *
    len(
        z_values
    )
)


print(
    "Total elements:",
    (
        len(
            x_values
        )
        -
        1
    )
    *
    (
        len(
            y_values
        )
        -
        1
    )
    *
    (
        len(
            z_values
        )
        -
        1
    )
)


# ============================================================
# PCA RECONSTRUCTION
# ============================================================

def reconstruct_interface(
    mean,
    basis,
    coefficients,
):

    displacement_flat = (

        mean

        +

        np.dot(
            basis,
            coefficients
        )
    )


    return displacement_flat.reshape(
        -1,
        3,
    )


# ============================================================
# MAP Y,Z -> DISPLACEMENT
# ============================================================

def make_displacement_map(
    coordinates,
    displacement,
):

    mapping = {}


    for index in range(
        len(
            coordinates
        )
    ):

        key = (

            round(
                float(
                    coordinates[
                        index,
                        1
                    ]
                ),
                8,
            ),

            round(
                float(
                    coordinates[
                        index,
                        2
                    ]
                ),
                8,
            ),
        )


        mapping[
            key
        ] = displacement[
            index
        ]


    return mapping


# ============================================================
# READ DESIGN TABLE
# ============================================================

with open(
    DESIGN_FILE,
    "r"
) as file_object:

    reader = csv.DictReader(
        file_object
    )


    design_rows = list(
        reader
    )


print("")
print(
    "Design cases:",
    len(
        design_rows
    )
)


# ============================================================
# CREATE EACH ABAQUS INPUT FILE
# ============================================================

for case_number, row in enumerate(
    design_rows
):

    case_id = row[
        "CaseID"
    ]


    E1 = float(
        row[
            "E1"
        ]
    )


    E2 = float(
        row[
            "E2"
        ]
    )


    G12 = float(
        row[
            "G12"
        ]
    )


    # ========================================================
    # LEFT INTERFACE PCA COEFFICIENTS
    # ========================================================

    left_coefficients = np.asarray(
        [
            float(
                row[
                    "c_{}_{}".format(
                        interface_left_name,
                        mode_index,
                    )
                ]
            )

            for mode_index in range(
                left_modes
            )
        ],
        dtype=np.float64,
    )


    # ========================================================
    # RIGHT INTERFACE PCA COEFFICIENTS
    # ========================================================

    right_coefficients = np.asarray(
        [
            float(
                row[
                    "c_{}_{}".format(
                        interface_right_name,
                        mode_index,
                    )
                ]
            )

            for mode_index in range(
                right_modes
            )
        ],
        dtype=np.float64,
    )


    # ========================================================
    # RECONSTRUCT FULL BOUNDARY DISPLACEMENT
    # ========================================================

    left_displacement = reconstruct_interface(
        left_mean,
        left_basis,
        left_coefficients,
    )


    right_displacement = reconstruct_interface(
        right_mean,
        right_basis,
        right_coefficients,
    )


    left_displacement_map = make_displacement_map(
        left_coordinates,
        left_displacement,
    )


    right_displacement_map = make_displacement_map(
        right_coordinates,
        right_displacement,
    )


    # ========================================================
    # BUILD NODES
    # ========================================================

    node_map = {}

    nodes = []

    node_label = 1


    for ix, x_value in enumerate(
        x_values
    ):

        for iy, y_value in enumerate(
            y_values
        ):

            for iz, z_value in enumerate(
                z_values
            ):

                node_map[
                    (
                        ix,
                        iy,
                        iz,
                    )
                ] = node_label


                nodes.append(
                    (
                        node_label,
                        float(
                            x_value
                        ),
                        float(
                            y_value
                        ),
                        float(
                            z_value
                        ),
                    )
                )


                node_label += 1


    # ========================================================
    # BUILD C3D8R ELEMENTS
    # ========================================================

    elements = []

    element_label = 1


    for ix in range(
        len(
            x_values
        )
        -
        1
    ):

        for iy in range(
            len(
                y_values
            )
            -
            1
        ):

            for iz in range(
                len(
                    z_values
                )
                -
                1
            ):

                n1 = node_map[
                    (
                        ix,
                        iy,
                        iz,
                    )
                ]


                n2 = node_map[
                    (
                        ix + 1,
                        iy,
                        iz,
                    )
                ]


                n3 = node_map[
                    (
                        ix + 1,
                        iy + 1,
                        iz,
                    )
                ]


                n4 = node_map[
                    (
                        ix,
                        iy + 1,
                        iz,
                    )
                ]


                n5 = node_map[
                    (
                        ix,
                        iy,
                        iz + 1,
                    )
                ]


                n6 = node_map[
                    (
                        ix + 1,
                        iy,
                        iz + 1,
                    )
                ]


                n7 = node_map[
                    (
                        ix + 1,
                        iy + 1,
                        iz + 1,
                    )
                ]


                n8 = node_map[
                    (
                        ix,
                        iy + 1,
                        iz + 1,
                    )
                ]


                elements.append(
                    (
                        element_label,
                        n1,
                        n2,
                        n3,
                        n4,
                        n5,
                        n6,
                        n7,
                        n8,
                    )
                )


                element_label += 1


    # ========================================================
    # OUTPUT INPUT FILE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        "{}.inp".format(
            case_id
        ),
    )


    with open(
        output_file,
        "w"
    ) as file_object:

        # ====================================================
        # HEADING
        # ====================================================

        file_object.write(
            "*Heading\n"
        )


        file_object.write(
            "** Adaptive enrichment case {}\n".format(
                case_id
            )
        )


        file_object.write(
            "** Segment = {}\n".format(
                args.segment
            )
        )


        file_object.write(
            "** Material axes: 1=X, 2=Y, 3=Z\n"
        )


        # ====================================================
        # PART
        # ====================================================

        file_object.write(
            "*Part, name=BLOCK\n"
        )


        # ====================================================
        # NODES
        # ====================================================

        file_object.write(
            "*Node\n"
        )


        for node in nodes:

            file_object.write(
                "{}, {:.12f}, {:.12f}, {:.12f}\n".format(
                    node[
                        0
                    ],
                    node[
                        1
                    ],
                    node[
                        2
                    ],
                    node[
                        3
                    ],
                )
            )


        # ====================================================
        # ELEMENTS
        # ====================================================

        file_object.write(
            "*Element, type=C3D8R\n"
        )


        for element in elements:

            file_object.write(
                "{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                    element[
                        0
                    ],
                    element[
                        1
                    ],
                    element[
                        2
                    ],
                    element[
                        3
                    ],
                    element[
                        4
                    ],
                    element[
                        5
                    ],
                    element[
                        6
                    ],
                    element[
                        7
                    ],
                    element[
                        8
                    ],
                )
            )


        # ====================================================
        # ELEMENT SET
        # ====================================================

        file_object.write(
            "*Elset, elset=EALL, generate\n"
        )


        file_object.write(
            "1, {}, 1\n".format(
                len(
                    elements
                )
            )
        )


        # ====================================================
        # MATERIAL ORIENTATION
        #
        # Local material direction:
        #
        # 1 = global X
        # 2 = global Y
        # 3 = global Z
        #
        # Abaqus requires this because Engineering Constants
        # define an orthotropic material.
        # ====================================================

        file_object.write(
            "*Orientation, "
            "name=ORI_COMPOSITE, "
            "system=RECTANGULAR, "
            "definition=COORDINATES\n"
        )


        file_object.write(
            "1.0, 0.0, 0.0, "
            "0.0, 1.0, 0.0\n"
        )


        file_object.write(
            "3, 0.0\n"
        )


        # ====================================================
        # SOLID SECTION
        #
        # IMPORTANT:
        # orientation=ORI_COMPOSITE is required for the
        # orthotropic material.
        # ====================================================

        file_object.write(
            "*Solid Section, "
            "elset=EALL, "
            "material=COMPOSITE, "
            "orientation=ORI_COMPOSITE\n"
        )


        file_object.write(
            ",\n"
        )


        file_object.write(
            "*End Part\n"
        )


        # ====================================================
        # ASSEMBLY
        # ====================================================

        file_object.write(
            "*Assembly, name=Assembly\n"
        )


        file_object.write(
            "*Instance, name=BLOCK-1, part=BLOCK\n"
        )


        file_object.write(
            "*End Instance\n"
        )


        file_object.write(
            "*End Assembly\n"
        )


        # ====================================================
        # MATERIAL
        # ====================================================

        file_object.write(
            "*Material, name=COMPOSITE\n"
        )


        file_object.write(
            "*Elastic, type=ENGINEERING CONSTANTS\n"
        )


        # ----------------------------------------------------
        # Engineering constants:
        #
        # E1  = variable
        # E2  = variable
        # E3  = 12000 MPa
        #
        # nu12 = 0.28
        # nu13 = 0.28
        # nu23 = 0.40
        #
        # G12 = variable
        # G13 = 4500 MPa
        # G23 = 3500 MPa
        # ----------------------------------------------------

        file_object.write(
            "{:.12e}, {:.12e}, {:.12e}, "
            "{:.12e}, {:.12e}, {:.12e}, "
            "{:.12e}, {:.12e}\n".format(
                E1,
                E2,
                12000.0,
                0.28,
                0.28,
                0.40,
                G12,
                4500.0,
            )
        )


        file_object.write(
            "{:.12e}\n".format(
                3500.0
            )
        )


        # ====================================================
        # ANALYSIS STEP
        # ====================================================

        file_object.write(
            "*Step, name=Loading, nlgeom=NO\n"
        )


        file_object.write(
            "*Static\n"
        )


        file_object.write(
            "0.1, 1.0, 1.0e-06, 0.1\n"
        )


        # ====================================================
        # PRESCRIBED INTERFACE DISPLACEMENTS
        # ====================================================

        file_object.write(
            "*Boundary\n"
        )


        # ====================================================
        # LEFT FACE
        # ====================================================

        ix = 0


        for iy, y_value in enumerate(
            y_values
        ):

            for iz, z_value in enumerate(
                z_values
            ):

                node = node_map[
                    (
                        ix,
                        iy,
                        iz,
                    )
                ]


                key = (

                    round(
                        float(
                            y_value
                        ),
                        8,
                    ),

                    round(
                        float(
                            z_value
                        ),
                        8,
                    ),
                )


                if key not in left_displacement_map:

                    raise RuntimeError(
                        "Missing left interface displacement "
                        "for Y,Z = {}".format(
                            key
                        )
                    )


                displacement = left_displacement_map[
                    key
                ]


                for component in range(
                    3
                ):

                    file_object.write(
                        "BLOCK-1.{}, {}, {}, {:.12e}\n".format(
                            node,
                            component + 1,
                            component + 1,
                            displacement[
                                component
                            ],
                        )
                    )


        # ====================================================
        # RIGHT FACE
        # ====================================================

        ix = (
            len(
                x_values
            )
            -
            1
        )


        for iy, y_value in enumerate(
            y_values
        ):

            for iz, z_value in enumerate(
                z_values
            ):

                node = node_map[
                    (
                        ix,
                        iy,
                        iz,
                    )
                ]


                key = (

                    round(
                        float(
                            y_value
                        ),
                        8,
                    ),

                    round(
                        float(
                            z_value
                        ),
                        8,
                    ),
                )


                if key not in right_displacement_map:

                    raise RuntimeError(
                        "Missing right interface displacement "
                        "for Y,Z = {}".format(
                            key
                        )
                    )


                displacement = right_displacement_map[
                    key
                ]


                for component in range(
                    3
                ):

                    file_object.write(
                        "BLOCK-1.{}, {}, {}, {:.12e}\n".format(
                            node,
                            component + 1,
                            component + 1,
                            displacement[
                                component
                            ],
                        )
                    )


        # ====================================================
        # FIELD OUTPUT
        #
        # For adaptive enrichment we need:
        #
        # U  = displacement
        # RF = nodal reaction forces
        #
        # We do not request LE here because it is not needed
        # for building the enrichment dataset.
        # ====================================================

        file_object.write(
            "*Output, field\n"
        )


        file_object.write(
            "*Node Output\n"
        )


        file_object.write(
            "U, RF\n"
        )


        # Optional stress output.
        file_object.write(
            "*Element Output, directions=YES\n"
        )


        file_object.write(
            "S, E\n"
        )


        file_object.write(
            "*End Step\n"
        )


    print(
        "[{}/{}] Created: {}".format(
            case_number + 1,
            len(
                design_rows
            ),
            output_file,
        )
    )


print("")
print(
    "================================================"
)


print(
    "INPUT GENERATION COMPLETE"
)


print(
    "================================================"
)


print(
    "Segment:",
    args.segment
)


print(
    "Cases:",
    len(
        design_rows
    )
)


print(
    "Output directory:"
)


print(
    OUTPUT_DIR
)