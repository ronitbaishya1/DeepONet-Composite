import os
import csv
import argparse

import numpy as np


# ============================================================
# AUTOMATIC PROJECT DIRECTORY
#
# Expected structure:
#
# ODBS/
# ├── scripts/
# │   └── 17_generate_bulk_inp.py
# ├── interfaces/
# ├── designs/
# ├── left_inp/
# └── right_inp/
#
# BASE_DIR is automatically the parent of "scripts".
# No Windows username is hard-coded.
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
    "interfaces"
)

DESIGN_DIR = os.path.join(
    BASE_DIR,
    "designs"
)


print("")
print("========================================")
print("PROJECT PATHS")
print("========================================")
print("SCRIPT_DIR    =", SCRIPT_DIR)
print("BASE_DIR      =", BASE_DIR)
print("INTERFACE_DIR =", INTERFACE_DIR)
print("DESIGN_DIR    =", DESIGN_DIR)
print("")


# ============================================================
# CHECK REQUIRED DIRECTORIES
# ============================================================

if not os.path.isdir(
    BASE_DIR
):

    raise RuntimeError(
        "BASE_DIR does not exist: {}".format(
            BASE_DIR
        )
    )


if not os.path.isdir(
    INTERFACE_DIR
):

    raise RuntimeError(
        "Interface directory does not exist: {}".format(
            INTERFACE_DIR
        )
    )


if not os.path.isdir(
    DESIGN_DIR
):

    raise RuntimeError(
        "Design directory does not exist: {}".format(
            DESIGN_DIR
        )
    )


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
# SEGMENT SETTINGS
# ============================================================

if args.segment == "left":

    LEFT_NAME = "m6"
    RIGHT_NAME = "m2"

    DESIGN_FILE = os.path.join(
        DESIGN_DIR,
        "left_bulk_design_80.csv"
    )

    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        "left_inp"
    )

else:

    LEFT_NAME = "p2"
    RIGHT_NAME = "p6"

    DESIGN_FILE = os.path.join(
        DESIGN_DIR,
        "right_bulk_design_80.csv"
    )

    OUTPUT_DIR = os.path.join(
        BASE_DIR,
        "right_inp"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

LEFT_INTERFACE_FILE = os.path.join(
    INTERFACE_DIR,
    "interface_{}.npz".format(
        LEFT_NAME
    )
)

RIGHT_INTERFACE_FILE = os.path.join(
    INTERFACE_DIR,
    "interface_{}.npz".format(
        RIGHT_NAME
    )
)


if not os.path.isfile(
    LEFT_INTERFACE_FILE
):

    raise FileNotFoundError(
        "Missing interface file: {}".format(
            LEFT_INTERFACE_FILE
        )
    )


if not os.path.isfile(
    RIGHT_INTERFACE_FILE
):

    raise FileNotFoundError(
        "Missing interface file: {}".format(
            RIGHT_INTERFACE_FILE
        )
    )


if not os.path.isfile(
    DESIGN_FILE
):

    raise FileNotFoundError(
        "Missing design file: {}".format(
            DESIGN_FILE
        )
    )


# ============================================================
# FIXED MATERIAL PROPERTIES
#
# Units:
# E, G = MPa
#
# Material axes:
# 1 = X
# 2 = Y
# 3 = Z
# ============================================================

E3 = 12000.0

NU12 = 0.28
NU13 = 0.28
NU23 = 0.40

G13 = 4500.0
G23 = 3500.0


# ============================================================
# LOAD INTERFACE PCA DATA
# ============================================================

left_data = np.load(
    LEFT_INTERFACE_FILE
)

right_data = np.load(
    RIGHT_INTERFACE_FILE
)


left_coordinates = left_data[
    "coordinates"
]

right_coordinates = right_data[
    "coordinates"
]


left_mean = left_data[
    "mean"
]

right_mean = right_data[
    "mean"
]


left_basis = left_data[
    "basis"
]

right_basis = right_data[
    "basis"
]


n_left_modes = left_basis.shape[
    1
]

n_right_modes = right_basis.shape[
    1
]


x_left = float(
    left_data[
        "actual_x"
    ][
        0
    ]
)

x_right = float(
    right_data[
        "actual_x"
    ][
        0
    ]
)


# ============================================================
# VERIFY PCA DIMENSIONS
# ============================================================

if left_mean.shape[
    0
] != left_basis.shape[
    0
]:

    raise RuntimeError(
        "Left PCA mean/basis dimension mismatch."
    )


if right_mean.shape[
    0
] != right_basis.shape[
    0
]:

    raise RuntimeError(
        "Right PCA mean/basis dimension mismatch."
    )


if left_mean.shape[
    0
] != left_coordinates.shape[
    0
] * 3:

    raise RuntimeError(
        "Left PCA vector is inconsistent with interface node count."
    )


if right_mean.shape[
    0
] != right_coordinates.shape[
    0
] * 3:

    raise RuntimeError(
        "Right PCA vector is inconsistent with interface node count."
    )


# ============================================================
# VERIFY Y-Z CROSS-SECTION COMPATIBILITY
# ============================================================

left_yz = left_coordinates[
    :,
    1:3
]

right_yz = right_coordinates[
    :,
    1:3
]


if left_yz.shape != right_yz.shape:

    raise RuntimeError(
        "Left and right interfaces have different node counts."
    )


if not np.allclose(
    left_yz,
    right_yz,
    atol=1.0e-9
):

    raise RuntimeError(
        "Left and right interface Y-Z coordinates do not match."
    )


# ============================================================
# CROSS-SECTION COORDINATES
# ============================================================

y_values = np.unique(
    np.round(
        left_coordinates[
            :,
            1
        ],
        decimals=10
    )
)


z_values = np.unique(
    np.round(
        left_coordinates[
            :,
            2
        ],
        decimals=10
    )
)


y_values.sort()

z_values.sort()


number_interface_nodes = (
    len(
        y_values
    )
    *
    len(
        z_values
    )
)


if number_interface_nodes != left_coordinates.shape[
    0
]:

    raise RuntimeError(
        "Interface nodes do not form a complete structured Y-Z grid."
    )


# ============================================================
# X GRID
#
# Current NO block length:
#
# left:
# -6.0 -> -1.8 = 4.2 mm
#
# right:
# +1.8 -> +6.0 = 4.2 mm
#
# Using dx = 0.6 mm:
#
# 4.2 / 0.6 = 7 elements
# ============================================================

DX = 0.6


segment_length = (
    x_right
    - x_left
)


number_x_elements = int(
    round(
        segment_length
        /
        DX
    )
)


if number_x_elements < 1:

    raise RuntimeError(
        "Invalid segment length."
    )


actual_dx = (
    segment_length
    /
    number_x_elements
)


x_values = np.linspace(
    x_left,
    x_right,
    number_x_elements + 1
)


# ============================================================
# PRINT GEOMETRY INFORMATION
# ============================================================

print("")
print("========================================")
print("HYBRID BULK INPUT GENERATION")
print("========================================")
print("Segment            :", args.segment)
print("Left interface     :", LEFT_NAME)
print("Right interface    :", RIGHT_NAME)
print("x_left             :", x_left)
print("x_right            :", x_right)
print("Segment length     :", segment_length)
print("Requested dx       :", DX)
print("Actual dx          :", actual_dx)
print("X elements         :", number_x_elements)
print("X nodes            :", len(x_values))
print("Y nodes            :", len(y_values))
print("Z nodes            :", len(z_values))
print("Interface nodes    :", number_interface_nodes)
print("Left PCA modes     :", n_left_modes)
print("Right PCA modes    :", n_right_modes)
print("Design file        :", DESIGN_FILE)
print("Output directory   :", OUTPUT_DIR)
print("")


# ============================================================
# GENERATE NODES
#
# Ordering:
#
# x
#   -> y
#       -> z
#
# ============================================================

node_map = {}

nodes = []

node_label = 1


for ix, x in enumerate(
    x_values
):

    for iy, y in enumerate(
        y_values
    ):

        for iz, z in enumerate(
            z_values
        ):

            node_map[
                (
                    ix,
                    iy,
                    iz
                )
            ] = node_label


            nodes.append(
                (
                    node_label,
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                    float(
                        z
                    )
                )
            )


            node_label += 1


# ============================================================
# GENERATE C3D8R ELEMENTS
# ============================================================

elements = []

element_label = 1


for ix in range(
    len(
        x_values
    )
    - 1
):

    for iy in range(
        len(
            y_values
        )
        - 1
    ):

        for iz in range(
            len(
                z_values
            )
            - 1
        ):

            n1 = node_map[
                (
                    ix,
                    iy,
                    iz
                )
            ]

            n2 = node_map[
                (
                    ix + 1,
                    iy,
                    iz
                )
            ]

            n3 = node_map[
                (
                    ix + 1,
                    iy + 1,
                    iz
                )
            ]

            n4 = node_map[
                (
                    ix,
                    iy + 1,
                    iz
                )
            ]

            n5 = node_map[
                (
                    ix,
                    iy,
                    iz + 1
                )
            ]

            n6 = node_map[
                (
                    ix + 1,
                    iy,
                    iz + 1
                )
            ]

            n7 = node_map[
                (
                    ix + 1,
                    iy + 1,
                    iz + 1
                )
            ]

            n8 = node_map[
                (
                    ix,
                    iy + 1,
                    iz + 1
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
                    n8
                )
            )


            element_label += 1


# ============================================================
# LEFT / RIGHT FACE NODE LABELS
#
# IMPORTANT:
# PCA ordering was created using:
#
# np.lexsort((z, y))
#
# Therefore:
#
# y is primary
# z is secondary
#
# The loops below reproduce that order.
# ============================================================

left_face_labels = []

right_face_labels = []


for iy in range(
    len(
        y_values
    )
):

    for iz in range(
        len(
            z_values
        )
    ):

        left_face_labels.append(
            node_map[
                (
                    0,
                    iy,
                    iz
                )
            ]
        )


        right_face_labels.append(
            node_map[
                (
                    len(
                        x_values
                    )
                    - 1,
                    iy,
                    iz
                )
            ]
        )


if len(
    left_face_labels
) != left_coordinates.shape[
    0
]:

    raise RuntimeError(
        "Left face generated node count does not match PCA interface."
    )


if len(
    right_face_labels
) != right_coordinates.shape[
    0
]:

    raise RuntimeError(
        "Right face generated node count does not match PCA interface."
    )


# ============================================================
# READ DESIGN CSV
# ============================================================

with open(
    DESIGN_FILE,
    "r",
    newline=""
) as file_object:

    reader = csv.DictReader(
        file_object
    )

    design_rows = list(
        reader
    )


if len(
    design_rows
) == 0:

    raise RuntimeError(
        "Design CSV contains no cases."
    )


print(
    "Design cases        :",
    len(
        design_rows
    )
)

print(
    "Total nodes/block   :",
    len(
        nodes
    )
)

print(
    "Total elements/block:",
    len(
        elements
    )
)

print("")


# ============================================================
# HELPERS
# ============================================================

def reconstruct_face(
    mean,
    basis,
    coefficients
):

    flattened = (
        mean
        +
        basis.dot(
            coefficients
        )
    )


    return flattened.reshape(
        -1,
        3
    )


def write_label_lines(
    file_object,
    labels
):

    for start in range(
        0,
        len(
            labels
        ),
        16
    ):

        subset = labels[
            start:
            start + 16
        ]


        line = ", ".join(
            [
                str(
                    value
                )
                for value in subset
            ]
        )


        file_object.write(
            line
            +
            "\n"
        )


# ============================================================
# GENERATE ALL INPUT FILES
# ============================================================

for row_index, row in enumerate(
    design_rows
):

    case_id = row[
        "CaseID"
    ]


    E1 = float(
        row[
            "E1_MPa"
        ]
    )


    E2 = float(
        row[
            "E2_MPa"
        ]
    )


    G12 = float(
        row[
            "G12_MPa"
        ]
    )


    # ========================================================
    # LEFT PCA COEFFICIENTS
    # ========================================================

    c_left = np.array(
        [
            float(
                row[
                    "cL_{:02d}".format(
                        mode_index + 1
                    )
                ]
            )

            for mode_index in range(
                n_left_modes
            )
        ],
        dtype=np.float64
    )


    # ========================================================
    # RIGHT PCA COEFFICIENTS
    # ========================================================

    c_right = np.array(
        [
            float(
                row[
                    "cR_{:02d}".format(
                        mode_index + 1
                    )
                ]
            )

            for mode_index in range(
                n_right_modes
            )
        ],
        dtype=np.float64
    )


    # ========================================================
    # RECONSTRUCT INTERFACE DISPLACEMENTS
    # ========================================================

    displacement_left = reconstruct_face(
        left_mean,
        left_basis,
        c_left
    )


    displacement_right = reconstruct_face(
        right_mean,
        right_basis,
        c_right
    )


    if displacement_left.shape != (
        len(
            left_face_labels
        ),
        3
    ):

        raise RuntimeError(
            "Left displacement reconstruction shape error."
        )


    if displacement_right.shape != (
        len(
            right_face_labels
        ),
        3
    ):

        raise RuntimeError(
            "Right displacement reconstruction shape error."
        )


    # ========================================================
    # OUTPUT FILE
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        case_id
        +
        ".inp"
    )


    with open(
        output_file,
        "w"
    ) as inp:

        # ====================================================
        # HEADING
        # ====================================================

        inp.write(
            "*HEADING\n"
        )

        inp.write(
            "Hybrid bulk operator case {}\n".format(
                case_id
            )
        )


        # ====================================================
        # PART
        # ====================================================

        inp.write(
            "*PART, NAME=BLOCK\n"
        )


        # ====================================================
        # NODES
        # ====================================================

        inp.write(
            "*NODE\n"
        )


        for (
            label,
            x,
            y,
            z
        ) in nodes:

            inp.write(
                "{}, {:.10f}, {:.10f}, {:.10f}\n".format(
                    label,
                    x,
                    y,
                    z
                )
            )


        # ====================================================
        # ELEMENTS
        # ====================================================

        inp.write(
            "*ELEMENT, TYPE=C3D8R, ELSET=EALL\n"
        )


        for element in elements:

            inp.write(
                ", ".join(
                    [
                        str(
                            value
                        )
                        for value in element
                    ]
                )
                +
                "\n"
            )


        # ====================================================
        # NODE SETS
        # ====================================================

        inp.write(
            "*NSET, NSET=LEFT_FACE\n"
        )


        write_label_lines(
            inp,
            left_face_labels
        )


        inp.write(
            "*NSET, NSET=RIGHT_FACE\n"
        )


        write_label_lines(
            inp,
            right_face_labels
        )


        # ====================================================
        # MATERIAL ORIENTATION
        #
        # Axis 1 = global X
        # Axis 2 = global Y
        # Axis 3 = global Z
        # ====================================================

        inp.write(
            "*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR\n"
        )

        inp.write(
            "1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n"
        )


        # ====================================================
        # SECTION
        # ====================================================

        inp.write(
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT, ORIENTATION=MAT_ORI\n"
        )

        inp.write(
            ",\n"
        )


        inp.write(
            "*END PART\n"
        )


        # ====================================================
        # ASSEMBLY
        # ====================================================

        inp.write(
            "*ASSEMBLY, NAME=ASSEMBLY\n"
        )


        inp.write(
            "*INSTANCE, NAME=BLOCK-1, PART=BLOCK\n"
        )


        inp.write(
            "*END INSTANCE\n"
        )


        inp.write(
            "*END ASSEMBLY\n"
        )


        # ====================================================
        # MATERIAL
        # ====================================================

        inp.write(
            "*MATERIAL, NAME=MAT\n"
        )


        inp.write(
            "*ELASTIC, TYPE=ENGINEERING CONSTANTS\n"
        )


        inp.write(
            "{:.10f}, {:.10f}, {:.10f}, "
            "{:.10f}, {:.10f}, {:.10f}, "
            "{:.10f}, {:.10f}\n".format(
                E1,
                E2,
                E3,
                NU12,
                NU13,
                NU23,
                G12,
                G13
            )
        )


        inp.write(
            "{:.10f}\n".format(
                G23
            )
        )


        # ====================================================
        # LOAD STEP
        # ====================================================

        inp.write(
            "*STEP, NAME=Loading, NLGEOM=YES\n"
        )


        inp.write(
            "*STATIC\n"
        )


        inp.write(
            "0.05, 1.0, 1.0E-08, 0.1\n"
        )


        # ====================================================
        # PRESCRIBED LEFT FACE
        # ====================================================

        inp.write(
            "*BOUNDARY\n"
        )


        for local_index, label in enumerate(
            left_face_labels
        ):

            values = displacement_left[
                local_index
            ]


            inp.write(
                "BLOCK-1.{}, 1, 1, {:.12e}\n".format(
                    label,
                    values[
                        0
                    ]
                )
            )


            inp.write(
                "BLOCK-1.{}, 2, 2, {:.12e}\n".format(
                    label,
                    values[
                        1
                    ]
                )
            )


            inp.write(
                "BLOCK-1.{}, 3, 3, {:.12e}\n".format(
                    label,
                    values[
                        2
                    ]
                )
            )


        # ====================================================
        # PRESCRIBED RIGHT FACE
        # ====================================================

        for local_index, label in enumerate(
            right_face_labels
        ):

            values = displacement_right[
                local_index
            ]


            inp.write(
                "BLOCK-1.{}, 1, 1, {:.12e}\n".format(
                    label,
                    values[
                        0
                    ]
                )
            )


            inp.write(
                "BLOCK-1.{}, 2, 2, {:.12e}\n".format(
                    label,
                    values[
                        1
                    ]
                )
            )


            inp.write(
                "BLOCK-1.{}, 3, 3, {:.12e}\n".format(
                    label,
                    values[
                        2
                    ]
                )
            )


        # ====================================================
        # FIELD OUTPUT
        # ====================================================

        inp.write(
            "*OUTPUT, FIELD, FREQUENCY=1\n"
        )


        inp.write(
            "*NODE OUTPUT\n"
        )


        inp.write(
            "U, RF\n"
        )


        inp.write(
            "*ELEMENT OUTPUT, DIRECTIONS=YES\n"
        )


        inp.write(
            "S, LE\n"
        )


        inp.write(
            "*END STEP\n"
        )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        row_index == 0
        or (
            row_index + 1
        ) % 10 == 0
        or (
            row_index + 1
        ) == len(
            design_rows
        )
    ):

        print(
            "Generated {}/{} : {}".format(
                row_index + 1,
                len(
                    design_rows
                ),
                case_id
            )
        )


# ============================================================
# COMPLETE
# ============================================================

print("")
print("========================================")
print("STEP 17 COMPLETE")
print("========================================")
print("Segment :", args.segment)
print("Cases   :", len(design_rows))
print("Folder  :", OUTPUT_DIR)
print("")
