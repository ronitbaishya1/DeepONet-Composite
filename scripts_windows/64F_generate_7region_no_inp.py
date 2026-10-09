from __future__ import print_function

import os
import csv
import argparse

import numpy as np


# ============================================================
# PROJECT ROOT
#
# Expected Windows structure:
#
# ODBS/
# ├── scripts/
# │   └── 64F_generate_7region_no_inp.py
# │
# ├── candidate_interfaces_7region/
# │   ├── interface_left_outer.npz
# │   ├── interface_left_inner.npz
# │   ├── interface_center_left.npz
# │   ├── interface_center_right.npz
# │   ├── interface_right_inner.npz
# │   └── interface_right_outer.npz
# │
# ├── candidate_designs_7region/
# │   ├── outer_left_candidate_design_80.csv
# │   ├── inner_left_candidate_design_80.csv
# │   ├── inner_right_candidate_design_80.csv
# │   └── outer_right_candidate_design_80.csv
# │
# └── candidate_inp_7region/
#     ├── outer_left/
#     ├── inner_left/
#     ├── inner_right/
#     └── outer_right/
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
    "candidate_interfaces_7region"
)


DESIGN_DIR = os.path.join(
    BASE_DIR,
    "candidate_designs_7region"
)


OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "candidate_inp_7region"
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "outer_left",
        "inner_left",
        "inner_right",
        "outer_right",
    ],
    required=True,
)


parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help=(
        "0 = generate every case. "
        "Use --limit 1 for smoke test."
    ),
)


args = parser.parse_args()


# ============================================================
# FIXED MATERIAL PROPERTIES
# ============================================================

E3 = 12000.0

NU12 = 0.28
NU13 = 0.28
NU23 = 0.40

G13 = 4500.0
G23 = 3500.0


DX = 0.6


# ============================================================
# FILES
# ============================================================

DESIGN_FILE = os.path.join(
    DESIGN_DIR,
    "{}_candidate_design_80.csv".format(
        args.segment
    )
)


OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    args.segment
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


if not os.path.isfile(
    DESIGN_FILE
):

    raise FileNotFoundError(
        DESIGN_FILE
    )


# ============================================================
# READ DESIGN
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
        "Design CSV is empty."
    )


if args.limit > 0:

    design_rows = design_rows[
        :args.limit
    ]


first_row = design_rows[
    0
]


x_min = float(
    first_row[
        "XMin"
    ]
)


x_max = float(
    first_row[
        "XMax"
    ]
)


left_boundary_type = first_row[
    "LeftBoundaryType"
].strip().lower()


right_boundary_type = first_row[
    "RightBoundaryType"
].strip().lower()


left_interface_name = first_row[
    "LeftInterface"
].strip()


right_interface_name = first_row[
    "RightInterface"
].strip()


def normalize_interface_name(
    value
):

    if value.upper() in [
        "NONE",
        "NULL",
        "",
    ]:

        return None

    return value


left_interface_name = normalize_interface_name(
    left_interface_name
)


right_interface_name = normalize_interface_name(
    right_interface_name
)


# ============================================================
# LOAD INTERFACE
# ============================================================

def load_interface(
    interface_name
):

    if interface_name is None:

        return None


    filename = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        )
    )


    if not os.path.isfile(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


    return np.load(
        filename
    )


left_interface = load_interface(
    left_interface_name
)


right_interface = load_interface(
    right_interface_name
)


if (
    left_interface is None
    and
    right_interface is None
):

    raise RuntimeError(
        "At least one interface is required."
    )


# ============================================================
# CROSS SECTION SOURCE
# ============================================================

if left_interface is not None:

    reference_interface = (
        left_interface
    )

else:

    reference_interface = (
        right_interface
    )


reference_coordinates = reference_interface[
    "coordinates"
].astype(
    np.float64
)


y_values = np.unique(
    np.round(
        reference_coordinates[
            :,
            1
        ],
        decimals=10
    )
)


z_values = np.unique(
    np.round(
        reference_coordinates[
            :,
            2
        ],
        decimals=10
    )
)


y_values.sort()
z_values.sort()


number_face_nodes = (
    len(
        y_values
    )
    *
    len(
        z_values
    )
)


if number_face_nodes != (
    reference_coordinates.shape[
        0
    ]
):

    raise RuntimeError(
        "Cross section is not a complete Y-Z grid."
    )


# ============================================================
# VERIFY INTERFACE GRID
# ============================================================

for interface_name, interface in [
    (
        left_interface_name,
        left_interface,
    ),
    (
        right_interface_name,
        right_interface,
    ),
]:

    if interface is None:

        continue


    coordinates = interface[
        "coordinates"
    ].astype(
        np.float64
    )


    yz = coordinates[
        :,
        1:3
    ]


    reference_yz = (
        reference_coordinates[
            :,
            1:3
        ]
    )


    if yz.shape != (
        reference_yz.shape
    ):

        raise RuntimeError(
            "Interface {} has incompatible grid."
            .format(
                interface_name
            )
        )


    if not np.allclose(
        yz,
        reference_yz,
        atol=1.0e-8,
    ):

        raise RuntimeError(
            "Interface {} Y-Z coordinates differ."
            .format(
                interface_name
            )
        )


# ============================================================
# VERIFY INTERFACE X LOCATIONS
# ============================================================

if left_interface is not None:

    interface_x = float(
        left_interface[
            "actual_x"
        ][0]
    )


    if not np.isclose(
        interface_x,
        x_min,
        atol=1.0e-7,
    ):

        raise RuntimeError(
            (
                "Left interface {} is at x={} "
                "but segment starts at {}."
            ).format(
                left_interface_name,
                interface_x,
                x_min,
            )
        )


if right_interface is not None:

    interface_x = float(
        right_interface[
            "actual_x"
        ][0]
    )


    if not np.isclose(
        interface_x,
        x_max,
        atol=1.0e-7,
    ):

        raise RuntimeError(
            (
                "Right interface {} is at x={} "
                "but segment ends at {}."
            ).format(
                right_interface_name,
                interface_x,
                x_max,
            )
        )


# ============================================================
# X GRID
# ============================================================

segment_length = (
    x_max
    -
    x_min
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


if not np.isclose(
    actual_dx,
    DX,
    atol=1.0e-8,
):

    raise RuntimeError(
        (
            "Segment cannot use dx=0.6 exactly. "
            "Length={} actual_dx={}"
        ).format(
            segment_length,
            actual_dx,
        )
    )


x_values = np.linspace(
    x_min,
    x_max,
    number_x_elements + 1
)


# ============================================================
# GENERATE NODES
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
                    iz,
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
                    ),
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


# ============================================================
# LEFT / RIGHT FACE LABELS
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
                    iz,
                )
            ]
        )


        right_face_labels.append(
            node_map[
                (
                    len(
                        x_values
                    )
                    -
                    1,
                    iy,
                    iz,
                )
            ]
        )


# ============================================================
# HELPERS
# ============================================================

def write_label_lines(
    file_object,
    labels
):

    for start in range(
        0,
        len(
            labels
        ),
        16,
    ):

        subset = labels[
            start:
            start + 16
        ]


        file_object.write(
            ", ".join(
                [
                    str(
                        label
                    )

                    for label
                    in subset
                ]
            )
            +
            "\n"
        )


def reconstruct_interface(
    interface,
    coefficients
):

    mean = interface[
        "mean"
    ].astype(
        np.float64
    )


    basis = interface[
        "basis"
    ].astype(
        np.float64
    )


    result = (
        mean
        +
        basis.dot(
            coefficients
        )
    )


    return result.reshape(
        -1,
        3
    )


# ============================================================
# MODE COUNTS
# ============================================================

if left_interface is None:

    n_left_modes = 0

else:

    n_left_modes = (
        left_interface[
            "basis"
        ].shape[1]
    )


if right_interface is None:

    n_right_modes = 0

else:

    n_right_modes = (
        right_interface[
            "basis"
        ].shape[1]
    )


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64F — GENERATE 7-REGION NO INPUTS"
)

print(
    "=========================================="
)

print(
    "Segment:",
    args.segment
)

print(
    "x_min / x_max:",
    x_min,
    x_max,
)

print(
    "Length:",
    segment_length
)

print(
    "X elements:",
    number_x_elements
)

print(
    "Nodes:",
    len(
        nodes
    )
)

print(
    "Elements:",
    len(
        elements
    )
)

print(
    "Left BC:",
    left_boundary_type,
    left_interface_name,
)

print(
    "Right BC:",
    right_boundary_type,
    right_interface_name,
)

print(
    "Left modes:",
    n_left_modes
)

print(
    "Right modes:",
    n_right_modes
)

print(
    "Cases being generated:",
    len(
        design_rows
    )
)

print("")


# ============================================================
# GENERATE CASES
# ============================================================

generation_rows = []


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
    # LEFT INTERFACE DISPLACEMENT
    # ========================================================

    displacement_left = None


    if left_interface is not None:

        coefficients = np.asarray(
            [
                float(
                    row[
                        "cL_{:02d}".format(
                            mode_index + 1
                        )
                    ]
                )

                for mode_index
                in range(
                    n_left_modes
                )
            ],
            dtype=np.float64,
        )


        displacement_left = (
            reconstruct_interface(
                left_interface,
                coefficients,
            )
        )


    # ========================================================
    # RIGHT INTERFACE DISPLACEMENT
    # ========================================================

    displacement_right = None


    if right_interface is not None:

        coefficients = np.asarray(
            [
                float(
                    row[
                        "cR_{:02d}".format(
                            mode_index + 1
                        )
                    ]
                )

                for mode_index
                in range(
                    n_right_modes
                )
            ],
            dtype=np.float64,
        )


        displacement_right = (
            reconstruct_interface(
                right_interface,
                coefficients,
            )
        )


    # ========================================================
    # OUTPUT INPUT FILE
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

        inp.write(
            "*HEADING\n"
        )

        inp.write(
            "Seven-region NO block: {}\n".format(
                case_id
            )
        )

        inp.write(
            "** Segment = {}\n".format(
                args.segment
            )
        )

        inp.write(
            "** X range = {}, {}\n".format(
                x_min,
                x_max,
            )
        )

        inp.write(
            "** Left boundary = {}\n".format(
                left_boundary_type
            )
        )

        inp.write(
            "** Right boundary = {}\n".format(
                right_boundary_type
            )
        )


        # ====================================================
        # PART
        # ====================================================

        inp.write(
            "*PART, NAME=BLOCK\n"
        )


        inp.write(
            "*NODE\n"
        )


        for (
            label,
            x,
            y,
            z,
        ) in nodes:

            inp.write(
                "{}, {:.10f}, {:.10f}, {:.10f}\n"
                .format(
                    label,
                    x,
                    y,
                    z,
                )
            )


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

                        for value
                        in element
                    ]
                )
                +
                "\n"
            )


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
        # MATERIAL AXES
        # 1 = X
        # 2 = Y
        # 3 = Z
        # ====================================================

        inp.write(
            "*ORIENTATION, NAME=MAT_ORI, SYSTEM=RECTANGULAR\n"
        )

        inp.write(
            "1.0, 0.0, 0.0, "
            "0.0, 1.0, 0.0\n"
        )


        inp.write(
            "*SOLID SECTION, "
            "ELSET=EALL, "
            "MATERIAL=MAT, "
            "ORIENTATION=MAT_ORI\n"
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
            "{:.10f}, {:.10f}\n"
            .format(
                E1,
                E2,
                E3,
                NU12,
                NU13,
                NU23,
                G12,
                G13,
            )
        )


        inp.write(
            "{:.10f}\n".format(
                G23
            )
        )


        # ====================================================
        # STEP
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
        # DIRICHLET INTERFACES ONLY
        #
        # IMPORTANT:
        #
        # If a side is "free", NO boundary condition is
        # written. Abaqus therefore gives that side its
        # natural zero-traction boundary condition.
        # ====================================================

        if (
            displacement_left is not None
            or
            displacement_right is not None
        ):

            inp.write(
                "*BOUNDARY\n"
            )


        if displacement_left is not None:

            for local_index, label in enumerate(
                left_face_labels
            ):

                values = displacement_left[
                    local_index
                ]


                for dof in range(
                    3
                ):

                    inp.write(
                        "BLOCK-1.{}, {}, {}, {:.12e}\n"
                        .format(
                            label,
                            dof + 1,
                            dof + 1,
                            values[
                                dof
                            ],
                        )
                    )


        if displacement_right is not None:

            for local_index, label in enumerate(
                right_face_labels
            ):

                values = displacement_right[
                    local_index
                ]


                for dof in range(
                    3
                ):

                    inp.write(
                        "BLOCK-1.{}, {}, {}, {:.12e}\n"
                        .format(
                            label,
                            dof + 1,
                            dof + 1,
                            values[
                                dof
                            ],
                        )
                    )


        # ====================================================
        # OUTPUTS
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


    generation_rows.append(
        {
            "CaseID":
                case_id,

            "Segment":
                args.segment,

            "XMin":
                x_min,

            "XMax":
                x_max,

            "Nodes":
                len(
                    nodes
                ),

            "Elements":
                len(
                    elements
                ),

            "LeftBoundary":
                left_boundary_type,

            "RightBoundary":
                right_boundary_type,

            "LeftInterface":
                (
                    left_interface_name
                    if left_interface_name
                    is not None
                    else
                    "NONE"
                ),

            "RightInterface":
                (
                    right_interface_name
                    if right_interface_name
                    is not None
                    else
                    "NONE"
                ),
        }
    )


    print(
        "[{}/{}] Generated {}"
        .format(
            row_index + 1,
            len(
                design_rows
            ),
            case_id,
        )
    )


# ============================================================
# SAVE GENERATION SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "generation_summary.csv"
)


with open(
    summary_file,
    "w",
    newline=""
) as file_object:

    fieldnames = [
        "CaseID",
        "Segment",
        "XMin",
        "XMax",
        "Nodes",
        "Elements",
        "LeftBoundary",
        "RightBoundary",
        "LeftInterface",
        "RightInterface",
    ]


    writer = csv.DictWriter(
        file_object,
        fieldnames=
            fieldnames,
    )


    writer.writeheader()


    for row in generation_rows:

        writer.writerow(
            row
        )


print("")
print(
    "=========================================="
)

print(
    "STEP 64F COMPLETE"
)

print(
    "=========================================="
)

print(
    "Segment:",
    args.segment
)

print(
    "Generated:",
    len(
        generation_rows
    )
)

print(
    "Output:",
    OUTPUT_DIR
)

print("")