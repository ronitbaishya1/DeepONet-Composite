from __future__ import print_function

import os
import json

import numpy as np


# ============================================================
# AUTOMATIC PROJECT ROOT
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(
        __file__
    )
)


BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

BASELINE_FILE = os.path.join(
    BASE_DIR,
    "baseline",
    "3Point.inp",
)


INTERFACE_DIR = os.path.join(
    BASE_DIR,
    "interfaces_7region",
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "patch_templates_7region",
)


if not os.path.isdir(
    OUTPUT_DIR
):

    os.makedirs(
        OUTPUT_DIR
    )


if not os.path.isfile(
    BASELINE_FILE
):

    raise FileNotFoundError(
        BASELINE_FILE
    )


# ============================================================
# NEW SEVEN-REGION FE PATCH CONFIGURATION
#
# Full partition:
#
# NO_OL [-12,-9.6]
# FE_L  [-9.6,-6.6]
# NO_L  [-6.6,-1.8]
# FE_C  [-1.8,+1.8]
# NO_R  [+1.8,+6.6]
# FE_R  [+6.6,+9.6]
# NO_OR [+9.6,+12]
#
#
# Contact numbering in original Abaqus deck:
#
# CP-1 = RIGHT SUPPORT
# CP-2 = LOADING NOSE
# CP-3 = LEFT SUPPORT
# ============================================================

PATCHES = {

    # ========================================================
    # LEFT SUPPORT PATCH
    # ========================================================

    "left": {

        "x_min":
            -9.6,

        "x_max":
            -6.6,

        "point_instance":
            "Point-3",

        "contact_number":
            3,

        "contact_side":
            "S6",

        "contact_set_name":
            "_CP-3-Composite-1_S6",

        "composite_surface_name":
            "CP-3-Composite-1",

        "point_surface_name":
            "CP-3-Point-3",

        "smoothing_name":
            "CP-3-Composite-1-Point-3",

        "picked_set":
            "_PickedSet19",

        "initial_bc_type":
            "support",

        "interfaces":
            [
                "left_outer",
                "left_inner",
            ],
    },


    # ========================================================
    # CENTER NOSE PATCH
    # ========================================================

    "center": {

        "x_min":
            -1.8,

        "x_max":
            1.8,

        "point_instance":
            "Point-2",

        "contact_number":
            2,

        "contact_side":
            "S4",

        "contact_set_name":
            "_CP-2-Composite-1_S4",

        "composite_surface_name":
            "CP-2-Composite-1",

        "point_surface_name":
            "CP-2-Point-2",

        "smoothing_name":
            "CP-2-Composite-1-Point-2",

        "picked_set":
            "_PickedSet23",

        "initial_bc_type":
            "nose",

        "interfaces":
            [
                "center_left",
                "center_right",
            ],
    },


    # ========================================================
    # RIGHT SUPPORT PATCH
    # ========================================================

    "right": {

        "x_min":
            6.6,

        "x_max":
            9.6,

        "point_instance":
            "Point-1",

        "contact_number":
            1,

        "contact_side":
            "S6",

        "contact_set_name":
            "_CP-1-Composite-1_S6",

        "composite_surface_name":
            "CP-1-Composite-1",

        "point_surface_name":
            "CP-1-Point-1",

        "smoothing_name":
            "CP-1-Composite-1-Point-1",

        "picked_set":
            "_PickedSet21",

        "initial_bc_type":
            "support",

        "interfaces":
            [
                "right_inner",
                "right_outer",
            ],
    },
}


# ============================================================
# READ ORIGINAL FULL INPUT FILE
# ============================================================

with open(
    BASELINE_FILE,
    "r"
) as file_object:

    lines = file_object.read().splitlines()


# ============================================================
# BASIC HELPERS
# ============================================================

def lower(
    text
):

    return text.strip().lower()


# ============================================================
# FIND PART
# ============================================================

def find_part_range(
    all_lines,
    part_name,
):

    start = None
    end = None


    target = "name={}".format(
        part_name.lower()
    )


    for index, line in enumerate(
        all_lines
    ):

        if (
            lower(
                line
            ).startswith(
                "*part"
            )
            and
            target
            in
            lower(
                line
            )
        ):

            start = index

            break


    if start is None:

        raise RuntimeError(
            "Part not found: {}".format(
                part_name
            )
        )


    for index in range(
        start + 1,
        len(
            all_lines
        ),
    ):

        if lower(
            all_lines[
                index
            ]
        ).startswith(
            "*end part"
        ):

            end = index

            break


    if end is None:

        raise RuntimeError(
            "End Part not found for {}".format(
                part_name
            )
        )


    return (
        start,
        end,
    )


# ============================================================
# PARSE COMPOSITE NODES / ELEMENTS
# ============================================================

def parse_composite_mesh(
    part_lines,
):

    nodes = {}

    elements = {}

    mode = None


    for line in part_lines:

        stripped = line.strip()


        if lower(
            stripped
        ).startswith(
            "*node"
        ):

            mode = "node"

            continue


        if lower(
            stripped
        ).startswith(
            "*element"
        ):

            mode = "element"

            continue


        if stripped.startswith(
            "*"
        ):

            mode = None

            continue


        if not stripped:

            continue


        if stripped.startswith(
            "**"
        ):

            continue


        # ====================================================
        # NODE
        # ====================================================

        if mode == "node":

            pieces = [

                item.strip()

                for item
                in stripped.split(
                    ","
                )

                if item.strip()
            ]


            if len(
                pieces
            ) >= 4:

                label = int(
                    pieces[
                        0
                    ]
                )


                nodes[
                    label
                ] = (

                    float(
                        pieces[
                            1
                        ]
                    ),

                    float(
                        pieces[
                            2
                        ]
                    ),

                    float(
                        pieces[
                            3
                        ]
                    ),
                )


        # ====================================================
        # ELEMENT
        # ====================================================

        elif mode == "element":

            pieces = [

                item.strip()

                for item
                in stripped.split(
                    ","
                )

                if item.strip()
            ]


            if len(
                pieces
            ) >= 9:

                label = int(
                    pieces[
                        0
                    ]
                )


                connectivity = [

                    int(
                        value
                    )

                    for value
                    in pieces[
                        1:
                    ]
                ]


                elements[
                    label
                ] = connectivity


    return (
        nodes,
        elements,
    )


# ============================================================
# KEYWORD BLOCKS
# ============================================================

def keyword_blocks(
    source_lines,
):

    blocks = []

    current = None


    for line in source_lines:

        stripped = line.strip()


        if stripped.startswith(
            "**"
        ):

            continue


        if (
            stripped.startswith(
                "*"
            )
            and
            not stripped.startswith(
                "**"
            )
        ):

            if current is not None:

                blocks.append(
                    current
                )


            current = [
                line
            ]


        else:

            if current is not None:

                current.append(
                    line
                )


    if current is not None:

        blocks.append(
            current
        )


    return blocks


# ============================================================
# FIND ASSEMBLY
# ============================================================

def find_assembly_range(
    all_lines,
):

    start = None
    end = None


    for index, line in enumerate(
        all_lines
    ):

        if lower(
            line
        ).startswith(
            "*assembly"
        ):

            start = index

            break


    if start is None:

        raise RuntimeError(
            "Assembly not found."
        )


    for index in range(
        start + 1,
        len(
            all_lines
        ),
    ):

        if lower(
            all_lines[
                index
            ]
        ).startswith(
            "*end assembly"
        ):

            end = index

            break


    if end is None:

        raise RuntimeError(
            "End Assembly not found."
        )


    return (
        start,
        end,
    )


# ============================================================
# FIND INSTANCE TRANSLATION
# ============================================================

def find_instance_translation(
    assembly_lines,
    instance_name,
):

    target = "name={}".format(
        instance_name.lower()
    )


    for index, line in enumerate(
        assembly_lines
    ):

        if (
            lower(
                line
            ).startswith(
                "*instance"
            )
            and
            target
            in
            lower(
                line
            )
        ):

            next_index = index + 1


            while next_index < len(
                assembly_lines
            ):

                candidate = assembly_lines[
                    next_index
                ].strip()


                if candidate.startswith(
                    "**"
                ):

                    next_index += 1

                    continue


                if candidate.startswith(
                    "*"
                ):

                    return None


                if candidate:

                    pieces = [

                        item.strip()

                        for item
                        in candidate.split(
                            ","
                        )

                        if item.strip()
                    ]


                    if len(
                        pieces
                    ) >= 3:

                        return (

                            float(
                                pieces[
                                    0
                                ]
                            ),

                            float(
                                pieces[
                                    1
                                ]
                            ),

                            float(
                                pieces[
                                    2
                                ]
                            ),
                        )


                next_index += 1


    raise RuntimeError(
        "Instance translation not found: {}".format(
            instance_name
        )
    )


# ============================================================
# FIND KEYWORD BLOCK
# ============================================================

def find_keyword_block(
    all_lines,
    header_prefix,
):

    blocks = keyword_blocks(
        all_lines
    )


    for block in blocks:

        if lower(
            block[
                0
            ]
        ).startswith(
            header_prefix.lower()
        ):

            return block


    return None


# ============================================================
# GENERATE SET RANGE
# ============================================================

def expand_generate(
    start,
    end,
    increment,
):

    return list(
        range(
            start,
            end + 1,
            increment,
        )
    )


# ============================================================
# ORIGINAL COMPOSITE CONTACT ELEMENTS
# ============================================================

def get_original_composite_contact_elements(
    assembly_lines,
    set_name,
):

    blocks = keyword_blocks(
        assembly_lines
    )


    target = "elset={}".format(
        set_name.lower()
    )


    for block in blocks:

        header = lower(
            block[
                0
            ]
        )


        if (
            header.startswith(
                "*elset"
            )
            and
            target
            in
            header
        ):

            values = []


            for line in block[
                1:
            ]:

                pieces = [

                    piece.strip()

                    for piece
                    in line.split(
                        ","
                    )

                    if piece.strip()
                ]


                values.extend(
                    [
                        int(
                            value
                        )

                        for value
                        in pieces
                    ]
                )


            if "generate" in header:

                if len(
                    values
                ) != 3:

                    raise RuntimeError(
                        (
                            "Unexpected generate set format: {}"
                        ).format(
                            set_name
                        )
                    )


                return expand_generate(

                    values[
                        0
                    ],

                    values[
                        1
                    ],

                    values[
                        2
                    ],
                )


            return values


    raise RuntimeError(
        "Composite contact set not found: {}".format(
            set_name
        )
    )


# ============================================================
# WRITE INTEGER SET
# ============================================================

def write_integer_set(
    file_object,
    values,
    values_per_line=16,
):

    values = list(
        values
    )


    for start in range(
        0,
        len(
            values
        ),
        values_per_line,
    ):

        subset = values[
            start:
            start
            +
            values_per_line
        ]


        file_object.write(
            ", ".join(
                [
                    str(
                        value
                    )

                    for value
                    in subset
                ]
            )
            +
            "\n"
        )


# ============================================================
# EXTRACT ORIGINAL PARTS
# ============================================================

composite_start, composite_end = find_part_range(
    lines,
    "Composite",
)


point_start, point_end = find_part_range(
    lines,
    "Point",
)


composite_part_lines = lines[
    composite_start:
    composite_end + 1
]


point_part_lines = lines[
    point_start:
    point_end + 1
]


nodes, elements = parse_composite_mesh(
    composite_part_lines
)


print("")
print(
    "Original composite nodes:",
    len(
        nodes
    ),
)


print(
    "Original composite elements:",
    len(
        elements
    ),
)


# ============================================================
# SHOW AVAILABLE X PLANES
#
# Useful diagnostic:
# verifies that all six selected interfaces are actually
# present in the original mesh.
# ============================================================

available_x = sorted(
    set(
        [
            round(
                float(
                    xyz[
                        0
                    ]
                ),
                7,
            )

            for xyz
            in nodes.values()
        ]
    )
)


print("")
print(
    "Available X planes:"
)

print(
    available_x
)


# ============================================================
# ASSEMBLY
# ============================================================

assembly_start, assembly_end = find_assembly_range(
    lines
)


assembly_lines = lines[
    assembly_start:
    assembly_end + 1
]


assembly_blocks = keyword_blocks(
    assembly_lines
)


# ============================================================
# LOAD SIX NEW INTERFACE DATASETS
# ============================================================

interface_data = {}


INTERFACE_NAMES = [

    "left_outer",

    "left_inner",

    "center_left",

    "center_right",

    "right_inner",

    "right_outer",
]


for interface_name in INTERFACE_NAMES:

    interface_file = os.path.join(
        INTERFACE_DIR,
        "interface_{}.npz".format(
            interface_name
        ),
    )


    if not os.path.isfile(
        interface_file
    ):

        raise FileNotFoundError(
            interface_file
        )


    interface_data[
        interface_name
    ] = np.load(
        interface_file
    )


# ============================================================
# PRE-CHECK ALL INTERFACE X LOCATIONS
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "CHECKING SIX INTERFACE PLANES"
)

print(
    "=========================================="
)


for interface_name in INTERFACE_NAMES:

    interface = interface_data[
        interface_name
    ]


    target_x = float(
        interface[
            "actual_x"
        ][0]
    )


    closest_x = min(

        available_x,

        key=lambda value:
            abs(
                value
                -
                target_x
            ),
    )


    difference = abs(
        closest_x
        -
        target_x
    )


    print(
        "{:<15s} target={: .8f} closest={: .8f} diff={:.3e}"
        .format(
            interface_name,
            target_x,
            closest_x,
            difference,
        )
    )


    if difference > 1.0e-5:

        raise RuntimeError(
            (
                "Interface {} is not present in the "
                "baseline mesh."
            ).format(
                interface_name
            )
        )


# ============================================================
# METADATA
# ============================================================

metadata = {

    "baseline_file":
        BASELINE_FILE,

    "patches":
        {},
}


# ============================================================
# BUILD EACH PATCH
# ============================================================

for patch_name, config in PATCHES.items():

    print("")
    print(
        "=========================================="
    )

    print(
        "BUILDING PATCH:",
        patch_name.upper()
    )

    print(
        "=========================================="
    )


    x_min = config[
        "x_min"
    ]


    x_max = config[
        "x_max"
    ]


    # ========================================================
    # RETAIN NODES
    # ========================================================

    retained_nodes = {

        label

        for label, xyz in nodes.items()

        if (
            xyz[
                0
            ]
            >=
            x_min
            -
            1.0e-5

            and

            xyz[
                0
            ]
            <=
            x_max
            +
            1.0e-5
        )
    }


    # ========================================================
    # RETAIN ELEMENTS
    # ========================================================

    retained_elements = {

        label

        for label, connectivity
        in elements.items()

        if all(
            node_label
            in
            retained_nodes

            for node_label
            in connectivity
        )
    }


    if len(
        retained_nodes
    ) == 0:

        raise RuntimeError(
            "Patch {} retained zero nodes."
            .format(
                patch_name
            )
        )


    if len(
        retained_elements
    ) == 0:

        raise RuntimeError(
            "Patch {} retained zero elements."
            .format(
                patch_name
            )
        )


    # ========================================================
    # CONTACT ELEMENTS
    # ========================================================

    original_contact_elements = (
        get_original_composite_contact_elements(

            assembly_lines,

            config[
                "contact_set_name"
            ],
        )
    )


    contact_elements = [

        label

        for label
        in original_contact_elements

        if label
        in
        retained_elements
    ]


    if len(
        contact_elements
    ) == 0:

        raise RuntimeError(
            (
                "No contact elements retained for patch {}."
            ).format(
                patch_name
            )
        )


    # ========================================================
    # INTERFACE NODE LABELS
    # ========================================================

    patch_interface_metadata = {}


    for interface_name in config[
        "interfaces"
    ]:

        data = interface_data[
            interface_name
        ]


        target_x = float(
            data[
                "actual_x"
            ][0]
        )


        target_coordinates = data[
            "coordinates"
        ].astype(
            np.float64
        )


        # ====================================================
        # COLLECT NODES AT THIS X PLANE
        # ====================================================

        interface_nodes = [

            (
                label,
                xyz
            )

            for label, xyz
            in nodes.items()

            if (
                label
                in
                retained_nodes

                and

                abs(
                    xyz[
                        0
                    ]
                    -
                    target_x
                )
                <
                1.0e-5
            )
        ]


        interface_nodes.sort(

            key=lambda item: (

                item[
                    1
                ][
                    1
                ],

                item[
                    1
                ][
                    2
                ],
            )
        )


        node_labels = [

            item[
                0
            ]

            for item
            in interface_nodes
        ]


        yz_coordinates = np.asarray(
            [

                [
                    item[
                        1
                    ][
                        1
                    ],

                    item[
                        1
                    ][
                        2
                    ],
                ]

                for item
                in interface_nodes
            ],
            dtype=np.float64,
        )


        target_yz = target_coordinates[
            :,
            1:3
        ]


        print(
            "{} -> patch nodes = {}, PCA nodes = {}"
            .format(
                interface_name,
                yz_coordinates.shape[
                    0
                ],
                target_yz.shape[
                    0
                ],
            )
        )


        # ====================================================
        # NODE COUNT CHECK
        # ====================================================

        if yz_coordinates.shape != (
            target_yz.shape
        ):

            raise RuntimeError(
                (
                    "Interface shape mismatch for {}. "
                    "Patch={} PCA={}"
                ).format(
                    interface_name,
                    yz_coordinates.shape,
                    target_yz.shape,
                )
            )


        # ====================================================
        # ORDER / COORDINATE CHECK
        # ====================================================

        if not np.allclose(
            yz_coordinates,
            target_yz,
            atol=1.0e-6,
        ):

            maximum_difference = float(
                np.max(
                    np.abs(
                        yz_coordinates
                        -
                        target_yz
                    )
                )
            )


            raise RuntimeError(
                (
                    "Interface ordering mismatch for {}. "
                    "Max YZ difference={}"
                ).format(
                    interface_name,
                    maximum_difference,
                )
            )


        patch_interface_metadata[
            interface_name
        ] = {

            "x":
                target_x,

            "node_labels":
                [
                    int(
                        value
                    )

                    for value
                    in node_labels
                ],
        }


    # ========================================================
    # POINT TRANSLATION
    # ========================================================

    point_translation = find_instance_translation(

        assembly_lines,

        config[
            "point_instance"
        ],
    )


    if point_translation is None:

        point_translation = (
            0.0,
            0.0,
            0.0,
        )


    # ========================================================
    # SELECT ORIGINAL POINT INSTANCE BLOCKS
    # ========================================================

    point_instance_lower = config[
        "point_instance"
    ].lower()


    contact_number = config[
        "contact_number"
    ]


    selected_point_blocks = []


    for block in assembly_blocks:

        header = lower(
            block[
                0
            ]
        )


        include = False


        # ----------------------------------------------------
        # Point-specific sets
        # ----------------------------------------------------

        if (
            "instance={}".format(
                point_instance_lower
            )
            in
            header
        ):

            include = True


        # ----------------------------------------------------
        # Point contact surface
        # ----------------------------------------------------

        if (
            header.startswith(
                "*surface"
            )
            and
            (
                "name={}".format(
                    config[
                        "point_surface_name"
                    ].lower()
                )
                in
                header
            )
        ):

            include = True


        # ----------------------------------------------------
        # Surface-smoothing auxiliary surfaces
        # ----------------------------------------------------

        smoothing_surface_prefix = (
            "_cp-{}-composite-1-{}"
            "_ssm_"
        ).format(
            contact_number,
            point_instance_lower,
        )


        if (
            header.startswith(
                "*surface"
            )
            and
            smoothing_surface_prefix
            in
            header
        ):

            include = True


        # ----------------------------------------------------
        # Rigid-body definition
        # ----------------------------------------------------

        if (
            header.startswith(
                "*rigid body"
            )
            and
            config[
                "picked_set"
            ].lower()
            in
            header
        ):

            include = True


        # ----------------------------------------------------
        # Never copy instance start/end blocks
        # ----------------------------------------------------

        if header.startswith(
            "*instance"
        ):

            include = False


        if header.startswith(
            "*end instance"
        ):

            include = False


        # ----------------------------------------------------
        # Never copy original Composite instance sets
        # ----------------------------------------------------

        if (
            "instance=composite-1"
            in
            header
        ):

            include = False


        if include:

            selected_point_blocks.append(
                block
            )


    # ========================================================
    # SURFACE SMOOTHING
    # ========================================================

    smoothing_header = (
        "*surface smoothing, "
        "name={}"
    ).format(
        config[
            "smoothing_name"
        ]
    )


    smoothing_block = find_keyword_block(
        lines,
        smoothing_header,
    )


    if smoothing_block is None:

        raise RuntimeError(
            (
                "Surface smoothing block not found: {}"
            ).format(
                config[
                    "smoothing_name"
                ]
            )
        )


    # ========================================================
    # OUTPUT TEMPLATE
    # ========================================================

    template_file = os.path.join(
        OUTPUT_DIR,
        "{}_template.inp".format(
            patch_name
        ),
    )


    with open(
        template_file,
        "w"
    ) as out:

        # ====================================================
        # HEADING
        # ====================================================

        out.write(
            "*Heading\n"
        )


        out.write(
            "** Seven-region FE-NO patch: {}\n"
            .format(
                patch_name
            )
        )


        out.write(
            (
                "** Composite x-range: "
                "{:.6f} to {:.6f}\n"
            ).format(
                x_min,
                x_max,
            )
        )


        out.write(
            "*Preprint, echo=NO, model=NO, "
            "history=NO, contact=NO\n"
        )


        # ====================================================
        # COMPOSITE PART
        # ====================================================

        out.write(
            "*Part, name=Composite\n"
        )


        out.write(
            "*Node\n"
        )


        for label in sorted(
            retained_nodes
        ):

            xyz = nodes[
                label
            ]


            out.write(
                (
                    "{}, {:.10f}, {:.10f}, {:.10f}\n"
                ).format(
                    label,
                    xyz[
                        0
                    ],
                    xyz[
                        1
                    ],
                    xyz[
                        2
                    ],
                )
            )


        out.write(
            "*Element, type=C3D8R\n"
        )


        for label in sorted(
            retained_elements
        ):

            connectivity = elements[
                label
            ]


            out.write(
                "{}, {}\n".format(

                    label,

                    ", ".join(
                        [
                            str(
                                value
                            )

                            for value
                            in connectivity
                        ]
                    ),
                )
            )


        out.write(
            "*Elset, elset=Set-1\n"
        )


        write_integer_set(

            out,

            sorted(
                retained_elements
            ),
        )


        # ====================================================
        # MATERIAL ORIENTATION
        # ====================================================

        out.write(
            "*Orientation, name=Ori-1\n"
        )


        out.write(
            "1., 0., 0., 0., 1., 0.\n"
        )


        out.write(
            "1, 0.\n"
        )


        out.write(
            (
                "*Solid Section, elset=Set-1, "
                "orientation=Ori-1, "
                "material=GlassEpoxy\n"
            )
        )


        out.write(
            ",\n"
        )


        out.write(
            "*End Part\n"
        )


        # ====================================================
        # ORIGINAL ROLLER / NOSE PART
        # ====================================================

        for line in point_part_lines:

            out.write(
                line
                +
                "\n"
            )


        # ====================================================
        # ASSEMBLY
        # ====================================================

        out.write(
            "*Assembly, name=Assembly\n"
        )


        out.write(
            "*Instance, name=Composite-1, part=Composite\n"
        )


        out.write(
            "*End Instance\n"
        )


        out.write(
            (
                "*Instance, name={}, part=Point\n"
            ).format(
                config[
                    "point_instance"
                ]
            )
        )


        out.write(
            (
                "{:.10f}, {:.10f}, {:.10f}\n"
            ).format(
                point_translation[
                    0
                ],
                point_translation[
                    1
                ],
                point_translation[
                    2
                ],
            )
        )


        out.write(
            "*End Instance\n"
        )


        # ====================================================
        # COMPOSITE CONTACT ELSET
        # ====================================================

        out.write(
            (
                "*Elset, elset={}, internal, "
                "instance=Composite-1\n"
            ).format(
                config[
                    "contact_set_name"
                ]
            )
        )


        write_integer_set(

            out,

            contact_elements,
        )


        # ====================================================
        # COMPOSITE CONTACT SURFACE
        # ====================================================

        out.write(
            "*Surface, type=ELEMENT, name={}\n"
            .format(
                config[
                    "composite_surface_name"
                ]
            )
        )


        out.write(
            "{}, {}\n".format(
                config[
                    "contact_set_name"
                ],
                config[
                    "contact_side"
                ],
            )
        )


        # ====================================================
        # ORIGINAL POINT SETS / SURFACES / RIGID BODY
        # ====================================================

        for block in selected_point_blocks:

            for line in block:

                out.write(
                    line
                    +
                    "\n"
                )


        out.write(
            "*End Assembly\n"
        )


        # ====================================================
        # GLASS-EPOXY MATERIAL
        #
        # Filled later by hybrid runtime.
        # ====================================================

        out.write(
            "*Material, name=GlassEpoxy\n"
        )


        out.write(
            "*Elastic, type=ENGINEERING CONSTANTS\n"
        )


        out.write(
            "@@GLASS_LINE1@@\n"
        )


        out.write(
            "@@GLASS_LINE2@@\n"
        )


        # ====================================================
        # STEEL MATERIAL
        # ====================================================

        out.write(
            "*Material, name=Steel\n"
        )


        out.write(
            "*Elastic\n"
        )


        out.write(
            "200000., 0.3\n"
        )


        # ====================================================
        # CONTACT PROPERTY
        # ====================================================

        out.write(
            "*Surface Interaction, name=IntProp-1\n"
        )


        out.write(
            "1.,\n"
        )


        out.write(
            "*Friction\n"
        )


        out.write(
            "0.,\n"
        )


        out.write(
            "*Surface Behavior, pressure-overclosure=HARD\n"
        )


        # ====================================================
        # SURFACE SMOOTHING
        # ====================================================

        for line in smoothing_block:

            out.write(
                line
                +
                "\n"
            )


        # ====================================================
        # INITIAL RIGID TOOL BOUNDARY CONDITIONS
        # ====================================================

        if config[
            "initial_bc_type"
        ] == "support":

            out.write(
                "*Boundary\n"
            )


            out.write(
                "Set-9, ENCASTRE\n"
            )


        elif config[
            "initial_bc_type"
        ] == "nose":

            out.write(
                "*Boundary\n"
            )


            out.write(
                "Set-10, 1, 1\n"
            )


            out.write(
                "Set-10, 3, 3\n"
            )


            out.write(
                "Set-10, 4, 4\n"
            )


            out.write(
                "Set-10, 5, 5\n"
            )


            out.write(
                "Set-10, 6, 6\n"
            )


        # ====================================================
        # CONTACT PAIR
        # ====================================================

        out.write(
            (
                "*Contact Pair, interaction=IntProp-1, "
                "type=SURFACE TO SURFACE, "
                "geometric correction={}\n"
            ).format(
                config[
                    "smoothing_name"
                ]
            )
        )


        out.write(
            "{}, {}\n".format(
                config[
                    "point_surface_name"
                ],
                config[
                    "composite_surface_name"
                ],
            )
        )


        # ====================================================
        # LOADING STEP
        # ====================================================

        out.write(
            "*Step, name=Loading, nlgeom=YES, inc=1000\n"
        )


        out.write(
            (
                "*Static, stabilize=0.0002, "
                "allsdtol=0.05, continue=NO\n"
            )
        )


        out.write(
            "0.01, 1., 1e-06, 0.05\n"
        )


        # ====================================================
        # CENTER NOSE DISPLACEMENT
        # ====================================================

        if patch_name == "center":

            out.write(
                "*Boundary\n"
            )


            out.write(
                "Set-10, 2, 2, -0.2\n"
            )


        # ====================================================
        # INTERFACE DISPLACEMENTS
        #
        # Inserted at runtime by Broyden.
        # ====================================================

        out.write(
            "*Boundary\n"
        )


        out.write(
            "@@INTERFACE_BC@@\n"
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        out.write(
            "*Restart, write, frequency=0\n"
        )


        out.write(
            "*Output, field, number interval=1\n"
        )


        out.write(
            "*Node Output\n"
        )


        out.write(
            "U, RF\n"
        )


        out.write(
            "*Element Output, directions=YES\n"
        )


        out.write(
            "LE, S\n"
        )


        out.write(
            "*Contact Output, variable=PRESELECT\n"
        )


        out.write(
            "*End Step\n"
        )


    # ========================================================
    # SAVE PATCH METADATA
    # ========================================================

    metadata[
        "patches"
    ][
        patch_name
    ] = {

        "template_file":
            template_file,

        "x_min":
            float(
                x_min
            ),

        "x_max":
            float(
                x_max
            ),

        "point_instance":
            config[
                "point_instance"
            ],

        "number_nodes":
            int(
                len(
                    retained_nodes
                )
            ),

        "number_elements":
            int(
                len(
                    retained_elements
                )
            ),

        "number_contact_elements":
            int(
                len(
                    contact_elements
                )
            ),

        "interfaces":
            patch_interface_metadata,
    }


    print("")
    print(
        "Composite nodes:",
        len(
            retained_nodes
        ),
    )


    print(
        "Composite elements:",
        len(
            retained_elements
        ),
    )


    print(
        "Contact elements:",
        len(
            contact_elements
        ),
    )


    print(
        "Template:",
        template_file
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_file = os.path.join(
    OUTPUT_DIR,
    "patch_metadata.json",
)


with open(
    metadata_file,
    "w"
) as file_object:

    json.dump(
        metadata,
        file_object,
        indent=4,
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64R COMPLETE"
)

print(
    "=========================================="
)

print(
    "Metadata:",
    metadata_file
)

print("")