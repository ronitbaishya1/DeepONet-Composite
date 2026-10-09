import os
import shutil
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--segment",
    choices=["left", "right"],
    required=True,
)

parser.add_argument(
    "--source_dir",
    required=True,
)

parser.add_argument(
    "--extracted_dir",
    required=True,
)

parser.add_argument(
    "--output_dir",
    required=True,
)

args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# LOAD ORIGINAL LOCAL DATASET
# ============================================================

split_file = os.path.join(
    args.source_dir,
    "split_assignment.csv",
)

split = pd.read_csv(
    split_file
)


case_ids = (
    split[
        "CaseID"
    ]
    .astype(str)
    .to_list()
)


coordinates = np.load(
    os.path.join(
        args.source_dir,
        "coordinates.npy",
    )
).astype(
    np.float64
)


print("")
print("==========================================")
print("STEP 61")
print("==========================================")
print("Segment:", args.segment)
print("Cases:", len(case_ids))
print("Nodes:", coordinates.shape[0])


# ============================================================
# READ FEM IP FIELDS
# ============================================================

all_stress = []
all_strain = []

reference_ip_coordinates = None
reference_element_labels = None


for case_number, case_id in enumerate(
    case_ids
):

    filename = os.path.join(
        args.extracted_dir,
        "{}_ip.csv".format(
            case_id
        ),
    )


    if not os.path.isfile(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


    dataframe = pd.read_csv(
        filename
    )


    dataframe = (
        dataframe
        .sort_values(
            "ElementLabel"
        )
        .reset_index(
            drop=True
        )
    )


    ip_coordinates = dataframe[
        [
            "X",
            "Y",
            "Z",
        ]
    ].to_numpy(
        dtype=np.float64
    )


    element_labels = dataframe[
        "ElementLabel"
    ].to_numpy(
        dtype=int
    )


    if reference_ip_coordinates is None:

        reference_ip_coordinates = (
            ip_coordinates.copy()
        )

        reference_element_labels = (
            element_labels.copy()
        )

    else:

        if not np.array_equal(
            reference_element_labels,
            element_labels,
        ):

            raise RuntimeError(
                "Element ordering changed for {}".format(
                    case_id
                )
            )


        maximum_difference = float(
            np.max(
                np.abs(
                    ip_coordinates
                    -
                    reference_ip_coordinates
                )
            )
        )


        if maximum_difference > 1.0e-7:

            raise RuntimeError(
                "IP coordinates changed in {}. Max = {}"
                .format(
                    case_id,
                    maximum_difference,
                )
            )


    stress = dataframe[
        [
            "S11",
            "S22",
            "S33",
            "S12",
            "S13",
            "S23",
        ]
    ].to_numpy(
        dtype=np.float32
    )


    strain = dataframe[
        [
            "LE11",
            "LE22",
            "LE33",
            "LE12",
            "LE13",
            "LE23",
        ]
    ].to_numpy(
        dtype=np.float32
    )


    all_stress.append(
        stress
    )


    all_strain.append(
        strain
    )


    print(
        "[{}/{}] {}".format(
            case_number + 1,
            len(case_ids),
            case_id,
        )
    )


S = np.asarray(
    all_stress,
    dtype=np.float32,
)


LE = np.asarray(
    all_strain,
    dtype=np.float32,
)


# ============================================================
# BUILD STRUCTURED ELEMENT CONNECTIVITY
#
# Needed later for:
#
# displacement -> strain
#
# using the C3D8R element B matrix.
# ============================================================

x_values = np.unique(
    np.round(
        coordinates[:, 0],
        8,
    )
)

y_values = np.unique(
    np.round(
        coordinates[:, 1],
        8,
    )
)

z_values = np.unique(
    np.round(
        coordinates[:, 2],
        8,
    )
)


x_values.sort()
y_values.sort()
z_values.sort()


coordinate_to_node = {}


for node_index, xyz in enumerate(
    coordinates
):

    key = (
        round(
            float(xyz[0]),
            8,
        ),

        round(
            float(xyz[1]),
            8,
        ),

        round(
            float(xyz[2]),
            8,
        ),
    )

    coordinate_to_node[key] = (
        node_index
    )


def nearest_interval(
    values,
    midpoint,
):

    candidate_midpoints = (
        0.5
        *
        (
            values[:-1]
            +
            values[1:]
        )
    )


    index = int(
        np.argmin(
            np.abs(
                candidate_midpoints
                -
                midpoint
            )
        )
    )


    difference = abs(
        candidate_midpoints[index]
        -
        midpoint
    )


    if difference > 1.0e-5:

        raise RuntimeError(
            "Could not locate element interval."
        )


    return index


connectivity = []

B_matrices = []


# Natural coordinates matching the exact element ordering
# used in Step 17.
natural_coordinates = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ],
    dtype=np.float64,
)


for element_index, centroid in enumerate(
    reference_ip_coordinates
):

    ix = nearest_interval(
        x_values,
        centroid[0],
    )

    iy = nearest_interval(
        y_values,
        centroid[1],
    )

    iz = nearest_interval(
        z_values,
        centroid[2],
    )


    x0 = x_values[ix]
    x1 = x_values[ix + 1]

    y0 = y_values[iy]
    y1 = y_values[iy + 1]

    z0 = z_values[iz]
    z1 = z_values[iz + 1]


    element_coordinates = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]


    element_nodes = []


    for xyz in element_coordinates:

        key = (
            round(
                float(xyz[0]),
                8,
            ),

            round(
                float(xyz[1]),
                8,
            ),

            round(
                float(xyz[2]),
                8,
            ),
        )


        if key not in coordinate_to_node:

            raise RuntimeError(
                "Element node not found: {}".format(
                    key
                )
            )


        element_nodes.append(
            coordinate_to_node[
                key
            ]
        )


    connectivity.append(
        element_nodes
    )


    dx = float(
        x1
        -
        x0
    )

    dy = float(
        y1
        -
        y0
    )

    dz = float(
        z1
        -
        z0
    )


    B = np.zeros(
        (
            6,
            24,
        ),
        dtype=np.float64,
    )


    for local_node in range(
        8
    ):

        xi = natural_coordinates[
            local_node,
            0
        ]

        eta = natural_coordinates[
            local_node,
            1
        ]

        zeta = natural_coordinates[
            local_node,
            2
        ]


        dN_dx = (
            xi
            /
            (
                4.0
                *
                dx
            )
        )

        dN_dy = (
            eta
            /
            (
                4.0
                *
                dy
            )
        )

        dN_dz = (
            zeta
            /
            (
                4.0
                *
                dz
            )
        )


        column = (
            3
            *
            local_node
        )


        # epsilon_11
        B[
            0,
            column
        ] = dN_dx


        # epsilon_22
        B[
            1,
            column + 1
        ] = dN_dy


        # epsilon_33
        B[
            2,
            column + 2
        ] = dN_dz


        # gamma_12
        B[
            3,
            column
        ] = dN_dy

        B[
            3,
            column + 1
        ] = dN_dx


        # gamma_13
        B[
            4,
            column
        ] = dN_dz

        B[
            4,
            column + 2
        ] = dN_dx


        # gamma_23
        B[
            5,
            column + 1
        ] = dN_dz

        B[
            5,
            column + 2
        ] = dN_dy


    B_matrices.append(
        B
    )


connectivity = np.asarray(
    connectivity,
    dtype=np.int64,
)


B_matrices = np.asarray(
    B_matrices,
    dtype=np.float32,
)


# ============================================================
# SAVE MECHANICS ARRAYS
# ============================================================

np.save(
    os.path.join(
        args.output_dir,
        "ip_coordinates.npy",
    ),
    reference_ip_coordinates.astype(
        np.float32
    ),
)


np.save(
    os.path.join(
        args.output_dir,
        "element_labels.npy",
    ),
    reference_element_labels,
)


np.save(
    os.path.join(
        args.output_dir,
        "element_node_indices.npy",
    ),
    connectivity,
)


np.save(
    os.path.join(
        args.output_dir,
        "B_center.npy",
    ),
    B_matrices,
)


np.save(
    os.path.join(
        args.output_dir,
        "LE.npy",
    ),
    LE,
)


np.save(
    os.path.join(
        args.output_dir,
        "S.npy",
    ),
    S,
)


# ============================================================
# COPY EXISTING LOCAL DATA
# ============================================================

files_to_copy = [
    "branch_inputs.npy",
    "coordinates.npy",
    "node_labels.npy",
    "U1.npy",
    "U2.npy",
    "U3.npy",
    "g_left.npy",
    "g_right.npy",
    "split_assignment.csv",
]


for filename in files_to_copy:

    source = os.path.join(
        args.source_dir,
        filename,
    )

    destination = os.path.join(
        args.output_dir,
        filename,
    )


    if not os.path.isfile(
        source
    ):

        raise FileNotFoundError(
            source
        )


    shutil.copyfile(
        source,
        destination,
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("")
print("==========================================")
print("STEP 61 COMPLETE")
print("==========================================")

print(
    "Cases:",
    S.shape[0],
)

print(
    "Integration points:",
    S.shape[1],
)

print(
    "LE shape:",
    LE.shape,
)

print(
    "S shape:",
    S.shape,
)

print(
    "Connectivity:",
    connectivity.shape,
)

print(
    "B matrices:",
    B_matrices.shape,
)

print(
    "Finite LE:",
    np.isfinite(
        LE
    ).all(),
)

print(
    "Finite S:",
    np.isfinite(
        S
    ).all(),
)