import os
import json
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "all",
        "outer_left",
        "inner_left",
        "inner_right",
        "outer_right",
    ],
    default="all",
)


parser.add_argument(
    "--design_dir",
    default=
        "data/candidate_partition_designs_7region",
)


parser.add_argument(
    "--interface_dir",
    default=
        "data/candidate_partition_interfaces_7region",
)


parser.add_argument(
    "--extracted_root",
    default=
        "data/windows_candidate_7region_extracted",
)


parser.add_argument(
    "--output_root",
    default=
        "data/hybrid_bulk_7region",
)


parser.add_argument(
    "--seed",
    type=int,
    default=12345,
)


args = parser.parse_args()


SEGMENTS = [
    "outer_left",
    "inner_left",
    "inner_right",
    "outer_right",
]


if args.segment == "all":

    selected_segments = SEGMENTS

else:

    selected_segments = [
        args.segment
    ]


# ============================================================
# B MATRIX
#
# IMPORTANT:
#
# Rows 3,4,5 are ENGINEERING shear:
#
# gamma12, gamma13, gamma23
#
# Abaqus LE12/LE13/LE23 are tensor shear strains.
# We will account for the factor of 2 later whenever
# kinematic/constitutive consistency is imposed.
# ============================================================

def build_c3d8r_B(
    element_coordinates
):

    signs = np.asarray(
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


    dN_natural = (
        signs
        /
        8.0
    )


    jacobian = (
        dN_natural.T
        @
        element_coordinates
    )


    inverse_jacobian = np.linalg.inv(
        jacobian
    )


    dN_global = (
        dN_natural
        @
        inverse_jacobian
    )


    B = np.zeros(
        (
            6,
            24,
        ),
        dtype=np.float64,
    )


    for node_index in range(
        8
    ):

        dNx = dN_global[
            node_index,
            0
        ]

        dNy = dN_global[
            node_index,
            1
        ]

        dNz = dN_global[
            node_index,
            2
        ]


        column = (
            3
            *
            node_index
        )


        B[
            0,
            column
        ] = dNx


        B[
            1,
            column + 1
        ] = dNy


        B[
            2,
            column + 2
        ] = dNz


        B[
            3,
            column
        ] = dNy

        B[
            3,
            column + 1
        ] = dNx


        B[
            4,
            column
        ] = dNz

        B[
            4,
            column + 2
        ] = dNx


        B[
            5,
            column + 1
        ] = dNz

        B[
            5,
            column + 2
        ] = dNy


    return B


# ============================================================
# ONE SEGMENT
# ============================================================

def assemble_segment(
    segment
):

    print("")
    print(
        "=========================================="
    )

    print(
        "ASSEMBLING:",
        segment.upper()
    )

    print(
        "=========================================="
    )


    design_file = os.path.join(
        args.design_dir,
        "{}_candidate_design_80.csv".format(
            segment
        ),
    )


    extracted_dir = os.path.join(
        args.extracted_root,
        segment,
    )


    output_dir = os.path.join(
        args.output_root,
        segment,
    )


    os.makedirs(
        output_dir,
        exist_ok=True,
    )


    design = pd.read_csv(
        design_file
    )


    if len(
        design
    ) == 0:

        raise RuntimeError(
            "Design is empty."
        )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    first = design.iloc[
        0
    ]


    left_interface_name = str(
        first[
            "LeftInterface"
        ]
    )


    right_interface_name = str(
        first[
            "RightInterface"
        ]
    )


    if left_interface_name.upper() in [
        "NONE",
        "NAN",
    ]:

        left_interface_name = None


    if right_interface_name.upper() in [
        "NONE",
        "NAN",
    ]:

        right_interface_name = None


    left_interface = None
    right_interface = None


    if left_interface_name is not None:

        left_interface = np.load(
            os.path.join(
                args.interface_dir,
                "interface_{}.npz".format(
                    left_interface_name
                ),
            )
        )


    if right_interface_name is not None:

        right_interface = np.load(
            os.path.join(
                args.interface_dir,
                "interface_{}.npz".format(
                    right_interface_name
                ),
            )
        )


    n_left_modes = (
        0
        if left_interface is None
        else
        left_interface[
            "basis"
        ].shape[1]
    )


    n_right_modes = (
        0
        if right_interface is None
        else
        right_interface[
            "basis"
        ].shape[1]
    )


    # ========================================================
    # STORAGE
    # ========================================================

    all_U = []

    all_S = []

    all_LE = []

    all_rf_left = []

    all_rf_right = []

    all_g_left = []

    all_g_right = []


    reference_coordinates = None

    reference_node_labels = None

    reference_ip_coordinates = None

    reference_element_labels = None


    # ========================================================
    # CASE LOOP
    # ========================================================

    for case_number, row in design.iterrows():

        case_id = str(
            row[
                "CaseID"
            ]
        )


        node_file = os.path.join(
            extracted_dir,
            case_id
            +
            "_nodes.csv",
        )


        ip_file = os.path.join(
            extracted_dir,
            case_id
            +
            "_ip.csv",
        )


        if not os.path.isfile(
            node_file
        ):

            raise FileNotFoundError(
                node_file
            )


        if not os.path.isfile(
            ip_file
        ):

            raise FileNotFoundError(
                ip_file
            )


        nodes = pd.read_csv(
            node_file
        )


        nodes = nodes.sort_values(
            [
                "X",
                "Y",
                "Z",
            ]
        ).reset_index(
            drop=True
        )


        ip = pd.read_csv(
            ip_file
        )


        ip = ip.sort_values(
            "ElementLabel"
        ).reset_index(
            drop=True
        )


        coordinates = nodes[
            [
                "X",
                "Y",
                "Z",
            ]
        ].to_numpy(
            dtype=np.float64
        )


        node_labels = nodes[
            "NodeLabel"
        ].to_numpy(
            dtype=int
        )


        ip_coordinates = ip[
            [
                "X",
                "Y",
                "Z",
            ]
        ].to_numpy(
            dtype=np.float64
        )


        element_labels = ip[
            "ElementLabel"
        ].to_numpy(
            dtype=int
        )


        if reference_coordinates is None:

            reference_coordinates = (
                coordinates.copy()
            )

            reference_node_labels = (
                node_labels.copy()
            )

            reference_ip_coordinates = (
                ip_coordinates.copy()
            )

            reference_element_labels = (
                element_labels.copy()
            )


        else:

            if not np.allclose(
                coordinates,
                reference_coordinates,
                atol=1.0e-9,
            ):

                raise RuntimeError(
                    "Node coordinates differ for {}."
                    .format(
                        case_id
                    )
                )


            if not np.allclose(
                ip_coordinates,
                reference_ip_coordinates,
                atol=1.0e-9,
            ):

                raise RuntimeError(
                    "IP coordinates differ for {}."
                    .format(
                        case_id
                    )
                )


        # ====================================================
        # FIELD DATA
        # ====================================================

        U = nodes[
            [
                "U1",
                "U2",
                "U3",
            ]
        ].to_numpy(
            dtype=np.float64
        )


        S = ip[
            [
                "S11",
                "S22",
                "S33",
                "S12",
                "S13",
                "S23",
            ]
        ].to_numpy(
            dtype=np.float64
        )


        LE = ip[
            [
                "LE11",
                "LE22",
                "LE33",
                "LE12",
                "LE13",
                "LE23",
            ]
        ].to_numpy(
            dtype=np.float64
        )


        all_U.append(
            U
        )

        all_S.append(
            S
        )

        all_LE.append(
            LE
        )


        # ====================================================
        # LEFT INTERFACE FORCE
        # ====================================================

        if left_interface is not None:

            x_left = float(
                left_interface[
                    "actual_x"
                ][0]
            )


            face = nodes[
                np.isclose(
                    nodes[
                        "X"
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    x_left,
                    atol=1.0e-7,
                )
            ].copy()


            face = face.sort_values(
                [
                    "Y",
                    "Z",
                ]
            )


            rf = face[
                [
                    "RF1",
                    "RF2",
                    "RF3",
                ]
            ].to_numpy(
                dtype=np.float64
            ).reshape(
                -1
            )


            basis = left_interface[
                "basis"
            ].astype(
                np.float64
            )


            if rf.shape[
                0
            ] != basis.shape[
                0
            ]:

                raise RuntimeError(
                    "Left RF dimension mismatch."
                )


            g = (
                basis.T
                @
                rf
            )


            all_rf_left.append(
                rf
            )

            all_g_left.append(
                g
            )


        # ====================================================
        # RIGHT INTERFACE FORCE
        # ====================================================

        if right_interface is not None:

            x_right = float(
                right_interface[
                    "actual_x"
                ][0]
            )


            face = nodes[
                np.isclose(
                    nodes[
                        "X"
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    x_right,
                    atol=1.0e-7,
                )
            ].copy()


            face = face.sort_values(
                [
                    "Y",
                    "Z",
                ]
            )


            rf = face[
                [
                    "RF1",
                    "RF2",
                    "RF3",
                ]
            ].to_numpy(
                dtype=np.float64
            ).reshape(
                -1
            )


            basis = right_interface[
                "basis"
            ].astype(
                np.float64
            )


            if rf.shape[
                0
            ] != basis.shape[
                0
            ]:

                raise RuntimeError(
                    "Right RF dimension mismatch."
                )


            g = (
                basis.T
                @
                rf
            )


            all_rf_right.append(
                rf
            )

            all_g_right.append(
                g
            )


        print(
            "[{}/{}] {}"
            .format(
                case_number + 1,
                len(
                    design
                ),
                case_id,
            )
        )


    # ========================================================
    # STACK FIELDS
    # ========================================================

    all_U = np.asarray(
        all_U,
        dtype=np.float32,
    )


    all_S = np.asarray(
        all_S,
        dtype=np.float32,
    )


    all_LE = np.asarray(
        all_LE,
        dtype=np.float32,
    )


    # ========================================================
    # BUILD STRUCTURED CONNECTIVITY
    # ========================================================

    x_values = np.unique(
        np.round(
            reference_coordinates[
                :,
                0
            ],
            decimals=10,
        )
    )


    y_values = np.unique(
        np.round(
            reference_coordinates[
                :,
                1
            ],
            decimals=10,
        )
    )


    z_values = np.unique(
        np.round(
            reference_coordinates[
                :,
                2
            ],
            decimals=10,
        )
    )


    x_values.sort()
    y_values.sort()
    z_values.sort()


    coordinate_to_index = {}


    for node_index, xyz in enumerate(
        reference_coordinates
    ):

        key = tuple(
            np.round(
                xyz,
                decimals=10,
            )
        )


        coordinate_to_index[
            key
        ] = node_index


    connectivity = []

    B_matrices = []

    expected_centroids = []


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

                x0 = x_values[
                    ix
                ]

                x1 = x_values[
                    ix + 1
                ]

                y0 = y_values[
                    iy
                ]

                y1 = y_values[
                    iy + 1
                ]

                z0 = z_values[
                    iz
                ]

                z1 = z_values[
                    iz + 1
                ]


                element_coordinates = np.asarray(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    ],
                    dtype=np.float64,
                )


                element_indices = []


                for xyz in element_coordinates:

                    key = tuple(
                        np.round(
                            xyz,
                            decimals=10,
                        )
                    )


                    if key not in (
                        coordinate_to_index
                    ):

                        raise RuntimeError(
                            "Missing structured node {}."
                            .format(
                                key
                            )
                        )


                    element_indices.append(
                        coordinate_to_index[
                            key
                        ]
                    )


                element_indices = np.asarray(
                    element_indices,
                    dtype=np.int64,
                )


                connectivity.append(
                    element_indices
                )


                B_matrices.append(
                    build_c3d8r_B(
                        element_coordinates
                    )
                )


                expected_centroids.append(
                    np.mean(
                        element_coordinates,
                        axis=0,
                    )
                )


    connectivity = np.asarray(
        connectivity,
        dtype=np.int64,
    )


    B_matrices = np.asarray(
        B_matrices,
        dtype=np.float32,
    )


    expected_centroids = np.asarray(
        expected_centroids,
        dtype=np.float64,
    )


    if connectivity.shape[
        0
    ] != reference_ip_coordinates.shape[
        0
    ]:

        raise RuntimeError(
            "Connectivity/IP element count mismatch."
        )


    if not np.allclose(
        expected_centroids,
        reference_ip_coordinates,
        atol=1.0e-7,
    ):

        maximum_difference = np.max(
            np.abs(
                expected_centroids
                -
                reference_ip_coordinates
            )
        )


        raise RuntimeError(
            (
                "Structured centroid ordering does not match "
                "ODB element ordering. Max difference = {}"
            ).format(
                maximum_difference
            )
        )


    # ========================================================
    # BRANCH INPUT
    # ========================================================

    branch_columns = [
        "E1_MPa",
        "E2_MPa",
        "G12_MPa",
    ]


    if left_interface is not None:

        branch_columns += [
            "cL_{:02d}".format(
                index + 1
            )

            for index
            in range(
                n_left_modes
            )
        ]


    if right_interface is not None:

        branch_columns += [
            "cR_{:02d}".format(
                index + 1
            )

            for index
            in range(
                n_right_modes
            )
        ]


    branch_inputs = design[
        branch_columns
    ].to_numpy(
        dtype=np.float32
    )


    # ========================================================
    # FORCE ARRAYS
    # ========================================================

    number_cases = len(
        design
    )


    if left_interface is None:

        rf_left = np.zeros(
            (
                number_cases,
                0,
            ),
            dtype=np.float32,
        )

        g_left = np.zeros(
            (
                number_cases,
                0,
            ),
            dtype=np.float32,
        )

    else:

        rf_left = np.asarray(
            all_rf_left,
            dtype=np.float32,
        )

        g_left = np.asarray(
            all_g_left,
            dtype=np.float32,
        )


    if right_interface is None:

        rf_right = np.zeros(
            (
                number_cases,
                0,
            ),
            dtype=np.float32,
        )

        g_right = np.zeros(
            (
                number_cases,
                0,
            ),
            dtype=np.float32,
        )

    else:

        rf_right = np.asarray(
            all_rf_right,
            dtype=np.float32,
        )

        g_right = np.asarray(
            all_g_right,
            dtype=np.float32,
        )


    g_all = np.concatenate(
        [
            g_left,
            g_right,
        ],
        axis=1,
    )


    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(
            output_dir,
            "branch_inputs.npy",
        ),
        branch_inputs,
    )


    np.save(
        os.path.join(
            output_dir,
            "coordinates.npy",
        ),
        reference_coordinates.astype(
            np.float32
        ),
    )


    np.save(
        os.path.join(
            output_dir,
            "node_labels.npy",
        ),
        reference_node_labels,
    )


    np.save(
        os.path.join(
            output_dir,
            "ip_coordinates.npy",
        ),
        reference_ip_coordinates.astype(
            np.float32
        ),
    )


    np.save(
        os.path.join(
            output_dir,
            "element_labels.npy",
        ),
        reference_element_labels,
    )


    np.save(
        os.path.join(
            output_dir,
            "element_node_indices.npy",
        ),
        connectivity,
    )


    np.save(
        os.path.join(
            output_dir,
            "B_center.npy",
        ),
        B_matrices,
    )


    np.save(
        os.path.join(
            output_dir,
            "U1.npy",
        ),
        all_U[
            :,
            :,
            0
        ],
    )


    np.save(
        os.path.join(
            output_dir,
            "U2.npy",
        ),
        all_U[
            :,
            :,
            1
        ],
    )


    np.save(
        os.path.join(
            output_dir,
            "U3.npy",
        ),
        all_U[
            :,
            :,
            2
        ],
    )


    np.save(
        os.path.join(
            output_dir,
            "S.npy",
        ),
        all_S,
    )


    np.save(
        os.path.join(
            output_dir,
            "LE.npy",
        ),
        all_LE,
    )


    np.save(
        os.path.join(
            output_dir,
            "rf_left.npy",
        ),
        rf_left,
    )


    np.save(
        os.path.join(
            output_dir,
            "rf_right.npy",
        ),
        rf_right,
    )


    np.save(
        os.path.join(
            output_dir,
            "g_left.npy",
        ),
        g_left,
    )


    np.save(
        os.path.join(
            output_dir,
            "g_right.npy",
        ),
        g_right,
    )


    np.save(
        os.path.join(
            output_dir,
            "g_all.npy",
        ),
        g_all,
    )


    # ========================================================
    # DETERMINISTIC 56 / 12 / 12 SPLIT
    # ========================================================

    rng = np.random.default_rng(
        args.seed
    )


    indices = np.arange(
        number_cases
    )


    rng.shuffle(
        indices
    )


    n_train = int(
        round(
            0.70
            *
            number_cases
        )
    )


    n_validation = int(
        round(
            0.15
            *
            number_cases
        )
    )


    train_indices = indices[
        :n_train
    ]


    validation_indices = indices[
        n_train:
        n_train
        +
        n_validation
    ]


    test_indices = indices[
        n_train
        +
        n_validation:
    ]


    split_labels = np.empty(
        number_cases,
        dtype=object,
    )


    split_labels[
        train_indices
    ] = "train"


    split_labels[
        validation_indices
    ] = "validation"


    split_labels[
        test_indices
    ] = "test"


    split_table = pd.DataFrame(
        {
            "Index":
                np.arange(
                    number_cases
                ),

            "CaseID":
                design[
                    "CaseID"
                ],

            "Source":
                design[
                    "Source"
                ],

            "Split":
                split_labels,
        }
    )


    split_table.to_csv(
        os.path.join(
            output_dir,
            "split_assignment.csv",
        ),
        index=False,
    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "segment":
            segment,

        "cases":
            int(
                number_cases
            ),

        "nodes":
            int(
                reference_coordinates.shape[
                    0
                ]
            ),

        "elements":
            int(
                reference_ip_coordinates.shape[
                    0
                ]
            ),

        "branch_dimension":
            int(
                branch_inputs.shape[
                    1
                ]
            ),

        "left_interface":
            left_interface_name,

        "right_interface":
            right_interface_name,

        "left_modes":
            int(
                n_left_modes
            ),

        "right_modes":
            int(
                n_right_modes
            ),

        "force_dimension":
            int(
                g_all.shape[
                    1
                ]
            ),

        "B_matrix_shear_convention":
            (
                "engineering shear gamma12, "
                "gamma13, gamma23"
            ),

        "Abaqus_LE_shear_convention":
            (
                "tensor logarithmic shear LE12, "
                "LE13, LE23"
            ),

        "train_cases":
            int(
                len(
                    train_indices
                )
            ),

        "validation_cases":
            int(
                len(
                    validation_indices
                )
            ),

        "test_cases":
            int(
                len(
                    test_indices
                )
            ),
    }


    with open(
        os.path.join(
            output_dir,
            "metadata.json",
        ),
        "w",
    ) as file_object:

        json.dump(
            metadata,
            file_object,
            indent=4,
        )


    print("")
    print(
        json.dumps(
            metadata,
            indent=4,
        )
    )


# ============================================================
# RUN
# ============================================================

for segment in selected_segments:

    assemble_segment(
        segment
    )


print("")
print(
    "=========================================="
)

print(
    "STEP 64J COMPLETE"
)

print(
    "=========================================="
)