import numpy as np
import torch


# ============================================================
# STRUCTURED C3D8R FINITE-ELEMENT PHYSICS
# ============================================================

class StructuredHexPhysics:

    def __init__(
        self,
        coordinates,
        device,
    ):

        self.device = device


        coordinates = np.asarray(
            coordinates,
            dtype=np.float64,
        )


        self.number_nodes = coordinates.shape[
            0
        ]


        # ====================================================
        # UNIQUE GRID COORDINATES
        # ====================================================

        x_values = np.sort(
            np.unique(
                np.round(
                    coordinates[
                        :,
                        0
                    ],
                    8,
                )
            )
        )


        y_values = np.sort(
            np.unique(
                np.round(
                    coordinates[
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
                    coordinates[
                        :,
                        2
                    ],
                    8,
                )
            )
        )


        expected_nodes = (

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


        if expected_nodes != self.number_nodes:

            raise RuntimeError(

                "The local mesh is not a complete "
                "structured grid.\n"
                "Expected {} nodes but found {}."
                .format(
                    expected_nodes,
                    self.number_nodes,
                )
            )


        # ====================================================
        # COORDINATE -> NODE NUMBER
        # ====================================================

        coordinate_map = {}


        for node_index, xyz in enumerate(
            coordinates
        ):

            key = (

                round(
                    float(
                        xyz[
                            0
                        ]
                    ),
                    8,
                ),

                round(
                    float(
                        xyz[
                            1
                        ]
                    ),
                    8,
                ),

                round(
                    float(
                        xyz[
                            2
                        ]
                    ),
                    8,
                ),
            )


            coordinate_map[
                key
            ] = node_index


        # ====================================================
        # BUILD C3D8R CONNECTIVITY
        # ====================================================

        connectivity = []


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
                        ix
                        +
                        1
                    ]


                    y0 = y_values[
                        iy
                    ]

                    y1 = y_values[
                        iy
                        +
                        1
                    ]


                    z0 = z_values[
                        iz
                    ]

                    z1 = z_values[
                        iz
                        +
                        1
                    ]


                    # ----------------------------------------
                    # Standard 8-node hex ordering
                    # ----------------------------------------

                    points = [

                        (
                            x0,
                            y0,
                            z0,
                        ),

                        (
                            x1,
                            y0,
                            z0,
                        ),

                        (
                            x1,
                            y1,
                            z0,
                        ),

                        (
                            x0,
                            y1,
                            z0,
                        ),

                        (
                            x0,
                            y0,
                            z1,
                        ),

                        (
                            x1,
                            y0,
                            z1,
                        ),

                        (
                            x1,
                            y1,
                            z1,
                        ),

                        (
                            x0,
                            y1,
                            z1,
                        ),
                    ]


                    element = []


                    for point in points:

                        key = (

                            round(
                                float(
                                    point[
                                        0
                                    ]
                                ),
                                8,
                            ),

                            round(
                                float(
                                    point[
                                        1
                                    ]
                                ),
                                8,
                            ),

                            round(
                                float(
                                    point[
                                        2
                                    ]
                                ),
                                8,
                            ),
                        )


                        if key not in coordinate_map:

                            raise RuntimeError(

                                "Could not find mesh node "
                                "at coordinate {}".format(
                                    key
                                )
                            )


                        element.append(
                            coordinate_map[
                                key
                            ]
                        )


                    connectivity.append(
                        element
                    )


        connectivity = np.asarray(
            connectivity,
            dtype=np.int64,
        )


        self.connectivity = torch.tensor(

            connectivity,

            dtype=torch.long,

            device=device,
        )


        self.number_elements = (
            connectivity.shape[
                0
            ]
        )


        # ====================================================
        # ELEMENT SIZE
        # ====================================================

        dx_values = np.diff(
            x_values
        )


        dy_values = np.diff(
            y_values
        )


        dz_values = np.diff(
            z_values
        )


        if not np.allclose(
            dx_values,
            dx_values[
                0
            ],
        ):

            raise RuntimeError(
                "x spacing is not uniform."
            )


        if not np.allclose(
            dy_values,
            dy_values[
                0
            ],
        ):

            raise RuntimeError(
                "y spacing is not uniform."
            )


        if not np.allclose(
            dz_values,
            dz_values[
                0
            ],
        ):

            raise RuntimeError(
                "z spacing is not uniform."
            )


        self.dx = float(
            dx_values[
                0
            ]
        )


        self.dy = float(
            dy_values[
                0
            ]
        )


        self.dz = float(
            dz_values[
                0
            ]
        )


        self.element_volume = (

            self.dx
            *
            self.dy
            *
            self.dz
        )


        # ====================================================
        # C3D8R B MATRIX
        #
        # B converts:
        #
        # nodal displacement
        #
        # into
        #
        # strain
        #
        # epsilon = B u
        #
        # B size = 6 x 24
        # ====================================================

        signs = np.array(
            [

                [
                    -1.0,
                    -1.0,
                    -1.0,
                ],

                [
                    +1.0,
                    -1.0,
                    -1.0,
                ],

                [
                    +1.0,
                    +1.0,
                    -1.0,
                ],

                [
                    -1.0,
                    +1.0,
                    -1.0,
                ],

                [
                    -1.0,
                    -1.0,
                    +1.0,
                ],

                [
                    +1.0,
                    -1.0,
                    +1.0,
                ],

                [
                    +1.0,
                    +1.0,
                    +1.0,
                ],

                [
                    -1.0,
                    +1.0,
                    +1.0,
                ],
            ],
            dtype=np.float64,
        )


        gradients = np.zeros(
            (
                8,
                3,
            ),
            dtype=np.float64,
        )


        gradients[
            :,
            0
        ] = (

            signs[
                :,
                0
            ]

            /

            (
                4.0
                *
                self.dx
            )
        )


        gradients[
            :,
            1
        ] = (

            signs[
                :,
                1
            ]

            /

            (
                4.0
                *
                self.dy
            )
        )


        gradients[
            :,
            2
        ] = (

            signs[
                :,
                2
            ]

            /

            (
                4.0
                *
                self.dz
            )
        )


        B = np.zeros(
            (
                6,
                24,
            ),
            dtype=np.float64,
        )


        for node in range(
            8
        ):

            dNdx = gradients[
                node,
                0
            ]


            dNdy = gradients[
                node,
                1
            ]


            dNdz = gradients[
                node,
                2
            ]


            start = (
                3
                *
                node
            )


            # ----------------------------------------------
            # Normal strain exx
            # ----------------------------------------------

            B[
                0,
                start
            ] = dNdx


            # ----------------------------------------------
            # Normal strain eyy
            # ----------------------------------------------

            B[
                1,
                start
                +
                1
            ] = dNdy


            # ----------------------------------------------
            # Normal strain ezz
            # ----------------------------------------------

            B[
                2,
                start
                +
                2
            ] = dNdz


            # ----------------------------------------------
            # Engineering shear strain gamma_xy
            # ----------------------------------------------

            B[
                3,
                start
            ] = dNdy


            B[
                3,
                start
                +
                1
            ] = dNdx


            # ----------------------------------------------
            # Engineering shear strain gamma_xz
            # ----------------------------------------------

            B[
                4,
                start
            ] = dNdz


            B[
                4,
                start
                +
                2
            ] = dNdx


            # ----------------------------------------------
            # Engineering shear strain gamma_yz
            # ----------------------------------------------

            B[
                5,
                start
                +
                1
            ] = dNdz


            B[
                5,
                start
                +
                2
            ] = dNdy


        self.B = torch.tensor(

            B,

            dtype=torch.float32,

            device=device,
        )


        # ====================================================
        # GLOBAL DOF NUMBERS
        # ====================================================

        dof_indices = []


        for element in connectivity:

            element_dofs = []


            for node in element:

                element_dofs.extend(
                    [

                        3
                        *
                        node,

                        3
                        *
                        node
                        +
                        1,

                        3
                        *
                        node
                        +
                        2,
                    ]
                )


            dof_indices.append(
                element_dofs
            )


        self.dof_indices = torch.tensor(

            np.asarray(
                dof_indices,
                dtype=np.int64,
            ),

            dtype=torch.long,

            device=device,
        )


        # ====================================================
        # INTERIOR DOFS
        #
        # The two x-end surfaces have prescribed displacement.
        #
        # Therefore we only check FEM equilibrium inside
        # the Neural Operator region.
        # ====================================================

        x_min = np.min(
            coordinates[
                :,
                0
            ]
        )


        x_max = np.max(
            coordinates[
                :,
                0
            ]
        )


        free_nodes = (

            (
                np.abs(
                    coordinates[
                        :,
                        0
                    ]
                    -
                    x_min
                )
                >
                1.0e-6
            )

            &

            (
                np.abs(
                    coordinates[
                        :,
                        0
                    ]
                    -
                    x_max
                )
                >
                1.0e-6
            )
        )


        free_dofs = []


        for node_index, is_free in enumerate(
            free_nodes
        ):

            if is_free:

                free_dofs.extend(
                    [

                        3
                        *
                        node_index,

                        3
                        *
                        node_index
                        +
                        1,

                        3
                        *
                        node_index
                        +
                        2,
                    ]
                )


        self.free_dofs = torch.tensor(

            free_dofs,

            dtype=torch.long,

            device=device,
        )


        # ====================================================
        # INFORMATION
        # ====================================================

        print("")

        print(
            "StructuredHexPhysics created"
        )


        print(
            "Nodes:",
            self.number_nodes
        )


        print(
            "Elements:",
            self.number_elements
        )


        print(
            "Free DOFs:",
            len(
                free_dofs
            )
        )


        print(
            "Element size:",
            self.dx,
            self.dy,
            self.dz,
        )


    # ========================================================
    # ORTHOTROPIC MATERIAL MATRIX
    # ========================================================

    def material_matrix(
        self,
        material_parameters,
    ):

        # ----------------------------------------------------
        # Variable material properties
        # ----------------------------------------------------

        E1 = material_parameters[
            :,
            0
        ]


        E2 = material_parameters[
            :,
            1
        ]


        G12 = material_parameters[
            :,
            2
        ]


        # ----------------------------------------------------
        # Fixed material properties
        # ----------------------------------------------------

        E3 = torch.full_like(
            E1,
            12000.0,
        )


        nu12 = torch.full_like(
            E1,
            0.28,
        )


        nu13 = torch.full_like(
            E1,
            0.28,
        )


        nu23 = torch.full_like(
            E1,
            0.40,
        )


        G13 = torch.full_like(
            E1,
            4500.0,
        )


        G23 = torch.full_like(
            E1,
            3500.0,
        )


        batch_size = E1.shape[
            0
        ]


        # ====================================================
        # NORMAL COMPLIANCE MATRIX
        # ====================================================

        compliance = torch.zeros(

            (
                batch_size,
                3,
                3,
            ),

            dtype=torch.float32,

            device=self.device,
        )


        compliance[
            :,
            0,
            0
        ] = (
            1.0
            /
            E1
        )


        compliance[
            :,
            1,
            1
        ] = (
            1.0
            /
            E2
        )


        compliance[
            :,
            2,
            2
        ] = (
            1.0
            /
            E3
        )


        compliance[
            :,
            0,
            1
        ] = (
            -
            nu12
            /
            E1
        )


        compliance[
            :,
            1,
            0
        ] = compliance[
            :,
            0,
            1
        ]


        compliance[
            :,
            0,
            2
        ] = (
            -
            nu13
            /
            E1
        )


        compliance[
            :,
            2,
            0
        ] = compliance[
            :,
            0,
            2
        ]


        compliance[
            :,
            1,
            2
        ] = (
            -
            nu23
            /
            E2
        )


        compliance[
            :,
            2,
            1
        ] = compliance[
            :,
            1,
            2
        ]


        normal_stiffness = torch.linalg.inv(
            compliance
        )


        # ====================================================
        # COMPLETE 6x6 ORTHOTROPIC MATRIX
        # ====================================================

        C = torch.zeros(

            (
                batch_size,
                6,
                6,
            ),

            dtype=torch.float32,

            device=self.device,
        )


        C[
            :,
            0:3,
            0:3
        ] = normal_stiffness


        C[
            :,
            3,
            3
        ] = G12


        C[
            :,
            4,
            4
        ] = G13


        C[
            :,
            5,
            5
        ] = G23


        return C


    # ========================================================
    # CALCULATE INTERNAL NODAL FORCE
    # ========================================================

    def internal_force(
        self,
        displacement,
        material_parameters,
    ):

        # ----------------------------------------------------
        # displacement shape:
        #
        # batch x nodes x 3
        # ----------------------------------------------------

        batch_size = displacement.shape[
            0
        ]


        # ====================================================
        # GET THE 24 DISPLACEMENTS OF EACH ELEMENT
        # ====================================================

        element_displacement = displacement[
            :,
            self.connectivity,
            :
        ].reshape(

            batch_size,
            self.number_elements,
            24,
        )


        # ====================================================
        # STRAIN
        #
        # epsilon = B u
        #
        # B:
        # 6 x 24
        #
        # u:
        # batch x element x 24
        #
        # strain:
        # batch x element x 6
        # ====================================================

        strain = torch.einsum(

            "ij,bej->bei",

            self.B,

            element_displacement,
        )


        # ====================================================
        # MATERIAL MATRIX
        # ====================================================

        C = self.material_matrix(
            material_parameters
        )


        # ====================================================
        # STRESS
        #
        # sigma = C epsilon
        #
        # stress:
        # batch x element x 6
        # ====================================================

        stress = torch.einsum(

            "bij,bej->bei",

            C,

            strain,
        )


        # ====================================================
        # ELEMENT INTERNAL FORCE
        #
        # f_element = B^T sigma V
        #
        # THIS IS THE CORRECTED LINE.
        #
        # B is stored as:
        #
        # [6, 24]
        #
        # Therefore:
        #
        # i = 6
        # j = 24
        #
        # stress also uses i = 6.
        # ====================================================

        element_force = torch.einsum(

            "ij,bei->bej",

            self.B,

            stress,
        )


        element_force = (

            element_force

            *

            self.element_volume
        )


        # ====================================================
        # ASSEMBLE ELEMENT FORCES INTO GLOBAL NODAL FORCES
        # ====================================================

        global_force = torch.zeros(

            (
                batch_size,
                self.number_nodes
                *
                3,
            ),

            dtype=displacement.dtype,

            device=self.device,
        )


        indices = (

            self.dof_indices

            .reshape(
                1,
                -1,
            )

            .expand(
                batch_size,
                -1,
            )
        )


        global_force.scatter_add_(

            dim=1,

            index=
                indices,

            src=
                element_force.reshape(
                    batch_size,
                    -1,
                ),
        )


        return global_force


    # ========================================================
    # INTERIOR EQUILIBRIUM RESIDUAL
    # ========================================================

    def free_residual(
        self,
        displacement,
        material_parameters,
    ):

        global_force = self.internal_force(

            displacement,

            material_parameters,
        )


        free_force = global_force[
            :,
            self.free_dofs
        ]


        return free_force