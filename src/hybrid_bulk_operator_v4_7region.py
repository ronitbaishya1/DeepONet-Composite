import torch
import torch.nn as nn


# ============================================================
# MLP
# ============================================================

class MLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth=4,
    ):

        super().__init__()


        layers = []


        current_dim = input_dim


        for _ in range(
            depth - 1
        ):

            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_dim,
                )
            )


            layers.append(
                nn.GELU()
            )


            current_dim = hidden_dim


        layers.append(
            nn.Linear(
                current_dim,
                output_dim,
            )
        )


        self.network = nn.Sequential(
            *layers
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# VECTOR DEEPONET
# ============================================================

class VectorDeepONet(
    nn.Module
):

    def __init__(
        self,
        branch_dim,
        output_dim,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        self.output_dim = (
            output_dim
        )


        self.latent_dim = (
            latent_dim
        )


        self.branch_network = MLP(

            input_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            output_dim=
                output_dim
                *
                latent_dim,

            depth=
                depth,
        )


        self.trunk_network = MLP(

            input_dim=
                3,

            hidden_dim=
                hidden_dim,

            output_dim=
                output_dim
                *
                latent_dim,

            depth=
                depth,
        )


        self.bias = nn.Parameter(
            torch.zeros(
                output_dim
            )
        )


    def forward(
        self,
        branch,
        coordinates,
    ):

        batch_size = branch.shape[
            0
        ]


        number_points = coordinates.shape[
            0
        ]


        branch_features = (
            self.branch_network(
                branch
            )
            .reshape(
                batch_size,
                self.output_dim,
                self.latent_dim,
            )
        )


        trunk_features = (
            self.trunk_network(
                coordinates
            )
            .reshape(
                number_points,
                self.output_dim,
                self.latent_dim,
            )
        )


        output = torch.einsum(

            "bop,nop->bno",

            branch_features,

            trunk_features,
        )


        output = (
            output
            +
            self.bias.reshape(
                1,
                1,
                -1,
            )
        )


        return output


# ============================================================
# V4 MECHANICS OPERATOR
# ============================================================

class HybridBulkOperatorV4SevenRegion(
    nn.Module
):

    def __init__(
        self,
        branch_dim,
        number_force_coefficients,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        self.branch_dim = (
            branch_dim
        )


        self.number_force_coefficients = (
            number_force_coefficients
        )


        # ====================================================
        # NODAL DISPLACEMENT
        # ====================================================

        self.displacement_operator = (
            VectorDeepONet(

                branch_dim=
                    branch_dim,

                output_dim=
                    3,

                hidden_dim=
                    hidden_dim,

                latent_dim=
                    latent_dim,

                depth=
                    depth,
            )
        )


        # ====================================================
        # ELEMENT/IP STRAIN
        # ====================================================

        self.strain_operator = (
            VectorDeepONet(

                branch_dim=
                    branch_dim,

                output_dim=
                    6,

                hidden_dim=
                    hidden_dim,

                latent_dim=
                    latent_dim,

                depth=
                    depth,
            )
        )


        # ====================================================
        # ELEMENT/IP STRESS
        # ====================================================

        self.stress_operator = (
            VectorDeepONet(

                branch_dim=
                    branch_dim,

                output_dim=
                    6,

                hidden_dim=
                    hidden_dim,

                latent_dim=
                    latent_dim,

                depth=
                    depth,
            )
        )


        # ====================================================
        # FORCE PCA COEFFICIENTS
        # ====================================================

        self.force_coefficient_head = MLP(

            input_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            output_dim=
                number_force_coefficients,

            depth=
                depth,
        )


    def forward(

        self,

        branch,

        node_coordinates,

        ip_coordinates,
    ):

        displacement = (
            self.displacement_operator(

                branch,

                node_coordinates,
            )
        )


        strain = (
            self.strain_operator(

                branch,

                ip_coordinates,
            )
        )


        stress = (
            self.stress_operator(

                branch,

                ip_coordinates,
            )
        )


        force_coefficients = (
            self.force_coefficient_head(
                branch
            )
        )


        return (
            displacement,
            strain,
            stress,
            force_coefficients,
        )