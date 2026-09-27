import torch
import torch.nn as nn


# ============================================================
# BASIC MLP
# ============================================================

class MLP(nn.Module):

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
# ONE SCALAR DEEPONET
# ============================================================

class ScalarDeepONet(nn.Module):

    def __init__(
        self,
        branch_dim,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        self.branch_network = MLP(

            input_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            output_dim=
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
                latent_dim,

            depth=
                depth,
        )


        self.bias = nn.Parameter(
            torch.zeros(
                1
            )
        )


    def forward(
        self,
        branch,
        coordinates,
    ):

        branch_features = self.branch_network(
            branch
        )


        trunk_features = self.trunk_network(
            coordinates
        )


        output = torch.einsum(

            "bp,np->bn",

            branch_features,

            trunk_features,
        )


        output = (
            output
            +
            self.bias
        )


        return output


# ============================================================
# TRACTION/FORCE-AWARE LOCAL OPERATOR
# ============================================================

class HybridBulkOperatorV3(nn.Module):

    def __init__(
        self,
        branch_dim,
        number_force_coefficients=16,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        self.branch_dim = branch_dim

        self.number_force_coefficients = (
            number_force_coefficients
        )


        # ----------------------------------------------------
        # THREE DISPLACEMENT COMPONENTS
        # ----------------------------------------------------

        self.u1_operator = ScalarDeepONet(

            branch_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            latent_dim=
                latent_dim,

            depth=
                depth,
        )


        self.u2_operator = ScalarDeepONet(

            branch_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            latent_dim=
                latent_dim,

            depth=
                depth,
        )


        self.u3_operator = ScalarDeepONet(

            branch_dim=
                branch_dim,

            hidden_dim=
                hidden_dim,

            latent_dim=
                latent_dim,

            depth=
                depth,
        )


        # ----------------------------------------------------
        # INTERFACE FORCE PCA COEFFICIENTS
        # ----------------------------------------------------

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
        coordinates,
    ):

        u1 = self.u1_operator(
            branch,
            coordinates,
        )


        u2 = self.u2_operator(
            branch,
            coordinates,
        )


        u3 = self.u3_operator(
            branch,
            coordinates,
        )


        displacement = torch.stack(
            [
                u1,
                u2,
                u3,
            ],
            dim=-1,
        )


        force_coefficients = (
            self.force_coefficient_head(
                branch
            )
        )


        return (
            displacement,
            force_coefficients,
        )