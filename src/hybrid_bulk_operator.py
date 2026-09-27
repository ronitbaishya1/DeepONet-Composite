import torch
import torch.nn as nn


# ============================================================
# GENERIC MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.Tanh,
    ):

        super().__init__()


        layers = []

        previous_dim = input_dim


        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )


            layers.append(
                activation()
            )


            previous_dim = hidden_dim


        layers.append(
            nn.Linear(
                previous_dim,
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
# SCALAR DEEPONET
# ============================================================

class ScalarDeepONet(nn.Module):

    def __init__(
        self,
        branch_dim,
        trunk_dim=3,
        latent_dim=128,
        hidden_dim=128,
    ):

        super().__init__()


        self.branch_net = MLP(
            input_dim=branch_dim,
            hidden_dims=[
                hidden_dim,
                hidden_dim,
                hidden_dim,
            ],
            output_dim=latent_dim,
        )


        self.trunk_net = MLP(
            input_dim=trunk_dim,
            hidden_dims=[
                hidden_dim,
                hidden_dim,
                hidden_dim,
            ],
            output_dim=latent_dim,
        )


        self.bias = nn.Parameter(
            torch.zeros(
                1
            )
        )


    def forward(
        self,
        branch_input,
        coordinates,
    ):

        branch_features = self.branch_net(
            branch_input
        )


        trunk_features = self.trunk_net(
            coordinates
        )


        output = torch.einsum(
            "bi,ni->bn",
            branch_features,
            trunk_features,
        )


        return (
            output
            +
            self.bias
        )


# ============================================================
# HYBRID BULK OPERATOR
#
# Inputs:
#
# branch =
# [
#   E1,
#   E2,
#   G12,
#   c_left...,
#   c_right...
# ]
#
# trunk =
# [x,y,z]
#
# Outputs:
#
# U1,U2,U3 at arbitrary coordinates
#
# plus
#
# generalized interface forces
# [g_left, g_right]
# ============================================================

class HybridBulkOperator(nn.Module):

    def __init__(
        self,
        branch_dim,
        number_force_outputs,
        latent_dim=128,
        hidden_dim=128,
    ):

        super().__init__()


        self.branch_dim = (
            branch_dim
        )


        self.number_force_outputs = (
            number_force_outputs
        )


        # ----------------------------------------------------
        # Three separate displacement operators
        # ----------------------------------------------------

        self.u1_operator = ScalarDeepONet(
            branch_dim=branch_dim,
            trunk_dim=3,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )


        self.u2_operator = ScalarDeepONet(
            branch_dim=branch_dim,
            trunk_dim=3,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )


        self.u3_operator = ScalarDeepONet(
            branch_dim=branch_dim,
            trunk_dim=3,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )


        # ----------------------------------------------------
        # Generalized force / Dirichlet-to-Neumann head
        # ----------------------------------------------------

        self.force_head = MLP(
            input_dim=branch_dim,
            hidden_dims=[
                hidden_dim,
                hidden_dim,
                hidden_dim,
            ],
            output_dim=number_force_outputs,
        )


    def forward(
        self,
        branch_input,
        coordinates,
    ):

        U1 = self.u1_operator(
            branch_input,
            coordinates,
        )


        U2 = self.u2_operator(
            branch_input,
            coordinates,
        )


        U3 = self.u3_operator(
            branch_input,
            coordinates,
        )


        displacement = torch.stack(
            [
                U1,
                U2,
                U3,
            ],
            dim=-1,
        )


        generalized_force = self.force_head(
            branch_input
        )


        return (
            displacement,
            generalized_force,
        )