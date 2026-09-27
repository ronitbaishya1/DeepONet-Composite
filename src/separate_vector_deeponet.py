import torch
import torch.nn as nn


# ============================================================
# BASIC MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
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
                nn.Tanh()
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

        return self.network(x)


# ============================================================
# ONE SCALAR DEEPONET
# ============================================================

class ScalarDeepONet(nn.Module):

    def __init__(
        self,
        branch_dim,
        trunk_dim,
        latent_dim=128,
        hidden_dims=(128, 128, 128),
    ):

        super().__init__()

        self.branch = MLP(
            branch_dim,
            hidden_dims,
            latent_dim,
        )

        self.trunk = MLP(
            trunk_dim,
            hidden_dims,
            latent_dim,
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )


    def forward(
        self,
        branch_input,
        trunk_input,
    ):

        branch_features = self.branch(
            branch_input
        )

        # ----------------------------------------------------
        # COMMON TRUNK COORDINATES
        #
        # branch: [B, P]
        # trunk : [N, 3]
        # output: [B, N]
        # ----------------------------------------------------

        if trunk_input.ndim == 2:

            trunk_features = self.trunk(
                trunk_input
            )

            output = torch.matmul(
                branch_features,
                trunk_features.T,
            )

            return (
                output
                + self.bias
            )


        # ----------------------------------------------------
        # CASE-SPECIFIC TRUNK COORDINATES
        #
        # branch: [B, P]
        # trunk : [B, N, 3]
        # output: [B, N]
        #
        # This will be useful later when L/h varies.
        # ----------------------------------------------------

        if trunk_input.ndim == 3:

            B, N, D = trunk_input.shape

            trunk_flat = trunk_input.reshape(
                B * N,
                D,
            )

            trunk_features = self.trunk(
                trunk_flat
            )

            trunk_features = trunk_features.reshape(
                B,
                N,
                -1,
            )

            output = torch.einsum(
                "bq,bnq->bn",
                branch_features,
                trunk_features,
            )

            return (
                output
                + self.bias
            )


        raise ValueError(
            "trunk_input must have shape "
            "[N,D] or [B,N,D]"
        )


# ============================================================
# THREE INDEPENDENT DISPLACEMENT OPERATORS
# ============================================================

class SeparateVectorDeepONet(nn.Module):

    def __init__(
        self,
        branch_dim=3,
        trunk_dim=3,
        latent_dim=128,
        hidden_dims=(128, 128, 128),
    ):

        super().__init__()

        self.branch_dim = branch_dim

        self.U1_operator = ScalarDeepONet(
            branch_dim,
            trunk_dim,
            latent_dim,
            hidden_dims,
        )

        self.U2_operator = ScalarDeepONet(
            branch_dim,
            trunk_dim,
            latent_dim,
            hidden_dims,
        )

        self.U3_operator = ScalarDeepONet(
            branch_dim,
            trunk_dim,
            latent_dim,
            hidden_dims,
        )


    def forward(
        self,
        branch_input,
        trunk_input,
    ):

        U1 = self.U1_operator(
            branch_input,
            trunk_input,
        )

        U2 = self.U2_operator(
            branch_input,
            trunk_input,
        )

        U3 = self.U3_operator(
            branch_input,
            trunk_input,
        )

        return torch.stack(
            [U1, U2, U3],
            dim=-1,
        )