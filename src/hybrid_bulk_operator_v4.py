import torch
import torch.nn as nn

from src.hybrid_bulk_operator_v3 import (
    MLP,
    ScalarDeepONet,
)


# ============================================================
# VECTOR DEEPONET
# ============================================================

class VectorDeepONet(nn.Module):

    def __init__(
        self,
        branch_dim,
        number_components,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        self.number_components = (
            number_components
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
                latent_dim,
            depth=
                depth,
        )


        self.trunk_network = MLP(
            input_dim=3,
            hidden_dim=
                hidden_dim,
            output_dim=
                number_components
                *
                latent_dim,
            depth=
                depth,
        )


        self.bias = nn.Parameter(
            torch.zeros(
                number_components
            )
        )


    def forward(
        self,
        branch,
        coordinates,
    ):

        branch_features = (
            self.branch_network(
                branch
            )
        )


        trunk_features = (
            self.trunk_network(
                coordinates
            )
        )


        trunk_features = (
            trunk_features.reshape(
                coordinates.shape[0],
                self.number_components,
                self.latent_dim,
            )
        )


        output = torch.einsum(
            "bp,ncp->bnc",
            branch_features,
            trunk_features,
        )


        return (
            output
            +
            self.bias.reshape(
                1,
                1,
                -1,
            )
        )


# ============================================================
# V4
#
# Predicts:
#
# 1. displacement U
# 2. strain LE directly
# 3. stress S directly
# 4. interface force PCA coefficients
# ============================================================

class HybridBulkOperatorV4(nn.Module):

    def __init__(
        self,
        branch_dim,
        number_force_coefficients=16,
        hidden_dim=128,
        latent_dim=128,
        depth=4,
    ):

        super().__init__()


        # ----------------------------------------------------
        # KEEP V3 NAMES EXACTLY
        #
        # This lets us transfer the successful Step-40 weights.
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


        # ----------------------------------------------------
        # NEW DIRECT MECHANICS OPERATORS
        # ----------------------------------------------------

        self.strain_operator = VectorDeepONet(
            branch_dim=
                branch_dim,
            number_components=6,
            hidden_dim=
                hidden_dim,
            latent_dim=
                latent_dim,
            depth=
                depth,
        )


        self.stress_operator = VectorDeepONet(
            branch_dim=
                branch_dim,
            number_components=6,
            hidden_dim=
                hidden_dim,
            latent_dim=
                latent_dim,
            depth=
                depth,
        )


    def forward(
        self,
        branch,
        nodal_coordinates,
        ip_coordinates,
    ):

        u1 = self.u1_operator(
            branch,
            nodal_coordinates,
        )


        u2 = self.u2_operator(
            branch,
            nodal_coordinates,
        )


        u3 = self.u3_operator(
            branch,
            nodal_coordinates,
        )


        displacement = torch.stack(
            [
                u1,
                u2,
                u3,
            ],
            dim=-1,
        )


        strain = self.strain_operator(
            branch,
            ip_coordinates,
        )


        stress = self.stress_operator(
            branch,
            ip_coordinates,
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