import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VectorDeepONet(nn.Module):
    """
    Branch input : [E1, E2, G12]  -> normalized
    Trunk input  : [x, y, z]      -> normalized
    Output       : [U1, U2, U3]   -> normalized

    One shared trunk basis is used, while the branch generates
    component-specific coefficients for U1/U2/U3.
    """
    def __init__(
        self,
        branch_dim=3,
        trunk_dim=3,
        hidden=(128, 128, 128),
        latent_dim=128,
        n_components=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components

        self.branch = MLP(
            branch_dim,
            list(hidden),
            n_components * latent_dim,
        )
        self.trunk = MLP(
            trunk_dim,
            list(hidden),
            latent_dim,
        )
        self.bias = nn.Parameter(torch.zeros(n_components))

    def forward(self, branch_input, trunk_input):
        # branch_input: [B, 3]
        # trunk_input : [N, 3]

        B = branch_input.shape[0]

        branch_features = self.branch(branch_input)
        branch_features = branch_features.view(
            B,
            self.n_components,
            self.latent_dim,
        )  # [B, C, Q]

        trunk_features = self.trunk(trunk_input)  # [N, Q]

        # [B, N, C]
        out = torch.einsum(
            "bcq,nq->bnc",
            branch_features,
            trunk_features,
        )

        return out + self.bias.view(1, 1, -1)
