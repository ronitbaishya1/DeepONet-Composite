import numpy as np
import torch

from src.orthotropic_mechanics import (
    normalize_parameters,
    normalize_coordinates,
    coordinate_chain_rule_scale,
    denormalize_displacement,
    infinitesimal_strain,
    strain_tensor_to_abaqus_voigt,
    orthotropic_stiffness_matrix,
)


# ============================================================
# MECHANICS SCALES
# ============================================================

def calculate_mechanics_scales(
    LE_tensor,
    S_tensor,
    train_indices,
):
    """
    Component-wise scales using TRAINING CASES ONLY.

    LE_tensor : [cases, IPs, 6]
    S_tensor  : [cases, IPs, 6]
    """

    LE_train = LE_tensor[
        train_indices
    ].reshape(
        -1,
        6,
    )

    S_train = S_tensor[
        train_indices
    ].reshape(
        -1,
        6,
    )

    strain_scale = LE_train.std(
        axis=0
    )

    stress_scale = S_train.std(
        axis=0
    )

    # Prevent tiny components from receiving enormous weights.
    strain_scale = np.maximum(
        strain_scale,
        1.0e-5,
    )

    stress_scale = np.maximum(
        stress_scale,
        1.0,
    )

    return (
        strain_scale.astype(
            np.float32
        ),
        stress_scale.astype(
            np.float32
        ),
    )


# ============================================================
# DERIVED MECHANICS AT QUERY POINTS
# ============================================================

def derived_mechanics_single(
    model,
    parameter_physical,
    coordinates_physical,
    normalization,
    create_graph,
):
    """
    Returns:
        displacement [N,3] mm
        strain       [N,6] Abaqus-style engineering shear
        stress       [N,6] MPa
    """

    parameter_normalized = (
        normalize_parameters(
            parameter_physical,
            normalization,
        )
        .unsqueeze(0)
    )

    coordinates_normalized = (
        normalize_coordinates(
            coordinates_physical,
            normalization,
        )
        .clone()
        .detach()
        .requires_grad_(True)
    )

    displacement_normalized = model(
        parameter_normalized,
        coordinates_normalized,
    )[0]

    displacement_physical = (
        denormalize_displacement(
            displacement_normalized,
            normalization,
        )
    )

    chain_scale = (
        coordinate_chain_rule_scale(
            normalization,
            dtype=coordinates_normalized.dtype,
            device=coordinates_normalized.device,
        )
    )

    gradient_rows = []

    for component in range(3):

        gradient_normalized = (
            torch.autograd.grad(
                displacement_physical[
                    :,
                    component,
                ].sum(),
                coordinates_normalized,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )

        gradient_rows.append(
            gradient_normalized
            * chain_scale
        )

    displacement_gradient = torch.stack(
        gradient_rows,
        dim=1,
    )

    strain_tensor = (
        infinitesimal_strain(
            displacement_gradient
        )
    )

    strain_voigt = (
        strain_tensor_to_abaqus_voigt(
            strain_tensor
        )
    )

    stiffness = (
        orthotropic_stiffness_matrix(
            parameter_physical
        )
    )

    stress_voigt = (
        strain_voigt
        @ stiffness.T
    )

    return (
        displacement_physical,
        strain_voigt,
        stress_voigt,
    )


# ============================================================
# BATCH MECHANICS SUPERVISION LOSS
# ============================================================

def mechanics_supervision_loss(
    model,
    parameters_physical,
    coordinates_physical,
    strain_target,
    stress_target,
    normalization,
    strain_scale,
    stress_scale,
    component_indices,
    create_graph=True,
):
    """
    parameters_physical: [B,3]

    coordinates_physical:
        [N,3]

    strain_target:
        [B,N,6]

    stress_target:
        [B,N,6]
    """

    strain_losses = []
    stress_losses = []

    strain_scale = torch.as_tensor(
        strain_scale,
        dtype=parameters_physical.dtype,
        device=parameters_physical.device,
    )

    stress_scale = torch.as_tensor(
        stress_scale,
        dtype=parameters_physical.dtype,
        device=parameters_physical.device,
    )

    component_indices = torch.as_tensor(
        component_indices,
        dtype=torch.long,
        device=parameters_physical.device,
    )

    for b in range(
        parameters_physical.shape[0]
    ):

        (
            _,
            strain_prediction,
            stress_prediction,
        ) = derived_mechanics_single(
            model=model,
            parameter_physical=parameters_physical[
                b
            ],
            coordinates_physical=coordinates_physical,
            normalization=normalization,
            create_graph=create_graph,
        )

        strain_difference = (
            strain_prediction[
                :,
                component_indices,
            ]
            -
            strain_target[
                b,
                :,
                component_indices,
            ]
        )

        stress_difference = (
            stress_prediction[
                :,
                component_indices,
            ]
            -
            stress_target[
                b,
                :,
                component_indices,
            ]
        )

        normalized_strain_difference = (
            strain_difference
            /
            strain_scale[
                component_indices
            ]
        )

        normalized_stress_difference = (
            stress_difference
            /
            stress_scale[
                component_indices
            ]
        )

        strain_losses.append(
            torch.mean(
                normalized_strain_difference
                ** 2
            )
        )

        stress_losses.append(
            torch.mean(
                normalized_stress_difference
                ** 2
            )
        )

    return (
        torch.stack(
            strain_losses
        ).mean(),
        torch.stack(
            stress_losses
        ).mean(),
    )


# ============================================================
# CONTACT-AWARE INTERIOR COLLOCATION
# ============================================================

def sample_contact_aware_interior(
    nodal_coordinates,
    number_points,
    device,
    rng,
):
    """
    Samples physical points inside the specimen while avoiding
    the sharp loading-nose and support neighborhoods.

    Current specimen:
        x approx [-12,12]
        y approx [0,4]
        z approx [0,8]

    Contacts:
        nose    x ~ 0,   top
        support x ~ -8, bottom
        support x ~ +8, bottom
    """

    minimum = nodal_coordinates.min(
        axis=0
    )

    maximum = nodal_coordinates.max(
        axis=0
    )

    # Stay away from external surfaces.
    lower = minimum + np.array(
        [
            0.25,
            0.20,
            0.25,
        ],
        dtype=np.float32,
    )

    upper = maximum - np.array(
        [
            0.25,
            0.20,
            0.25,
        ],
        dtype=np.float32,
    )

    accepted = []

    while len(
        accepted
    ) < number_points:

        candidates = rng.uniform(
            lower,
            upper,
            size=(
                number_points * 3,
                3,
            ),
        )

        x = candidates[
            :,
            0
        ]

        y = candidates[
            :,
            1
        ]

        ymax = maximum[
            1
        ]

        ymin = minimum[
            1
        ]

        # Nose contact neighborhood.
        nose_region = (
            np.abs(
                x
            ) < 3.5
        ) & (
            y > (
                ymax
                - 1.0
            )
        )

        # Left support neighborhood.
        left_support_region = (
            np.abs(
                x + 8.0
            ) < 3.5
        ) & (
            y < (
                ymin
                + 1.0
            )
        )

        # Right support neighborhood.
        right_support_region = (
            np.abs(
                x - 8.0
            ) < 3.5
        ) & (
            y < (
                ymin
                + 1.0
            )
        )

        valid = ~(
            nose_region
            |
            left_support_region
            |
            right_support_region
        )

        valid_points = candidates[
            valid
        ]

        for point in valid_points:

            accepted.append(
                point
            )

            if len(
                accepted
            ) >= number_points:

                break

    accepted = np.asarray(
        accepted[
            :number_points
        ],
        dtype=np.float32,
    )

    return torch.tensor(
        accepted,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# EQUILIBRIUM LOSS FOR ONE CASE
# ============================================================

def equilibrium_loss_single(
    model,
    parameter_physical,
    coordinates_physical,
    normalization,
    residual_reference=562.5,
    create_graph_second=True,
):
    """
    Static strong-form equilibrium:

        div(sigma) = 0

    residual_reference:
        approximate MPa/mm scaling used to nondimensionalize.
    """

    parameter_normalized = (
        normalize_parameters(
            parameter_physical,
            normalization,
        )
        .unsqueeze(0)
    )

    coordinate_normalized = (
        normalize_coordinates(
            coordinates_physical,
            normalization,
        )
        .clone()
        .detach()
        .requires_grad_(True)
    )

    displacement_normalized = model(
        parameter_normalized,
        coordinate_normalized,
    )[0]

    displacement_physical = (
        denormalize_displacement(
            displacement_normalized,
            normalization,
        )
    )

    chain_scale = (
        coordinate_chain_rule_scale(
            normalization,
            dtype=coordinate_normalized.dtype,
            device=coordinate_normalized.device,
        )
    )

    gradient_rows = []

    # First derivatives must retain a graph because
    # equilibrium needs second derivatives.
    for component in range(3):

        gradient_normalized = (
            torch.autograd.grad(
                displacement_physical[
                    :,
                    component,
                ].sum(),
                coordinate_normalized,
                create_graph=True,
                retain_graph=True,
            )[0]
        )

        gradient_rows.append(
            gradient_normalized
            * chain_scale
        )

    gradient = torch.stack(
        gradient_rows,
        dim=1,
    )

    strain = 0.5 * (
        gradient
        +
        gradient.transpose(
            1,
            2,
        )
    )

    strain_engineering = torch.stack(
        [
            strain[
                :,
                0,
                0,
            ],

            strain[
                :,
                1,
                1,
            ],

            strain[
                :,
                2,
                2,
            ],

            2.0
            * strain[
                :,
                0,
                1,
            ],

            2.0
            * strain[
                :,
                0,
                2,
            ],

            2.0
            * strain[
                :,
                1,
                2,
            ],
        ],
        dim=1,
    )

    stiffness = (
        orthotropic_stiffness_matrix(
            parameter_physical
        )
    )

    stress = (
        strain_engineering
        @ stiffness.T
    )

    stress_gradients = []

    for component in range(6):

        gradient_normalized = (
            torch.autograd.grad(
                stress[
                    :,
                    component,
                ].sum(),
                coordinate_normalized,
                create_graph=create_graph_second,
                retain_graph=True,
            )[0]
        )

        stress_gradients.append(
            gradient_normalized
            * chain_scale
        )

    (
        grad_S11,
        grad_S22,
        grad_S33,
        grad_S12,
        grad_S13,
        grad_S23,
    ) = stress_gradients

    residual_x = (
        grad_S11[
            :,
            0
        ]
        +
        grad_S12[
            :,
            1
        ]
        +
        grad_S13[
            :,
            2
        ]
    )

    residual_y = (
        grad_S12[
            :,
            0
        ]
        +
        grad_S22[
            :,
            1
        ]
        +
        grad_S23[
            :,
            2
        ]
    )

    residual_z = (
        grad_S13[
            :,
            0
        ]
        +
        grad_S23[
            :,
            1
        ]
        +
        grad_S33[
            :,
            2
        ]
    )

    residual = torch.stack(
        [
            residual_x,
            residual_y,
            residual_z,
        ],
        dim=1,
    )

    residual_normalized = (
        residual
        /
        residual_reference
    )

    return torch.mean(
        residual_normalized
        ** 2
    )