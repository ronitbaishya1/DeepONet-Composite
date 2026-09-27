import torch


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_parameters(
    parameters_physical,
    normalization,
):

    mean = torch.as_tensor(
        normalization["parameter_mean"],
        dtype=parameters_physical.dtype,
        device=parameters_physical.device,
    )

    std = torch.as_tensor(
        normalization["parameter_std"],
        dtype=parameters_physical.dtype,
        device=parameters_physical.device,
    )

    return (
        parameters_physical
        - mean
    ) / std


def normalize_coordinates(
    coordinates_physical,
    normalization,
):

    coordinate_min = torch.as_tensor(
        normalization["coordinate_min"],
        dtype=coordinates_physical.dtype,
        device=coordinates_physical.device,
    )

    coordinate_max = torch.as_tensor(
        normalization["coordinate_max"],
        dtype=coordinates_physical.dtype,
        device=coordinates_physical.device,
    )

    return (
        2.0
        * (
            coordinates_physical
            - coordinate_min
        )
        / (
            coordinate_max
            - coordinate_min
        )
        - 1.0
    )


def coordinate_chain_rule_scale(
    normalization,
    dtype,
    device,
):

    coordinate_min = torch.as_tensor(
        normalization["coordinate_min"],
        dtype=dtype,
        device=device,
    )

    coordinate_max = torch.as_tensor(
        normalization["coordinate_max"],
        dtype=dtype,
        device=device,
    )

    # ========================================================
    # x_norm =
    # 2*(x-xmin)/(xmax-xmin)-1
    #
    # Therefore:
    #
    # dx_norm/dx =
    # 2/(xmax-xmin)
    # ========================================================

    return (
        2.0
        / (
            coordinate_max
            - coordinate_min
        )
    )


def denormalize_displacement(
    displacement_normalized,
    normalization,
):

    mean = torch.as_tensor(
        normalization["output_mean"],
        dtype=displacement_normalized.dtype,
        device=displacement_normalized.device,
    )

    std = torch.as_tensor(
        normalization["output_std"],
        dtype=displacement_normalized.dtype,
        device=displacement_normalized.device,
    )

    return (
        displacement_normalized
        * std
        + mean
    )


# ============================================================
# DISPLACEMENT + GRADIENT
# ============================================================

def displacement_and_gradient(
    model,
    parameters_physical,
    coordinates_physical,
    normalization,
    create_graph=False,
):

    # --------------------------------------------------------
    # Branch:
    # [E1,E2,G12]
    # --------------------------------------------------------

    parameter_normalized = (
        normalize_parameters(
            parameters_physical,
            normalization,
        )
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # Trunk coordinates need autograd.
    # --------------------------------------------------------

    coordinate_normalized = (
        normalize_coordinates(
            coordinates_physical,
            normalization,
        )
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # --------------------------------------------------------
    # Network prediction
    #
    # shape = [N,3]
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Chain-rule correction from normalized coordinate
    # derivatives to physical mm coordinates.
    # --------------------------------------------------------

    chain_scale = (
        coordinate_chain_rule_scale(
            normalization,
            dtype=coordinate_normalized.dtype,
            device=coordinate_normalized.device,
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
                coordinate_normalized,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )

        gradient_physical = (
            gradient_normalized
            * chain_scale
        )

        gradient_rows.append(
            gradient_physical
        )

    # --------------------------------------------------------
    # grad_u[n,i,j]
    #
    # i = displacement component
    # j = derivative direction
    #
    # Examples:
    #
    # grad_u[:,0,0] = dU1/dx
    # grad_u[:,0,1] = dU1/dy
    # grad_u[:,1,0] = dU2/dx
    # --------------------------------------------------------

    displacement_gradient = torch.stack(
        gradient_rows,
        dim=1,
    )

    return (
        displacement_physical,
        displacement_gradient,
    )


# ============================================================
# INFINITESIMAL STRAIN
# ============================================================

def infinitesimal_strain(
    displacement_gradient,
):

    return (
        0.5
        * (
            displacement_gradient
            + displacement_gradient.transpose(
                1,
                2,
            )
        )
    )


# ============================================================
# LEFT HENCKY / LOGARITHMIC STRAIN
#
# Abaqus LE is based on:
#
# LE = ln(V)
#
# where:
#
# B = F F^T
# V = sqrt(B)
#
# Hence:
#
# LE = 0.5 log(B)
#
# This is used only as a diagnostic comparison against Abaqus.
# ============================================================

def left_hencky_strain(
    displacement_gradient,
):

    number_points = (
        displacement_gradient.shape[0]
    )

    identity = torch.eye(
        3,
        dtype=displacement_gradient.dtype,
        device=displacement_gradient.device,
    )

    identity = identity.unsqueeze(
        0
    ).expand(
        number_points,
        -1,
        -1,
    )

    # --------------------------------------------------------
    # F = I + grad(u)
    # --------------------------------------------------------

    deformation_gradient = (
        identity
        + displacement_gradient
    )

    # --------------------------------------------------------
    # Left Cauchy-Green:
    #
    # B = F F^T
    # --------------------------------------------------------

    left_cauchy_green = (
        deformation_gradient
        @ deformation_gradient.transpose(
            1,
            2,
        )
    )

    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            left_cauchy_green
        )
    )

    eigenvalues = torch.clamp(
        eigenvalues,
        min=1.0e-12,
    )

    # --------------------------------------------------------
    # Principal logarithmic strains:
    #
    # 0.5 log(lambda_B)
    # --------------------------------------------------------

    log_principal = (
        0.5
        * torch.log(
            eigenvalues
        )
    )

    logarithmic_strain = (
        eigenvectors
        @ torch.diag_embed(
            log_principal
        )
        @ eigenvectors.transpose(
            1,
            2,
        )
    )

    return logarithmic_strain


# ============================================================
# TENSOR COMPONENTS
#
# Pure tensor convention:
#
# [e11,e22,e33,e12,e13,e23]
# ============================================================

def strain_tensor_to_tensor_voigt(
    strain,
):

    return torch.stack(
        [
            strain[:, 0, 0],
            strain[:, 1, 1],
            strain[:, 2, 2],

            strain[:, 0, 1],
            strain[:, 0, 2],
            strain[:, 1, 2],
        ],
        dim=1,
    )


# ============================================================
# ABAQUS STRAIN OUTPUT CONVENTION
#
# Abaqus reports shear strain as engineering shear:
#
# gamma12 = 2*e12
# gamma13 = 2*e13
# gamma23 = 2*e23
#
# Thus compare this function directly to:
#
# LE11,LE22,LE33,LE12,LE13,LE23
# ============================================================

def strain_tensor_to_abaqus_voigt(
    strain,
):

    return torch.stack(
        [
            strain[:, 0, 0],
            strain[:, 1, 1],
            strain[:, 2, 2],

            2.0 * strain[:, 0, 1],
            2.0 * strain[:, 0, 2],
            2.0 * strain[:, 1, 2],
        ],
        dim=1,
    )


# ============================================================
# ORTHOTROPIC STIFFNESS MATRIX
#
# Current orientation:
#
# 1 = X
# 2 = Y
# 3 = Z
#
# Strain vector:
#
# [e11,e22,e33,gamma12,gamma13,gamma23]
#
# Stress vector:
#
# [S11,S22,S33,S12,S13,S23]
# ============================================================

def orthotropic_stiffness_matrix(
    parameters_physical,
    E3=12000.0,
    nu12=0.28,
    nu13=0.28,
    nu23=0.40,
    G13=4500.0,
    G23=3500.0,
):

    dtype = parameters_physical.dtype
    device = parameters_physical.device

    E1 = parameters_physical[0]
    E2 = parameters_physical[1]
    G12 = parameters_physical[2]

    E3 = torch.as_tensor(
        E3,
        dtype=dtype,
        device=device,
    )

    nu12 = torch.as_tensor(
        nu12,
        dtype=dtype,
        device=device,
    )

    nu13 = torch.as_tensor(
        nu13,
        dtype=dtype,
        device=device,
    )

    nu23 = torch.as_tensor(
        nu23,
        dtype=dtype,
        device=device,
    )

    G13 = torch.as_tensor(
        G13,
        dtype=dtype,
        device=device,
    )

    G23 = torch.as_tensor(
        G23,
        dtype=dtype,
        device=device,
    )

    compliance = torch.zeros(
        (6, 6),
        dtype=dtype,
        device=device,
    )

    # --------------------------------------------------------
    # Normal terms
    # --------------------------------------------------------

    compliance[0, 0] = (
        1.0 / E1
    )

    compliance[1, 1] = (
        1.0 / E2
    )

    compliance[2, 2] = (
        1.0 / E3
    )

    # --------------------------------------------------------
    # Poisson coupling
    #
    # Symmetric compliance automatically enforces
    # reciprocal relationships.
    # --------------------------------------------------------

    compliance[0, 1] = (
        -nu12 / E1
    )

    compliance[1, 0] = (
        compliance[0, 1]
    )

    compliance[0, 2] = (
        -nu13 / E1
    )

    compliance[2, 0] = (
        compliance[0, 2]
    )

    compliance[1, 2] = (
        -nu23 / E2
    )

    compliance[2, 1] = (
        compliance[1, 2]
    )

    # --------------------------------------------------------
    # Shear terms
    # --------------------------------------------------------

    compliance[3, 3] = (
        1.0 / G12
    )

    compliance[4, 4] = (
        1.0 / G13
    )

    compliance[5, 5] = (
        1.0 / G23
    )

    stiffness = torch.linalg.inv(
        compliance
    )

    return stiffness


# ============================================================
# STRESS FROM SMALL STRAIN
# ============================================================

def stress_from_small_strain(
    strain_tensor,
    parameters_physical,
):

    # --------------------------------------------------------
    # Constitutive matrix uses engineering shear strains.
    # --------------------------------------------------------

    engineering_strain = (
        strain_tensor_to_abaqus_voigt(
            strain_tensor
        )
    )

    stiffness = (
        orthotropic_stiffness_matrix(
            parameters_physical
        )
    )

    stress = (
        engineering_strain
        @ stiffness.T
    )

    return stress