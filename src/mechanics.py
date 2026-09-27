import torch


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_parameters(p_phys, norm):
    mean = torch.as_tensor(
        norm["parameter_mean"],
        dtype=p_phys.dtype,
        device=p_phys.device,
    )
    std = torch.as_tensor(
        norm["parameter_std"],
        dtype=p_phys.dtype,
        device=p_phys.device,
    )

    return (p_phys - mean) / std


def normalize_coordinates(x_phys, norm):
    x_min = torch.as_tensor(
        norm["coordinate_min"],
        dtype=x_phys.dtype,
        device=x_phys.device,
    )
    x_max = torch.as_tensor(
        norm["coordinate_max"],
        dtype=x_phys.dtype,
        device=x_phys.device,
    )

    return 2.0 * (
        x_phys - x_min
    ) / (
        x_max - x_min
    ) - 1.0


def physical_coordinate_scale(norm, dtype, device):
    x_min = torch.as_tensor(
        norm["coordinate_min"],
        dtype=dtype,
        device=device,
    )
    x_max = torch.as_tensor(
        norm["coordinate_max"],
        dtype=dtype,
        device=device,
    )

    # dx_norm/dx_phys
    return 2.0 / (x_max - x_min)


def denormalize_displacement(u_norm, norm):
    mean = torch.as_tensor(
        norm["output_mean"],
        dtype=u_norm.dtype,
        device=u_norm.device,
    )
    std = torch.as_tensor(
        norm["output_std"],
        dtype=u_norm.dtype,
        device=u_norm.device,
    )

    return u_norm * std + mean


# ============================================================
# DISPLACEMENT GRADIENT
# ============================================================

def displacement_and_gradient_single(
    model,
    p_phys,
    x_phys,
    norm,
    create_graph=False,
):
    """
    p_phys : [3]
    x_phys : [N,3]

    Returns
    -------
    u_phys : [N,3]
    grad_u : [N,3,3]

    grad_u[n,i,j] = d u_i / d x_j
    """

    p_norm = normalize_parameters(
        p_phys,
        norm,
    ).unsqueeze(0)

    x_norm = normalize_coordinates(
        x_phys,
        norm,
    ).clone().detach().requires_grad_(True)

    u_norm = model(
        p_norm,
        x_norm,
    )[0]

    u_phys = denormalize_displacement(
        u_norm,
        norm,
    )

    coord_scale = physical_coordinate_scale(
        norm,
        dtype=x_norm.dtype,
        device=x_norm.device,
    )

    grads = []

    for component in range(3):

        g_norm = torch.autograd.grad(
            u_phys[:, component].sum(),
            x_norm,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        # Chain rule:
        # du/dx_phys = du/dx_norm * dx_norm/dx_phys
        g_phys = g_norm * coord_scale

        grads.append(g_phys)

    grad_u = torch.stack(
        grads,
        dim=1,
    )

    return u_phys, grad_u


# ============================================================
# SMALL STRAIN
# ============================================================

def infinitesimal_strain_tensor(grad_u):
    """
    epsilon = 0.5*(grad_u + grad_u^T)

    Returns [N,3,3]
    """
    return 0.5 * (
        grad_u
        + grad_u.transpose(1, 2)
    )


def strain_tensor_to_voigt_tensor_shear(eps):
    """
    Tensor shear convention:
    [e11,e22,e33,e12,e13,e23]
    """
    return torch.stack(
        [
            eps[:, 0, 0],
            eps[:, 1, 1],
            eps[:, 2, 2],
            eps[:, 0, 1],
            eps[:, 0, 2],
            eps[:, 1, 2],
        ],
        dim=1,
    )


def strain_tensor_to_voigt_engineering_shear(eps):
    """
    Engineering shear convention for constitutive matrix:
    [e11,e22,e33,gamma12,gamma13,gamma23]
    where gamma_ij = 2*e_ij.
    """
    return torch.stack(
        [
            eps[:, 0, 0],
            eps[:, 1, 1],
            eps[:, 2, 2],
            2.0 * eps[:, 0, 1],
            2.0 * eps[:, 0, 2],
            2.0 * eps[:, 1, 2],
        ],
        dim=1,
    )


# ============================================================
# OPTIONAL HENCKY / LOG STRAIN FOR VALIDATION
# ============================================================

def right_hencky_strain(grad_u):
    """
    Experimental validation helper.

    F = I + grad_u
    C = F^T F
    H = 0.5 * log(C)

    This is useful for comparing a finite-strain measure against
    Abaqus LE, but do NOT automatically assume it reproduces
    Abaqus component output under every finite-rotation setting.
    """

    N = grad_u.shape[0]

    I = torch.eye(
        3,
        dtype=grad_u.dtype,
        device=grad_u.device,
    ).unsqueeze(0).expand(N, -1, -1)

    F = I + grad_u

    C = torch.matmul(
        F.transpose(1, 2),
        F,
    )

    eigvals, eigvecs = torch.linalg.eigh(C)

    eigvals = torch.clamp(
        eigvals,
        min=1.0e-12,
    )

    log_diag = 0.5 * torch.log(eigvals)

    H = torch.matmul(
        eigvecs,
        torch.matmul(
            torch.diag_embed(log_diag),
            eigvecs.transpose(1, 2),
        ),
    )

    return H


# ============================================================
# ORTHOTROPIC LINEAR ELASTICITY
# ============================================================

def orthotropic_stiffness(
    p_phys,
    E3=12000.0,
    nu12=0.28,
    nu13=0.28,
    nu23=0.40,
    G13=4500.0,
    G23=3500.0,
):
    """
    p_phys = [E1,E2,G12] in MPa

    Returns 6x6 stiffness matrix C for
    engineering-shear Voigt strain:
    [e11,e22,e33,gamma12,gamma13,gamma23]
    """

    dtype = p_phys.dtype
    device = p_phys.device

    E1 = p_phys[0]
    E2 = p_phys[1]
    G12 = p_phys[2]

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

    S = torch.zeros(
        (6, 6),
        dtype=dtype,
        device=device,
    )

    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3

    # Reciprocity gives symmetric compliance:
    # nu21/E2 = nu12/E1, etc.
    S[0, 1] = -nu12 / E1
    S[1, 0] = S[0, 1]

    S[0, 2] = -nu13 / E1
    S[2, 0] = S[0, 2]

    S[1, 2] = -nu23 / E2
    S[2, 1] = S[1, 2]

    S[3, 3] = 1.0 / G12
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G23

    return torch.linalg.inv(S)


def stress_from_small_strain(
    eps_tensor,
    p_phys,
):
    """
    Returns:
    [s11,s22,s33,s12,s13,s23] MPa
    """

    eng_strain = (
        strain_tensor_to_voigt_engineering_shear(
            eps_tensor
        )
    )

    C = orthotropic_stiffness(p_phys)

    return torch.matmul(
        eng_strain,
        C.T,
    )
