"""Small-strain J2 (von Mises) plasticity with linear isotropic hardening, vectorised in torch.

Voigt order is ``(xx, yy, zz, yz, xz, xy)``. Strains carry engineering shear components
(``gamma = 2 * eps``); stresses carry tensor components. The same 6-component formulation
serves 3D solids and 2D plane strain / generalized plane strain (where ``yz = xz = 0``).

Return mapping and consistent tangent follow Simo & Hughes, *Computational Inelasticity*
(1998), Box 3.1/3.2 with the yield function ``||s|| - sqrt(2/3) (sigma_y0 + H alpha)``, where
``alpha`` is the equivalent (uniaxial) plastic strain. In uniaxial stress the model gives the
bilinear curve with elastic modulus E and post-yield tangent ``E H / (E + H)``. Phases without
a yield stress are linear elastic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_SQRT_2_3 = math.sqrt(2.0 / 3.0)


@dataclass(frozen=True)
class ElementMaterial:
    """Per-element material parameters as tensors of shape ``(n_elements,)``."""

    shear_modulus: torch.Tensor
    bulk_modulus: torch.Tensor
    yield_stress: torch.Tensor  # +inf for elastic phases
    hardening_modulus: torch.Tensor


@dataclass(frozen=True)
class PlasticState:
    plastic_strain: torch.Tensor  # (n, 6), engineering shear
    equivalent_plastic_strain: torch.Tensor  # (n,)


def element_material(
    youngs_modulus: torch.Tensor,
    poisson_ratio: torch.Tensor,
    yield_stress: torch.Tensor,
    hardening_modulus: torch.Tensor,
) -> ElementMaterial:
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    bulk = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    return ElementMaterial(shear, bulk, yield_stress, hardening_modulus)


def initial_state(n: int, dtype: torch.dtype, device: torch.device) -> PlasticState:
    return PlasticState(
        plastic_strain=torch.zeros(n, 6, dtype=dtype, device=device),
        equivalent_plastic_strain=torch.zeros(n, dtype=dtype, device=device),
    )


def j2_return_mapping(
    strain: torch.Tensor,
    state: PlasticState,
    material: ElementMaterial,
) -> tuple[torch.Tensor, torch.Tensor, PlasticState, torch.Tensor]:
    """Return ``(stress, consistent_tangent, new_state, yielding)`` for strains ``(n, 6)``."""
    mu = material.shear_modulus
    kappa = material.bulk_modulus
    hardening = material.hardening_modulus

    elastic_strain = strain - state.plastic_strain
    volumetric = elastic_strain[:, :3].sum(dim=1)
    dev_normal = elastic_strain[:, :3] - volumetric[:, None] / 3.0
    dev_shear = 0.5 * elastic_strain[:, 3:]  # tensor shear components
    s_trial = 2.0 * mu[:, None] * torch.cat([dev_normal, dev_shear], dim=1)
    s_norm = torch.sqrt((s_trial[:, :3] ** 2).sum(dim=1) + 2.0 * (s_trial[:, 3:] ** 2).sum(dim=1))

    yield_radius = _SQRT_2_3 * (material.yield_stress + hardening * state.equivalent_plastic_strain)
    f_trial = s_norm - yield_radius
    yielding = f_trial > 0.0

    safe_norm = torch.where(s_norm > 0.0, s_norm, torch.ones_like(s_norm))
    direction = s_trial / safe_norm[:, None]
    dgamma = torch.where(
        yielding, f_trial / (2.0 * mu * (1.0 + hardening / (3.0 * mu))), torch.zeros_like(f_trial)
    )

    deviatoric = s_trial - (2.0 * mu * dgamma)[:, None] * direction
    pressure = kappa * volumetric
    stress = deviatoric.clone()
    stress[:, :3] = stress[:, :3] + pressure[:, None]

    engineering_direction = torch.cat([direction[:, :3], 2.0 * direction[:, 3:]], dim=1)
    new_state = PlasticState(
        plastic_strain=state.plastic_strain + dgamma[:, None] * engineering_direction,
        equivalent_plastic_strain=state.equivalent_plastic_strain + _SQRT_2_3 * dgamma,
    )

    theta = torch.where(yielding, 1.0 - 2.0 * mu * dgamma / safe_norm, torch.ones_like(s_norm))
    theta_bar = torch.where(
        yielding, 1.0 / (1.0 + hardening / (3.0 * mu)) - (1.0 - theta), torch.zeros_like(s_norm)
    )
    tangent = _isotropic_tangent(kappa, mu * theta) - (2.0 * mu * theta_bar)[:, None, None] * (
        direction[:, :, None] * direction[:, None, :]
    )
    return stress, tangent, new_state, yielding


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    mean = stress[:, :3].mean(dim=1, keepdim=True)
    dev = stress[:, :3] - mean
    return torch.sqrt(1.5 * ((dev**2).sum(dim=1) + 2.0 * (stress[:, 3:] ** 2).sum(dim=1)))


def _isotropic_tangent(bulk: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    """``K 1(x)1 + 2 G (I_sym - 1/3 1(x)1)`` in engineering-shear Voigt form."""
    n = bulk.shape[0]
    tangent = torch.zeros(n, 6, 6, dtype=bulk.dtype, device=bulk.device)
    lam = bulk - 2.0 * shear / 3.0
    tangent[:, :3, :3] = lam[:, None, None]
    diag = torch.arange(3, device=bulk.device)
    tangent[:, diag, diag] = tangent[:, diag, diag] + 2.0 * shear[:, None]
    shear_idx = torch.arange(3, 6, device=bulk.device)
    tangent[:, shear_idx, shear_idx] = shear[:, None]
    return tangent
