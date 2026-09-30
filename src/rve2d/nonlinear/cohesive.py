"""Vectorised zero-thickness cohesive elements for fibre/matrix interfaces.

2D: 4-node line pairs; 3D: 6-node triangle pairs. Integration is nodal (Newton-Cotes, the
default) or Gauss (2-point line rule, 3-point triangle rule).
The traction-separation law is any ``diffcohesive.laws.TractionSeparationLaw``: it maps local
separations ``(..., ndim)`` (normal first, then shear) and history to tractions, so the laws
shipped with diffcohesive (mixed-mode bilinear with BK / power-law closure, the Alfano shape
library, frictional, neural) all plug in unchanged. The law tangent ``dT/d(delta)`` comes from
forward-mode autograd over all integration points at once, which keeps the element tangent
consistent for any law, analytic or learned.

Kinematics use the reference facet frame (small-strain, small-rotation setting, matching the
bulk formulation): ``delta_local = R (u_top - u_bottom)`` with ``R`` rows ``[n, t (, t2)]`` and
``n`` pointing into the fibre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

_GAUSS_LINE = (
    torch.tensor(
        [
            [0.5 * (1.0 - 1.0 / math.sqrt(3.0)), 0.5 * (1.0 + 1.0 / math.sqrt(3.0))],
            [0.5 * (1.0 + 1.0 / math.sqrt(3.0)), 0.5 * (1.0 - 1.0 / math.sqrt(3.0))],
        ],
        dtype=torch.float64,
    ),
    torch.tensor([0.5, 0.5], dtype=torch.float64),
)
_TRIANGLE_RULE = (
    torch.tensor(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=torch.float64,
    ),
    torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64),
)


@dataclass(frozen=True)
class CohesiveGeometry:
    rotation: torch.Tensor  # (n_coh, dim, dim), local = R @ global
    weights: torch.Tensor  # (n_coh, n_q): quadrature weight * facet measure
    shape: torch.Tensor  # (n_q, n_face_nodes)
    dofs: torch.Tensor  # (n_coh, 2 * n_face_nodes * dim): bottom dofs then top dofs
    measure: torch.Tensor  # (n_coh,) facet length (2D) or area (3D)


def cohesive_geometry(
    points: torch.Tensor, connectivity: torch.Tensor, integration: str = "nodal"
) -> CohesiveGeometry:
    """Reference geometry of the cohesive elements.

    ``integration="nodal"`` (Newton-Cotes: integration points at the node pairs) decouples the
    node pairs and avoids the spurious traction oscillations of Gauss integration on stiff
    interfaces (Schellekens & de Borst, IJNME 36, 1993); ``"gauss"`` uses the interior rules.
    """
    if integration not in {"nodal", "gauss"}:
        raise ValueError(f"cohesive integration must be 'nodal' or 'gauss', got {integration!r}")
    dim = points.shape[1]
    n_face = dim
    bottom = connectivity[:, :n_face]
    coords = points[bottom]  # (n_coh, n_face, dim)
    if dim == 2:
        edge = coords[:, 1] - coords[:, 0]
        length = torch.linalg.norm(edge, dim=1)
        t = edge / length[:, None]
        n = torch.stack([-t[:, 1], t[:, 0]], dim=1)
        rotation = torch.stack([n, t], dim=1)
        shape, w = _GAUSS_LINE
        measure = length
    else:
        a = coords[:, 1] - coords[:, 0]
        b = coords[:, 2] - coords[:, 0]
        normal = torch.linalg.cross(a, b, dim=1)
        double_area = torch.linalg.norm(normal, dim=1)
        n = normal / double_area[:, None]
        t1 = a / torch.linalg.norm(a, dim=1)[:, None]
        t2 = torch.linalg.cross(n, t1, dim=1)
        rotation = torch.stack([n, t1, t2], dim=1)
        shape, w = _TRIANGLE_RULE
        measure = 0.5 * double_area
    if integration == "nodal":
        shape = torch.eye(n_face, dtype=torch.float64)
        w = torch.full((n_face,), 1.0 / n_face, dtype=torch.float64)
    shape = shape.to(dtype=points.dtype, device=points.device)
    weights = w.to(dtype=points.dtype, device=points.device)[None, :] * measure[:, None]
    comp = torch.arange(dim, device=points.device)
    dofs = (connectivity[:, :, None] * dim + comp).reshape(connectivity.shape[0], -1)
    return CohesiveGeometry(rotation, weights, shape, dofs, measure)


def evaluate_cohesive(
    law: Any,
    geometry: CohesiveGeometry,
    u_full: torch.Tensor,
    history: torch.Tensor,
    with_tangent: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(forces, tangents, new_history, damage, delta_local)``.

    ``forces``: (n_coh, n_elem_dofs) element internal forces; ``tangents``: (n_coh, n, n).
    ``history`` has shape (n_coh, n_q) or (n_coh, n_q, state_dim).
    """
    n_coh, n_q = geometry.weights.shape
    dim = geometry.rotation.shape[1]
    n_face = geometry.shape.shape[1]
    u = u_full[geometry.dofs].reshape(n_coh, 2, n_face, dim)
    jump_nodes = u[:, 1] - u[:, 0]  # (n_coh, n_face, dim): top minus bottom
    jump = torch.einsum("qa,cad->cqd", geometry.shape, jump_nodes)
    delta_local = torch.einsum("cij,cqj->cqi", geometry.rotation, jump)

    flat_delta = delta_local.reshape(n_coh * n_q, dim)
    flat_history = history.reshape(n_coh * n_q, *history.shape[2:])
    traction, new_history, damage = law(flat_delta, flat_history)
    traction = traction.reshape(n_coh, n_q, dim)
    traction_global = torch.einsum("cji,cqj->cqi", geometry.rotation, traction)
    f_top = torch.einsum("cq,qa,cqi->cai", geometry.weights, geometry.shape, traction_global)
    forces = torch.cat([-f_top, f_top], dim=1).reshape(n_coh, -1)

    tangents = None
    if with_tangent:
        d_traction = _law_jacobian(law, flat_delta, flat_history).reshape(n_coh, n_q, dim, dim)
        d_global = torch.einsum(
            "cji,cqjl,clk->cqik", geometry.rotation, d_traction, geometry.rotation
        )
        k_tt = torch.einsum(
            "cq,qa,qb,cqik->caibk", geometry.weights, geometry.shape, geometry.shape, d_global
        )
        k_tt = k_tt.reshape(n_coh, n_face * dim, n_face * dim)
        tangents = torch.cat(
            [torch.cat([k_tt, -k_tt], dim=2), torch.cat([-k_tt, k_tt], dim=2)], dim=1
        )
    return (
        forces,
        tangents,
        new_history.reshape(history.shape),
        damage.reshape(n_coh, n_q),
        delta_local,
    )


def _law_jacobian(law: Any, delta: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    def traction_only(d: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = law(d, h)[0]
        return result

    jacobian: torch.Tensor = torch.func.vmap(torch.func.jacfwd(traction_only, argnums=0))(
        delta, history
    )
    return jacobian
