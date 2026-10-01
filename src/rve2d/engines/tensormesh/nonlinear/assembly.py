"""Residual and tangent assembly for the nonlinear RVE problem.

Unknowns are the reduced displacement fluctuations ``w`` and the stress-controlled macro strain
components ``E_f``. The element strain is ``eps_e = B_e w_e + E`` (6-component Voigt, see
``material.py``), so the macro strain enters every bulk element directly and never the
cohesive separations (duplicated nodes share coordinates). With ``V`` the RVE volume:

* fluctuation residual   ``r_w = sum_e V_e B_e^T sigma_e + sum_c f_c``
* macro residual         ``r_E = sum_e V_e sigma_e = V * sigma_bar``  (Hill-Mandel average)

and the consistent tangent blocks ``K_ww = sum V B^T C B + K_coh``, ``K_wE = sum V B^T C``,
``K_EE = sum V C``. Periodic ties and fixed DOFs are applied through the reduced-DOF map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rve2d.engines.tensormesh.constraints import DofMap
from rve2d.engines.tensormesh.mesh import RVEMesh
from rve2d.engines.tensormesh.nonlinear.cohesive import (
    CohesiveGeometry,
    cohesive_geometry,
    evaluate_cohesive,
)
from rve2d.engines.tensormesh.nonlinear.material import (
    ElementMaterial,
    PlasticState,
    return_mapping,
)


@dataclass(frozen=True)
class BulkGeometry:
    strain_matrix: torch.Tensor  # (n_cells, 6, n_elem_dofs)
    volume: torch.Tensor  # (n_cells,)
    dofs: torch.Tensor  # (n_cells, n_elem_dofs)


@dataclass(frozen=True)
class SparseSystem:
    """Square sparse matrix in COO form; repeated ``(row, col)`` entries are summed."""

    row: torch.Tensor
    col: torch.Tensor
    value: torch.Tensor
    size: int


@dataclass
class Evaluation:
    """Everything a Newton iteration needs, already reduced to the free unknowns."""

    residual: torch.Tensor  # (n_reduced + n_free_macro,)
    matrix: SparseSystem | None  # bordered tangent (None when not requested)
    macro_stress: torch.Tensor  # (6,) volume-averaged stress
    force_scale: float
    stress: torch.Tensor
    plastic_state: PlasticState
    yielding: torch.Tensor
    cohesive_history: torch.Tensor
    damage: torch.Tensor
    delta_local: torch.Tensor


def bulk_geometry(points: torch.Tensor, cells: torch.Tensor) -> BulkGeometry:
    dim = points.shape[1]
    coords = points[cells]
    n_cells = cells.shape[0]
    if dim == 2:
        x, y = coords[..., 0], coords[..., 1]
        det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        dndx = (
            torch.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], dim=1)
            / det[:, None]
        )
        dndy = (
            torch.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], dim=1)
            / det[:, None]
        )
        strain_matrix = torch.zeros(n_cells, 6, 6, dtype=points.dtype, device=points.device)
        strain_matrix[:, 0, 0::2] = dndx
        strain_matrix[:, 1, 1::2] = dndy
        strain_matrix[:, 5, 0::2] = dndy
        strain_matrix[:, 5, 1::2] = dndx
        volume = 0.5 * det.abs()
    else:
        ones = torch.ones(n_cells, 4, 1, dtype=points.dtype, device=points.device)
        coeff = torch.linalg.inv(torch.cat([ones, coords], dim=2))  # rows: [a; dN/dx; dN/dy; dN/dz]
        grads = coeff[:, 1:, :]  # (n_cells, 3, 4)
        strain_matrix = torch.zeros(n_cells, 6, 12, dtype=points.dtype, device=points.device)
        strain_matrix[:, 0, 0::3] = grads[:, 0]
        strain_matrix[:, 1, 1::3] = grads[:, 1]
        strain_matrix[:, 2, 2::3] = grads[:, 2]
        strain_matrix[:, 3, 1::3] = grads[:, 2]
        strain_matrix[:, 3, 2::3] = grads[:, 1]
        strain_matrix[:, 4, 0::3] = grads[:, 2]
        strain_matrix[:, 4, 2::3] = grads[:, 0]
        strain_matrix[:, 5, 0::3] = grads[:, 1]
        strain_matrix[:, 5, 1::3] = grads[:, 0]
        volume = torch.linalg.det(torch.cat([ones, coords], dim=2)).abs() / 6.0
    comp = torch.arange(dim, device=points.device)
    dofs = (cells[:, :, None] * dim + comp).reshape(n_cells, -1)
    return BulkGeometry(strain_matrix, volume, dofs)


class RVESystem:
    """Precomputed geometry plus the evaluation of residual and tangent for given unknowns."""

    def __init__(
        self,
        mesh: RVEMesh,
        dof_map: DofMap,
        material: ElementMaterial,
        law: Any,
        macro_active: list[int],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        cohesive_integration: str = "nodal",
    ) -> None:
        self.mesh = mesh
        self.dim = mesh.dim
        self.device = device
        self.dtype = dtype
        self.points = torch.as_tensor(mesh.points, dtype=dtype, device=device)
        cells = torch.as_tensor(mesh.cells, dtype=torch.long, device=device)
        self.bulk = bulk_geometry(self.points, cells)
        self.volume = float(self.bulk.volume.sum())
        self.material = material
        self.law = law
        connectivity = torch.as_tensor(mesh.cohesive, dtype=torch.long, device=device)
        self.cohesive: CohesiveGeometry | None = (
            cohesive_geometry(self.points, connectivity, cohesive_integration)
            if mesh.n_cohesive
            else None
        )
        self.n_full = mesh.n_nodes * self.dim
        self.map = torch.as_tensor(dof_map.full_to_reduced, dtype=torch.long, device=device)
        self.n_reduced = dof_map.n_reduced
        self.macro_active = list(macro_active)

    # -- state helpers -------------------------------------------------------------------
    def expand(self, w_reduced: torch.Tensor) -> torch.Tensor:
        """Full fluctuation vector from the reduced unknowns (fixed DOFs are zero)."""
        full = torch.zeros(self.n_full, dtype=self.dtype, device=self.device)
        mask = self.map >= 0
        full[mask] = w_reduced[self.map[mask]]
        return full

    def reduce_vector(self, full: torch.Tensor) -> torch.Tensor:
        mask = self.map >= 0
        out = torch.zeros(self.n_reduced, *full.shape[1:], dtype=full.dtype, device=full.device)
        return out.index_add(0, self.map[mask], full[mask])

    def cohesive_quadrature_points(self) -> int:
        return 0 if self.cohesive is None else int(self.cohesive.weights.shape[1])

    # -- evaluation ----------------------------------------------------------------------
    def evaluate(
        self,
        w_reduced: torch.Tensor,
        macro_strain: torch.Tensor,
        plastic_state: PlasticState,
        cohesive_history: torch.Tensor,
        free_macro: list[int],
        target_macro_stress: torch.Tensor,
        with_tangent: bool = True,
    ) -> Evaluation:
        w_full = self.expand(w_reduced)
        strain_matrix = self.bulk.strain_matrix
        vol = self.bulk.volume
        strain = (
            torch.einsum("eij,ej->ei", strain_matrix, w_full[self.bulk.dofs])
            + macro_strain[None, :]
        )
        stress, tangent, new_state, yielding = return_mapping(strain, plastic_state, self.material)

        element_forces = torch.einsum("eji,ej->ei", strain_matrix, stress) * vol[:, None]
        residual_full = torch.zeros(self.n_full, dtype=self.dtype, device=self.device)
        residual_full = residual_full.index_add(
            0, self.bulk.dofs.reshape(-1), element_forces.reshape(-1)
        )
        force_scale = float(torch.linalg.norm(element_forces))
        macro_residual = (stress * vol[:, None]).sum(dim=0)  # = V * sigma_bar

        coh_forces = coh_tangents = None
        new_history = cohesive_history
        damage = torch.zeros(0, dtype=self.dtype, device=self.device)
        delta_local = torch.zeros(0, dtype=self.dtype, device=self.device)
        if self.cohesive is not None:
            coh_forces, coh_tangents, new_history, damage, delta_local = evaluate_cohesive(
                self.law, self.cohesive, w_full, cohesive_history, with_tangent=with_tangent
            )
            residual_full = residual_full.index_add(
                0, self.cohesive.dofs.reshape(-1), coh_forces.reshape(-1)
            )
            force_scale = float(np.hypot(force_scale, float(torch.linalg.norm(coh_forces))))

        residual = torch.cat(
            [
                self.reduce_vector(residual_full),
                macro_residual[free_macro] - self.volume * target_macro_stress[free_macro],
            ]
        )
        matrix = None
        if with_tangent:
            matrix = self._bordered_matrix(strain_matrix, vol, tangent, coh_tangents, free_macro)
        return Evaluation(
            residual=residual,
            matrix=matrix,
            macro_stress=macro_residual / self.volume,
            force_scale=force_scale,
            stress=stress,
            plastic_state=new_state,
            yielding=yielding,
            cohesive_history=new_history,
            damage=damage,
            delta_local=delta_local,
        )

    def _bordered_matrix(
        self,
        strain_matrix: torch.Tensor,
        vol: torch.Tensor,
        tangent: torch.Tensor,
        coh_tangents: torch.Tensor | None,
        free_macro: list[int],
    ) -> SparseSystem:
        """COO triplets of the reduced, bordered tangent (duplicates are summed by the solver)."""
        c_times_b = torch.einsum("eij,ejk->eik", tangent, strain_matrix)
        k_bulk = torch.einsum("eji,ejk->eik", strain_matrix, c_times_b) * vol[:, None, None]
        rows = [self.bulk.dofs[:, :, None].expand_as(k_bulk).reshape(-1)]
        cols = [self.bulk.dofs[:, None, :].expand_as(k_bulk).reshape(-1)]
        vals = [k_bulk.reshape(-1)]
        if coh_tangents is not None and self.cohesive is not None:
            d = self.cohesive.dofs
            rows.append(d[:, :, None].expand_as(coh_tangents).reshape(-1))
            cols.append(d[:, None, :].expand_as(coh_tangents).reshape(-1))
            vals.append(coh_tangents.reshape(-1))
        row = self.map[torch.cat(rows)]
        col = self.map[torch.cat(cols)]
        val = torch.cat(vals)
        keep = (row >= 0) & (col >= 0)
        row, col, val = row[keep], col[keep], val[keep]

        n_free = len(free_macro)
        size = self.n_reduced + n_free
        if n_free:
            # K_wE, K_Ew (reduced) and K_EE for the stress-controlled macro components; the
            # material tangent need not be symmetric (non-associative flow, damage).
            k_we_elem = (
                torch.einsum("eji,ejk->eik", strain_matrix, tangent[:, :, free_macro])
                * vol[:, None, None]
            )
            k_ew_elem = (  # [e, j, k] = (C[free_k, :] B)[j]
                torch.einsum("eki,eij->ejk", tangent[:, free_macro, :], strain_matrix)
                * vol[:, None, None]
            )
            k_we = self._reduce_columns(k_we_elem, n_free)  # (n_reduced, n_free)
            k_ew = self._reduce_columns(k_ew_elem, n_free)  # (n_reduced, n_free), transposed
            k_ee = (tangent[:, free_macro][:, :, free_macro] * vol[:, None, None]).sum(dim=0)
            r_idx = torch.arange(self.n_reduced, device=self.device)
            border = self.n_reduced + torch.arange(n_free, device=self.device)
            rr = r_idx[:, None].expand(-1, n_free).reshape(-1)
            cc = border[None, :].expand(self.n_reduced, -1).reshape(-1)
            ee_r = border[:, None].expand(-1, n_free).reshape(-1)
            ee_c = border[None, :].expand(n_free, -1).reshape(-1)
            row = torch.cat([row, rr, cc, ee_r])
            col = torch.cat([col, cc, rr, ee_c])
            val = torch.cat([val, k_we.reshape(-1), k_ew.reshape(-1), k_ee.reshape(-1)])
        return SparseSystem(row.detach(), col.detach(), val.detach(), size)

    def _reduce_columns(self, element_columns: torch.Tensor, n_free: int) -> torch.Tensor:
        """Assemble per-element ``(n_elem_dofs, n_free)`` blocks and reduce them."""
        full = torch.zeros(self.n_full, n_free, dtype=self.dtype, device=self.device)
        full = full.index_add(0, self.bulk.dofs.reshape(-1), element_columns.reshape(-1, n_free))
        return self.reduce_vector(full)
