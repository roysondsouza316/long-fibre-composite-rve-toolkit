"""VTU field output of the nonlinear RVE solve (bulk and interface)."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import torch

from rve2d.nonlinear.assembly import Evaluation, RVESystem
from rve2d.nonlinear.material import von_mises
from rve2d.nonlinear.records import COMPONENTS
from rve2d.nonlinear.solver import SolverState


def total_displacement(system: RVESystem, state: SolverState) -> np.ndarray:
    """Fluctuation plus the affine part ``E x`` at every node, padded to 3 components."""
    w = system.expand(state.w).reshape(-1, system.dim)
    e = state.macro_strain
    grad = torch.stack(
        [
            torch.stack([e[0], 0.5 * e[5], 0.5 * e[4]]),
            torch.stack([0.5 * e[5], e[1], 0.5 * e[3]]),
            torch.stack([0.5 * e[4], 0.5 * e[3], e[2]]),
        ]
    )[: system.dim, : system.dim]
    u = w + system.points @ grad.T
    out = np.zeros((system.mesh.n_nodes, 3))
    out[:, : system.dim] = u.detach().cpu().numpy()
    return out


def write_fields(
    output_dir: Path,
    stem: str,
    system: RVESystem,
    state: SolverState,
    evaluation: Evaluation,
) -> list[Path]:
    mesh = system.mesh
    points = np.zeros((mesh.n_nodes, 3))
    points[:, : mesh.dim] = mesh.points
    displacement = total_displacement(system, state)
    stress = evaluation.stress.detach().cpu().numpy()
    cell_data: dict[str, list[np.ndarray]] = {
        "phase": [mesh.phase.astype(np.int32)],
        "equivalent_plastic_strain": [
            state.plastic.equivalent_plastic_strain.detach().cpu().numpy()
        ],
        "von_mises": [von_mises(evaluation.stress).detach().cpu().numpy()],
    }
    for index, comp in enumerate(COMPONENTS):
        cell_data[f"stress_{comp}"] = [stress[:, index]]
    bulk_type = "triangle" if mesh.dim == 2 else "tetra"
    bulk_path = output_dir / f"{stem}.vtu"
    meshio.write(
        bulk_path,
        meshio.Mesh(
            points,
            [(bulk_type, mesh.cells)],
            point_data={"displacement": displacement},
            cell_data=cell_data,
        ),
    )
    written = [bulk_path]
    if mesh.n_cohesive and evaluation.damage.numel():
        damage = evaluation.damage.detach().cpu().numpy()
        delta = evaluation.delta_local.detach().cpu().numpy()  # (n_coh, n_q, dim)
        facet_type = "line" if mesh.dim == 2 else "triangle"
        interface_path = output_dir / f"{stem}_interface.vtu"
        meshio.write(
            interface_path,
            meshio.Mesh(
                points,
                [(facet_type, mesh.cohesive[:, : mesh.dim])],
                point_data={"displacement": displacement},
                cell_data={
                    "damage": [damage.mean(axis=1)],
                    "max_damage": [damage.max(axis=1)],
                    "normal_opening": [delta[:, :, 0].mean(axis=1)],
                    "shear_opening": [np.linalg.norm(delta[:, :, 1:], axis=2).mean(axis=1)],
                },
            ),
        )
        written.append(interface_path)
    return written
