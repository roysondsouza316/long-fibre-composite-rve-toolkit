"""Fluctuation constraints for the RVE: periodic node matching or zero boundary fluctuation.

The unknowns of the RVE solves are the displacement *fluctuations* ``w`` (the total
displacement is ``E x + w`` for macro strain ``E``). Constraints are expressed as a map from
every full DOF to an independent (reduced) DOF, or to ``-1`` for a DOF fixed at zero.

* ``periodic``: every node on an image face (x = x_max, ...) is tied to its partner on the
  mirror face; edge and corner chains are resolved to a single master. Partners are matched
  per side (matrix-side node to matrix-side node, fibre-side duplicate to fibre-side
  duplicate), so fibres that cross the boundary keep their cohesive interface. Rigid-body
  translation is removed by fixing the fluctuation of one interior matrix node, which is not
  part of any periodic tie.
* ``dirichlet``: the fluctuation is fixed at zero on the whole boundary (affine boundary
  displacement, the "KUBC" upper-bound condition).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rve2d.engines.tensormesh.mesh import IntArray, RVEMesh
from rve2d.exceptions import SolverError


@dataclass(frozen=True)
class DofMap:
    """``full_to_reduced[i]`` is the reduced index of full DOF ``i`` or ``-1`` if fixed."""

    full_to_reduced: IntArray
    n_reduced: int
    anchor_node: int | None


def boundary_tolerance(mesh: RVEMesh) -> float:
    return 1e-8 * float(np.max(mesh.upper - mesh.lower))


def boundary_nodes(mesh: RVEMesh) -> IntArray:
    tol = boundary_tolerance(mesh)
    on_face = (np.abs(mesh.points - mesh.lower) < tol) | (np.abs(mesh.points - mesh.upper) < tol)
    return np.flatnonzero(on_face.any(axis=1)).astype(np.int64)


def periodic_masters(mesh: RVEMesh) -> IntArray:
    """Master node of every node under periodicity (a node is its own master if untied)."""
    tol = boundary_tolerance(mesh)
    master = np.arange(mesh.n_nodes, dtype=np.int64)
    for axis in range(mesh.dim):
        others = [a for a in range(mesh.dim) if a != axis]
        mirror = np.flatnonzero(np.abs(mesh.points[:, axis] - mesh.lower[axis]) < tol)
        image = np.flatnonzero(np.abs(mesh.points[:, axis] - mesh.upper[axis]) < tol)
        lookup = {_match_key(mesh, int(node), others, tol): int(node) for node in mirror}
        for node in image:
            partner = lookup.get(_match_key(mesh, int(node), others, tol))
            if partner is None:
                coords = ", ".join(f"{value:.6g}" for value in mesh.points[node])
                raise SolverError(
                    "Periodic boundary conditions need a node-matched periodic mesh: no partner "
                    f"found for the node at ({coords}). Build the geometry with "
                    "periodic_compatible: true."
                )
            master[node] = partner
    for _ in range(mesh.dim + 1):
        master = master[master]
    return master


def build_dof_map(mesh: RVEMesh, boundary_condition: str, components: int | None = None) -> DofMap:
    """Reduced-DOF map with ``components`` displacement components per node (default: the
    mesh dimension; three on a 2D mesh for generalized plane strain), numbered node-major."""
    ncomp = mesh.dim if components is None else components
    if boundary_condition == "periodic":
        master = periodic_masters(mesh)
        anchor = _interior_anchor(mesh)
        independent = np.unique(master)
        independent = independent[independent != anchor]
        reduced_node = np.full(mesh.n_nodes, -1, dtype=np.int64)
        reduced_node[independent] = np.arange(independent.size, dtype=np.int64)
        node_index = reduced_node[master]
        n_reduced_nodes = independent.size
    elif boundary_condition == "dirichlet":
        fixed = np.zeros(mesh.n_nodes, dtype=bool)
        fixed[boundary_nodes(mesh)] = True
        node_index = np.full(mesh.n_nodes, -1, dtype=np.int64)
        free_nodes = np.flatnonzero(~fixed)
        node_index[free_nodes] = np.arange(free_nodes.size, dtype=np.int64)
        n_reduced_nodes = free_nodes.size
        anchor = None
    else:
        raise SolverError(f"Unsupported boundary condition {boundary_condition!r}.")
    full_to_reduced = np.full(mesh.n_nodes * ncomp, -1, dtype=np.int64)
    has_dofs = node_index >= 0
    for component in range(ncomp):
        full_to_reduced[np.flatnonzero(has_dofs) * ncomp + component] = (
            node_index[has_dofs] * ncomp + component
        )
    return DofMap(
        full_to_reduced=full_to_reduced, n_reduced=int(n_reduced_nodes * ncomp), anchor_node=anchor
    )


def _match_key(mesh: RVEMesh, node: int, others: list[int], tol: float) -> tuple[int, ...]:
    coords = np.round(mesh.points[node, others] / tol).astype(np.int64)
    return (*coords.tolist(), int(mesh.node_side[node]))


def _interior_anchor(mesh: RVEMesh) -> int:
    """Matrix-side node closest to the RVE centre that lies off the boundary."""
    candidates = np.ones(mesh.n_nodes, dtype=bool)
    candidates[boundary_nodes(mesh)] = False
    candidates &= mesh.node_side == 0
    interface = (
        np.unique(mesh.cohesive[:, : mesh.dim]) if mesh.n_cohesive else np.zeros(0, dtype=np.int64)
    )
    candidates[interface] = False
    pool = np.flatnonzero(candidates)
    if pool.size == 0:
        raise SolverError("Could not find an interior node to anchor the periodic fluctuation.")
    centre = 0.5 * (mesh.lower + mesh.upper)
    return int(pool[np.argmin(np.linalg.norm(mesh.points[pool] - centre, axis=1))])
