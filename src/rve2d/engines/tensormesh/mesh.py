"""RVE mesh preparation for the nonlinear solver: read the gmsh mesh and insert
zero-thickness cohesive elements on every fibre/matrix interface facet.

Interface insertion duplicates each interface node: matrix cells keep the original node,
fibre cells are re-pointed to the duplicate. A cohesive element joins an interface facet's
matrix-side nodes ("bottom") to the coincident fibre-side nodes ("top"), ordered so that the
facet normal computed from the bottom nodes points into the fibre. The separation used by the
traction-separation law is ``u_top - u_bottom = u_fibre - u_matrix`` expressed in that frame,
so a positive normal separation means the interface is opening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rve2d.exceptions import SolverError
from rve2d.mesh_io import read_mesh

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

# Local facets of linear simplices (node indices within the cell).
_CELL_FACETS: dict[int, IntArray] = {
    2: np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
    3: np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int64),
}


@dataclass(frozen=True)
class RVEMesh:
    """Bulk simplex mesh plus cohesive interface elements.

    ``cohesive`` rows are ``[bottom_0, ..., bottom_{d-1}, top_0, ..., top_{d-1}]`` where
    bottom nodes sit on the matrix side and top nodes on the fibre side.
    """

    dim: int
    points: FloatArray
    cells: IntArray
    phase: IntArray
    cohesive: IntArray
    node_side: IntArray
    lower: FloatArray
    upper: FloatArray
    matrix_phase_id: int
    fibre_phase_id: int

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_cohesive(self) -> int:
        return int(self.cohesive.shape[0])


def read_rve_mesh(
    mesh_path: str | Path,
    matrix_phase_id: int = 1,
    fibre_phase_id: int = 2,
) -> tuple[int, FloatArray, IntArray, IntArray]:
    """Read a gmsh RVE mesh and return ``(dim, points, cells, phase)``.

    Only linear triangles (2D) or linear tetrahedra (3D) are supported. Nodes not used by any
    bulk cell are dropped and the remaining nodes renumbered compactly.
    """
    mesh = read_mesh(mesh_path)
    cell_type = "tetra" if any(block.type == "tetra" for block in mesh.cells) else "triangle"
    blocks = [block.data for block in mesh.cells if block.type == cell_type]
    if not blocks:
        raise SolverError(
            "The solvers need a linear triangle (2D) or tetrahedron (3D) mesh; "
            "found cell types " + ", ".join(sorted({block.type for block in mesh.cells})) + "."
        )
    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    if physical is None:
        raise SolverError("Mesh is missing gmsh physical tags for the bulk cells.")
    cells = np.vstack([np.asarray(block, dtype=np.int64) for block in blocks])
    phase = np.asarray(physical, dtype=np.int64).reshape(-1)
    unknown = sorted(set(np.unique(phase).tolist()) - {matrix_phase_id, fibre_phase_id})
    if unknown:
        raise SolverError(f"Bulk cells carry unexpected physical tags {unknown}.")
    dim = 3 if cell_type == "tetra" else 2

    used, compact = np.unique(cells, return_inverse=True)
    points = np.asarray(mesh.points[used, :dim], dtype=np.float64)
    return dim, points, compact.reshape(cells.shape).astype(np.int64), phase


def insert_cohesive_interfaces(
    dim: int,
    points: FloatArray,
    cells: IntArray,
    phase: IntArray,
    matrix_phase_id: int = 1,
    fibre_phase_id: int = 2,
) -> RVEMesh:
    """Duplicate fibre-side interface nodes and build oriented cohesive elements."""
    n_nodes = points.shape[0]
    local = _CELL_FACETS[dim]
    n_local = local.shape[0]
    facet_nodes = cells[:, local].reshape(-1, dim)  # (n_cells * n_local, dim)
    keys = np.sort(facet_nodes, axis=1)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)

    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    shared = np.flatnonzero(counts == 2)
    occ_a = order[starts[shared]]
    occ_b = order[starts[shared] + 1]
    cell_a = occ_a // n_local
    cell_b = occ_b // n_local
    phase_a = phase[cell_a]
    phase_b = phase[cell_b]
    is_interface = (
        (phase_a != phase_b)
        & np.isin(phase_a, [matrix_phase_id, fibre_phase_id])
        & np.isin(phase_b, [matrix_phase_id, fibre_phase_id])
    )
    groups = shared[is_interface]
    fibre_cell = np.where(
        phase_a[is_interface] == fibre_phase_id, cell_a[is_interface], cell_b[is_interface]
    )
    bottom = unique_keys[groups].astype(np.int64)  # matrix-side (original) node ids

    interface_nodes = np.unique(bottom)
    duplicate_of = np.full(n_nodes, -1, dtype=np.int64)
    duplicate_of[interface_nodes] = n_nodes + np.arange(interface_nodes.size, dtype=np.int64)

    new_cells = cells.copy()
    fibre_rows = phase == fibre_phase_id
    fibre_cells = new_cells[fibre_rows]
    remapped = duplicate_of[fibre_cells]
    new_cells[fibre_rows] = np.where(remapped >= 0, remapped, fibre_cells)

    new_points = np.vstack([points, points[interface_nodes]])
    node_side = np.concatenate(
        [np.zeros(n_nodes, dtype=np.int64), np.ones(interface_nodes.size, dtype=np.int64)]
    )

    # Orient every facet so that the normal built from the bottom nodes points into the fibre.
    if bottom.shape[0]:
        normals = _facet_normals(points, bottom)
        facet_centre = points[bottom].mean(axis=1)
        fibre_centre = points[cells[fibre_cell]].mean(axis=1)
        flip = np.einsum("ij,ij->i", normals, fibre_centre - facet_centre) < 0.0
        bottom[flip, 0], bottom[flip, 1] = bottom[flip, 1].copy(), bottom[flip, 0].copy()
        measures = np.linalg.norm(_facet_normals(points, bottom), axis=1)
        if np.any(measures <= 1e-14 * max(float(np.ptp(points)), 1.0) ** (dim - 1)):
            raise SolverError("Found a degenerate (zero-size) fibre/matrix interface facet.")
    top = duplicate_of[bottom]
    cohesive = (
        np.hstack([bottom, top]) if bottom.shape[0] else np.zeros((0, 2 * dim), dtype=np.int64)
    )

    return RVEMesh(
        dim=dim,
        points=new_points,
        cells=new_cells,
        phase=phase.copy(),
        cohesive=cohesive.astype(np.int64),
        node_side=node_side,
        lower=points.min(axis=0),
        upper=points.max(axis=0),
        matrix_phase_id=matrix_phase_id,
        fibre_phase_id=fibre_phase_id,
    )


def build_rve_mesh(
    mesh_path: str | Path,
    matrix_phase_id: int = 1,
    fibre_phase_id: int = 2,
    cohesive_interfaces: bool = True,
) -> RVEMesh:
    """Read a gmsh RVE mesh and (optionally) insert cohesive interface elements."""
    dim, points, cells, phase = read_rve_mesh(mesh_path, matrix_phase_id, fibre_phase_id)
    if cohesive_interfaces:
        return insert_cohesive_interfaces(
            dim, points, cells, phase, matrix_phase_id, fibre_phase_id
        )
    return RVEMesh(
        dim=dim,
        points=points,
        cells=cells,
        phase=phase,
        cohesive=np.zeros((0, 2 * dim), dtype=np.int64),
        node_side=np.zeros(points.shape[0], dtype=np.int64),
        lower=points.min(axis=0),
        upper=points.max(axis=0),
        matrix_phase_id=matrix_phase_id,
        fibre_phase_id=fibre_phase_id,
    )


def _facet_normals(points: FloatArray, facets: IntArray) -> FloatArray:
    """Unnormalised facet normals: left normal of the edge (2D) or the triangle normal (3D)."""
    if facets.shape[1] == 2:
        edge = points[facets[:, 1]] - points[facets[:, 0]]
        return np.column_stack([-edge[:, 1], edge[:, 0]])
    a = points[facets[:, 1]] - points[facets[:, 0]]
    b = points[facets[:, 2]] - points[facets[:, 0]]
    return np.cross(a, b)
