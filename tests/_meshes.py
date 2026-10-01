"""Structured RVE meshes written as gmsh .msh files (no gmsh needed) for solver tests.

Every mesh is the unit square/cube with a square (2D) or square-prism (3D, along z) fibre
in ``[0.3, 0.7]^2``; physical tag 1 is the matrix and 2 the fibre.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np


def square(n: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y] for y in xs for x in xs])
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return points, np.array(cells)


def cube(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Kuhn split of the unit cube into 6n^3 tetrahedra (mixed orientations on purpose)."""
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y, z] for z in xs for y in xs for x in xs])

    def node(i: int, j: int, k: int) -> int:
        return (k * (n + 1) + j) * (n + 1) + i

    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v = [node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)]
                for first, second in ((1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2)):
                    cells.append([v[0], v[first], v[first | second], v[7]])
    return points, np.array(cells)


def extrude(points: np.ndarray, triangles: np.ndarray, layers: int, depth: float):
    """Stack ``layers`` prism layers on a 2D triangle mesh and split each prism into three
    tetrahedra by global vertex order (conforming across prisms). Returns the 3D points, the
    tetrahedra and, for every tetrahedron, the index of its parent triangle."""
    n2 = points.shape[0]
    zs = np.linspace(0.0, depth, layers + 1)
    points3 = np.vstack([np.column_stack([points, np.full(n2, z)]) for z in zs])
    tets, parent = [], []
    for layer in range(layers):
        for index, tri in enumerate(triangles):
            a, b, c = sorted(int(v) for v in tri)
            low = [a + layer * n2, b + layer * n2, c + layer * n2]
            up = [v + n2 for v in low]
            for tet in ([low[0], low[1], low[2], up[0]],
                        [low[1], low[2], up[0], up[1]],
                        [low[2], up[0], up[1], up[2]]):  # fmt: skip
                tets.append(tet)
                parent.append(index)
    return points3, np.array(tets), np.array(parent)


def fibre_phase(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    centre = points[cells][:, :, :2].mean(axis=1)
    return np.where(np.all((centre > 0.3) & (centre < 0.7), axis=1), 2, 1)


def write_msh(path: Path, points: np.ndarray, cells: np.ndarray, phase: np.ndarray) -> Path:
    cell_type = "triangle" if cells.shape[1] == 3 else "tetra"
    padded = np.zeros((points.shape[0], 3))
    padded[:, : points.shape[1]] = points
    meshio.write(
        path,
        meshio.Mesh(
            padded,
            [(cell_type, cells)],
            cell_data={"gmsh:physical": [phase], "gmsh:geometrical": [phase]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    return path


def square_msh(path: Path, n: int = 8) -> Path:
    points, cells = square(n)
    return write_msh(path, points, cells, fibre_phase(points, cells))


def cube_msh(path: Path, n: int = 4) -> Path:
    points, cells = cube(n)
    return write_msh(path, points, cells, fibre_phase(points, cells))


def extruded_msh(path: Path, n: int = 8, layers: int = 2, depth: float = 0.5) -> Path:
    points, triangles = square(n)
    phase2 = fibre_phase(points, triangles)
    points3, tets, parent = extrude(points, triangles, layers, depth)
    return write_msh(path, points3, tets, phase2[parent])
