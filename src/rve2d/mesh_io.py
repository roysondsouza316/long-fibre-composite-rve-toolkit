"""Read mesh files with meshio, without its side effects on the console and the process, and
extrude 2D triangle meshes into conforming tetrahedral layers."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from numpy.typing import NDArray

from rve2d.exceptions import MeshingError


def read_mesh(path: str | Path) -> meshio.Mesh:
    """Read a mesh file; ``.msh`` files are read as gmsh.

    ``meshio.read`` tries the ANSYS reader first for ``.msh`` files (printing its failure to
    stdout) and exits the process when no reader succeeds. Here a missing file raises
    ``FileNotFoundError`` and an unreadable one ``MeshingError``.
    """
    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    try:
        if mesh_path.suffix.lower() == ".msh":
            return meshio.gmsh.read(mesh_path)
        return meshio.read(mesh_path)
    except (Exception, SystemExit) as exc:
        detail = str(exc) or type(exc).__name__
        raise MeshingError(f"Could not read the mesh file {mesh_path}: {detail}") from exc


def extrude_triangles(
    points: NDArray[np.float64], triangles: NDArray[np.int64], depth: float, layers: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Stack ``layers`` prism layers of total ``depth`` along z on a 2D triangle mesh and
    split every prism into three tetrahedra by global vertex order, which keeps neighbouring
    prisms conforming. Returns the 3D points (layer by layer), the tetrahedra and the index of
    the parent triangle of every tetrahedron."""
    if depth <= 0.0 or layers < 1:
        raise MeshingError("Extrusion needs a positive depth and at least one layer.")
    count = points.shape[0]
    planar = np.asarray(points, dtype=np.float64)[:, :2]
    heights = np.linspace(0.0, depth, layers + 1)
    points3 = np.vstack([np.column_stack([planar, np.full(count, z)]) for z in heights])
    low = np.sort(np.asarray(triangles, dtype=np.int64), axis=1)
    tets = []
    for layer in range(layers):
        lo = low + layer * count
        up = lo + count
        split = (
            (lo[:, 0], lo[:, 1], lo[:, 2], up[:, 0]),
            (lo[:, 1], lo[:, 2], up[:, 0], up[:, 1]),
            (lo[:, 2], up[:, 0], up[:, 1], up[:, 2]),
        )
        tets.append(np.stack([np.column_stack(tet) for tet in split], axis=1).reshape(-1, 4))
    parent = np.tile(np.repeat(np.arange(low.shape[0], dtype=np.int64), 3), layers)
    return points3, np.vstack(tets), parent


def extrude_mesh(
    mesh_path: str | Path,
    output_path: str | Path,
    depth: float | None = None,
    layers: int = 1,
) -> Path:
    """Extrude a 2D gmsh triangle mesh along z into tetrahedra (``extrude_triangles``) and
    write it as a gmsh mesh; every tetrahedron keeps the tags of its triangle.

    One layer on a periodic 2D mesh, solved with periodic boundary conditions, is the
    z-invariant 3D problem of the cross-section: the fluctuations of the top and bottom nodes
    are equal, so every tetrahedron carries the strain of its triangle (with the longitudinal
    shears that a 2D solve lacks). ``depth`` defaults to the mean edge length.
    """
    mesh = read_mesh(mesh_path)
    blocks = [index for index, block in enumerate(mesh.cells) if block.type == "triangle"]
    if not blocks:
        raise MeshingError(f"{mesh_path} has no linear triangles to extrude.")
    triangles = np.vstack([np.asarray(mesh.cells[index].data, dtype=np.int64) for index in blocks])
    tags = {
        key: np.concatenate([np.asarray(mesh.cell_data[key][i]).reshape(-1) for i in blocks])
        for key in ("gmsh:physical", "gmsh:geometrical")
        if key in mesh.cell_data
    }
    if "gmsh:physical" not in tags:
        raise MeshingError(f"{mesh_path} has no physical tags on its triangles.")
    tags.setdefault("gmsh:geometrical", tags["gmsh:physical"])
    if depth is None:
        corners = mesh.points[triangles][:, :, :2]
        edges = corners - np.roll(corners, 1, axis=1)
        depth = float(np.linalg.norm(edges, axis=2).mean())
    points3, tets, parent = extrude_triangles(mesh.points, triangles, depth, layers)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        output,
        meshio.Mesh(
            points3,
            [("tetra", tets)],
            cell_data={key: [values[parent]] for key, values in tags.items()},
        ),
        file_format="gmsh22",
        binary=False,
    )
    return output
