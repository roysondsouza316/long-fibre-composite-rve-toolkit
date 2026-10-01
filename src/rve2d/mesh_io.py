"""Read mesh files with meshio, without its side effects on the console and the process."""

from __future__ import annotations

from pathlib import Path

import meshio

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
