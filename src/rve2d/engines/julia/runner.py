"""Run the Julia engine: locate Julia, prepare the bundled environment, run a task.

The Julia sources ship inside the Python package (``FerriteRVE/``, ``DiffCohesive/`` and
``run.jl`` next to this file). Every run uses that environment explicitly
(``julia --project=.../FerriteRVE``), so nothing has to be installed into the global Julia
environment. The first run sets it up (resolves and downloads Ferrite.jl and friends and
precompiles, a few minutes) and records the Julia dependencies it was set up for, so an
upgrade of rve2d that changes them sets it up again. If the installed package directory is
read-only, the sources are copied to a user cache directory first.

Environment variables: ``RVE2D_JULIA`` (Julia executable, default ``julia`` on PATH) and
``RVE2D_JULIA_TIMEOUT`` (seconds per run, default 86400).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from rve2d.exceptions import SolverError
from rve2d.mesh_io import read_mesh

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT = "FerriteRVE"
ENTRY = "run.jl"
MIN_JULIA = (1, 11)
DEFAULT_TIMEOUT = 86_400.0
SETUP_COMMAND = "using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()"
SETUP_STAMP = ".rve2d-setup"  # in FerriteRVE/: digest of the Project.toml files it was set up for


def julia_executable() -> str:
    candidate = os.environ.get("RVE2D_JULIA") or shutil.which("julia")
    if not candidate:
        raise SolverError(
            "The Julia engine needs Julia 1.11 or newer on PATH (https://julialang.org/downloads"
            "; or set RVE2D_JULIA to the executable). Use `engine: tensormesh` to solve without "
            "Julia, or run `rve2d doctor` to check the setup."
        )
    return candidate


def julia_version(executable: str) -> tuple[int, int, int] | None:
    try:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    words = completed.stdout.split()
    if len(words) < 3:
        return None
    parts = words[2].split("-")[0].split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except ValueError:
        return None


def engine_dir() -> Path:
    """Directory with ``run.jl`` and the Julia packages, writable so they can be set up.

    A read-only install is copied to ``$XDG_CACHE_HOME/rve2d`` (default ``~/.cache``), in a
    folder named after the content of the Julia sources, so every version gets its own copy.
    """
    if os.access(ENGINE_DIR / PROJECT, os.W_OK):
        return ENGINE_DIR
    sources = _digest(
        ENGINE_DIR,
        (
            path
            for path in ENGINE_DIR.rglob("*")
            if path.suffix in {".jl", ".toml", ".csv"} and path.name != "Manifest.toml"
        ),
    )
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    target = cache_root / "rve2d" / f"julia-engine-{sources}"
    if not (target / ENTRY).exists():
        shutil.copytree(
            ENGINE_DIR,
            target,
            ignore=shutil.ignore_patterns("*.py", "__pycache__", "Manifest.toml", SETUP_STAMP),
            dirs_exist_ok=True,
        )
    return target


def project_dir() -> Path:
    return engine_dir() / PROJECT


def is_instantiated() -> bool:
    """The environment is set up for the current Julia dependencies."""
    project = project_dir()
    stamp = project / SETUP_STAMP
    return (
        (project / "Manifest.toml").exists()
        and stamp.exists()
        and stamp.read_text(encoding="utf-8").strip() == _dependency_digest(project)
    )


def ensure_environment(executable: str | None = None, quiet: bool = False) -> Path:
    """Set up (resolve, download, precompile) the engine environment unless it is current."""
    julia = executable or julia_executable()
    project = project_dir()
    if is_instantiated():
        return project
    version = julia_version(julia)
    if version is not None and version[:2] < MIN_JULIA:
        raise SolverError(
            f"The Julia engine needs Julia {MIN_JULIA[0]}.{MIN_JULIA[1]} or newer; "
            f"{julia} is {'.'.join(map(str, version))}."
        )
    if not quiet:
        print(
            "Setting up the Julia engine (first use: downloads Ferrite.jl and precompiles; "
            "this takes a few minutes)...",
            file=sys.stderr,
            flush=True,
        )
    completed = _run([julia, f"--project={project}", "-e", SETUP_COMMAND], timeout=None)
    if completed.returncode != 0:
        raise SolverError(
            "Could not set up the Julia engine environment.\n"
            f"{_tail(completed.stdout + completed.stderr)}"
        )
    (project / SETUP_STAMP).write_text(_dependency_digest(project) + "\n", encoding="utf-8")
    return project


def _dependency_digest(project: Path) -> str:
    base = project.parent
    return _digest(base, [project / "Project.toml", base / "DiffCohesive" / "Project.toml"])


def _digest(base: Path, paths: Iterable[Path]) -> str:
    """Short hash of the names (relative to ``base``) and contents of ``paths``."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.relative_to(base).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def run_task(input_path: Path, log_path: Path) -> None:
    """Run the task described by ``input_path`` (a TOML file); write the log to ``log_path``."""
    julia = julia_executable()
    project = ensure_environment(julia)
    command = [julia, f"--project={project}", str(project.parent / ENTRY), str(input_path)]
    timeout = float(os.environ.get("RVE2D_JULIA_TIMEOUT", DEFAULT_TIMEOUT))
    completed = _run(command, timeout=timeout)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SolverError(
            f"The Julia engine failed (log: {log_path}).\n"
            f"{_tail(completed.stdout + completed.stderr)}"
        )


def _run(command: list[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise SolverError(f"Could not start Julia ({command[0]}): {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SolverError(
            f"The Julia engine did not finish within {timeout:g} s (RVE2D_JULIA_TIMEOUT)."
        ) from exc


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def write_toml(path: Path, top: dict[str, Any], tables: dict[str, dict[str, Any]]) -> Path:
    """Write a TOML file with top-level keys and one level of tables."""
    lines = [f"{key} = {_toml_value(value)}" for key, value in top.items()]
    for name, table in tables.items():
        lines.append(f"\n[{name}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in table.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise SolverError("Cannot pass NaN to the Julia engine.")
        return repr(value) if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, np.ndarray):
        return _toml_value(value.tolist())
    raise SolverError(f"Cannot write {value!r} to the Julia input file.")


def export_mesh_arrays(
    mesh_path: Path, out_dir: Path, dimension: int, prefix: str
) -> dict[str, Path]:
    """CSV node coordinates, 1-based cells and physical tags of the bulk mesh for Julia."""
    mesh = read_mesh(mesh_path)
    cell_type = "tetra" if dimension == 3 else "triangle"
    blocks = [block.data for block in mesh.cells if block.type == cell_type]
    if not blocks:
        raise SolverError(
            f"The {dimension}D solvers need linear {cell_type} cells; found "
            + ", ".join(sorted({block.type for block in mesh.cells}))
            + "."
        )
    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    if physical is None:
        raise SolverError("Mesh is missing gmsh physical tags for the bulk cells.")
    paths = {
        "nodes": out_dir / f"{prefix}_nodes.csv",
        "cells": out_dir / f"{prefix}_cells.csv",
        "cell_tags": out_dir / f"{prefix}_cell_tags.csv",
    }
    np.savetxt(paths["nodes"], np.asarray(mesh.points[:, :dimension]), delimiter=",", fmt="%.17g")
    cells = np.vstack([np.asarray(block, dtype=np.int64) for block in blocks]) + 1
    np.savetxt(paths["cells"], cells, delimiter=",", fmt="%d")
    tags = np.asarray(physical, dtype=np.int64).reshape(-1, 1)
    np.savetxt(paths["cell_tags"], tags, delimiter=",", fmt="%d")
    return {key: path.resolve() for key, path in paths.items()}
