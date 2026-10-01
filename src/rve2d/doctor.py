"""``rve2d doctor``: check what is installed for meshing and for each solver engine."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = False


def run_checks(setup_julia: bool = False) -> list[Check]:
    """Environment checks; ``setup_julia`` also instantiates the Julia engine environment."""
    checks = [
        Check("python", sys.version_info >= (3, 11), sys.version.split()[0], required=True),
        _module("numpy", required=True),
        _module("scipy", required=True),
        _module("meshio", required=True),
        _module("gmsh", "meshing (pip install gmsh, plus libGLU on Linux)", required=True),
        _module("h5py", "XDMF export (the default mesh format)", required=True),
        _module("torch", "TensorMesh engine, nonlinear solve (pip install '.[nonlinear]')"),
        _module("diffcohesive", "TensorMesh engine, cohesive laws (pip install '.[nonlinear]')"),
        _module("tensormesh", "TensorMesh engine, GPU sparse solves (pip install '.[nonlinear]')"),
    ]
    checks.extend(_julia_checks(setup_julia))
    return checks


def _module(name: str, purpose: str = "", required: bool = False) -> Check:
    try:
        importlib.import_module(name)
    except ImportError:
        return Check(
            name, False, f"not installed - {purpose}" if purpose else "not installed", required
        )
    except OSError as exc:  # e.g. gmsh without its system libraries
        return Check(name, False, f"installed but cannot load: {exc}", required)
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "installed"
    return Check(name, True, f"{version} - {purpose}" if purpose else version, required)


def _julia_checks(setup: bool) -> list[Check]:
    from rve2d.engines.julia import runner
    from rve2d.exceptions import SolverError

    try:
        julia = runner.julia_executable()
    except SolverError:
        return [Check("julia", False, "not found - Julia engine (julialang.org, 1.11 or newer)")]
    version = runner.julia_version(julia)
    if version is None:
        return [Check("julia", False, f"{julia} does not report a version")]
    text = ".".join(map(str, version))
    if version[:2] < runner.MIN_JULIA:
        return [Check("julia", False, f"{text} at {julia}; the Julia engine needs 1.11+")]
    checks = [Check("julia", True, f"{text} at {julia}")]
    if setup and not runner.is_instantiated():
        try:
            runner.ensure_environment(julia)
        except SolverError as exc:
            checks.append(Check("julia engine", False, str(exc)))
            return checks
    if runner.is_instantiated():
        checks.append(Check("julia engine", True, f"ready ({runner.project_dir()})"))
    else:
        checks.append(
            Check(
                "julia engine",
                False,
                "not set up yet: happens on first use, or run `rve2d doctor --setup-julia`",
            )
        )
    return checks


def format_checks(checks: list[Check]) -> str:
    width = max(len(check.name) for check in checks)
    lines = []
    for check in checks:
        mark = "ok " if check.ok else ("ERR" if check.required else " - ")
        lines.append(f"[{mark}] {check.name.ljust(width)}  {check.detail}")
    return "\n".join(lines)
