"""Solver engines of rve2d.

Two interchangeable engines solve the same problems with the same formulations and give the
same results to round-off:

* ``python`` (:mod:`rve2d.engines.python`): linear homogenization with NumPy/SciPy (no extra
  dependencies) and the nonlinear solve (plasticity + cohesive interfaces) with PyTorch,
  diffcohesive and TensorMesh (``pip install "rve2d-fibre[nonlinear]"``).
* ``julia`` (:mod:`rve2d.engines.julia`): Ferrite.jl through the bundled ``FerriteRVE``
  Julia package, with the ``DiffCohesive`` Julia package for the cohesive laws; needs
  Julia 1.11+ (the environment is set up on first use).

Engine-independent inputs and outputs live in :mod:`rve2d.engines.common`. Choose the engine
per config section: ``solver.engine`` (linear homogenization) and ``nonlinear.engine``.
"""

from __future__ import annotations

from pathlib import Path

from rve2d.config import RVEConfig
from rve2d.engines.common.homogenization import HomogenizationResult
from rve2d.engines.common.homogenization import run_homogenization as _run_homogenization
from rve2d.engines.common.records import NonlinearResult
from rve2d.exceptions import ConfigError

ENGINES = ("python", "julia")

__all__ = ["ENGINES", "HomogenizationResult", "NonlinearResult", "homogenize", "solve_nonlinear"]


def homogenize(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    geometry_metadata: dict[str, object] | None = None,
    engine: str | None = None,
) -> HomogenizationResult:
    """Linear homogenization (``solver`` section) with ``engine`` or ``solver.engine``."""
    chosen = _check(engine or config.solver.engine)
    if chosen == "julia":
        from rve2d.engines.julia.homogenization import solve
    else:
        from rve2d.engines.python.homogenization import solve
    return _run_homogenization(config, mesh_path, output_dir, chosen, solve, geometry_metadata)


def solve_nonlinear(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    engine: str | None = None,
) -> NonlinearResult:
    """Nonlinear solve (``nonlinear`` section) with ``engine`` or ``nonlinear.engine``."""
    chosen = _check(engine or config.nonlinear.engine)
    if chosen == "julia":
        from rve2d.engines.julia.nonlinear import run_nonlinear
    else:
        from rve2d.engines.python.nonlinear.driver import run_nonlinear
    return run_nonlinear(config, mesh_path, output_dir)


def _check(engine: str) -> str:
    if engine not in ENGINES:
        raise ConfigError(f"Unknown engine {engine!r}; use one of {', '.join(ENGINES)}.")
    return engine
