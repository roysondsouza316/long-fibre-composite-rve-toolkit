"""End to end on a real gmsh mesh: build the periodic 2D example and homogenize it.

Needs the gmsh Python module (skipped otherwise). The Python engine always runs; the Julia
engine runs on the same mesh when its environment has been set up, and must agree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from rve2d.config import load_config
from rve2d.engines.julia import runner
from rve2d.workflow import build_and_solve_rve, solve_homogenization

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "2d" / "synthetic" / "periodic_solve.yaml"
)


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # ImportError, or OSError when system libraries (libGLU) are missing
        return False
    return True


@pytest.mark.skipif(not _gmsh_available(), reason="gmsh python module not available")
def test_periodic_example_with_both_engines(tmp_path: Path) -> None:
    config = load_config(EXAMPLE)
    build = build_and_solve_rve(config, output_dir=str(tmp_path), basename="rve", engine="python")
    result = build.homogenization_result
    assert result is not None and result.engine == "python"
    stiffness = np.asarray(result.homogenized_stiffness)
    assert stiffness.shape == (3, 3)
    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0, atol=1e-9 * stiffness.max())
    assert np.linalg.eigvalsh(stiffness).min() > 0.0
    constants = result.engineering_constants
    assert all(value > 0.0 for value in constants.values())

    if shutil.which("julia") is None or not runner.is_instantiated():
        pytest.skip("Julia engine environment not set up")
    mesh = tmp_path / "rve.msh"
    julia = solve_homogenization(config, mesh, tmp_path / "julia", engine="julia")
    np.testing.assert_allclose(
        julia.homogenized_stiffness, stiffness, rtol=0, atol=1e-10 * stiffness.max()
    )
