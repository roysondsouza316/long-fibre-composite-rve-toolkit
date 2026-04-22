"""End-to-end integration test against the real Julia/Ferrite stack.

Skipped automatically when ``julia`` or the ``gmsh`` Python module is not
available, so this test is safe to ship and run in CI behind an opt-in flag.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "2d"
    / "synthetic"
    / "periodic_solve.yaml"
)


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:
        return False
    return True


def _julia_available() -> bool:
    return shutil.which("julia") is not None


@pytest.mark.skipif(not EXAMPLE.exists(), reason="periodic_solve example missing")
@pytest.mark.skipif(not _julia_available(), reason="julia not on PATH")
@pytest.mark.skipif(not _gmsh_available(), reason="gmsh python module not installed")
def test_julia_periodic_homogenization_2d(tmp_path: Path) -> None:
    from rve2d.config import load_config
    from rve2d.workflow import build_and_solve_rve

    config = load_config(EXAMPLE)
    result = build_and_solve_rve(config, output_dir=str(tmp_path), basename="rve_test")

    assert result.ferrite_result is not None
    stiffness = result.ferrite_result.homogenized_stiffness
    assert len(stiffness) == 3
    assert all(len(row) == 3 for row in stiffness)

    # Diagonal entries must be positive for a physically valid stiffness.
    for i in range(3):
        assert stiffness[i][i] > 0.0, f"non-positive C[{i},{i}]: {stiffness[i][i]}"

    # Symmetry within numerical tolerance (relative).
    for i in range(3):
        for j in range(i + 1, 3):
            avg = 0.5 * (abs(stiffness[i][j]) + abs(stiffness[j][i]))
            if avg == 0.0:
                continue
            rel = abs(stiffness[i][j] - stiffness[j][i]) / avg
            assert rel < 1e-3, (
                f"C not symmetric at ({i},{j}): "
                f"{stiffness[i][j]} vs {stiffness[j][i]}"
            )

    eng = result.ferrite_result.engineering_constants
    for key in ("ex", "ey", "gxy"):
        assert eng[key] > 0.0
