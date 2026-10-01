"""Julia backend of the nonlinear solve (``nonlinear.backend: julia``).

The input-file test needs neither Julia nor PyTorch. The end-to-end test runs when ``julia``
is on PATH and the ``julia/NonlinearRVE`` environment has been instantiated (it has a
Manifest.toml); it compares the Julia and Python backends when PyTorch is installed too.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import tomllib
from pathlib import Path

import meshio
import numpy as np
import pytest

from rve2d.config import RVEConfig, config_from_dict
from rve2d.nonlinear.julia_bridge import JULIA_PROJECT, _toml_value, write_julia_input


def structured_mesh(path: Path, n: int = 8) -> Path:
    """Periodic 2D mesh of the unit square with a square fibre in [0.3, 0.7]^2."""
    xs = np.linspace(0.0, 1.0, n + 1)
    points = np.array([[x, y, 0.0] for y in xs for x in xs])
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    cells_arr = np.array(cells)
    centre = points[cells_arr, :2].mean(axis=1)
    phase = np.where(np.all((centre > 0.3) & (centre < 0.7), axis=1), 2, 1)
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("triangle", cells_arr)],
            cell_data={"gmsh:physical": [phase], "gmsh:geometrical": [phase]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    return path


def nonlinear_config(backend: str, **nonlinear: object) -> RVEConfig:
    section: dict[str, object] = {
        "enabled": True,
        "backend": backend,
        "matrix": {
            "youngs_modulus": 3500.0,
            "poisson_ratio": 0.35,
            "yield_stress": 40.0,
            "hardening_modulus": 300.0,
        },
        "fibre": {"youngs_modulus": 70000.0, "poisson_ratio": 0.2},
        "interface": {
            "penalty_stiffness": 1.0e6,
            "normal_strength": 20.0,
            "shear_strength": 30.0,
            "mode_i_toughness": 0.05,
            "mode_ii_toughness": 0.1,
        },
        "load": {"max_strain": 0.012, "steps": 8},
    }
    section.update(nonlinear)
    return config_from_dict(
        {
            "mode": "synthetic",
            "synthetic": {
                "domain_width": 1.0,
                "domain_height": 1.0,
                "fibre_radius": 0.2,
                "target_volume_fraction": 0.1,
                "periodic_compatible": True,
            },
            "nonlinear": section,
        }
    )


def test_julia_input_file_describes_the_solve(tmp_path: Path) -> None:
    mesh = structured_mesh(tmp_path / "rve.msh")
    path = write_julia_input(nonlinear_config("julia"), mesh, tmp_path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["dimension"] == 2 and data["kinematics"] == "generalized_plane_strain"
    assert data["stress_scale"] == 3500.0
    assert "yield_stress" not in data["fibre"]  # elastic phase
    assert data["interface"]["law"] == "bilinear_mixed_mode"
    assert "shear_penalty_stiffness" not in data["interface"]
    # Voigt components are 1-based for Julia: load xx, stress-free yy, zz, xy
    assert data["load"]["prescribed_components"] == [1]
    assert sorted(data["load"]["stress_components"]) == [2, 3, 6]
    assert sorted(data["load"]["fixed_zero"]) == [4, 5]
    nodes = np.loadtxt(data["nodes"], delimiter=",")
    cells = np.loadtxt(data["cells"], delimiter=",", dtype=int)
    assert nodes.shape == (81, 2) and cells.min() == 1 and cells.max() == 81
    assert _toml_value(float("inf")) == "inf" and _toml_value([1, 2]) == "[1, 2]"


def test_julia_backend_rejects_gpu() -> None:
    from rve2d.exceptions import ConfigError

    with pytest.raises(ConfigError, match="CPU"):
        nonlinear_config("julia", device="cuda")


def _julia_ready() -> bool:
    return shutil.which("julia") is not None and (JULIA_PROJECT / "Manifest.toml").exists()


requires_julia = pytest.mark.skipif(
    not _julia_ready(), reason="julia/NonlinearRVE environment not instantiated"
)


@requires_julia
def test_julia_backend_solves_plasticity_and_debonding(tmp_path: Path) -> None:
    from rve2d.workflow import solve_nonlinear

    mesh = structured_mesh(tmp_path / "rve.msh", n=12)
    result = solve_nonlinear(nonlinear_config("julia"), mesh, tmp_path)
    assert result.completed
    assert result.records[-1].max_damage > 0.1  # interface damage and plasticity are active
    assert result.records[-1].yielded_fraction > 0.0
    for name in ("nonlinear_summary.json", "nonlinear_final.vtu", "nonlinear_final_interface.vtu"):
        assert (tmp_path / name).exists()
    assert math.isfinite(result.peak_stress) and result.peak_stress > 0.0


@requires_julia
def test_julia_backend_matches_python_backend(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    from rve2d.workflow import solve_nonlinear

    mesh = structured_mesh(tmp_path / "rve.msh", n=12)
    config = nonlinear_config("julia")
    python_config = dataclasses.replace(
        config, nonlinear=dataclasses.replace(config.nonlinear, backend="python")
    )
    julia = solve_nonlinear(config, mesh, tmp_path / "julia")
    python = solve_nonlinear(python_config, mesh, tmp_path / "python")
    assert julia.completed and python.completed
    assert [r.time for r in julia.records] == pytest.approx([r.time for r in python.records])
    assert [r.iterations for r in julia.records] == [r.iterations for r in python.records]
    stress_p = np.array([r.macro_stress for r in python.records])
    stress_j = np.array([r.macro_stress for r in julia.records])
    assert np.abs(stress_p - stress_j).max() < 1e-9 * np.abs(stress_p).max()
    damage_p = [r.mean_damage for r in python.records]
    assert [r.mean_damage for r in julia.records] == pytest.approx(damage_p, abs=1e-10)
