"""Julia engine of the nonlinear solve (``nonlinear.engine: julia``).

The input-file test needs neither Julia nor PyTorch. The end-to-end test runs when ``julia``
is on PATH and the Julia engine environment has been set up (``rve2d doctor --setup-julia``);
it compares the Julia and TensorMesh engines when PyTorch is installed too.
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
from rve2d.engines.julia import runner
from rve2d.engines.julia.nonlinear import write_julia_input
from rve2d.engines.julia.runner import _toml_value


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


def nonlinear_config(engine: str, **nonlinear: object) -> RVEConfig:
    section: dict[str, object] = {
        "enabled": True,
        "engine": engine,
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


def test_julia_engine_rejects_gpu() -> None:
    from rve2d.exceptions import ConfigError

    with pytest.raises(ConfigError, match="CPU"):
        nonlinear_config("julia", device="cuda")


def _julia_ready() -> bool:
    return shutil.which("julia") is not None and runner.is_instantiated()


requires_julia = pytest.mark.skipif(
    not _julia_ready(), reason="Julia engine environment not set up"
)


@requires_julia
def test_julia_engine_solves_plasticity_and_debonding(tmp_path: Path) -> None:
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
def test_julia_engine_matches_tensormesh_engine(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    from rve2d.workflow import solve_nonlinear

    mesh = structured_mesh(tmp_path / "rve.msh", n=12)
    config = nonlinear_config("julia")
    tm_config = dataclasses.replace(
        config, nonlinear=dataclasses.replace(config.nonlinear, engine="tensormesh")
    )
    julia = solve_nonlinear(config, mesh, tmp_path / "julia")
    tensormesh = solve_nonlinear(tm_config, mesh, tmp_path / "tensormesh")
    assert julia.completed and tensormesh.completed
    assert [r.time for r in julia.records] == pytest.approx([r.time for r in tensormesh.records])
    assert [r.iterations for r in julia.records] == [r.iterations for r in tensormesh.records]
    stress_p = np.array([r.macro_stress for r in tensormesh.records])
    stress_j = np.array([r.macro_stress for r in julia.records])
    assert np.abs(stress_p - stress_j).max() < 1e-9 * np.abs(stress_p).max()
    damage_p = [r.mean_damage for r in tensormesh.records]
    assert [r.mean_damage for r in julia.records] == pytest.approx(damage_p, abs=1e-10)


@requires_julia
def test_engines_agree_on_longitudinal_shear_of_an_extruded_layer(tmp_path: Path) -> None:
    """The shear curve of the laminate pipeline for a 2D RVE: xz shear on one periodic layer
    of tetrahedra extruded from the 2D mesh, where every node is on the top or bottom face."""
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    from rve2d.mesh_io import extrude_mesh
    from rve2d.workflow import solve_nonlinear

    flat = structured_mesh(tmp_path / "flat.msh", n=8)
    mesh = extrude_mesh(flat, tmp_path / "layer.msh", depth=0.125)
    base = nonlinear_config("julia")
    load = dataclasses.replace(base.nonlinear.load, component="xz", max_strain=0.04, steps=8)
    nonlinear = dataclasses.replace(base.nonlinear, kinematics="solid", load=load)
    config = dataclasses.replace(base, dimension=3, nonlinear=nonlinear)
    tm_config = dataclasses.replace(
        config, nonlinear=dataclasses.replace(nonlinear, engine="tensormesh")
    )
    julia = solve_nonlinear(config, mesh, tmp_path / "julia")
    tensormesh = solve_nonlinear(tm_config, mesh, tmp_path / "tensormesh")
    assert julia.completed and tensormesh.completed
    assert tensormesh.records[-1].yielded_fraction > 0.0  # nonlinear in shear
    assert [r.iterations for r in julia.records] == [r.iterations for r in tensormesh.records]
    stress_p = np.array([r.macro_stress for r in tensormesh.records])
    stress_j = np.array([r.macro_stress for r in julia.records])
    assert np.abs(stress_p - stress_j).max() < 1e-9 * np.abs(stress_p).max()
    # only the loaded shear carries stress (uniaxial stress in xz)
    assert np.abs(stress_p[:, [0, 1, 2, 3, 5]]).max() < 1e-6 * np.abs(stress_p[:, 4]).max()


@requires_julia
@pytest.mark.parametrize("saturation", [None, 55.0])
def test_engines_agree_with_a_pressure_dependent_damaging_matrix(
    tmp_path: Path, saturation: float | None
) -> None:
    """Paraboloidal plasticity (compressive yield 1.5x tensile, plastic Poisson 0.3) with
    linear or Voce hardening and ductile damage in the matrix, under transverse
    compression."""
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    from rve2d.workflow import solve_nonlinear

    mesh = structured_mesh(tmp_path / "rve.msh", n=12)
    matrix = {
        "youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yield_stress": 40.0,
        "hardening_modulus": 300.0, "compressive_yield_stress": 60.0,
        "plastic_poisson_ratio": 0.3, "damage_onset_strain": 0.002, "fracture_energy": 0.3,
    }  # fmt: skip
    if saturation is not None:
        matrix.update({"saturation_stress": saturation, "hardening_modulus": 3000.0})
    config = nonlinear_config(
        "julia", matrix=matrix, load={"max_strain": -0.03, "steps": 12}
    )
    tm_config = dataclasses.replace(
        config, nonlinear=dataclasses.replace(config.nonlinear, engine="tensormesh")
    )
    julia = solve_nonlinear(config, mesh, tmp_path / "julia")
    tensormesh = solve_nonlinear(tm_config, mesh, tmp_path / "tensormesh")
    assert julia.completed and tensormesh.completed
    assert tensormesh.records[-1].mean_bulk_damage > 0.05  # the matrix damages ...
    stress_xx = [r.macro_stress[0] for r in tensormesh.records]
    assert abs(stress_xx[-1]) < 0.5 * max(abs(s) for s in stress_xx)  # ... and softens
    assert [r.iterations for r in julia.records] == [r.iterations for r in tensormesh.records]
    stress_p = np.array([r.macro_stress for r in tensormesh.records])
    stress_j = np.array([r.macro_stress for r in julia.records])
    assert np.abs(stress_p - stress_j).max() < 1e-9 * np.abs(stress_p).max()
    damage_p = [r.mean_bulk_damage for r in tensormesh.records]
    assert [r.mean_bulk_damage for r in julia.records] == pytest.approx(damage_p, abs=1e-10)
