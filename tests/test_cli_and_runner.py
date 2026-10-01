"""Mesh reading, CLI error reporting and the Julia engine's environment handling.

None of these need Julia, gmsh or PyTorch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _meshes import square_msh
from pytest import CaptureFixture, MonkeyPatch

from rve2d.cli import main
from rve2d.engines.julia import runner
from rve2d.exceptions import MeshingError
from rve2d.mesh_io import read_mesh

CONFIG = """\
mode: synthetic
synthetic: {domain_width: 1.0, domain_height: 1.0, fibre_radius: 0.1,
            target_volume_fraction: 0.1, periodic_compatible: true}
solver: {enabled: true, kinematics: plane_strain, boundary_condition: periodic}
"""


def test_read_mesh_is_quiet_and_reports_bad_files(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    mesh = read_mesh(square_msh(tmp_path / "rve.msh"))
    assert mesh.points.shape[0] == 81
    assert capsys.readouterr().out == ""  # meshio.read prints a failed ANSYS attempt
    with pytest.raises(FileNotFoundError, match="not found"):
        read_mesh(tmp_path / "missing.msh")
    (tmp_path / "bad.msh").write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n1 0 0")
    with pytest.raises(MeshingError, match="bad.msh"):  # meshio.read would exit the process
        read_mesh(tmp_path / "bad.msh")


def test_cli_reports_errors_in_one_line(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config = tmp_path / "solve.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    out = ["--output-dir", str(tmp_path / "out")]
    assert main(["solve", str(config), str(tmp_path / "missing.msh"), *out]) == 1
    error = capsys.readouterr().err
    assert error.startswith("rve2d: error: Mesh file not found") and error.count("\n") == 1
    with pytest.raises(FileNotFoundError):
        main(["--traceback", "solve", str(config), str(tmp_path / "missing.msh"), *out])
    config.write_text(CONFIG.replace("plane_strain", "solid"), encoding="utf-8")
    assert main(["validate-config", str(config)]) == 1
    assert "2D solves use kinematics" in capsys.readouterr().err


def test_cli_solve_writes_the_summary(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config = tmp_path / "solve.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    mesh = square_msh(tmp_path / "rve.msh")
    assert main(["solve", str(config), str(mesh), "--output-dir", str(tmp_path / "out")]) == 0
    assert '"engine": "tensormesh"' in capsys.readouterr().out
    assert (tmp_path / "out" / "homogenization_summary.json").exists()


def test_read_only_install_is_copied_to_the_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    real_access = os.access
    installed = runner.ENGINE_DIR / runner.PROJECT

    def access(path: str | os.PathLike[str], mode: int) -> bool:
        return False if Path(path) == installed else real_access(path, mode)

    monkeypatch.setattr(runner.os, "access", access)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = runner.engine_dir()
    assert target.parent == tmp_path / "rve2d" and target.name.startswith("julia-engine-")
    assert runner.engine_dir() == target  # stable for unchanged sources
    assert (target / runner.ENTRY).is_file()
    for package in ("FerriteRVE", "DiffCohesive"):
        assert (target / package / "Project.toml").is_file()
        assert (target / package / "src" / f"{package}.jl").is_file()
    copied = {path.name for path in target.rglob("*")}
    assert not any(name.endswith(".py") for name in copied)
    assert "Manifest.toml" not in copied and runner.SETUP_STAMP not in copied
    assert runner.project_dir() == target / "FerriteRVE"
    assert not runner.is_instantiated()


def test_environment_is_set_up_again_when_the_dependencies_change(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    for package in ("FerriteRVE", "DiffCohesive"):
        (tmp_path / package).mkdir()
        (tmp_path / package / "Project.toml").write_text(f'name = "{package}"\n')
    project = tmp_path / "FerriteRVE"
    monkeypatch.setattr(runner, "project_dir", lambda: project)
    (project / "Manifest.toml").write_text("")
    assert not runner.is_instantiated()  # no stamp: never set up by rve2d
    (project / runner.SETUP_STAMP).write_text(runner._dependency_digest(project) + "\n")
    assert runner.is_instantiated()
    (tmp_path / "DiffCohesive" / "Project.toml").write_text('name = "DiffCohesive"\n[deps]\n')
    assert not runner.is_instantiated()


def test_missing_julia_is_reported(monkeypatch: MonkeyPatch) -> None:
    from rve2d.exceptions import SolverError

    monkeypatch.setenv("RVE2D_JULIA", "")
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    with pytest.raises(SolverError, match="engine: tensormesh"):
        runner.julia_executable()


def test_engine_names_and_removed_options() -> None:
    from rve2d.config import ConfigError, config_from_dict

    base = {
        "mode": "synthetic",
        "synthetic": {
            "domain_width": 1.0,
            "domain_height": 1.0,
            "fibre_radius": 0.1,
            "target_volume_fraction": 0.1,
            "periodic_compatible": True,
        },
    }
    legacy = config_from_dict(base | {"solver": {"enabled": True, "engine": "python"}})
    assert legacy.solver.engine == "tensormesh"  # older configs keep working
    with pytest.raises(ConfigError, match="tensormesh"):
        config_from_dict(base | {"solver": {"enabled": True, "engine": "fortran"}})
    with pytest.raises(ConfigError, match="PARDISO"):
        config_from_dict(base | {"nonlinear": {"linear_solver": "pardiso"}})
