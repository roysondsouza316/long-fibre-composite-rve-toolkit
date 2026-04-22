from __future__ import annotations

from pathlib import Path

from rve2d.config import load_config


def test_image_example_enables_solver() -> None:
    config = load_config(Path("examples/2d/image/basic_solve.yaml"))
    assert config.mode == "image"
    assert config.solver.enabled
    assert config.solver.boundary_condition == "dirichlet"
    assert config.solver.write_vtk


def test_oriented_image_example_loads_phase_angles() -> None:
    config = load_config(Path("examples/2d/image/oriented_solve.yaml"))
    assert config.mode == "image"
    assert config.solver.matrix_material_model == "orthotropic"
    assert config.solver.fibre_material_model == "orthotropic"
    assert config.solver.matrix_material_angle_deg == 15.0
    assert config.solver.fibre_material_angle_deg == 30.0
