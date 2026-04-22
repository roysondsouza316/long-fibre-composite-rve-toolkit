from __future__ import annotations

import pytest

from rve2d.config import ConfigError, config_from_dict, load_config


def test_3d_synthetic_config_requires_depth() -> None:
    with pytest.raises(ConfigError):
        config_from_dict(
            {
                "dimension": 3,
                "mode": "synthetic",
                "synthetic": {
                    "domain_width": 1.0,
                    "domain_height": 1.0,
                    "fibre_radius": 0.05,
                    "target_volume_fraction": 0.2,
                },
            }
        )


def test_3d_solver_requires_solid_kinematics() -> None:
    with pytest.raises(ConfigError):
        config_from_dict(
            {
                "dimension": 3,
                "mode": "synthetic",
                "synthetic": {
                    "domain_width": 1.0,
                    "domain_height": 1.0,
                    "domain_depth": 1.0,
                    "fibre_radius": 0.05,
                    "target_volume_fraction": 0.2,
                },
                "solver": {"enabled": True, "kinematics": "plane_strain"},
            }
        )


def test_3d_solver_accepts_axis_aligned_orthotropic_materials() -> None:
    config = config_from_dict(
        {
            "dimension": 3,
            "mode": "synthetic",
            "synthetic": {
                "domain_width": 1.0,
                "domain_height": 1.0,
                "domain_depth": 1.0,
                "fibre_radius": 0.05,
                "target_volume_fraction": 0.2,
            },
            "solver": {
                "enabled": True,
                "kinematics": "solid",
                "fibre_material_model": "orthotropic",
                "fibre_e1": 100.0,
                "fibre_e2": 10.0,
                "fibre_e3": 10.0,
                "fibre_g12": 5.0,
                "fibre_g13": 5.0,
                "fibre_g23": 4.0,
                "fibre_nu12": 0.2,
                "fibre_nu13": 0.2,
                "fibre_nu23": 0.25,
            },
        }
    )
    assert config.solver.fibre_material_model == "orthotropic"


def test_3d_solver_rejects_rotated_orthotropic_materials() -> None:
    config = config_from_dict(
        {
            "dimension": 3,
            "mode": "synthetic",
            "synthetic": {
                "domain_width": 1.0,
                "domain_height": 1.0,
                "domain_depth": 1.0,
                "fibre_radius": 0.05,
                "target_volume_fraction": 0.2,
            },
            "solver": {
                "enabled": True,
                "kinematics": "solid",
                "fibre_material_model": "orthotropic",
                "fibre_e1": 100.0,
                "fibre_e2": 10.0,
                "fibre_e3": 10.0,
                "fibre_g12": 5.0,
                "fibre_g13": 5.0,
                "fibre_g23": 4.0,
                "fibre_nu12": 0.2,
                "fibre_nu13": 0.2,
                "fibre_nu23": 0.25,
                "fibre_material_angle_x_deg": 10.0,
                "fibre_material_angle_y_deg": 5.0,
                "fibre_material_angle_z_deg": 20.0,
            },
        }
    )
    assert config.solver.fibre_material_angle_z_deg == pytest.approx(20.0)


def test_load_3d_example_config() -> None:
    config = load_config("examples/3d/synthetic/basic.yaml")
    assert config.dimension == 3
    assert config.synthetic is not None
    assert config.synthetic.domain_depth == pytest.approx(0.3)
