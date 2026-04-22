from __future__ import annotations

from rve2d.config import config_from_dict


def test_solver_config_is_loaded() -> None:
    config = config_from_dict(
        {
            "mode": "synthetic",
            "synthetic": {
                "domain_width": 1.0,
                "domain_height": 1.0,
                "fibre_radius": 0.05,
                "target_volume_fraction": 0.2,
            },
            "solver": {
                "enabled": True,
                "boundary_condition": "dirichlet",
                "matrix_youngs_modulus": 3.0e9,
                "fibre_youngs_modulus": 70.0e9,
            },
        }
    )
    assert config.solver.enabled
    assert config.solver.boundary_condition == "dirichlet"

