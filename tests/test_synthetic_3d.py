from __future__ import annotations

import pytest

from rve2d.config import SyntheticGenerationConfig
from rve2d.synthetic_generation.cylindrical import generate_cylindrical_fibre_rve
from rve2d.validation.checks import validate_geometry


def test_generate_cylindrical_fibre_rve_hits_reasonable_volume_fraction() -> None:
    config = SyntheticGenerationConfig(
        domain_width=1.0,
        domain_height=1.0,
        domain_depth=0.4,
        fibre_radius=0.05,
        target_volume_fraction=0.2,
        min_spacing=0.005,
        edge_clearance=0.01,
        random_seed=3,
    )
    result = generate_cylindrical_fibre_rve(config)
    assert result.geometry.dimension == 3
    assert result.geometry.fibre_volume_fraction == pytest.approx(0.2, abs=0.03)
    report = validate_geometry(result.geometry, minimum_spacing_requirement=0.005)
    assert report.valid


def test_generate_cylindrical_fibre_rve_sets_periodic_pairs() -> None:
    config = SyntheticGenerationConfig(
        domain_width=1.0,
        domain_height=1.0,
        domain_depth=0.5,
        fibre_radius=0.08,
        target_volume_fraction=0.1,
        periodic_compatible=True,
    )
    result = generate_cylindrical_fibre_rve(config)
    assert {pair.name for pair in result.geometry.periodic_pairs} == {
        "x_periodic",
        "y_periodic",
        "z_periodic",
    }
