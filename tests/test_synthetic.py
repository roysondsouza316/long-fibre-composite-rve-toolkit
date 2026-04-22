from __future__ import annotations

from rve2d.config import SyntheticGenerationConfig
from rve2d.synthetic_generation.circular import generate_circular_fibre_rve


def test_generate_circular_fibre_rve_respects_spacing() -> None:
    config = SyntheticGenerationConfig(
        domain_width=1.0,
        domain_height=1.0,
        fibre_radius=0.05,
        target_volume_fraction=0.15,
        min_spacing=0.01,
        random_seed=42,
    )
    result = generate_circular_fibre_rve(config)
    fibres = result.geometry.circular_fibres
    for index, fibre_a in enumerate(fibres):
        for fibre_b in fibres[index + 1 :]:
            center_distance = (
                (fibre_a.center_x - fibre_b.center_x) ** 2
                + (fibre_a.center_y - fibre_b.center_y) ** 2
            ) ** 0.5
            assert center_distance >= fibre_a.radius + fibre_b.radius + config.min_spacing


def test_generate_circular_fibre_rve_hits_reasonable_volume_fraction() -> None:
    config = SyntheticGenerationConfig(
        domain_width=1.0,
        domain_height=1.0,
        fibre_radius=0.05,
        target_volume_fraction=0.2,
        random_seed=7,
    )
    result = generate_circular_fibre_rve(config)
    assert abs(result.geometry.fibre_volume_fraction - 0.2) < 0.02
