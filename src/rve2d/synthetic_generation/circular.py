from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rve2d.config import SyntheticGenerationConfig
from rve2d.models import CircleFibre, Domain2D, GeometryModel, periodic_boundary_pairs
from rve2d.synthetic_generation.packing import pack_fibre_centres, resolve_fibre_count


@dataclass(frozen=True)
class SyntheticPlacementResult:
    geometry: GeometryModel
    attempts: int
    requested_fibre_count: int


def generate_circular_fibre_rve(config: SyntheticGenerationConfig) -> SyntheticPlacementResult:
    """Generate a 2D RVE of equal circular fibres (see ``SyntheticGenerationConfig``).

    With ``periodic_wrapping`` the returned geometry lists, for every fibre that crosses the
    domain boundary, its periodic images with the same ``fibre_id``; the mesher keeps the parts
    inside the domain.
    """
    domain = Domain2D(width=config.domain_width, height=config.domain_height)
    fibre_count = _resolve_fibre_count(config, domain.area)
    packing = pack_fibre_centres(
        config,
        fibre_count,
        domain.width,
        domain.height,
        origin_x=domain.origin_x,
        origin_y=domain.origin_y,
    )

    geometry = GeometryModel(
        domain=domain,
        circular_fibres=[
            CircleFibre(center_x=x, center_y=y, radius=config.fibre_radius, fibre_id=fibre_id)
            for fibre_id, x, y in packing.placements()
        ],
        periodic_pairs=periodic_boundary_pairs(domain) if config.periodic_compatible else [],
        metadata={
            "generation_mode": "synthetic",
            "orientation_deg": config.orientation_deg,
            "target_volume_fraction": config.target_volume_fraction,
            "requested_fibre_count": fibre_count,
            **packing.metadata(),
        },
    )
    return SyntheticPlacementResult(
        geometry=geometry,
        attempts=packing.attempts,
        requested_fibre_count=fibre_count,
    )


def _resolve_fibre_count(config: SyntheticGenerationConfig, domain_area: float) -> int:
    return resolve_fibre_count(config, domain_area, np.pi * config.fibre_radius**2)
