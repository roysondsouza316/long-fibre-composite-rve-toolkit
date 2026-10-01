from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rve2d.config import SyntheticGenerationConfig
from rve2d.exceptions import RVEError
from rve2d.models import CylinderFibre, Domain3D, GeometryModel, periodic_boundary_pairs


@dataclass(frozen=True)
class SyntheticPlacement3DResult:
    geometry: GeometryModel
    attempts: int
    requested_fibre_count: int


def generate_cylindrical_fibre_rve(config: SyntheticGenerationConfig) -> SyntheticPlacement3DResult:
    if config.domain_depth is None:
        raise RVEError("3D synthetic generation requires domain_depth.")
    domain = Domain3D(
        width=config.domain_width,
        height=config.domain_height,
        depth=config.domain_depth,
    )
    fibre_count = _resolve_fibre_count(config, domain.volume)
    rng = np.random.default_rng(config.random_seed)
    lower_x = domain.origin_x + config.fibre_radius + config.edge_clearance
    lower_y = domain.origin_y + config.fibre_radius + config.edge_clearance
    upper_x = domain.x_max - config.fibre_radius - config.edge_clearance
    upper_y = domain.y_max - config.fibre_radius - config.edge_clearance
    if lower_x >= upper_x or lower_y >= upper_y:
        raise RVEError("Fibre radius and edge clearance leave no room for placement.")

    min_center_distance = 2.0 * config.fibre_radius + config.min_spacing
    centres: list[tuple[float, float]] = []
    attempts = 0
    while len(centres) < fibre_count and attempts < config.max_attempts:
        attempts += 1
        candidate_x = float(rng.uniform(lower_x, upper_x))
        candidate_y = float(rng.uniform(lower_y, upper_y))
        if _is_non_overlapping(candidate_x, candidate_y, centres, min_center_distance):
            centres.append((candidate_x, candidate_y))

    if len(centres) != fibre_count:
        raise RVEError(
            "Could not place "
            f"{fibre_count} fibres without overlap after {config.max_attempts} attempts."
        )

    geometry = GeometryModel(
        domain=domain,
        cylindrical_fibres=[
            CylinderFibre(
                center_x=x,
                center_y=y,
                radius=config.fibre_radius,
                z_min=domain.origin_z,
                z_max=domain.z_max,
                fibre_id=index + 1,
            )
            for index, (x, y) in enumerate(centres)
        ],
        boundary_labels={
            "left": 11,
            "right": 12,
            "front": 13,
            "back": 14,
            "bottom": 15,
            "top": 16,
        },
        periodic_pairs=periodic_boundary_pairs(domain) if config.periodic_compatible else [],
        metadata={
            "generation_mode": "synthetic_3d",
            "dimension": 3,
            "orientation_vector": [0.0, 0.0, 1.0],
            "phase_orientation_rotations_deg": {
                "matrix": {
                    "x": (
                        0.0
                        if config.matrix_orientation_angle_x_deg is None
                        else float(config.matrix_orientation_angle_x_deg)
                    ),
                    "y": (
                        0.0
                        if config.matrix_orientation_angle_y_deg is None
                        else float(config.matrix_orientation_angle_y_deg)
                    ),
                    "z": (
                        0.0
                        if config.matrix_orientation_angle_z_deg is None
                        else float(config.matrix_orientation_angle_z_deg)
                    ),
                },
                "fibre": {
                    "x": (
                        0.0
                        if config.fibre_orientation_angle_x_deg is None
                        else float(config.fibre_orientation_angle_x_deg)
                    ),
                    "y": (
                        0.0
                        if config.fibre_orientation_angle_y_deg is None
                        else float(config.fibre_orientation_angle_y_deg)
                    ),
                    "z": (
                        0.0
                        if config.fibre_orientation_angle_z_deg is None
                        else float(config.fibre_orientation_angle_z_deg)
                    ),
                },
            },
            "target_volume_fraction": config.target_volume_fraction,
            "requested_fibre_count": fibre_count,
        },
    )
    return SyntheticPlacement3DResult(
        geometry=geometry,
        attempts=attempts,
        requested_fibre_count=fibre_count,
    )


def _resolve_fibre_count(config: SyntheticGenerationConfig, domain_volume: float) -> int:
    if config.fibre_count is not None:
        return config.fibre_count
    if config.domain_depth is None:
        raise RVEError("3D synthetic generation requires domain_depth.")
    fibre_volume = np.pi * config.fibre_radius**2 * config.domain_depth
    count = int(round(config.target_volume_fraction * domain_volume / fibre_volume))
    return max(count, 1)


def _is_non_overlapping(
    candidate_x: float,
    candidate_y: float,
    centres: list[tuple[float, float]],
    min_center_distance: float,
) -> bool:
    min_distance_sq = min_center_distance**2
    for center_x, center_y in centres:
        delta_x = candidate_x - center_x
        delta_y = candidate_y - center_y
        if delta_x * delta_x + delta_y * delta_y < min_distance_sq:
            return False
    return True
