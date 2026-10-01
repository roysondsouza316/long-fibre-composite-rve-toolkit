from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rve2d.config import SyntheticGenerationConfig
from rve2d.exceptions import RVEError
from rve2d.models import CylinderFibre, Domain3D, GeometryModel, periodic_boundary_pairs
from rve2d.synthetic_generation.packing import pack_fibre_centres, resolve_fibre_count


@dataclass(frozen=True)
class SyntheticPlacement3DResult:
    geometry: GeometryModel
    attempts: int
    requested_fibre_count: int


def generate_cylindrical_fibre_rve(config: SyntheticGenerationConfig) -> SyntheticPlacement3DResult:
    """Generate a 3D RVE of equal cylindrical fibres along z (see ``SyntheticGenerationConfig``).

    The cross-section is packed exactly like the 2D generator; every cylinder spans the full
    domain depth. With ``periodic_wrapping`` cylinders crossing an x or y face are listed
    together with their periodic images (same ``fibre_id``).
    """
    if config.domain_depth is None:
        raise RVEError("3D synthetic generation requires domain_depth.")
    domain = Domain3D(
        width=config.domain_width,
        height=config.domain_height,
        depth=config.domain_depth,
    )
    fibre_count = _resolve_fibre_count(config, domain.volume)
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
        cylindrical_fibres=[
            CylinderFibre(
                center_x=x,
                center_y=y,
                radius=config.fibre_radius,
                z_min=domain.origin_z,
                z_max=domain.z_max,
                fibre_id=fibre_id,
            )
            for fibre_id, x, y in packing.placements()
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
            **packing.metadata(),
        },
    )
    return SyntheticPlacement3DResult(
        geometry=geometry,
        attempts=packing.attempts,
        requested_fibre_count=fibre_count,
    )


def _resolve_fibre_count(config: SyntheticGenerationConfig, domain_volume: float) -> int:
    if config.fibre_count is not None:
        return config.fibre_count
    if config.domain_depth is None:
        raise RVEError("3D synthetic generation requires domain_depth.")
    fibre_volume = np.pi * config.fibre_radius**2 * config.domain_depth
    return resolve_fibre_count(config, domain_volume, fibre_volume)
