from __future__ import annotations

from dataclasses import dataclass

from rve2d.config import ImageImportConfig
from rve2d.image_import.raster import extract_mask_polygons
from rve2d.models import Domain3D, ExtrudedPolygonFibre, GeometryModel, periodic_boundary_pairs


@dataclass(frozen=True)
class MaskImport3DResult:
    geometry: GeometryModel
    connected_regions: int
    removed_artifacts: int
    clipped_regions: int


def import_mask_geometry_3d(config: ImageImportConfig) -> MaskImport3DResult:
    """Extrude the 2D mask polygons (see :func:`import_mask_geometry`) through the depth."""
    if config.extrusion_depth is None:
        raise ValueError("3D image import requires extrusion_depth.")
    extracted = extract_mask_polygons(config)
    domain = Domain3D(
        width=extracted.domain_width,
        height=extracted.domain_height,
        depth=config.extrusion_depth,
    )
    geometry = GeometryModel(
        domain=domain,
        extruded_polygonal_fibres=[
            ExtrudedPolygonFibre(
                points=polygon,
                z_min=domain.origin_z,
                z_max=domain.z_max,
                fibre_id=index + 1,
            )
            for index, polygon in enumerate(extracted.polygons)
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
            "generation_mode": "image_3d_extruded",
            "dimension": 3,
            "image_path": config.image_path,
            "mask_shape": list(extracted.mask_shape),
            "extrusion_depth": config.extrusion_depth,
            "phase_orientation_rotations_deg": {
                "matrix": {
                    "x": _angle(config.matrix_orientation_angle_x_deg),
                    "y": _angle(config.matrix_orientation_angle_y_deg),
                    "z": _angle(config.matrix_orientation_angle_z_deg),
                },
                "fibre": {
                    "x": _angle(config.fibre_orientation_angle_x_deg),
                    "y": _angle(config.fibre_orientation_angle_y_deg),
                    "z": _angle(config.fibre_orientation_angle_z_deg),
                },
            },
            "connected_components": extracted.connected_regions,
            "removed_artifacts": extracted.removed_artifacts,
        },
    )
    return MaskImport3DResult(
        geometry=geometry,
        connected_regions=extracted.connected_regions,
        removed_artifacts=extracted.removed_artifacts,
        clipped_regions=extracted.clipped_regions,
    )


def _angle(value: float | None) -> float:
    return 0.0 if value is None else float(value)
