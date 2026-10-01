from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.io import imread
from skimage.segmentation import clear_border

from rve2d.config import ImageImportConfig
from rve2d.image_import.mask_to_geometry import (
    _contour_to_polygon,
    _count_clipped_regions,
    _remove_small_regions,
)
from rve2d.models import Domain3D, ExtrudedPolygonFibre, GeometryModel, periodic_boundary_pairs


@dataclass(frozen=True)
class MaskImport3DResult:
    geometry: GeometryModel
    connected_regions: int
    removed_artifacts: int
    clipped_regions: int


def import_mask_geometry_3d(config: ImageImportConfig) -> MaskImport3DResult:
    if config.extrusion_depth is None:
        raise ValueError("3D image import requires extrusion_depth.")
    image = imread(config.image_path, as_gray=True)
    mask = image > config.threshold
    if config.invert:
        mask = ~mask
    mask = ndimage.binary_fill_holes(mask)
    removed_artifacts = 0
    if config.min_artifact_area_px > 0:
        mask, removed_artifacts = _remove_small_regions(
            mask.astype(bool),
            config.min_artifact_area_px,
        )
    clipped_regions = _count_clipped_regions(mask)
    if config.clear_border:
        mask = clear_border(mask)  # type: ignore[no-untyped-call]

    _, connected_regions = ndimage.label(mask)
    contours = measure.find_contours(mask.astype(float), 0.5)  # type: ignore[no-untyped-call]
    polygons = [_contour_to_polygon(contour, config, mask.shape) for contour in contours]
    polygons = [polygon for polygon in polygons if polygon.shape[0] >= 4]

    domain_width = (
        config.domain_width
        if config.domain_width is not None
        else mask.shape[1] * config.pixel_size
    )
    domain_height = (
        config.domain_height
        if config.domain_height is not None
        else mask.shape[0] * config.pixel_size
    )
    domain = Domain3D(width=domain_width, height=domain_height, depth=config.extrusion_depth)
    geometry = GeometryModel(
        domain=domain,
        extruded_polygonal_fibres=[
            ExtrudedPolygonFibre(
                points=np.asarray(polygon, dtype=np.float64),
                z_min=domain.origin_z,
                z_max=domain.z_max,
                fibre_id=index + 1,
            )
            for index, polygon in enumerate(polygons)
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
            "mask_shape": list(mask.shape),
            "extrusion_depth": config.extrusion_depth,
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
            "connected_components": int(connected_regions),
            "removed_artifacts": removed_artifacts,
        },
    )
    return MaskImport3DResult(
        geometry=geometry,
        connected_regions=int(connected_regions),
        removed_artifacts=removed_artifacts,
        clipped_regions=clipped_regions,
    )
