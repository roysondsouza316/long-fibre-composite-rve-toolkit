from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.io import imread
from skimage.segmentation import clear_border

from rve2d.config import ImageImportConfig
from rve2d.geometry_cleanup.cleanup import close_polygon, filter_small_polygons
from rve2d.models import Domain2D, GeometryModel, PeriodicBoundaryPair, PolygonFibre


@dataclass(frozen=True)
class MaskImportResult:
    geometry: GeometryModel
    connected_regions: int
    removed_artifacts: int
    clipped_regions: int


def import_mask_geometry(config: ImageImportConfig) -> MaskImportResult:
    image_path = Path(config.image_path)
    image = imread(image_path, as_gray=True)
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
    polygons = filter_small_polygons(polygons, minimum_area=(config.pixel_size**2))

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
    domain = Domain2D(width=domain_width, height=domain_height)
    geometry = GeometryModel(
        domain=domain,
        polygonal_fibres=[
            PolygonFibre(points=polygon, fibre_id=index + 1)
            for index, polygon in enumerate(polygons)
        ],
        periodic_pairs=_periodic_pairs(domain) if config.periodic_compatible else [],
        metadata={
            "generation_mode": "image",
            "image_path": str(image_path),
            "mask_shape": list(mask.shape),
            "connected_components": int(connected_regions),
            "removed_artifacts": removed_artifacts,
        },
    )
    return MaskImportResult(
        geometry=geometry,
        connected_regions=int(connected_regions),
        removed_artifacts=removed_artifacts,
        clipped_regions=clipped_regions,
    )


def _contour_to_polygon(
    contour: np.ndarray,
    config: ImageImportConfig,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    simplified = measure.approximate_polygon(  # type: ignore[no-untyped-call]
        contour,
        tolerance=config.simplify_tolerance,
    )
    if simplified.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    rows, cols = mask_shape
    points = np.zeros((simplified.shape[0], 2), dtype=np.float64)
    points[:, 0] = simplified[:, 1] * config.pixel_size
    points[:, 1] = (rows - simplified[:, 0]) * config.pixel_size
    return close_polygon(points)


def _count_clipped_regions(mask: np.ndarray) -> int:
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    clipped_mask = mask & border
    _, count = ndimage.label(clipped_mask)
    return int(count)


def _remove_small_regions(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0

    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    kept_mask = keep[labels]
    removed_artifacts = int(np.count_nonzero((~keep)[1:]))
    return kept_mask, removed_artifacts


def _periodic_pairs(domain: Domain2D) -> list[PeriodicBoundaryPair]:
    return [
        PeriodicBoundaryPair(
            name="x_periodic",
            source="left",
            target="right",
            translation=(domain.width, 0.0),
        ),
        PeriodicBoundaryPair(
            name="y_periodic",
            source="bottom",
            target="top",
            translation=(0.0, domain.height),
        ),
    ]
