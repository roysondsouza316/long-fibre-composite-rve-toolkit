from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rve2d.config import ImageImportConfig
from rve2d.image_import.raster import extract_mask_polygons
from rve2d.models import Domain2D, GeometryModel, PolygonFibre, periodic_boundary_pairs


@dataclass(frozen=True)
class MaskImportResult:
    geometry: GeometryModel
    connected_regions: int
    removed_artifacts: int
    clipped_regions: int


def import_mask_geometry(config: ImageImportConfig) -> MaskImportResult:
    """Build a 2D geometry whose polygonal fibres are the foreground regions of a mask image.

    See :func:`rve2d.image_import.raster.extract_mask_polygons` for the image conventions.
    """
    extracted = extract_mask_polygons(config)
    domain = Domain2D(width=extracted.domain_width, height=extracted.domain_height)
    geometry = GeometryModel(
        domain=domain,
        polygonal_fibres=[
            PolygonFibre(points=polygon, fibre_id=index + 1)
            for index, polygon in enumerate(extracted.polygons)
        ],
        periodic_pairs=periodic_boundary_pairs(domain) if config.periodic_compatible else [],
        metadata={
            "generation_mode": "image",
            "image_path": str(Path(config.image_path)),
            "mask_shape": list(extracted.mask_shape),
            "connected_components": extracted.connected_regions,
            "removed_artifacts": extracted.removed_artifacts,
        },
    )
    return MaskImportResult(
        geometry=geometry,
        connected_regions=extracted.connected_regions,
        removed_artifacts=extracted.removed_artifacts,
        clipped_regions=extracted.clipped_regions,
    )
