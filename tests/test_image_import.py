from __future__ import annotations

from pathlib import Path

from rve2d.config import ImageImportConfig
from rve2d.image_import.mask_to_geometry import import_mask_geometry


def test_import_mask_geometry_from_pgm(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.pgm"
    mask_path.write_text(
        "\n".join(
            [
                "P2",
                "8 8",
                "255",
                "0 0 0 0 0 0 0 0",
                "0 0 255 255 255 255 0 0",
                "0 0 255 255 255 255 0 0",
                "0 0 255 255 255 255 0 0",
                "0 0 255 255 255 255 0 0",
                "0 0 255 255 255 255 0 0",
                "0 0 0 0 0 0 0 0",
                "0 0 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    config = ImageImportConfig(
        image_path=str(mask_path),
        pixel_size=0.1,
        threshold=0.5,
        min_artifact_area_px=1,
        simplify_tolerance=0.0,
    )
    result = import_mask_geometry(config)
    assert result.geometry.fibre_count >= 1
    assert result.connected_regions == 1
    assert result.removed_artifacts == 0
