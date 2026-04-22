from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.io import imsave

from rve2d.config import SemToSyntheticConfig
from rve2d.synthetic_generation.sem_image import generate_circular_fibre_rve_from_sem


def test_generate_circular_fibre_rve_from_sem_extracts_circle_measurements(tmp_path: Path) -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    rr, cc = disk((20, 22), 8, shape=image.shape)
    image[rr, cc] = 255
    rr, cc = disk((55, 52), 10, shape=image.shape)
    image[rr, cc] = 255
    image_path = tmp_path / "fibres.png"
    imsave(image_path, image)

    config = SemToSyntheticConfig(
        image_path=str(image_path),
        pixel_size=2.0,
        threshold=0.5,
        smoothing_sigma=0.0,
        min_artifact_area_px=20,
        min_region_area_px=20,
        drop_boundary_fibres=False,
    )
    result = generate_circular_fibre_rve_from_sem(config)

    assert result.extracted_fibre_count == 2
    radii = sorted(fibre.radius for fibre in result.geometry.circular_fibres)
    assert abs(radii[0] - 16.0) < 2.0
    assert abs(radii[1] - 20.0) < 2.0
    centers = sorted(
        (round(fibre.center_x, 1), round(fibre.center_y, 1))
        for fibre in result.geometry.circular_fibres
    )
    assert centers == [(44.0, 120.0), (104.0, 50.0)]


def test_generate_circular_fibre_rve_from_sem_separates_touching_fibres(tmp_path: Path) -> None:
    image = np.zeros((96, 96), dtype=np.uint8)
    rr, cc = disk((48, 38), 16, shape=image.shape)
    image[rr, cc] = 255
    rr, cc = disk((48, 58), 16, shape=image.shape)
    image[rr, cc] = 255
    image_path = tmp_path / "touching_fibres.png"
    imsave(image_path, image)

    config = SemToSyntheticConfig(
        image_path=str(image_path),
        pixel_size=1.0,
        threshold=0.5,
        smoothing_sigma=0.0,
        separation_min_distance_px=10,
        separation_peak_threshold_px=5.0,
        min_artifact_area_px=20,
        min_region_area_px=20,
        drop_boundary_fibres=False,
    )
    result = generate_circular_fibre_rve_from_sem(config)

    assert result.extracted_fibre_count == 2
