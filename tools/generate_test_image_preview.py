from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.filters import gaussian
from skimage.io import imread, imsave

from rve2d.config import load_config
from rve2d.image_import.mask_to_geometry import import_mask_geometry


def main() -> int:
    image = imread("examples/2d/sem/sem_sample.png", as_gray=True)
    smoothed = gaussian(image, sigma=2.0, preserve_range=True)
    mask = smoothed > 0.42
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))

    output_dir = Path("outputs/2d/sem/import")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "preprocessed_mask.png"
    imsave(mask_path, (mask.astype(np.uint8) * 255))

    config = load_config("examples/2d/sem/masked_import.yaml")
    if config.image is None:
        raise RuntimeError("Image config is required.")

    result = import_mask_geometry(config.image)

    rgb = np.dstack([image, image, image]).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    overlay = rgb.copy()
    overlay[mask] = 0.65 * overlay[mask] + 0.35 * np.array([1.0, 0.1, 0.1], dtype=np.float32)

    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:
        ij = np.rint(contour).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, overlay.shape[0] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, overlay.shape[1] - 1)
        overlay[ij[:, 0], ij[:, 1]] = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    preview_path = output_dir / "sem_import_preview.png"
    imsave(preview_path, (overlay * 255).astype(np.uint8))

    summary = {
        "preview_path": str(preview_path),
        "mask_path": str(mask_path),
        "connected_regions": result.connected_regions,
        "removed_artifacts": result.removed_artifacts,
        "clipped_regions": result.clipped_regions,
        "fibre_count": result.geometry.fibre_count,
        "domain_width": result.geometry.domain.width,
        "domain_height": result.geometry.domain.height,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
