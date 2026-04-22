from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.transform import hough_circle, hough_circle_peaks


def main() -> int:
    image = imread("examples/2d/sem/sem_sample.png", as_gray=True).astype(float)
    smoothed = gaussian(image, sigma=1.5, preserve_range=True)
    edges = canny(smoothed, sigma=2.0, low_threshold=0.08, high_threshold=0.2)
    radii = np.arange(18, 46, 2)
    hough_responses = hough_circle(edges, radii)
    accums, center_xs, center_ys, detected_radii = hough_circle_peaks(
        hough_responses,
        radii,
        total_num_peaks=120,
        min_xdistance=12,
        min_ydistance=12,
        threshold=0.22 * np.max(hough_responses),
    )
    candidates = sorted(
        zip(accums, center_xs, center_ys, detected_radii, strict=True),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    kept: list[tuple[float, int, int, int]] = []
    for score, center_x, center_y, radius in candidates:
        if (
            center_x - radius < 4
            or center_y - radius < 4
            or center_x + radius >= image.shape[1] - 4
            or center_y + radius >= image.shape[0] - 4
        ):
            continue
        keep = True
        for _, other_x, other_y, other_radius in kept:
            distance = float(np.hypot(center_x - other_x, center_y - other_y))
            if distance < max(radius, other_radius) * 0.85:
                keep = False
                break
            if distance < radius + other_radius + 4 and distance < min(radius, other_radius) * 1.3:
                keep = False
                break
        if keep:
            kept.append((float(score), int(center_x), int(center_y), int(radius)))

    overlay = np.dstack([image, image, image]).astype(float)
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    for _, center_x, center_y, radius in kept:
        distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        ring = np.abs(distances - radius) <= 1.2
        fill = distances < radius
        overlay[ring] = np.array([0.0, 1.0, 1.0])
        overlay[fill] = 0.85 * overlay[fill] + 0.15 * np.array([1.0, 0.15, 0.15])

    output_dir = Path("outputs/2d/sem/hough_proto")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "hough_preview.png"
    imsave(preview_path, np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8))
    print({"count": len(kept), "preview": str(preview_path), "first10": kept[:10]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
