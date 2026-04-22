from __future__ import annotations

import numpy as np


def close_polygon(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float64)
    first = points[0]
    last = points[-1]
    if np.allclose(first, last):
        return points.astype(np.float64)
    return np.vstack([points, first]).astype(np.float64)


def polygon_area(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def filter_small_polygons(polygons: list[np.ndarray], minimum_area: float) -> list[np.ndarray]:
    return [
        polygon
        for polygon in polygons
        if polygon.shape[0] >= 4 and polygon_area(polygon) >= minimum_area
    ]
