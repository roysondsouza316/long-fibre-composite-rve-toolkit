from __future__ import annotations

import math

import numpy as np
import pytest

from rve2d.models import (
    CircleFibre,
    CylinderFibre,
    Domain2D,
    Domain3D,
    ExtrudedPolygonFibre,
    GeometryModel,
    PolygonFibre,
    circle_rectangle_intersection_area,
    clip_polygon_to_rectangle,
    clipped_polygon_area,
)


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)


def test_circle_rectangle_area_special_cases() -> None:
    radius = 0.2
    full = math.pi * radius**2
    assert circle_rectangle_intersection_area(0.5, 0.5, radius, 0.0, 0.0, 1.0, 1.0) == full
    assert circle_rectangle_intersection_area(0.0, 0.5, radius, 0.0, 0.0, 1.0, 1.0) == (
        pytest.approx(full / 2.0, rel=1e-12)
    )
    assert circle_rectangle_intersection_area(1.0, 1.0, radius, 0.0, 0.0, 1.0, 1.0) == (
        pytest.approx(full / 4.0, rel=1e-12)
    )
    assert circle_rectangle_intersection_area(1.5, 0.5, radius, 0.0, 0.0, 1.0, 1.0) == 0.0
    # Circular segment of height h = radius - d cut off by the line x = 1.
    d = 0.1
    segment = radius**2 * math.acos(d / radius) - d * math.sqrt(radius**2 - d**2)
    assert circle_rectangle_intersection_area(1.0 - d, 0.5, radius, 0.0, 0.0, 1.0, 1.0) == (
        pytest.approx(full - segment, rel=1e-12)
    )


def test_circle_rectangle_area_matches_fine_polygon() -> None:
    rng = np.random.default_rng(1)
    angles = np.linspace(0.0, 2.0 * math.pi, 8192, endpoint=False)
    for _ in range(100):
        center_x, center_y = rng.uniform(-1.0, 2.0, size=2)
        radius = float(rng.uniform(0.05, 1.5))
        x0, y0 = rng.uniform(-1.0, 1.0, size=2)
        x1, y1 = x0 + rng.uniform(0.01, 2.0), y0 + rng.uniform(0.01, 2.0)
        polygon = np.column_stack(
            [center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)]
        )
        exact = circle_rectangle_intersection_area(center_x, center_y, radius, x0, y0, x1, y1)
        approximate = clipped_polygon_area(polygon, x0, y0, x1, y1)
        # The 8192-gon misses (2 pi / n)^2 / 6 ~ 1e-7 of the disk area.
        assert exact == pytest.approx(approximate, abs=2e-7 * math.pi * radius**2)


def test_polygon_clipping_handles_concave_polygons() -> None:
    # L-shaped polygon of area 3 whose two arms leave the unit-height strip 0 <= y <= 1.
    polygon = np.array(
        [[0.0, -1.0], [2.0, -1.0], [2.0, 0.5], [1.0, 0.5], [1.0, 2.0], [0.0, 2.0], [0.0, -1.0]]
    )
    assert clipped_polygon_area(polygon, -5.0, -5.0, 5.0, 5.0) == pytest.approx(4.5)
    assert clipped_polygon_area(polygon, -5.0, 0.0, 5.0, 1.0) == pytest.approx(1.0 + 0.5)
    clipped = clip_polygon_to_rectangle(polygon, -5.0, 0.0, 5.0, 1.0)
    assert clipped[:, 1].min() >= 0.0
    assert clipped[:, 1].max() <= 1.0
    assert clipped_polygon_area(polygon, 3.0, 3.0, 4.0, 4.0) == 0.0


def test_fibre_volume_fraction_clips_fibres_to_the_domain_2d() -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=10.0, height=10.0),
        circular_fibres=[
            CircleFibre(center_x=5.0, center_y=5.0, radius=1.0, fibre_id=1),
            CircleFibre(center_x=10.0, center_y=5.0, radius=1.0, fibre_id=2),  # half inside
            CircleFibre(center_x=20.0, center_y=5.0, radius=1.0, fibre_id=3),  # outside
        ],
        polygonal_fibres=[
            PolygonFibre(points=_square(-1.0, 2.0, 1.0, 4.0), fibre_id=4),  # half inside
            PolygonFibre(points=_square(12.0, 12.0, 14.0, 14.0), fibre_id=5),  # outside
        ],
    )
    expected = (math.pi + math.pi / 2.0 + 2.0) / 100.0
    assert geometry.fibre_volume_fraction == pytest.approx(expected, rel=1e-12)
    assert geometry.matrix_area == pytest.approx(100.0 * (1.0 - expected), rel=1e-12)


def test_fibre_volume_fraction_clips_fibres_to_the_domain_3d() -> None:
    geometry = GeometryModel(
        domain=Domain3D(width=10.0, height=10.0, depth=2.0),
        cylindrical_fibres=[
            CylinderFibre(
                center_x=0.0, center_y=0.0, radius=1.0, z_min=0.0, z_max=2.0, fibre_id=1
            ),
            CylinderFibre(
                center_x=5.0, center_y=5.0, radius=1.0, z_min=-1.0, z_max=1.0, fibre_id=2
            ),
        ],
        extruded_polygonal_fibres=[
            ExtrudedPolygonFibre(
                points=_square(9.0, 4.0, 11.0, 6.0), z_min=0.0, z_max=3.0, fibre_id=3
            ),
        ],
    )
    expected_volume = math.pi / 4.0 * 2.0 + math.pi * 1.0 + 2.0 * 2.0
    assert geometry.fibre_volume == pytest.approx(expected_volume, rel=1e-12)
    assert geometry.fibre_volume_fraction == pytest.approx(expected_volume / 200.0, rel=1e-12)


def test_fibre_count_counts_periodic_images_once() -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            CircleFibre(center_x=0.02, center_y=0.5, radius=0.05, fibre_id=1),
            CircleFibre(center_x=1.02, center_y=0.5, radius=0.05, fibre_id=1),
            CircleFibre(center_x=0.5, center_y=0.5, radius=0.05, fibre_id=2),
        ],
    )
    assert geometry.fibre_count == 2
    assert geometry.fibre_instance_count == 3
    assert geometry.fibre_volume_fraction == pytest.approx(2.0 * math.pi * 0.05**2, rel=1e-12)
