"""Tests for the image -> mask -> polygon pipeline shared by the 2D and 3D image imports.

Area tolerances. Contours are traced by marching squares at level 0.5 on the binary mask, so
they run along pixel edges except at pixel corners: every convex corner of the pixelated
outline loses a right triangle with 0.5 px legs (1/8 px^2) and every concave corner gains one.
Walking once around a hole-free outline turns by 360 degrees, so (#convex - #concave) corners
= 4 for a fibre inside the image and the polygon area is exactly ``pixels - 0.5 px^2``. Where
a fibre is cut by the domain boundary the outline meets the boundary at right angles that are
*not* cut (the border pixels are replicated outwards before tracing), so with J such junctions
and K domain corners inside the fibre the deficit is ``(4 - J - K) / 8 px^2``. The tests use
``simplify_tolerance=0`` to check these values to floating-point accuracy; Douglas-Peucker
simplification is a separate, deliberately lossy step whose effect is tested separately
(boundary vertices preserved, polygons stay valid).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import ndimage
from skimage.draw import disk
from skimage.filters import gaussian
from skimage.io import imsave

from rve2d.config import ImageImportConfig, load_config
from rve2d.image_import import raster
from rve2d.image_import.mask_to_geometry import import_mask_geometry
from rve2d.image_import.mask_to_geometry_3d import import_mask_geometry_3d
from rve2d.image_import.raster import mask_to_polygons, normalise_image
from rve2d.models import Domain2D

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _save(path: Path, data: np.ndarray) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(path, data, check_contrast=False)
    return path


def _write_pgm16(path: Path, data: np.ndarray) -> Path:
    rows, cols = data.shape
    header = f"P5\n{cols} {rows}\n65535\n".encode()
    path.write_bytes(header + data.astype(">u2").tobytes())
    return path


def _area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _segments_intersect(p: np.ndarray, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> bool:
    """Closed segments pq and rs share at least one point (exact for our coordinates)."""

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return bool(
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    d1, d2 = orient(r, s, p), orient(r, s, q)
    d3, d4 = orient(p, q, r), orient(p, q, s)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (d1 == 0 and on_segment(r, s, p))
        or (d2 == 0 and on_segment(r, s, q))
        or (d3 == 0 and on_segment(p, q, r))
        or (d4 == 0 and on_segment(p, q, s))
    )


def _ring_defect(ring: np.ndarray) -> str | None:
    """Brute-force check of an open ring; returns why it is not a simple polygon, or None."""
    count = ring.shape[0]
    if count < 3:
        return "fewer than three vertices"
    edges = np.roll(ring, -1, axis=0) - ring
    if not np.all(np.hypot(edges[:, 0], edges[:, 1]) > 0.0):
        return "zero-length edge"
    if _area(np.vstack([ring, ring[:1]])) == 0.0:
        return "zero area"
    for i in range(count):
        a, b = edges[i - 1], edges[i]
        if a[0] * b[1] - a[1] * b[0] == 0.0 and a @ b < 0.0:
            return f"edge {i} folds back onto the previous one"
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            if _segments_intersect(ring[i], ring[(i + 1) % count], ring[j], ring[(j + 1) % count]):
                return f"edges {i} and {j} intersect"
    return None


def _assert_valid_polygon(points: np.ndarray, width: float, height: float) -> None:
    """Closed, counter-clockwise, inside the domain, no degenerate edges, no self-contact."""
    assert points.ndim == 2 and points.shape[1] == 2
    assert np.array_equal(points[0], points[-1]), "polygon must be closed"
    ring = points[:-1]
    assert _area(points) > 0.0, "polygon must be counter-clockwise"
    assert ring[:, 0].min() >= 0.0 and ring[:, 0].max() <= width
    assert ring[:, 1].min() >= 0.0 and ring[:, 1].max() <= height
    defect = _ring_defect(ring)
    assert defect is None, defect


def _boundary_vertices(points: np.ndarray, width: float, height: float) -> np.ndarray:
    ring = points[:-1]
    on_boundary = (
        (ring[:, 0] == 0.0)
        | (ring[:, 0] == width)
        | (ring[:, 1] == 0.0)
        | (ring[:, 1] == height)
    )
    return ring[on_boundary]


def _marching_squares_area_px(mask: np.ndarray) -> float:
    """Exact area enclosed by the level-0.5 contours of a hole-free mask, in px^2.

    Each 2x2 cell with one foreground pixel cuts a 1/8 px^2 corner, each cell with three
    adds one, and a saddle cell (two diagonal pixels, kept apart) cuts two. Cells that would
    involve pixels outside the image do not contribute: fibres end square on the boundary.
    """
    m = mask.astype(np.int64)
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    total = a + b + c + d
    ones = np.count_nonzero(total == 1)
    threes = np.count_nonzero(total == 3)
    saddles = np.count_nonzero((total == 2) & (a == d))
    return float(m.sum()) - ones / 8.0 + threes / 8.0 - saddles / 4.0


def _config(path: Path, **kwargs: Any) -> ImageImportConfig:
    options: dict[str, Any] = {
        "pixel_size": 0.5,
        "threshold": 0.5,
        "min_artifact_area_px": 1,
        "simplify_tolerance": 0.0,
    }
    options.update(kwargs)
    return ImageImportConfig(image_path=str(path), **options)


def _squares_mask() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:10, 0:10] = True  # top-left corner
    mask[15:25, 30:40] = True  # right edge
    mask[15:25, 12:22] = True  # interior
    mask[30:40, 30:40] = True  # bottom-right corner
    mask[30:40, 5:15] = True  # bottom edge
    return mask


# --------------------------------------------------------------------------------------------
# Image normalisation (8-bit greyscale used to be compared on a 0..255 scale)
# --------------------------------------------------------------------------------------------


def test_normalise_image_maps_every_storage_format_to_unit_grey() -> None:
    grey = np.array([[0, 51, 128, 255]], dtype=np.uint8)
    expected = grey / 255.0
    np.testing.assert_allclose(normalise_image(grey), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        normalise_image(grey.astype(np.uint16) * 257), expected, rtol=0, atol=1e-15
    )
    # Pillow returns 16-bit PGM data in an int32 array.
    np.testing.assert_allclose(
        normalise_image(grey.astype(np.int32) * 257), expected, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(normalise_image(np.dstack([grey] * 3)), normalise_image(grey))
    rgba = np.dstack([grey] * 3 + [np.zeros_like(grey)])  # alpha is ignored, not composited
    np.testing.assert_array_equal(normalise_image(rgba), normalise_image(grey))
    np.testing.assert_array_equal(
        normalise_image(np.dstack([grey, np.full_like(grey, 255)])), normalise_image(grey)
    )
    np.testing.assert_array_equal(
        normalise_image(np.array([[True, False]])), np.array([[1.0, 0.0]])
    )
    np.testing.assert_array_equal(
        normalise_image(np.array([[0.0, 0.25, 1.0]], dtype=np.float32)), [[0.0, 0.25, 1.0]]
    )
    # 0/1 integer images are binary masks, not almost-black 8-bit images.
    np.testing.assert_array_equal(
        normalise_image(np.array([[0, 1]], dtype=np.uint8)), [[0.0, 1.0]]
    )
    colour = np.zeros((1, 1, 3), dtype=np.uint8)
    colour[0, 0] = (255, 0, 0)
    assert normalise_image(colour)[0, 0] == pytest.approx(0.2125)  # rgb2gray luminance


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.0, 2.0]]),
        np.array([[np.nan, 0.5]]),
        np.array([[-1, 5]], dtype=np.int16),
        np.zeros((2, 2, 5), dtype=np.uint8),
        np.zeros((2, 2, 2, 2), dtype=np.uint8),
    ],
)
def test_normalise_image_rejects_ambiguous_data(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalise_image(bad)


def test_mask_import_is_independent_of_storage_format(tmp_path: Path) -> None:
    mask = _squares_mask()
    rr, cc = disk((5, 25), 4, shape=mask.shape)
    mask[rr, cc] = True
    # A real greyscale image: grey background, bright fibres, threshold in between.
    grey8 = np.where(mask, 204, 51).astype(np.uint8)  # 0.8 and 0.2
    grey16 = grey8.astype(np.uint16) * 257
    paths = {
        "grey8": _save(tmp_path / "grey8.png", grey8),
        "grey16": _save(tmp_path / "grey16.png", grey16),
        "rgb": _save(tmp_path / "rgb.png", np.dstack([grey8] * 3)),
        "rgba": _save(
            tmp_path / "rgba.png", np.dstack([grey8] * 3 + [np.full_like(grey8, 255)])
        ),
        "grey8_tif": _save(tmp_path / "grey8.tif", grey8),
        "float_tif": _save(tmp_path / "float.tif", grey8 / 255.0),
        "pgm16": _write_pgm16(tmp_path / "grey16.pgm", grey16),
    }
    expected_px = float(mask.sum())
    for threshold in (0.42, 0.5, 0.6):
        results = {
            name: import_mask_geometry(_config(path, threshold=threshold))
            for name, path in paths.items()
        }
        reference = results["rgb"].geometry
        assert reference.fibre_count == 6
        assert reference.fibre_volume_fraction == pytest.approx(
            expected_px / mask.size, abs=6 * 0.5 / mask.size
        )
        for name, result in results.items():
            fibres = result.geometry.polygonal_fibres
            assert len(fibres) == len(reference.polygonal_fibres), name
            for ours, ref in zip(fibres, reference.polygonal_fibres, strict=True):
                assert np.array_equal(ours.points, ref.points), name


def test_binary_mask_formats_still_work(tmp_path: Path) -> None:
    mask = _squares_mask()
    expected = import_mask_geometry(_config(_save(tmp_path / "m.png", mask * np.uint8(255))))
    for name, data in {
        "bool.png": mask,
        "zero_one.png": mask.astype(np.uint8),  # 0/1 label-style mask
    }.items():
        result = import_mask_geometry(_config(_save(tmp_path / name, data)))
        assert [f.points.tolist() for f in result.geometry.polygonal_fibres] == [
            f.points.tolist() for f in expected.geometry.polygonal_fibres
        ], name


# --------------------------------------------------------------------------------------------
# Fibres touching the image border
# --------------------------------------------------------------------------------------------


def test_border_squares_end_on_the_boundary_with_exact_areas(tmp_path: Path) -> None:
    mask = _squares_mask()
    s = 0.5
    result = import_mask_geometry(
        _config(_save(tmp_path / "squares.png", mask * np.uint8(255)), pixel_size=s)
    )
    geometry = result.geometry
    assert isinstance(geometry.domain, Domain2D)
    width, height = geometry.domain.width, geometry.domain.height
    assert (width, height) == (20.0, 20.0)
    assert geometry.fibre_count == 5
    assert result.clipped_regions == 4

    by_centroid = {
        tuple(np.round(f.points[:-1].mean(axis=0) / s).astype(int)): f.points
        for f in geometry.polygonal_fibres
    }
    corner = next(p for c, p in by_centroid.items() if c[0] < 10 and c[1] > 30)
    right_edge = next(p for c, p in by_centroid.items() if c[0] > 30 and 15 < c[1] < 25)
    interior = next(p for c, p in by_centroid.items() if 12 < c[0] < 22)

    # (4 - J - K)/8 px^2 deficits: corner J=2, K=1; edge J=2; interior J=0.
    assert _area(corner) == pytest.approx((100 - 1 / 8) * s**2, abs=1e-12)
    assert _area(right_edge) == pytest.approx((100 - 2 / 8) * s**2, abs=1e-12)
    assert _area(interior) == pytest.approx((100 - 4 / 8) * s**2, abs=1e-12)
    total = sum(_area(f.points) for f in geometry.polygonal_fibres)
    assert total == pytest.approx((500 - 2 * 1 / 8 - 2 * 2 / 8 - 4 / 8) * s**2, abs=1e-12)
    # Vf within 5 fibres x 0.5 px^2 of the pixel fraction (here 0.3117 vs 0.3125).
    assert geometry.fibre_volume_fraction == pytest.approx(mask.mean(), abs=5 * 0.5 / mask.size)

    for polygon in by_centroid.values():
        _assert_valid_polygon(polygon, width, height)
    # The corner fibre contains the domain corner and runs along both boundaries.
    assert [0.0, height] in corner.tolist()
    left = _boundary_vertices(corner, width, height)
    assert sorted(map(tuple, left)) == [(0.0, 15.0), (0.0, 20.0), (5.0, 20.0)]
    # The right-edge fibre covers rows 15..24 -> y in [7.5, 12.5] on x = width.
    on_right = _boundary_vertices(right_edge, width, height)
    assert sorted(map(tuple, on_right)) == [(20.0, 7.5), (20.0, 12.5)]
    assert len(_boundary_vertices(interior, width, height)) == 0


def test_reviewer_case_volume_fraction_matches_pixels(tmp_path: Path) -> None:
    # Two 10x10 fibres on the bottom-right corner and the right edge of a 40x40 image:
    # true Vf 0.125 (the old importer reported 0.087 because border contours were closed
    # by chords and the bottom/right fibres stopped one pixel short of the boundary).
    mask = np.zeros((40, 40), dtype=bool)
    mask[30:40, 30:40] = True
    mask[10:20, 30:40] = True
    result = import_mask_geometry(_config(_save(tmp_path / "m.png", mask * np.uint8(255))))
    vf = result.geometry.fibre_volume_fraction
    assert vf == pytest.approx(0.125, abs=2 * 0.5 / 1600)
    assert vf == pytest.approx((200 - 1 / 8 - 2 / 8) / 1600, abs=1e-12)


@pytest.mark.parametrize(
    ("center", "radius", "junctions", "corners"),
    [
        ((20, 20), 8, 0, 0),  # interior
        ((15, 2), 7, 2, 0),  # cut by the left edge
        ((38, 22), 6, 2, 0),  # cut by the bottom edge
        ((1, 38), 6, 2, 1),  # covers the top-right corner
        ((39, 0), 9, 2, 1),  # covers the bottom-left corner
    ],
)
def test_disc_areas(
    tmp_path: Path, center: tuple[int, int], radius: int, junctions: int, corners: int
) -> None:
    mask = np.zeros((40, 40), dtype=bool)
    rr, cc = disk(center, radius, shape=mask.shape)
    mask[rr, cc] = True
    result = import_mask_geometry(
        _config(_save(tmp_path / "disc.png", mask * np.uint8(255)), pixel_size=1.0)
    )
    (fibre,) = result.geometry.polygonal_fibres
    expected = mask.sum() - (4 - junctions - corners) / 8
    assert _area(fibre.points) == pytest.approx(expected, abs=1e-9)
    _assert_valid_polygon(fibre.points, 40.0, 40.0)
    assert len(_boundary_vertices(fibre.points, 40.0, 40.0)) == junctions + corners


@pytest.mark.parametrize("seed", range(12))
def test_random_masks_give_valid_polygons_with_exact_area(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(12, 40)), int(rng.integers(12, 40)))
    if seed % 3 == 0:
        mask = rng.random(shape) < 0.45  # salt and pepper: saddles, single pixels, slivers
    else:
        smooth = gaussian(rng.random(shape), sigma=float(rng.uniform(1.0, 2.5)))
        mask = smooth > np.quantile(smooth, rng.uniform(0.3, 0.7))
    mask = ndimage.binary_fill_holes(mask)
    _, regions = ndimage.label(mask)
    height, width = float(shape[0]), float(shape[1])

    exact = mask_to_polygons(mask, 1.0, simplify_tolerance=0.0)
    assert len(exact) == regions
    assert sum(_area(p) for p in exact) == pytest.approx(_marching_squares_area_px(mask), abs=1e-9)
    for polygon in exact:
        _assert_valid_polygon(polygon, width, height)
    pinned = sorted(
        tuple(v) for polygon in exact for v in _boundary_vertices(polygon, width, height)
    )

    for tolerance in (0.5, 1.0, 2.0):
        simplified = mask_to_polygons(mask, 1.0, simplify_tolerance=tolerance)
        assert len(simplified) == regions
        for polygon in simplified:
            _assert_valid_polygon(polygon, width, height)
        # Simplification never moves or drops a vertex on the domain boundary.
        assert pinned == sorted(
            tuple(v) for polygon in simplified for v in _boundary_vertices(polygon, width, height)
        )


@pytest.mark.parametrize("seed", range(6))
def test_random_masks_with_fractional_domain_crops(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    shape = (int(rng.integers(16, 40)), int(rng.integers(16, 40)))
    smooth = gaussian(rng.random(shape), sigma=1.5)
    mask = ndimage.binary_fill_holes(smooth > np.median(smooth))
    width = float(rng.uniform(0.4, 0.95) * shape[1])
    height = float(rng.uniform(0.4, 0.95) * shape[0])
    for tolerance in (0.0, 1.0):
        polygons = mask_to_polygons(
            mask, 1.0, simplify_tolerance=tolerance, domain_width=width, domain_height=height
        )
        assert polygons
        for polygon in polygons:
            _assert_valid_polygon(polygon, width, height)
            # Fibres reaching the crop line end exactly on it.
            if polygon[:, 0].max() > width - 1e-6:
                assert polygon[:, 0].max() == width
            if polygon[:, 1].max() > height - 1e-6:
                assert polygon[:, 1].max() == height


def test_ring_validity_check_matches_brute_force() -> None:
    rng = np.random.default_rng(11)
    outcomes = {True: 0, False: 0}
    for _ in range(4000):
        count = int(rng.integers(3, 9))
        ring = rng.integers(0, 6, size=(count, 2)).astype(np.float64) / 2.0
        expected = _ring_defect(ring) is None
        assert raster._is_valid_ring(ring) == expected, ring.tolist()
        outcomes[expected] += 1
    assert min(outcomes.values()) > 100  # both simple and non-simple rings were exercised
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert raster._is_valid_ring(square)
    bow_tie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert not raster._is_valid_ring(bow_tie)
    spike = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert not raster._is_valid_ring(spike)
    touching = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
    assert not raster._is_valid_ring(touching)


def test_simplification_keeps_border_fibres_on_the_boundary(tmp_path: Path) -> None:
    mask = np.zeros((60, 60), dtype=bool)
    for center, radius in (((0, 0), 14), ((30, 59), 10), ((59, 25), 12), ((30, 25), 11)):
        rr, cc = disk(center, radius, shape=mask.shape)
        mask[rr, cc] = True
    path = _save(tmp_path / "discs.png", mask * np.uint8(255))
    exact = import_mask_geometry(_config(path, pixel_size=1.0)).geometry
    simplified = import_mask_geometry(
        _config(path, pixel_size=1.0, simplify_tolerance=1.0)
    ).geometry
    assert simplified.fibre_count == exact.fibre_count == 4
    for ours, ref in zip(simplified.polygonal_fibres, exact.polygonal_fibres, strict=True):
        _assert_valid_polygon(ours.points, 60.0, 60.0)
        assert len(ours.points) < len(ref.points)
        np.testing.assert_array_equal(
            np.sort(_boundary_vertices(ours.points, 60.0, 60.0), axis=0),
            np.sort(_boundary_vertices(ref.points, 60.0, 60.0), axis=0),
        )
    # Douglas-Peucker only moves outlines by up to 1 px; for these 10-14 px fibres that is a
    # few percent of Vf, while border handling itself is exact (see the tests above).
    assert simplified.fibre_volume_fraction == pytest.approx(
        exact.fibre_volume_fraction, rel=0.05
    )


def test_domain_smaller_than_image_crops_fibres(tmp_path: Path) -> None:
    mask = _squares_mask()
    path = _save(tmp_path / "squares.png", mask * np.uint8(255))
    # Crop to the left 30 columns and the bottom 20 rows (pixel size 1).
    result = import_mask_geometry(
        _config(path, pixel_size=1.0, domain_width=30.0, domain_height=20.0)
    )
    geometry = result.geometry
    cropped = mask[20:, :30]
    assert geometry.fibre_count == 2  # the bottom-edge square and the cut interior square
    for fibre in geometry.polygonal_fibres:
        _assert_valid_polygon(fibre.points, 30.0, 20.0)
    total = sum(_area(f.points) for f in geometry.polygonal_fibres)
    assert total == pytest.approx(_marching_squares_area_px(cropped), abs=1e-9)
    # A crop through the middle of a pixel cuts that pixel column in half.
    half = import_mask_geometry(_config(path, pixel_size=1.0, domain_width=17.5)).geometry
    for fibre in half.polygonal_fibres:
        _assert_valid_polygon(fibre.points, 17.5, 40.0)
    assert max(f.points[:, 0].max() for f in half.polygonal_fibres) == 17.5


def test_domain_larger_than_image_keeps_fibres_on_the_image(tmp_path: Path) -> None:
    mask = _squares_mask()
    path = _save(tmp_path / "squares.png", mask * np.uint8(255))
    geometry = import_mask_geometry(
        _config(path, pixel_size=1.0, domain_width=50.0, domain_height=45.0)
    ).geometry
    reference = import_mask_geometry(_config(path, pixel_size=1.0)).geometry
    assert geometry.fibre_area == pytest.approx(reference.fibre_area, abs=1e-12)
    for fibre in geometry.polygonal_fibres:
        _assert_valid_polygon(fibre.points, 40.0, 40.0)


def test_domain_equal_to_image_up_to_rounding_snaps_to_the_boundary(tmp_path: Path) -> None:
    mask = _squares_mask()
    path = _save(tmp_path / "squares.png", mask * np.uint8(255))
    pixel_size = 1.7058823529411764  # SEM example: 40 px * s != 68.23529411764706 exactly
    width = 68.2352941176471
    geometry = import_mask_geometry(
        _config(path, pixel_size=pixel_size, domain_width=width, domain_height=width)
    ).geometry
    xs = np.concatenate([f.points[:, 0] for f in geometry.polygonal_fibres])
    ys = np.concatenate([f.points[:, 1] for f in geometry.polygonal_fibres])
    assert xs.max() == width and ys.max() == width


# --------------------------------------------------------------------------------------------
# 3D import = 2D polygons extruded
# --------------------------------------------------------------------------------------------


def test_3d_import_extrudes_the_2d_polygons(tmp_path: Path) -> None:
    mask = _squares_mask()
    rr, cc = disk((5, 25), 4, shape=mask.shape)
    mask[rr, cc] = True
    path = _save(tmp_path / "m.png", np.where(mask, 204, 51).astype(np.uint8))
    for tolerance in (0.0, 1.0):
        config = _config(path, simplify_tolerance=tolerance, extrusion_depth=3.0)
        flat = import_mask_geometry(config)
        solid = import_mask_geometry_3d(config)
        assert solid.removed_artifacts == flat.removed_artifacts
        assert solid.clipped_regions == flat.clipped_regions
        fibres = solid.geometry.extruded_polygonal_fibres
        assert len(fibres) == len(flat.geometry.polygonal_fibres) == 6
        for extruded, polygon in zip(fibres, flat.geometry.polygonal_fibres, strict=True):
            assert np.array_equal(extruded.points, polygon.points)
            assert (extruded.z_min, extruded.z_max) == (0.0, 3.0)
            assert extruded.fibre_id == polygon.fibre_id
        assert solid.geometry.fibre_volume_fraction == pytest.approx(
            flat.geometry.fibre_volume_fraction, abs=1e-12
        )


# --------------------------------------------------------------------------------------------
# Meshing: fibres on the boundary leave no matrix slivers
# --------------------------------------------------------------------------------------------


def _image_config_file(
    tmp_path: Path, image_path: Path, dimension: int = 2, extra: tuple[str, ...] = ()
) -> Path:
    config_path = tmp_path / f"config_{dimension}d.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: image",
                f"dimension: {dimension}",
                "image:",
                f"  image_path: {image_path}",
                "  pixel_size: 0.05",
                "  threshold: 0.5",
                "  min_artifact_area_px: 4",
                "  simplify_tolerance: 0.5",
                *extra,
                "mesh:",
                "  element_size_min: 0.05",
                "  element_size_max: 0.2",
                "  verbosity: 0",
                "export:",
                "  formats: [msh]",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_border_fibres_mesh_without_slivers(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    import meshio

    from rve2d.workflow import build_rve

    image_path = _save(tmp_path / "squares.png", _squares_mask() * np.uint8(255))
    config_path = _image_config_file(tmp_path, image_path)
    build = build_rve(load_config(config_path), output_dir=str(tmp_path / "out"))
    assert build.quality_report.clipped_fibres == 0
    mesh = meshio.read(build.mesh_files[0])
    areas = {1: 0.0, 2: 0.0}
    for block, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        if block.type != "triangle":
            continue
        tri = mesh.points[block.data][:, :, :2]
        signed = 0.5 * np.abs(
            (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
        )
        for tag in (1, 2):
            areas[tag] += float(signed[tags == tag].sum())
    image_config = load_config(config_path).image
    assert image_config is not None
    fibre_area = import_mask_geometry(image_config).geometry.fibre_area
    assert areas[2] == pytest.approx(fibre_area, rel=1e-9)
    assert areas[1] + areas[2] == pytest.approx(4.0, rel=1e-9)


@pytest.mark.parametrize("dimension", [2, 3])
def test_periodic_mask_with_border_fibres_meshes_periodically(
    tmp_path: Path, dimension: int
) -> None:
    # Fibres cut by the image edges reappear on the opposite edge. Their boundary segments
    # must coincide exactly for gmsh to pair the boundaries (the old importer stopped the
    # right/bottom fibres one pixel short, so the periodic build failed).
    pytest.importorskip("gmsh")
    from rve2d.workflow import build_rve

    size = 48
    mask = np.zeros((size, size), dtype=bool)
    for (row, col), radius in (((20, 0), 7), ((0, 30), 6), ((0, 0), 8), ((25, 24), 9)):
        for shift_row in (-size, 0, size):
            for shift_col in (-size, 0, size):
                rr, cc = disk((row + shift_row, col + shift_col), radius, shape=mask.shape)
                mask[rr, cc] = True
    image_path = _save(tmp_path / "periodic.png", mask * np.uint8(255))
    extra = ("  periodic_compatible: true",)
    if dimension == 3:
        extra += ("  extrusion_depth: 0.5",)
    config_path = _image_config_file(tmp_path, image_path, dimension, extra)
    build = build_rve(load_config(config_path), output_dir=str(tmp_path / "out"))
    assert build.quality_report.valid
