"""Raster helpers shared by the image-import (2D and 3D) and SEM pipelines.

Conventions used by every pipeline that reads an image:

* **Intensities.** Images are converted to float64 grey levels in ``[0, 1]`` whatever their
  storage format (8/16-bit integers, booleans, floats, grey, RGB or RGBA), so the thresholds
  in the configs always refer to that 0-1 scale.
* **Coordinates.** Pixel ``(row, col)`` of an image with ``rows`` rows is the square
  ``[col*s, (col+1)*s] x [(rows-row-1)*s, (rows-row)*s]``: x points right, y points up and the
  origin is the bottom-left corner of the image. The pixel centre is
  ``((col + 0.5)*s, (rows - row - 0.5)*s)`` and the image spans exactly
  ``[0, cols*s] x [0, rows*s]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.io import imread
from skimage.measure import approximate_polygon, find_contours
from skimage.segmentation import clear_border

from rve2d.config import ImageImportConfig
from rve2d.geometry_cleanup.cleanup import close_polygon, filter_small_polygons
from rve2d.models import FloatArray

BoolArray = NDArray[np.bool_]
LabelArray = NDArray[np.int32]

# Same weights as skimage.color.rgb2gray.
_LUMINANCE_WEIGHTS = np.array([0.2125, 0.7154, 0.0721], dtype=np.float64)
# Pillow returns 16-bit PGM data, rescaled to 0..65535, in an int32 array.
_UINT16_MAX = 65535.0
# Distances below this (in pixels) are treated as zero when cleaning clipped polygons.
_SNAP_PX = 1e-9


@dataclass(frozen=True)
class MaskPolygons:
    """Fibre polygons extracted from a segmented image (see :func:`extract_mask_polygons`)."""

    polygons: list[FloatArray]
    mask_shape: tuple[int, int]
    connected_regions: int
    removed_artifacts: int
    clipped_regions: int
    domain_width: float
    domain_height: float


# --------------------------------------------------------------------------------------------
# Reading images
# --------------------------------------------------------------------------------------------


def load_grayscale_image(path: str | Path) -> FloatArray:
    """Read an image file as float64 grey levels in ``[0, 1]`` (see :func:`normalise_image`)."""
    image_path = Path(path)
    raw = np.asarray(imread(image_path))
    try:
        return normalise_image(raw)
    except ValueError as exc:
        raise ValueError(f"Cannot use '{image_path}' as a grey-level image: {exc}") from exc


def normalise_image(image: NDArray[Any]) -> FloatArray:
    """Return ``image`` as a 2D float64 array of grey levels in ``[0, 1]``.

    * Integer images are scaled by the maximum of their storage format (8-bit: 255,
      16-bit: 65535, ...). Integer images whose values are all 0 or 1 are binary masks and
      are kept as 0/1.
    * Boolean images map to 0/1; floating-point images must already lie in ``[0, 1]``.
    * Colour images are reduced to luminance (``skimage.color.rgb2gray`` weights) after
      dropping the alpha channel, which is ignored rather than composited. Colour images
      whose channels are all equal keep their exact grey values, so the same grey image gives
      identical intensities whether it is stored as grey, RGB or RGBA.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        channels = array.shape[2]
        if channels in (1, 2):  # grey, grey + alpha
            array = array[:, :, 0]
        elif channels in (3, 4):  # RGB, RGBA
            array = array[:, :, :3]
        else:
            raise ValueError(f"unsupported image with {channels} channels")
    elif array.ndim != 2:
        raise ValueError(f"expected a 2D grey or colour image, got shape {array.shape}")

    unit = _to_unit_interval(array)
    if unit.ndim == 2:
        return unit
    red, green, blue = unit[:, :, 0], unit[:, :, 1], unit[:, :, 2]
    if np.array_equal(red, green) and np.array_equal(red, blue):
        return np.ascontiguousarray(red)
    return np.asarray(unit @ _LUMINANCE_WEIGHTS, dtype=np.float64)


def _to_unit_interval(array: NDArray[Any]) -> FloatArray:
    if array.dtype == np.bool_:
        return array.astype(np.float64)
    if np.issubdtype(array.dtype, np.integer):
        if array.size == 0:
            return array.astype(np.float64)
        low = int(array.min())
        high = int(array.max())
        if low < 0:
            raise ValueError("integer image contains negative values")
        info = np.iinfo(array.dtype)
        if high <= 1:
            scale = 1.0
        elif np.issubdtype(array.dtype, np.unsignedinteger):
            scale = float(info.max)
        elif array.dtype.itemsize >= 4 and high <= _UINT16_MAX:
            scale = _UINT16_MAX
        else:
            scale = float(info.max)
        # Same arithmetic as skimage.util.img_as_float.
        return np.asarray(np.multiply(array, 1.0 / scale, dtype=np.float64), dtype=np.float64)
    if np.issubdtype(array.dtype, np.floating):
        values = array.astype(np.float64)
        if values.size and not (
            bool(np.isfinite(values).all())
            and float(values.min()) >= 0.0
            and float(values.max()) <= 1.0
        ):
            raise ValueError(
                "floating-point intensities must be finite and lie in [0, 1] "
                f"(found {float(np.nanmin(values)):g}..{float(np.nanmax(values)):g}); "
                "rescale the image or store it as an 8/16-bit image"
            )
        return values
    raise ValueError(f"unsupported image data type {array.dtype}")


# --------------------------------------------------------------------------------------------
# Binary masks
# --------------------------------------------------------------------------------------------


def threshold_mask(image: FloatArray, threshold: float, invert: bool = False) -> BoolArray:
    """Foreground = ``image > threshold`` (or its complement when ``invert``)."""
    mask = np.asarray(image > threshold, dtype=bool)
    return ~mask if invert else mask


def fill_holes(mask: BoolArray) -> BoolArray:
    return np.asarray(ndimage.binary_fill_holes(mask), dtype=bool)


def label_regions(mask: BoolArray) -> tuple[LabelArray, int]:
    """4-connected labelling of ``mask`` (``scipy.ndimage.label`` defaults)."""
    labels, count = ndimage.label(mask)
    return np.asarray(labels, dtype=np.int32), int(count)


def remove_small_regions(mask: BoolArray, minimum_area: int) -> tuple[BoolArray, int]:
    """Drop connected regions smaller than ``minimum_area`` pixels; return the mask and count."""
    labels, count = label_regions(mask)
    if count == 0:
        return mask, 0
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    removed_artifacts = int(np.count_nonzero(~keep[1:]))
    return np.asarray(keep[labels], dtype=bool), removed_artifacts


def count_clipped_regions(mask: BoolArray) -> int:
    """Number of connected foreground runs on the image border."""
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    _, count = label_regions(mask & border)
    return count


def clear_border_mask(mask: BoolArray) -> BoolArray:
    cleared = clear_border(mask)  # type: ignore[no-untyped-call]
    return np.asarray(cleared, dtype=bool)


# --------------------------------------------------------------------------------------------
# Contours and coordinates
# --------------------------------------------------------------------------------------------


def find_level_contours(image: FloatArray, level: float) -> list[FloatArray]:
    """``skimage.measure.find_contours`` with typed output: ``(n, 2)`` arrays of (row, col)."""
    contours = find_contours(image, level)  # type: ignore[no-untyped-call]
    return [np.asarray(contour, dtype=np.float64) for contour in contours]


def simplify_polyline(points: FloatArray, tolerance: float) -> FloatArray:
    """Douglas-Peucker simplification (``skimage.measure.approximate_polygon``).

    The first and last points are always kept; a chain whose first and last points coincide
    is simplified as a closed contour.
    """
    simplified = approximate_polygon(points, tolerance=tolerance)  # type: ignore[no-untyped-call]
    return np.asarray(simplified, dtype=np.float64)


def pixel_to_physical(points_rc: FloatArray, n_rows: int, pixel_size: float) -> FloatArray:
    """Map ``(row, col)`` pixel-centre coordinates to physical ``(x, y)`` coordinates.

    Pixel centres land on ``((col + 0.5)*s, (n_rows - row - 0.5)*s)``, so an image with
    ``n_rows x n_cols`` pixels spans ``[0, n_cols*s] x [0, n_rows*s]``.
    """
    points = np.asarray(points_rc, dtype=np.float64)
    xy = np.empty((points.shape[0], 2), dtype=np.float64)
    xy[:, 0] = (points[:, 1] + 0.5) * pixel_size
    xy[:, 1] = (n_rows - points[:, 0] - 0.5) * pixel_size
    return xy


# --------------------------------------------------------------------------------------------
# Image -> mask -> polygons
# --------------------------------------------------------------------------------------------


def extract_mask_polygons(config: ImageImportConfig) -> MaskPolygons:
    """Threshold and clean the configured image, then convert its fibres to polygons.

    This is the common part of the 2D and 3D image imports: the 3D import extrudes exactly
    these polygons.
    """
    image = load_grayscale_image(config.image_path)
    mask = fill_holes(threshold_mask(image, config.threshold, config.invert))
    removed_artifacts = 0
    if config.min_artifact_area_px > 0:
        mask, removed_artifacts = remove_small_regions(mask, config.min_artifact_area_px)
    clipped_regions = count_clipped_regions(mask)
    if config.clear_border:
        mask = clear_border_mask(mask)
    _, connected_regions = label_regions(mask)

    rows, cols = mask.shape
    domain_width = (
        config.domain_width if config.domain_width is not None else cols * config.pixel_size
    )
    domain_height = (
        config.domain_height if config.domain_height is not None else rows * config.pixel_size
    )
    if domain_width <= 0.0 or domain_height <= 0.0:
        raise ValueError("Image import domain_width and domain_height must be positive.")
    polygons = mask_to_polygons(
        mask,
        config.pixel_size,
        simplify_tolerance=config.simplify_tolerance,
        domain_width=domain_width,
        domain_height=domain_height,
    )
    polygons = filter_small_polygons(polygons, minimum_area=config.pixel_size**2)
    return MaskPolygons(
        polygons=polygons,
        mask_shape=(rows, cols),
        connected_regions=connected_regions,
        removed_artifacts=removed_artifacts,
        clipped_regions=clipped_regions,
        domain_width=domain_width,
        domain_height=domain_height,
    )


def mask_to_polygons(
    mask: BoolArray,
    pixel_size: float,
    simplify_tolerance: float = 0.0,
    domain_width: float | None = None,
    domain_height: float | None = None,
) -> list[FloatArray]:
    """Convert the foreground regions of ``mask`` into fibre polygons in physical units.

    Each polygon is closed (last point == first point), counter-clockwise and simple. Contours
    come from marching squares at level 0.5, i.e. they run along pixel edges and cut each
    convex pixel corner by a 1/8 px^2 triangle (concave corners gain the same amount).

    Fibres cut by the domain boundary end exactly on it: the border pixels are replicated one
    pixel outwards and the result is padded with background, so every contour closes outside
    the image, and the contours are then clipped to the domain rectangle. The part of a
    fibre on the boundary is therefore a straight segment along the boundary that meets the
    rest of the outline at a right angle, and a fibre covering an image corner contains that
    corner exactly. ``domain_width``/``domain_height`` (default: the image extent) crop the
    image when smaller than it; when larger, fibres stop at the image edge.

    ``simplify_tolerance`` (pixels) applies Douglas-Peucker simplification. Vertices on the
    domain boundary are never removed, and a simplified outline that would self-intersect
    or collapse is replaced by the unsimplified one.
    """
    rows, cols = mask.shape
    if mask.size == 0:
        return []
    x_axis = _axis_extent(cols, pixel_size, domain_width)
    y_axis = _axis_extent(rows, pixel_size, domain_height)
    if x_axis.pixels == 0 or y_axis.pixels == 0:
        return []

    window = np.asarray(mask[rows - y_axis.pixels :, : x_axis.pixels], dtype=bool)
    window_rows = y_axis.pixels
    box = _PixelBox(
        row_min=window_rows - y_axis.span_px - 0.5,
        row_max=window_rows - 0.5,
        col_min=-0.5,
        col_max=x_axis.span_px - 0.5,
    )
    padded = np.pad(window, 1, mode="edge")
    padded = np.pad(padded, 1, mode="constant", constant_values=False)

    polygons: list[FloatArray] = []
    for contour in find_level_contours(padded.astype(np.float64), 0.5):
        ring = _contour_to_ring(contour - 2.0, box, simplify_tolerance)
        if ring.shape[0] < 3:
            continue
        polygons.append(
            _ring_to_physical(ring, box, window_rows, pixel_size, x_axis.extent, y_axis.extent)
        )
    return polygons


@dataclass(frozen=True)
class _AxisExtent:
    pixels: int  # image pixels kept along this axis, counted from the domain origin
    span_px: float  # domain extent along this axis in pixels (where the clip line sits)
    extent: float  # physical extent of the fibre geometry along this axis


def _axis_extent(n_pixels: int, pixel_size: float, domain_size: float | None) -> _AxisExtent:
    image_size = n_pixels * pixel_size
    if domain_size is None or math.isclose(domain_size, image_size, rel_tol=1e-9):
        extent = image_size if domain_size is None else domain_size
        return _AxisExtent(pixels=n_pixels, span_px=float(n_pixels), extent=extent)
    if domain_size > image_size:
        # The image does not reach the far side of the domain: fibres stop at the image edge.
        return _AxisExtent(pixels=n_pixels, span_px=float(n_pixels), extent=image_size)
    span_px = max(domain_size / pixel_size, 0.0)
    if math.isclose(span_px, round(span_px), rel_tol=0.0, abs_tol=1e-9):
        span_px = float(round(span_px))
    return _AxisExtent(
        pixels=min(n_pixels, math.ceil(span_px)),
        span_px=span_px,
        extent=domain_size,
    )


@dataclass(frozen=True)
class _PixelBox:
    """Domain rectangle in (row, col) pixel-centre coordinates of the cropped image."""

    row_min: float  # top edge, y = domain height
    row_max: float  # bottom edge, y = 0
    col_min: float  # left edge, x = 0
    col_max: float  # right edge, x = domain width

    def contains_strictly(self, ring: FloatArray) -> bool:
        return bool(
            np.all(
                (ring[:, 0] > self.row_min)
                & (ring[:, 0] < self.row_max)
                & (ring[:, 1] > self.col_min)
                & (ring[:, 1] < self.col_max)
            )
        )

    def on_boundary(self, ring: FloatArray) -> BoolArray:
        return np.asarray(
            (ring[:, 0] == self.row_min)
            | (ring[:, 0] == self.row_max)
            | (ring[:, 1] == self.col_min)
            | (ring[:, 1] == self.col_max),
            dtype=bool,
        )


def _contour_to_ring(contour: FloatArray, box: _PixelBox, tolerance: float) -> FloatArray:
    """Clip and simplify one closed contour; returns an open ring of (row, col) vertices."""
    ring = contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour
    if box.contains_strictly(ring):
        pinned = np.zeros(ring.shape[0], dtype=bool)
    else:
        ring = _clip_ring(ring, box)
        pinned = box.on_boundary(ring)
    if ring.shape[0] < 3:
        return ring
    simplified = _simplify_ring(ring, pinned, tolerance)
    if simplified is ring or _is_valid_ring(simplified):
        return simplified
    return ring


def _clip_ring(ring: FloatArray, box: _PixelBox) -> FloatArray:
    ring = _clip_half_plane(ring, axis=1, limit=box.col_min, keep_greater=True)
    ring = _clip_half_plane(ring, axis=1, limit=box.col_max, keep_greater=False)
    ring = _clip_half_plane(ring, axis=0, limit=box.row_min, keep_greater=True)
    ring = _clip_half_plane(ring, axis=0, limit=box.row_max, keep_greater=False)
    return _drop_repeated_vertices(ring)


def _clip_half_plane(
    ring: FloatArray,
    axis: int,
    limit: float,
    keep_greater: bool,
) -> FloatArray:
    """Sutherland-Hodgman clip of a closed ring against ``ring[:, axis] >= limit`` (or ``<=``).

    Points created on the clip line get exactly ``limit`` as coordinate. The excursions that
    are clipped away are the replicated border pixels, i.e. bumps whose base on the clip line
    lies inside the fibre, so joining consecutive crossing points along the clip line is
    exact (no spurious bridges).
    """
    if ring.shape[0] == 0:
        return ring
    coordinate = ring[:, axis]
    inside = coordinate >= limit if keep_greater else coordinate <= limit
    if bool(inside.all()):
        return ring
    if not bool(inside.any()):
        return ring[:0]
    previous = np.roll(ring, 1, axis=0)
    crossing = inside != np.roll(inside, 1)
    delta = np.where(crossing, coordinate - previous[:, axis], 1.0)
    t = (limit - previous[:, axis]) / delta
    points = previous + t[:, np.newaxis] * (ring - previous)
    points[:, axis] = limit
    # Sutherland-Hodgman order: the crossing point of edge (k-1, k), then vertex k if inside.
    slots = np.empty((2 * ring.shape[0], 2), dtype=np.float64)
    slots[0::2] = points
    slots[1::2] = ring
    keep = np.empty(2 * ring.shape[0], dtype=bool)
    keep[0::2] = crossing
    keep[1::2] = inside
    return slots[keep]


def _drop_repeated_vertices(ring: FloatArray) -> FloatArray:
    if ring.shape[0] < 2:
        return ring
    difference = ring - np.roll(ring, 1, axis=0)
    distinct = np.hypot(difference[:, 0], difference[:, 1]) > _SNAP_PX
    if not bool(distinct.any()):
        return ring[:1]
    return ring[distinct]


def _simplify_ring(ring: FloatArray, pinned: BoolArray, tolerance: float) -> FloatArray:
    """Douglas-Peucker simplification of an open ring that keeps every pinned vertex."""
    count = ring.shape[0]
    if tolerance <= 0.0 or count <= 3:
        return ring
    pins = np.flatnonzero(pinned)
    if pins.size == 0:
        closed = np.vstack([ring, ring[:1]])
        return simplify_polyline(closed, tolerance)[:-1]
    start = int(pins[0])
    ring = np.roll(ring, -start, axis=0)
    bounds = [int(pin) - start for pin in pins] + [count]
    pieces: list[FloatArray] = []
    for first, last in zip(bounds[:-1], bounds[1:], strict=True):
        if last - first == 1:
            pieces.append(ring[first : first + 1])
            continue
        chain = ring[first : last + 1] if last < count else np.vstack([ring[first:], ring[:1]])
        pieces.append(simplify_polyline(chain, tolerance)[:-1])
    return np.vstack(pieces)


def _ring_to_physical(
    ring: FloatArray,
    box: _PixelBox,
    window_rows: int,
    pixel_size: float,
    width: float,
    height: float,
) -> FloatArray:
    xy = pixel_to_physical(ring, window_rows, pixel_size)
    # Vertices on the domain boundary get the exact boundary coordinate.
    xy[ring[:, 1] == box.col_min, 0] = 0.0
    xy[ring[:, 1] == box.col_max, 0] = width
    xy[ring[:, 0] == box.row_max, 1] = 0.0
    xy[ring[:, 0] == box.row_min, 1] = height
    xy[:, 0] = np.clip(xy[:, 0], 0.0, width)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, height)
    if signed_area(xy) < 0.0:
        xy = np.vstack([xy[:1], xy[:0:-1]])
    return close_polygon(xy)


# --------------------------------------------------------------------------------------------
# Polygon checks
# --------------------------------------------------------------------------------------------


def signed_area(ring: FloatArray) -> float:
    """Shoelace area of a ring (open or closed); positive when counter-clockwise."""
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_valid_ring(ring: FloatArray) -> bool:
    """True for an open ring with >= 3 vertices, no zero-length edge, non-zero area and no
    self-intersection (touching counts as intersecting)."""
    if ring.shape[0] < 3:
        return False
    edges = np.roll(ring, -1, axis=0) - ring
    if bool(np.any(np.all(edges == 0.0, axis=1))):
        return False
    if signed_area(ring) == 0.0:
        return False
    return not _has_self_intersection(ring)


def _has_self_intersection(ring: FloatArray) -> bool:
    count = ring.shape[0]
    start = ring
    end = np.roll(ring, -1, axis=0)
    direction = end - start
    previous_direction = np.roll(direction, 1, axis=0)
    # Adjacent edges only overlap when the outline doubles back on itself.
    cross = previous_direction[:, 0] * direction[:, 1] - previous_direction[:, 1] * direction[:, 0]
    dot = np.sum(previous_direction * direction, axis=1)
    if bool(np.any((cross == 0.0) & (dot < 0.0))):
        return True
    if count < 4:
        return False
    # Only edges whose x-ranges overlap can meet: sweep over the edges sorted by their left
    # end and pair each one with the following edges that start before it ends.
    x_low = np.minimum(start[:, 0], end[:, 0])
    x_high = np.maximum(start[:, 0], end[:, 0])
    order = np.argsort(x_low, kind="stable")
    position = np.arange(count)
    stop = np.searchsorted(x_low[order], x_high[order], side="right")
    span = np.maximum(stop - position - 1, 0)
    total = int(span.sum())
    if total == 0:
        return False
    first = np.repeat(position, span)
    second = first + 1 + np.arange(total) - np.repeat(np.cumsum(span) - span, span)
    edge_a = order[first]
    edge_b = order[second]
    gap = np.abs(edge_a - edge_b)
    non_adjacent = (gap != 1) & (gap != count - 1)
    edge_a = edge_a[non_adjacent]
    edge_b = edge_b[non_adjacent]
    hits = _segments_touch(start[edge_a], end[edge_a], start[edge_b], end[edge_b])
    return bool(hits.any())


def _segments_touch(
    p0: FloatArray,
    p1: FloatArray,
    q0: FloatArray,
    q1: FloatArray,
) -> BoolArray:
    """Element-wise test whether the closed segments p0-p1 and q0-q1 share a point."""
    o1 = _orientation(p0, p1, q0)
    o2 = _orientation(p0, p1, q1)
    o3 = _orientation(q0, q1, p0)
    o4 = _orientation(q0, q1, p1)
    proper = (np.sign(o1) * np.sign(o2) < 0) & (np.sign(o3) * np.sign(o4) < 0)
    touching = (
        ((o1 == 0) & _within_box(p0, p1, q0))
        | ((o2 == 0) & _within_box(p0, p1, q1))
        | ((o3 == 0) & _within_box(q0, q1, p0))
        | ((o4 == 0) & _within_box(q0, q1, p1))
    )
    return np.asarray(proper | touching, dtype=bool)


def _orientation(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    return np.asarray(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]),
        dtype=np.float64,
    )


def _within_box(a: FloatArray, b: FloatArray, c: FloatArray) -> BoolArray:
    return np.asarray(
        (np.minimum(a[..., 0], b[..., 0]) <= c[..., 0])
        & (c[..., 0] <= np.maximum(a[..., 0], b[..., 0]))
        & (np.minimum(a[..., 1], b[..., 1]) <= c[..., 1])
        & (c[..., 1] <= np.maximum(a[..., 1], b[..., 1])),
        dtype=bool,
    )
