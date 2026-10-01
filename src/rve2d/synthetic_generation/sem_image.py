from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import CircleModel, EllipseModel, regionprops
from skimage.segmentation import watershed

from rve2d.config import SemToSyntheticConfig
from rve2d.geometry_cleanup.cleanup import close_polygon
from rve2d.image_import.raster import (
    BoolArray,
    LabelArray,
    clear_border_mask,
    count_clipped_regions,
    fill_holes,
    find_level_contours,
    label_regions,
    load_grayscale_image,
    pixel_to_physical,
)
from rve2d.models import (
    CircleFibre,
    Domain2D,
    FloatArray,
    GeometryModel,
    PolygonFibre,
    periodic_boundary_pairs,
)


@dataclass(frozen=True)
class SemSyntheticExtractionResult:
    geometry: GeometryModel
    extracted_fibre_count: int
    removed_artifacts: int
    clipped_regions: int
    threshold_used: float


@dataclass(frozen=True)
class _BoxMask:
    """Binary pixel mask stored by its bounding box.

    ``mask[i, j]`` is image pixel ``(row0 + i, col0 + j)``; every pixel outside the box is
    False. Keeping candidate masks this small (instead of one full-image mask per candidate)
    makes fitting, overlap tests and shrinking independent of the image size.
    """

    row0: int
    col0: int
    mask: BoolArray

    @property
    def row1(self) -> int:
        return self.row0 + int(self.mask.shape[0])

    @property
    def col1(self) -> int:
        return self.col0 + int(self.mask.shape[1])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.row0, self.row1, self.col0, self.col1)

    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

    def intersection_count(self, other: _BoxMask) -> int:
        row0 = max(self.row0, other.row0)
        row1 = min(self.row1, other.row1)
        col0 = max(self.col0, other.col0)
        col1 = min(self.col1, other.col1)
        if row0 >= row1 or col0 >= col1:
            return 0
        mine = self.mask[row0 - self.row0 : row1 - self.row0, col0 - self.col0 : col1 - self.col0]
        theirs = other.mask[
            row0 - other.row0 : row1 - other.row0, col0 - other.col0 : col1 - other.col0
        ]
        return int(np.count_nonzero(mine & theirs))

    def touches_image_border(self, image_shape: tuple[int, int]) -> bool:
        if self.mask.size == 0:
            return False
        rows, cols = image_shape
        return bool(
            (self.row0 == 0 and self.mask[0, :].any())
            or (self.row1 == rows and self.mask[-1, :].any())
            or (self.col0 == 0 and self.mask[:, 0].any())
            or (self.col1 == cols and self.mask[:, -1].any())
        )


@dataclass(frozen=True)
class _ShapeCandidate:
    kind: str
    score: float
    area_px: float
    pixel_mask: _BoxMask
    center_col_px: float
    center_row_px: float
    radius_px: float | None = None
    ellipse_axes_px: tuple[float, float] | None = None
    ellipse_angle_rad: float | None = None


@dataclass(frozen=True)
class _FibreDetection:
    candidates: list[_ShapeCandidate]
    image_shape: tuple[int, int]
    threshold: float
    removed_artifacts: int
    clipped_regions: int


class _Region(Protocol):
    """The ``skimage.measure.regionprops`` attributes used here."""

    @property
    def area(self) -> float: ...

    @property
    def bbox(self) -> tuple[int, int, int, int]: ...

    @property
    def image(self) -> BoolArray: ...


def generate_circular_fibre_rve_from_sem(
    config: SemToSyntheticConfig,
) -> SemSyntheticExtractionResult:
    image_path = Path(config.image_path)
    image = load_grayscale_image(image_path)
    detection = _detect_fibres(image, config)
    kept_candidates = detection.candidates

    rows, cols = detection.image_shape
    domain_width = (
        config.domain_width if config.domain_width is not None else cols * config.pixel_size
    )
    domain_height = (
        config.domain_height if config.domain_height is not None else rows * config.pixel_size
    )
    domain = Domain2D(width=domain_width, height=domain_height)

    circular_fibres: list[CircleFibre] = []
    polygonal_fibres: list[PolygonFibre] = []
    circle_count = 0
    ellipse_count = 0
    for index, candidate in enumerate(kept_candidates, start=1):
        if candidate.kind == "circle":
            circle_count += 1
            assert candidate.radius_px is not None
            center = pixel_to_physical(
                np.array([[candidate.center_row_px, candidate.center_col_px]]),
                rows,
                config.pixel_size,
            )[0]
            circular_fibres.append(
                CircleFibre(
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    radius=candidate.radius_px * config.pixel_size,
                    fibre_id=index,
                )
            )
            continue

        ellipse_count += 1
        assert candidate.ellipse_axes_px is not None
        assert candidate.ellipse_angle_rad is not None
        polygonal_fibres.append(
            PolygonFibre(
                points=_ellipse_polygon_points(
                    candidate.center_col_px,
                    candidate.center_row_px,
                    candidate.ellipse_axes_px[0],
                    candidate.ellipse_axes_px[1],
                    candidate.ellipse_angle_rad,
                    rows,
                    config.pixel_size,
                    config.ellipse_sample_points,
                ),
                fibre_id=index,
            )
        )

    geometry = GeometryModel(
        domain=domain,
        circular_fibres=circular_fibres,
        polygonal_fibres=polygonal_fibres,
        periodic_pairs=periodic_boundary_pairs(domain) if config.periodic_compatible else [],
        metadata={
            "generation_mode": "sem_to_synthetic",
            "image_path": str(image_path),
            "mask_shape": list(detection.image_shape),
            "extracted_fibre_count": len(kept_candidates),
            "circle_count": circle_count,
            "ellipse_count": ellipse_count,
            "removed_artifacts": detection.removed_artifacts,
            "threshold_used": detection.threshold,
        },
    )
    return SemSyntheticExtractionResult(
        geometry=geometry,
        extracted_fibre_count=len(kept_candidates),
        removed_artifacts=detection.removed_artifacts,
        clipped_regions=detection.clipped_regions,
        threshold_used=detection.threshold,
    )


def _detect_fibres(image: FloatArray, config: SemToSyntheticConfig) -> _FibreDetection:
    """Segment ``image`` (grey levels in [0, 1]) and fit one circle/ellipse per fibre.

    Candidate positions are in pixel-centre coordinates (column, row).
    """
    smoothed = _gaussian(image, config.smoothing_sigma)
    threshold = (
        float(config.threshold) if config.threshold is not None else _otsu_threshold(smoothed)
    )
    mask = smoothed > threshold
    if config.invert:
        mask = ~mask
    mask = fill_holes(mask)
    clipped_regions = count_clipped_regions(mask)
    if config.clear_border:
        mask = clear_border_mask(mask)

    labels, _ = label_regions(mask)
    if config.min_artifact_area_px > 0:
        labels, removed_artifacts = _remove_small_regions(labels, config.min_artifact_area_px)
    else:
        removed_artifacts = 0
    if config.separate_touching_fibres:
        labels = _separate_touching_regions(
            labels > 0,
            min_distance=config.separation_min_distance_px,
            peak_threshold=config.separation_peak_threshold_px,
        )

    image_shape = (int(labels.shape[0]), int(labels.shape[1]))
    candidates = _extract_candidates(labels, config)
    kept_candidates = _prune_candidates(candidates, config.max_candidate_overlap_fraction)
    kept_candidates = _resolve_candidate_overlaps(kept_candidates, image_shape)
    return _FibreDetection(
        candidates=kept_candidates,
        image_shape=image_shape,
        threshold=threshold,
        removed_artifacts=removed_artifacts,
        clipped_regions=clipped_regions,
    )


def _extract_candidates(
    labels: LabelArray,
    config: SemToSyntheticConfig,
) -> list[_ShapeCandidate]:
    image_shape = (int(labels.shape[0]), int(labels.shape[1]))
    candidates: list[_ShapeCandidate] = []
    for region in _region_properties(labels):
        if int(region.area) < config.min_region_area_px:
            continue
        candidate = _fit_region_candidate(region, image_shape, config)
        if candidate is None:
            continue
        if config.drop_boundary_fibres and candidate.pixel_mask.touches_image_border(
            image_shape
        ):
            continue
        candidates.append(candidate)
    return candidates


def _fit_region_candidate(
    region: _Region,
    image_shape: tuple[int, int],
    config: SemToSyntheticConfig,
) -> _ShapeCandidate | None:
    min_row, min_col, _, _ = (int(value) for value in region.bbox)
    region_image = np.asarray(region.image, dtype=bool)
    padded = np.pad(region_image, 1, mode="constant", constant_values=False)
    contours = find_level_contours(padded.astype(np.float64), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda points: points.shape[0])
    contour_xy = np.column_stack((contour[:, 1] + min_col - 1, contour[:, 0] + min_row - 1))
    if contour_xy.shape[0] < 8:
        return None

    region_mask = _BoxMask(min_row, min_col, region_image)
    circle_candidate = _fit_circle_candidate(contour_xy, region_mask, image_shape)
    ellipse_candidate = _fit_ellipse_candidate(
        contour_xy,
        region_mask,
        image_shape,
        config.ellipse_sample_points,
    )

    axis_ratio = None
    if ellipse_candidate is not None and ellipse_candidate.ellipse_axes_px is not None:
        major = max(ellipse_candidate.ellipse_axes_px)
        minor = min(ellipse_candidate.ellipse_axes_px)
        if minor > 0.0:
            axis_ratio = major / minor

    if circle_candidate is None:
        return ellipse_candidate
    if ellipse_candidate is None:
        return circle_candidate
    if axis_ratio is not None and axis_ratio >= config.ellipse_axis_ratio_threshold:
        return (
            ellipse_candidate
            if ellipse_candidate.score >= circle_candidate.score * 0.9
            else circle_candidate
        )
    return (
        circle_candidate
        if circle_candidate.score >= ellipse_candidate.score - 0.02
        else ellipse_candidate
    )


def _estimate_circle_model(contour_xy: FloatArray) -> CircleModel | None:
    """Fit a circle, or return None when the fit fails.

    ``CircleModel.from_estimate`` signals failure with a falsy ``FailedEstimation`` object in
    scikit-image 0.26 (not ``None``), so test for a real model instead of comparing to None.
    """
    model = CircleModel.from_estimate(contour_xy)  # type: ignore[no-untyped-call]
    return model if isinstance(model, CircleModel) else None


def _estimate_ellipse_model(contour_xy: FloatArray) -> EllipseModel | None:
    """Fit an ellipse, or return None when the fit fails (see ``_estimate_circle_model``)."""
    model = EllipseModel.from_estimate(contour_xy)  # type: ignore[no-untyped-call]
    return model if isinstance(model, EllipseModel) else None


def _fit_circle_candidate(
    contour_xy: FloatArray,
    region_mask: _BoxMask,
    image_shape: tuple[int, int],
) -> _ShapeCandidate | None:
    model = _estimate_circle_model(contour_xy)
    if model is None:
        return None
    center_col, center_row = (float(value) for value in model.center)
    radius = float(model.radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    if not (np.isfinite(center_col) and np.isfinite(center_row)):
        return None
    pixel_mask = _shape_mask_from_circle(image_shape, center_row, center_col, radius)
    return _ShapeCandidate(
        kind="circle",
        score=_mask_iou(pixel_mask, region_mask),
        area_px=float(pixel_mask.count()),
        pixel_mask=pixel_mask,
        center_col_px=center_col,
        center_row_px=center_row,
        radius_px=radius,
    )


def _fit_ellipse_candidate(
    contour_xy: FloatArray,
    region_mask: _BoxMask,
    image_shape: tuple[int, int],
    sample_points: int,
) -> _ShapeCandidate | None:
    model = _estimate_ellipse_model(contour_xy)
    if model is None:
        return None
    center_col, center_row = (float(value) for value in model.center)
    axis_a, axis_b = (float(value) for value in model.axis_lengths)
    angle = float(model.theta)
    if not np.isfinite(axis_a) or not np.isfinite(axis_b) or axis_a <= 0.0 or axis_b <= 0.0:
        return None
    if not (np.isfinite(center_col) and np.isfinite(center_row) and np.isfinite(angle)):
        return None
    pixel_mask = _shape_mask_from_ellipse(
        image_shape,
        center_row,
        center_col,
        axis_a,
        axis_b,
        angle,
        sample_points,
    )
    return _ShapeCandidate(
        kind="ellipse",
        score=_mask_iou(pixel_mask, region_mask),
        area_px=float(pixel_mask.count()),
        pixel_mask=pixel_mask,
        center_col_px=center_col,
        center_row_px=center_row,
        ellipse_axes_px=(axis_a, axis_b),
        ellipse_angle_rad=angle,
    )


def _shape_mask_from_circle(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    radius: float,
) -> _BoxMask:
    """Pixels whose centre satisfies ``(row - r0)^2 + (col - c0)^2 <= radius^2``.

    The test is only evaluated on the circle's bounding box (plus a one-pixel margin), with the
    same expression as a full-image evaluation, so the result is pixel-for-pixel identical.
    """
    row0 = max(0, math.floor(center_row - radius) - 1)
    row1 = min(shape[0], math.ceil(center_row + radius) + 2)
    col0 = max(0, math.floor(center_col - radius) - 1)
    col1 = min(shape[1], math.ceil(center_col + radius) + 2)
    if row0 >= row1 or col0 >= col1:
        return _empty_box_mask()
    rows = np.arange(row0, row1)[:, np.newaxis]
    cols = np.arange(col0, col1)[np.newaxis, :]
    inside = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2
    return _trimmed_box_mask(row0, col0, inside)


def _shape_mask_from_ellipse(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    axis_a: float,
    axis_b: float,
    angle: float,
    sample_points: int,
) -> _BoxMask:
    polygon = _ellipse_polygon_pixels(center_col, center_row, axis_a, axis_b, angle, sample_points)
    return _polygon_mask(shape, polygon)


def _ellipse_polygon_points(
    center_col_px: float,
    center_row_px: float,
    axis_a_px: float,
    axis_b_px: float,
    angle: float,
    rows: int,
    pixel_size: float,
    sample_points: int,
) -> FloatArray:
    polygon_px = _ellipse_polygon_pixels(
        center_col_px,
        center_row_px,
        axis_a_px,
        axis_b_px,
        angle,
        sample_points,
    )
    polygon = pixel_to_physical(polygon_px[:, ::-1], rows, pixel_size)
    return close_polygon(polygon)


def _ellipse_polygon_pixels(
    center_col_px: float,
    center_row_px: float,
    axis_a_px: float,
    axis_b_px: float,
    angle: float,
    sample_points: int,
) -> FloatArray:
    """Ellipse outline sampled at ``sample_points`` points, as (col, row) pixel coordinates."""
    t = np.linspace(0.0, 2.0 * np.pi, sample_points, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = center_col_px + axis_a_px * cos_t * cos_a - axis_b_px * sin_t * sin_a
    y = center_row_px + axis_a_px * cos_t * sin_a + axis_b_px * sin_t * cos_a
    return np.column_stack((x, y))


def _polygon_mask(shape: tuple[int, int], polygon_xy: FloatArray) -> _BoxMask:
    """Even-odd rasterisation of a closed polygon given as (col, row) pixel coordinates.

    A pixel is inside when a ray from its centre towards +col crosses an odd number of edges.
    Only edges with ``min(y) <= row < max(y)`` cross row ``row``, and within a row a pixel
    can only be inside between the smallest and largest crossing abscissa, so the test is
    evaluated on that bounding box only (same expressions as a full-image evaluation, hence
    the same pixels).
    """
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return _empty_box_mask()
    row0 = max(0, math.ceil(float(y.min())))
    row1 = min(shape[0], math.ceil(float(y.max())))
    if row0 >= row1:
        return _empty_box_mask()
    rows = np.arange(row0, row1)[:, np.newaxis]
    # Edge k joins vertex k-1 to vertex k.
    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    crosses = (y > rows) != (y_prev > rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_col = (x_prev - x) * (rows - y) / ((y_prev - y) + 1e-12) + x
    crossing_values = crossing_col[crosses]
    if crossing_values.size == 0:
        return _empty_box_mask()
    if np.all(np.isfinite(crossing_values)):
        col0 = max(0, math.ceil(float(crossing_values.min())))
        col1 = min(shape[1], math.ceil(float(crossing_values.max())))
    else:
        col0, col1 = 0, shape[1]
    if col0 >= col1:
        return _empty_box_mask()
    cols = np.arange(col0, col1)[np.newaxis, :]
    inside = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for edge in range(x.shape[0]):
        inside ^= crosses[:, edge : edge + 1] & (cols < crossing_col[:, edge : edge + 1])
    return _trimmed_box_mask(row0, col0, inside)


def _trimmed_box_mask(row0: int, col0: int, mask: BoolArray) -> _BoxMask:
    rows_used = np.flatnonzero(mask.any(axis=1))
    if rows_used.size == 0:
        return _empty_box_mask()
    cols_used = np.flatnonzero(mask.any(axis=0))
    first_row, last_row = int(rows_used[0]), int(rows_used[-1]) + 1
    first_col, last_col = int(cols_used[0]), int(cols_used[-1]) + 1
    return _BoxMask(
        row0 + first_row,
        col0 + first_col,
        mask[first_row:last_row, first_col:last_col],
    )


def _empty_box_mask() -> _BoxMask:
    return _BoxMask(0, 0, np.zeros((0, 0), dtype=bool))


def _mask_iou(mask_a: _BoxMask, mask_b: _BoxMask) -> float:
    intersection = mask_a.intersection_count(mask_b)
    union = mask_a.count() + mask_b.count() - intersection
    return float(intersection / union) if union else 0.0


def _boxes_overlap(box: tuple[int, int, int, int], boxes: NDArray[np.int64]) -> BoolArray:
    """Which rows of ``boxes`` ([row0, row1, col0, col1], half-open) overlap ``box``."""
    row0, row1, col0, col1 = box
    return np.asarray(
        (boxes[:, 0] < row1)
        & (row0 < boxes[:, 1])
        & (boxes[:, 2] < col1)
        & (col0 < boxes[:, 3]),
        dtype=bool,
    )


def _box_array(candidates: list[_ShapeCandidate]) -> NDArray[np.int64]:
    boxes = np.zeros((len(candidates), 4), dtype=np.int64)
    for index, candidate in enumerate(candidates):
        boxes[index] = candidate.pixel_mask.bounds
    return boxes


def _prune_candidates(
    candidates: list[_ShapeCandidate],
    max_overlap_fraction: float,
) -> list[_ShapeCandidate]:
    """Greedy selection by (score, area): drop candidates overlapping an accepted one too much.

    Only accepted candidates whose bounding boxes intersect the candidate's can overlap it.
    """
    ordered = sorted(candidates, key=lambda item: (item.score, item.area_px), reverse=True)
    kept: list[_ShapeCandidate] = []
    kept_boxes = np.zeros((len(ordered), 4), dtype=np.int64)
    for candidate in ordered:
        overlaps_too_much = False
        neighbours = np.flatnonzero(
            _boxes_overlap(candidate.pixel_mask.bounds, kept_boxes[: len(kept)])
        )
        for kept_index in neighbours:
            accepted = kept[int(kept_index)]
            intersection = candidate.pixel_mask.intersection_count(accepted.pixel_mask)
            if intersection == 0:
                continue
            overlap_fraction = intersection / min(candidate.area_px, accepted.area_px)
            if overlap_fraction > max_overlap_fraction:
                overlaps_too_much = True
                break
        if not overlaps_too_much:
            kept_boxes[len(kept)] = candidate.pixel_mask.bounds
            kept.append(candidate)
    return kept


def _resolve_candidate_overlaps(
    candidates: list[_ShapeCandidate],
    image_shape: tuple[int, int],
    max_iterations: int = 16,
    shrink_factor: float = 0.98,
) -> list[_ShapeCandidate]:
    """Shrink the lower-scoring candidate of every overlapping pair until no pixel is shared.

    Pairs are visited in the same order as an all-pairs sweep, but pairs whose bounding boxes
    are disjoint (and therefore cannot share a pixel) are skipped.
    """
    resolved = list(candidates)
    boxes = _box_array(resolved)
    count = len(resolved)
    for _ in range(max_iterations):
        changed = False
        for first_index in range(count):
            second_index = first_index + 1
            while second_index < count:
                following = np.flatnonzero(
                    _boxes_overlap(resolved[first_index].pixel_mask.bounds, boxes[second_index:])
                )
                if following.size == 0:
                    break
                second_index += int(following[0])
                first = resolved[first_index]
                second = resolved[second_index]
                if first.pixel_mask.intersection_count(second.pixel_mask) > 0:
                    shrink_index = first_index if first.score < second.score else second_index
                    shrunk = _shrink_candidate(resolved[shrink_index], shrink_factor, image_shape)
                    if shrunk is not None:
                        resolved[shrink_index] = shrunk
                        boxes[shrink_index] = shrunk.pixel_mask.bounds
                        changed = True
                second_index += 1
        if not changed:
            break
    return resolved


def _shrink_candidate(
    candidate: _ShapeCandidate,
    shrink_factor: float,
    image_shape: tuple[int, int],
) -> _ShapeCandidate | None:
    if candidate.kind == "circle":
        assert candidate.radius_px is not None
        new_radius = candidate.radius_px * shrink_factor
        if new_radius <= 1.0:
            return None
        pixel_mask = _shape_mask_from_circle(
            image_shape,
            candidate.center_row_px,
            candidate.center_col_px,
            new_radius,
        )
        return _ShapeCandidate(
            kind="circle",
            score=candidate.score,
            area_px=float(pixel_mask.count()),
            pixel_mask=pixel_mask,
            center_col_px=candidate.center_col_px,
            center_row_px=candidate.center_row_px,
            radius_px=new_radius,
        )

    assert candidate.ellipse_axes_px is not None
    assert candidate.ellipse_angle_rad is not None
    new_axes = (
        candidate.ellipse_axes_px[0] * shrink_factor,
        candidate.ellipse_axes_px[1] * shrink_factor,
    )
    if min(new_axes) <= 1.0:
        return None
    pixel_mask = _shape_mask_from_ellipse(
        image_shape,
        candidate.center_row_px,
        candidate.center_col_px,
        new_axes[0],
        new_axes[1],
        candidate.ellipse_angle_rad,
        48,
    )
    return _ShapeCandidate(
        kind="ellipse",
        score=candidate.score,
        area_px=float(pixel_mask.count()),
        pixel_mask=pixel_mask,
        center_col_px=candidate.center_col_px,
        center_row_px=candidate.center_row_px,
        ellipse_axes_px=new_axes,
        ellipse_angle_rad=candidate.ellipse_angle_rad,
    )


def _remove_small_regions(labels: LabelArray, minimum_area: int) -> tuple[LabelArray, int]:
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    relabeled, _ = label_regions(filtered > 0)
    removed_artifacts = int(np.count_nonzero(~keep[1:]))
    return relabeled, removed_artifacts


def _separate_touching_regions(
    mask: BoolArray,
    min_distance: int,
    peak_threshold: float,
) -> LabelArray:
    distance = np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)
    peak_coords = _peak_local_max(distance, mask, min_distance, peak_threshold)
    if peak_coords.size == 0:
        relabeled, _ = label_regions(mask)
        return relabeled

    peak_mask = np.zeros_like(mask, dtype=bool)
    peak_mask[tuple(peak_coords.T)] = True
    markers, _ = label_regions(peak_mask)
    separated = _watershed(-distance, markers, mask)
    # Renumber the watershed labels 1..n in increasing order (the smallest value, i.e. the
    # background 0, maps to 0).
    _, consecutive = np.unique(separated, return_inverse=True)
    return np.asarray(consecutive.reshape(separated.shape), dtype=np.int32)


def _region_properties(labels: LabelArray) -> list[_Region]:
    regions = regionprops(labels)  # type: ignore[no-untyped-call]
    return cast(list[_Region], regions)


def _gaussian(image: FloatArray, sigma: float) -> FloatArray:
    smoothed = gaussian(image, sigma=sigma, preserve_range=True)  # type: ignore[no-untyped-call]
    return np.asarray(smoothed, dtype=np.float64)


def _otsu_threshold(image: FloatArray) -> float:
    return float(threshold_otsu(image))  # type: ignore[no-untyped-call]


def _peak_local_max(
    distance: FloatArray,
    mask: BoolArray,
    min_distance: int,
    threshold_abs: float,
) -> NDArray[np.intp]:
    peaks = peak_local_max(  # type: ignore[no-untyped-call]
        distance,
        labels=mask,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )
    return np.asarray(peaks, dtype=np.intp)


def _watershed(image: FloatArray, markers: LabelArray, mask: BoolArray) -> LabelArray:
    labels = watershed(image, markers, mask=mask)  # type: ignore[no-untyped-call]
    return np.asarray(labels, dtype=np.int32)
