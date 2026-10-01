from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread
from skimage.measure import CircleModel, EllipseModel, find_contours, regionprops
from skimage.segmentation import clear_border, watershed

from rve2d.config import SemToSyntheticConfig
from rve2d.geometry_cleanup.cleanup import close_polygon
from rve2d.models import (
    CircleFibre,
    Domain2D,
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
class _ShapeCandidate:
    kind: str
    score: float
    area_px: float
    pixel_mask: np.ndarray
    center_col_px: float
    center_row_px: float
    radius_px: float | None = None
    ellipse_axes_px: tuple[float, float] | None = None
    ellipse_angle_rad: float | None = None


def _estimate_circle_model(contour_xy: np.ndarray) -> CircleModel | None:
    model = CircleModel.from_estimate(contour_xy)
    return model if model is not None else None


def _estimate_ellipse_model(contour_xy: np.ndarray) -> EllipseModel | None:
    model = EllipseModel.from_estimate(contour_xy)
    return model if model is not None else None


def generate_circular_fibre_rve_from_sem(
    config: SemToSyntheticConfig,
) -> SemSyntheticExtractionResult:
    image_path = Path(config.image_path)
    image = imread(image_path, as_gray=True)
    smoothed = gaussian(image, sigma=config.smoothing_sigma, preserve_range=True)
    threshold = (
        config.threshold if config.threshold is not None else float(threshold_otsu(smoothed))
    )
    mask = smoothed > threshold
    if config.invert:
        mask = ~mask
    mask = ndimage.binary_fill_holes(mask)
    clipped_regions = _count_clipped_regions(mask)
    if config.clear_border:
        mask = clear_border(mask)  # type: ignore[no-untyped-call]

    labels, _ = ndimage.label(mask)
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

    rows, cols = mask.shape
    domain_width = (
        config.domain_width if config.domain_width is not None else cols * config.pixel_size
    )
    domain_height = (
        config.domain_height if config.domain_height is not None else rows * config.pixel_size
    )
    domain = Domain2D(width=domain_width, height=domain_height)

    candidates = _extract_candidates(labels, config)
    kept_candidates = _prune_candidates(candidates, config.max_candidate_overlap_fraction)
    kept_candidates = _resolve_candidate_overlaps(kept_candidates)

    circular_fibres: list[CircleFibre] = []
    polygonal_fibres: list[PolygonFibre] = []
    circle_count = 0
    ellipse_count = 0
    for index, candidate in enumerate(kept_candidates, start=1):
        if candidate.kind == "circle":
            circle_count += 1
            assert candidate.radius_px is not None
            circular_fibres.append(
                CircleFibre(
                    center_x=candidate.center_col_px * config.pixel_size,
                    center_y=(rows - candidate.center_row_px) * config.pixel_size,
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
            "mask_shape": list(mask.shape),
            "extracted_fibre_count": len(kept_candidates),
            "circle_count": circle_count,
            "ellipse_count": ellipse_count,
            "removed_artifacts": removed_artifacts,
            "threshold_used": threshold,
        },
    )
    return SemSyntheticExtractionResult(
        geometry=geometry,
        extracted_fibre_count=len(kept_candidates),
        removed_artifacts=removed_artifacts,
        clipped_regions=clipped_regions,
        threshold_used=float(threshold),
    )


def _extract_candidates(
    labels: np.ndarray,
    config: SemToSyntheticConfig,
) -> list[_ShapeCandidate]:
    candidates: list[_ShapeCandidate] = []
    for region in regionprops(labels):
        if int(region.area) < config.min_region_area_px:
            continue
        candidate = _fit_region_candidate(region, labels.shape, config)
        if candidate is None:
            continue
        if config.drop_boundary_fibres and _touches_boundary(candidate.pixel_mask):
            continue
        candidates.append(candidate)
    return candidates


def _fit_region_candidate(
    region: object,
    image_shape: tuple[int, int],
    config: SemToSyntheticConfig,
) -> _ShapeCandidate | None:
    min_row, min_col, _, _ = region.bbox
    padded = np.pad(region.image.astype(bool), 1, mode="constant", constant_values=False)
    contours = find_contours(padded.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda points: points.shape[0])
    contour_xy = np.column_stack((contour[:, 1] + min_col - 1, contour[:, 0] + min_row - 1))
    if contour_xy.shape[0] < 8:
        return None

    region_mask = np.zeros(image_shape, dtype=bool)
    max_row, max_col = region.bbox[2], region.bbox[3]
    region_mask[min_row:max_row, min_col:max_col] = region.image.astype(bool)

    circle_candidate = _fit_circle_candidate(contour_xy, region_mask)
    ellipse_candidate = _fit_ellipse_candidate(
        contour_xy,
        region_mask,
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


def _fit_circle_candidate(
    contour_xy: np.ndarray, region_mask: np.ndarray
) -> _ShapeCandidate | None:
    model = _estimate_circle_model(contour_xy)
    if model is None:
        return None
    center_col, center_row = (float(value) for value in model.center)
    radius = float(model.radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    pixel_mask = _shape_mask_from_circle(region_mask.shape, center_row, center_col, radius)
    return _ShapeCandidate(
        kind="circle",
        score=_mask_iou(pixel_mask, region_mask),
        area_px=float(np.count_nonzero(pixel_mask)),
        pixel_mask=pixel_mask,
        center_col_px=center_col,
        center_row_px=center_row,
        radius_px=radius,
    )


def _fit_ellipse_candidate(
    contour_xy: np.ndarray,
    region_mask: np.ndarray,
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
    pixel_mask = _shape_mask_from_ellipse(
        region_mask.shape,
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
        area_px=float(np.count_nonzero(pixel_mask)),
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
) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2


def _shape_mask_from_ellipse(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    axis_a: float,
    axis_b: float,
    angle: float,
    sample_points: int,
) -> np.ndarray:
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
) -> np.ndarray:
    polygon_px = _ellipse_polygon_pixels(
        center_col_px,
        center_row_px,
        axis_a_px,
        axis_b_px,
        angle,
        sample_points,
    )
    polygon = np.zeros_like(polygon_px)
    polygon[:, 0] = polygon_px[:, 0] * pixel_size
    polygon[:, 1] = (rows - polygon_px[:, 1]) * pixel_size
    return close_polygon(polygon.astype(np.float64))


def _ellipse_polygon_pixels(
    center_col_px: float,
    center_row_px: float,
    axis_a_px: float,
    axis_b_px: float,
    angle: float,
    sample_points: int,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, sample_points, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x = center_col_px + axis_a_px * cos_t * cos_a - axis_b_px * sin_t * sin_a
    y = center_row_px + axis_a_px * cos_t * sin_a + axis_b_px * sin_t * cos_a
    return np.column_stack((x, y))


def _polygon_mask(shape: tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(shape)
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    inside = np.zeros(shape, dtype=bool)
    j = len(x) - 1
    for i in range(len(x)):
        intersects = ((y[i] > rows) != (y[j] > rows)) & (
            cols < (x[j] - x[i]) * (rows - y[i]) / ((y[j] - y[i]) + 1e-12) + x[i]
        )
        inside ^= intersects
        j = i
    return inside


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _prune_candidates(
    candidates: list[_ShapeCandidate],
    max_overlap_fraction: float,
) -> list[_ShapeCandidate]:
    kept: list[_ShapeCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.score, item.area_px), reverse=True):
        overlaps_too_much = False
        for accepted in kept:
            intersection = np.count_nonzero(candidate.pixel_mask & accepted.pixel_mask)
            if intersection == 0:
                continue
            overlap_fraction = intersection / min(candidate.area_px, accepted.area_px)
            if overlap_fraction > max_overlap_fraction:
                overlaps_too_much = True
                break
        if not overlaps_too_much:
            kept.append(candidate)
    return kept


def _resolve_candidate_overlaps(
    candidates: list[_ShapeCandidate],
    max_iterations: int = 16,
    shrink_factor: float = 0.98,
) -> list[_ShapeCandidate]:
    resolved = list(candidates)
    for _ in range(max_iterations):
        changed = False
        for first_index in range(len(resolved)):
            for second_index in range(first_index + 1, len(resolved)):
                intersection = np.count_nonzero(
                    resolved[first_index].pixel_mask & resolved[second_index].pixel_mask
                )
                if intersection == 0:
                    continue
                shrink_index = (
                    first_index
                    if resolved[first_index].score < resolved[second_index].score
                    else second_index
                )
                shrunk = _shrink_candidate(resolved[shrink_index], shrink_factor)
                if shrunk is None:
                    continue
                resolved[shrink_index] = shrunk
                changed = True
        if not changed:
            break
    return resolved


def _shrink_candidate(candidate: _ShapeCandidate, shrink_factor: float) -> _ShapeCandidate | None:
    if candidate.kind == "circle":
        assert candidate.radius_px is not None
        new_radius = candidate.radius_px * shrink_factor
        if new_radius <= 1.0:
            return None
        pixel_mask = _shape_mask_from_circle(
            candidate.pixel_mask.shape,
            candidate.center_row_px,
            candidate.center_col_px,
            new_radius,
        )
        return _ShapeCandidate(
            kind="circle",
            score=candidate.score,
            area_px=float(np.count_nonzero(pixel_mask)),
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
        candidate.pixel_mask.shape,
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
        area_px=float(np.count_nonzero(pixel_mask)),
        pixel_mask=pixel_mask,
        center_col_px=candidate.center_col_px,
        center_row_px=candidate.center_row_px,
        ellipse_axes_px=new_axes,
        ellipse_angle_rad=candidate.ellipse_angle_rad,
    )


def _remove_small_regions(labels: np.ndarray, minimum_area: int) -> tuple[np.ndarray, int]:
    counts = np.bincount(labels.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    relabeled, _ = ndimage.label(filtered > 0)
    removed_artifacts = int(np.count_nonzero((~keep)[1:]))
    return relabeled, removed_artifacts


def _count_clipped_regions(mask: np.ndarray) -> int:
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    clipped_mask = mask & border
    _, count = ndimage.label(clipped_mask)
    return int(count)


def _separate_touching_regions(
    mask: np.ndarray,
    min_distance: int,
    peak_threshold: float,
) -> np.ndarray:
    distance = ndimage.distance_transform_edt(mask)
    peak_coords = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance,
        threshold_abs=peak_threshold,
    )
    if peak_coords.size == 0:
        relabeled, _ = ndimage.label(mask)
        return relabeled

    peak_mask = np.zeros_like(mask, dtype=bool)
    peak_mask[tuple(peak_coords.T)] = True
    markers, _ = ndimage.label(peak_mask)
    separated = watershed(-distance, markers, mask=mask)
    relabeled = np.zeros_like(separated, dtype=np.int32)
    for new_label, old_label in enumerate(np.unique(separated)[1:], start=1):
        relabeled[separated == old_label] = new_label
    return relabeled
