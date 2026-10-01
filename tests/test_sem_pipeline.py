"""Regression tests for the SEM -> synthetic pipeline (input formats, model fits, speed-ups)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import ndimage
from skimage.draw import disk, ellipse
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.measure import CircleModel, EllipseModel, find_contours, regionprops
from skimage.segmentation import watershed

from rve2d.config import SemToSyntheticConfig
from rve2d.image_import.raster import load_grayscale_image
from rve2d.synthetic_generation import sem_image
from rve2d.synthetic_generation.sem_image import generate_circular_fibre_rve_from_sem

BACKGROUND = 0.2
FIBRE = 0.8


def _sem_like_image() -> np.ndarray:
    """160 x 160 grey image with isolated, touching and overlapping fibres and one ellipse."""
    fibres = np.zeros((160, 160), dtype=bool)
    discs = [
        ((30, 30), 13),
        ((30, 52), 12),  # overlaps the previous disc: one blob, split by the watershed
        ((35, 120), 14),
        ((80, 70), 15),
        ((88, 96), 13),  # touches/overlaps the previous disc
        ((70, 4), 12),  # cut by the left border
        ((130, 40), 14),
        ((140, 140), 12),
    ]
    for (row, col), radius in discs:
        rr, cc = disk((row, col), radius, shape=fibres.shape)
        fibres[rr, cc] = True
    rr, cc = ellipse(122, 98, 10, 19, shape=fibres.shape, rotation=np.deg2rad(30.0))
    fibres[rr, cc] = True
    image = np.where(fibres, FIBRE, BACKGROUND)
    return np.asarray(gaussian(image, sigma=1.0, preserve_range=True), dtype=np.float64)


def _config(image_path: Path, **overrides: Any) -> SemToSyntheticConfig:
    base = SemToSyntheticConfig(
        image_path=str(image_path),
        pixel_size=0.5,
        smoothing_sigma=1.0,
        threshold=0.42,
        separate_touching_fibres=True,
        separation_min_distance_px=8,
        separation_peak_threshold_px=4.0,
        min_artifact_area_px=20,
        min_region_area_px=150,
        ellipse_axis_ratio_threshold=1.15,
        ellipse_sample_points=48,
        max_candidate_overlap_fraction=0.2,
        drop_boundary_fibres=True,
    )
    return replace(base, **overrides)


def _save(path: Path, data: np.ndarray) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(path, data, check_contrast=False)
    return path


# --------------------------------------------------------------------------------------------
# Straightforward reference: the original algorithm with one full-image mask per candidate and
# all-pairs overlap checks. The optimised pipeline must reproduce it exactly.
# --------------------------------------------------------------------------------------------


@dataclass
class _Ref:
    kind: str
    score: float
    area_px: float
    mask: np.ndarray
    center_col_px: float
    center_row_px: float
    radius_px: float | None = None
    ellipse_axes_px: tuple[float, float] | None = None
    ellipse_angle_rad: float | None = None


def _ref_polygon_mask(shape: tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(shape)
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    inside = np.zeros(shape, dtype=bool)
    j = len(x) - 1
    for i in range(len(x)):
        inside ^= ((y[i] > rows) != (y[j] > rows)) & (
            cols < (x[j] - x[i]) * (rows - y[i]) / ((y[j] - y[i]) + 1e-12) + x[i]
        )
        j = i
    return inside


def _ref_ellipse_mask(
    shape: tuple[int, int], cr: float, cc: float, a: float, b: float, angle: float, n: int
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = cc + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
    y = cr + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
    return _ref_polygon_mask(shape, np.column_stack((x, y)))


def _ref_circle_mask(shape: tuple[int, int], cr: float, cc: float, r: float) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return np.asarray((rows - cr) ** 2 + (cols - cc) ** 2 <= r**2)


def _ref_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _reference_detection(
    image: np.ndarray, config: SemToSyntheticConfig, stats: dict[str, int]
) -> list[_Ref]:
    smoothed = gaussian(image, sigma=config.smoothing_sigma, preserve_range=True)
    assert config.threshold is not None
    mask = ndimage.binary_fill_holes(smoothed > config.threshold)
    labels, _ = ndimage.label(mask)
    counts = np.bincount(labels.ravel())
    keep = counts >= config.min_artifact_area_px
    keep[0] = False
    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    labels, _ = ndimage.label(filtered > 0)
    foreground = labels > 0
    distance = ndimage.distance_transform_edt(foreground)
    peaks = peak_local_max(
        distance,
        labels=foreground,
        min_distance=config.separation_min_distance_px,
        threshold_abs=config.separation_peak_threshold_px,
    )
    peak_mask = np.zeros_like(foreground)
    peak_mask[tuple(peaks.T)] = True
    markers, _ = ndimage.label(peak_mask)
    separated = watershed(-distance, markers, mask=foreground)
    labels = np.zeros_like(separated, dtype=np.int32)
    for new_label, old_label in enumerate(np.unique(separated)[1:], start=1):
        labels[separated == old_label] = new_label

    shape = labels.shape
    candidates: list[_Ref] = []
    for region in regionprops(labels):
        if int(region.area) < config.min_region_area_px:
            continue
        min_row, min_col, _, _ = region.bbox
        padded = np.pad(region.image, 1)
        contour = max(find_contours(padded.astype(float), 0.5), key=len)
        xy = np.column_stack((contour[:, 1] + min_col - 1, contour[:, 0] + min_row - 1))
        if xy.shape[0] < 8:
            continue
        region_mask = labels == region.label
        circle = None
        model = CircleModel.from_estimate(xy)
        if isinstance(model, CircleModel) and np.isfinite(model.radius) and model.radius > 0:
            cc, cr = (float(v) for v in model.center)
            r = float(model.radius)
            m = _ref_circle_mask(shape, cr, cc, r)
            circle = _Ref("circle", _ref_iou(m, region_mask), float(m.sum()), m, cc, cr, r)
        ell = None
        model_e = EllipseModel.from_estimate(xy)
        if isinstance(model_e, EllipseModel):
            cc, cr = (float(v) for v in model_e.center)
            a, b = (float(v) for v in model_e.axis_lengths)
            angle = float(model_e.theta)
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                n = config.ellipse_sample_points
                m = _ref_ellipse_mask(shape, cr, cc, a, b, angle, n)
                score = _ref_iou(m, region_mask)
                ell = _Ref("ellipse", score, float(m.sum()), m, cc, cr, None, (a, b), angle)
        if circle is None or ell is None:
            chosen = circle if ell is None else ell
        else:
            assert ell.ellipse_axes_px is not None
            ratio = max(ell.ellipse_axes_px) / min(ell.ellipse_axes_px)
            if ratio >= config.ellipse_axis_ratio_threshold:
                chosen = ell if ell.score >= circle.score * 0.9 else circle
            else:
                chosen = circle if circle.score >= ell.score - 0.02 else ell
        if chosen is None:
            continue
        on_border = (
            chosen.mask[0].any()
            or chosen.mask[-1].any()
            or chosen.mask[:, 0].any()
            or chosen.mask[:, -1].any()
        )
        if config.drop_boundary_fibres and on_border:
            continue
        candidates.append(chosen)

    kept: list[_Ref] = []
    for cand in sorted(candidates, key=lambda c: (c.score, c.area_px), reverse=True):
        rejected = any(
            np.count_nonzero(cand.mask & other.mask) / min(cand.area_px, other.area_px)
            > config.max_candidate_overlap_fraction
            for other in kept
            if np.count_nonzero(cand.mask & other.mask)
        )
        stats["rejected"] += int(rejected)
        if not rejected:
            kept.append(cand)

    for _ in range(16):
        changed = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if not np.count_nonzero(kept[i].mask & kept[j].mask):
                    continue
                k = i if kept[i].score < kept[j].score else j
                c = kept[k]
                if c.kind == "circle":
                    assert c.radius_px is not None
                    r = c.radius_px * 0.98
                    if r <= 1.0:
                        continue
                    m = _ref_circle_mask(shape, c.center_row_px, c.center_col_px, r)
                    kept[k] = replace(c, area_px=float(m.sum()), mask=m, radius_px=r)
                else:
                    assert c.ellipse_axes_px is not None and c.ellipse_angle_rad is not None
                    axes = (c.ellipse_axes_px[0] * 0.98, c.ellipse_axes_px[1] * 0.98)
                    if min(axes) <= 1.0:
                        continue
                    m = _ref_ellipse_mask(
                        shape, c.center_row_px, c.center_col_px, *axes, c.ellipse_angle_rad, 48
                    )
                    kept[k] = replace(c, area_px=float(m.sum()), mask=m, ellipse_axes_px=axes)
                stats["shrinks"] += 1
                changed = True
        if not changed:
            break
    return kept


def _expand(box: Any, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[box.row0 : box.row1, box.col0 : box.col1] = box.mask
    return out


def _full_mask(candidate: Any, shape: tuple[int, int]) -> np.ndarray:
    return _expand(candidate.pixel_mask, shape)


def test_bounding_box_masks_match_full_image_rasterisation() -> None:
    rng = np.random.default_rng(7)
    shape = (60, 50)
    boxes = []
    for _ in range(240):
        center_row, center_col = rng.uniform(-12.0, 72.0), rng.uniform(-12.0, 62.0)
        if rng.random() < 0.5:
            radius = rng.uniform(0.3, 16.0)
            box = sem_image._shape_mask_from_circle(shape, center_row, center_col, radius)
            reference = _ref_circle_mask(shape, center_row, center_col, radius)
        else:
            a, b = rng.uniform(0.5, 18.0, size=2)
            angle = rng.uniform(-np.pi, np.pi)
            n = int(rng.integers(8, 64))
            box = sem_image._shape_mask_from_ellipse(
                shape, center_row, center_col, a, b, angle, n
            )
            reference = _ref_ellipse_mask(shape, center_row, center_col, a, b, angle, n)
        assert np.array_equal(_expand(box, shape), reference)
        assert box.count() == np.count_nonzero(reference)
        on_border = bool(
            reference[0].any() or reference[-1].any() or reference[:, 0].any()
            or reference[:, -1].any()
        )
        assert box.touches_image_border(shape) == on_border
        boxes.append((box, reference))

    all_bounds = np.array([box.bounds for box, _ in boxes], dtype=np.int64)
    for index, (box, reference) in enumerate(boxes):
        overlaps = sem_image._boxes_overlap(box.bounds, all_bounds)
        for other_index, (other, other_reference) in enumerate(boxes):
            shared = int(np.count_nonzero(reference & other_reference))
            assert box.intersection_count(other) == shared
            if shared:
                assert overlaps[other_index], (index, other_index)
            if other_index > index:
                assert sem_image._mask_iou(box, other) == _ref_iou(reference, other_reference)


@pytest.mark.parametrize(
    ("overrides", "expect", "fibre_count"),
    [
        ({}, "shrinks", 8),  # the left-border fibre is dropped
        ({"max_candidate_overlap_fraction": 0.0}, "rejected", 6),
        ({"drop_boundary_fibres": False}, "shrinks", 9),
    ],
)
def test_detected_fibres_match_full_image_reference(
    tmp_path: Path, overrides: dict[str, Any], expect: str, fibre_count: int
) -> None:
    image_path = _save(tmp_path / "sem.tif", _sem_like_image())
    config = _config(image_path, **overrides)
    image = load_grayscale_image(image_path)

    stats = {"rejected": 0, "shrinks": 0}
    reference = _reference_detection(image, config, stats)
    detection = sem_image._detect_fibres(image, config)

    # The image must exercise the branch this case is about, or the comparison proves little.
    assert stats[expect] > 0
    assert {c.kind for c in reference} == {"circle", "ellipse"}
    assert len(detection.candidates) == len(reference) == fibre_count
    for ours, ref in zip(detection.candidates, reference, strict=True):
        assert ours.kind == ref.kind
        assert ours.score == ref.score
        assert ours.area_px == ref.area_px
        assert ours.center_col_px == ref.center_col_px
        assert ours.center_row_px == ref.center_row_px
        assert ours.radius_px == ref.radius_px
        assert ours.ellipse_axes_px == ref.ellipse_axes_px
        assert ours.ellipse_angle_rad == ref.ellipse_angle_rad
        assert np.array_equal(_full_mask(ours, image.shape), ref.mask)

    # Public API: same fibres, placed with the pixel-centre convention.
    result = generate_circular_fibre_rve_from_sem(config)
    rows = image.shape[0]
    s = config.pixel_size
    expected_circles = [
        ((ref.center_col_px + 0.5) * s, (rows - ref.center_row_px - 0.5) * s, ref.radius_px * s)
        for ref in reference
        if ref.kind == "circle" and ref.radius_px is not None
    ]
    got_circles = [(f.center_x, f.center_y, f.radius) for f in result.geometry.circular_fibres]
    assert got_circles == expected_circles
    assert len(result.geometry.polygonal_fibres) == sum(r.kind == "ellipse" for r in reference)
    assert result.extracted_fibre_count == len(reference)


# --------------------------------------------------------------------------------------------
# Input formats (bug: 8-bit greyscale read as 0..255 against 0..1 thresholds)
# --------------------------------------------------------------------------------------------


def _write_formats(tmp_path: Path, image: np.ndarray) -> dict[str, Path]:
    grey8 = np.round(image * 255).astype(np.uint8)
    grey16 = grey8.astype(np.uint16) * 257  # same grey levels at 16 bits
    alpha = np.full_like(grey8, 255)
    return {
        "grey8": _save(tmp_path / "grey8.png", grey8),
        "grey16": _save(tmp_path / "grey16.png", grey16),
        "rgb": _save(tmp_path / "rgb.png", np.dstack([grey8] * 3)),
        "rgba": _save(tmp_path / "rgba.png", np.dstack([grey8] * 3 + [alpha])),
    }


def test_sem_detection_is_independent_of_storage_format(tmp_path: Path) -> None:
    paths = _write_formats(tmp_path, _sem_like_image())
    results = {
        name: generate_circular_fibre_rve_from_sem(_config(path))
        for name, path in paths.items()
    }
    reference = results["rgb"]
    assert reference.extracted_fibre_count >= 7
    assert reference.threshold_used == pytest.approx(0.42)
    for result in results.values():
        assert result.geometry.circular_fibres == reference.geometry.circular_fibres
        assert len(result.geometry.polygonal_fibres) == len(reference.geometry.polygonal_fibres)
        for ours, ref in zip(
            result.geometry.polygonal_fibres, reference.geometry.polygonal_fibres, strict=True
        ):
            assert np.array_equal(ours.points, ref.points)


def test_otsu_threshold_is_reported_on_the_unit_scale(tmp_path: Path) -> None:
    paths = _write_formats(tmp_path, _sem_like_image())
    result = generate_circular_fibre_rve_from_sem(_config(paths["grey8"], threshold=None))
    assert BACKGROUND < result.threshold_used < FIBRE
    assert result.extracted_fibre_count >= 7


# --------------------------------------------------------------------------------------------
# Model fitting failures (scikit-image 0.26 returns a falsy FailedEstimation, not None)
# --------------------------------------------------------------------------------------------


def test_failed_shape_fits_are_rejected() -> None:
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    failed = CircleModel.from_estimate(collinear)
    assert failed is not None and not failed  # the behaviour the old `is None` check missed
    assert sem_image._estimate_circle_model(collinear) is None
    assert sem_image._estimate_ellipse_model(collinear) is None

    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    circle_xy = np.column_stack((5.0 + 3.0 * np.cos(angles), -2.0 + 3.0 * np.sin(angles)))
    circle = sem_image._estimate_circle_model(circle_xy)
    assert circle is not None
    assert float(circle.radius) == pytest.approx(3.0)
    ellipse_xy = np.column_stack((5.0 + 4.0 * np.cos(angles), -2.0 + 2.0 * np.sin(angles)))
    fitted = sem_image._estimate_ellipse_model(ellipse_xy)
    assert fitted is not None
    assert sorted(float(v) for v in fitted.axis_lengths) == pytest.approx([2.0, 4.0])
