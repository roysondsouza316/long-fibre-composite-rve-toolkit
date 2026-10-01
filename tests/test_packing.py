from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rve2d.config import (
    ConfigError,
    MeshConfig,
    SyntheticGenerationConfig,
    config_from_dict,
    load_config,
)
from rve2d.exceptions import RVEError
from rve2d.models import GeometryModel, circle_rectangle_intersection_area
from rve2d.synthetic_generation.circular import generate_circular_fibre_rve
from rve2d.synthetic_generation.cylindrical import generate_cylindrical_fibre_rve
from rve2d.synthetic_generation.packing import (
    mesh_compatibility_warnings,
    periodic_image_shifts,
    resolve_boundary_clearance,
)
from rve2d.validation.checks import validate_geometry

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).resolve().parent / "data" / "rsa_reference_centres.json"

RADIUS = 3.5
GAP = 0.35
CELL = 70.0


def _relaxation_config(
    target: float,
    seed: int,
    *,
    wrapping: bool = True,
    size: float = CELL,
    gap: float = GAP,
    **overrides: Any,
) -> SyntheticGenerationConfig:
    return SyntheticGenerationConfig(
        domain_width=size,
        domain_height=size,
        fibre_radius=RADIUS,
        target_volume_fraction=target,
        min_spacing=gap,
        random_seed=seed,
        periodic_compatible=True,
        periodic_wrapping=wrapping,
        packing_algorithm="relaxation",
        **overrides,
    )


def _distinct_centres(geometry: GeometryModel) -> np.ndarray:
    fibres = geometry.cylindrical_fibres if geometry.dimension == 3 else geometry.circular_fibres
    first: dict[int, tuple[float, float]] = {}
    for fibre in fibres:
        first.setdefault(fibre.fibre_id, (fibre.center_x, fibre.center_y))
    return np.array(list(first.values()), dtype=np.float64)


def _pairwise_gaps(centres: np.ndarray, radius: float, box: np.ndarray | None) -> np.ndarray:
    delta = centres[:, None, :] - centres[None, :, :]
    if box is not None:
        delta -= box * np.round(delta / box)
    distance = np.hypot(delta[..., 0], delta[..., 1])
    upper = np.triu_indices(centres.shape[0], k=1)
    return distance[upper] - 2.0 * radius


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("target", [0.60, 0.65])
def test_relaxation_with_wrapping_reaches_high_volume_fraction(target: float, seed: int) -> None:
    config = _relaxation_config(target, seed)
    result = generate_circular_fibre_rve(config)
    geometry = result.geometry

    expected_count = round(target * CELL**2 / (math.pi * RADIUS**2))
    assert geometry.fibre_count == expected_count
    # Periodic images complete every crossing fibre, so the fraction is exact.
    assert geometry.fibre_volume_fraction == pytest.approx(
        expected_count * math.pi * RADIUS**2 / CELL**2, rel=1e-12
    )
    assert abs(geometry.fibre_volume_fraction - target) <= 0.5 * math.pi * RADIUS**2 / CELL**2
    assert geometry.fibre_instance_count > geometry.fibre_count  # some fibres wrap around

    centres = _distinct_centres(geometry)
    assert np.all(centres >= 0.0) and np.all(centres < CELL)
    gaps = _pairwise_gaps(centres, RADIUS, np.array([CELL, CELL]))
    assert gaps.min() >= GAP

    # No near-tangent cut: every fibre surface keeps the clearance from faces and corners.
    clearance = resolve_boundary_clearance(config)
    assert clearance == pytest.approx(GAP)
    to_faces = np.minimum(centres, CELL - centres)
    assert np.all(np.abs(to_faces - RADIUS) >= clearance)
    corners = CELL * np.round(centres / CELL)
    to_corner = np.hypot(*(centres - corners).T)
    assert np.all(np.abs(to_corner - RADIUS) >= clearance)

    report = validate_geometry(geometry, minimum_spacing_requirement=GAP)
    assert report.valid, report.warnings
    assert report.clipped_fibres == 0
    assert report.wrapped_fibres > 0
    assert report.minimum_spacing is not None and report.minimum_spacing >= GAP
    assert report.minimum_boundary_clearance is not None
    assert report.minimum_boundary_clearance >= clearance


@pytest.mark.parametrize("target", [0.60, 0.65])
def test_relaxation_without_wrapping_keeps_fibres_inside(target: float) -> None:
    config = _relaxation_config(target, 7, wrapping=False, edge_clearance=0.1)
    geometry = generate_circular_fibre_rve(config).geometry
    centres = _distinct_centres(geometry)
    assert geometry.fibre_count == geometry.fibre_instance_count
    assert geometry.fibre_volume_fraction == pytest.approx(target, abs=0.01)
    wall_gap = max(config.edge_clearance, resolve_boundary_clearance(config))
    assert np.all(centres - RADIUS >= wall_gap - 1e-12)
    assert np.all(CELL - centres - RADIUS >= wall_gap - 1e-12)
    assert _pairwise_gaps(centres, RADIUS, None).min() >= GAP
    report = validate_geometry(geometry, minimum_spacing_requirement=GAP)
    assert report.valid, report.warnings


def test_relaxation_is_reproducible_from_the_seed() -> None:
    first = _distinct_centres(generate_circular_fibre_rve(_relaxation_config(0.6, 11)).geometry)
    second = _distinct_centres(generate_circular_fibre_rve(_relaxation_config(0.6, 11)).geometry)
    other = _distinct_centres(generate_circular_fibre_rve(_relaxation_config(0.6, 12)).geometry)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_relaxation_packs_three_hundred_fibres_quickly() -> None:
    config = _relaxation_config(0.65, 0, size=140.0, gap=0.5)
    start = time.perf_counter()
    geometry = generate_circular_fibre_rve(config).geometry
    elapsed = time.perf_counter() - start
    assert geometry.fibre_count == 331
    gaps = _pairwise_gaps(_distinct_centres(geometry), RADIUS, np.array([140.0, 140.0]))
    assert gaps.min() >= 0.5
    # Typically about one second; the bound only guards against pathological slow-downs.
    assert elapsed < 30.0


def test_cylinders_use_the_same_cross_section_packing() -> None:
    config_2d = _relaxation_config(0.6, 4)
    config_3d = _relaxation_config(0.6, 4, domain_depth=10.0)
    circles = generate_circular_fibre_rve(config_2d).geometry
    cylinders = generate_cylindrical_fibre_rve(config_3d).geometry
    assert [(c.center_x, c.center_y, c.fibre_id) for c in circles.circular_fibres] == [
        (c.center_x, c.center_y, c.fibre_id) for c in cylinders.cylindrical_fibres
    ]
    assert all(c.z_min == 0.0 and c.z_max == 10.0 for c in cylinders.cylindrical_fibres)
    assert cylinders.fibre_volume_fraction == pytest.approx(circles.fibre_volume_fraction)
    report = validate_geometry(cylinders, minimum_spacing_requirement=GAP)
    assert report.valid, report.warnings
    assert report.wrapped_fibres > 0


def test_periodic_images_tile_each_wrapped_fibre() -> None:
    geometry = generate_circular_fibre_rve(_relaxation_config(0.6, 2)).geometry
    area_by_id: dict[int, float] = {}
    for fibre in geometry.circular_fibres:
        area_by_id[fibre.fibre_id] = area_by_id.get(
            fibre.fibre_id, 0.0
        ) + circle_rectangle_intersection_area(
            fibre.center_x, fibre.center_y, fibre.radius, 0.0, 0.0, CELL, CELL
        )
    for area in area_by_id.values():
        assert area == pytest.approx(math.pi * RADIUS**2, rel=1e-12)


def test_periodic_image_shifts_include_corner_images_only_when_needed() -> None:
    near_corner = periodic_image_shifts(0.5, 0.5, 1.0, 0.0, 0.0, 10.0, 10.0)
    assert sorted(near_corner) == sorted([(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
    # Corner outside the disk: the disk crosses both faces but needs no diagonal image.
    crossing_both = periodic_image_shifts(0.9, 0.9, 1.0, 0.0, 0.0, 10.0, 10.0)
    assert sorted(crossing_both) == sorted([(10.0, 0.0), (0.0, 10.0)])
    assert periodic_image_shifts(5.0, 5.0, 1.0, 0.0, 0.0, 10.0, 10.0) == []
    far_side = periodic_image_shifts(9.5, 5.0, 1.0, 0.0, 0.0, 10.0, 10.0)
    assert far_side == [(-10.0, 0.0)]


def test_wrapped_random_sequential_placement() -> None:
    config = SyntheticGenerationConfig(
        domain_width=CELL,
        domain_height=CELL,
        fibre_radius=RADIUS,
        target_volume_fraction=0.4,
        min_spacing=GAP,
        random_seed=5,
        periodic_compatible=True,
        periodic_wrapping=True,
    )
    geometry = generate_circular_fibre_rve(config).geometry
    centres = _distinct_centres(geometry)
    assert _pairwise_gaps(centres, RADIUS, np.array([CELL, CELL])).min() >= GAP
    report = validate_geometry(geometry, minimum_spacing_requirement=GAP)
    assert report.valid, report.warnings
    assert report.wrapped_fibres > 0
    assert geometry.metadata["packing_algorithm"] == "random_sequential"
    assert geometry.metadata["periodic_wrapping"] is True


@pytest.mark.parametrize(
    "example",
    [
        "examples/2d/synthetic/periodic_solve.yaml",
        "examples/3d/synthetic/periodic_solve.yaml",
        "examples/2d/synthetic/basic.yaml",
        "examples/2d/synthetic/orthotropic_solve.yaml",
    ],
)
def test_random_sequential_reproduces_previous_layouts(example: str) -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))[example]
    config = load_config(ROOT / example)
    assert config.synthetic is not None
    assert config.synthetic.packing_algorithm == "random_sequential"
    assert not config.synthetic.periodic_wrapping
    if config.dimension == 3:
        result_3d = generate_cylindrical_fibre_rve(config.synthetic)
        fibres = [(f.center_x, f.center_y) for f in result_3d.geometry.cylindrical_fibres]
        attempts = result_3d.attempts
        geometry = result_3d.geometry
    else:
        result_2d = generate_circular_fibre_rve(config.synthetic)
        fibres = [(f.center_x, f.center_y) for f in result_2d.geometry.circular_fibres]
        attempts = result_2d.attempts
        geometry = result_2d.geometry
    assert attempts == reference["attempts"]
    assert geometry.fibre_count == reference["fibre_count"]
    assert len(fibres) == len(reference["centres"])
    for actual, expected in zip(fibres, reference["centres"], strict=True):
        assert actual == pytest.approx(tuple(expected), abs=1e-12)


def test_random_sequential_failure_names_the_options_to_enable() -> None:
    config = SyntheticGenerationConfig(
        domain_width=CELL,
        domain_height=CELL,
        fibre_radius=RADIUS,
        target_volume_fraction=0.55,
        min_spacing=GAP,
        random_seed=0,
        max_attempts=5_000,
        periodic_compatible=True,
    )
    with pytest.raises(RVEError) as error:
        generate_circular_fibre_rve(config)
    message = str(error.value)
    assert "packing_algorithm: relaxation" in message
    assert "periodic_wrapping: true" in message
    # The same config succeeds once the new options are enabled.
    relaxed = generate_circular_fibre_rve(
        _relaxation_config(0.55, 0, max_attempts=5_000)
    ).geometry
    assert relaxed.fibre_volume_fraction == pytest.approx(0.55, abs=0.005)


def test_relaxation_reports_impossible_targets() -> None:
    with pytest.raises(RVEError, match="hexagonal"):
        generate_circular_fibre_rve(_relaxation_config(0.88, 0))


def test_default_boundary_clearance() -> None:
    legacy = SyntheticGenerationConfig(
        domain_width=1.0, domain_height=1.0, fibre_radius=0.05, target_volume_fraction=0.2
    )
    assert resolve_boundary_clearance(legacy) == 0.0
    assert resolve_boundary_clearance(_relaxation_config(0.6, 0)) == pytest.approx(GAP)
    small_gap = _relaxation_config(0.6, 0, gap=0.05)
    assert resolve_boundary_clearance(small_gap) == pytest.approx(0.1 * RADIUS)
    explicit = _relaxation_config(0.6, 0, boundary_clearance=0.2)
    assert resolve_boundary_clearance(explicit) == pytest.approx(0.2)


def test_mesh_compatibility_warnings() -> None:
    config = _relaxation_config(0.6, 0)
    assert mesh_compatibility_warnings(config, MeshConfig(element_size_min=0.3)) == []
    warnings = mesh_compatibility_warnings(config, MeshConfig(element_size_min=1.0))
    assert any("min_spacing" in warning for warning in warnings)
    assert any("boundary_clearance" in warning for warning in warnings)
    legacy = SyntheticGenerationConfig(
        domain_width=1.0, domain_height=1.0, fibre_radius=0.05, target_volume_fraction=0.2
    )
    assert mesh_compatibility_warnings(legacy, MeshConfig()) == []


def _synthetic_payload(**overrides: Any) -> dict[str, Any]:
    synthetic: dict[str, Any] = {
        "domain_width": CELL,
        "domain_height": CELL,
        "fibre_radius": RADIUS,
        "target_volume_fraction": 0.6,
        "min_spacing": GAP,
        "periodic_compatible": True,
    }
    synthetic.update(overrides)
    return {"mode": "synthetic", "synthetic": synthetic}


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"packing_algorithm": "shaking"}, "packing_algorithm"),
        ({"packing_algorithm": "relaxation", "min_spacing": 0.0}, "min_spacing > 0"),
        ({"periodic_wrapping": True, "periodic_compatible": False}, "periodic_compatible"),
        ({"periodic_wrapping": True, "boundary_clearance": RADIUS}, "boundary_clearance"),
        ({"periodic_wrapping": True, "domain_width": 14.0}, "periodic_wrapping requires"),
        ({"boundary_clearance": -1.0}, "boundary_clearance"),
        ({"max_relaxation_iterations": 0}, "max_relaxation_iterations"),
        ({"shake_sweeps": -1}, "shake_sweeps"),
    ],
)
def test_invalid_packing_options_are_rejected(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        config_from_dict(_synthetic_payload(**overrides))


@pytest.mark.parametrize(
    "example",
    ["examples/2d/synthetic/high_vf_periodic.yaml", "examples/3d/synthetic/high_vf_periodic.yaml"],
)
def test_high_volume_fraction_examples_generate(example: str) -> None:
    config = load_config(ROOT / example)
    assert config.synthetic is not None
    assert config.synthetic.packing_algorithm == "relaxation"
    assert config.synthetic.periodic_wrapping
    assert mesh_compatibility_warnings(config.synthetic, config.mesh) == []
    if config.dimension == 3:
        geometry = generate_cylindrical_fibre_rve(config.synthetic).geometry
    else:
        geometry = generate_circular_fibre_rve(config.synthetic).geometry
    assert geometry.fibre_volume_fraction >= 0.60
    report = validate_geometry(
        geometry,
        minimum_spacing_requirement=config.synthetic.min_spacing,
        element_size=config.mesh.element_size_min,
    )
    assert report.valid, report.warnings
    assert not [note for note in report.notes if note.startswith("Mesh resolution")]
