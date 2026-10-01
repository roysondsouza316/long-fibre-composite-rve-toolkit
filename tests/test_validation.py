from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rve2d.export.writers import write_metadata_bundle
from rve2d.models import (
    CircleFibre,
    CylinderFibre,
    Domain2D,
    Domain3D,
    GeometryModel,
    PeriodicBoundaryPair,
    PolygonFibre,
)
from rve2d.validation.checks import validate_geometry


def test_validate_geometry_flags_clipping_and_spacing() -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            CircleFibre(center_x=0.05, center_y=0.05, radius=0.06, fibre_id=1),
            CircleFibre(center_x=0.18, center_y=0.15, radius=0.06, fibre_id=2),
        ],
    )
    report = validate_geometry(geometry, minimum_spacing_requirement=0.05)
    assert not report.valid
    assert report.clipped_fibres == 1
    assert report.minimum_spacing is not None
    assert report.minimum_spacing < 0.05


def test_validate_geometry_flags_3d_clipping_and_spacing() -> None:
    geometry = GeometryModel(
        domain=Domain3D(width=1.0, height=1.0, depth=0.5),
        cylindrical_fibres=[
            CylinderFibre(
                center_x=0.05,
                center_y=0.05,
                radius=0.06,
                z_min=0.0,
                z_max=0.5,
                fibre_id=1,
            ),
            CylinderFibre(
                center_x=0.18,
                center_y=0.15,
                radius=0.06,
                z_min=-0.01,
                z_max=0.5,
                fibre_id=2,
            ),
        ],
    )
    report = validate_geometry(geometry, minimum_spacing_requirement=0.05)
    assert not report.valid
    assert report.clipped_fibres == 2
    assert report.minimum_spacing is not None
    assert report.minimum_spacing < 0.05


def _periodic_pairs_2d(width: float, height: float) -> list[PeriodicBoundaryPair]:
    return [
        PeriodicBoundaryPair("x_periodic", "left", "right", (width, 0.0)),
        PeriodicBoundaryPair("y_periodic", "bottom", "top", (0.0, height)),
    ]


def test_removed_specks_are_informational() -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[CircleFibre(center_x=0.5, center_y=0.5, radius=0.1, fibre_id=1)],
    )
    report = validate_geometry(geometry, disconnected_artifacts=5)
    assert report.valid
    assert report.warnings == []
    assert report.removed_artifacts == 5
    assert report.disconnected_artifacts == 5  # deprecated alias, same value
    assert any("removed" in note for note in report.notes)
    assert validate_geometry(geometry, removed_artifacts=2).removed_artifacts == 2


def test_quality_report_json_keeps_the_old_key(tmp_path: Path) -> None:
    geometry = GeometryModel(domain=Domain2D(width=1.0, height=1.0))
    report = validate_geometry(geometry, removed_artifacts=3)
    write_metadata_bundle(tmp_path, geometry, report, phase_tags={}, boundary_tags={})
    payload = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["removed_artifacts"] == 3
    assert payload["disconnected_artifacts"] == 3
    assert payload["notes"]


def test_wrapped_periodic_fibres_are_not_clipped() -> None:
    radius = 0.05
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            # fibre 1 crosses the left face, fibre 2 contains the bottom-left corner
            CircleFibre(center_x=0.02, center_y=0.5, radius=radius, fibre_id=1),
            CircleFibre(center_x=1.02, center_y=0.5, radius=radius, fibre_id=1),
            CircleFibre(center_x=0.02, center_y=0.02, radius=radius, fibre_id=2),
            CircleFibre(center_x=1.02, center_y=0.02, radius=radius, fibre_id=2),
            CircleFibre(center_x=0.02, center_y=1.02, radius=radius, fibre_id=2),
            CircleFibre(center_x=1.02, center_y=1.02, radius=radius, fibre_id=2),
            CircleFibre(center_x=0.5, center_y=0.5, radius=radius, fibre_id=3),
        ],
        periodic_pairs=_periodic_pairs_2d(1.0, 1.0),
    )
    report = validate_geometry(geometry, minimum_spacing_requirement=0.01)
    assert report.valid, report.warnings
    assert report.clipped_fibres == 0
    assert report.wrapped_fibres == 2
    assert geometry.fibre_count == 3
    assert geometry.fibre_volume_fraction == pytest.approx(3 * np.pi * radius**2)


def test_periodic_fibre_missing_an_image_is_clipped() -> None:
    radius = 0.05
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            # the diagonal image (1.02, 1.02) of the corner fibre is missing
            CircleFibre(center_x=0.02, center_y=0.02, radius=radius, fibre_id=1),
            CircleFibre(center_x=1.02, center_y=0.02, radius=radius, fibre_id=1),
            CircleFibre(center_x=0.02, center_y=1.02, radius=radius, fibre_id=1),
        ],
        periodic_pairs=_periodic_pairs_2d(1.0, 1.0),
    )
    report = validate_geometry(geometry)
    assert not report.valid
    assert report.clipped_fibres == 1
    assert report.wrapped_fibres == 0


def test_periodic_spacing_uses_minimum_image_distances() -> None:
    # Two fibres near opposite faces: far apart in the box, 0.01 apart across the boundary.
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            CircleFibre(center_x=0.055, center_y=0.5, radius=0.05, fibre_id=1),
            CircleFibre(center_x=0.945, center_y=0.5, radius=0.05, fibre_id=2),
        ],
        periodic_pairs=_periodic_pairs_2d(1.0, 1.0),
    )
    report = validate_geometry(geometry, minimum_spacing_requirement=0.02)
    assert report.minimum_spacing == pytest.approx(0.01)
    assert not report.valid
    non_periodic = GeometryModel(domain=geometry.domain, circular_fibres=geometry.circular_fibres)
    assert validate_geometry(non_periodic).minimum_spacing == pytest.approx(0.79)


def test_element_size_notes_flag_gaps_thinner_than_an_element() -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=1.0, height=1.0),
        circular_fibres=[
            CircleFibre(center_x=0.3, center_y=0.5, radius=0.1, fibre_id=1),
            CircleFibre(center_x=0.505, center_y=0.5, radius=0.1, fibre_id=2),
        ],
    )
    coarse = validate_geometry(geometry, element_size=0.02)
    assert coarse.valid
    assert any(note.startswith("Mesh resolution") for note in coarse.notes)
    fine = validate_geometry(geometry, element_size=0.004)
    assert not [note for note in fine.notes if note.startswith("Mesh resolution")]


def test_fibres_outside_the_domain_are_not_clipped() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    geometry = GeometryModel(
        domain=Domain2D(width=10.0, height=10.0),
        polygonal_fibres=[
            PolygonFibre(points=square + 4.0, fibre_id=1),  # inside
            PolygonFibre(points=square + [9.5, 4.0], fibre_id=2),  # crosses x = 10
            PolygonFibre(points=square + 12.0, fibre_id=3),  # entirely outside (image window)
        ],
    )
    report = validate_geometry(geometry)
    assert report.clipped_fibres == 1
    assert report.outside_fibres == 1
    assert any("outside the domain" in note for note in report.notes)
    assert geometry.fibre_volume_fraction == pytest.approx(1.5 / 100.0)
