from __future__ import annotations

from rve2d.models import CircleFibre, CylinderFibre, Domain2D, Domain3D, GeometryModel
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
