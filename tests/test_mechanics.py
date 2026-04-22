from __future__ import annotations

import numpy as np
import pytest

from rve2d.config import ConfigError, config_from_dict
from rve2d.ferrite_bridge import (
    _engineering_constants_from_stiffness,
    _extract_summary_payload,
    _orthotropic_constitutive_matrix,
    _resolve_material_rotation_deg,
)


def test_engineering_constants_from_isotropic_plane_stress_stiffness() -> None:
    youngs_modulus = 10.0
    poisson_ratio = 0.25
    factor = youngs_modulus / (1.0 - poisson_ratio**2)
    stiffness = [
        [factor, factor * poisson_ratio, 0.0],
        [factor * poisson_ratio, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - poisson_ratio) / 2.0],
    ]
    engineering = _engineering_constants_from_stiffness(stiffness)
    assert engineering["ex"] == pytest.approx(youngs_modulus)
    assert engineering["ey"] == pytest.approx(youngs_modulus)
    assert engineering["gxy"] == pytest.approx(youngs_modulus / (2.0 * (1.0 + poisson_ratio)))
    assert engineering["nuxy"] == pytest.approx(poisson_ratio)
    assert engineering["nuyx"] == pytest.approx(poisson_ratio)


def test_engineering_constants_from_isotropic_solid_stiffness() -> None:
    youngs_modulus = 15.0
    poisson_ratio = 0.28
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    stiffness = [
        [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
        [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0, 0.0, 0.0],
        [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
    ]
    engineering = _engineering_constants_from_stiffness(stiffness)
    assert engineering["ex"] == pytest.approx(youngs_modulus)
    assert engineering["ey"] == pytest.approx(youngs_modulus)
    assert engineering["ez"] == pytest.approx(youngs_modulus)
    assert engineering["gxy"] == pytest.approx(shear_modulus)
    assert engineering["gyz"] == pytest.approx(shear_modulus)
    assert engineering["gxz"] == pytest.approx(shear_modulus)
    assert engineering["nuxy"] == pytest.approx(poisson_ratio)
    assert engineering["nuxz"] == pytest.approx(poisson_ratio)
    assert engineering["nuyz"] == pytest.approx(poisson_ratio)


def test_orthotropic_plane_strain_is_rejected() -> None:
    with pytest.raises(ConfigError):
        config_from_dict(
            {
                "mode": "synthetic",
                "synthetic": {
                    "domain_width": 1.0,
                    "domain_height": 1.0,
                    "fibre_radius": 0.05,
                    "target_volume_fraction": 0.2,
                },
                "solver": {
                    "enabled": True,
                    "kinematics": "plane_strain",
                    "fibre_material_model": "orthotropic",
                    "fibre_e1": 100.0,
                    "fibre_e2": 10.0,
                    "fibre_g12": 5.0,
                    "fibre_nu12": 0.2,
                },
            }
        )


def test_rotated_orthotropic_stiffness_changes_with_angle() -> None:
    base = _orthotropic_constitutive_matrix(
        e1=140.0,
        e2=12.0,
        e3=None,
        g12=5.0,
        g13=None,
        g23=None,
        nu12=0.22,
        nu13=None,
        nu23=None,
        kinematics="plane_stress",
        rotation_deg=(0.0, 0.0, 0.0),
    )
    rotated = _orthotropic_constitutive_matrix(
        e1=140.0,
        e2=12.0,
        e3=None,
        g12=5.0,
        g13=None,
        g23=None,
        nu12=0.22,
        nu13=None,
        nu23=None,
        kinematics="plane_stress",
        rotation_deg=(0.0, 0.0, 45.0),
    )
    assert rotated[0, 2] != pytest.approx(0.0)
    assert not np.allclose(base, rotated)


def test_resolve_material_rotation_uses_geometry_fallback() -> None:
    rotation = _resolve_material_rotation_deg(
        dimension=2,
        phase_label="fibre",
        configured_angle_deg=None,
        configured_angle_x_deg=None,
        configured_angle_y_deg=None,
        configured_angle_z_deg=None,
        geometry_metadata={"orientation_deg": 30.0},
        fallback_key="orientation_deg",
    )
    assert rotation == pytest.approx((0.0, 0.0, 30.0))


def test_resolve_material_rotation_uses_phase_metadata_in_3d() -> None:
    rotation = _resolve_material_rotation_deg(
        dimension=3,
        phase_label="fibre",
        configured_angle_deg=None,
        configured_angle_x_deg=None,
        configured_angle_y_deg=None,
        configured_angle_z_deg=None,
        geometry_metadata={
            "phase_orientation_rotations_deg": {
                "fibre": {"x": 10.0, "y": 5.0, "z": 20.0},
            }
        },
        fallback_key=None,
    )
    assert rotation == pytest.approx((10.0, 5.0, 20.0))


def test_explicit_solver_rotation_overrides_phase_metadata() -> None:
    rotation = _resolve_material_rotation_deg(
        dimension=3,
        phase_label="fibre",
        configured_angle_deg=None,
        configured_angle_x_deg=1.0,
        configured_angle_y_deg=2.0,
        configured_angle_z_deg=3.0,
        geometry_metadata={
            "phase_orientation_rotations_deg": {
                "fibre": {"x": 10.0, "y": 5.0, "z": 20.0},
            }
        },
        fallback_key=None,
    )
    assert rotation == pytest.approx((1.0, 2.0, 3.0))


def test_axis_aligned_orthotropic_solid_stiffness_is_symmetric_positive_diagonal() -> None:
    stiffness = _orthotropic_constitutive_matrix(
        e1=120.0,
        e2=15.0,
        e3=18.0,
        g12=6.0,
        g13=7.0,
        g23=5.0,
        nu12=0.22,
        nu13=0.20,
        nu23=0.25,
        kinematics="solid",
        rotation_deg=(0.0, 0.0, 0.0),
    )
    assert stiffness.shape == (6, 6)
    assert np.allclose(stiffness, stiffness.T)
    assert np.all(np.diag(stiffness) > 0.0)


def test_rotated_orthotropic_solid_stiffness_changes_with_rotation() -> None:
    base = _orthotropic_constitutive_matrix(
        e1=120.0,
        e2=15.0,
        e3=18.0,
        g12=6.0,
        g13=7.0,
        g23=5.0,
        nu12=0.22,
        nu13=0.20,
        nu23=0.25,
        kinematics="solid",
        rotation_deg=(0.0, 0.0, 0.0),
    )
    rotated = _orthotropic_constitutive_matrix(
        e1=120.0,
        e2=15.0,
        e3=18.0,
        g12=6.0,
        g13=7.0,
        g23=5.0,
        nu12=0.22,
        nu13=0.20,
        nu23=0.25,
        kinematics="solid",
        rotation_deg=(20.0, 10.0, 30.0),
    )
    assert np.allclose(rotated, rotated.T)
    assert not np.allclose(base, rotated)


def test_extract_summary_payload_parses_boundary_tractions() -> None:
    payload = _extract_summary_payload(
        "\n".join(
            [
                "C_ROW_1=1.0,2.0,3.0",
                "C_ROW_2=4.0,5.0,6.0",
                "C_ROW_3=7.0,8.0,9.0",
                "SIGMA_BAR_1=1.0,0.0,0.0",
                "SIGMA_BAR_2=0.0,1.0,0.0",
                "SIGMA_BAR_3=0.0,0.0,1.0",
                "TRACTION_1_RIGHT=10.0,11.0",
                "TRACTION_1_LEFT=-10.0,-11.0",
                "TRACTION_1_TOP=12.0,13.0",
                "TRACTION_1_BOTTOM=-12.0,-13.0",
            ]
        )
    )
    assert payload["boundary_tractions"][0]["right"] == pytest.approx([10.0, 11.0])
    assert payload["boundary_tractions"][0]["bottom"] == pytest.approx([-12.0, -13.0])
