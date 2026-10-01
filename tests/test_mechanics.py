from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rve2d.config import ConfigError, RVEConfig, config_from_dict
from rve2d.engines.common.homogenization import engineering_constants, resolve_material_rotation
from rve2d.engines.common.materials import PhaseElasticity, constitutive_matrix


def _isotropic_solid(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = lame
    stiffness[np.arange(3), np.arange(3)] += 2.0 * shear
    stiffness[np.arange(3, 6), np.arange(3, 6)] = shear
    return stiffness


def _orthotropic(**overrides: float) -> PhaseElasticity:
    values: dict[str, Any] = {
        "e1": 120.0,
        "e2": 15.0,
        "e3": 18.0,
        "g12": 6.0,
        "g13": 7.0,
        "g23": 5.0,
        "nu12": 0.22,
        "nu13": 0.20,
        "nu23": 0.25,
    }
    values.update(overrides)
    return PhaseElasticity("orthotropic", 0.0, 0.0, **values)


def _solver_config(dimension: int = 2, **solver: object) -> RVEConfig:
    synthetic: dict[str, object] = {
        "domain_width": 1.0,
        "domain_height": 1.0,
        "fibre_radius": 0.05,
        "target_volume_fraction": 0.2,
    }
    if dimension == 3:
        synthetic["domain_depth"] = 1.0
    return config_from_dict(
        {
            "mode": "synthetic",
            "dimension": dimension,
            "synthetic": synthetic,
            "solver": {"enabled": True, **solver},
        }
    )


def test_engineering_constants_from_isotropic_plane_stress_stiffness() -> None:
    youngs_modulus, poisson_ratio = 10.0, 0.25
    factor = youngs_modulus / (1.0 - poisson_ratio**2)
    stiffness = factor * np.array(
        [[1.0, poisson_ratio, 0.0], [poisson_ratio, 1.0, 0.0], [0.0, 0.0, (1 - poisson_ratio) / 2]]
    )
    engineering = engineering_constants(stiffness, "plane_stress")
    assert engineering["ex"] == pytest.approx(youngs_modulus)
    assert engineering["ey"] == pytest.approx(youngs_modulus)
    assert engineering["gxy"] == pytest.approx(youngs_modulus / (2.0 * (1.0 + poisson_ratio)))
    assert engineering["nuxy"] == pytest.approx(poisson_ratio)
    assert engineering["nuyx"] == pytest.approx(poisson_ratio)


def test_plane_strain_constants_are_labelled_as_plane_strain_moduli() -> None:
    youngs_modulus, poisson_ratio = 3.5, 0.35
    stiffness = _isotropic_solid(youngs_modulus, poisson_ratio)[np.ix_([0, 1, 5], [0, 1, 5])]
    engineering = engineering_constants(stiffness, "plane_strain")
    assert "ex" not in engineering
    assert engineering["ex_plane_strain"] == pytest.approx(youngs_modulus / (1 - poisson_ratio**2))
    assert engineering["nuxy_plane_strain"] == pytest.approx(poisson_ratio / (1 - poisson_ratio))


def test_engineering_constants_from_isotropic_solid_stiffness() -> None:
    youngs_modulus, poisson_ratio = 15.0, 0.28
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    for kinematics in ("solid", "generalized_plane_strain"):
        engineering = engineering_constants(
            _isotropic_solid(youngs_modulus, poisson_ratio), kinematics
        )
        for name in ("ex", "ey", "ez"):
            assert engineering[name] == pytest.approx(youngs_modulus)
        for name in ("gxy", "gyz", "gxz"):
            assert engineering[name] == pytest.approx(shear_modulus)
        for name in ("nuxy", "nuxz", "nuyz", "nuzy"):
            assert engineering[name] == pytest.approx(poisson_ratio)


def test_isotropic_matrices_match_closed_forms() -> None:
    phase = PhaseElasticity("isotropic", 3.5, 0.35)
    solid = _isotropic_solid(3.5, 0.35)
    np.testing.assert_allclose(constitutive_matrix(phase, "solid"), solid, rtol=1e-12)
    np.testing.assert_allclose(
        constitutive_matrix(phase, "plane_strain"), solid[np.ix_([0, 1, 5], [0, 1, 5])], rtol=1e-12
    )
    factor = 3.5 / (1.0 - 0.35**2)
    np.testing.assert_allclose(
        constitutive_matrix(phase, "plane_stress"),
        factor * np.array([[1.0, 0.35, 0.0], [0.35, 1.0, 0.0], [0.0, 0.0, 0.325]]),
        rtol=1e-12,
    )


def test_orthotropic_plane_strain_needs_the_3d_constants() -> None:
    in_plane_only = {
        "kinematics": "plane_strain",
        "fibre_material_model": "orthotropic",
        "fibre_e1": 100.0,
        "fibre_e2": 10.0,
        "fibre_g12": 5.0,
        "fibre_nu12": 0.2,
    }
    with pytest.raises(ConfigError, match="e3"):
        _solver_config(**in_plane_only)
    full = in_plane_only | {
        "fibre_e3": 10.0,
        "fibre_g13": 5.0,
        "fibre_g23": 4.0,
        "fibre_nu13": 0.2,
        "fibre_nu23": 0.25,
    }
    assert _solver_config(**full).solver.kinematics == "plane_strain"


def test_unstable_orthotropic_constants_are_rejected() -> None:
    with pytest.raises(ConfigError, match="positive definite"):
        _solver_config(
            kinematics="plane_stress",
            fibre_material_model="orthotropic",
            fibre_e1=10.0,
            fibre_e2=100.0,
            fibre_g12=5.0,
            fibre_nu12=0.5,  # nu12^2 > e1/e2
        )


def test_rotated_orthotropic_stiffness_changes_with_angle() -> None:
    phase = _orthotropic()
    base = constitutive_matrix(phase, "plane_stress")
    rotated = constitutive_matrix(phase, "plane_stress", (0.0, 0.0, 45.0))
    assert rotated[0, 2] != pytest.approx(0.0)
    assert not np.allclose(base, rotated)


def test_fibre_axis_along_z_for_generalized_plane_strain() -> None:
    phase = _orthotropic()
    stiffness = constitutive_matrix(phase, "generalized_plane_strain", (0.0, -90.0, 0.0))
    reference = constitutive_matrix(phase, "solid")
    assert stiffness[2, 2] == pytest.approx(reference[0, 0])  # axis 1 now along z
    assert stiffness[0, 0] == pytest.approx(reference[2, 2])


def test_axis_aligned_orthotropic_solid_stiffness_is_symmetric_positive_diagonal() -> None:
    stiffness = constitutive_matrix(_orthotropic(), "solid")
    assert stiffness.shape == (6, 6)
    assert np.allclose(stiffness, stiffness.T)
    assert np.all(np.diag(stiffness) > 0.0)


def test_rotated_orthotropic_solid_stiffness_changes_with_rotation() -> None:
    base = constitutive_matrix(_orthotropic(), "solid")
    rotated = constitutive_matrix(_orthotropic(), "solid", (20.0, 10.0, 30.0))
    assert np.allclose(rotated, rotated.T)
    assert not np.allclose(base, rotated)
    # a rotation preserves the eigenvalues of the Mandel (orthonormal-basis) form
    mandel = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
    assert np.linalg.eigvalsh(mandel @ rotated @ mandel) == pytest.approx(
        np.linalg.eigvalsh(mandel @ base @ mandel)
    )


def test_resolve_material_rotation_uses_geometry_fallback() -> None:
    config = _solver_config(kinematics="plane_stress")
    rotation = resolve_material_rotation(config, "fibre", {"orientation_deg": 30.0})
    assert rotation == pytest.approx((0.0, 0.0, 30.0))
    assert resolve_material_rotation(config, "matrix", {"orientation_deg": 30.0}) == (0, 0, 0)


def test_plane_stress_rotations_are_in_plane_only() -> None:
    config = _solver_config(kinematics="plane_stress", fibre_material_angle_z_deg=15.0)
    assert resolve_material_rotation(config, "fibre", {"orientation_deg": 30.0}) == (0, 0, 15)
    with pytest.raises(ConfigError, match="in plane only"):
        _solver_config(kinematics="plane_stress", fibre_material_angle_y_deg=-90.0)
    with pytest.raises(ConfigError, match="cannot mix"):
        _solver_config(
            kinematics="plane_stress", fibre_material_angle_deg=5.0, fibre_material_angle_z_deg=5.0
        )
    gps = _solver_config(kinematics="generalized_plane_strain", fibre_material_angle_y_deg=-90.0)
    assert resolve_material_rotation(gps, "fibre", None) == (0, -90, 0)


def test_resolve_material_rotation_uses_phase_metadata_in_3d() -> None:
    config = _solver_config(3, kinematics="solid")
    metadata = {"phase_orientation_rotations_deg": {"fibre": {"x": 10.0, "y": 5.0, "z": 20.0}}}
    assert resolve_material_rotation(config, "fibre", metadata) == pytest.approx((10, 5, 20))


def test_explicit_solver_rotation_overrides_phase_metadata() -> None:
    config = _solver_config(
        3,
        kinematics="solid",
        fibre_material_angle_x_deg=1.0,
        fibre_material_angle_y_deg=2.0,
        fibre_material_angle_z_deg=3.0,
    )
    metadata = {"phase_orientation_rotations_deg": {"fibre": {"x": 10.0, "y": 5.0, "z": 20.0}}}
    assert resolve_material_rotation(config, "fibre", metadata) == pytest.approx((1, 2, 3))
