"""Laminate module: stacking notation, ply properties, CLT and the 3D effective stiffness."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from _meshes import cube, write_msh
from _plies import CARBON, carbon_ply, isotropic_ply

from rve2d.config import config_from_dict
from rve2d.engines.common.materials import PhaseElasticity, compliance_3d, constitutive_matrix
from rve2d.engines.julia import runner
from rve2d.exceptions import ConfigError
from rve2d.laminate import (
    Laminate,
    abd_matrix,
    effective_3d_stiffness,
    laminate_constants,
    parse_stacking_sequence,
    ply_stiffness_from_rve,
)


@pytest.mark.parametrize(
    ("text", "angles"),
    [
        ("[0/90/90/0]", [0, 90, 90, 0]),
        ("[0/90/90/0]T", [0, 90, 90, 0]),
        ("[0/90]s", [0, 90, 90, 0]),
        ("[0/90]2s", [0, 90, 0, 90, 90, 0, 90, 0]),
        ("[0/90]3", [0, 90, 0, 90, 0, 90]),
        ("[0/±45/90]s", [0, 45, -45, 90, 90, -45, 45, 0]),
        ("[0/+-45/90]S", [0, 45, -45, 90, 90, -45, 45, 0]),
        ("[∓30]", [-30, 30]),
        ("[0_2/90]s", [0, 0, 90, 90, 0, 0]),
        ("[0₂/90]s", [0, 0, 90, 90, 0, 0]),
        ("[(0/90)_2/45]", [0, 90, 0, 90, 45]),
        ("[(±45)2/0]", [45, -45, 45, -45, 0]),
        ("[0/90/45]sb", [0, 90, 45, 90, 0]),
        ("[0 / -45 / 22.5]", [0, -45, 22.5]),
        ("0, 90, -45", [0, 90, -45]),
        ([0, 90, 45], [0, 90, 45]),
    ],
)
def test_stacking_notation(text: str | list[float], angles: list[float]) -> None:
    assert parse_stacking_sequence(text) == pytest.approx(angles)


@pytest.mark.parametrize("text", ["[0/9x0]", "[]", "[0/90", "[(0/90]", "[0/90]b", "[0_0]"])
def test_malformed_stacking_sequences_are_rejected(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_stacking_sequence(text)


def test_unidirectional_laminate_has_the_ply_constants() -> None:
    ply = carbon_ply()
    laminate = Laminate.from_sequence(ply, "[0_4]", 0.125)
    constants = laminate_constants(laminate)
    assert constants["ex"] == pytest.approx(CARBON["e1"])
    assert constants["ey"] == pytest.approx(CARBON["e2"])
    assert constants["gxy"] == pytest.approx(CARBON["g12"])
    assert constants["nuxy"] == pytest.approx(CARBON["nu12"])
    assert constants["flexural_ex"] == pytest.approx(CARBON["e1"])
    tolerance = 1e-12 * np.abs(ply.stiffness).max()
    np.testing.assert_allclose(
        effective_3d_stiffness(laminate), ply.stiffness, rtol=0, atol=tolerance
    )


def test_off_axis_modulus_follows_the_transformation_formula() -> None:
    laminate = Laminate.from_sequence(carbon_ply(), [30.0], 1.0)
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    e1, e2, g12, nu12 = CARBON["e1"], CARBON["e2"], CARBON["g12"], CARBON["nu12"]
    expected = 1.0 / (c**4 / e1 + (1.0 / g12 - 2.0 * nu12 / e1) * s**2 * c**2 + s**4 / e2)
    assert laminate_constants(laminate)["ex"] == pytest.approx(expected)


def test_cross_ply_abd_matches_hand_calculation() -> None:
    t = 0.125
    laminate = Laminate.from_sequence(carbon_ply(), "[0/90]s", t)
    q = laminate.reduced_stiffness(0.0)
    abd = abd_matrix(laminate)
    a, b, d = abd[:3, :3], abd[:3, 3:], abd[3:, 3:]
    assert a[0, 0] == pytest.approx(2 * t * (q[0, 0] + q[1, 1]))
    assert a[0, 1] == pytest.approx(4 * t * q[0, 1])
    assert a[2, 2] == pytest.approx(4 * t * q[2, 2])
    assert np.abs(b).max() < 1e-9 * np.abs(a).max()
    assert d[0, 0] == pytest.approx(2.0 / 3.0 * (7 * q[0, 0] + q[1, 1]) * t**3)
    constants = laminate_constants(laminate)
    assert constants["symmetric"] and constants["balanced"]


def test_angle_ply_coupling_flags() -> None:
    ply = carbon_ply()
    balanced = laminate_constants(Laminate.from_sequence(ply, "[±45]s", 0.125))
    assert balanced["symmetric"] and balanced["balanced"] and abs(balanced["eta_xy_x"]) < 1e-12
    unbalanced = laminate_constants(Laminate.from_sequence(ply, "[30]s", 0.125))
    assert not unbalanced["balanced"] and abs(unbalanced["eta_xy_x"]) > 0.1
    unsymmetric = laminate_constants(Laminate.from_sequence(ply, "[0/90]", 0.125))
    assert not unsymmetric["symmetric"]


def test_isotropic_plies_give_an_isotropic_laminate() -> None:
    ply = isotropic_ply()
    laminate = Laminate.from_sequence(ply, "[0/±45/90/30]s", 0.2)
    constants = laminate_constants(laminate)
    assert constants["ex"] == pytest.approx(3500.0)
    assert constants["ey"] == pytest.approx(3500.0)
    assert constants["nuxy"] == pytest.approx(0.35)
    assert constants["gxy"] == pytest.approx(3500.0 / 2.7)
    tolerance = 1e-12 * np.abs(ply.stiffness).max()
    np.testing.assert_allclose(
        effective_3d_stiffness(laminate), ply.stiffness, rtol=0, atol=tolerance
    )


@pytest.mark.parametrize("sequence", ["[0/90]s", "[0/±45/90]s", "[±30/0_2]s"])
def test_3d_effective_stiffness_agrees_with_clt_in_plane(sequence: str) -> None:
    laminate = Laminate.from_sequence(carbon_ply(), sequence, 0.15)
    clt = laminate_constants(laminate)
    compliance = np.linalg.inv(effective_3d_stiffness(laminate))
    assert 1.0 / compliance[0, 0] == pytest.approx(clt["ex"], rel=1e-10)
    assert 1.0 / compliance[1, 1] == pytest.approx(clt["ey"], rel=1e-10)
    assert 1.0 / compliance[5, 5] == pytest.approx(clt["gxy"], rel=1e-10)
    assert -compliance[0, 1] / compliance[0, 0] == pytest.approx(clt["nuxy"], rel=1e-10)


def test_ply_axes_from_an_rve_with_fibres_along_z() -> None:
    phase = PhaseElasticity("orthotropic", 0.0, 0.0, **CARBON)
    rve = constitutive_matrix(phase, "solid", (0.0, -90.0, 0.0))  # axis 1 of the ply along z
    ply = ply_stiffness_from_rve(rve, transverse_isotropy=False)
    expected = np.linalg.inv(compliance_3d(phase))
    np.testing.assert_allclose(ply, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


def test_transversely_isotropic_average() -> None:
    rng = np.random.default_rng(3)
    noisy = carbon_ply().stiffness * (1.0 + 0.02 * rng.standard_normal((6, 6)))
    average = ply_stiffness_from_rve(
        0.5 * (noisy + noisy.T)[np.ix_([1, 2, 0, 5, 3, 4], [1, 2, 0, 5, 3, 4])]
    )
    assert average[1, 1] == pytest.approx(average[2, 2])
    assert average[0, 1] == pytest.approx(average[0, 2])
    assert average[4, 4] == pytest.approx(average[5, 5])
    assert average[3, 3] == pytest.approx(0.5 * (average[1, 1] - average[1, 2]))
    assert np.abs(average[:3, 3:]).max() < 1e-9 * np.abs(average).max()


def layered_cube_msh(path: Path, n: int = 4) -> Path:
    points, cells = cube(n)
    centre_z = points[cells][:, :, 2].mean(axis=1)
    return write_msh(path, points, cells, np.where(centre_z < 0.5, 1, 2))


@pytest.mark.parametrize("engine", ["tensormesh", "julia"])
def test_3d_effective_stiffness_equals_a_layered_fe_rve(tmp_path: Path, engine: str) -> None:
    """A periodic cube of two plies (z < 0.5 at +30 deg, z > 0.5 at -60 deg) homogenized by
    the finite-element engine: linear tetrahedra represent the piecewise-uniform fields of a
    layered medium exactly, so the result must equal the closed-form stack formula."""
    if engine == "julia" and not (shutil.which("julia") and runner.is_instantiated()):
        pytest.skip("Julia engine environment not set up")
    from rve2d.workflow import solve_homogenization

    angles = (30.0, -60.0)
    solver: dict[str, object] = {
        "enabled": True,
        "kinematics": "solid",
        "boundary_condition": "periodic",
    }
    for phase, angle in zip(("matrix", "fibre"), angles, strict=True):
        solver[f"{phase}_material_model"] = "orthotropic"
        solver[f"{phase}_material_angle_z_deg"] = angle
        solver.update({f"{phase}_{name}": value for name, value in CARBON.items()})
    config = config_from_dict(
        {
            "mode": "synthetic",
            "dimension": 3,
            "synthetic": {
                "domain_width": 1.0,
                "domain_height": 1.0,
                "domain_depth": 1.0,
                "fibre_radius": 0.1,
                "target_volume_fraction": 0.1,
                "periodic_compatible": True,
            },
            "solver": solver,
        }
    )
    result = solve_homogenization(
        config, layered_cube_msh(tmp_path / "rve.msh"), tmp_path, engine=engine
    )
    laminate = Laminate(carbon_ply(), angles, (0.5, 0.5))
    expected = effective_3d_stiffness(laminate)
    np.testing.assert_allclose(
        np.asarray(result.homogenized_stiffness),
        expected,
        rtol=0,
        atol=1e-9 * np.abs(expected).max(),
    )
