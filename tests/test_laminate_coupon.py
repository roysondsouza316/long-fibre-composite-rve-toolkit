"""Laminate coupon tests in tension and compression with every ply model: elastic limit,
RVE-curve reproduction, maximum stress, Hashin and continuum damage against closed forms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _plies import CARBON, carbon_ply

from rve2d.exceptions import ConfigError
from rve2d.laminate import (
    CouponSettings,
    FractureEnergies,
    Laminate,
    PlyCurve,
    PlyStrengths,
    build_ply_model,
    coupon_test,
    laminate_constants,
)
from rve2d.laminate.damage import DAMAGE_CAP, _secant

# A softening transverse curve (MPa): linear to 50 MPa at 0.5 %, down to 10 MPa at 2 %
TRANSVERSE = PlyCurve(np.array([0.0, 0.005, 0.02]), np.array([0.0, 50.0, 10.0]))
# An elastic-plastic shear curve (MPa): G12 = 5 GPa up to 60 MPa, hardening slowly after
SHEAR = PlyCurve(np.array([0.0, 0.012, 0.08]), np.array([0.0, 60.0, 90.0]))
STRENGTHS = PlyStrengths(xt=1500.0, xc=1000.0, yt=50.0, yc=150.0, s12=70.0)
ENERGIES = FractureEnergies(
    fibre_tension=90.0, fibre_compression=80.0, matrix_tension=0.3, matrix_compression=1.0
)


def ply_with_strengths() -> object:
    ply = carbon_ply(strengths=STRENGTHS)
    from dataclasses import replace

    return replace(ply, fracture_energies=ENERGIES)


# -- elastic and RVE-curve models ----------------------------------------------------------
@pytest.mark.parametrize(
    ("sequence", "direction", "constant"),
    [
        ("[0/±45/90]s", "x", "ex"),
        ("[0/±45/90]s", "y", "ey"),
        ("[0/±45/90]s", "xy", "gxy"),
        ("[30_4]", "x", "ex"),  # unbalanced: shear-extension coupling
        ("[0/90]", "x", "ex"),  # unsymmetric: bending-extension coupling, free curvature
    ],
)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_elastic_response_matches_clt(
    sequence: str, direction: str, constant: str, sign: float
) -> None:
    laminate = Laminate.from_sequence(carbon_ply(), sequence, 0.125)
    settings = CouponSettings(direction=direction, max_strain=sign * 0.01, steps=4)
    result = coupon_test(laminate, settings, "elastic")
    assert result.completed
    expected = laminate_constants(laminate)[constant]
    np.testing.assert_allclose(result.stress[1:] / result.strain[1:], expected, rtol=1e-8)


def test_transverse_laminate_reproduces_the_rve_curve() -> None:
    laminate = Laminate.from_sequence(carbon_ply(transverse_tension=TRANSVERSE), "[90_4]", 0.2)
    result = coupon_test(laminate, CouponSettings(max_strain=0.018, steps=36))
    expected = [TRANSVERSE.stress_at(strain) for strain in result.strain]
    np.testing.assert_allclose(result.stress, expected, rtol=1e-8, atol=1e-8)
    summary = result.summary()
    assert summary["peak_stress"] == pytest.approx(50.0)
    # flagged on the step that reaches the peak strain of the curve (steps of 0.05 %)
    assert summary["first_ply_failure"]["event"] == "transverse tension peak passed"
    assert summary["first_ply_failure"]["strain"] == pytest.approx(0.005, abs=6e-4)


def test_transverse_compression_follows_the_compression_curve() -> None:
    compression = PlyCurve(np.array([0.0, 0.01, 0.03]), np.array([0.0, 120.0, 110.0]))
    ply = carbon_ply(transverse_tension=TRANSVERSE, transverse_compression=compression)
    laminate = Laminate.from_sequence(ply, "[90_4]", 0.2)
    result = coupon_test(laminate, CouponSettings(max_strain=-0.025, steps=50))
    expected = [-compression.stress_at(strain) for strain in result.strain]
    np.testing.assert_allclose(result.stress, expected, rtol=1e-8, atol=1e-8)
    assert result.summary()["first_ply_failure"]["event"] == "transverse compression peak passed"


def test_fibre_failure_ends_a_unidirectional_test() -> None:
    laminate = Laminate.from_sequence(carbon_ply(strengths=PlyStrengths(xt=1500.0)), "[0_4]", 0.125)
    result = coupon_test(laminate, CouponSettings(max_strain=0.02, steps=100))
    summary = result.summary()
    assert summary["first_fibre_failure"]["event"] == "fibre tension failure"
    # the failure is located within 2 % of a step (2e-4): no step-size error in the strength
    assert summary["first_fibre_failure"]["strain"] == pytest.approx(
        1500.0 / CARBON["e1"], abs=5e-6
    )
    assert summary["peak_stress"] == pytest.approx(1500.0, rel=1e-3)
    assert result.stress[-1] == 0.0 and result.message == "all load-carrying plies failed"


def test_cross_ply_knee_then_fibre_failure() -> None:
    ply = carbon_ply(transverse_tension=TRANSVERSE, strengths=PlyStrengths(xt=1500.0))
    laminate = Laminate.from_sequence(ply, "[0/90]s", 0.125)
    result = coupon_test(laminate, CouponSettings(max_strain=0.02, steps=200))
    events = [(e["event"], e["ply"]) for e in result.events]
    assert events[:2] == [
        ("transverse tension peak passed", 2),
        ("transverse tension peak passed", 3),
    ]
    assert ("fibre tension failure", 1) in events and ("fibre tension failure", 4) in events
    strain, stress = result.strain, result.stress
    peak = int(np.argmax(stress))
    # 0-ply fibres fail (sigma_1 = Xt; the softened 90 plies change the Poisson contraction)
    assert strain[peak] == pytest.approx(1500.0 / CARBON["e1"], abs=1e-4)
    initial = stress[1] / strain[1]
    after_knee = (stress[peak] - stress[60]) / (strain[peak] - strain[60])
    assert after_knee < 0.95 * initial  # the 90 plies have softened (strain > 0.5 %)
    assert np.all(np.diff(stress[: peak + 1]) > 0)  # the 0 plies keep loading
    assert result.stress[-1] < 0.05 * stress[peak]  # load lost after fibre failure


def test_angle_ply_follows_the_shear_curve() -> None:
    ply = carbon_ply(shear=SHEAR, transverse_tension=TRANSVERSE)
    laminate = Laminate.from_sequence(ply, "[±45]2s", 0.125)
    elastic = laminate_constants(laminate)["ex"]
    result = coupon_test(laminate, CouponSettings(max_strain=0.03, steps=60))
    assert result.stress[1] / result.strain[1] == pytest.approx(elastic, rel=1e-6)
    assert result.stress[-1] < 0.6 * elastic * result.strain[-1]  # strongly nonlinear


def test_secant_unloading_memory() -> None:
    loaded, kappa = _secant(TRANSVERSE, 0.01, 0.0, 1e4)
    assert kappa == 0.01 and loaded == pytest.approx(TRANSVERSE.stress_at(0.01))
    unloaded, kept = _secant(TRANSVERSE, 0.005, kappa, 1e4)
    assert kept == 0.01 and unloaded == pytest.approx(0.5 * loaded)


def test_elongation_force_and_files(tmp_path: Path) -> None:
    laminate = Laminate.from_sequence(carbon_ply(), "[0/90]s", 0.125)
    settings = CouponSettings(max_strain=0.01, steps=5, gauge_length=150.0, width=25.0)
    result = coupon_test(laminate, settings, "elastic")
    last = result.records[-1]
    assert last["elongation"] == pytest.approx(0.01 * 150.0)
    assert last["force"] == pytest.approx(last["stress"] * 0.5 * 25.0)
    curve, summary = result.write(tmp_path)
    header = curve.read_text().splitlines()[0]
    assert header.startswith("strain,stress,") and "elongation" in header and "force" in header
    assert '"initial_modulus"' in summary.read_text()


def test_rising_curves_and_curve_ends_are_reported() -> None:
    rising = PlyCurve(np.array([0.0, 0.01, 0.02]), np.array([0.0, 40.0, 50.0]))
    laminate = Laminate.from_sequence(carbon_ply(transverse_tension=rising), "[90_4]", 0.2)
    result = coupon_test(laminate, CouponSettings(max_strain=0.03, steps=30))
    summary = result.summary()
    assert summary["first_ply_failure"] is None  # no peak inside a rising curve
    assert summary["first_curve_exceeded"]["event"] == "transverse tension curve exceeded"
    assert summary["first_curve_exceeded"]["strain"] == pytest.approx(0.021)  # first step past 2 %
    assert result.stress[-1] == pytest.approx(50.0)  # held at the last value of the curve


# -- maximum stress, Hashin, continuum damage ----------------------------------------------
def cross_ply_first_failure_strain(laminate: Laminate) -> float:
    """Laminate strain at which the 90 plies of a symmetric cross-ply reach Yt (elastic CLT)."""
    q = laminate.reduced_stiffness(0.0)
    nu_xy = laminate_constants(laminate)["nuxy"]
    return STRENGTHS.yt / (q[1, 1] - q[0, 1] * nu_xy)  # 90 ply: eps_2 = eps_x, eps_1 = eps_y


@pytest.mark.parametrize(
    ("sign", "mode", "strength", "modulus"),
    [
        (1.0, "fibre tension", 1500.0, CARBON["e1"]),
        (-1.0, "fibre compression", 1000.0, CARBON["e1"]),
    ],
)
@pytest.mark.parametrize("model", ["max_stress", "hashin"])
def test_unidirectional_failure_at_the_fibre_strength(
    model: str, sign: float, mode: str, strength: float, modulus: float
) -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[0_4]", 0.125)
    result = coupon_test(laminate, CouponSettings(max_strain=sign * 0.02, steps=200), model)
    summary = result.summary()
    assert summary["first_fibre_failure"]["event"] == f"{mode} failure"
    assert abs(summary["peak_stress"]) == pytest.approx(strength, rel=1e-3)
    assert abs(summary["first_fibre_failure"]["strain"]) == pytest.approx(
        strength / modulus, abs=5e-6
    )
    assert abs(result.stress[-1]) < 0.15 * strength  # the plies are discounted


def test_max_stress_cross_ply_first_and_last_ply_failure() -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[0/90]s", 0.125)
    result = coupon_test(laminate, CouponSettings(max_strain=0.02, steps=400), "max_stress")
    summary = result.summary()
    first = summary["first_ply_failure"]
    assert first["event"] == "matrix tension failure" and first["ply"] in (2, 3)
    assert first["strain"] == pytest.approx(cross_ply_first_failure_strain(laminate), abs=2e-6)
    stress = result.stress
    drop = int(np.argmax(result.strain >= first["strain"] - 1e-12))
    assert stress[drop] < stress[drop - 1] * (1 + 1e-9) or stress[drop + 1] < stress[drop]
    # after the 90 plies are discounted, the 0 plies carry the load up to fibre failure
    assert summary["first_fibre_failure"]["strain"] == pytest.approx(
        1500.0 / CARBON["e1"], abs=1e-4
    )


def test_hashin_interaction_lowers_the_off_axis_strength() -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[30_4]", 0.125)
    settings = CouponSettings(max_strain=0.02, steps=400)
    max_stress = coupon_test(laminate, settings, "max_stress").summary()["first_ply_failure"]
    hashin = coupon_test(laminate, settings, "hashin").summary()["first_ply_failure"]
    # combined transverse tension and shear fail earlier with the interactive criterion
    assert hashin["event"] == "matrix tension failure"
    assert hashin["strain"] < max_stress["strain"] - 1e-4


def test_hashin_keeps_a_residual_stiffness() -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[90_4]", 0.125)
    result = coupon_test(laminate, CouponSettings(max_strain=0.01, steps=100), "hashin")
    summary = result.summary()
    assert summary["first_ply_failure"]["event"] == "matrix tension failure"
    assert summary["peak_stress"] == pytest.approx(50.0, rel=1e-3)
    # after failure E2 keeps 20 % (Tserpes et al.): the stress-strain slope drops to about 0.2
    slope = (result.stress[-1] - result.stress[-11]) / (result.strain[-1] - result.strain[-11])
    assert slope == pytest.approx(0.2 * result.stress[1] / result.strain[1], rel=0.02)


@pytest.mark.parametrize(
    ("sequence", "sign", "strength", "energy", "modulus"),
    [
        ("[90_4]", 1.0, 50.0, 0.3, CARBON["e2"]),  # matrix tension
        ("[0_4]", 1.0, 1500.0, 90.0, CARBON["e1"]),  # fibre tension
        ("[0_4]", -1.0, 1000.0, 80.0, CARBON["e1"]),  # fibre compression
    ],
)
def test_continuum_damage_peaks_at_the_strength_and_dissipates_the_fracture_energy(
    sequence: str, sign: float, strength: float, energy: float, modulus: float
) -> None:
    length = 0.5  # mm
    laminate = Laminate.from_sequence(ply_with_strengths(), sequence, 0.125)
    onset = strength / modulus
    end = 2.0 * energy / (length * strength)  # strain where the softening line reaches zero
    settings = CouponSettings(max_strain=sign * 1.2 * end, steps=2000, stop_fraction=1e-3)
    result = coupon_test(laminate, settings, "continuum_damage", characteristic_length=length)
    summary = result.summary()
    assert not summary["warnings"]
    assert abs(summary["peak_stress"]) == pytest.approx(strength, rel=0.01)
    step = abs(settings.max_strain) / settings.steps  # the peak is the first step past onset
    assert abs(summary["strain_at_peak"]) == pytest.approx(onset, abs=1.01 * step)
    # area under the stress-strain curve = fracture energy / crack band (per unit volume)
    stress, strain = np.abs(result.stress), np.abs(result.strain)
    work = float(np.sum(0.5 * (stress[1:] + stress[:-1]) * np.diff(strain)))
    assert work == pytest.approx(energy / length, rel=0.02)
    # linear softening: halfway down the stress is half the strength
    middle = int(np.argmin(np.abs(strain - 0.5 * (onset + end))))
    assert stress[middle] == pytest.approx(0.5 * strength, rel=0.05)


def test_continuum_damage_warns_about_snap_back() -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[90_4]", 0.125)
    critical = 2.0 * 0.3 * CARBON["e2"] / 50.0**2  # 2 G E / Y^2
    result = coupon_test(
        laminate, CouponSettings(max_strain=0.01, steps=50), "continuum_damage",
        characteristic_length=2.0 * critical,
    )  # fmt: skip
    assert result.summary()["warnings"] and "matrix tension" in result.summary()["warnings"][0]
    assert result.stress[-1] < 0.05 * result.summary()["peak_stress"]


def test_continuum_damage_states_reach_the_cap_and_stop_the_test() -> None:
    laminate = Laminate.from_sequence(ply_with_strengths(), "[0_4]", 0.125)
    result = coupon_test(
        laminate, CouponSettings(max_strain=0.5, steps=500), "continuum_damage",
        characteristic_length=0.5,
    )  # fmt: skip
    events = [event["event"] for event in result.events]
    assert "fibre tension damage onset" in events
    assert result.message  # stopped once the stress fell below 5 % of the peak
    assert DAMAGE_CAP < 1.0


def test_models_report_missing_strengths_and_energies() -> None:
    ply = carbon_ply(strengths=PlyStrengths(xt=1500.0, xc=1000.0, yt=50.0))
    with pytest.raises(ConfigError, match="yc, s12"):
        build_ply_model("max_stress", ply)
    with pytest.raises(ConfigError, match="fracture_energies"):
        build_ply_model("continuum_damage", carbon_ply(strengths=STRENGTHS))
    with pytest.raises(ConfigError, match="Unknown ply damage model"):
        build_ply_model("tsai_wu", ply)
