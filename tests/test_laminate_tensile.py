"""Laminate tensile test: elastic limit, RVE-curve reproduction, fibre failure, outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _plies import CARBON, carbon_ply

from rve2d.laminate import Laminate, PlyCurve, laminate_constants
from rve2d.laminate.tensile import TensileSettings, _secant, tensile_test

# A softening transverse curve (MPa): linear to 50 MPa at 0.5 %, down to 10 MPa at 2 %
TRANSVERSE = PlyCurve(np.array([0.0, 0.005, 0.02]), np.array([0.0, 50.0, 10.0]))
# An elastic-plastic shear curve (MPa): G12 = 5 GPa up to 60 MPa, hardening slowly after
SHEAR = PlyCurve(np.array([0.0, 0.012, 0.08]), np.array([0.0, 60.0, 90.0]))


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
def test_elastic_response_matches_clt(sequence: str, direction: str, constant: str) -> None:
    laminate = Laminate.from_sequence(carbon_ply(), sequence, 0.125)
    settings = TensileSettings(direction=direction, max_strain=0.01, steps=4)
    result = tensile_test(laminate, settings, elastic_only=True)
    assert result.completed
    expected = laminate_constants(laminate)[constant]
    np.testing.assert_allclose(result.stress[1:] / result.strain[1:], expected, rtol=1e-8)


def test_transverse_laminate_reproduces_the_rve_curve() -> None:
    laminate = Laminate.from_sequence(carbon_ply(transverse_tension=TRANSVERSE), "[90_4]", 0.2)
    result = tensile_test(laminate, TensileSettings(max_strain=0.018, steps=36))
    expected = [TRANSVERSE.stress_at(strain) for strain in result.strain]
    np.testing.assert_allclose(result.stress, expected, rtol=1e-8, atol=1e-8)
    summary = result.summary()
    assert summary["peak_stress"] == pytest.approx(50.0)
    # flagged on the step that reaches the peak strain of the curve (steps of 0.05 %)
    assert summary["first_transverse_damage"]["strain"] == pytest.approx(0.005, abs=6e-4)


def test_fibre_failure_ends_a_unidirectional_test() -> None:
    ply = carbon_ply(longitudinal_tensile_strength=1500.0)
    laminate = Laminate.from_sequence(ply, "[0_4]", 0.125)
    result = tensile_test(laminate, TensileSettings(max_strain=0.02, steps=100))
    summary = result.summary()
    assert summary["first_fibre_failure"] is not None
    strain_at_failure = 1500.0 / CARBON["e1"]
    assert summary["first_fibre_failure"]["strain"] == pytest.approx(strain_at_failure, abs=2e-4)
    assert summary["peak_stress"] == pytest.approx(1500.0, rel=0.02)
    assert result.stress[-1] == 0.0 and result.message == "all load-carrying plies failed"


def test_cross_ply_knee_then_fibre_failure() -> None:
    ply = carbon_ply(transverse_tension=TRANSVERSE, longitudinal_tensile_strength=1500.0)
    laminate = Laminate.from_sequence(ply, "[0/90]s", 0.125)
    result = tensile_test(laminate, TensileSettings(max_strain=0.02, steps=200))
    events = [(e["event"], e["ply"]) for e in result.events]
    assert events[:2] == [("transverse peak passed", 2), ("transverse peak passed", 3)]
    assert ("fibre failure", 1) in events and ("fibre failure", 4) in events
    strain, stress = result.strain, result.stress
    peak = int(np.argmax(stress))
    assert strain[peak] == pytest.approx(1500.0 / CARBON["e1"], abs=2e-4)  # 0-ply fibres fail
    initial = stress[1] / strain[1]
    after_knee = (stress[peak] - stress[60]) / (strain[peak] - strain[60])
    assert after_knee < 0.95 * initial  # the 90 plies have softened (strain > 0.5 %)
    assert np.all(np.diff(stress[: peak + 1]) > 0)  # the 0 plies keep loading
    assert result.stress[-1] < 0.05 * stress[peak]  # load lost after fibre failure


def test_angle_ply_follows_the_shear_curve() -> None:
    ply = carbon_ply(shear=SHEAR, transverse_tension=TRANSVERSE)
    laminate = Laminate.from_sequence(ply, "[±45]2s", 0.125)
    elastic = laminate_constants(laminate)["ex"]
    result = tensile_test(laminate, TensileSettings(max_strain=0.03, steps=60))
    assert result.stress[1] / result.strain[1] == pytest.approx(elastic, rel=1e-6)
    assert result.stress[-1] < 0.6 * elastic * result.strain[-1]  # strongly nonlinear


def test_secant_unloading_memory() -> None:
    loaded, kappa = _secant(TRANSVERSE, 0.01, 0.0, 1e4)
    assert kappa == 0.01 and loaded == pytest.approx(TRANSVERSE.stress_at(0.01))
    unloaded, kept = _secant(TRANSVERSE, 0.005, kappa, 1e4)
    assert kept == 0.01 and unloaded == pytest.approx(0.5 * loaded)


def test_elongation_force_and_files(tmp_path: Path) -> None:
    laminate = Laminate.from_sequence(carbon_ply(), "[0/90]s", 0.125)
    settings = TensileSettings(max_strain=0.01, steps=5, gauge_length=150.0, width=25.0)
    result = tensile_test(laminate, settings, elastic_only=True)
    last = result.records[-1]
    assert last["elongation"] == pytest.approx(0.01 * 150.0)
    assert last["force"] == pytest.approx(last["stress"] * 0.5 * 25.0)
    curve, summary = result.write(tmp_path)
    header = curve.read_text().splitlines()[0]
    assert header.startswith("strain,stress,") and "elongation" in header and "force" in header
    assert '"initial_modulus"' in summary.read_text()
