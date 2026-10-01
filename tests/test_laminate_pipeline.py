"""The laminate pipeline end to end on small RVEs (needs gmsh; the coupon tests with RVE ply
curves also need the TensorMesh engine's nonlinear extra, parity tests need Julia)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rve2d.cli import main
from rve2d.config import config_from_dict
from rve2d.engines.julia import runner
from rve2d.exceptions import ConfigError
from rve2d.laminate import Laminate, laminate_constants
from rve2d.laminate.pipeline import run_laminate_pipeline


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # ImportError, or OSError when system libraries are missing
        return False
    return True


pytestmark = pytest.mark.skipif(not _gmsh_available(), reason="gmsh not available")


def payload(curves: bool, dimension: int = 2) -> dict[str, Any]:
    synthetic: dict[str, Any] = {
        "domain_width": 0.02,
        "domain_height": 0.02,
        "fibre_radius": 0.0035,
        "target_volume_fraction": 0.2,
        "min_spacing": 0.0005,
        "edge_clearance": 0.0005,
        "random_seed": 4,
        "periodic_compatible": True,
    }
    if dimension == 3:
        synthetic["domain_depth"] = 0.003
    data: dict[str, Any] = {
        "mode": "synthetic",
        "dimension": dimension,
        "synthetic": synthetic,
        "mesh": {"element_size_min": 0.0008, "element_size_max": 0.0025, "verbosity": 0},
        "export": {"basename": "rve", "formats": ["msh"]},
        "solver": {
            "enabled": True,
            "kinematics": "generalized_plane_strain" if dimension == 2 else "solid",
            "boundary_condition": "periodic",
            "matrix_youngs_modulus": 3500.0,
            "matrix_poisson_ratio": 0.35,
            "fibre_youngs_modulus": 70000.0,
            "fibre_poisson_ratio": 0.2,
            "write_vtk": False,
        },
        "laminate": {
            "enabled": True,
            "ply_thickness": 0.125,
            "stacking_sequences": ["[0]4", "[0/90]s", "[±45]s"],
        },
    }
    if curves:
        data["nonlinear"] = {
            "enabled": True,
            "matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yield_stress": 60.0,
                       "hardening_modulus": 300.0, "compressive_yield_stress": 90.0,
                       "plastic_poisson_ratio": 0.3},
            "fibre": {"youngs_modulus": 70000.0, "poisson_ratio": 0.2},
            "interface": {"penalty_stiffness": 1.0e8, "normal_strength": 50.0,
                          "shear_strength": 75.0, "mode_i_toughness": 0.002,
                          "mode_ii_toughness": 0.006, "viscosity": 1.0e-4},
        }  # fmt: skip
        data["laminate"].update(
            {
                "damage_models": ["rve_curves", "max_stress", "hashin", "continuum_damage"],
                # fibre failure at about 1.5 % strain in tension and 1 % in compression
                "strengths": {"longitudinal_tension": 250.0, "longitudinal_compression": 170.0},
                "fracture_energies": {"fibre_tension": 20.0, "fibre_compression": 20.0,
                                      "matrix_tension": 0.5, "matrix_compression": 2.0},
                "characteristic_length": 0.1,
                "transverse_max_strain": 0.01,
                "shear_max_strain": 0.02,
                "curve_steps": 8,
                "coupon_tests": {
                    "tension": {"max_strain": 0.02, "steps": 40, "gauge_length": 100.0,
                                "width": 20.0},
                    "compression": {"max_strain": -0.015, "steps": 30},
                },
            }
        )  # fmt: skip
    return data


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stiffness_pipeline_and_ply_reuse(tmp_path: Path) -> None:
    config = config_from_dict(payload(curves=False))
    result = run_laminate_pipeline(config, tmp_path / "run")
    ply = result.ply
    constants = ply.engineering_constants()
    summary = json.loads((tmp_path / "run/ply/elastic/homogenization_summary.json").read_text())
    vf = summary["fibre_volume_fraction"]
    # the fibres (RVE z) are ply axis 1: E1 is the rule of mixtures (plus a Poisson effect)
    assert constants["e1"] == pytest.approx(vf * 70000.0 + (1.0 - vf) * 3500.0, rel=0.01)
    assert constants["e2"] < 0.5 * constants["e1"]
    assert ply.source["kinematics"] == "generalized_plane_strain"
    table = {row["laminate"]: row for row in rows(result.summary_csv)}
    assert float(table["[0]4"]["ex"]) == pytest.approx(constants["e1"], rel=1e-10)
    assert float(table["[0]4"]["ey"]) == pytest.approx(constants["e2"], rel=1e-10)
    assert float(table["[0/90]s"]["ex"]) == pytest.approx(float(table["[0/90]s"]["ey"]))
    for row in table.values():
        folder = Path(row["folder"])
        assert (folder / "abd.csv").exists() and (folder / "constants.json").exists()
    # change the stacking sequences and reuse the saved ply properties (no RVE solve)
    data = payload(curves=False)
    data["laminate"]["stacking_sequences"] = ["[0/±60]s"]
    again = run_laminate_pipeline(
        config_from_dict(data), tmp_path / "again", ply_path=result.ply_path
    )
    assert not (tmp_path / "again" / "rve").exists()
    expected = laminate_constants(Laminate.from_sequence(ply, "[0/±60]s", 0.125))
    assert float(rows(again.summary_csv)[0]["ex"]) == pytest.approx(expected["ex"])


@pytest.mark.skipif(
    not (shutil.which("julia") and runner.is_instantiated()), reason="Julia engine not set up"
)
def test_ply_properties_agree_between_engines(tmp_path: Path) -> None:
    config = config_from_dict(payload(curves=False))
    tensormesh = run_laminate_pipeline(config, tmp_path / "tm", engine="tensormesh")
    julia = run_laminate_pipeline(config, tmp_path / "jl", engine="julia")
    assert julia.ply.source["engine"] == "julia"
    scale = np.abs(tensormesh.ply.stiffness).max()
    np.testing.assert_allclose(julia.ply.stiffness, tensormesh.ply.stiffness, atol=1e-10 * scale)


def test_coupon_pipeline_with_every_damage_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    config = config_from_dict(payload(curves=True))
    result = run_laminate_pipeline(config, tmp_path / "run", engine="tensormesh")
    ply = result.ply
    assert ply.transverse_tension is not None and ply.shear is not None
    assert ply.transverse_compression is not None
    assert (tmp_path / "run" / "ply" / "shear" / "rve_layer.msh").exists()
    assert 0.0 < ply.transverse_tension.peak_stress < 200.0
    # the pressure-dependent matrix: compression is stronger than tension
    assert ply.transverse_compression.peak_stress > ply.transverse_tension.peak_stress
    strengths = ply.resolved_strengths()
    assert strengths.yt == ply.transverse_tension.peak_stress
    assert strengths.yc == ply.transverse_compression.peak_stress
    assert strengths.s12 == ply.shear.peak_stress
    assert ply.strength_sources() == {
        "xt": "input", "xc": "input", "yt": "rve", "yc": "rve", "s12": "rve", "s23": None
    }  # fmt: skip

    table = {(r["laminate"], r["model"], r["test"]): r for r in rows(result.coupon_csv)}
    assert len(table) == 3 * 4 * 2
    e1 = ply.engineering_constants()["e1"]
    for model in ("rve_curves", "max_stress", "hashin", "continuum_damage"):
        tension, compression = table["[0]4", model, "tension"], table["[0]4", model, "compression"]
        # [0]4: fibre-dominated, linear up to Xt / Xc in every model
        assert float(tension["peak_stress"]) == pytest.approx(250.0, rel=0.03)
        assert float(compression["peak_stress"]) == pytest.approx(-170.0, rel=0.03)
        assert float(tension["first_fibre_failure_strain"]) == pytest.approx(250 / e1, abs=6e-4)
        # cross-ply: matrix cracking in the 90 plies first, then fibre failure in the 0 plies
        cross = table["[0/90]s", model, "tension"]
        assert float(cross["first_ply_failure_strain"]) < float(cross["first_fibre_failure_strain"])
        angle = table["[±45]s", model, "tension"]
        assert abs(float(angle["peak_stress"])) < abs(float(cross["peak_stress"]))
        csv_path = result.output_directory / "laminates" / "0-4" / model / "tension.csv"
        assert Path(tension["csv"]) == csv_path and csv_path.exists()
    # the strength-based models see first ply failure at the transverse strength of the 90 ply
    for model in ("max_stress", "hashin"):
        assert "matrix tension" in table["[0/90]s", model, "tension"]["first_ply_failure"]

    summary = json.loads(result.summary_json.read_text())
    assert summary["ply_curves"]["shear"]["peak_stress"] > 0.0
    assert summary["damage_models"] == ["rve_curves", "max_stress", "hashin", "continuum_damage"]
    assert set(summary["coupon_tests"]) == {"tension", "compression"}
    assert all(Path(plot).exists() for plot in summary["plots"])
    curve = Path(table["[0/90]s", "hashin", "tension"]["csv"])
    header = curve.read_text().splitlines()[0]
    assert "elongation" in header and "force" in header

    # change the damage models and tests, reuse the saved ply (curves and strengths)
    data = payload(curves=True)
    data["laminate"].update(
        {"damage_models": ["max_stress"], "coupon_tests": {"shear": {"direction": "xy",
                                                                     "max_strain": 0.02}}}
    )  # fmt: skip
    again = run_laminate_pipeline(
        config_from_dict(data), tmp_path / "again", ply_path=result.ply_path
    )
    assert again.ply.resolved_strengths() == strengths
    shear = rows(again.coupon_csv)
    assert {row["test"] for row in shear} == {"shear"} and len(shear) == 3


def test_cli_prints_the_coupon_tests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Strength-based models with given strengths need no ply curves (no nonlinear solve)."""
    data = payload(curves=False)
    data["laminate"].update(
        {
            "damage_models": ["max_stress", "hashin"],
            "ply_curves": False,
            "strengths": {"longitudinal_tension": 250.0, "longitudinal_compression": 170.0,
                          "transverse_tension": 20.0, "transverse_compression": 60.0,
                          "in_plane_shear": 30.0},
            "coupon_tests": {"tension": {"max_strain": 0.02, "steps": 40}},
        }
    )  # fmt: skip
    config_path = tmp_path / "laminate.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["laminate", str(config_path), "--output-dir", str(tmp_path / "out")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["damage_models"] == ["max_stress", "hashin"]
    assert printed["ply_strength_sources"]["yt"] == "input" and printed["warnings"] == []
    tests = printed["laminates"]["[0/90]s"]["coupon_tests"]
    assert set(tests) == {"max_stress", "hashin"}
    assert tests["max_stress"]["tension"]["first_ply_failure"] == "matrix tension failure"
    unidirectional = printed["laminates"]["[0]4"]["coupon_tests"]["hashin"]["tension"]
    assert unidirectional["peak_stress"] == pytest.approx(250.0, rel=1e-3)
    assert not (tmp_path / "out" / "ply" / "transverse_tension").exists()


def test_laminate_config_validation() -> None:
    data = payload(curves=True)
    data["laminate"]["damage_models"] = ["max_stress", "puck"]
    with pytest.raises(ConfigError, match="damage_models"):
        config_from_dict(data)
    data = payload(curves=True)
    del data["laminate"]["strengths"]["longitudinal_compression"]
    with pytest.raises(ConfigError, match="longitudinal_compression"):
        config_from_dict(data)
    data = payload(curves=True)
    del data["laminate"]["fracture_energies"]
    with pytest.raises(ConfigError, match="fracture_energies"):
        config_from_dict(data)
    data = payload(curves=True)
    data["laminate"]["coupon_tests"]["bad name"] = {"max_strain": 0.01}
    with pytest.raises(ConfigError, match="test names"):
        config_from_dict(data)
    data = payload(curves=True)
    del data["nonlinear"]
    with pytest.raises(ConfigError, match="ply curves"):
        config_from_dict(data)
    # given strengths remove the curves the strength models need
    data["laminate"]["damage_models"] = ["max_stress", "hashin"]
    data["laminate"]["strengths"].update(
        {"transverse_tension": 35.0, "transverse_compression": 114.0, "in_plane_shear": 72.0}
    )
    config = config_from_dict(data)
    assert config.laminate.needed_curves() == ()
    data["laminate"]["damage_models"] = ["rve_curves"]
    with pytest.raises(ConfigError, match="ply curves"):
        config_from_dict(data)
