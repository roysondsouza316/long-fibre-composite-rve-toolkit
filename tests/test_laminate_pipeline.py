"""The laminate pipeline end to end on small RVEs (needs gmsh; the tensile test with ply
curves also needs the TensorMesh engine's nonlinear extra, parity tests need Julia)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rve2d.config import config_from_dict
from rve2d.engines.julia import runner
from rve2d.laminate import Laminate, laminate_constants
from rve2d.laminate.pipeline import run_laminate_pipeline


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # ImportError, or OSError when system libraries are missing
        return False
    return True


pytestmark = pytest.mark.skipif(not _gmsh_available(), reason="gmsh not available")


def payload(tensile: bool, dimension: int = 2) -> dict[str, Any]:
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
            "longitudinal_tensile_strength": 250.0,  # fibre failure at about 1.5 % strain
            "tensile_test": {"enabled": tensile, "max_strain": 0.02, "steps": 40,
                             "gauge_length": 100.0, "width": 20.0},
        },  # fmt: skip
    }
    if tensile:
        data["nonlinear"] = {
            "enabled": True,
            "matrix": {"youngs_modulus": 3500.0, "poisson_ratio": 0.35, "yield_stress": 60.0,
                       "hardening_modulus": 300.0},
            "fibre": {"youngs_modulus": 70000.0, "poisson_ratio": 0.2},
            "interface": {"penalty_stiffness": 1.0e8, "normal_strength": 50.0,
                          "shear_strength": 75.0, "mode_i_toughness": 0.002,
                          "mode_ii_toughness": 0.006},
        }  # fmt: skip
        data["laminate"].update(
            {"transverse_max_strain": 0.01, "shear_max_strain": 0.02, "curve_steps": 8}
        )
    return data


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stiffness_pipeline_and_ply_reuse(tmp_path: Path) -> None:
    config = config_from_dict(payload(tensile=False))
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
    data = payload(tensile=False)
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
    config = config_from_dict(payload(tensile=False))
    tensormesh = run_laminate_pipeline(config, tmp_path / "tm", engine="tensormesh")
    julia = run_laminate_pipeline(config, tmp_path / "jl", engine="julia")
    assert julia.ply.source["engine"] == "julia"
    scale = np.abs(tensormesh.ply.stiffness).max()
    np.testing.assert_allclose(julia.ply.stiffness, tensormesh.ply.stiffness, atol=1e-10 * scale)


def test_tensile_pipeline_with_rve_curves(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffcohesive")
    config = config_from_dict(payload(tensile=True))
    result = run_laminate_pipeline(config, tmp_path / "run", engine="tensormesh")
    ply = result.ply
    assert ply.transverse_tension is not None and ply.shear is not None
    assert (tmp_path / "run" / "ply" / "shear" / "rve_layer.msh").exists()
    assert 0.0 < ply.transverse_tension.peak_stress < 200.0
    table = {row["laminate"]: row for row in rows(result.summary_csv)}
    assert float(table["[0]4"]["test_peak_stress"]) == pytest.approx(250.0, rel=0.03)
    e1 = ply.engineering_constants()["e1"]
    assert float(table["[0]4"]["first_fibre_failure_strain"]) == pytest.approx(250.0 / e1, abs=6e-4)
    assert float(table["[±45]s"]["test_peak_stress"]) < float(table["[0/90]s"]["test_peak_stress"])
    summary = json.loads(result.summary_json.read_text())
    assert summary["ply_curves"]["shear"]["peak_stress"] > 0.0
    curve = Path(table["[0/90]s"]["folder"]) / "tensile_test.csv"
    header = curve.read_text().splitlines()[0]
    assert "elongation" in header and "force" in header
