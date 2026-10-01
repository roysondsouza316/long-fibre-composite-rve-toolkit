from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rve2d.engines.common.homogenization import HomogenizationResult
from rve2d.exceptions import SolverError
from rve2d.study import run_batch_study, unique_case_names
from rve2d.validation.checks import QualityReport
from rve2d.workflow import BuildResult


@dataclass(frozen=True)
class _DummyConfig:
    name: str


def _fake_build_and_solve_rve(
    config: _DummyConfig,
    output_dir: str | None = None,
    basename: str | None = None,
    engine: str | None = None,
) -> BuildResult:
    assert output_dir is not None and basename is not None
    if config.name == "broken":
        raise SolverError("singular stiffness matrix")
    case_dir = Path(output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "homogenization_summary.json"
    response_files = [
        case_dir / "homogenized_stiffness.csv",
        case_dir / "engineering_constants.csv",
        case_dir / "stress_strain_response.csv",
        case_dir / "traction_response.csv",
    ]
    for path in response_files:
        path.write_text("header\n", encoding="utf-8")
    summary_payload = {
        "fibre_volume_fraction": 0.33,
        "boundary_condition": "periodic",
        "kinematics": "plane_stress",
        "material_angles_deg": {"matrix": 10.0, "fibre": 25.0},
        "response_files": [str(path) for path in response_files],
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    vtk_path = case_dir / "homogenization.vtu"
    vtk_path.write_text("<VTKFile/>", encoding="utf-8")
    return BuildResult(
        output_directory=case_dir,
        mesh_files=[],
        metadata_files=[summary_path],
        quality_report=QualityReport(
            valid=True, minimum_spacing=0.0, clipped_fibres=0, disconnected_artifacts=0
        ),
        geometry_metadata={},
        homogenization_result=HomogenizationResult(
            engine=engine or "python",
            kinematics="plane_stress",
            summary_path=summary_path,
            log_path=None,
            vtk_path=vtk_path,
            homogenized_stiffness=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            engineering_constants={"ex": 1.0, "ey": 2.0, "gxy": 3.0, "nuxy": 0.1, "nuyx": 0.2},
            response_files=response_files,
        ),
    )


@pytest.fixture
def fake_pipeline(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("rve2d.study.load_config", lambda path: _DummyConfig(Path(path).stem))
    monkeypatch.setattr("rve2d.study.build_and_solve_rve", _fake_build_and_solve_rve)


def _rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.usefixtures("fake_pipeline")
def test_batch_study_writes_summary_csv(tmp_path: Path) -> None:
    cases, summary_csv = run_batch_study(
        [tmp_path / "case_a.yaml", tmp_path / "case_b.yaml"], tmp_path / "out", engine="julia"
    )
    assert [case.status for case in cases] == ["ok", "ok"]
    rows = _rows(summary_csv)
    assert len(rows) == 2
    assert rows[0]["engine"] == "julia"
    assert rows[0]["boundary_condition"] == "periodic"
    assert rows[0]["matrix_angle_deg"] == "10.0"
    assert rows[0]["fibre_angle_deg"] == "25.0"
    assert rows[0]["ex"] == "1.0"
    assert rows[0]["stress_strain_csv"].endswith("stress_strain_response.csv")
    assert rows[0]["traction_csv"].endswith("traction_response.csv")


@pytest.mark.usefixtures("fake_pipeline")
def test_failing_case_is_recorded_and_the_study_goes_on(tmp_path: Path) -> None:
    configs = [tmp_path / "broken.yaml", tmp_path / "good.yaml"]
    cases, summary_csv = run_batch_study(configs, tmp_path / "out")
    assert [case.status for case in cases] == ["failed", "ok"]
    rows = _rows(summary_csv)
    assert rows[0]["status"] == "failed" and "singular" in rows[0]["error"]
    assert rows[1]["status"] == "ok" and rows[1]["ex"] == "1.0"
    with pytest.raises(SolverError):
        run_batch_study(configs, tmp_path / "out_fail_fast", fail_fast=True)
    assert len(_rows(tmp_path / "out_fail_fast" / "study_summary.csv")) == 1


@pytest.mark.usefixtures("fake_pipeline")
def test_cases_with_the_same_file_name_get_separate_folders(tmp_path: Path) -> None:
    configs = [tmp_path / "2d" / "periodic_solve.yaml", tmp_path / "3d" / "periodic_solve.yaml"]
    cases, _ = run_batch_study(configs, tmp_path / "out")
    assert [case.case_name for case in cases] == ["2d_periodic_solve", "3d_periodic_solve"]
    assert cases[0].output_directory != cases[1].output_directory


def test_unique_case_names() -> None:
    paths = [Path("a/x.yaml"), Path("b/x.yaml"), Path("a/y.yaml"), Path("a/y.yaml")]
    assert unique_case_names(paths) == ["a_x", "b_x", "y", "y_2"]
