from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pytest import MonkeyPatch

from rve2d.ferrite_bridge import FerriteSolveResult
from rve2d.study import run_batch_study
from rve2d.validation.checks import QualityReport
from rve2d.workflow import BuildResult


@dataclass(frozen=True)
class _DummyConfig:
    name: str


def test_batch_study_writes_summary_csv(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_config(path: str | Path) -> _DummyConfig:
        return _DummyConfig(name=Path(path).stem)

    def fake_build_and_solve_rve(
        config: _DummyConfig,
        output_dir: str | None = None,
        basename: str | None = None,
    ) -> BuildResult:
        assert output_dir is not None
        assert basename is not None
        case_dir = Path(output_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "ferrite_homogenization_summary.json"
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
        vtk_path = case_dir / "ferrite_homogenization.vtu"
        vtk_path.write_text("<VTKFile/>", encoding="utf-8")
        return BuildResult(
            output_directory=case_dir,
            mesh_files=[],
            metadata_files=[summary_path],
            quality_report=QualityReport(
                valid=True,
                minimum_spacing=0.0,
                clipped_fibres=0,
                disconnected_artifacts=0,
            ),
            geometry_metadata={},
            ferrite_result=FerriteSolveResult(
                summary_path=summary_path,
                stdout_path=case_dir / "ferrite_homogenization_stdout.txt",
                vtk_path=vtk_path,
                homogenized_stiffness=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                engineering_constants={
                    "ex": 1.0,
                    "ey": 2.0,
                    "gxy": 3.0,
                    "nuxy": 0.1,
                    "nuyx": 0.2,
                },
                response_files=response_files,
            ),
        )

    monkeypatch.setattr("rve2d.study.load_config", fake_load_config)
    monkeypatch.setattr("rve2d.study.build_and_solve_rve", fake_build_and_solve_rve)

    case_results, summary_csv = run_batch_study(
        [tmp_path / "case_a.yaml", tmp_path / "case_b.yaml"],
        tmp_path / "batch_outputs",
    )

    assert len(case_results) == 2
    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["boundary_condition"] == "periodic"
    assert rows[0]["matrix_angle_deg"] == "10.0"
    assert rows[0]["fibre_angle_deg"] == "25.0"
    assert rows[0]["stress_strain_csv"].endswith("stress_strain_response.csv")
    assert rows[0]["traction_csv"].endswith("traction_response.csv")
