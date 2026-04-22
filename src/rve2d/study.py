from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rve2d.config import load_config
from rve2d.workflow import BuildResult, build_and_solve_rve


@dataclass(frozen=True)
class StudyCaseResult:
    case_name: str
    config_path: Path
    result: BuildResult


def run_batch_study(
    config_paths: list[str | Path],
    output_dir: str | Path,
) -> tuple[list[StudyCaseResult], Path]:
    root_dir = Path(output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    case_results: list[StudyCaseResult] = []

    for config_path in config_paths:
        path = Path(config_path)
        config = load_config(path)
        case_name = path.stem
        case_output_dir = root_dir / case_name
        result = build_and_solve_rve(config, output_dir=str(case_output_dir), basename=case_name)
        case_results.append(
            StudyCaseResult(
                case_name=case_name,
                config_path=path.resolve(),
                result=result,
            )
        )

    summary_csv = root_dir / "study_summary.csv"
    _write_study_summary(summary_csv, case_results)
    return case_results, summary_csv


def _write_study_summary(summary_csv: Path, case_results: list[StudyCaseResult]) -> None:
    fieldnames = [
        "case_name",
        "config_path",
        "output_directory",
        "fibre_volume_fraction",
        "ex",
        "ey",
        "ez",
        "gyz",
        "gxz",
        "gxy",
        "nuxy",
        "nuyx",
        "nuxz",
        "nuzx",
        "nuyz",
        "nuzy",
        "boundary_condition",
        "kinematics",
        "matrix_angle_deg",
        "fibre_angle_deg",
        "vtk_path",
        "summary_path",
        "stiffness_csv",
        "engineering_csv",
        "stress_strain_csv",
        "traction_csv",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in case_results:
            ferrite = case.result.ferrite_result
            if ferrite is None:
                continue
            summary = ferrite.summary_path
            summary_payload = summary.read_text(encoding="utf-8")
            row = {
                "case_name": case.case_name,
                "config_path": str(case.config_path),
                "output_directory": str(case.result.output_directory),
                "fibre_volume_fraction": case.result.geometry_metadata.get(
                    "target_volume_fraction",
                    "",
                ),
                "ex": ferrite.engineering_constants.get("ex", ""),
                "ey": ferrite.engineering_constants.get("ey", ""),
                "ez": ferrite.engineering_constants.get("ez", ""),
                "gyz": ferrite.engineering_constants.get("gyz", ""),
                "gxz": ferrite.engineering_constants.get("gxz", ""),
                "gxy": ferrite.engineering_constants.get("gxy", ""),
                "nuxy": ferrite.engineering_constants.get("nuxy", ""),
                "nuyx": ferrite.engineering_constants.get("nuyx", ""),
                "nuxz": ferrite.engineering_constants.get("nuxz", ""),
                "nuzx": ferrite.engineering_constants.get("nuzx", ""),
                "nuyz": ferrite.engineering_constants.get("nuyz", ""),
                "nuzy": ferrite.engineering_constants.get("nuzy", ""),
                "boundary_condition": "unknown",
                "kinematics": "unknown",
                "matrix_angle_deg": "",
                "fibre_angle_deg": "",
                "vtk_path": str(ferrite.vtk_path) if ferrite.vtk_path else "",
                "summary_path": str(summary),
                "stiffness_csv": "",
                "engineering_csv": "",
                "stress_strain_csv": "",
                "traction_csv": "",
            }

            payload = json.loads(summary_payload)
            row["fibre_volume_fraction"] = payload.get("fibre_volume_fraction", "")
            row["boundary_condition"] = payload.get("boundary_condition", "")
            row["kinematics"] = payload.get("kinematics", "")
            angles = payload.get("material_angles_deg", {})
            row["matrix_angle_deg"] = angles.get("matrix", "")
            row["fibre_angle_deg"] = angles.get("fibre", "")
            response_files = [Path(path) for path in payload.get("response_files", [])]
            for response_path in response_files:
                if response_path.name == "homogenized_stiffness.csv":
                    row["stiffness_csv"] = str(response_path)
                elif response_path.name == "engineering_constants.csv":
                    row["engineering_csv"] = str(response_path)
                elif response_path.name == "stress_strain_response.csv":
                    row["stress_strain_csv"] = str(response_path)
                elif response_path.name == "traction_response.csv":
                    row["traction_csv"] = str(response_path)
            writer.writerow(row)
