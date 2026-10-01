"""Batch studies: run build-and-solve for several configs and collect one summary CSV.

Each case gets its own output folder (named after the config file, with parent folders
added when two configs share a file name). A failing case is recorded with its error and the
study goes on (unless ``fail_fast``); ``study_summary.csv`` is written in every case.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rve2d.config import load_config
from rve2d.workflow import BuildResult, build_and_solve_rve

ENGINEERING_ORDER = (
    "ex", "ey", "ez", "gyz", "gxz", "gxy", "nuxy", "nuyx", "nuxz", "nuzx", "nuyz", "nuzy",
)  # fmt: skip
FIELDS = (
    "case_name",
    "status",
    "error",
    "config_path",
    "output_directory",
    "engine",
    "fibre_volume_fraction",
    *ENGINEERING_ORDER,
    *(f"{name}_plane_strain" for name in ("ex", "ey", "gxy", "nuxy", "nuyx")),
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
    "nonlinear_completed",
    "nonlinear_peak_stress",
    "nonlinear_strain_at_peak",
    "nonlinear_summary_path",
)
RESPONSE_COLUMNS = {
    "homogenized_stiffness.csv": "stiffness_csv",
    "engineering_constants.csv": "engineering_csv",
    "stress_strain_response.csv": "stress_strain_csv",
    "traction_response.csv": "traction_csv",
}


@dataclass(frozen=True)
class StudyCaseResult:
    case_name: str
    config_path: Path
    output_directory: Path
    status: str  # "ok" or "failed"
    result: BuildResult | None = None
    error: str | None = None


def run_batch_study(
    config_paths: list[str | Path] | list[Path],
    output_dir: str | Path,
    engine: str | None = None,
    fail_fast: bool = False,
) -> tuple[list[StudyCaseResult], Path]:
    root_dir = Path(output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in config_paths]
    cases: list[StudyCaseResult] = []
    summary_csv = root_dir / "study_summary.csv"
    for path, name in zip(paths, unique_case_names(paths), strict=True):
        case_dir = root_dir / name
        try:
            config = load_config(path)
            result = build_and_solve_rve(
                config, output_dir=str(case_dir), basename=name, engine=engine
            )
        except Exception as exc:  # one failing case must not lose the others
            cases.append(
                StudyCaseResult(
                    name, path.resolve(), case_dir, "failed", error=f"{type(exc).__name__}: {exc}"
                )
            )
            if fail_fast:
                _write_study_summary(summary_csv, cases)
                raise
            continue
        cases.append(StudyCaseResult(name, path.resolve(), case_dir, "ok", result=result))
    _write_study_summary(summary_csv, cases)
    return cases, summary_csv


def unique_case_names(paths: list[Path]) -> list[str]:
    """Config file stems, extended with parent folder names where different files share a
    stem; repeated entries of the same file get ``_2``, ``_3``, ... appended."""
    resolved = [path.resolve() for path in paths]
    names = []
    for path in resolved:
        others = {other for other in resolved if other.stem == path.stem and other != path}
        depth = 1
        while depth < len(path.parts) and any(
            other.parts[-depth:] == path.parts[-depth:] for other in others
        ):
            depth += 1
        names.append("_".join([*path.parts[-depth:-1], path.stem]))
    seen: Counter[str] = Counter()
    unique = []
    for name in names:
        seen[name] += 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return unique


def _write_study_summary(summary_csv: Path, cases: list[StudyCaseResult]) -> None:
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(_row(case))


def _row(case: StudyCaseResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_name": case.case_name,
        "status": case.status,
        "error": case.error or "",
        "config_path": str(case.config_path),
        "output_directory": str(case.output_directory),
    }
    result = case.result
    if result is None:
        return row
    homogenization = result.homogenization_result
    if homogenization is not None:
        payload = json.loads(homogenization.summary_path.read_text(encoding="utf-8"))
        angles = payload.get("material_angles_deg", {})
        row.update(homogenization.engineering_constants)
        row.update(
            {
                "engine": homogenization.engine,
                "fibre_volume_fraction": payload.get("fibre_volume_fraction", ""),
                "boundary_condition": payload.get("boundary_condition", ""),
                "kinematics": payload.get("kinematics", ""),
                "matrix_angle_deg": angles.get("matrix", ""),
                "fibre_angle_deg": angles.get("fibre", ""),
                "vtk_path": str(homogenization.vtk_path) if homogenization.vtk_path else "",
                "summary_path": str(homogenization.summary_path),
            }
        )
        for path in homogenization.response_files:
            column = RESPONSE_COLUMNS.get(Path(path).name)
            if column is not None:
                row[column] = str(path)
    nonlinear = result.nonlinear_result
    if nonlinear is not None:
        row.update(
            {
                "nonlinear_completed": nonlinear.completed,
                "nonlinear_peak_stress": nonlinear.peak_stress,
                "nonlinear_strain_at_peak": nonlinear.strain_at_peak,
                "nonlinear_summary_path": str(nonlinear.summary_path),
            }
        )
    return row
