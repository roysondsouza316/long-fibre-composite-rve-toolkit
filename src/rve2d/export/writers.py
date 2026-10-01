from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import meshio

from rve2d.exceptions import RVEError
from rve2d.mesh_io import read_mesh
from rve2d.models import GeometryModel
from rve2d.validation.checks import QualityReport


def convert_mesh_formats(mesh_path: str | Path, formats: list[str], basename: str) -> list[Path]:
    source_path = Path(mesh_path)
    written_paths = [source_path]
    requested_formats = {fmt.lower().lstrip(".") for fmt in formats}
    if requested_formats == {"msh"}:
        return written_paths

    mesh = read_mesh(source_path)
    for output_format in sorted(requested_formats - {"msh"}):
        destination = source_path.with_name(f"{basename}.{output_format}")
        try:
            meshio.write(destination, mesh)
        except ModuleNotFoundError as exc:
            raise RVEError(
                f"Writing '.{output_format}' requires an additional backend dependency: {exc.name}."
            ) from exc
        written_paths.append(destination)
    return written_paths


def write_metadata_bundle(
    output_dir: str | Path,
    geometry: GeometryModel,
    quality_report: QualityReport,
    phase_tags: dict[str, int],
    boundary_tags: dict[str, int],
    write_quality_json: bool = True,
    write_phase_json: bool = True,
    write_periodic_json: bool = True,
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    geometry_summary = {
        "domain": asdict(geometry.domain),
        "fibre_count": geometry.fibre_count,
        "fibre_volume_fraction": geometry.fibre_volume_fraction,
        "metadata": geometry.metadata,
    }
    summary_path = directory / "geometry_summary.json"
    summary_path.write_text(json.dumps(geometry_summary, indent=2), encoding="utf-8")
    written_paths.append(summary_path)

    if write_phase_json:
        phase_path = directory / "phase_tags.json"
        phase_payload = {"phase_tags": phase_tags, "boundary_tags": boundary_tags}
        phase_path.write_text(json.dumps(phase_payload, indent=2), encoding="utf-8")
        written_paths.append(phase_path)

    if write_quality_json:
        quality_path = directory / "quality_report.json"
        quality_path.write_text(json.dumps(_jsonable(quality_report), indent=2), encoding="utf-8")
        written_paths.append(quality_path)

    if geometry.periodic_pairs and write_periodic_json:
        periodic_path = directory / "periodic_pairs.json"
        periodic_path.write_text(
            json.dumps({"periodic_pairs": _jsonable(geometry.periodic_pairs)}, indent=2),
            encoding="utf-8",
        )
        written_paths.append(periodic_path)

    return written_paths


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
