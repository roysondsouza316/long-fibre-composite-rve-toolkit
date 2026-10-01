"""Nonlinear RVE solve with the Julia engine (``FerriteRVE`` on Ferrite.jl + DiffCohesive.jl).

``nonlinear.engine: julia`` exports the mesh arrays and a TOML input, runs the engine (see
``rve2d.engines.julia.runner``) and reads back the same response CSV the TensorMesh engine
writes; the summary JSON has the same layout. PyTorch is not needed.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rve2d.config import PhaseMaterialConfig, RVEConfig
from rve2d.engines.common.records import (
    NonlinearResult,
    load_path_from_config,
    read_response_csv,
    response_summary,
    write_summary_json,
)
from rve2d.engines.julia.runner import export_mesh_arrays, run_task, write_toml
from rve2d.exceptions import ConfigError

RESULT_NAME = "julia_nonlinear_result.toml"


def run_nonlinear(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
) -> NonlinearResult:
    nl = config.nonlinear
    if not nl.enabled:
        raise ConfigError("Nonlinear solve requested but nonlinear.enabled is false in the config.")
    started = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = write_julia_input(config, Path(mesh_path), out_dir)
    (out_dir / RESULT_NAME).unlink(missing_ok=True)
    log_path = out_dir / "nonlinear_log.txt"
    run_task(input_path, log_path)

    result = tomllib.loads((out_dir / RESULT_NAME).read_text(encoding="utf-8"))
    response_path = out_dir / "nonlinear_response.csv"
    records = read_response_csv(response_path)
    response = response_summary(records, nl.load)
    interface = nl.interface if nl.interface is not None and nl.interface.enabled else None
    field_files = [Path(path) for path in result["field_files"]]
    summary: dict[str, Any] = {
        "engine": "julia",
        "dimension": config.dimension,
        "kinematics": nl.resolved_kinematics(config.dimension),
        "boundary_condition": nl.boundary_condition,
        "load": asdict(nl.load),
        "materials": {
            "matrix": None if nl.matrix is None else asdict(nl.matrix),
            "fibre": None if nl.fibre is None else asdict(nl.fibre),
        },
        "interface": None if interface is None else asdict(interface),
        "mesh": {
            "nodes": result["nodes"],
            "bulk_cells": result["bulk_cells"],
            "cohesive_elements": result["cohesive_elements"],
            "unknowns": result["unknowns"],
        },
        "device": "cpu",
        "linear_solver": result["linear_solver"],
        "julia_version": result["julia_version"],
        "completed": result["completed"],
        "reached_load_time": result["reached_time"],
        "steps": result["steps"],
        "newton_iterations": result["newton_iterations"],
        "increment_cuts": result["increment_cuts"],
        "messages": result["messages"],
        **response,
        "julia_runtime_seconds": result["runtime_seconds"],
        "runtime_seconds": round(time.time() - started, 2),
        "response_csv": str(response_path),
        "field_files": [str(path) for path in field_files],
        "log_path": str(log_path),
    }
    summary_path = write_summary_json(out_dir / "nonlinear_summary.json", summary)
    return NonlinearResult(
        summary_path=summary_path,
        response_path=response_path,
        field_files=field_files,
        completed=bool(result["completed"]),
        peak_stress=response["peak_stress"],
        strain_at_peak=response["strain_at_peak"],
        records=records,
    )


def write_julia_input(config: RVEConfig, mesh_path: Path, out_dir: Path) -> Path:
    """Write the mesh arrays and the TOML input of the Julia solver; return the TOML path."""
    nl = config.nonlinear
    assert nl.matrix is not None and nl.fibre is not None
    arrays = export_mesh_arrays(mesh_path, out_dir, config.dimension, "nonlinear")
    kinematics = nl.resolved_kinematics(config.dimension)
    load = load_path_from_config(kinematics, nl.load)
    tables: dict[str, dict[str, Any]] = {
        "matrix": _phase_table(nl.matrix),
        "fibre": _phase_table(nl.fibre),
        "load": {
            "prescribed_components": [k + 1 for k in load.prescribed_strain],
            "prescribed_values": list(load.prescribed_strain.values()),
            "fixed_zero": [k + 1 for k in load.fixed_zero],
            "stress_components": [k + 1 for k in load.stress_controlled],
            "stress_values": list(load.stress_controlled.values()),
            "unload": load.unload,
        },
        "newton": {
            "max_iterations": nl.newton_max_iterations,
            "tolerance": nl.newton_tolerance,
            "steps": nl.load.steps,
            "max_cuts": nl.max_step_cuts,
            "growth": 1.5,
            "line_search": True,
        },
    }
    interface = nl.interface
    if interface is not None and interface.enabled:
        tables["interface"] = {
            key: value
            for key, value in asdict(interface).items()
            if key != "enabled" and value is not None
        }
    top: dict[str, Any] = {
        "task": "nonlinear",
        "dimension": config.dimension,
        "kinematics": kinematics,
        "boundary_condition": nl.boundary_condition,
        "nodes": str(arrays["nodes"]),
        "cells": str(arrays["cells"]),
        "cell_tags": str(arrays["cell_tags"]),
        "output_dir": str(out_dir.resolve()),
        "matrix_phase_id": nl.matrix_phase_id,
        "fibre_phase_id": nl.fibre_phase_id,
        "stress_scale": nl.matrix.youngs_modulus,
        "output_every": nl.output_every,
    }
    return write_toml(out_dir / "julia_nonlinear_input.toml", top, tables)


def _phase_table(phase: PhaseMaterialConfig) -> dict[str, Any]:
    return {key: value for key, value in asdict(phase).items() if value is not None}
