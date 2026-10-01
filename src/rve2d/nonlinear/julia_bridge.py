"""Nonlinear RVE solve with the Julia backend: Ferrite.jl + DiffCohesive.jl.

``nonlinear.backend: julia`` exports the mesh arrays and a TOML input, runs
``julia/ferrite_nonlinear_rve.jl`` in the ``julia/NonlinearRVE`` environment (instantiated on
first use) and reads back the same response CSV the Python backend writes; the summary JSON
has the same layout. PyTorch is not needed.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from rve2d.config import PhaseMaterialConfig, RVEConfig
from rve2d.exceptions import ConfigError, SolverError
from rve2d.nonlinear.records import (
    NonlinearResult,
    load_path_from_config,
    read_response_csv,
    response_summary,
    write_summary_json,
)

JULIA_ROOT = Path(__file__).resolve().parents[3] / "julia"
JULIA_PROJECT = JULIA_ROOT / "NonlinearRVE"
JULIA_SCRIPT = JULIA_ROOT / "ferrite_nonlinear_rve.jl"


def run_julia_nonlinear(
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
    julia = shutil.which("julia")
    if julia is None:
        raise SolverError("nonlinear.backend: julia needs the `julia` executable on PATH.")
    if not JULIA_SCRIPT.exists():
        raise SolverError(f"Julia solver not found at {JULIA_SCRIPT} (needs a source checkout).")

    input_path = write_julia_input(config, Path(mesh_path), out_dir)
    command = [julia, f"--project={JULIA_PROJECT}", str(JULIA_SCRIPT), str(input_path)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path = out_dir / "julia_nonlinear_stdout.txt"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SolverError(
            "Julia nonlinear solve failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout (tail):\n{completed.stdout[-4000:]}\n"
            f"stderr (tail):\n{completed.stderr[-4000:]}"
        )

    result = tomllib.loads((out_dir / "julia_nonlinear_result.toml").read_text(encoding="utf-8"))
    response_path = out_dir / "nonlinear_response.csv"
    records = read_response_csv(response_path)
    response = response_summary(records, nl.load)
    interface = nl.interface if nl.interface is not None and nl.interface.enabled else None
    field_files = [Path(path) for path in result["field_files"]]
    summary: dict[str, Any] = {
        "backend": "julia",
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
        "julia_log": str(log_path),
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
    arrays = _export_mesh_arrays(mesh_path, out_dir, config.dimension)
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
    lines = [f"{key} = {_toml_value(value)}" for key, value in top.items()]
    for name, table in tables.items():
        lines.append(f"\n[{name}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in table.items())
    path = out_dir / "julia_nonlinear_input.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _phase_table(phase: PhaseMaterialConfig) -> dict[str, Any]:
    return {key: value for key, value in asdict(phase).items() if value is not None}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise SolverError("Cannot pass NaN to the Julia solver.")
        return repr(value) if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise SolverError(f"Cannot write {value!r} to the Julia input file.")


def _export_mesh_arrays(mesh_path: Path, out_dir: Path, dimension: int) -> dict[str, Path]:
    mesh = meshio.read(mesh_path)
    cell_type = "tetra" if dimension == 3 else "triangle"
    blocks = [block.data for block in mesh.cells if block.type == cell_type]
    if not blocks:
        raise SolverError(
            f"The {dimension}D nonlinear solver needs linear {cell_type} cells; found "
            + ", ".join(sorted({block.type for block in mesh.cells}))
            + "."
        )
    physical = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    if physical is None:
        raise SolverError("Mesh is missing gmsh physical tags for the bulk cells.")
    paths = {
        "nodes": out_dir / "nonlinear_nodes.csv",
        "cells": out_dir / "nonlinear_cells.csv",
        "cell_tags": out_dir / "nonlinear_cell_tags.csv",
    }
    np.savetxt(paths["nodes"], np.asarray(mesh.points[:, :dimension]), delimiter=",", fmt="%.17g")
    cells = np.vstack([np.asarray(block, dtype=np.int64) for block in blocks]) + 1
    np.savetxt(paths["cells"], cells, delimiter=",", fmt="%d")
    tags = np.asarray(physical, dtype=np.int64).reshape(-1, 1)
    np.savetxt(paths["cell_tags"], tags, delimiter=",", fmt="%d")
    return {key: path.resolve() for key, path in paths.items()}
