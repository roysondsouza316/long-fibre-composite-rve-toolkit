"""Linear homogenization with the Julia engine (``FerriteRVE.homogenize`` on Ferrite.jl).

Writes the mesh arrays and a TOML input, runs the engine and reads back its result file;
the engine-independent outputs are written by ``rve2d.engines.common.homogenization``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from rve2d.engines.common.homogenization import EngineSolution, HomogenizationProblem
from rve2d.engines.julia.runner import export_mesh_arrays, run_task, write_toml

RESULT_NAME = "julia_homogenization_result.toml"


def solve(problem: HomogenizationProblem) -> EngineSolution:
    out_dir = problem.output_dir
    arrays = export_mesh_arrays(problem.mesh_path, out_dir, problem.dimension, "homogenization")
    input_path = write_toml(
        out_dir / "julia_homogenization_input.toml",
        {
            "task": "homogenization",
            "dimension": problem.dimension,
            "kinematics": problem.kinematics,
            "boundary_condition": problem.boundary_condition,
            "nodes": str(arrays["nodes"]),
            "cells": str(arrays["cells"]),
            "cell_tags": str(arrays["cell_tags"]),
            "output_dir": str(out_dir),
            "matrix_phase_id": problem.matrix_phase_id,
            "fibre_phase_id": problem.fibre_phase_id,
            "write_vtk": problem.write_vtk,
            "matrix_stiffness": problem.matrix_stiffness.tolist(),
            "fibre_stiffness": problem.fibre_stiffness.tolist(),
        },
        {},
    )
    (out_dir / RESULT_NAME).unlink(missing_ok=True)
    log_path = out_dir / "homogenization_log.txt"
    run_task(input_path, log_path)
    result = tomllib.loads((out_dir / RESULT_NAME).read_text(encoding="utf-8"))
    faces = result["traction_faces"]
    vtk = result.get("vtk_path") or ""
    anchor = int(result.get("anchor_node", 0))  # 1-based in Julia, 0 when there is none
    return EngineSolution(
        stiffness=np.asarray(result["stiffness"], dtype=np.float64),
        tractions=[dict(zip(faces, case, strict=True)) for case in result["tractions"]],
        fibre_volume_fraction=float(result["fibre_volume_fraction"]),
        unknowns=int(result["unknowns"]),
        vtk_path=Path(vtk) if vtk else None,
        log_path=log_path,
        details={
            "anchor_node": anchor - 1 if anchor > 0 else None,
            "julia_version": result.get("julia_version"),
            "julia_runtime_seconds": result.get("runtime_seconds"),
        },
    )
