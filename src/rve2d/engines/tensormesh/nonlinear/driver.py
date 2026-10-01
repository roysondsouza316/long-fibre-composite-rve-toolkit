"""Run the nonlinear RVE solve described by the ``nonlinear`` config section."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rve2d.config import (
    NONLINEAR_ACTIVE_COMPONENTS,
    NonlinearSolveConfig,
    PhaseMaterialConfig,
    RVEConfig,
)
from rve2d.engines.common.records import (
    COMPONENT_INDEX,
    NonlinearResult,
    StepRecord,
    load_path_from_config,
    response_summary,
    write_response_csv,
    write_summary_json,
)
from rve2d.engines.tensormesh.constraints import build_dof_map
from rve2d.engines.tensormesh.mesh import RVEMesh, build_rve_mesh
from rve2d.engines.tensormesh.nonlinear import linear_solver
from rve2d.engines.tensormesh.nonlinear.assembly import Evaluation, RVESystem
from rve2d.engines.tensormesh.nonlinear.laws import build_traction_law
from rve2d.engines.tensormesh.nonlinear.material import ElementMaterial, element_material
from rve2d.engines.tensormesh.nonlinear.output import write_fields
from rve2d.engines.tensormesh.nonlinear.solver import (
    NewtonSettings,
    SolverState,
    initial_solver_state,
    solve_load_path,
)
from rve2d.exceptions import ConfigError

__all__ = ["NonlinearResult", "load_path_from_config", "run_nonlinear"]


def phase_material(
    mesh: RVEMesh, nl: NonlinearSolveConfig, device: torch.device
) -> ElementMaterial:
    assert nl.matrix is not None and nl.fibre is not None
    is_fibre = mesh.phase == mesh.fibre_phase_id

    def per_element(pick: Any) -> torch.Tensor:
        values = np.where(is_fibre, pick(nl.fibre), pick(nl.matrix))
        return torch.as_tensor(values, dtype=torch.float64, device=device)

    def yield_stress(phase: PhaseMaterialConfig) -> float:
        return float("inf") if phase.yield_stress is None else phase.yield_stress

    return element_material(
        per_element(lambda p: p.youngs_modulus),
        per_element(lambda p: p.poisson_ratio),
        per_element(yield_stress),
        per_element(lambda p: p.hardening_modulus),
    )


def run_nonlinear(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
) -> NonlinearResult:
    nl = config.nonlinear
    if not nl.enabled:
        raise ConfigError("Nonlinear solve requested but nonlinear.enabled is false in the config.")
    assert nl.matrix is not None and nl.fibre is not None
    started = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(nl.device)
    kinematics = nl.resolved_kinematics(config.dimension)
    interface = nl.interface if nl.interface is not None and nl.interface.enabled else None

    mesh = build_rve_mesh(
        mesh_path, nl.matrix_phase_id, nl.fibre_phase_id, cohesive_interfaces=interface is not None
    )
    if mesh.dim != config.dimension:
        raise ConfigError(f"Mesh is {mesh.dim}D but the config has dimension {config.dimension}.")
    dof_map = build_dof_map(mesh, nl.boundary_condition)
    law = None
    if interface is not None:
        assert interface.penalty_stiffness is not None and interface.normal_strength is not None
        assert interface.mode_i_toughness is not None
        law = build_traction_law(
            interface.law,
            penalty_stiffness=interface.penalty_stiffness,
            normal_strength=interface.normal_strength,
            shear_strength=interface.shear_strength,
            mode_i_toughness=interface.mode_i_toughness,
            mode_ii_toughness=interface.mode_ii_toughness,
            bk_exponent=interface.bk_exponent,
            mixed_mode_criterion=interface.mixed_mode_criterion,
            viscosity=interface.viscosity,
            shear_penalty_stiffness=interface.shear_penalty_stiffness,
        ).to(device)
    active = [COMPONENT_INDEX[c] for c in NONLINEAR_ACTIVE_COMPONENTS[kinematics]]
    system = RVESystem(
        mesh,
        dof_map,
        phase_material(mesh, nl, device),
        law,
        active,
        device,
        cohesive_integration="nodal" if interface is None else interface.integration,
    )
    load = load_path_from_config(kinematics, nl.load)
    settings = NewtonSettings(
        max_iterations=nl.newton_max_iterations,
        tolerance=nl.newton_tolerance,
        steps=nl.load.steps,
        max_cuts=nl.max_step_cuts,
        linear_backend=nl.linear_solver,
    )
    state = initial_solver_state(system, int(getattr(law, "state_dim", 1)))
    field_files: list[Path] = []

    def on_step(record: StepRecord, current: SolverState, evaluation: Evaluation) -> None:
        step = len(records_seen) + 1
        records_seen.append(record)
        if nl.output_every > 0 and step % nl.output_every == 0:
            field_files.extend(
                write_fields(out_dir, f"nonlinear_step_{step:04d}", system, current, evaluation)
            )

    records_seen: list[StepRecord] = []
    outcome = solve_load_path(
        system, load, settings, state, nl.matrix.youngs_modulus, on_step=on_step
    )
    field_files.extend(
        write_fields(out_dir, "nonlinear_final", system, outcome.state, outcome.last_evaluation)
    )

    response_path = write_response_csv(out_dir / "nonlinear_response.csv", outcome.records)
    response = response_summary(outcome.records, nl.load)
    summary: dict[str, Any] = {
        "engine": "tensormesh",
        "dimension": mesh.dim,
        "kinematics": kinematics,
        "boundary_condition": nl.boundary_condition,
        "load": asdict(nl.load),
        "materials": {"matrix": asdict(nl.matrix), "fibre": asdict(nl.fibre)},
        "interface": None if interface is None else asdict(interface),
        "mesh": {
            "nodes": mesh.n_nodes,
            "bulk_cells": mesh.n_cells,
            "cohesive_elements": mesh.n_cohesive,
            "unknowns": system.n_reduced + len(load.free),
        },
        "device": str(device),
        "linear_solver": linear_solver.resolve_backend(nl.linear_solver, device),
        "completed": outcome.completed,
        "reached_load_time": outcome.reached_time,
        "steps": len(outcome.records) - 1,
        "newton_iterations": outcome.total_iterations,
        "increment_cuts": outcome.cuts,
        "messages": outcome.messages,
        **response,
        "runtime_seconds": round(time.time() - started, 2),
        "response_csv": str(response_path),
        "field_files": [str(path) for path in field_files],
    }
    summary_path = write_summary_json(out_dir / "nonlinear_summary.json", summary)
    return NonlinearResult(
        summary_path=summary_path,
        response_path=response_path,
        field_files=field_files,
        completed=outcome.completed,
        peak_stress=response["peak_stress"],
        strain_at_peak=response["strain_at_peak"],
        records=outcome.records,
    )
