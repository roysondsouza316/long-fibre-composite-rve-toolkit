"""Incremental Newton solution of the nonlinear RVE problem under mixed macro control.

Each macro strain component is either *prescribed* (follows the load factor, or stays at
zero) or *stress-controlled* (an unknown whose conjugate macro stress follows its target,
typically zero). This covers e.g. uniaxial stress along one axis with all other macro
stresses free, or uniaxial strain. History variables (plastic strain, cohesive damage
history) are committed only after a converged increment; non-converged increments are cut in
half and retried, and the increment grows back after easy convergence -- the standard
cohesive-zone stepping strategy (also used by diffcohesive's adaptive stepping).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from rve2d.exceptions import SolverError
from rve2d.nonlinear import linear_solver
from rve2d.nonlinear.assembly import Evaluation, RVESystem
from rve2d.nonlinear.material import PlasticState, initial_state


@dataclass(frozen=True)
class LoadPath:
    """Macro loading as a function of pseudo-time ``t`` in ``[0, end_time]``."""

    prescribed_strain: dict[int, float]  # component -> strain at load factor 1
    fixed_zero: list[int]  # components held at zero strain (inactive or strain-controlled at 0)
    stress_controlled: dict[int, float]  # component -> target macro stress at load factor 1
    unload: bool = False

    @property
    def end_time(self) -> float:
        return 2.0 if self.unload else 1.0

    def factor(self, t: float) -> float:
        return t if t <= 1.0 else 2.0 - t

    @property
    def free(self) -> list[int]:
        return sorted(self.stress_controlled)


@dataclass(frozen=True)
class NewtonSettings:
    max_iterations: int = 25
    tolerance: float = 1e-8
    steps: int = 40
    max_cuts: int = 8
    growth: float = 1.5
    line_search: bool = True
    linear_backend: str = "auto"


@dataclass
class SolverState:
    w: torch.Tensor
    macro_strain: torch.Tensor
    plastic: PlasticState
    cohesive_history: torch.Tensor


@dataclass
class StepRecord:
    time: float
    load_factor: float
    macro_strain: list[float]
    macro_stress: list[float]
    iterations: int
    max_damage: float
    damaged_fraction: float
    mean_damage: float
    yielded_fraction: float
    mean_eqps_matrix: float
    mean_eqps_fibre: float
    work_density: float


@dataclass
class SolveOutcome:
    records: list[StepRecord]
    state: SolverState
    last_evaluation: Evaluation
    completed: bool
    reached_time: float
    total_iterations: int
    cuts: int
    messages: list[str] = field(default_factory=list)


def initial_solver_state(system: RVESystem, law_state_dim: int) -> SolverState:
    dtype, device = system.dtype, system.device
    n_q = system.cohesive_quadrature_points()
    n_coh = system.mesh.n_cohesive
    shape: tuple[int, ...] = (n_coh, n_q) if law_state_dim == 1 else (n_coh, n_q, law_state_dim)
    return SolverState(
        w=torch.zeros(system.n_reduced, dtype=dtype, device=device),
        macro_strain=torch.zeros(6, dtype=dtype, device=device),
        plastic=initial_state(system.mesh.n_cells, dtype, device),
        cohesive_history=torch.zeros(shape, dtype=dtype, device=device),
    )


def solve_load_path(
    system: RVESystem,
    load: LoadPath,
    settings: NewtonSettings,
    state: SolverState,
    stress_scale: float,
    on_step: Callable[[StepRecord, SolverState, Evaluation], None] | None = None,
) -> SolveOutcome:
    backend = linear_solver.resolve_backend(settings.linear_backend, system.device)
    free = load.free
    target_final = torch.zeros(6, dtype=system.dtype, device=system.device)
    for comp, value in load.stress_controlled.items():
        target_final[comp] = value
    force_floor = 1e-6 * stress_scale * system.volume ** ((system.dim - 1) / system.dim)
    stress_floor = 1e-6 * stress_scale

    first = _evaluate(system, state, free, target_final * 0.0, with_tangent=False)
    records = [_record(system, 0.0, 0.0, state, first, 0, 0.0)]
    last_eval = first
    t, dt0 = 0.0, load.end_time / settings.steps
    dt = dt0
    min_dt = dt0 / 2**settings.max_cuts
    total_iterations = cuts = 0
    messages: list[str] = []
    work = 0.0
    while t < load.end_time - 1e-12:
        dt = min(dt, load.end_time - t)
        t_new = t + dt
        set_time_step = getattr(system.law, "set_time_step", None)
        if set_time_step is not None:
            set_time_step(dt)
        lam = load.factor(t_new)
        trial = SolverState(
            state.w.clone(), state.macro_strain.clone(), state.plastic, state.cohesive_history
        )
        for comp, value in load.prescribed_strain.items():
            trial.macro_strain[comp] = lam * value
        for comp in load.fixed_zero:
            trial.macro_strain[comp] = 0.0
        converged, evaluation, iterations = _newton(
            system, trial, free, lam * target_final, settings, backend, force_floor, stress_floor
        )
        total_iterations += iterations
        if not converged:
            cuts += 1
            dt *= 0.5
            if dt < min_dt:
                messages.append(
                    f"Stopped at t={t:.6g}: no convergence after "
                    f"{settings.max_cuts} increment cuts."
                )
                break
            continue
        committed = SolverState(
            trial.w, trial.macro_strain, evaluation.plastic_state, evaluation.cohesive_history
        )
        work += _work_increment(records[-1], committed, evaluation)
        state, t, last_eval = committed, t_new, evaluation
        record = _record(system, t, lam, state, evaluation, iterations, work)
        records.append(record)
        if on_step is not None:
            on_step(record, state, evaluation)
        if iterations <= 4:
            dt = min(dt * settings.growth, dt0)
    return SolveOutcome(
        records=records,
        state=state,
        last_evaluation=last_eval,
        completed=t >= load.end_time - 1e-12,
        reached_time=t,
        total_iterations=total_iterations,
        cuts=cuts,
        messages=messages,
    )


def _evaluate(
    system: RVESystem,
    state: SolverState,
    free: list[int],
    target: torch.Tensor,
    with_tangent: bool = True,
) -> Evaluation:
    return system.evaluate(
        state.w,
        state.macro_strain,
        state.plastic,
        state.cohesive_history,
        free,
        target,
        with_tangent,
    )


def _newton(
    system: RVESystem,
    trial: SolverState,
    free: list[int],
    target: torch.Tensor,
    settings: NewtonSettings,
    backend: str,
    force_floor: float,
    stress_floor: float,
) -> tuple[bool, Evaluation, int]:
    n_red = system.n_reduced
    evaluation = _evaluate(system, trial, free, target)
    history: list[float] = []
    for iteration in range(1, settings.max_iterations + 1):
        error = _error(system, evaluation, free, force_floor, stress_floor)
        if not math.isfinite(error):
            return False, evaluation, iteration
        if error <= settings.tolerance:
            return True, evaluation, iteration - 1
        history.append(error)
        # Give up early on a stalled or diverging iteration: cutting the increment is cheaper.
        if len(history) >= 8 and error > 0.5 * history[-5]:
            return False, evaluation, iteration
        assert evaluation.matrix is not None
        try:
            step = linear_solver.solve(evaluation.matrix, -evaluation.residual, backend)
        except SolverError:
            return False, evaluation, iteration
        alpha = 1.0
        base_w, base_e = trial.w.clone(), trial.macro_strain.clone()
        for _ in range(4 if settings.line_search else 1):
            trial.w = base_w + alpha * step[:n_red]
            trial.macro_strain = base_e.clone()
            if free:
                trial.macro_strain[free] = base_e[free] + alpha * step[n_red:]
            candidate = _evaluate(system, trial, free, target)
            candidate_error = _error(system, candidate, free, force_floor, stress_floor)
            if math.isfinite(candidate_error) and candidate_error < error:
                break
            alpha *= 0.5
        evaluation = candidate
    final_error = _error(system, evaluation, free, force_floor, stress_floor)
    return final_error <= settings.tolerance, evaluation, settings.max_iterations


def _error(
    system: RVESystem,
    evaluation: Evaluation,
    free: list[int],
    force_floor: float,
    stress_floor: float,
) -> float:
    residual = evaluation.residual
    n_red = system.n_reduced
    force_error = float(torch.linalg.norm(residual[:n_red])) / max(
        evaluation.force_scale, force_floor
    )
    if not free:
        return force_error
    stress_ref = max(float(evaluation.macro_stress.abs().max()), stress_floor) * system.volume
    stress_error = float(residual[n_red:].abs().max()) / stress_ref
    return max(force_error, stress_error)


def _work_increment(previous: StepRecord, state: SolverState, evaluation: Evaluation) -> float:
    strain = state.macro_strain.detach().cpu().tolist()
    stress = evaluation.macro_stress.detach().cpu().tolist()
    return float(
        sum(
            0.5 * (previous.macro_stress[i] + stress[i]) * (strain[i] - previous.macro_strain[i])
            for i in range(6)
        )
    )


def _record(
    system: RVESystem,
    t: float,
    lam: float,
    state: SolverState,
    evaluation: Evaluation,
    iterations: int,
    work: float,
) -> StepRecord:
    vol = system.bulk.volume
    matrix = torch.as_tensor(system.mesh.phase == system.mesh.matrix_phase_id, device=system.device)
    fibre = ~matrix
    eqps = state.plastic.equivalent_plastic_strain

    def mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        weight = vol[mask].sum()
        return float((values[mask] * vol[mask]).sum() / weight) if float(weight) > 0 else 0.0

    max_damage = damaged = mean_damage = 0.0
    if system.cohesive is not None and evaluation.damage.numel():
        weights = system.cohesive.weights
        damage = evaluation.damage
        max_damage = float(damage.max())
        damaged = float((weights * (damage >= 0.99)).sum() / weights.sum())
        mean_damage = float((weights * damage).sum() / weights.sum())
    yielded = float(vol[eqps > 0].sum() / vol.sum())
    return StepRecord(
        time=t,
        load_factor=lam,
        macro_strain=state.macro_strain.detach().cpu().tolist(),
        macro_stress=evaluation.macro_stress.detach().cpu().tolist(),
        iterations=iterations,
        max_damage=max_damage,
        damaged_fraction=damaged,
        mean_damage=mean_damage,
        yielded_fraction=yielded,
        mean_eqps_matrix=mean(eqps, matrix),
        mean_eqps_fibre=mean(eqps, fibre),
        work_density=work,
    )
