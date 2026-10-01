"""Laminate coupon test in tension or compression: stress, strain, elongation and force of a
laminate with any ply damage model (``rve2d.laminate.damage``).

Laminate kinematics are those of classical lamination theory (mid-plane strains and
curvatures, plies in plane stress); the loading is strain controlled in one direction
(``x``, ``y`` or in-plane shear ``xy``; a negative ``max_strain`` is a compression test) with
the other force resultants and all moments at zero, as in a coupon test, so unsymmetric
laminates curve and unbalanced ones shear. Each ply is integrated with two Gauss points
through its thickness, which is exact for the elastic ABD response, and every load step is
solved by Newton iterations with a finite-difference Jacobian, a line search and step
cutting. Modes that fail suddenly (maximum stress, Hashin, fibre failure of the RVE-curve
model) are located by cutting the step back until it is a small fraction of the nominal one
(``failure_resolution``), so the peak stresses do not depend on the step size; the failed
ply is then degraded and the step solved again until no new failure appears, so the load is
redistributed within the step.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rve2d.exceptions import ConfigError
from rve2d.laminate.clt import Laminate, abd_matrix
from rve2d.laminate.damage import PlyModel, build_ply_model

FloatArray = NDArray[np.float64]

DIRECTIONS = {"x": 0, "y": 1, "xy": 2}
_GAUSS = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


@dataclass(frozen=True)
class CouponSettings:
    direction: str = "x"
    max_strain: float = 0.02  # negative: compression
    steps: int = 100
    gauge_length: float | None = None  # elongation = strain * gauge_length
    width: float | None = None  # force = stress * thickness * width
    stop_fraction: float = 0.05  # stop once the stress falls below this fraction of the peak
    failure_resolution: float = 0.02  # sudden failures are located within this many steps
    newton_tolerance: float = 1e-10
    newton_max_iterations: int = 30
    max_step_cuts: int = 12


@dataclass
class CouponResult:
    name: str
    model: str
    direction: str
    thickness: float
    records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = True
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def strain(self) -> FloatArray:
        return np.array([r["strain"] for r in self.records])

    @property
    def stress(self) -> FloatArray:
        return np.array([r["stress"] for r in self.records])

    def summary(self) -> dict[str, Any]:
        strain, stress = self.strain, self.stress
        peak = int(np.argmax(np.abs(stress))) if stress.size else 0
        modulus = float(stress[1] / strain[1]) if strain.size > 1 and strain[1] != 0 else math.nan

        def first(predicate: Any) -> dict[str, Any] | None:
            return next((event for event in self.events if predicate(event["event"])), None)

        return {
            "laminate": self.name,
            "model": self.model,
            "direction": self.direction,
            "thickness": self.thickness,
            "initial_modulus": modulus,
            "peak_stress": float(stress[peak]) if stress.size else math.nan,
            "strain_at_peak": float(strain[peak]) if strain.size else math.nan,
            "final_strain": float(strain[-1]) if strain.size else math.nan,
            # first ply failure: the first damage onset, failure or passed curve peak
            "first_ply_failure": first(lambda name: "exceeded" not in name),
            # first fibre failure (continuum damage: the first fibre damage onset)
            "first_fibre_failure": first(lambda name: name.startswith("fibre")),
            # a ply strained beyond its RVE curve (held at the last stress): extend the curve
            "first_curve_exceeded": first(lambda name: name.endswith("curve exceeded")),
            "completed": self.completed,
            "message": self.message,
            "warnings": self.warnings,
            "events": self.events,
        }

    def write(self, directory: str | Path, stem: str = "coupon_test") -> list[Path]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        curve_path = out / f"{stem}.csv"
        columns = list(self.records[0]) if self.records else ["strain", "stress"]
        with curve_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow(record)
        summary_path = out / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return [curve_path, summary_path]


def coupon_test(
    laminate: Laminate,
    settings: CouponSettings | None = None,
    model: str | PlyModel = "rve_curves",
    characteristic_length: float = 1.0,
) -> CouponResult:
    """Strain-controlled coupon test of ``laminate`` with the ply ``model`` (a name from
    ``rve2d.laminate.damage.MODELS``, ``"elastic"``, or a ``PlyModel``)."""
    settings = settings or CouponSettings()
    if settings.direction not in DIRECTIONS:
        raise ConfigError("The coupon test direction must be x, y or xy.")
    if settings.steps < 1 or settings.max_strain == 0.0:
        raise ConfigError("The coupon test needs a non-zero max_strain and at least one step.")
    law = (
        build_ply_model(model, laminate.ply, characteristic_length)
        if isinstance(model, str)
        else model
    )
    z = laminate.interfaces
    points = []  # (ply index, z, weight, rotation)
    for k, angle in enumerate(laminate.angles):
        middle, half = 0.5 * (z[k] + z[k + 1]), 0.5 * (z[k + 1] - z[k])
        rotation = _rotation(angle)
        for xi in _GAUSS:
            points.append((k, middle + xi * half, half, rotation))
    states: list[Any] = [law.initial_state() for _ in points]
    loaded = DIRECTIONS[settings.direction]
    free = [i for i in range(6) if i != loaded]
    abd = abd_matrix(laminate)
    force_scale = float(np.abs(abd[:3, :3]).max() * abs(settings.max_strain))
    moment_scale = float(np.abs(abd[3:, 3:]).max() * abs(settings.max_strain) / laminate.thickness)
    scales = np.array([force_scale] * 3 + [moment_scale] * 3)
    elastic = abd[np.ix_(free, free)]

    def resultants(unknowns: FloatArray) -> tuple[FloatArray, list[Any]]:
        total = np.zeros(6)
        trials = []
        for (_, zp, weight, rotation), state in zip(points, states, strict=True):
            strain = rotation @ (unknowns[:3] + zp * unknowns[3:])
            stress_ply, trial = law.stress(strain, state)
            stress_lam = rotation.T @ stress_ply
            total[:3] += weight * stress_lam
            total[3:] += weight * zp * stress_lam
            trials.append((stress_ply, trial))
        return total, trials

    def solve(target: float, guess: FloatArray) -> FloatArray | None:
        unknowns = guess.copy()
        unknowns[loaded] = target
        for _ in range(settings.newton_max_iterations):
            total, _ = resultants(unknowns)
            residual = total[free]
            if np.all(np.abs(residual) <= settings.newton_tolerance * scales[free]):
                return unknowns
            jacobian = np.zeros((5, 5))
            for column, index in enumerate(free):
                step = 1e-7 * max(abs(settings.max_strain), abs(unknowns[index]))
                if index >= 3:
                    step /= laminate.thickness
                shifted = unknowns.copy()
                shifted[index] += step
                jacobian[:, column] = (resultants(shifted)[0][free] - residual) / step
            # Failed or plateaued plies can leave a direction without stiffness (e.g. the
            # bending of a laminate whose plies all sit on a flat curve): a small multiple of
            # the elastic stiffness keeps the system solvable without slowing convergence.
            jacobian += 1e-6 * elastic
            try:
                update = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                return None
            norm = float(np.linalg.norm(residual / scales[free]))
            for _ in range(12):  # backtracking line search
                trial = unknowns.copy()
                trial[free] += update
                if (
                    float(np.linalg.norm(resultants(trial)[0][free] / scales[free])) < norm
                    or norm < 1e-14
                ):
                    unknowns = trial
                    break
                update *= 0.5
            else:
                unknowns = trial
        total, _ = resultants(unknowns)
        if np.all(np.abs(total[free]) <= 1e3 * settings.newton_tolerance * scales[free]):
            return unknowns
        return None

    result = CouponResult(laminate.name, law.name, settings.direction, laminate.thickness)
    unknowns = np.zeros(6)
    result.records.append(_record(laminate, settings, unknowns, np.zeros(6), law, states, points))
    increment = settings.max_strain / settings.steps
    applied, cuts, peak = 0.0, 0, 0.0
    seen: set[tuple[str, int]] = set()  # (event, ply) pairs already reported

    def report(name: str, ply: int, resultant: float, strain: float) -> None:
        if (name, ply) not in seen:
            seen.add((name, ply))
            result.events.append(_event(name, strain, resultant, laminate, ply))

    nominal = settings.max_strain / settings.steps
    while abs(applied) < abs(settings.max_strain) * (1 - 1e-12):
        target = applied + increment
        if abs(target) > abs(settings.max_strain):
            target = settings.max_strain
        solution = solve(target, unknowns)
        if solution is None:
            cuts += 1
            if cuts > settings.max_step_cuts:
                result.completed = False
                result.message = (
                    f"no equilibrium beyond strain {applied:.6g} after {settings.max_step_cuts} "
                    "increment cuts"
                )
                break
            increment *= 0.5
            continue
        total, trials = resultants(solution)
        if abs(increment) > settings.failure_resolution * abs(nominal) and any(
            law.failures(stress_ply, trial)[0] for stress_ply, trial in trials
        ):
            increment *= 0.5  # a ply fails suddenly in this step: locate the failure strain
            continue
        # commit the history; degrade suddenly failing plies and solve again until none fails
        while True:
            total, trials = resultants(solution)
            failed_now = False
            for index, (stress_ply, trial) in enumerate(trials):
                ply = points[index][0]
                for event in law.events(states[index], trial):
                    report(event, ply, total[loaded], target)
                modes, updated = law.failures(stress_ply, trial)
                states[index] = updated
                for event in modes:
                    report(event, ply, total[loaded], target)
                failed_now = failed_now or bool(modes)
            if not failed_now:
                break
            if all(law.fibre_failed(state) for state in states):
                solution = None
                break
            resolved = solve(target, solution)
            if resolved is None:
                solution = None
                break
            solution = resolved
        if solution is None:
            total = np.zeros(6)
            unknowns[loaded] = target
            result.records.append(_record(laminate, settings, unknowns, total, law, states, points))
            result.message = "all load-carrying plies failed"
            break
        unknowns = solution
        applied = target
        result.records.append(_record(laminate, settings, unknowns, total, law, states, points))
        stress = total[loaded] / laminate.thickness
        peak = max(peak, abs(stress))
        if peak > 0 and abs(stress) < settings.stop_fraction * peak:
            result.message = f"stress fell below {settings.stop_fraction:g} of the peak"
            break
        if abs(increment) < abs(nominal):
            increment = math.copysign(min(1.5 * abs(increment), abs(nominal)), nominal)
    result.warnings = law.warnings()
    return result


def _rotation(angle_deg: float) -> FloatArray:
    """Strain transformation laminate (x, y, xy) -> ply (1, 2, 12), engineering shear."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array(
        [
            [c * c, s * s, c * s],
            [s * s, c * c, -c * s],
            [-2 * c * s, 2 * c * s, c * c - s * s],
        ]
    )


def _event(
    name: str, strain: float, resultant: float, laminate: Laminate, ply: int
) -> dict[str, Any]:
    return {
        "event": name,
        "strain": float(strain),
        "stress": float(resultant / laminate.thickness),
        "ply": int(ply) + 1,
        "angle": float(laminate.angles[ply]),
    }


def _record(
    laminate: Laminate,
    settings: CouponSettings,
    unknowns: FloatArray,
    total: FloatArray,
    law: PlyModel,
    states: list[Any],
    points: list[Any],
) -> dict[str, Any]:
    loaded = DIRECTIONS[settings.direction]
    strain = float(unknowns[loaded])
    stress = float(total[loaded] / laminate.thickness)
    record: dict[str, Any] = {
        "strain": strain,
        "stress": stress,
        "strain_xx": float(unknowns[0]),
        "strain_yy": float(unknowns[1]),
        "strain_xy": float(unknowns[2]),
        "curvature_xx": float(unknowns[3]),
        "curvature_yy": float(unknowns[4]),
        "curvature_xy": float(unknowns[5]),
        "damaged_plies": len({points[i][0] for i, s in enumerate(states) if law.damaged(s)}),
        "fibre_failed_plies": len(
            {points[i][0] for i, s in enumerate(states) if law.fibre_failed(s)}
        ),
    }
    if settings.gauge_length is not None:
        record["elongation"] = strain * settings.gauge_length
    if settings.width is not None:
        record["force"] = stress * laminate.thickness * settings.width
    return record
