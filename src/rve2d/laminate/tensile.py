"""Laminate tensile test: stress, strain and elongation of a coupon with a given stacking
sequence, from the ply curves of the RVE solves.

Laminate kinematics are those of classical lamination theory (mid-plane strains and
curvatures, plies in plane stress); the loading is strain controlled in one direction
(``x``, ``y`` or in-plane shear ``xy``) with the other force resultants and all moments at
zero, as in a coupon test. Each ply is integrated with two Gauss points through its
thickness, which is exact for the elastic ABD response.

Ply behaviour in ply axes (1 = fibre, 2 = transverse, 12 = in-plane shear):

* fibre direction: linear elastic up to the longitudinal tensile or compressive strength
  (fibre failure is not part of the RVE model, so these strengths are inputs); a ply whose
  fibres fail carries no more load;
* transverse direction: the RVE uniaxial-stress curve (matrix plasticity and fibre/matrix
  debonding), driven by the transverse strain caused by the transverse stress; compression
  is linear elastic unless a compression curve is given;
* in-plane shear: the RVE shear curve;
* after a curve's maximum strain has been reached, unloading follows the secant to the
  origin (damage-like memory), and beyond the last sample the stress stays at its last value
  (reported as a "curve exceeded" event: extend the RVE curve to cover such strains).

The transverse and shear responses are taken as independent of each other (no interaction),
an approximation that is exact for loads that excite one mode in each ply, e.g. a [90]
laminate in transverse tension, where the laminate curve reproduces the RVE curve. With
linear curves the test reproduces the elastic CLT modulus exactly.
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
from rve2d.laminate.clt import IN_PLANE, Laminate, abd_matrix
from rve2d.laminate.ply import PlyCurve

FloatArray = NDArray[np.float64]

DIRECTIONS = {"x": 0, "y": 1, "xy": 2}
_GAUSS = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


@dataclass(frozen=True)
class TensileSettings:
    direction: str = "x"
    max_strain: float = 0.02
    steps: int = 100
    gauge_length: float | None = None  # elongation = strain * gauge_length
    width: float | None = None  # force = stress * thickness * width
    stop_fraction: float = 0.05  # stop once the stress falls below this fraction of the peak
    newton_tolerance: float = 1e-10
    newton_max_iterations: int = 30
    max_step_cuts: int = 12


@dataclass
class _PointState:
    kappa_tension: float = 0.0  # largest transverse tensile (mechanical) strain reached
    kappa_compression: float = 0.0
    kappa_shear: float = 0.0
    fibre_failed: bool = False


@dataclass(frozen=True)
class _PlyLaw:
    s11: float
    s12: float
    s22: float
    s66: float
    tension: PlyCurve | None
    compression: PlyCurve | None
    shear: PlyCurve | None
    xt: float | None
    xc: float | None

    def stress(
        self, strain: FloatArray, state: _PointState
    ) -> tuple[FloatArray, tuple[float, float, float]]:
        """Ply stress (1, 2, 12) for ply strain (1, 2, 12); also the trial history values."""
        if state.fibre_failed:
            return np.zeros(3), (state.kappa_tension, state.kappa_compression, state.kappa_shear)
        e1, e2, g12 = float(strain[0]), float(strain[1]), float(strain[2])
        tau, kappa_s = _secant(self.shear, abs(g12), state.kappa_shear, 1.0 / self.s66)
        tau = math.copysign(tau, g12)
        sigma2 = (e2 - self.s12 * e1 / self.s11) / (self.s22 - self.s12**2 / self.s11)
        kappa_t, kappa_c = state.kappa_tension, state.kappa_compression
        for _ in range(60):  # fixed point; contraction factor ~ nu12^2 E2 / E1 << 1
            sigma1 = (e1 - self.s12 * sigma2) / self.s11
            mechanical = e2 - self.s12 * sigma1  # transverse strain caused by sigma2
            if mechanical >= 0.0:
                value, kappa_t = _secant(
                    self.tension, mechanical, state.kappa_tension, 1 / self.s22
                )
                kappa_c = state.kappa_compression
            else:
                value, kappa_c = _secant(
                    self.compression, -mechanical, state.kappa_compression, 1 / self.s22
                )
                value, kappa_t = -value, state.kappa_tension
            if abs(value - sigma2) <= 1e-13 * (abs(value) + 1e-300) + 1e-300:
                sigma2 = value
                break
            sigma2 = value
        sigma1 = (e1 - self.s12 * sigma2) / self.s11
        return np.array([sigma1, sigma2, tau]), (kappa_t, kappa_c, kappa_s)

    def fibre_fails(self, sigma1: float) -> bool:
        return (self.xt is not None and sigma1 >= self.xt) or (
            self.xc is not None and sigma1 <= -self.xc
        )


def _secant(
    curve: PlyCurve | None, strain: float, kappa: float, modulus: float
) -> tuple[float, float]:
    """Stress magnitude for a strain magnitude with secant unloading below ``kappa``."""
    if curve is None:
        return modulus * strain, kappa
    if strain >= kappa:
        return curve.stress_at(strain), strain
    if kappa <= 0.0:
        return 0.0, kappa
    return curve.stress_at(kappa) * strain / kappa, kappa


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


@dataclass
class TensileResult:
    name: str
    direction: str
    thickness: float
    records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = True
    message: str = ""

    @property
    def strain(self) -> FloatArray:
        return np.array([r["strain"] for r in self.records])

    @property
    def stress(self) -> FloatArray:
        return np.array([r["stress"] for r in self.records])

    def summary(self) -> dict[str, Any]:
        strain, stress = self.strain, self.stress
        peak = int(np.argmax(stress)) if stress.size else 0
        modulus = float(stress[1] / strain[1]) if strain.size > 1 and strain[1] != 0 else math.nan
        first = {event["event"]: event for event in reversed(self.events)}
        return {
            "laminate": self.name,
            "direction": self.direction,
            "thickness": self.thickness,
            "initial_modulus": modulus,
            "peak_stress": float(stress[peak]) if stress.size else math.nan,
            "strain_at_peak": float(strain[peak]) if strain.size else math.nan,
            "final_strain": float(strain[-1]) if strain.size else math.nan,
            "first_transverse_damage": first.get("transverse peak passed"),
            "first_shear_damage": first.get("shear peak passed"),
            "first_fibre_failure": first.get("fibre failure"),
            # a ply strained beyond its RVE curve (held at the last stress): extend the curve
            "first_curve_exceeded": next(
                (event for event in self.events if event["event"].endswith("curve exceeded")),
                None,
            ),
            "completed": self.completed,
            "message": self.message,
            "events": self.events,
        }

    def write(self, directory: str | Path, stem: str = "tensile_test") -> list[Path]:
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


def tensile_test(
    laminate: Laminate,
    settings: TensileSettings | None = None,
    elastic_only: bool = False,
) -> TensileResult:
    """Strain-controlled test of ``laminate``; ``elastic_only`` ignores curves and strengths."""
    settings = settings or TensileSettings()
    if settings.direction not in DIRECTIONS:
        raise ConfigError("The tensile test direction must be x, y or xy.")
    if settings.steps < 1 or settings.max_strain == 0.0:
        raise ConfigError("The tensile test needs a non-zero max_strain and at least one step.")
    ply = laminate.ply
    compliance = np.linalg.inv(ply.stiffness)[np.ix_(IN_PLANE, IN_PLANE)]
    law = _PlyLaw(
        s11=float(compliance[0, 0]),
        s12=float(compliance[0, 1]),
        s22=float(compliance[1, 1]),
        s66=float(compliance[2, 2]),
        tension=None if elastic_only else ply.transverse_tension,
        compression=None if elastic_only else ply.transverse_compression,
        shear=None if elastic_only else ply.shear,
        xt=None if elastic_only else ply.longitudinal_tensile_strength,
        xc=None if elastic_only else ply.longitudinal_compressive_strength,
    )
    z = laminate.interfaces
    points = []  # (ply index, z, weight, rotation)
    for k, angle in enumerate(laminate.angles):
        middle, half = 0.5 * (z[k] + z[k + 1]), 0.5 * (z[k + 1] - z[k])
        rotation = _rotation(angle)
        for xi in _GAUSS:
            points.append((k, middle + xi * half, half, rotation))
    states = [_PointState() for _ in points]
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
            stress_ply, history = law.stress(strain, state)
            stress_lam = rotation.T @ stress_ply
            total[:3] += weight * stress_lam
            total[3:] += weight * zp * stress_lam
            trials.append((stress_ply, history))
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
            # A plateau of the ply curves can leave a direction without stiffness (e.g. the
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

    result = TensileResult(laminate.name, settings.direction, laminate.thickness)
    unknowns = np.zeros(6)
    result.records.append(_record(laminate, settings, unknowns, np.zeros(6), states, points))
    increment = settings.max_strain / settings.steps
    applied, cuts, peak = 0.0, 0, 0.0
    seen: set[tuple[str, int]] = set()  # (event, ply) pairs already reported
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
        # commit the history and check fibre failure; re-solve while new plies fail
        while True:
            total, trials = resultants(solution)
            failed_now = set()
            for (k, _, _, _), state, (stress_ply, history) in zip(
                points, states, trials, strict=True
            ):
                state.kappa_tension, state.kappa_compression, state.kappa_shear = history
                if not state.fibre_failed and law.fibre_fails(float(stress_ply[0])):
                    failed_now.add(k)
            if not failed_now:
                break
            for (k, _, _, _), state in zip(points, states, strict=True):
                if k in failed_now:
                    state.fibre_failed = True
            for k in sorted(failed_now):
                result.events.append(_event("fibre failure", target, total[loaded], laminate, k))
            if all(state.fibre_failed for state in states):
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
            result.records.append(_record(laminate, settings, unknowns, total, states, points))
            result.message = "all load-carrying plies failed"
            break
        unknowns = solution
        applied = target
        for (k, _, _, _), state in zip(points, states, strict=True):
            for mode, curve, kappa in (
                ("transverse", law.tension, state.kappa_tension),
                ("shear", law.shear, state.kappa_shear),
            ):
                if curve is None:
                    continue
                # (a relative tolerance keeps round-off at the end of a curve from counting)
                checks = [(f"{mode} curve exceeded", curve.max_strain * (1.0 + 1e-6))]
                if curve.strain_at_peak < curve.max_strain:  # the curve has a peak
                    checks.insert(0, (f"{mode} peak passed", curve.strain_at_peak))
                for event, limit in checks:
                    if kappa > limit and (event, k) not in seen:
                        seen.add((event, k))
                        result.events.append(_event(event, applied, total[loaded], laminate, k))
        result.records.append(_record(laminate, settings, unknowns, total, states, points))
        stress = total[loaded] / laminate.thickness
        peak = max(peak, abs(stress))
        if peak > 0 and abs(stress) < settings.stop_fraction * peak:
            result.message = f"stress fell below {settings.stop_fraction:g} of the peak"
            break
        if abs(increment) < abs(nominal):
            increment = math.copysign(min(1.5 * abs(increment), abs(nominal)), nominal)
    return result


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
    settings: TensileSettings,
    unknowns: FloatArray,
    total: FloatArray,
    states: list[_PointState],
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
        "fibre_failed_plies": len({points[i][0] for i, s in enumerate(states) if s.fibre_failed}),
    }
    if settings.gauge_length is not None:
        record["elongation"] = strain * settings.gauge_length
    if settings.width is not None:
        record["force"] = stress * laminate.thickness * settings.width
    return record
