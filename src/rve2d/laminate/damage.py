"""Ply damage models for the laminate coupon tests (tension and compression).

Each model maps the strain of one integration point in ply axes (1 = fibre, 2 = transverse,
12 = in-plane shear, engineering shear strain, plane stress) and its history to the ply
stress. From the simplest to the most elaborate:

* ``max_stress`` -- maximum stress criterion with ply discount: a mode fails when its stress
  reaches the strength (``sigma1 >= Xt``, ``-sigma1 >= Xc``, ``sigma2 >= Yt``,
  ``-sigma2 >= Yc``, ``|tau12| >= S12``); fibre failure removes the ply, matrix or shear
  failure removes its transverse and shear stiffness.
* ``hashin`` -- Hashin's (1980) interactive, mode-based criteria (fibre tension and
  compression, matrix tension and compression, with the transverse shear strength S23 in
  matrix compression) and progressive degradation: each failed mode leaves a residual
  fraction of the stiffness (``HASHIN_RESIDUAL``, after Tserpes et al., 2001).
* ``continuum_damage`` -- continuum damage mechanics in the Matzenmiller-Lubliner-Taylor form
  (as in Lapczyk & Hurtado, 2007, and Maimi et al., 2007): Hashin criteria on the effective
  stress start one damage variable per mode, which then grows with linear softening of the
  mode's equivalent stress-strain response such that a crack band of width
  ``characteristic_length`` dissipates the mode's fracture energy. The damage is gradual and
  energy-consistent; the post-peak response depends on the characteristic length.
* ``rve_curves`` -- the stress-strain curves of the nonlinear RVE solves (transverse tension
  and compression, in-plane shear) as secant laws with damage memory, and fibre failure at
  ``Xt`` / ``Xc`` (see ``CurveModel``).
* ``elastic`` -- linear, never fails (for reference).

The damaged plane-stress stiffness of the first three models is

    Q11 = (1-d1) E1 / D,  Q22 = (1-d2) E2 / D,  Q12 = (1-d1)(1-d2) nu12 E2 / D,
    Q66 = (1-d6) G12,     D = 1 - (1-d1)(1-d2) nu12 nu21.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from rve2d.exceptions import ConfigError

if TYPE_CHECKING:
    from rve2d.laminate.ply import PlyCurve, PlyProperties

FloatArray = NDArray[np.float64]

MODELS = ("rve_curves", "max_stress", "hashin", "continuum_damage")
FIBRE_TENSION, FIBRE_COMPRESSION = "fibre tension", "fibre compression"
MATRIX_TENSION, MATRIX_COMPRESSION = "matrix tension", "matrix compression"
SHEAR = "shear"
CDM_MODES = (FIBRE_TENSION, FIBRE_COMPRESSION, MATRIX_TENSION, MATRIX_COMPRESSION)
# residual fractions of (E1, E2, G12) after each Hashin mode (Tserpes et al., 2001; the shear
# modulus degrades with the transverse one after matrix failure)
HASHIN_RESIDUAL = {
    FIBRE_TENSION: (0.07, 0.07, 0.07),
    FIBRE_COMPRESSION: (0.14, 0.14, 0.14),
    MATRIX_TENSION: (1.0, 0.2, 0.2),
    MATRIX_COMPRESSION: (1.0, 0.4, 0.4),
}
DAMAGE_CAP = 0.999  # continuum damage: residual stiffness 0.1 %


@dataclass(frozen=True)
class PlyStrengths:
    """Ply strengths as positive magnitudes: fibre direction ``xt``, ``xc``; transverse ``yt``,
    ``yc``; in-plane shear ``s12``; transverse shear ``s23`` (Hashin's matrix compression
    criterion, ``yc / 2`` when not given)."""

    xt: float | None = None
    xc: float | None = None
    yt: float | None = None
    yc: float | None = None
    s12: float | None = None
    s23: float | None = None

    def require(self, names: tuple[str, ...], model: str) -> dict[str, float]:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ConfigError(
                f"The {model} model needs the ply strengths {', '.join(missing)}: give them in "
                "laminate.strengths or let the RVE curves provide them (yt, yc, s12)."
            )
        return {name: float(getattr(self, name)) for name in names}

    def to_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in ("xt", "xc", "yt", "yc", "s12", "s23")}


@dataclass(frozen=True)
class FractureEnergies:
    """Intralaminar fracture energies (per unit crack area) of the continuum damage model."""

    fibre_tension: float
    fibre_compression: float
    matrix_tension: float
    matrix_compression: float

    def of(self, mode: str) -> float:
        return float(getattr(self, mode.replace(" ", "_")))

    def to_dict(self) -> dict[str, float]:
        return {mode.replace(" ", "_"): self.of(mode) for mode in CDM_MODES}


@dataclass(frozen=True)
class PlyElasticity:
    """Plane-stress engineering constants of the ply."""

    e1: float
    e2: float
    nu12: float
    g12: float

    @staticmethod
    def from_compliance(compliance: FloatArray) -> PlyElasticity:
        """From the 6x6 ply compliance (Voigt 11, 22, 33, 23, 13, 12)."""
        e1 = 1.0 / compliance[0, 0]
        return PlyElasticity(
            float(e1), float(1.0 / compliance[1, 1]), float(-compliance[0, 1] * e1),
            float(1.0 / compliance[5, 5]),
        )  # fmt: skip

    @property
    def nu21(self) -> float:
        return self.nu12 * self.e2 / self.e1

    def stiffness(self, d1: float = 0.0, d2: float = 0.0, d6: float = 0.0) -> FloatArray:
        """Damaged plane-stress stiffness ``Q`` (1, 2, 12)."""
        a1, a2 = 1.0 - d1, 1.0 - d2
        det = 1.0 - a1 * a2 * self.nu12 * self.nu21
        q12 = a1 * a2 * self.nu12 * self.e2 / det
        return np.array(
            [
                [a1 * self.e1 / det, q12, 0.0],
                [q12, a2 * self.e2 / det, 0.0],
                [0.0, 0.0, (1.0 - d6) * self.g12],
            ]
        )


class PlyModel(ABC):
    """Stress of one integration point; states are immutable and committed by the caller."""

    name: str = ""

    @abstractmethod
    def initial_state(self) -> Any: ...

    @abstractmethod
    def stress(self, strain: FloatArray, state: Any) -> tuple[FloatArray, Any]:
        """Ply stress for ply strain and the committed state, with the trial state."""

    def failures(self, stress: FloatArray, state: Any) -> tuple[list[str], Any]:
        """Modes that fail suddenly at this converged stress (the step is then solved again
        with the degraded ply) and the updated state."""
        return [], state

    def events(self, old: Any, new: Any) -> list[str]:
        """Events (damage onsets, passed peaks) between two committed states."""
        return []

    def fibre_failed(self, state: Any) -> bool:
        return False

    def damaged(self, state: Any) -> bool:
        return False

    def warnings(self) -> list[str]:
        return []


class ElasticModel(PlyModel):
    name = "elastic"

    def __init__(self, elastic: PlyElasticity) -> None:
        self.q = elastic.stiffness()

    def initial_state(self) -> None:
        return None

    def stress(self, strain: FloatArray, state: Any) -> tuple[FloatArray, Any]:
        return self.q @ strain, state


# -- maximum stress and Hashin: sudden (ply-discount) degradation --------------------------
@dataclass(frozen=True)
class FailureState:
    failed: frozenset[str] = frozenset()


class _SuddenFailureModel(PlyModel):
    def __init__(self, elastic: PlyElasticity) -> None:
        self.elastic = elastic
        self._stiffness: dict[frozenset[str], FloatArray] = {}

    def initial_state(self) -> FailureState:
        return FailureState()

    @abstractmethod
    def degradation(self, failed: frozenset[str]) -> tuple[float, float, float]: ...

    @abstractmethod
    def failing(self, stress: FloatArray) -> list[str]: ...

    def stress(self, strain: FloatArray, state: FailureState) -> tuple[FloatArray, Any]:
        q = self._stiffness.get(state.failed)
        if q is None:
            q = self._stiffness[state.failed] = self.elastic.stiffness(
                *self.degradation(state.failed)
            )
        return q @ strain, state

    def failures(self, stress: FloatArray, state: FailureState) -> tuple[list[str], Any]:
        new = [mode for mode in self.failing(stress) if mode not in state.failed]
        if not new:
            return [], state
        return [f"{mode} failure" for mode in new], FailureState(state.failed | set(new))

    def fibre_failed(self, state: FailureState) -> bool:
        return bool(state.failed & {FIBRE_TENSION, FIBRE_COMPRESSION})

    def damaged(self, state: FailureState) -> bool:
        return bool(state.failed)


class MaxStressModel(_SuddenFailureModel):
    """Maximum stress criterion with ply discount."""

    name = "max_stress"

    def __init__(self, elastic: PlyElasticity, strengths: PlyStrengths) -> None:
        super().__init__(elastic)
        self.s = strengths.require(("xt", "xc", "yt", "yc", "s12"), self.name)

    def degradation(self, failed: frozenset[str]) -> tuple[float, float, float]:
        if failed & {FIBRE_TENSION, FIBRE_COMPRESSION}:
            return 1.0, 1.0, 1.0
        if failed:
            return 0.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    def failing(self, stress: FloatArray) -> list[str]:
        s1, s2, t12 = (float(v) for v in stress)
        s = self.s
        hits = (
            (FIBRE_TENSION, s1 >= s["xt"]),
            (FIBRE_COMPRESSION, -s1 >= s["xc"]),
            (MATRIX_TENSION, s2 >= s["yt"]),
            (MATRIX_COMPRESSION, -s2 >= s["yc"]),
            (SHEAR, abs(t12) >= s["s12"]),
        )
        return [mode for mode, hit in hits if hit]


def hashin_indices(stress: FloatArray, s: dict[str, float], alpha: float = 0.0) -> dict[str, float]:
    """Hashin's failure indices (>= 1: failure) of the modes active for the stress signs."""
    s1, s2, t12 = (float(v) for v in stress)
    shear = (t12 / s["s12"]) ** 2
    out = {}
    if s1 >= 0.0:
        out[FIBRE_TENSION] = (s1 / s["xt"]) ** 2 + alpha * shear
    else:
        out[FIBRE_COMPRESSION] = (s1 / s["xc"]) ** 2
    if s2 >= 0.0:
        out[MATRIX_TENSION] = (s2 / s["yt"]) ** 2 + shear
    else:
        s23 = s["s23"]
        out[MATRIX_COMPRESSION] = (
            (s2 / (2.0 * s23)) ** 2 + ((s["yc"] / (2.0 * s23)) ** 2 - 1.0) * s2 / s["yc"] + shear
        )
    return out


class HashinModel(_SuddenFailureModel):
    """Hashin criteria with mode-dependent residual stiffness (progressive degradation).

    ``alpha`` weighs the shear term of the fibre tension criterion: 0 (Hashin-Rotem, the
    default, avoids fibre failure predicted from shear alone) or 1 (Hashin 1980)."""

    name = "hashin"

    def __init__(
        self,
        elastic: PlyElasticity,
        strengths: PlyStrengths,
        alpha: float = 0.0,
        residual: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        super().__init__(elastic)
        self.s = strengths.require(("xt", "xc", "yt", "yc", "s12"), self.name)
        self.s["s23"] = strengths.s23 if strengths.s23 is not None else 0.5 * self.s["yc"]
        self.alpha = alpha
        self.residual = dict(HASHIN_RESIDUAL if residual is None else residual)

    def degradation(self, failed: frozenset[str]) -> tuple[float, float, float]:
        kept = [1.0, 1.0, 1.0]
        for mode in failed:
            for i, fraction in enumerate(self.residual[mode]):
                kept[i] *= fraction
        return 1.0 - kept[0], 1.0 - kept[1], 1.0 - kept[2]

    def failing(self, stress: FloatArray) -> list[str]:
        return [m for m, value in hashin_indices(stress, self.s, self.alpha).items() if value >= 1]


# -- continuum damage: gradual, energy-regularised softening --------------------------------
@dataclass(frozen=True)
class DamageState:
    """Per mode (``CDM_MODES``): the largest failure measure ``r = sqrt(F)`` reached, the work
    density of the mode at onset (``r = 1``, 0 before onset) and the damage."""

    r: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    onset_work: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    damage: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


class ContinuumDamageModel(PlyModel):
    """Hashin-initiated damage per mode with linear softening regularised by the crack band.

    For mode ``m`` with Hashin index ``F_m`` of the effective (undamaged) stress, ``r = sqrt(F)``
    grows linearly with the strain on a proportional path; damage starts at ``r = 1``, where
    the mode's work density (``sigma : eps`` of its components) is ``w0``, and follows

        d = r_f (r - 1) / (r (r_f - 1)),   r_f = 2 G_m / (L w0),

    so that the equivalent stress falls linearly to zero at ``r_f`` and the band dissipates
    ``G_m`` per unit area. ``r_f <= 1`` (``L`` above ``2 G_m E / X^2`` in uniaxial terms)
    would snap back; the softening is then instantaneous and a warning is reported.
    """

    name = "continuum_damage"

    def __init__(
        self,
        elastic: PlyElasticity,
        strengths: PlyStrengths,
        energies: FractureEnergies | None,
        characteristic_length: float,
        alpha: float = 0.0,
    ) -> None:
        if energies is None:
            raise ConfigError(
                "The continuum_damage model needs laminate.fracture_energies (fibre_tension, "
                "fibre_compression, matrix_tension, matrix_compression)."
            )
        if characteristic_length <= 0.0:
            raise ConfigError("laminate.characteristic_length must be positive.")
        self.elastic = elastic
        self.q0 = elastic.stiffness()
        self.s = strengths.require(("xt", "xc", "yt", "yc", "s12"), self.name)
        self.s["s23"] = strengths.s23 if strengths.s23 is not None else 0.5 * self.s["yc"]
        self.energies = energies
        self.length = characteristic_length
        self.alpha = alpha
        self._snap_back: set[str] = set()

    def initial_state(self) -> DamageState:
        return DamageState()

    def stress(self, strain: FloatArray, state: DamageState) -> tuple[FloatArray, Any]:
        effective = self.q0 @ strain
        e1, e2, g12 = (float(v) for v in strain)
        s1, s2, t12 = (float(v) for v in effective)
        indices = hashin_indices(effective, self.s, self.alpha)
        r, onset, damage = list(state.r), list(state.onset_work), list(state.damage)
        works = {
            FIBRE_TENSION: max(s1, 0.0) * max(e1, 0.0) + self.alpha * t12 * g12,
            FIBRE_COMPRESSION: max(-s1, 0.0) * max(-e1, 0.0),
            MATRIX_TENSION: max(s2, 0.0) * max(e2, 0.0) + t12 * g12,
            MATRIX_COMPRESSION: max(-s2, 0.0) * max(-e2, 0.0) + t12 * g12,
        }
        for k, mode in enumerate(CDM_MODES):
            if mode not in indices:
                continue
            measure = math.sqrt(max(indices[mode], 0.0))
            if measure <= r[k]:
                continue
            if onset[k] == 0.0 and measure >= 1.0:
                onset[k] = max(works[mode] / measure**2, 1e-300)  # work density at r = 1
            r[k] = measure
            if onset[k] > 0.0:
                damage[k] = self._damage(mode, r[k], onset[k])
        d1 = damage[0] if s1 >= 0.0 else damage[1]
        d2 = damage[2] if s2 >= 0.0 else damage[3]
        d6 = 1.0 - math.prod(1.0 - d for d in damage)
        q = self.elastic.stiffness(d1, d2, d6)
        return q @ strain, DamageState(tuple(r), tuple(onset), tuple(damage))

    def _damage(self, mode: str, r: float, onset_work: float) -> float:
        r_f = 2.0 * self.energies.of(mode) / (self.length * onset_work)
        if r_f <= 1.0 + 1e-9:
            self._snap_back.add(mode)
            return DAMAGE_CAP
        return min(r_f * (r - 1.0) / (r * (r_f - 1.0)), DAMAGE_CAP)

    def events(self, old: DamageState, new: DamageState) -> list[str]:
        out = []
        for k, mode in enumerate(CDM_MODES):
            if old.damage[k] == 0.0 and new.damage[k] > 0.0:
                out.append(f"{mode} damage onset")
            if old.damage[k] < DAMAGE_CAP and new.damage[k] >= DAMAGE_CAP:
                out.append(f"{mode} failure")
        return out

    def fibre_failed(self, state: DamageState) -> bool:
        return max(state.damage[0], state.damage[1]) >= DAMAGE_CAP

    def damaged(self, state: DamageState) -> bool:
        return max(state.damage) > 0.0

    def warnings(self) -> list[str]:
        return [
            f"{mode}: the characteristic length exceeds 2 G E / X^2, so the softening is "
            "instantaneous (snap-back); use a smaller characteristic_length"
            for mode in sorted(self._snap_back)
        ]


# -- RVE curves ------------------------------------------------------------------------------
@dataclass(frozen=True)
class CurveState:
    kappa_tension: float = 0.0  # largest transverse tensile (mechanical) strain reached
    kappa_compression: float = 0.0
    kappa_shear: float = 0.0
    failed: frozenset[str] = field(default_factory=frozenset)  # fibre modes


class CurveModel(PlyModel):
    """The ply curves of the RVE solves as secant laws with damage-like memory.

    * fibre direction: linear elastic up to ``Xt`` / ``Xc`` (inputs); a ply whose fibres fail
      carries no more load;
    * transverse: the RVE uniaxial-stress curve, driven by the transverse strain caused by the
      transverse stress; linear in compression unless a compression curve is given;
    * in-plane shear: the RVE shear curve;
    * after a curve's largest strain so far, unloading follows the secant to the origin, and
      beyond the last sample the stress stays at its last value ("curve exceeded" events).

    The transverse and shear responses are independent (no interaction), which is exact when
    each ply sees one mode, e.g. a [90] laminate in transverse tension.
    """

    name = "rve_curves"

    def __init__(self, ply: PlyProperties, strengths: PlyStrengths) -> None:
        compliance = ply.compliance[np.ix_([0, 1, 5], [0, 1, 5])]
        self.s11, self.s12 = float(compliance[0, 0]), float(compliance[0, 1])
        self.s22, self.s66 = float(compliance[1, 1]), float(compliance[2, 2])
        self.tension: PlyCurve | None = ply.transverse_tension
        self.compression: PlyCurve | None = ply.transverse_compression
        self.shear: PlyCurve | None = ply.shear
        self.xt, self.xc = strengths.xt, strengths.xc

    def initial_state(self) -> CurveState:
        return CurveState()

    def stress(self, strain: FloatArray, state: CurveState) -> tuple[FloatArray, Any]:
        if state.failed:
            return np.zeros(3), state
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
        return np.array([sigma1, sigma2, tau]), replace(
            state, kappa_tension=kappa_t, kappa_compression=kappa_c, kappa_shear=kappa_s
        )

    def failures(self, stress: FloatArray, state: CurveState) -> tuple[list[str], Any]:
        if state.failed:
            return [], state
        sigma1 = float(stress[0])
        if self.xt is not None and sigma1 >= self.xt:
            return [f"{FIBRE_TENSION} failure"], replace(state, failed=frozenset({FIBRE_TENSION}))
        if self.xc is not None and sigma1 <= -self.xc:
            mode = FIBRE_COMPRESSION
            return [f"{mode} failure"], replace(state, failed=frozenset({mode}))
        return [], state

    def events(self, old: CurveState, new: CurveState) -> list[str]:
        out = []
        for name, curve, before, after in (
            ("transverse tension", self.tension, old.kappa_tension, new.kappa_tension),
            ("transverse compression", self.compression, old.kappa_compression,
             new.kappa_compression),
            ("shear", self.shear, old.kappa_shear, new.kappa_shear),
        ):  # fmt: skip
            if curve is None:
                continue
            if curve.strain_at_peak < curve.max_strain and before <= curve.strain_at_peak < after:
                out.append(f"{name} peak passed")
            end = curve.max_strain * (1.0 + 1e-6)  # round-off at the end does not count
            if before <= end < after:
                out.append(f"{name} curve exceeded")
        return out

    def fibre_failed(self, state: CurveState) -> bool:
        return bool(state.failed)

    def damaged(self, state: CurveState) -> bool:
        return bool(state.failed) or any(
            curve is not None and kappa > curve.strain_at_peak
            for curve, kappa in (
                (self.tension, state.kappa_tension),
                (self.compression, state.kappa_compression),
                (self.shear, state.kappa_shear),
            )
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


def build_ply_model(
    name: str,
    ply: PlyProperties,
    characteristic_length: float = 1.0,
    hashin_alpha: float = 0.0,
) -> PlyModel:
    """The ply model ``name`` for ``ply`` (strengths: given values, else the RVE curves)."""
    elastic = PlyElasticity.from_compliance(ply.compliance)
    strengths = ply.resolved_strengths()
    if name == "elastic":
        return ElasticModel(elastic)
    if name == "rve_curves":
        return CurveModel(ply, strengths)
    if name == "max_stress":
        return MaxStressModel(elastic, strengths)
    if name == "hashin":
        return HashinModel(elastic, strengths, alpha=hashin_alpha)
    if name == "continuum_damage":
        return ContinuumDamageModel(
            elastic, strengths, ply.fracture_energies, characteristic_length, alpha=hashin_alpha
        )
    raise ConfigError(f"Unknown ply damage model {name!r}; use one of {', '.join(MODELS)}.")
