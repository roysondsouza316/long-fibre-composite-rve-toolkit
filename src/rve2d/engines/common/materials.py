"""Phase constitutive matrices for the linear homogenization (shared by both engines).

Matrices use Voigt notation ``(xx, yy, zz, yz, xz, xy)`` with engineering shear strains,
restricted to the components that are active for the kinematics:

* ``plane_stress`` (2D): the reduced stiffness ``Q`` on ``(xx, yy, xy)``; orthotropic phases
  are given by their in-plane constants and rotated in plane (``material_angle_deg``).
* ``plane_strain`` (2D): rows/columns ``(xx, yy, xy)`` of the 3D stiffness (``eps_zz = 0``).
* ``generalized_plane_strain`` (2D) and ``solid`` (3D): the full 3D 6x6 stiffness.

For every kinematics except plane stress, orthotropic phases need all nine 3D constants and
are rotated in 3D (``material_angle_x/y/z_deg``, applied as ``Rz Ry Rx``); a 2D
``material_angle_deg`` is a rotation about z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from rve2d.config import FerriteSolveConfig
from rve2d.exceptions import ConfigError

FloatArray = NDArray[np.float64]

VOIGT_NAMES = ("xx", "yy", "zz", "yz", "xz", "xy")
ACTIVE_VOIGT: dict[str, tuple[int, ...]] = {
    "plane_stress": (0, 1, 5),
    "plane_strain": (0, 1, 5),
    "generalized_plane_strain": (0, 1, 2, 3, 4, 5),
    "solid": (0, 1, 2, 3, 4, 5),
}


@dataclass(frozen=True)
class PhaseElasticity:
    """Elastic constants of one phase, as given in the ``solver`` config section."""

    model: str
    youngs_modulus: float
    poisson_ratio: float
    e1: float | None = None
    e2: float | None = None
    e3: float | None = None
    g12: float | None = None
    g13: float | None = None
    g23: float | None = None
    nu12: float | None = None
    nu13: float | None = None
    nu23: float | None = None


def phase_elasticity(solver: FerriteSolveConfig, phase: str) -> PhaseElasticity:
    """Read the elastic constants of ``phase`` (``"matrix"`` or ``"fibre"``) from the config."""
    return PhaseElasticity(
        model=getattr(solver, f"{phase}_material_model"),
        youngs_modulus=getattr(solver, f"{phase}_youngs_modulus"),
        poisson_ratio=getattr(solver, f"{phase}_poisson_ratio"),
        **{
            name: getattr(solver, f"{phase}_{name}")
            for name in ("e1", "e2", "e3", "g12", "g13", "g23", "nu12", "nu13", "nu23")
        },
    )


def constitutive_matrix(
    phase: PhaseElasticity,
    kinematics: str,
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Stiffness of a phase in the active Voigt basis of ``kinematics``."""
    if kinematics == "plane_stress":
        return _rotate_plane_stress(_plane_stress_stiffness(phase), rotation_deg[2])
    stiffness = _rotate_3d(np.asarray(np.linalg.inv(compliance_3d(phase))), rotation_deg)
    active = list(ACTIVE_VOIGT[kinematics])
    return np.ascontiguousarray(stiffness[np.ix_(active, active)])


def compliance_3d(phase: PhaseElasticity) -> FloatArray:
    """3D compliance in material axes (engineering shear)."""
    if phase.model == "isotropic":
        e, nu = phase.youngs_modulus, phase.poisson_ratio
        g = e / (2.0 * (1.0 + nu))
        e1 = e2 = e3 = e
        nu12 = nu13 = nu23 = nu
        g12 = g13 = g23 = g
    else:
        values = [phase.e1, phase.e2, phase.e3, phase.g12, phase.g13, phase.g23]
        values += [phase.nu12, phase.nu13, phase.nu23]
        if any(v is None for v in values):
            raise ConfigError(
                "Orthotropic phases need e1, e2, e3, g12, g13, g23, nu12, nu13 and nu23 for "
                "plane strain, generalized plane strain and 3D solves."
            )
        e1, e2, e3, g12, g13, g23, nu12, nu13, nu23 = (float(cast(float, v)) for v in values)
    return np.array(
        [
            [1.0 / e1, -nu12 / e1, -nu13 / e1, 0.0, 0.0, 0.0],
            [-nu12 / e1, 1.0 / e2, -nu23 / e2, 0.0, 0.0, 0.0],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / g23, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / g13, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g12],
        ]
    )


def _plane_stress_stiffness(phase: PhaseElasticity) -> FloatArray:
    if phase.model == "isotropic":
        e, nu = phase.youngs_modulus, phase.poisson_ratio
        e1 = e2 = e
        nu12 = nu
        g12 = e / (2.0 * (1.0 + nu))
    else:
        if phase.e1 is None or phase.e2 is None or phase.g12 is None or phase.nu12 is None:
            raise ConfigError("Orthotropic plane-stress phases need e1, e2, g12 and nu12.")
        e1, e2, g12, nu12 = phase.e1, phase.e2, phase.g12, phase.nu12
    compliance = np.array(
        [
            [1.0 / e1, -nu12 / e1, 0.0],
            [-nu12 / e1, 1.0 / e2, 0.0],
            [0.0, 0.0, 1.0 / g12],
        ]
    )
    return np.asarray(np.linalg.inv(compliance))


def _rotate_plane_stress(stiffness: FloatArray, angle_deg: float) -> FloatArray:
    """Rotate the reduced stiffness ``Q`` in plane by ``angle_deg`` (material axis 1 at that
    angle from x)."""
    if abs(angle_deg) < 1e-12:
        return stiffness
    full = np.zeros((6, 6))
    index = [0, 1, 5]
    full[np.ix_(index, index)] = stiffness
    # Out-of-plane entries do not mix with in-plane ones under a rotation about z.
    rotated = _rotate_3d(full, (0.0, 0.0, angle_deg))
    return np.ascontiguousarray(rotated[np.ix_(index, index)])


def rotation_matrix(rotation_deg: tuple[float, float, float]) -> FloatArray:
    """``Rz @ Ry @ Rx`` for angles about x, y and z in degrees (material to global axes)."""
    ax, ay, az = (np.deg2rad(angle) for angle in rotation_deg)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(rz @ ry @ rx)


def _rotate_3d(stiffness: FloatArray, rotation_deg: tuple[float, float, float]) -> FloatArray:
    """Stiffness in global axes for a material whose axes are rotated by ``rotation_deg``."""
    if max(abs(angle) for angle in rotation_deg) < 1e-12:
        return stiffness
    t = _voigt_rotation(rotation_matrix(rotation_deg))
    # Engineering-shear Voigt: global stress = T_s sigma_local, local strain = T_s^T eps_global
    return np.asarray(t @ stiffness @ t.T)


def _voigt_rotation(r: FloatArray) -> FloatArray:
    """6x6 stress transformation ``sigma_global = T sigma_local`` for ``x_global = r x_local``
    (tensor shear stresses; the strain transformation with engineering shears is ``T^-T``)."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    t = np.zeros((6, 6))
    for row, (i, j) in enumerate(pairs):
        for col, (k, m) in enumerate(pairs):
            if k == m:
                t[row, col] = r[i, k] * r[j, m]
            else:
                t[row, col] = r[i, k] * r[j, m] + r[i, m] * r[j, k]
    return t
