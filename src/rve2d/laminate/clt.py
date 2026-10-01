"""Laminate stiffness from ply properties and a stacking sequence.

Two homogenizations of the same stack of perfectly bonded plies (laminate axes: x the
reference in-plane direction, y in plane, z the stacking direction):

* **Classical lamination theory (CLT)**: plies in plane stress, Kirchhoff plate kinematics.
  ``A``, ``B``, ``D`` relate the force and moment resultants to the mid-plane strains and
  curvatures; the membrane and flexural engineering constants follow from the inverse of the
  ABD matrix (bending-extension coupling included, i.e. the laminate is free to curve).
* **3D effective stiffness** of the stack (a periodic laminate, or a thick laminate far
  from its faces): in-plane strains and out-of-plane stresses are the same in every ply.
  The 6x6 result is exact for this layered microstructure; its in-plane constants under
  in-plane loading equal the CLT membrane constants of a symmetric laminate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rve2d.engines.common.homogenization import engineering_constants
from rve2d.engines.common.materials import rotate_stiffness
from rve2d.exceptions import ConfigError
from rve2d.laminate.ply import PlyProperties
from rve2d.laminate.stacking import format_stacking_sequence, parse_stacking_sequence

FloatArray = NDArray[np.float64]

IN_PLANE = [0, 1, 5]  # xx, yy, xy in the Voigt order (xx, yy, zz, yz, xz, xy)
OUT_OF_PLANE = [2, 3, 4]  # zz, yz, xz


@dataclass(frozen=True)
class Laminate:
    """A stack of plies of one material, bottom to top."""

    ply: PlyProperties
    angles: tuple[float, ...]  # degrees from the laminate x axis, about z
    thicknesses: tuple[float, ...]
    name: str = ""

    @staticmethod
    def from_sequence(
        ply: PlyProperties, sequence: str | list[float], ply_thickness: float
    ) -> Laminate:
        if ply_thickness <= 0.0:
            raise ConfigError("ply_thickness must be positive.")
        angles = tuple(parse_stacking_sequence(sequence))
        name = sequence if isinstance(sequence, str) else format_stacking_sequence(list(angles))
        return Laminate(ply, angles, tuple(ply_thickness for _ in angles), name)

    @property
    def thickness(self) -> float:
        return float(sum(self.thicknesses))

    @property
    def interfaces(self) -> FloatArray:
        """z of the ply boundaries, from -h/2 (bottom) to h/2 (top)."""
        return np.concatenate([[0.0], np.cumsum(self.thicknesses)]) - 0.5 * self.thickness

    def ply_stiffness(self, angle: float) -> FloatArray:
        """3D stiffness of a ply at ``angle`` in laminate axes (ply axes rotated about z)."""
        return np.asarray(rotate_stiffness(self.ply.stiffness, (0.0, 0.0, float(angle))))

    def reduced_stiffness(self, angle: float) -> FloatArray:
        """Plane-stress stiffness ``Qbar`` (xx, yy, xy) of a ply at ``angle``."""
        compliance = np.linalg.inv(self.ply_stiffness(angle))
        return np.asarray(np.linalg.inv(compliance[np.ix_(IN_PLANE, IN_PLANE)]))


def abd_matrix(laminate: Laminate) -> FloatArray:
    """6x6 ABD matrix: ``[N; M] = [[A, B], [B, D]] [eps0; kappa]`` (engineering shear)."""
    z = laminate.interfaces
    a, b, d = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    for k, angle in enumerate(laminate.angles):
        q = laminate.reduced_stiffness(angle)
        a += q * (z[k + 1] - z[k])
        b += q * (z[k + 1] ** 2 - z[k] ** 2) / 2.0
        d += q * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
    return np.block([[a, b], [b, d]])


def laminate_constants(laminate: Laminate) -> dict[str, float | bool]:
    """Membrane and flexural engineering constants of the laminate (from inverse ABD)."""
    abd = abd_matrix(laminate)
    h = laminate.thickness
    inverse = np.linalg.inv(abd)
    a, d = inverse[:3, :3], inverse[3:, 3:]
    scale = np.abs(abd[:3, :3]).max()
    b_coupling = float(np.abs(abd[:3, 3:]).max() / (scale * h))
    return {
        "thickness": h,
        "plies": len(laminate.angles),
        "ex": float(1.0 / (h * a[0, 0])),
        "ey": float(1.0 / (h * a[1, 1])),
        "gxy": float(1.0 / (h * a[2, 2])),
        "nuxy": float(-a[0, 1] / a[0, 0]),
        "nuyx": float(-a[0, 1] / a[1, 1]),
        "eta_xy_x": float(a[0, 2] / a[0, 0]),  # shear strain per axial strain under Nx
        "eta_xy_y": float(a[1, 2] / a[1, 1]),
        "flexural_ex": float(12.0 / (h**3 * d[0, 0])),
        "flexural_ey": float(12.0 / (h**3 * d[1, 1])),
        "flexural_gxy": float(12.0 / (h**3 * d[2, 2])),
        "flexural_nuxy": float(-d[0, 1] / d[0, 0]),
        "symmetric": bool(b_coupling < 1e-12),
        "balanced": bool(abs(abd[0, 2]) + abs(abd[1, 2]) < 1e-12 * scale),
    }


def effective_3d_stiffness(laminate: Laminate) -> FloatArray:
    """Exact 6x6 stiffness (laminate axes) of the stack with uniform in-plane strains and
    out-of-plane stresses (thickness-weighted averages of the mixed ply relations)."""
    weights = np.asarray(laminate.thicknesses) / laminate.thickness
    i, o = IN_PLANE, OUT_OF_PLANE
    m_ii = np.zeros((3, 3))
    m_io = np.zeros((3, 3))
    m_oi = np.zeros((3, 3))
    m_oo = np.zeros((3, 3))
    for weight, angle in zip(weights, laminate.angles, strict=True):
        c = laminate.ply_stiffness(angle)
        c_oo_inv = np.linalg.inv(c[np.ix_(o, o)])
        # [sigma_i; eps_o] = [[c_ii - c_io c_oo^-1 c_oi, c_io c_oo^-1], [-c_oo^-1 c_oi, c_oo^-1]]
        #                    [eps_i; sigma_o]
        m_ii += weight * (c[np.ix_(i, i)] - c[np.ix_(i, o)] @ c_oo_inv @ c[np.ix_(o, i)])
        m_io += weight * (c[np.ix_(i, o)] @ c_oo_inv)
        m_oi += weight * (-c_oo_inv @ c[np.ix_(o, i)])
        m_oo += weight * c_oo_inv
    m_oo_inv = np.linalg.inv(m_oo)
    effective = np.zeros((6, 6))
    effective[np.ix_(i, i)] = m_ii - m_io @ m_oo_inv @ m_oi
    effective[np.ix_(i, o)] = m_io @ m_oo_inv
    effective[np.ix_(o, i)] = -m_oo_inv @ m_oi
    effective[np.ix_(o, o)] = m_oo_inv
    return np.asarray(0.5 * (effective + effective.T))


def effective_3d_constants(laminate: Laminate) -> dict[str, float]:
    """Engineering constants of the 3D effective laminate stiffness (z through thickness)."""
    return engineering_constants(effective_3d_stiffness(laminate), "solid")
