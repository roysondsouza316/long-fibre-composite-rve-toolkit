"""Ply fixtures shared by the laminate tests."""

from __future__ import annotations

import numpy as np

from rve2d.engines.common.materials import PhaseElasticity, compliance_3d, constitutive_matrix
from rve2d.laminate import PlyProperties, PlyStrengths

CARBON = {  # a carbon/epoxy-like transversely isotropic ply, MPa
    "e1": 140e3,
    "e2": 10e3,
    "e3": 10e3,
    "g12": 5e3,
    "g13": 5e3,
    "g23": 10e3 / (2 * 1.45),
    "nu12": 0.3,
    "nu13": 0.3,
    "nu23": 0.45,
}


def carbon_stiffness() -> np.ndarray:
    phase = PhaseElasticity("orthotropic", 0.0, 0.0, **CARBON)
    return np.asarray(np.linalg.inv(compliance_3d(phase)))


def carbon_ply(strengths: PlyStrengths | None = None, **curves: object) -> PlyProperties:
    return PlyProperties(
        stiffness=carbon_stiffness(),
        strengths=strengths or PlyStrengths(),
        **curves,  # type: ignore[arg-type]
    )


def isotropic_ply(e: float = 3500.0, nu: float = 0.35) -> PlyProperties:
    stiffness = constitutive_matrix(PhaseElasticity("isotropic", e, nu), "solid")
    return PlyProperties(stiffness=stiffness)
