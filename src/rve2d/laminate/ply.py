"""Ply (lamina) properties: elastic stiffness and stress-strain curves from RVE solves.

The RVEs have their fibres along z. A ply has its fibres along axis 1, the in-plane
transverse direction 2 and the through-thickness direction 3, so the RVE axes (z, x, y)
become the ply axes (1, 2, 3). Voigt order in ply axes: (11, 22, 33, 23, 13, 12) with
engineering shear strains.

* The elastic stiffness comes from the linear homogenization (6x6, so kinematics
  ``generalized_plane_strain`` in 2D or ``solid`` in 3D). A finite random RVE is only
  approximately transversely isotropic; by default the stiffness is averaged over rotations
  about the fibre axis, which makes it exactly transversely isotropic and removes the
  arbitrary choice of which cross-section axis is called 2.
* The stress-strain curves come from nonlinear RVE solves under uniaxial stress: transverse
  (ply 22 = RVE xx) and in-plane shear (ply 12 = RVE xz).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rve2d.engines.common.materials import rotation_matrix, voigt_rotation
from rve2d.exceptions import ConfigError

FloatArray = NDArray[np.float64]

PLY_VOIGT = ("11", "22", "33", "23", "13", "12")
# ply Voigt component i is RVE Voigt component RVE_TO_PLY[i] (RVE order xx yy zz yz xz xy)
RVE_TO_PLY = (2, 0, 1, 5, 3, 4)
# RVE load component that produces each ply curve
CURVE_COMPONENTS = {"transverse_tension": "xx", "transverse_compression": "xx", "shear": "xz"}


@dataclass(frozen=True)
class PlyCurve:
    """Stress against strain of one ply component under uniaxial stress, from zero load.

    Samples are taken in loading order (``strain`` increasing in magnitude); between samples
    the response is linear, beyond the last sample the stress stays at its last value.
    """

    strain: FloatArray
    stress: FloatArray
    source: str = ""

    def __post_init__(self) -> None:
        strain = np.asarray(self.strain, dtype=np.float64)
        stress = np.asarray(self.stress, dtype=np.float64)
        if strain.ndim != 1 or strain.shape != stress.shape or strain.size < 2:
            raise ConfigError("A ply curve needs matching strain and stress samples (two or more).")
        order = np.argsort(np.abs(strain), kind="stable")
        object.__setattr__(self, "strain", np.abs(strain[order]))
        object.__setattr__(self, "stress", np.abs(stress[order]))

    def stress_at(self, strain: float) -> float:
        """Stress magnitude at ``|strain|`` (the curve is used symmetrically)."""
        return float(np.interp(abs(strain), self.strain, self.stress))

    @property
    def peak_stress(self) -> float:
        return float(self.stress.max())

    @property
    def strain_at_peak(self) -> float:
        return float(self.strain[int(np.argmax(self.stress))])

    @property
    def max_strain(self) -> float:
        return float(self.strain[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "strain": self.strain.tolist(),
            "stress": self.stress.tolist(),
            "source": self.source,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PlyCurve:
        return PlyCurve(
            np.asarray(data["strain"], dtype=np.float64),
            np.asarray(data["stress"], dtype=np.float64),
            str(data.get("source", "")),
        )


@dataclass(frozen=True)
class PlyProperties:
    """A unidirectional ply: elastic stiffness in ply axes, optional curves and strengths."""

    stiffness: FloatArray  # 6x6, ply axes (11, 22, 33, 23, 13, 12)
    transverse_tension: PlyCurve | None = None
    transverse_compression: PlyCurve | None = None
    shear: PlyCurve | None = None
    longitudinal_tensile_strength: float | None = None
    longitudinal_compressive_strength: float | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def compliance(self) -> FloatArray:
        return np.asarray(np.linalg.inv(self.stiffness))

    def engineering_constants(self) -> dict[str, float]:
        s = self.compliance
        e1, e2, e3 = 1.0 / s[0, 0], 1.0 / s[1, 1], 1.0 / s[2, 2]
        return {
            "e1": float(e1),
            "e2": float(e2),
            "e3": float(e3),
            "g23": float(1.0 / s[3, 3]),
            "g13": float(1.0 / s[4, 4]),
            "g12": float(1.0 / s[5, 5]),
            "nu12": float(-s[0, 1] * e1),
            "nu13": float(-s[0, 2] * e1),
            "nu23": float(-s[1, 2] * e2),
        }

    def to_dict(self) -> dict[str, Any]:
        def curve(c: PlyCurve | None) -> dict[str, Any] | None:
            return None if c is None else c.to_dict()

        return {
            "axes": "1 = fibre, 2 = in-plane transverse, 3 = through-thickness",
            "voigt_components": list(PLY_VOIGT),
            "stiffness": self.stiffness.tolist(),
            "engineering_constants": self.engineering_constants(),
            "transverse_tension": curve(self.transverse_tension),
            "transverse_compression": curve(self.transverse_compression),
            "shear": curve(self.shear),
            "longitudinal_tensile_strength": self.longitudinal_tensile_strength,
            "longitudinal_compressive_strength": self.longitudinal_compressive_strength,
            "source": self.source,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PlyProperties:
        def curve(key: str) -> PlyCurve | None:
            value = data.get(key)
            return None if value is None else PlyCurve.from_dict(value)

        return PlyProperties(
            stiffness=np.asarray(data["stiffness"], dtype=np.float64),
            transverse_tension=curve("transverse_tension"),
            transverse_compression=curve("transverse_compression"),
            shear=curve("shear"),
            longitudinal_tensile_strength=data.get("longitudinal_tensile_strength"),
            longitudinal_compressive_strength=data.get("longitudinal_compressive_strength"),
            source=dict(data.get("source", {})),
        )

    @staticmethod
    def load(path: str | Path) -> PlyProperties:
        return PlyProperties.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def ply_stiffness_from_rve(
    rve_stiffness: FloatArray | list[list[float]], transverse_isotropy: bool = True
) -> FloatArray:
    """Ply stiffness (ply axes) from the 6x6 effective stiffness of an RVE with fibres along z."""
    c_rve = np.asarray(rve_stiffness, dtype=np.float64)
    if c_rve.shape != (6, 6):
        raise ConfigError(
            "Ply properties need the full 6x6 RVE stiffness: use kinematics "
            "generalized_plane_strain (2D) or solid (3D)."
        )
    index = list(RVE_TO_PLY)
    c_ply = np.ascontiguousarray(c_rve[np.ix_(index, index)])
    c_ply = 0.5 * (c_ply + c_ply.T)
    return transversely_isotropic_average(c_ply) if transverse_isotropy else c_ply


def transversely_isotropic_average(stiffness: FloatArray, samples: int = 36) -> FloatArray:
    """Average of ``stiffness`` over rotations about axis 1 (exact for ``samples`` > 4)."""
    total = np.zeros((6, 6))
    for angle in np.linspace(0.0, 360.0, samples, endpoint=False):
        t = voigt_rotation(rotation_matrix((float(angle), 0.0, 0.0)))
        total += t @ stiffness @ t.T
    average = total / samples
    return np.asarray(0.5 * (average + average.T))


def transverse_anisotropy(stiffness: FloatArray) -> float:
    """Relative difference between the stiffness and its transversely isotropic average."""
    average = transversely_isotropic_average(stiffness)
    return float(np.abs(stiffness - average).max() / np.abs(average).max())


def ply_from_homogenization(
    summary: dict[str, Any] | str | Path, transverse_isotropy: bool = True
) -> PlyProperties:
    """Elastic ply properties from a ``homogenization_summary.json`` (or its contents)."""
    if isinstance(summary, dict):
        data, origin = summary, str(summary.get("_path", ""))
    else:
        origin = str(summary)
        data = json.loads(Path(summary).read_text(encoding="utf-8"))
    kinematics = data.get("kinematics")
    if kinematics not in ("generalized_plane_strain", "solid"):
        raise ConfigError(
            f"Ply properties need the full 6x6 stiffness; the RVE was solved with {kinematics}. "
            "Use kinematics generalized_plane_strain (2D) or solid (3D)."
        )
    raw = ply_stiffness_from_rve(data["homogenized_stiffness_voigt"], transverse_isotropy=False)
    stiffness = transversely_isotropic_average(raw) if transverse_isotropy else raw
    return PlyProperties(
        stiffness=stiffness,
        source={
            "homogenization_summary": origin,
            "engine": data.get("engine"),
            "kinematics": kinematics,
            "fibre_volume_fraction": data.get("fibre_volume_fraction"),
            "transversely_isotropic_average": transverse_isotropy,
            "rve_transverse_anisotropy": transverse_anisotropy(raw),
        },
    )


def curve_from_response(path: str | Path, component: str) -> PlyCurve:
    """Ply curve from a ``nonlinear_response.csv`` (RVE macro strain and stress of
    ``component``)."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ConfigError(f"{path} has no response records.")
    strain = np.array([float(row[f"e_{component}"]) for row in rows])
    stress = np.array([float(row[f"s_{component}"]) for row in rows])
    if strain[0] != 0.0:
        strain, stress = np.concatenate([[0.0], strain]), np.concatenate([[0.0], stress])
    return PlyCurve(strain, stress, source=str(path))
