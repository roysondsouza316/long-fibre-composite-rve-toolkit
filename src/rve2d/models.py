from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Domain2D:
    width: float
    height: float
    origin_x: float = 0.0
    origin_y: float = 0.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x_max(self) -> float:
        return self.origin_x + self.width

    @property
    def y_max(self) -> float:
        return self.origin_y + self.height


@dataclass(frozen=True)
class Domain3D:
    width: float
    height: float
    depth: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def x_max(self) -> float:
        return self.origin_x + self.width

    @property
    def y_max(self) -> float:
        return self.origin_y + self.height

    @property
    def z_max(self) -> float:
        return self.origin_z + self.depth


@dataclass(frozen=True)
class CircleFibre:
    center_x: float
    center_y: float
    radius: float
    fibre_id: int


@dataclass(frozen=True)
class PolygonFibre:
    points: FloatArray
    fibre_id: int


@dataclass(frozen=True)
class CylinderFibre:
    center_x: float
    center_y: float
    radius: float
    z_min: float
    z_max: float
    fibre_id: int


@dataclass(frozen=True)
class ExtrudedPolygonFibre:
    points: FloatArray
    z_min: float
    z_max: float
    fibre_id: int


@dataclass(frozen=True)
class PeriodicBoundaryPair:
    name: str
    source: str
    target: str
    translation: tuple[float, ...]


Domain: TypeAlias = Domain2D | Domain3D


@dataclass
class GeometryModel:
    domain: Domain
    circular_fibres: list[CircleFibre] = field(default_factory=list)
    polygonal_fibres: list[PolygonFibre] = field(default_factory=list)
    cylindrical_fibres: list[CylinderFibre] = field(default_factory=list)
    extruded_polygonal_fibres: list[ExtrudedPolygonFibre] = field(default_factory=list)
    phase_labels: dict[str, int] = field(default_factory=lambda: {"matrix": 1, "fibre": 2})
    boundary_labels: dict[str, int] = field(
        default_factory=lambda: {"left": 11, "right": 12, "bottom": 13, "top": 14}
    )
    periodic_pairs: list[PeriodicBoundaryPair] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return 2 if isinstance(self.domain, Domain2D) else 3

    @property
    def fibre_count(self) -> int:
        return (
            len(self.circular_fibres)
            + len(self.polygonal_fibres)
            + len(self.cylindrical_fibres)
            + len(self.extruded_polygonal_fibres)
        )

    @property
    def fibre_measure(self) -> float:
        circular_area = sum(np.pi * fibre.radius**2 for fibre in self.circular_fibres)
        polygon_area = sum(
            abs(_signed_polygon_area(fibre.points)) for fibre in self.polygonal_fibres
        )
        cylinder_volume = sum(
            np.pi * fibre.radius**2 * (fibre.z_max - fibre.z_min)
            for fibre in self.cylindrical_fibres
        )
        extruded_polygon_volume = sum(
            abs(_signed_polygon_area(fibre.points)) * (fibre.z_max - fibre.z_min)
            for fibre in self.extruded_polygonal_fibres
        )
        return float(circular_area + polygon_area + cylinder_volume + extruded_polygon_volume)

    @property
    def fibre_area(self) -> float:
        if self.dimension != 2:
            raise ValueError("fibre_area is only defined for 2D geometries.")
        return self.fibre_measure

    @property
    def matrix_area(self) -> float:
        if not isinstance(self.domain, Domain2D):
            raise ValueError("matrix_area is only defined for 2D geometries.")
        return self.domain.area - self.fibre_area

    @property
    def fibre_volume(self) -> float:
        if self.dimension != 3:
            raise ValueError("fibre_volume is only defined for 3D geometries.")
        return self.fibre_measure

    @property
    def matrix_volume(self) -> float:
        if not isinstance(self.domain, Domain3D):
            raise ValueError("matrix_volume is only defined for 3D geometries.")
        return self.domain.volume - self.fibre_volume

    @property
    def fibre_volume_fraction(self) -> float:
        if isinstance(self.domain, Domain2D):
            if self.domain.area <= 0.0:
                return 0.0
            return self.fibre_area / self.domain.area
        if self.domain.volume <= 0.0:
            return 0.0
        return self.fibre_volume / self.domain.volume


def _signed_polygon_area(points: FloatArray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
