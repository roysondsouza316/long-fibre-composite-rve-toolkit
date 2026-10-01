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


def periodic_boundary_pairs(domain: Domain) -> list[PeriodicBoundaryPair]:
    """Opposite-face pairs of a periodic RVE: left/right and bottom/top in 2D; left/right,
    front/back and bottom/top in 3D. ``source`` is the mirror face, ``target`` its image."""
    if isinstance(domain, Domain2D):
        return [
            PeriodicBoundaryPair("x_periodic", "left", "right", (domain.width, 0.0)),
            PeriodicBoundaryPair("y_periodic", "bottom", "top", (0.0, domain.height)),
        ]
    return [
        PeriodicBoundaryPair("x_periodic", "left", "right", (domain.width, 0.0, 0.0)),
        PeriodicBoundaryPair("y_periodic", "front", "back", (0.0, domain.height, 0.0)),
        PeriodicBoundaryPair("z_periodic", "bottom", "top", (0.0, 0.0, domain.depth)),
    ]


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
        """Number of distinct fibres.

        Periodic images of a fibre that crosses the domain boundary share the ``fibre_id`` of
        that fibre, so they are counted once. Use :attr:`fibre_instance_count` for the number
        of fibre shapes (including images).
        """
        distinct: set[tuple[str, int]] = set()
        distinct.update(("circle", fibre.fibre_id) for fibre in self.circular_fibres)
        distinct.update(("polygon", fibre.fibre_id) for fibre in self.polygonal_fibres)
        distinct.update(("cylinder", fibre.fibre_id) for fibre in self.cylindrical_fibres)
        distinct.update(
            ("extruded_polygon", fibre.fibre_id) for fibre in self.extruded_polygonal_fibres
        )
        return len(distinct)

    @property
    def fibre_instance_count(self) -> int:
        """Number of fibre shapes, counting every periodic image separately."""
        return (
            len(self.circular_fibres)
            + len(self.polygonal_fibres)
            + len(self.cylindrical_fibres)
            + len(self.extruded_polygonal_fibres)
        )

    @property
    def fibre_measure(self) -> float:
        """Fibre area (2D) or volume (3D) inside the domain.

        Every fibre is clipped to the domain, exactly like the mesher does, so the value matches
        the meshed fibre phase for geometries whose fibres cross the domain boundary (image
        windows smaller than the image, periodic images of wrapped fibres).
        """
        domain = self.domain
        bounds = (domain.origin_x, domain.origin_y, domain.x_max, domain.y_max)
        circular_area = sum(
            circle_rectangle_intersection_area(
                fibre.center_x, fibre.center_y, fibre.radius, *bounds
            )
            for fibre in self.circular_fibres
        )
        polygon_area = sum(
            clipped_polygon_area(fibre.points, *bounds) for fibre in self.polygonal_fibres
        )
        cylinder_volume = 0.0
        extruded_polygon_volume = 0.0
        if isinstance(domain, Domain3D):
            cylinder_volume = sum(
                circle_rectangle_intersection_area(
                    fibre.center_x, fibre.center_y, fibre.radius, *bounds
                )
                * _clipped_length(fibre.z_min, fibre.z_max, domain.origin_z, domain.z_max)
                for fibre in self.cylindrical_fibres
            )
            extruded_polygon_volume = sum(
                clipped_polygon_area(fibre.points, *bounds)
                * _clipped_length(fibre.z_min, fibre.z_max, domain.origin_z, domain.z_max)
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


def circle_rectangle_intersection_area(
    center_x: float,
    center_y: float,
    radius: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> float:
    """Exact area of the intersection of a disk with an axis-aligned rectangle.

    A disk entirely inside the rectangle returns ``pi * radius**2`` exactly and a disk
    entirely outside returns 0.
    """
    if radius <= 0.0 or x_max <= x_min or y_max <= y_min:
        return 0.0
    if (
        center_x - radius >= x_min
        and center_x + radius <= x_max
        and center_y - radius >= y_min
        and center_y + radius <= y_max
    ):
        return float(np.pi * radius**2)
    if (
        center_x + radius <= x_min
        or center_x - radius >= x_max
        or center_y + radius <= y_min
        or center_y - radius >= y_max
    ):
        return 0.0
    area = _disk_box_area(
        x_min - center_x,
        x_max - center_x,
        y_min - center_y,
        y_max - center_y,
        radius,
    )
    return float(min(max(area, 0.0), np.pi * radius**2))


def _disk_box_area(x0: float, x1: float, y0: float, y1: float, radius: float) -> float:
    # Area of [x0, x1] x [y0, y1] intersected with the disk of the given radius centred at the
    # origin; y0 <= y1 is split at y = 0 so that the half-strip integral below applies.
    if y0 < 0.0:
        if y1 < 0.0:
            return _disk_box_area(x0, x1, -y1, -y0, radius)
        return _disk_box_area(x0, x1, 0.0, -y0, radius) + _disk_box_area(
            x0, x1, 0.0, y1, radius
        )
    return _disk_strip_area(x0, x1, y0, radius) - _disk_strip_area(x0, x1, y1, radius)


def _disk_strip_area(x0: float, x1: float, height: float, radius: float) -> float:
    # Area of the disk part with x0 <= x <= x1 and y >= height (height >= 0).
    if height >= radius:
        return 0.0
    half_chord = float(np.sqrt(radius * radius - height * height))
    lower = min(max(x0, -half_chord), half_chord)
    upper = min(max(x1, -half_chord), half_chord)
    return _strip_antiderivative(upper, height, radius) - _strip_antiderivative(
        lower, height, radius
    )


def _strip_antiderivative(x: float, height: float, radius: float) -> float:
    # Antiderivative of sqrt(radius^2 - x^2) - height.
    ratio = min(max(x / radius, -1.0), 1.0)
    root = float(np.sqrt(max(radius * radius - x * x, 0.0)))
    return 0.5 * (x * root + radius * radius * float(np.arcsin(ratio))) - height * x


def clip_polygon_to_rectangle(
    points: FloatArray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> FloatArray:
    """Clip a polygon (open or closed point list) to an axis-aligned rectangle.

    Uses Sutherland-Hodgman clipping. For a non-convex polygon whose intersection with the
    rectangle has several parts the result may contain zero-width bridges along the rectangle
    edges; its signed area is still the area of the intersection. The returned polygon is
    open (the first point is not repeated) and empty when nothing is inside.
    """
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if polygon.shape[0] > 1 and np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    for axis, bound, keep_greater in (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    ):
        if polygon.shape[0] == 0:
            break
        polygon = _clip_polygon_half_plane(polygon, axis, bound, keep_greater)
    return polygon


def _clip_polygon_half_plane(
    polygon: FloatArray,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> FloatArray:
    coordinate = polygon[:, axis]
    inside = coordinate >= bound if keep_greater else coordinate <= bound
    if bool(np.all(inside)):
        return polygon
    if not bool(np.any(inside)):
        return np.empty((0, 2), dtype=np.float64)
    output: list[FloatArray] = []
    count = polygon.shape[0]
    for index in range(count):
        current = polygon[index]
        following = polygon[(index + 1) % count]
        current_inside = bool(inside[index])
        following_inside = bool(inside[(index + 1) % count])
        if current_inside:
            output.append(current)
        if current_inside != following_inside:
            fraction = (bound - current[axis]) / (following[axis] - current[axis])
            crossing = current + fraction * (following - current)
            crossing[axis] = bound
            output.append(crossing)
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def clipped_polygon_area(
    points: FloatArray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> float:
    """Area of the part of a polygon inside an axis-aligned rectangle."""
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return 0.0
    if (
        float(polygon[:, 0].min()) >= x_min
        and float(polygon[:, 0].max()) <= x_max
        and float(polygon[:, 1].min()) >= y_min
        and float(polygon[:, 1].max()) <= y_max
    ):
        return abs(_signed_polygon_area(polygon))
    clipped = clip_polygon_to_rectangle(polygon, x_min, y_min, x_max, y_max)
    if clipped.shape[0] < 3:
        return 0.0
    return abs(_signed_polygon_area(clipped))


def _clipped_length(start: float, end: float, lower: float, upper: float) -> float:
    if start >= lower and end <= upper:
        return end - start
    return max(0.0, min(end, upper) - max(start, lower))
