from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rve2d.models import CircleFibre, CylinderFibre, Domain2D, Domain3D, GeometryModel


@dataclass(frozen=True)
class QualityReport:
    valid: bool
    minimum_spacing: float | None
    clipped_fibres: int
    disconnected_artifacts: int
    warnings: list[str] = field(default_factory=list)


def validate_geometry(
    geometry: GeometryModel,
    minimum_spacing_requirement: float = 0.0,
    disconnected_artifacts: int = 0,
) -> QualityReport:
    minimum_spacing = _minimum_spacing(geometry)
    clipped_fibres = _count_clipped_fibres(geometry)
    warnings: list[str] = []
    if minimum_spacing is not None and minimum_spacing < minimum_spacing_requirement:
        warnings.append(
            f"Minimum spacing {minimum_spacing:.6f} is below the requirement "
            f"{minimum_spacing_requirement:.6f}."
        )
    if clipped_fibres > 0:
        warnings.append(f"{clipped_fibres} fibres intersect the domain boundary.")
    if disconnected_artifacts > 1:
        warnings.append(
            f"{disconnected_artifacts} disconnected fibre regions detected in the input mask."
        )

    valid = not warnings
    return QualityReport(
        valid=valid,
        minimum_spacing=minimum_spacing,
        clipped_fibres=clipped_fibres,
        disconnected_artifacts=disconnected_artifacts,
        warnings=warnings,
    )


def _minimum_spacing(geometry: GeometryModel) -> float | None:
    if geometry.dimension == 3:
        return _minimum_spacing_cylindrical(geometry.cylindrical_fibres)
    return _minimum_spacing_circular(geometry.circular_fibres)


def _minimum_spacing_circular(fibres: list[CircleFibre]) -> float | None:
    if len(fibres) < 2:
        return None
    minimum_spacing: float | None = None
    for index, fibre_a in enumerate(fibres):
        for fibre_b in fibres[index + 1 :]:
            center_distance = float(
                np.hypot(
                    fibre_a.center_x - fibre_b.center_x,
                    fibre_a.center_y - fibre_b.center_y,
                )
            )
            edge_distance = center_distance - fibre_a.radius - fibre_b.radius
            minimum_spacing = (
                edge_distance
                if minimum_spacing is None
                else min(minimum_spacing, edge_distance)
            )
    return minimum_spacing


def _minimum_spacing_cylindrical(fibres: list[CylinderFibre]) -> float | None:
    if len(fibres) < 2:
        return None
    minimum_spacing: float | None = None
    for index, fibre_a in enumerate(fibres):
        for fibre_b in fibres[index + 1 :]:
            center_distance = float(
                np.hypot(
                    fibre_a.center_x - fibre_b.center_x,
                    fibre_a.center_y - fibre_b.center_y,
                )
            )
            edge_distance = center_distance - fibre_a.radius - fibre_b.radius
            minimum_spacing = (
                edge_distance
                if minimum_spacing is None
                else min(minimum_spacing, edge_distance)
            )
    return minimum_spacing


def _count_clipped_fibres(geometry: GeometryModel) -> int:
    clipped = 0
    if isinstance(geometry.domain, Domain2D):
        for fibre in geometry.circular_fibres:
            if (
                fibre.center_x - fibre.radius < geometry.domain.origin_x
                or fibre.center_y - fibre.radius < geometry.domain.origin_y
                or fibre.center_x + fibre.radius > geometry.domain.x_max
                or fibre.center_y + fibre.radius > geometry.domain.y_max
            ):
                clipped += 1
        for polygon in geometry.polygonal_fibres:
            points = polygon.points
            if points.size == 0:
                continue
            if (
                np.any(points[:, 0] < geometry.domain.origin_x)
                or np.any(points[:, 0] > geometry.domain.x_max)
                or np.any(points[:, 1] < geometry.domain.origin_y)
                or np.any(points[:, 1] > geometry.domain.y_max)
            ):
                clipped += 1
        return clipped

    assert isinstance(geometry.domain, Domain3D)
    for cylinder in geometry.cylindrical_fibres:
        if (
            cylinder.center_x - cylinder.radius < geometry.domain.origin_x
            or cylinder.center_y - cylinder.radius < geometry.domain.origin_y
            or cylinder.center_x + cylinder.radius > geometry.domain.x_max
            or cylinder.center_y + cylinder.radius > geometry.domain.y_max
            or cylinder.z_min < geometry.domain.origin_z
            or cylinder.z_max > geometry.domain.z_max
        ):
            clipped += 1
    for extruded_polygon in geometry.extruded_polygonal_fibres:
        points = extruded_polygon.points
        if points.size == 0:
            continue
        if (
            np.any(points[:, 0] < geometry.domain.origin_x)
            or np.any(points[:, 0] > geometry.domain.x_max)
            or np.any(points[:, 1] < geometry.domain.origin_y)
            or np.any(points[:, 1] > geometry.domain.y_max)
            or extruded_polygon.z_min < geometry.domain.origin_z
            or extruded_polygon.z_max > geometry.domain.z_max
        ):
            clipped += 1
    return clipped
