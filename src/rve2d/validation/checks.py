from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rve2d.models import (
    Domain3D,
    FloatArray,
    GeometryModel,
    circle_rectangle_intersection_area,
    clipped_polygon_area,
)

# A fibre part counts as present inside the domain when it covers more than this fraction of
# the fibre cross-section.
_PRESENT_FRACTION = 1e-9


@dataclass(frozen=True)
class QualityReport:
    """Geometry quality summary, serialised to ``quality_report.json``.

    ``valid`` is False exactly when ``warnings`` is not empty, i.e. when the geometry has a
    defect: fibres closer than the required minimum spacing (or overlapping), or fibres cut by
    the domain boundary other than complete periodic wraps. ``notes`` are informational and do
    not affect ``valid``.

    Attributes:
        minimum_spacing: Smallest surface-to-surface gap between two distinct circular or
            cylindrical fibres; periodic minimum-image distance in periodic geometries.
        clipped_fibres: Fibres cut by the domain boundary (only their part inside the domain is
            meshed); fibres entirely outside count as ``outside_fibres``. In a periodic geometry
            a fibre that crosses the boundary together with all its periodic images is complete
            and counted in ``wrapped_fibres`` instead.
        disconnected_artifacts: Deprecated alias of ``removed_artifacts`` with the same value,
            kept so that existing readers of ``quality_report.json`` keep working. (The image
            importers always reported removed noise specks under this name.)
        removed_artifacts: Small regions (noise specks below ``min_artifact_area_px``) removed
            from the input mask during image import. Informational only.
        wrapped_fibres: Fibres crossing a periodic boundary that are represented by all their
            periodic images.
        outside_fibres: Fibres lying entirely outside the domain (for example outside an image
            window smaller than the image); the mesher drops them. Informational only.
        minimum_boundary_clearance: Smallest distance between a circular or cylindrical fibre
            surface and a domain face or corner that it does not cross, or by which it crosses
            one. Values much smaller than the mesh size produce sliver elements.
        notes: Informational messages (removed specks, wrapped fibres, mesh resolution).
    """

    valid: bool
    minimum_spacing: float | None
    clipped_fibres: int
    disconnected_artifacts: int = 0
    warnings: list[str] = field(default_factory=list)
    removed_artifacts: int = 0
    wrapped_fibres: int = 0
    outside_fibres: int = 0
    minimum_boundary_clearance: float | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep the deprecated alias consistent with the new field, whichever one was given.
        if self.removed_artifacts == 0 and self.disconnected_artifacts != 0:
            object.__setattr__(self, "removed_artifacts", self.disconnected_artifacts)
        elif self.disconnected_artifacts != self.removed_artifacts:
            object.__setattr__(self, "disconnected_artifacts", self.removed_artifacts)


def validate_geometry(
    geometry: GeometryModel,
    minimum_spacing_requirement: float = 0.0,
    disconnected_artifacts: int = 0,
    *,
    removed_artifacts: int | None = None,
    element_size: float | None = None,
) -> QualityReport:
    """Check fibre spacing and boundary clipping of ``geometry``.

    Args:
        geometry: Geometry to check. Geometries with ``periodic_pairs`` are treated as
            periodic: spacing uses minimum-image distances, and fibres that cross the boundary
            together with their periodic images (same ``fibre_id``) are not clipped.
        minimum_spacing_requirement: Gaps between fibres below this value are a defect.
        disconnected_artifacts: Deprecated name of ``removed_artifacts``.
        removed_artifacts: Number of noise specks removed from the input mask (informational).
        element_size: Smallest mesh element size (``mesh.element_size_min``). When given, gaps
            between fibres, and between fibres and domain faces or corners, that are thinner
            than one element are reported in ``notes`` because they mesh into slivers.
    """
    removed = disconnected_artifacts if removed_artifacts is None else removed_artifacts
    periods = _periods(geometry)
    minimum_spacing = _minimum_spacing(geometry, periods)
    clipped_fibres, wrapped_fibres, outside_fibres = _count_clipped_fibres(
        geometry, periods
    )
    boundary_clearance = _minimum_boundary_clearance(geometry)

    warnings: list[str] = []
    if minimum_spacing is not None and minimum_spacing < minimum_spacing_requirement:
        warnings.append(
            f"Minimum spacing {minimum_spacing:.6f} is below the requirement "
            f"{minimum_spacing_requirement:.6f}."
        )
    if clipped_fibres > 0:
        if periods is None:
            warnings.append(f"{clipped_fibres} fibres intersect the domain boundary.")
        else:
            warnings.append(
                f"{clipped_fibres} fibres intersect the periodic domain boundary without all "
                "of their periodic images."
            )

    notes: list[str] = []
    if removed > 0:
        notes.append(
            f"{removed} small regions (noise below min_artifact_area_px) were removed from the "
            "input mask; they are not part of the geometry."
        )
    if wrapped_fibres > 0:
        notes.append(
            f"{wrapped_fibres} fibres cross the periodic boundary and are represented by their "
            "periodic images."
        )
    if outside_fibres > 0:
        notes.append(
            f"{outside_fibres} fibres lie entirely outside the domain and are not meshed."
        )
    if element_size is not None:
        notes.extend(_mesh_resolution_notes(minimum_spacing, boundary_clearance, element_size))

    return QualityReport(
        valid=not warnings,
        minimum_spacing=minimum_spacing,
        clipped_fibres=clipped_fibres,
        disconnected_artifacts=removed,
        warnings=warnings,
        removed_artifacts=removed,
        wrapped_fibres=wrapped_fibres,
        outside_fibres=outside_fibres,
        minimum_boundary_clearance=boundary_clearance,
        notes=notes,
    )


def _mesh_resolution_notes(
    minimum_spacing: float | None,
    boundary_clearance: float | None,
    element_size: float,
) -> list[str]:
    notes: list[str] = []
    if minimum_spacing is not None and 0.0 <= minimum_spacing < element_size:
        notes.append(
            f"Mesh resolution: the smallest fibre gap {minimum_spacing:.6g} is thinner than "
            f"one element ({element_size:.6g}); elements bridging it will be slivers. Increase "
            "min_spacing or refine the mesh."
        )
    if boundary_clearance is not None and boundary_clearance < element_size:
        notes.append(
            f"Mesh resolution: a fibre surface comes within {boundary_clearance:.6g} of a "
            "domain face or corner (on either side of it), less than one element "
            f"({element_size:.6g}); the thin region in between meshes into sliver elements. "
            "Increase boundary_clearance (or edge_clearance) or refine the mesh."
        )
    return notes


def _periods(geometry: GeometryModel) -> tuple[float, float] | None:
    """In-plane periods (x, y) of a periodic geometry, or None for a non-periodic one."""
    if not geometry.periodic_pairs:
        return None
    period_x = 0.0
    period_y = 0.0
    for pair in geometry.periodic_pairs:
        if len(pair.translation) > 0 and pair.translation[0] != 0.0:
            period_x = abs(float(pair.translation[0]))
        if len(pair.translation) > 1 and pair.translation[1] != 0.0:
            period_y = abs(float(pair.translation[1]))
    if period_x <= 0.0 or period_y <= 0.0:
        return None
    return period_x, period_y


@dataclass(frozen=True)
class _FibreShape:
    """One fibre instance: a circle (``radius >= 0``, ``points`` = centre) or a polygon."""

    key: tuple[str, int]
    points: FloatArray
    radius: float
    z_range: tuple[float, float] | None


def _fibre_shapes(geometry: GeometryModel) -> list[_FibreShape]:
    shapes: list[_FibreShape] = []
    for circle in geometry.circular_fibres:
        shapes.append(
            _FibreShape(
                ("circle", circle.fibre_id),
                np.array([[circle.center_x, circle.center_y]]),
                circle.radius,
                None,
            )
        )
    for polygon in geometry.polygonal_fibres:
        shapes.append(
            _FibreShape(
                ("polygon", polygon.fibre_id),
                np.asarray(polygon.points, dtype=np.float64).reshape(-1, 2),
                -1.0,
                None,
            )
        )
    for cylinder in geometry.cylindrical_fibres:
        shapes.append(
            _FibreShape(
                ("cylinder", cylinder.fibre_id),
                np.array([[cylinder.center_x, cylinder.center_y]]),
                cylinder.radius,
                (cylinder.z_min, cylinder.z_max),
            )
        )
    for extruded in geometry.extruded_polygonal_fibres:
        shapes.append(
            _FibreShape(
                ("extruded_polygon", extruded.fibre_id),
                np.asarray(extruded.points, dtype=np.float64).reshape(-1, 2),
                -1.0,
                (extruded.z_min, extruded.z_max),
            )
        )
    return shapes


def _bounds(geometry: GeometryModel) -> tuple[float, float, float, float]:
    domain = geometry.domain
    return (domain.origin_x, domain.origin_y, domain.x_max, domain.y_max)


def _is_inside_domain(shape: _FibreShape, geometry: GeometryModel) -> bool:
    """True when a positive part of the fibre lies inside the domain (it gets meshed)."""
    bounds = _bounds(geometry)
    domain = geometry.domain
    if shape.z_range is not None and isinstance(domain, Domain3D):
        overlap = min(shape.z_range[1], domain.z_max) - max(shape.z_range[0], domain.origin_z)
        if overlap <= 0.0:
            return False
    if shape.radius >= 0.0:
        inside = circle_rectangle_intersection_area(
            float(shape.points[0, 0]), float(shape.points[0, 1]), shape.radius, *bounds
        )
        return inside > _PRESENT_FRACTION * math.pi * shape.radius**2
    if shape.points.shape[0] < 3:
        return False
    extent = float(np.ptp(shape.points[:, 0]) * np.ptp(shape.points[:, 1]))
    return clipped_polygon_area(shape.points, *bounds) > _PRESENT_FRACTION * extent


def _crosses_domain_boundary(shape: _FibreShape, geometry: GeometryModel) -> bool:
    domain = geometry.domain
    if shape.z_range is not None and isinstance(domain, Domain3D):
        if shape.z_range[0] < domain.origin_z or shape.z_range[1] > domain.z_max:
            return True
    return _crosses_boundary(shape.points, shape.radius, _bounds(geometry))


def _circle_data(
    geometry: GeometryModel,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Centres, radii and ids of the circles (2D) or cylinders (3D) reaching into the domain."""
    kind = "cylinder" if geometry.dimension == 3 else "circle"
    kept = [
        shape
        for shape in _fibre_shapes(geometry)
        if shape.key[0] == kind and _is_inside_domain(shape, geometry)
    ]
    centres = np.array([shape.points[0] for shape in kept], dtype=np.float64)
    radii = np.array([shape.radius for shape in kept], dtype=np.float64)
    ids = np.array([shape.key[1] for shape in kept], dtype=np.int64)
    return centres.reshape(-1, 2), radii, ids


def _minimum_spacing(
    geometry: GeometryModel,
    periods: tuple[float, float] | None,
) -> float | None:
    centres, radii, ids = _circle_data(geometry)
    if periods is not None and centres.shape[0] > 0:
        # One representative per fibre: periodic images are the same fibre.
        _, first = np.unique(ids, return_index=True)
        first = np.sort(first)
        centres, radii, ids = centres[first], radii[first], ids[first]
        box = np.array(periods, dtype=np.float64)
        origin = np.array([geometry.domain.origin_x, geometry.domain.origin_y])
        centres = _wrap_into_box(centres - origin, box)
        return _minimum_gap(centres, radii, ids, box)
    return _minimum_gap(centres, radii, ids, None)


def _wrap_into_box(points: FloatArray, box: FloatArray) -> FloatArray:
    wrapped = points - box * np.floor(points / box)
    wrapped = np.where(wrapped >= box, wrapped - box, wrapped)
    return np.where(wrapped < 0.0, 0.0, wrapped)


def _minimum_gap(
    centres: FloatArray,
    radii: FloatArray,
    ids: NDArray[np.int64],
    box: FloatArray | None,
) -> float | None:
    """Smallest ``distance - r_i - r_j`` over pairs of distinct fibres."""
    count = centres.shape[0]
    if count < 2:
        return None
    tree = cKDTree(centres, boxsize=box) if box is not None else cKDTree(centres)
    largest = float(radii.max())
    extent = float(np.max(np.ptp(centres, axis=0))) if box is None else float(box.max())
    reach = 2.0 * largest + max(largest, 1e-12)
    while True:
        pairs = np.asarray(tree.query_pairs(reach, output_type="ndarray"), dtype=np.intp)
        pairs = pairs.reshape(-1, 2)
        pairs = pairs[ids[pairs[:, 0]] != ids[pairs[:, 1]]]
        if pairs.shape[0]:
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            delta = centres[pairs[:, 0]] - centres[pairs[:, 1]]
            if box is not None:
                delta -= box * np.round(delta / box)
            gaps = np.hypot(delta[:, 0], delta[:, 1]) - radii[pairs[:, 0]] - radii[pairs[:, 1]]
            smallest = float(gaps.min())
            # Pairs farther apart than `reach` have gaps above reach - 2 * largest.
            if smallest <= reach - 2.0 * largest:
                return smallest
        if reach > 2.0 * extent + 4.0 * largest:
            return None if pairs.shape[0] == 0 else smallest
        reach *= 2.0


def _count_clipped_fibres(
    geometry: GeometryModel,
    periods: tuple[float, float] | None,
) -> tuple[int, int, int]:
    """Return ``(clipped, wrapped, outside)`` fibre counts.

    Without periodicity every fibre shape that crosses the domain boundary is clipped, unless
    it lies entirely outside the domain (the mesher drops it). With periodicity, shapes sharing
    a ``fibre_id`` are one fibre, which is wrapped when all its periodic images are present.
    """
    shapes = _fibre_shapes(geometry)
    clipped = 0
    wrapped = 0
    outside = 0
    if periods is None:
        for shape in shapes:
            if not _crosses_domain_boundary(shape, geometry):
                continue
            if _is_inside_domain(shape, geometry):
                clipped += 1
            else:
                outside += 1
        return clipped, 0, outside

    groups: dict[tuple[str, int], list[_FibreShape]] = {}
    for shape in shapes:
        groups.setdefault(shape.key, []).append(shape)
    period = np.array(periods, dtype=np.float64)
    for members in groups.values():
        if not any(_is_inside_domain(member, geometry) for member in members):
            outside += 1
            continue
        if not any(_crosses_domain_boundary(member, geometry) for member in members):
            continue
        if _is_complete_periodic_wrap(members, geometry, period):
            wrapped += 1
        else:
            clipped += 1
    return clipped, wrapped, outside


def _is_complete_periodic_wrap(
    members: list[_FibreShape],
    geometry: GeometryModel,
    period: FloatArray,
) -> bool:
    """Instances sharing a ``fibre_id`` form a complete wrap when they are translates of each
    other by whole in-plane periods and every translate overlapping the domain is present.
    Leaving the domain through a z face is never a periodic wrap."""
    domain = geometry.domain
    if isinstance(domain, Domain3D):
        for member in members:
            if member.z_range is not None and (
                member.z_range[0] < domain.origin_z or member.z_range[1] > domain.z_max
            ):
                return False
    reference = members[0]
    present: set[tuple[int, int]] = set()
    for member in members:
        shift = _whole_period_shift(
            reference.points, member.points, reference.radius, member.radius, period
        )
        if shift is None:
            return False
        present.add(shift)
    required = _required_shifts(reference.points, reference.radius, _bounds(geometry), period)
    return required <= present


def _crosses_boundary(
    shape: FloatArray,
    radius: float,
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, y_min, x_max, y_max = bounds
    if radius >= 0.0:
        center_x, center_y = float(shape[0, 0]), float(shape[0, 1])
        return (
            center_x - radius < x_min
            or center_y - radius < y_min
            or center_x + radius > x_max
            or center_y + radius > y_max
        )
    if shape.size == 0:
        return False
    return bool(
        np.any(shape[:, 0] < x_min)
        or np.any(shape[:, 0] > x_max)
        or np.any(shape[:, 1] < y_min)
        or np.any(shape[:, 1] > y_max)
    )


def _whole_period_shift(
    reference: FloatArray,
    shape: FloatArray,
    reference_radius: float,
    radius: float,
    period: FloatArray,
) -> tuple[int, int] | None:
    """Integer period multiples (k, l) with shape == reference + (k, l) * period, else None."""
    if shape.shape != reference.shape or reference.size == 0:
        return None
    if reference_radius >= 0.0 and not math.isclose(radius, reference_radius, rel_tol=1e-9):
        return None
    offset = shape[0] - reference[0]
    multiples = np.round(offset / period)
    tolerance = 1e-7 * float(period.max())
    if not np.allclose(shape - reference, multiples * period, rtol=0.0, atol=tolerance):
        return None
    return int(multiples[0]), int(multiples[1])


def _required_shifts(
    reference: FloatArray,
    radius: float,
    bounds: tuple[float, float, float, float],
    period: FloatArray,
) -> set[tuple[int, int]]:
    """Whole-period translates of ``reference`` that overlap the domain interior."""
    x_min, y_min, x_max, y_max = bounds
    if radius >= 0.0:
        lower = reference[0] - radius
        upper = reference[0] + radius
        full = math.pi * radius**2
    else:
        lower = reference.min(axis=0)
        upper = reference.max(axis=0)
        full = clipped_polygon_area(
            reference,
            float(lower[0]),
            float(lower[1]),
            float(upper[0]),
            float(upper[1]),
        )
    k_range = range(
        math.floor((x_min - float(upper[0])) / period[0]),
        math.ceil((x_max - float(lower[0])) / period[0]) + 1,
    )
    l_range = range(
        math.floor((y_min - float(upper[1])) / period[1]),
        math.ceil((y_max - float(lower[1])) / period[1]) + 1,
    )
    required: set[tuple[int, int]] = set()
    for k in k_range:
        for m in l_range:
            shift = np.array([k * period[0], m * period[1]])
            if radius >= 0.0:
                area = circle_rectangle_intersection_area(
                    float(reference[0, 0] + shift[0]),
                    float(reference[0, 1] + shift[1]),
                    radius,
                    *bounds,
                )
            else:
                area = clipped_polygon_area(reference + shift, *bounds)
            if area > _PRESENT_FRACTION * full:
                required.add((k, m))
    return required


def _minimum_boundary_clearance(geometry: GeometryModel) -> float | None:
    """Smallest distance between a circle (cylinder cross-section) reaching into the domain and
    a domain face or corner that it does not cross, or by which it crosses one."""
    centres, radii, _ = _circle_data(geometry)
    if centres.shape[0] == 0:
        return None
    x_min, y_min, x_max, y_max = _bounds(geometry)
    smallest: float | None = None
    for (center_x, center_y), radius in zip(centres.tolist(), radii.tolist(), strict=True):
        candidates: list[float] = []
        # Faces: only where the closest point of the face line lies on the face itself.
        if y_min <= center_y <= y_max:
            candidates.append(abs(abs(center_x - x_min) - radius))
            candidates.append(abs(abs(center_x - x_max) - radius))
        if x_min <= center_x <= x_max:
            candidates.append(abs(abs(center_y - y_min) - radius))
            candidates.append(abs(abs(center_y - y_max) - radius))
        for corner_x in (x_min, x_max):
            for corner_y in (y_min, y_max):
                distance = math.hypot(center_x - corner_x, center_y - corner_y)
                candidates.append(abs(distance - radius))
        nearest = min(candidates)
        smallest = nearest if smallest is None else min(smallest, nearest)
    return smallest
