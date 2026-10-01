from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rve2d.config import MeshConfig
from rve2d.exceptions import MeshingError
from rve2d.models import (
    Domain,
    Domain2D,
    Domain3D,
    FloatArray,
    GeometryModel,
    PeriodicBoundaryPair,
)

# OpenCASCADE enlarges bounding boxes by its confusion tolerance (1e-7, absolute); geometric
# comparisons below never use a tolerance smaller than a few times that value.
_OCC_CONFUSION = 1e-7


@dataclass(frozen=True)
class MeshBuildResult:
    mesh_path: Path
    phase_tags: dict[str, int]
    boundary_tags: dict[str, int]


def build_mesh_with_gmsh(
    geometry: GeometryModel,
    mesh_config: MeshConfig,
    output_path: str | Path,
) -> MeshBuildResult:
    """Mesh the matrix and the fibre phase of ``geometry`` with gmsh (OpenCASCADE kernel).

    Fibres are clipped to the domain: the domain is fragmented with all fibre shapes, so the
    fibre/matrix interfaces are conforming (shared nodes), and fibre parts outside the domain
    are discarded. Periodic images of wrapped fibres therefore contribute exactly the part of
    the fibre that lies inside the domain. For periodic geometries the boundary pieces of
    opposite faces are paired geometrically and meshed with ``setPeriodic``, which makes the
    boundary meshes node-matched.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise MeshingError(
            "gmsh is required for meshing. Install the optional dependency with `.[gmsh]`."
        ) from exc

    mesh_path = Path(output_path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", mesh_config.verbosity)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_config.element_size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_config.element_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", mesh_config.algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_config.mesh_order)
        if mesh_config.elements_per_circle > 0:
            # Resolve curved fibre boundaries: element size follows the local curvature so a
            # full circle gets about this many elements (still bounded by the min/max sizes).
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", mesh_config.elements_per_circle)
        gmsh.model.add(mesh_path.stem)

        if geometry.dimension == 2:
            matrix_tags, fibre_tags, boundary_entities, boundary_dim, mesh_dim = _build_2d_model(
                gmsh,
                geometry,
            )
        else:
            matrix_tags, fibre_tags, boundary_entities, boundary_dim, mesh_dim = _build_3d_model(
                gmsh,
                geometry,
            )

        gmsh.model.addPhysicalGroup(mesh_dim, matrix_tags, tag=geometry.phase_labels["matrix"])
        gmsh.model.setPhysicalName(mesh_dim, geometry.phase_labels["matrix"], "matrix")
        gmsh.model.addPhysicalGroup(mesh_dim, fibre_tags, tag=geometry.phase_labels["fibre"])
        gmsh.model.setPhysicalName(mesh_dim, geometry.phase_labels["fibre"], "fibre")

        if geometry.periodic_pairs:
            _apply_periodic_meshing(gmsh, geometry, boundary_entities, boundary_dim)
        for name, entity_tags in boundary_entities.items():
            if entity_tags:
                label = geometry.boundary_labels[name]
                gmsh.model.addPhysicalGroup(boundary_dim, entity_tags, tag=label)
                gmsh.model.setPhysicalName(boundary_dim, label, name)

        gmsh.model.mesh.generate(mesh_dim)
        if mesh_config.recombine:
            gmsh.model.mesh.recombine()
        gmsh.write(str(mesh_path))
    finally:
        gmsh.finalize()

    return MeshBuildResult(
        mesh_path=mesh_path,
        phase_tags=dict(geometry.phase_labels),
        boundary_tags=dict(geometry.boundary_labels),
    )


def _build_2d_model(
    gmsh: Any,
    geometry: GeometryModel,
) -> tuple[list[int], list[int], dict[str, list[int]], int, int]:
    assert isinstance(geometry.domain, Domain2D)
    domain = geometry.domain
    tolerance = _geometry_tolerance(domain)
    rect_tag = gmsh.model.occ.addRectangle(
        domain.origin_x,
        domain.origin_y,
        0.0,
        domain.width,
        domain.height,
    )

    fibre_surfaces: list[tuple[int, int]] = []
    for fibre in geometry.circular_fibres:
        if not _overlaps_domain(
            (fibre.center_x - fibre.radius, fibre.center_y - fibre.radius),
            (fibre.center_x + fibre.radius, fibre.center_y + fibre.radius),
            domain,
            tolerance,
        ):
            continue
        disk_tag = gmsh.model.occ.addDisk(
            fibre.center_x,
            fibre.center_y,
            0.0,
            fibre.radius,
            fibre.radius,
        )
        fibre_surfaces.append((2, disk_tag))

    for polygon_fibre in geometry.polygonal_fibres:
        points = _polygon_vertices(polygon_fibre.points, tolerance)
        if points.shape[0] < 3 or not _overlaps_domain(
            tuple(points.min(axis=0)), tuple(points.max(axis=0)), domain, tolerance
        ):
            continue
        surface_tag = _add_polygon_surface(gmsh, points, 0.0)
        fibre_surfaces.append((2, surface_tag))

    matrix_tags, fibre_tags = _fragment_and_clip(gmsh, (2, rect_tag), fibre_surfaces)
    boundary_entities = _classify_outer_boundaries_2d(gmsh, geometry)
    return matrix_tags, fibre_tags, boundary_entities, 1, 2


def _build_3d_model(
    gmsh: Any,
    geometry: GeometryModel,
) -> tuple[list[int], list[int], dict[str, list[int]], int, int]:
    assert isinstance(geometry.domain, Domain3D)
    domain = geometry.domain
    tolerance = _geometry_tolerance(domain)
    box_tag = gmsh.model.occ.addBox(
        domain.origin_x,
        domain.origin_y,
        domain.origin_z,
        domain.width,
        domain.height,
        domain.depth,
    )

    fibre_volumes: list[tuple[int, int]] = []
    for fibre in geometry.cylindrical_fibres:
        if not _overlaps_domain(
            (fibre.center_x - fibre.radius, fibre.center_y - fibre.radius, fibre.z_min),
            (fibre.center_x + fibre.radius, fibre.center_y + fibre.radius, fibre.z_max),
            domain,
            tolerance,
        ):
            continue
        cylinder_tag = gmsh.model.occ.addCylinder(
            fibre.center_x,
            fibre.center_y,
            fibre.z_min,
            0.0,
            0.0,
            fibre.z_max - fibre.z_min,
            fibre.radius,
        )
        fibre_volumes.append((3, cylinder_tag))

    for polygon_fibre in geometry.extruded_polygonal_fibres:
        points = _polygon_vertices(polygon_fibre.points, tolerance)
        if points.shape[0] < 3 or not _overlaps_domain(
            (*points.min(axis=0), polygon_fibre.z_min),
            (*points.max(axis=0), polygon_fibre.z_max),
            domain,
            tolerance,
        ):
            continue
        volume_tag = _add_extruded_polygon_volume(
            gmsh, points, polygon_fibre.z_min, polygon_fibre.z_max
        )
        fibre_volumes.append((3, volume_tag))

    matrix_tags, fibre_tags = _fragment_and_clip(gmsh, (3, box_tag), fibre_volumes)
    boundary_entities = _classify_outer_boundaries_3d(gmsh, geometry)
    return matrix_tags, fibre_tags, boundary_entities, 2, 3


def _fragment_and_clip(
    gmsh: Any,
    domain_dimtag: tuple[int, int],
    fibre_dimtags: list[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Fragment the domain with the fibres and drop every fibre piece outside the domain.

    Fragmenting (rather than cutting the fibres out of the domain) yields one shared boundary
    entity per fibre/matrix interface, so the mesh is conforming, and splits fibres that cross
    the domain boundary into an inside and an outside part. A piece belongs to the fibre phase
    if it comes from both the domain and a fibre, to the matrix if it comes from the domain
    only, and is removed if it comes from fibres only.
    """
    dim = domain_dimtag[0]
    if not fibre_dimtags:
        gmsh.model.occ.synchronize()
        return [domain_dimtag[1]], []
    _, result_map = gmsh.model.occ.fragment([domain_dimtag], fibre_dimtags)
    domain_pieces = {tag for piece_dim, tag in result_map[0] if piece_dim == dim}
    fibre_pieces = {
        tag for pieces in result_map[1:] for piece_dim, tag in pieces if piece_dim == dim
    }
    outside_pieces = sorted(fibre_pieces - domain_pieces)
    if outside_pieces:
        gmsh.model.occ.remove([(dim, tag) for tag in outside_pieces], recursive=True)
    gmsh.model.occ.synchronize()
    matrix_tags = sorted(domain_pieces - fibre_pieces)
    fibre_tags = sorted(domain_pieces & fibre_pieces)
    if not matrix_tags:
        kind = "surface" if dim == 2 else "volume"
        raise MeshingError(f"gmsh did not return a matrix {kind} after fragmenting the domain.")
    return matrix_tags, fibre_tags


def _geometry_tolerance(domain: Domain) -> float:
    if isinstance(domain, Domain2D):
        size = min(domain.width, domain.height)
    else:
        size = min(domain.width, domain.height, domain.depth)
    return max(1e-9, size * 1e-6, 10.0 * _OCC_CONFUSION)


def _overlaps_domain(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    domain: Domain,
    tolerance: float,
) -> bool:
    """True when a fibre bounding box overlaps the domain by more than ``tolerance``.

    Fibres entirely outside the domain (or merely touching it) are skipped before the boolean
    operation.
    """
    if isinstance(domain, Domain2D):
        domain_lower: tuple[float, ...] = (domain.origin_x, domain.origin_y)
        domain_upper: tuple[float, ...] = (domain.x_max, domain.y_max)
    else:
        domain_lower = (domain.origin_x, domain.origin_y, domain.origin_z)
        domain_upper = (domain.x_max, domain.y_max, domain.z_max)
    for low, high, domain_low, domain_high in zip(
        lower, upper, domain_lower, domain_upper, strict=True
    ):
        if min(high, domain_high) - max(low, domain_low) <= tolerance:
            return False
    return True


def _polygon_vertices(points: FloatArray, tolerance: float) -> FloatArray:
    """Open polygon vertex list without the closing point and without repeated vertices."""
    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if vertices.shape[0] == 0:
        return vertices
    keep = [0]
    for index in range(1, vertices.shape[0]):
        if float(np.max(np.abs(vertices[index] - vertices[keep[-1]]))) > tolerance:
            keep.append(index)
    vertices = vertices[keep]
    while vertices.shape[0] > 1 and float(np.max(np.abs(vertices[-1] - vertices[0]))) <= tolerance:
        vertices = vertices[:-1]
    return vertices


def _add_polygon_surface(gmsh: Any, points: FloatArray, z: float) -> int:
    point_tags = [
        gmsh.model.occ.addPoint(float(point[0]), float(point[1]), z) for point in points
    ]
    line_tags: list[int] = []
    for index in range(len(point_tags)):
        start = point_tags[index]
        end = point_tags[(index + 1) % len(point_tags)]
        line_tags.append(gmsh.model.occ.addLine(start, end))
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    return int(gmsh.model.occ.addPlaneSurface([loop]))


def _add_extruded_polygon_volume(
    gmsh: Any,
    points: FloatArray,
    z_min: float,
    z_max: float,
) -> int:
    surface = _add_polygon_surface(gmsh, points, z_min)
    extruded = gmsh.model.occ.extrude([(2, surface)], 0.0, 0.0, z_max - z_min)
    volumes = [tag for dim, tag in extruded if dim == 3]
    if len(volumes) != 1:
        raise MeshingError("gmsh extrusion did not return a single fibre volume.")
    return int(volumes[0])


def _classify_outer_boundaries_2d(gmsh: Any, geometry: GeometryModel) -> dict[str, list[int]]:
    assert isinstance(geometry.domain, Domain2D)
    domain = geometry.domain
    planes = (
        ("left", 0, domain.origin_x),
        ("right", 0, domain.x_max),
        ("bottom", 1, domain.origin_y),
        ("top", 1, domain.y_max),
    )
    return _classify_by_plane(gmsh, 1, planes, max(domain.width, domain.height))


def _classify_outer_boundaries_3d(gmsh: Any, geometry: GeometryModel) -> dict[str, list[int]]:
    assert isinstance(geometry.domain, Domain3D)
    domain = geometry.domain
    planes = (
        ("left", 0, domain.origin_x),
        ("right", 0, domain.x_max),
        ("front", 1, domain.origin_y),
        ("back", 1, domain.y_max),
        ("bottom", 2, domain.origin_z),
        ("top", 2, domain.z_max),
    )
    return _classify_by_plane(gmsh, 2, planes, max(domain.width, domain.height, domain.depth))


def _classify_by_plane(
    gmsh: Any,
    entity_dim: int,
    planes: tuple[tuple[str, int, float], ...],
    domain_size: float,
) -> dict[str, list[int]]:
    """Assign boundary curves (2D) or surfaces (3D) to the outer faces of the domain.

    Membership is decided from exact geometry -- the entity's centre of mass and all of its
    boundary vertices must lie on the face plane -- rather than from bounding boxes, which
    OpenCASCADE pads by an absolute ~1e-7 and so misclassifies RVEs whose size is given in
    millimetres or metres (e.g. a 0.06 mm window).
    """
    tolerance = 1e-7 * domain_size
    boundaries: dict[str, list[int]] = {name: [] for name, _, _ in planes}
    for _, tag in gmsh.model.getEntities(dim=entity_dim):
        samples = [gmsh.model.occ.getCenterOfMass(entity_dim, tag)]
        for sub_dim, sub_tag in gmsh.model.getBoundary(
            [(entity_dim, tag)], combined=False, oriented=False, recursive=True
        ):
            if sub_dim == 0:
                samples.append(gmsh.model.getValue(0, sub_tag, []))
        for name, axis, value in planes:
            if all(abs(point[axis] - value) <= tolerance for point in samples):
                boundaries[name].append(tag)
                break
    return boundaries


def _apply_periodic_meshing(
    gmsh: Any,
    geometry: GeometryModel,
    boundary_entities: dict[str, list[int]],
    entity_dim: int,
) -> None:
    tolerance = _geometry_tolerance(geometry.domain)
    for pair in geometry.periodic_pairs:
        source_entities = boundary_entities.get(pair.source, [])
        target_entities = boundary_entities.get(pair.target, [])
        if not source_entities or not target_entities:
            raise MeshingError(
                "Periodic meshing requires boundary entities for "
                f"'{pair.source}' and '{pair.target}'."
            )
        targets, sources = _match_periodic_entities(
            gmsh,
            entity_dim,
            source_entities,
            target_entities,
            pair,
            tolerance,
        )
        gmsh.model.mesh.setPeriodic(
            entity_dim,
            targets,
            sources,
            _affine_translation(pair.translation),
        )


def _match_periodic_entities(
    gmsh: Any,
    entity_dim: int,
    source_entities: list[int],
    target_entities: list[int],
    pair: PeriodicBoundaryPair,
    tolerance: float,
) -> tuple[list[int], list[int]]:
    """Pair every target boundary piece with the source piece it is a translate of.

    Fibres that cross a periodic face split it into several pieces. Each target piece must
    match exactly one source piece: same size (length or area) and a centre of mass that
    coincides after the periodic translation. Matching is purely geometric, so it does not
    depend on the order in which gmsh numbers the pieces.
    """
    translation = np.zeros(3)
    translation[: len(pair.translation)] = pair.translation
    source_centres = np.array(
        [gmsh.model.occ.getCenterOfMass(entity_dim, tag) for tag in source_entities],
        dtype=np.float64,
    )
    source_sizes = np.array(
        [gmsh.model.occ.getMass(entity_dim, tag) for tag in source_entities],
        dtype=np.float64,
    )
    matched_targets: list[int] = []
    matched_sources: list[int] = []
    used: set[int] = set()
    unmatched: list[str] = []
    for target in target_entities:
        centre = np.asarray(gmsh.model.occ.getCenterOfMass(entity_dim, target))
        size = float(gmsh.model.occ.getMass(entity_dim, target))
        offset = np.max(np.abs(source_centres + translation - centre), axis=1)
        size_error = np.abs(source_sizes - size)
        candidates = np.flatnonzero(
            (offset <= tolerance) & (size_error <= 1e-6 * max(abs(size), 1e-300) + tolerance**2)
        )
        if candidates.size != 1 or int(candidates[0]) in used:
            location = ", ".join(f"{value:.6g}" for value in centre)
            unmatched.append(f"{pair.target} piece {target} centred at ({location})")
            continue
        used.add(int(candidates[0]))
        matched_targets.append(target)
        matched_sources.append(source_entities[int(candidates[0])])
    for index, source in enumerate(source_entities):
        if index not in used:
            location = ", ".join(f"{value:.6g}" for value in source_centres[index])
            unmatched.append(f"{pair.source} piece {source} centred at ({location})")
    if unmatched:
        raise MeshingError(
            f"Periodic meshing could not pair the '{pair.target}' boundary with "
            f"'{pair.source}' (translation {pair.translation}): "
            + "; ".join(unmatched[:6])
            + (" ..." if len(unmatched) > 6 else "")
            + ". The geometry is not periodic across this boundary, for example because a "
            "fibre crosses it without its periodic image."
        )
    return matched_targets, matched_sources


def _affine_translation(translation: tuple[float, ...]) -> list[float]:
    tx = translation[0] if len(translation) > 0 else 0.0
    ty = translation[1] if len(translation) > 1 else 0.0
    tz = translation[2] if len(translation) > 2 else 0.0
    return [
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0,
    ]
