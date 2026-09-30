from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rve2d.config import MeshConfig
from rve2d.exceptions import MeshingError
from rve2d.models import (
    Domain2D,
    Domain3D,
    ExtrudedPolygonFibre,
    GeometryModel,
    PolygonFibre,
)


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
    rect_tag = gmsh.model.occ.addRectangle(
        geometry.domain.origin_x,
        geometry.domain.origin_y,
        0.0,
        geometry.domain.width,
        geometry.domain.height,
    )

    fibre_surfaces: list[tuple[int, int]] = []
    for fibre in geometry.circular_fibres:
        disk_tag = gmsh.model.occ.addDisk(
            fibre.center_x,
            fibre.center_y,
            0.0,
            fibre.radius,
            fibre.radius,
        )
        fibre_surfaces.append((2, disk_tag))

    for polygon_fibre in geometry.polygonal_fibres:
        surface_tag = _add_polygon_surface(gmsh, polygon_fibre)
        fibre_surfaces.append((2, surface_tag))

    matrix_result, _ = gmsh.model.occ.cut(
        [(2, rect_tag)],
        fibre_surfaces,
        removeObject=True,
        removeTool=False,
    )
    gmsh.model.occ.synchronize()
    matrix_tags = [tag for dim, tag in matrix_result if dim == 2]
    fibre_tags = [tag for dim, tag in fibre_surfaces if dim == 2]
    if not matrix_tags:
        raise MeshingError("gmsh did not return a matrix surface after boolean cut.")
    boundary_entities = _classify_outer_boundaries_2d(gmsh, geometry)
    return matrix_tags, fibre_tags, boundary_entities, 1, 2


def _build_3d_model(
    gmsh: Any,
    geometry: GeometryModel,
) -> tuple[list[int], list[int], dict[str, list[int]], int, int]:
    assert isinstance(geometry.domain, Domain3D)
    box_tag = gmsh.model.occ.addBox(
        geometry.domain.origin_x,
        geometry.domain.origin_y,
        geometry.domain.origin_z,
        geometry.domain.width,
        geometry.domain.height,
        geometry.domain.depth,
    )

    fibre_volumes: list[tuple[int, int]] = []
    for fibre in geometry.cylindrical_fibres:
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
        volume_tag = _add_extruded_polygon_volume(gmsh, polygon_fibre)
        fibre_volumes.append((3, volume_tag))

    matrix_result, _ = gmsh.model.occ.cut(
        [(3, box_tag)],
        fibre_volumes,
        removeObject=True,
        removeTool=False,
    )
    gmsh.model.occ.synchronize()
    matrix_tags = [tag for dim, tag in matrix_result if dim == 3]
    fibre_tags = [tag for dim, tag in fibre_volumes if dim == 3]
    if not matrix_tags:
        raise MeshingError("gmsh did not return a matrix volume after boolean cut.")
    boundary_entities = _classify_outer_boundaries_3d(gmsh, geometry)
    return matrix_tags, fibre_tags, boundary_entities, 2, 3


def _add_polygon_surface(gmsh: Any, polygon_fibre: PolygonFibre) -> int:
    point_tags: list[int] = []
    mesh_size = 0.0
    for point in polygon_fibre.points[:-1]:
        point_tags.append(gmsh.model.occ.addPoint(float(point[0]), float(point[1]), 0.0, mesh_size))
    line_tags: list[int] = []
    for index in range(len(point_tags)):
        start = point_tags[index]
        end = point_tags[(index + 1) % len(point_tags)]
        line_tags.append(gmsh.model.occ.addLine(start, end))
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    return int(gmsh.model.occ.addPlaneSurface([loop]))


def _add_extruded_polygon_volume(gmsh: Any, polygon_fibre: ExtrudedPolygonFibre) -> int:
    point_tags: list[int] = []
    for point in polygon_fibre.points[:-1]:
        point_tags.append(
            gmsh.model.occ.addPoint(
                float(point[0]),
                float(point[1]),
                polygon_fibre.z_min,
            )
        )
    line_tags: list[int] = []
    for index in range(len(point_tags)):
        start = point_tags[index]
        end = point_tags[(index + 1) % len(point_tags)]
        line_tags.append(gmsh.model.occ.addLine(start, end))
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    surface = gmsh.model.occ.addPlaneSurface([loop])
    extruded = gmsh.model.occ.extrude(
        [(2, surface)],
        0.0,
        0.0,
        polygon_fibre.z_max - polygon_fibre.z_min,
    )
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
    for pair in geometry.periodic_pairs:
        source_entities = boundary_entities.get(pair.source, [])
        target_entities = boundary_entities.get(pair.target, [])
        if not source_entities or not target_entities:
            raise MeshingError(
                "Periodic meshing requires boundary entities for "
                f"'{pair.source}' and '{pair.target}'."
            )
        ordered_source = _sort_periodic_entities(gmsh, source_entities, pair.source, entity_dim)
        ordered_target = _sort_periodic_entities(gmsh, target_entities, pair.target, entity_dim)
        if len(ordered_source) != len(ordered_target):
            raise MeshingError(
                "Periodic meshing requires matching entity counts for "
                f"'{pair.source}' and '{pair.target}'."
            )
        gmsh.model.mesh.setPeriodic(
            entity_dim,
            ordered_target,
            ordered_source,
            _affine_translation(pair.translation),
        )


def _sort_periodic_entities(
    gmsh: Any,
    entity_tags: list[int],
    boundary_name: str,
    entity_dim: int,
) -> list[int]:
    if entity_dim == 1:
        if boundary_name in {"left", "right"}:
            def key_fn(tag: int) -> tuple[float, float, float]:
                midpoint = _entity_midpoint(gmsh, 1, tag)
                return (midpoint[1], midpoint[0], midpoint[2])
        else:
            def key_fn(tag: int) -> tuple[float, float, float]:
                midpoint = _entity_midpoint(gmsh, 1, tag)
                return (midpoint[0], midpoint[1], midpoint[2])
    else:
        if boundary_name in {"left", "right"}:
            def key_fn(tag: int) -> tuple[float, float, float]:
                midpoint = _entity_midpoint(gmsh, 2, tag)
                return (midpoint[1], midpoint[2], midpoint[0])
        elif boundary_name in {"front", "back"}:
            def key_fn(tag: int) -> tuple[float, float, float]:
                midpoint = _entity_midpoint(gmsh, 2, tag)
                return (midpoint[0], midpoint[2], midpoint[1])
        else:
            def key_fn(tag: int) -> tuple[float, float, float]:
                midpoint = _entity_midpoint(gmsh, 2, tag)
                return (midpoint[0], midpoint[1], midpoint[2])
    return sorted(entity_tags, key=key_fn)


def _entity_midpoint(gmsh: Any, entity_dim: int, entity_tag: int) -> tuple[float, float, float]:
    x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(entity_dim, entity_tag)
    return ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5)


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
