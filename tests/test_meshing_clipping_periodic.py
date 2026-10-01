"""Meshing of fibres that cross the domain boundary: clipping, conformity, periodicity.

All checks read the written ``.msh`` with meshio and are self-contained.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import pytest
from scipy.spatial import cKDTree

from rve2d.config import ImageImportConfig, MeshConfig, SyntheticGenerationConfig, load_config
from rve2d.exceptions import MeshingError
from rve2d.models import (
    CircleFibre,
    CylinderFibre,
    Domain2D,
    Domain3D,
    ExtrudedPolygonFibre,
    GeometryModel,
    PeriodicBoundaryPair,
    PolygonFibre,
)
from rve2d.synthetic_generation.circular import generate_circular_fibre_rve
from rve2d.synthetic_generation.cylindrical import generate_cylindrical_fibre_rve

gmsh = pytest.importorskip("gmsh")

from rve2d.meshing.gmsh_builder import build_mesh_with_gmsh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS_3D = {"left": 11, "right": 12, "front": 13, "back": 14, "bottom": 15, "top": 16}
MATRIX, FIBRE = 1, 2


# --------------------------------------------------------------------------------------------
# Mesh inspection helpers
# --------------------------------------------------------------------------------------------


def _read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mesh = meshio.read(path)
    dim = 3 if any(block.type == "tetra" for block in mesh.cells) else 2
    cell_type = "tetra" if dim == 3 else "triangle"
    cells = np.vstack([block.data for block in mesh.cells if block.type == cell_type])
    tags = np.concatenate(
        [
            data
            for block, data in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True)
            if block.type == cell_type
        ]
    )
    return np.asarray(mesh.points[:, :dim], dtype=np.float64), cells, tags, dim


def _cell_measures(points: np.ndarray, cells: np.ndarray, dim: int) -> np.ndarray:
    corners = points[cells]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    if dim == 2:
        return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    third = corners[:, 3] - corners[:, 0]
    return np.abs(np.einsum("ij,ij->i", first, np.cross(second, third))) / 6.0


def _meshed_fibre_fraction(
    points: np.ndarray, cells: np.ndarray, tags: np.ndarray, dim: int
) -> float:
    measures = _cell_measures(points, cells, dim)
    return float(measures[tags == FIBRE].sum() / measures.sum())


def _facets(cells: np.ndarray, dim: int) -> np.ndarray:
    local = [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    return np.vstack([np.sort(cells[:, list(index)], axis=1) for index in local])


def _assert_conforming(
    points: np.ndarray,
    cells: np.ndarray,
    tags: np.ndarray,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> int:
    """No duplicated nodes, no hanging facets, every fibre/matrix interface facet shared once.

    Returns the number of fibre/matrix interface facets.
    """
    size = float(np.max(upper - lower))
    tolerance = 1e-9 * size
    used = np.unique(cells)
    duplicates = cKDTree(points[used]).query_pairs(tolerance)
    assert not duplicates, f"{len(duplicates)} pairs of distinct nodes share coordinates"

    facets = _facets(cells, dim)
    owners = np.tile(np.arange(cells.shape[0]), dim + 1)
    unique, inverse, counts = np.unique(facets, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    assert counts.max() <= 2, "a facet is shared by more than two cells"
    # Facets used once must lie on the outer boundary; anything else is a crack/non-conformity.
    single = unique[counts == 1]
    coordinates = points[single]
    on_boundary = np.zeros(single.shape[0], dtype=bool)
    for axis in range(dim):
        on_boundary |= np.all(np.abs(coordinates[:, :, axis] - lower[axis]) < tolerance, axis=1)
        on_boundary |= np.all(np.abs(coordinates[:, :, axis] - upper[axis]) < tolerance, axis=1)
    assert np.all(on_boundary), f"{int(np.sum(~on_boundary))} interior facets are unmatched"

    # Interface facets: shared by one fibre cell and one matrix cell, and meshed only once.
    owner_tags = tags[owners]
    has_fibre = np.zeros(unique.shape[0], dtype=bool)
    has_matrix = np.zeros(unique.shape[0], dtype=bool)
    np.logical_or.at(has_fibre, inverse, owner_tags == FIBRE)
    np.logical_or.at(has_matrix, inverse, owner_tags == MATRIX)
    interface = has_fibre & has_matrix
    assert np.all(counts[interface] == 2)
    return int(np.count_nonzero(interface))


def _assert_periodic_nodes(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    """Every node on a max face has a partner on the min face (translated) and vice versa."""
    size = float(np.max(upper - lower))
    tolerance = 1e-9 * size
    for axis in range(points.shape[1]):
        on_min = points[np.abs(points[:, axis] - lower[axis]) < tolerance]
        on_max = points[np.abs(points[:, axis] - upper[axis]) < tolerance]
        assert on_min.shape[0] > 0 and on_max.shape[0] > 0
        shift = np.zeros(points.shape[1])
        shift[axis] = upper[axis] - lower[axis]
        to_min, _ = cKDTree(on_min).query(on_max - shift)
        to_max, _ = cKDTree(on_max).query(on_min + shift)
        assert to_min.max() <= tolerance, f"axis {axis}: max-face node without partner"
        assert to_max.max() <= tolerance, f"axis {axis}: min-face node without partner"


def _boundary_nodes_touching_fibres(
    points: np.ndarray,
    cells: np.ndarray,
    tags: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    axis: int,
) -> int:
    tolerance = 1e-9 * float(np.max(upper - lower))
    fibre_nodes = np.unique(cells[tags == FIBRE])
    coordinate = points[fibre_nodes, axis]
    on_face = (np.abs(coordinate - lower[axis]) < tolerance) | (
        np.abs(coordinate - upper[axis]) < tolerance
    )
    return int(np.count_nonzero(on_face))


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)


def _blob(center_x: float, center_y: float, radius: float, phase: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    radii = radius * (1.0 + 0.15 * np.sin(3.0 * angles + phase))
    points = np.column_stack([center_x + radii * np.cos(angles), center_y + radii * np.sin(angles)])
    return np.vstack([points, points[:1]])


# Image-like polygons on a 20 x 20 window: crossing faces, a corner, inside and fully outside.
WINDOW_POLYGONS = [
    _square(15.0, 5.0, 25.0, 12.0),
    _square(-3.0, 14.0, 4.0, 18.0),
    _blob(19.0, 19.0, 3.0, 0.3),
    _blob(10.0, 10.0, 3.0, 1.1),
    _square(25.0, 25.0, 35.0, 35.0),
    _blob(10.0, 21.5, 2.5, 2.0),
]


def _mesh(geometry: GeometryModel, path: Path, size_min: float, size_max: float) -> Path:
    config = MeshConfig(element_size_min=size_min, element_size_max=size_max, verbosity=0)
    return build_mesh_with_gmsh(geometry, config, path).mesh_path


def _assert_inside(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    tolerance = 1e-9 * float(np.max(upper - lower))
    assert np.all(points.min(axis=0) >= lower - tolerance)
    assert np.all(points.max(axis=0) <= upper + tolerance)


# --------------------------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------------------------


def test_polygons_larger_than_the_domain_are_clipped_2d(tmp_path: Path) -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=20.0, height=20.0),
        polygonal_fibres=[
            PolygonFibre(points=points, fibre_id=index + 1)
            for index, points in enumerate(WINDOW_POLYGONS)
        ],
    )
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "poly.msh", 0.2, 1.0))
    lower, upper = np.zeros(2), np.full(2, 20.0)
    _assert_inside(points, lower, upper)
    assert np.isclose(points[:, 0].max(), 20.0) and np.isclose(points[:, 1].max(), 20.0)
    assert _meshed_fibre_fraction(points, cells, tags, dim) == pytest.approx(
        geometry.fibre_volume_fraction, rel=1e-9
    )
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0


def test_extruded_polygons_larger_than_the_domain_are_clipped_3d(tmp_path: Path) -> None:
    geometry = GeometryModel(
        domain=Domain3D(width=20.0, height=20.0, depth=4.0),
        extruded_polygonal_fibres=[
            ExtrudedPolygonFibre(
                points=points,
                z_min=-1.0 if index == 0 else 0.0,
                z_max=5.0 if index == 2 else 4.0,
                fibre_id=index + 1,
            )
            for index, points in enumerate(WINDOW_POLYGONS)
        ],
        boundary_labels=dict(LABELS_3D),
    )
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "extruded.msh", 0.5, 2.0))
    lower, upper = np.zeros(3), np.array([20.0, 20.0, 4.0])
    _assert_inside(points, lower, upper)
    assert _meshed_fibre_fraction(points, cells, tags, dim) == pytest.approx(
        geometry.fibre_volume_fraction, rel=1e-9
    )
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0


# Circles on an 8 x 8 domain: crossing a face, containing a corner, inside, outside, and
# mostly outside.
CROSSING_CIRCLES = [
    (7.6, 3.0, 1.2),
    (0.3, 0.4, 1.2),
    (4.0, 5.0, 1.2),
    (12.0, 4.0, 1.2),
    (4.0, 8.7, 1.2),
]
CIRCLE_DOMAIN = 8.0


def test_circles_crossing_the_domain_are_clipped(tmp_path: Path) -> None:
    geometry = GeometryModel(
        domain=Domain2D(width=CIRCLE_DOMAIN, height=CIRCLE_DOMAIN),
        circular_fibres=[
            CircleFibre(center_x=x, center_y=y, radius=r, fibre_id=index + 1)
            for index, (x, y, r) in enumerate(CROSSING_CIRCLES)
        ],
    )
    unclipped = sum(math.pi * r**2 for _, _, r in CROSSING_CIRCLES) / CIRCLE_DOMAIN**2
    assert geometry.fibre_volume_fraction < 0.6 * unclipped
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "circles.msh", 0.04, 0.075))
    lower, upper = np.zeros(2), np.full(2, CIRCLE_DOMAIN)
    _assert_inside(points, lower, upper)
    # About 100 segments per circle: the inscribed polygons lose ~7e-4 of the fibre area,
    # which is all that separates mesh and geometry.
    assert _meshed_fibre_fraction(points, cells, tags, dim) == pytest.approx(
        geometry.fibre_volume_fraction, rel=1e-3
    )
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0


def test_cylinders_crossing_the_domain_are_clipped(tmp_path: Path) -> None:
    geometry = GeometryModel(
        domain=Domain3D(width=CIRCLE_DOMAIN, height=CIRCLE_DOMAIN, depth=0.5),
        cylindrical_fibres=[
            CylinderFibre(
                center_x=x, center_y=y, radius=r, z_min=0.0, z_max=0.5, fibre_id=index + 1
            )
            for index, (x, y, r) in enumerate(CROSSING_CIRCLES)
        ],
        boundary_labels=dict(LABELS_3D),
    )
    unclipped = sum(math.pi * r**2 for _, _, r in CROSSING_CIRCLES) / CIRCLE_DOMAIN**2
    assert geometry.fibre_volume_fraction < 0.6 * unclipped
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "cylinders.msh", 0.1, 0.2))
    lower, upper = np.zeros(3), np.array([CIRCLE_DOMAIN, CIRCLE_DOMAIN, 0.5])
    _assert_inside(points, lower, upper)
    # About 38 segments per circle: faceting loses ~5e-3 of the fibre volume (the exact 3D
    # clipping check is the extruded-polygon test above).
    meshed = _meshed_fibre_fraction(points, cells, tags, dim)
    assert meshed == pytest.approx(geometry.fibre_volume_fraction, rel=1e-2)
    assert meshed < geometry.fibre_volume_fraction
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0


def test_image_window_smaller_than_the_image_is_cropped(tmp_path: Path) -> None:
    from skimage.io import imsave

    from rve2d.image_import.mask_to_geometry import import_mask_geometry

    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[15:25, 0:10] = 255
    mask[30:40, 30:40] = 255
    mask[2:8, 15:28] = 255
    mask[24:34, 14:26] = 255  # crosses the window corner region
    image_path = tmp_path / "mask.png"
    imsave(image_path, mask, check_contrast=False)
    config = ImageImportConfig(
        image_path=str(image_path),
        pixel_size=1.0,
        simplify_tolerance=0.0,
        min_artifact_area_px=1,
        domain_width=20.0,
        domain_height=20.0,
    )
    geometry = import_mask_geometry(config).geometry
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "window.msh", 0.5, 1.5))
    lower, upper = np.zeros(2), np.full(2, 20.0)
    _assert_inside(points, lower, upper)
    centroids = points[cells].mean(axis=1)
    assert np.all(centroids <= 20.0)
    assert np.count_nonzero(tags == FIBRE) > 0
    assert _meshed_fibre_fraction(points, cells, tags, dim) == pytest.approx(
        geometry.fibre_volume_fraction, rel=1e-9
    )
    _assert_conforming(points, cells, tags, dim, lower, upper)


# --------------------------------------------------------------------------------------------
# Periodic wrapping
# --------------------------------------------------------------------------------------------


def _wrapped_config(**overrides: Any) -> SyntheticGenerationConfig:
    values: dict[str, Any] = {
        "domain_width": 42.0,
        "domain_height": 42.0,
        "fibre_radius": 3.5,
        "target_volume_fraction": 0.6,
        "min_spacing": 0.5,
        "random_seed": 1,
        "periodic_compatible": True,
        "periodic_wrapping": True,
        "packing_algorithm": "relaxation",
    }
    values.update(overrides)
    return SyntheticGenerationConfig(**values)


def test_wrapped_fibres_give_node_matched_periodic_mesh_2d(tmp_path: Path) -> None:
    geometry = generate_circular_fibre_rve(_wrapped_config()).geometry
    assert geometry.fibre_instance_count > geometry.fibre_count
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "wrapped2d.msh", 0.5, 1.0))
    lower, upper = np.zeros(2), np.full(2, 42.0)
    _assert_inside(points, lower, upper)
    _assert_periodic_nodes(points, lower, upper)
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0
    # Fibre pieces meet the periodic faces, so the boundaries are split by fibres.
    for axis in range(2):
        assert _boundary_nodes_touching_fibres(points, cells, tags, lower, upper, axis) > 0
    # ~22 segments per fibre: the inscribed polygons lose ~1.4 % of the fibre area.
    meshed = _meshed_fibre_fraction(points, cells, tags, dim)
    assert meshed == pytest.approx(geometry.fibre_volume_fraction, rel=0.03)
    assert meshed < geometry.fibre_volume_fraction


def test_wrapped_fibres_give_node_matched_periodic_mesh_3d(tmp_path: Path) -> None:
    config = _wrapped_config(domain_width=28.0, domain_height=28.0, domain_depth=3.0, random_seed=2)
    geometry = generate_cylindrical_fibre_rve(config).geometry
    assert geometry.fibre_instance_count > geometry.fibre_count
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "wrapped3d.msh", 0.5, 1.5))
    lower, upper = np.zeros(3), np.array([28.0, 28.0, 3.0])
    _assert_inside(points, lower, upper)
    _assert_periodic_nodes(points, lower, upper)
    assert _assert_conforming(points, cells, tags, dim, lower, upper) > 0
    for axis in range(2):
        assert _boundary_nodes_touching_fibres(points, cells, tags, lower, upper, axis) > 0
    meshed = _meshed_fibre_fraction(points, cells, tags, dim)
    assert meshed == pytest.approx(geometry.fibre_volume_fraction, rel=0.05)


def test_periodic_meshing_rejects_a_fibre_without_its_image(tmp_path: Path) -> None:
    domain = Domain2D(width=1.0, height=1.0)
    geometry = GeometryModel(
        domain=domain,
        circular_fibres=[CircleFibre(center_x=0.02, center_y=0.5, radius=0.1, fibre_id=1)],
        periodic_pairs=[
            PeriodicBoundaryPair("x_periodic", "left", "right", (1.0, 0.0)),
            PeriodicBoundaryPair("y_periodic", "bottom", "top", (0.0, 1.0)),
        ],
    )
    with pytest.raises(MeshingError, match="not periodic"):
        _mesh(geometry, tmp_path / "broken.msh", 0.02, 0.05)


@pytest.mark.parametrize(
    "example",
    ["examples/2d/synthetic/high_vf_periodic.yaml", "examples/3d/synthetic/high_vf_periodic.yaml"],
)
def test_high_volume_fraction_examples_build(example: str, tmp_path: Path) -> None:
    from rve2d.workflow import build_rve

    config = load_config(ROOT / example)
    result = build_rve(config, output_dir=str(tmp_path), basename="rve")
    assert result.quality_report.valid, result.quality_report.warnings
    msh = next(path for path in result.mesh_files if path.suffix == ".msh")
    points, cells, tags, dim = _read_mesh(msh)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    _assert_periodic_nodes(points, lower, upper)
    _assert_conforming(points, cells, tags, dim, lower, upper)
    assert _meshed_fibre_fraction(points, cells, tags, dim) > 0.58


@pytest.mark.parametrize("dimension", [2, 3])
def test_fibre_containing_a_corner_meshes_periodically(dimension: int, tmp_path: Path) -> None:
    from rve2d.synthetic_generation.packing import periodic_image_shifts

    size, depth = 10.0, 1.5
    placements: list[tuple[int, float, float]] = []
    for fibre_id, (x, y) in enumerate([(0.5, 0.6), (9.2, 5.0), (5.0, 5.0)], start=1):
        placements.append((fibre_id, x, y))
        for shift_x, shift_y in periodic_image_shifts(x, y, 1.5, 0.0, 0.0, size, size):
            placements.append((fibre_id, x + shift_x, y + shift_y))
    assert len(placements) == 3 + 3 + 1  # corner fibre: 3 images, face fibre: 1 image
    if dimension == 2:
        geometry = GeometryModel(
            domain=Domain2D(width=size, height=size),
            circular_fibres=[
                CircleFibre(center_x=x, center_y=y, radius=1.5, fibre_id=i)
                for i, x, y in placements
            ],
            periodic_pairs=[
                PeriodicBoundaryPair("x_periodic", "left", "right", (size, 0.0)),
                PeriodicBoundaryPair("y_periodic", "bottom", "top", (0.0, size)),
            ],
        )
        lower, upper = np.zeros(2), np.full(2, size)
    else:
        geometry = GeometryModel(
            domain=Domain3D(width=size, height=size, depth=depth),
            cylindrical_fibres=[
                CylinderFibre(
                    center_x=x, center_y=y, radius=1.5, z_min=0.0, z_max=depth, fibre_id=i
                )
                for i, x, y in placements
            ],
            boundary_labels=dict(LABELS_3D),
            periodic_pairs=[
                PeriodicBoundaryPair("x_periodic", "left", "right", (size, 0.0, 0.0)),
                PeriodicBoundaryPair("y_periodic", "front", "back", (0.0, size, 0.0)),
                PeriodicBoundaryPair("z_periodic", "bottom", "top", (0.0, 0.0, depth)),
            ],
        )
        lower, upper = np.zeros(3), np.array([size, size, depth])
    assert geometry.fibre_volume_fraction == pytest.approx(3 * math.pi * 1.5**2 / size**2)
    points, cells, tags, dim = _read_mesh(_mesh(geometry, tmp_path / "corner.msh", 0.2, 0.5))
    _assert_inside(points, lower, upper)
    _assert_periodic_nodes(points, lower, upper)
    _assert_conforming(points, cells, tags, dim, lower, upper)
    corner_nodes = np.all(np.abs(points[:, :2]) < 1e-9, axis=1)
    fibre_nodes = np.zeros(points.shape[0], dtype=bool)
    fibre_nodes[np.unique(cells[tags == FIBRE])] = True
    assert np.any(corner_nodes & fibre_nodes)  # the corner lies inside the wrapped fibre
