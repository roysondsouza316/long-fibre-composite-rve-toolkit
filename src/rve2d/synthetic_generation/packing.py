"""Placement of equal circular fibre cross-sections in a rectangle.

Shared by the 2D (circles) and 3D (cylinders along z) synthetic generators. Two algorithms
are available (see :class:`rve2d.config.SyntheticGenerationConfig`):

* ``random_sequential``: random sequential adsorption (RSA). Without periodic wrapping this is
  the historical implementation and reproduces earlier layouts bit for bit.
* ``relaxation``: all fibres are dropped at random positions, then overlaps are removed by
  iteratively pushing apart every pair closer than ``2 * radius + min_spacing`` (collective
  rearrangement with periodic minimum-image distances when wrapping), followed by Monte Carlo
  "shaking" sweeps that randomise the contact network (in the spirit of Melro et al., 2008).

With ``periodic_wrapping`` fibre centres live in ``[origin, origin + size)`` and fibres may
cross the domain boundary. Positions where a fibre surface would come within
``boundary_clearance`` of a domain face or corner it does not cross are excluded, because such
near-tangent cuts create sliver elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rve2d.config import MeshConfig, SyntheticGenerationConfig
from rve2d.exceptions import RVEError

FloatArray = NDArray[np.float64]

#: Default ``boundary_clearance`` as a fraction of the fibre radius (when ``min_spacing`` is
#: smaller), used for wrapping and relaxation layouts.
DEFAULT_BOUNDARY_CLEARANCE_FRACTION = 0.1
#: Upper bound of the default boundary clearance as a fraction of the fibre radius.
MAX_DEFAULT_BOUNDARY_CLEARANCE_FRACTION = 0.5
#: Densest packing fraction of equal disks (hexagonal); used to reject impossible targets early.
_HEXAGONAL_PACKING_FRACTION = math.pi / (2.0 * math.sqrt(3.0))
# Pairs closer than contact * (1 + _DETECT) are pushed to contact * (1 + _PUSH). Pushing a bit
# beyond the detection threshold guarantees progress despite rounding.
_DETECT = 1e-9
_PUSH = 1e-6


@dataclass(frozen=True)
class FibrePacking:
    """Fibre centres produced by :func:`pack_fibre_centres`.

    ``centres`` holds one canonical centre per fibre (fibre ``i`` gets ``fibre_id = i + 1``).
    With periodic wrapping a canonical centre lies in ``[origin, origin + size)`` and the
    fibre may cross the boundary; :meth:`placements` adds the periodic images needed to cover
    the part of the fibre that wraps around.
    """

    centres: FloatArray
    radius: float
    origin_x: float
    origin_y: float
    width: float
    height: float
    algorithm: str
    periodic_wrapping: bool
    boundary_clearance: float
    attempts: int

    @property
    def fibre_count(self) -> int:
        return int(self.centres.shape[0])

    def metadata(self) -> dict[str, object]:
        """Provenance of the layout, stored in ``GeometryModel.metadata``."""
        return {
            "packing_algorithm": self.algorithm,
            "periodic_wrapping": self.periodic_wrapping,
            "boundary_clearance": self.boundary_clearance,
            "periodic_image_count": len(self.placements()) - self.fibre_count,
            "placement_attempts": self.attempts,
        }

    def placements(self) -> list[tuple[int, float, float]]:
        """Return ``(fibre_id, center_x, center_y)`` for every fibre and periodic image."""
        placements: list[tuple[int, float, float]] = []
        for index, (center_x, center_y) in enumerate(self.centres.tolist()):
            fibre_id = index + 1
            placements.append((fibre_id, center_x, center_y))
            if not self.periodic_wrapping:
                continue
            for shift_x, shift_y in periodic_image_shifts(
                center_x,
                center_y,
                self.radius,
                self.origin_x,
                self.origin_y,
                self.width,
                self.height,
            ):
                placements.append((fibre_id, center_x + shift_x, center_y + shift_y))
        return placements


def periodic_image_shifts(
    center_x: float,
    center_y: float,
    radius: float,
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    """Translations (other than zero) of a disk that still intersect the domain interior.

    A disk with its centre inside the domain needs an image for every face it crosses and,
    when it contains a domain corner, for that corner as well.
    """
    shifts: list[tuple[float, float]] = []
    for shift_x in (-width, 0.0, width):
        for shift_y in (-height, 0.0, height):
            if shift_x == 0.0 and shift_y == 0.0:
                continue
            image_x = center_x + shift_x
            image_y = center_y + shift_y
            nearest_x = min(max(image_x, origin_x), origin_x + width)
            nearest_y = min(max(image_y, origin_y), origin_y + height)
            if math.hypot(image_x - nearest_x, image_y - nearest_y) < radius:
                shifts.append((shift_x, shift_y))
    return shifts


def resolve_boundary_clearance(config: SyntheticGenerationConfig) -> float:
    """Boundary clearance applied by the generator (see ``SyntheticGenerationConfig``)."""
    if config.boundary_clearance is not None:
        return float(config.boundary_clearance)
    if config.packing_algorithm == "random_sequential" and not config.periodic_wrapping:
        return 0.0
    return min(
        max(config.min_spacing, DEFAULT_BOUNDARY_CLEARANCE_FRACTION * config.fibre_radius),
        MAX_DEFAULT_BOUNDARY_CLEARANCE_FRACTION * config.fibre_radius,
    )


def resolve_fibre_count(
    config: SyntheticGenerationConfig,
    domain_measure: float,
    fibre_measure: float,
) -> int:
    """Requested fibre count from ``fibre_count`` or ``target_volume_fraction``."""
    if config.fibre_count is not None:
        return config.fibre_count
    count = int(round(config.target_volume_fraction * domain_measure / fibre_measure))
    return max(count, 1)


def pack_fibre_centres(
    config: SyntheticGenerationConfig,
    fibre_count: int,
    width: float,
    height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> FibrePacking:
    """Place ``fibre_count`` equal fibre cross-sections according to ``config``."""
    rng = np.random.default_rng(config.random_seed)
    clearance = resolve_boundary_clearance(config)
    if config.periodic_wrapping:
        _check_wrapping_inputs(config, width, height, clearance)
    if config.packing_algorithm == "relaxation":
        centres, attempts = _relaxation_packing(
            config, fibre_count, width, height, origin_x, origin_y, clearance, rng
        )
    elif config.periodic_wrapping:
        centres, attempts = _random_sequential_wrapped(
            config, fibre_count, width, height, origin_x, origin_y, clearance, rng
        )
    else:
        centres, attempts = _random_sequential_legacy(
            config, fibre_count, width, height, origin_x, origin_y, clearance, rng
        )
    return FibrePacking(
        centres=centres,
        radius=config.fibre_radius,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
        algorithm=config.packing_algorithm,
        periodic_wrapping=config.periodic_wrapping,
        boundary_clearance=clearance,
        attempts=attempts,
    )


def mesh_compatibility_warnings(
    config: SyntheticGenerationConfig,
    mesh_config: MeshConfig,
) -> list[str]:
    """Warn when gaps the generator may create are thinner than about one mesh element.

    Fibre pairs closer than about one element, and fibre surfaces closer than that to a domain
    face, are meshed with sliver elements. The relaxation algorithm leaves many pairs exactly
    ``min_spacing`` apart, so there ``min_spacing`` itself must be resolved by the mesh.
    """
    element_size = mesh_config.element_size_min
    warnings: list[str] = []
    if config.packing_algorithm == "relaxation" and config.min_spacing < element_size:
        warnings.append(
            f"min_spacing {config.min_spacing:g} is smaller than mesh.element_size_min "
            f"{element_size:g}: the relaxation packing leaves many fibre pairs exactly "
            "min_spacing apart and the elements bridging those gaps will be slivers. "
            "Increase min_spacing to about one element size or refine the mesh."
        )
    clearance = resolve_boundary_clearance(config)
    if (config.periodic_wrapping or config.packing_algorithm == "relaxation") and (
        clearance < element_size
    ):
        warnings.append(
            f"boundary_clearance {clearance:g} is smaller than mesh.element_size_min "
            f"{element_size:g}: fibres passing that close to a domain face or corner (or "
            "crossing it by that little) produce sliver elements. Increase boundary_clearance "
            "to about one element size or refine the mesh."
        )
    return warnings


def _check_wrapping_inputs(
    config: SyntheticGenerationConfig,
    width: float,
    height: float,
    clearance: float,
) -> None:
    if not config.periodic_compatible:
        raise RVEError("periodic_wrapping requires periodic_compatible: true.")
    contact = 2.0 * config.fibre_radius + config.min_spacing
    if min(width, height) < 2.0 * contact:
        raise RVEError(
            "periodic_wrapping requires the domain to be at least 2 * (2 * fibre_radius + "
            "min_spacing) wide and high."
        )
    if clearance >= config.fibre_radius:
        raise RVEError("With periodic_wrapping, boundary_clearance must be below fibre_radius.")


# --------------------------------------------------------------------------------------------
# Random sequential adsorption
# --------------------------------------------------------------------------------------------


def _random_sequential_legacy(
    config: SyntheticGenerationConfig,
    fibre_count: int,
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
    clearance: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    # Historical algorithm: keep every floating-point operation and random draw unchanged so
    # that existing configs reproduce their geometry exactly.
    edge_clearance = config.edge_clearance
    if config.boundary_clearance is not None:
        edge_clearance = max(edge_clearance, clearance)
    x_max = origin_x + width
    y_max = origin_y + height
    lower_x = origin_x + config.fibre_radius + edge_clearance
    lower_y = origin_y + config.fibre_radius + edge_clearance
    upper_x = x_max - config.fibre_radius - edge_clearance
    upper_y = y_max - config.fibre_radius - edge_clearance
    if lower_x >= upper_x or lower_y >= upper_y:
        raise RVEError("Fibre radius and edge clearance leave no room for placement.")

    min_center_distance = 2.0 * config.fibre_radius + config.min_spacing
    centres: list[tuple[float, float]] = []
    attempts = 0
    while len(centres) < fibre_count and attempts < config.max_attempts:
        attempts += 1
        candidate_x = float(rng.uniform(lower_x, upper_x))
        candidate_y = float(rng.uniform(lower_y, upper_y))
        if _is_non_overlapping(candidate_x, candidate_y, centres, min_center_distance):
            centres.append((candidate_x, candidate_y))

    if len(centres) != fibre_count:
        raise RVEError(
            _random_sequential_failure_message(config, fibre_count, len(centres), width, height)
        )
    return np.asarray(centres, dtype=np.float64).reshape(-1, 2), attempts


def _is_non_overlapping(
    candidate_x: float,
    candidate_y: float,
    centres: list[tuple[float, float]],
    min_center_distance: float,
) -> bool:
    min_distance_sq = min_center_distance**2
    for center_x, center_y in centres:
        delta_x = candidate_x - center_x
        delta_y = candidate_y - center_y
        if delta_x * delta_x + delta_y * delta_y < min_distance_sq:
            return False
    return True


def _random_sequential_wrapped(
    config: SyntheticGenerationConfig,
    fibre_count: int,
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
    clearance: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    radius = config.fibre_radius
    box = np.array([width, height], dtype=np.float64)
    min_distance_sq = (2.0 * radius + config.min_spacing) ** 2
    placed = np.empty((fibre_count, 2), dtype=np.float64)
    count = 0
    attempts = 0
    batch = 256
    while count < fibre_count and attempts < config.max_attempts:
        # Draw candidates in batches; the boundary-band test is vectorised, the overlap test
        # is sequential because every accepted fibre constrains the next candidates.
        draws = min(batch, config.max_attempts - attempts)
        candidates = np.minimum(
            rng.uniform(0.0, 1.0, size=(draws, 2)) * box, np.nextafter(box, 0.0)
        )
        clear = _clear_of_boundary(candidates, radius, box, clearance)
        for index in range(draws):
            attempts += 1
            if not clear[index]:
                continue
            candidate = candidates[index]
            if count:
                delta = placed[:count] - candidate
                delta -= box * np.round(delta / box)
                if float(np.min(np.einsum("ij,ij->i", delta, delta))) < min_distance_sq:
                    continue
            placed[count] = candidate
            count += 1
            if count == fibre_count:
                break
    if count != fibre_count:
        raise RVEError(
            _random_sequential_failure_message(config, fibre_count, count, width, height)
        )
    placed[:, 0] += origin_x
    placed[:, 1] += origin_y
    return placed, attempts


def _random_sequential_failure_message(
    config: SyntheticGenerationConfig,
    fibre_count: int,
    placed: int,
    width: float,
    height: float,
) -> str:
    reached = placed * math.pi * config.fibre_radius**2 / (width * height)
    requested = fibre_count * math.pi * config.fibre_radius**2 / (width * height)
    advice = (
        "Set `packing_algorithm: relaxation` (requires `min_spacing > 0`) to reach fibre "
        "volume fractions of 0.6 and more"
    )
    if config.periodic_compatible and not config.periodic_wrapping:
        advice += (
            "; for periodic RVEs also set `periodic_wrapping: true` so that fibres may cross "
            "the domain boundary instead of leaving a matrix-only band along the edges"
        )
    return (
        f"Could not place {fibre_count} fibres without overlap after {config.max_attempts} "
        f"attempts: random sequential placement stopped at {placed} fibres (fibre volume "
        f"fraction {reached:.3f} of the requested {requested:.3f}) and saturates around "
        f"0.45-0.5. {advice}. Alternatively lower target_volume_fraction or min_spacing."
    )


# --------------------------------------------------------------------------------------------
# Overlap relaxation
# --------------------------------------------------------------------------------------------


def _relaxation_packing(
    config: SyntheticGenerationConfig,
    fibre_count: int,
    width: float,
    height: float,
    origin_x: float,
    origin_y: float,
    clearance: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    radius = config.fibre_radius
    contact = 2.0 * radius + config.min_spacing
    wrap = config.periodic_wrapping
    box = np.array([width, height], dtype=np.float64)
    if wrap:
        lower = np.zeros(2)
        upper = box.copy()
        available_area = width * height
    else:
        wall = max(config.edge_clearance, clearance)
        lower = np.array([radius + wall, radius + wall])
        upper = box - radius - wall
        if np.any(lower >= upper):
            raise RVEError("Fibre radius and edge clearance leave no room for placement.")
        # Fibre centres keep (radius + wall) from the faces; a gap of half the contact distance
        # is the share of free space a fibre near a face can use.
        available_area = float(np.prod(upper - lower + contact))
    packing_fraction = fibre_count * math.pi * (0.5 * contact) ** 2 / available_area
    if packing_fraction >= _HEXAGONAL_PACKING_FRACTION:
        raise RVEError(
            f"Cannot place {fibre_count} fibres of radius {radius:g} with min_spacing "
            f"{config.min_spacing:g}: even a perfect hexagonal arrangement would overlap "
            f"(effective packing fraction {packing_fraction:.3f} > "
            f"{_HEXAGONAL_PACKING_FRACTION:.3f}). Lower target_volume_fraction or min_spacing."
        )

    positions = lower + rng.uniform(0.0, 1.0, size=(fibre_count, 2)) * (upper - lower)
    if wrap:
        positions = _wrap(positions, box)
    detect = contact * (1.0 + _DETECT)
    push = contact * (1.0 + _PUSH)
    iterations = 0
    converged = False
    while iterations < config.max_relaxation_iterations:
        iterations += 1
        displacement, violations = _overlap_displacement(
            positions, box if wrap else None, detect, push, rng
        )
        candidate = positions + displacement
        if wrap:
            candidate = _wrap(candidate, box)
            candidate, band_violations = _project_out_of_bands(candidate, radius, box, clearance)
            violations += band_violations
            candidate = _wrap(candidate, box)
        else:
            candidate = np.clip(candidate, lower, upper)
        if violations == 0:
            converged = True
            break
        positions = candidate
    if not converged:
        raise RVEError(
            f"Relaxation packing did not remove all overlaps of {fibre_count} fibres within "
            f"{config.max_relaxation_iterations} iterations (effective packing fraction "
            f"{packing_fraction:.3f} including min_spacing; random packings of equal disks "
            "jam near 0.82). Lower target_volume_fraction or min_spacing, or increase "
            "max_relaxation_iterations."
        )

    if config.shake_sweeps > 0 and fibre_count > 1:
        positions = _shake(
            positions,
            radius,
            contact,
            box,
            lower,
            upper,
            clearance,
            wrap,
            config.shake_sweeps,
            rng,
        )
    _verify_packing(positions, radius, contact, box, lower, upper, clearance, wrap)
    positions = positions + np.array([origin_x, origin_y])
    return positions, iterations


def _overlap_displacement(
    positions: FloatArray,
    box: FloatArray | None,
    detect: float,
    push: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    displacement = np.zeros_like(positions)
    pairs = _close_pairs(positions, box, detect)
    if pairs.shape[0] == 0:
        return displacement, 0
    first = pairs[:, 0]
    second = pairs[:, 1]
    delta = positions[second] - positions[first]
    if box is not None:
        delta -= box * np.round(delta / box)
    distance = np.hypot(delta[:, 0], delta[:, 1])
    coincident = distance < 1e-12 * push
    if np.any(coincident):
        angles = rng.uniform(0.0, 2.0 * math.pi, size=int(np.count_nonzero(coincident)))
        delta[coincident] = np.column_stack([np.cos(angles), np.sin(angles)])
        distance[coincident] = 1.0
    unit = delta / distance[:, None]
    overlap = push - np.where(coincident, 0.0, distance)
    step = 0.5 * overlap[:, None] * unit
    np.add.at(displacement, first, -step)
    np.add.at(displacement, second, step)
    return displacement, int(pairs.shape[0])


def _close_pairs(
    positions: FloatArray,
    box: FloatArray | None,
    distance: float,
) -> NDArray[np.intp]:
    if positions.shape[0] < 2:
        return np.empty((0, 2), dtype=np.intp)
    tree = cKDTree(positions, boxsize=box) if box is not None else cKDTree(positions)
    pairs = np.asarray(tree.query_pairs(distance, output_type="ndarray"), dtype=np.intp)
    if pairs.shape[0] == 0:
        return pairs.reshape(0, 2)
    # query_pairs does not guarantee an order; sort for reproducible floating-point sums.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _wrap(positions: FloatArray, box: FloatArray) -> FloatArray:
    wrapped = positions - box * np.floor(positions / box)
    # Rounding can map -tiny to exactly the period; keep coordinates in [0, period).
    wrapped = np.where(wrapped >= box, wrapped - box, wrapped)
    return np.where(wrapped < 0.0, 0.0, wrapped)


def _band_edges(radius: float, clearance: float) -> tuple[float, float]:
    margin = 1e-7 * radius
    return radius - clearance - margin, radius + clearance + margin


def _clear_of_boundary(
    positions: FloatArray,
    radius: float,
    box: FloatArray,
    clearance: float,
) -> NDArray[np.bool_]:
    """True where a wrapped fibre keeps ``clearance`` from every face and corner it does not
    cross (positions relative to the domain origin, inside ``[0, box)``)."""
    ok = np.ones(positions.shape[0], dtype=bool)
    if clearance <= 0.0:
        return ok
    for axis in (0, 1):
        coordinate = positions[:, axis]
        to_face = np.minimum(coordinate, box[axis] - coordinate)
        ok &= np.abs(to_face - radius) >= clearance
    corner = box * np.round(positions / box)
    to_corner = np.hypot(*(positions - corner).T)
    ok &= np.abs(to_corner - radius) >= clearance
    return ok


def _project_out_of_bands(
    positions: FloatArray,
    radius: float,
    box: FloatArray,
    clearance: float,
) -> tuple[FloatArray, int]:
    """Move centres out of the near-tangent bands along faces and rings around corners."""
    if clearance <= 0.0:
        return positions, 0
    band_low, band_high = _band_edges(radius, clearance)
    projected = positions.copy()
    violations = 0
    for axis in (0, 1):
        coordinate = projected[:, axis]
        near_low_face = coordinate < 0.5 * box[axis]
        to_face = np.where(near_low_face, coordinate, box[axis] - coordinate)
        bad = np.abs(to_face - radius) < clearance
        if np.any(bad):
            violations += int(np.count_nonzero(bad))
            go_inside = (to_face < radius) & (band_low > 0.0)
            target = np.where(go_inside, band_low, band_high)
            new_coordinate = np.where(near_low_face, target, box[axis] - target)
            projected[bad, axis] = new_coordinate[bad]
    corner = box * np.round(projected / box)
    offset = projected - corner
    to_corner = np.hypot(offset[:, 0], offset[:, 1])
    bad = np.abs(to_corner - radius) < clearance
    if np.any(bad):
        violations += int(np.count_nonzero(bad))
        go_inside = (to_corner < radius) & (band_low > 0.0)
        target = np.where(go_inside, band_low, band_high)
        scale = target / np.maximum(to_corner, 1e-300)
        projected[bad] = (corner + offset * scale[:, None])[bad]
    return projected, violations


def _shake(
    positions: FloatArray,
    radius: float,
    contact: float,
    box: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    clearance: float,
    wrap: bool,
    sweeps: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Monte Carlo sweeps: random single-fibre moves accepted when no constraint is violated.

    Relaxation leaves most fibres exactly at contact distance; shaking spreads the
    nearest-neighbour distances without ever violating ``contact`` or the boundary bands.
    """
    positions = positions.copy()
    count = positions.shape[0]
    step = 0.1 * radius
    limit_sq = contact * contact * (1.0 + _DETECT)
    for _ in range(sweeps):
        reach = contact + 3.0 * step
        tree = cKDTree(positions, boxsize=box) if wrap else cKDTree(positions)
        neighbours = tree.query_ball_point(positions, reach)
        order = rng.permutation(count)
        moves = rng.uniform(-step, step, size=(count, 2))
        accepted = 0
        for index in order.tolist():
            proposal = positions[index] + moves[index]
            if wrap:
                proposal = _wrap(proposal[None, :], box)[0]
                if not _clear_of_boundary(proposal[None, :], radius, box, clearance)[0]:
                    continue
            elif np.any(proposal < lower) or np.any(proposal > upper):
                continue
            others = [other for other in neighbours[index] if other != index]
            if others:
                delta = positions[others] - proposal
                if wrap:
                    delta -= box * np.round(delta / box)
                if float(np.min(np.einsum("ij,ij->i", delta, delta))) < limit_sq:
                    continue
            positions[index] = proposal
            accepted += 1
        acceptance = accepted / count
        # Keep the acceptance rate moderate: large steps rarely succeed in dense packings.
        step *= float(np.clip(math.sqrt(max(acceptance, 1e-3) / 0.3), 0.5, 1.5))
        step = float(np.clip(step, 1e-3 * radius, 0.5 * radius))
    return positions


def _verify_packing(
    positions: FloatArray,
    radius: float,
    contact: float,
    box: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    clearance: float,
    wrap: bool,
) -> None:
    pairs = _close_pairs(positions, box if wrap else None, contact)
    if pairs.shape[0]:
        delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
        if wrap:
            delta -= box * np.round(delta / box)
        if float(np.min(np.hypot(delta[:, 0], delta[:, 1]))) < contact:
            raise RVEError("Relaxation packing produced fibres closer than min_spacing.")
    if wrap:
        if not bool(np.all(_clear_of_boundary(positions, radius, box, clearance))):
            raise RVEError("Relaxation packing left a fibre too close to the domain boundary.")
    elif np.any(positions < lower) or np.any(positions > upper):
        raise RVEError("Relaxation packing left a fibre outside the allowed region.")
