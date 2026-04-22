from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from rve2d.exceptions import ConfigError


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    domain_width: float
    domain_height: float
    fibre_radius: float
    target_volume_fraction: float
    domain_depth: float | None = None
    fibre_count: int | None = None
    min_spacing: float = 0.0
    edge_clearance: float = 0.0
    random_seed: int = 0
    max_attempts: int = 100_000
    periodic_compatible: bool = False
    orientation_deg: float = 0.0
    matrix_orientation_angle_x_deg: float | None = None
    matrix_orientation_angle_y_deg: float | None = None
    matrix_orientation_angle_z_deg: float | None = None
    fibre_orientation_angle_x_deg: float | None = None
    fibre_orientation_angle_y_deg: float | None = None
    fibre_orientation_angle_z_deg: float | None = None


@dataclass(frozen=True)
class ImageImportConfig:
    image_path: str
    pixel_size: float = 1.0
    extrusion_depth: float | None = None
    threshold: float = 0.5
    invert: bool = False
    min_artifact_area_px: int = 16
    clear_border: bool = False
    simplify_tolerance: float = 1.0
    domain_width: float | None = None
    domain_height: float | None = None
    periodic_compatible: bool = False
    matrix_orientation_angle_x_deg: float | None = None
    matrix_orientation_angle_y_deg: float | None = None
    matrix_orientation_angle_z_deg: float | None = None
    fibre_orientation_angle_x_deg: float | None = None
    fibre_orientation_angle_y_deg: float | None = None
    fibre_orientation_angle_z_deg: float | None = None


@dataclass(frozen=True)
class SemToSyntheticConfig:
    image_path: str
    pixel_size: float
    threshold: float | None = None
    invert: bool = False
    smoothing_sigma: float = 2.0
    edge_sigma: float = 2.0
    edge_low_threshold: float = 0.08
    edge_high_threshold: float = 0.2
    hough_min_radius_px: int = 18
    hough_max_radius_px: int = 46
    hough_radius_step_px: int = 2
    hough_total_num_peaks: int = 120
    hough_peak_threshold_rel: float = 0.22
    hough_min_center_distance_px: int = 12
    hough_overlap_buffer_px: float = 4.0
    separate_touching_fibres: bool = True
    separation_min_distance_px: int = 16
    separation_peak_threshold_px: float = 6.0
    min_artifact_area_px: int = 32
    min_region_area_px: int = 700
    ellipse_axis_ratio_threshold: float = 1.15
    ellipse_sample_points: int = 48
    max_candidate_overlap_fraction: float = 0.2
    drop_boundary_fibres: bool = True
    clear_border: bool = False
    domain_width: float | None = None
    domain_height: float | None = None
    periodic_compatible: bool = False


@dataclass(frozen=True)
class MeshConfig:
    element_size_min: float = 0.01
    element_size_max: float = 0.05
    algorithm: int = 6
    mesh_order: int = 1
    recombine: bool = False
    verbosity: int = 2


@dataclass(frozen=True)
class ExportConfig:
    output_dir: str = "outputs"
    basename: str = "rve2d"
    formats: list[str] = field(default_factory=lambda: ["msh", "xdmf"])
    write_phase_json: bool = True
    write_quality_json: bool = True
    write_periodic_json: bool = True


@dataclass(frozen=True)
class FerriteSolveConfig:
    enabled: bool = False
    boundary_condition: Literal["dirichlet", "periodic"] = "dirichlet"
    kinematics: Literal["plane_strain", "plane_stress", "solid"] = "plane_strain"
    matrix_material_model: Literal["isotropic", "orthotropic"] = "isotropic"
    matrix_youngs_modulus: float = 3.5e9
    matrix_poisson_ratio: float = 0.35
    matrix_e1: float | None = None
    matrix_e2: float | None = None
    matrix_e3: float | None = None
    matrix_g12: float | None = None
    matrix_g13: float | None = None
    matrix_g23: float | None = None
    matrix_nu12: float | None = None
    matrix_nu13: float | None = None
    matrix_nu23: float | None = None
    matrix_material_angle_deg: float | None = None
    matrix_material_angle_x_deg: float | None = None
    matrix_material_angle_y_deg: float | None = None
    matrix_material_angle_z_deg: float | None = None
    fibre_material_model: Literal["isotropic", "orthotropic"] = "isotropic"
    fibre_youngs_modulus: float = 70.0e9
    fibre_poisson_ratio: float = 0.2
    fibre_e1: float | None = None
    fibre_e2: float | None = None
    fibre_e3: float | None = None
    fibre_g12: float | None = None
    fibre_g13: float | None = None
    fibre_g23: float | None = None
    fibre_nu12: float | None = None
    fibre_nu13: float | None = None
    fibre_nu23: float | None = None
    fibre_material_angle_deg: float | None = None
    fibre_material_angle_x_deg: float | None = None
    fibre_material_angle_y_deg: float | None = None
    fibre_material_angle_z_deg: float | None = None
    matrix_cellset: str = "matrix"
    fibre_cellset: str = "fibre"
    matrix_phase_id: int = 1
    fibre_phase_id: int = 2
    write_vtk: bool = True


@dataclass(frozen=True)
class RVEConfig:
    mode: Literal["synthetic", "image", "sem_to_synthetic"]
    dimension: Literal[2, 3] = 2
    synthetic: SyntheticGenerationConfig | None = None
    image: ImageImportConfig | None = None
    sem_to_synthetic: SemToSyntheticConfig | None = None
    mesh: MeshConfig = field(default_factory=MeshConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    solver: FerriteSolveConfig = field(default_factory=FerriteSolveConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> RVEConfig:
    config_path = Path(path)
    payload = _load_payload(config_path)
    return config_from_dict(payload)


def config_from_dict(payload: dict[str, Any]) -> RVEConfig:
    mode = payload.get("mode")
    if mode not in {"synthetic", "image", "sem_to_synthetic"}:
        raise ConfigError(
            "Config 'mode' must be one of 'synthetic', 'image', or 'sem_to_synthetic'."
        )

    synthetic_payload = payload.get("synthetic")
    image_payload = payload.get("image")
    sem_to_synthetic_payload = payload.get("sem_to_synthetic")
    config = RVEConfig(
        dimension=payload.get("dimension", 2),
        mode=mode,
        synthetic=(
            SyntheticGenerationConfig(**synthetic_payload)
            if synthetic_payload is not None
            else None
        ),
        image=ImageImportConfig(**image_payload) if image_payload is not None else None,
        sem_to_synthetic=(
            SemToSyntheticConfig(**sem_to_synthetic_payload)
            if sem_to_synthetic_payload is not None
            else None
        ),
        mesh=MeshConfig(**payload.get("mesh", {})),
        export=ExportConfig(**payload.get("export", {})),
        solver=FerriteSolveConfig(**payload.get("solver", {})),
    )
    _validate_config(config)
    return config


def _load_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping.")
    return loaded


def _validate_config(config: RVEConfig) -> None:
    if config.dimension not in {2, 3}:
        raise ConfigError("Config 'dimension' must be either 2 or 3.")
    if config.mode == "synthetic":
        if config.synthetic is None:
            raise ConfigError("Synthetic mode requires a 'synthetic' section.")
        _validate_synthetic(config.synthetic, config.dimension)
    if config.mode == "image":
        if config.image is None:
            raise ConfigError("Image mode requires an 'image' section.")
        _validate_image(config.image, config.dimension)
    if config.mode == "sem_to_synthetic":
        if config.sem_to_synthetic is None:
            raise ConfigError("sem_to_synthetic mode requires a 'sem_to_synthetic' section.")
        _validate_sem_to_synthetic(config.sem_to_synthetic, config.dimension)
    if config.mesh.element_size_min <= 0.0 or config.mesh.element_size_max <= 0.0:
        raise ConfigError("Mesh element sizes must be positive.")
    if config.mesh.element_size_min > config.mesh.element_size_max:
        raise ConfigError("Mesh minimum element size cannot exceed the maximum.")
    if not config.export.formats:
        raise ConfigError("At least one export format must be configured.")
    _validate_solver(config)


def _validate_synthetic(config: SyntheticGenerationConfig, dimension: int) -> None:
    if config.domain_width <= 0.0 or config.domain_height <= 0.0:
        raise ConfigError("Synthetic domain dimensions must be positive.")
    if dimension == 3 and (config.domain_depth is None or config.domain_depth <= 0.0):
        raise ConfigError("3D synthetic configs require a positive domain_depth.")
    if config.fibre_radius <= 0.0:
        raise ConfigError("Fibre radius must be positive.")
    if not 0.0 < config.target_volume_fraction < 1.0:
        raise ConfigError("Target volume fraction must be between 0 and 1.")
    if config.fibre_count is not None and config.fibre_count <= 0:
        raise ConfigError("Fibre count must be positive when provided.")
    if config.min_spacing < 0.0 or config.edge_clearance < 0.0:
        raise ConfigError("Spacing and edge clearance must be non-negative.")
    if config.max_attempts <= 0:
        raise ConfigError("max_attempts must be positive.")


def _validate_image(config: ImageImportConfig, dimension: int) -> None:
    if config.pixel_size <= 0.0:
        raise ConfigError("pixel_size must be positive.")
    if dimension == 3 and (config.extrusion_depth is None or config.extrusion_depth <= 0.0):
        raise ConfigError("3D image configs require a positive extrusion_depth.")
    if config.min_artifact_area_px < 0:
        raise ConfigError("min_artifact_area_px must be non-negative.")
    if config.simplify_tolerance < 0.0:
        raise ConfigError("simplify_tolerance must be non-negative.")


def _validate_sem_to_synthetic(config: SemToSyntheticConfig, dimension: int) -> None:
    if dimension != 2:
        raise ConfigError("sem_to_synthetic mode is currently supported only for 2D.")
    if config.pixel_size <= 0.0:
        raise ConfigError("sem_to_synthetic pixel_size must be positive.")
    if config.threshold is not None and not 0.0 <= config.threshold <= 1.0:
        raise ConfigError("sem_to_synthetic threshold must lie between 0 and 1.")
    if config.smoothing_sigma < 0.0:
        raise ConfigError("sem_to_synthetic smoothing_sigma must be non-negative.")
    if config.edge_sigma < 0.0:
        raise ConfigError("sem_to_synthetic edge_sigma must be non-negative.")
    if not 0.0 <= config.edge_low_threshold <= 1.0:
        raise ConfigError("sem_to_synthetic edge_low_threshold must lie between 0 and 1.")
    if not 0.0 <= config.edge_high_threshold <= 1.0:
        raise ConfigError("sem_to_synthetic edge_high_threshold must lie between 0 and 1.")
    if config.edge_low_threshold > config.edge_high_threshold:
        raise ConfigError(
            "sem_to_synthetic edge_low_threshold cannot exceed edge_high_threshold."
        )
    if config.hough_min_radius_px < 1:
        raise ConfigError("sem_to_synthetic hough_min_radius_px must be at least 1.")
    if config.hough_max_radius_px < config.hough_min_radius_px:
        raise ConfigError(
            "sem_to_synthetic hough_max_radius_px cannot be smaller than hough_min_radius_px."
        )
    if config.hough_radius_step_px < 1:
        raise ConfigError("sem_to_synthetic hough_radius_step_px must be at least 1.")
    if config.hough_total_num_peaks < 1:
        raise ConfigError("sem_to_synthetic hough_total_num_peaks must be at least 1.")
    if not 0.0 < config.hough_peak_threshold_rel <= 1.0:
        raise ConfigError("sem_to_synthetic hough_peak_threshold_rel must lie in (0, 1].")
    if config.hough_min_center_distance_px < 1:
        raise ConfigError("sem_to_synthetic hough_min_center_distance_px must be at least 1.")
    if config.hough_overlap_buffer_px < 0.0:
        raise ConfigError("sem_to_synthetic hough_overlap_buffer_px must be non-negative.")
    if config.separation_min_distance_px < 1:
        raise ConfigError("sem_to_synthetic separation_min_distance_px must be at least 1.")
    if config.separation_peak_threshold_px < 0.0:
        raise ConfigError("sem_to_synthetic separation_peak_threshold_px must be non-negative.")
    if config.min_artifact_area_px < 0:
        raise ConfigError("sem_to_synthetic min_artifact_area_px must be non-negative.")
    if config.min_region_area_px < 1:
        raise ConfigError("sem_to_synthetic min_region_area_px must be at least 1.")
    if config.ellipse_axis_ratio_threshold < 1.0:
        raise ConfigError("sem_to_synthetic ellipse_axis_ratio_threshold must be at least 1.")
    if config.ellipse_sample_points < 8:
        raise ConfigError("sem_to_synthetic ellipse_sample_points must be at least 8.")
    if not 0.0 <= config.max_candidate_overlap_fraction <= 1.0:
        raise ConfigError(
            "sem_to_synthetic max_candidate_overlap_fraction must lie between 0 and 1."
        )


def _validate_solver(config: RVEConfig) -> None:
    solver = config.solver
    if config.dimension == 3 and solver.enabled and solver.kinematics != "solid":
        raise ConfigError("3D Ferrite solves require kinematics='solid'.")
    if solver.matrix_youngs_modulus <= 0.0 or solver.fibre_youngs_modulus <= 0.0:
        raise ConfigError("Ferrite solver Young's moduli must be positive.")
    for poisson in (solver.matrix_poisson_ratio, solver.fibre_poisson_ratio):
        if not -1.0 < poisson < 0.5:
            raise ConfigError("Ferrite solver Poisson ratios must lie between -1 and 0.5.")
    if solver.boundary_condition == "periodic":
        periodic_enabled = (
            bool(config.synthetic and config.synthetic.periodic_compatible)
            or bool(config.image and config.image.periodic_compatible)
            or bool(config.sem_to_synthetic and config.sem_to_synthetic.periodic_compatible)
        )
        if not periodic_enabled:
            raise ConfigError(
                "Periodic Ferrite solves require a config marked as periodic-compatible."
            )
    _validate_material_model(
        "matrix",
        solver.matrix_material_model,
        config.dimension,
        solver.matrix_e1,
        solver.matrix_e2,
        solver.matrix_e3,
        solver.matrix_g12,
        solver.matrix_g13,
        solver.matrix_g23,
        solver.matrix_nu12,
        solver.matrix_nu13,
        solver.matrix_nu23,
    )
    _validate_material_model(
        "fibre",
        solver.fibre_material_model,
        config.dimension,
        solver.fibre_e1,
        solver.fibre_e2,
        solver.fibre_e3,
        solver.fibre_g12,
        solver.fibre_g13,
        solver.fibre_g23,
        solver.fibre_nu12,
        solver.fibre_nu13,
        solver.fibre_nu23,
    )
    if solver.kinematics == "plane_strain" and (
        solver.matrix_material_model == "orthotropic"
        or solver.fibre_material_model == "orthotropic"
    ):
        raise ConfigError(
            "Orthotropic phase materials are currently supported only for plane_stress solves."
        )
    if config.dimension == 3:
        _validate_3d_rotation_inputs(
            "matrix",
            solver.matrix_material_angle_deg,
            solver.matrix_material_angle_x_deg,
            solver.matrix_material_angle_y_deg,
            solver.matrix_material_angle_z_deg,
        )
        _validate_3d_rotation_inputs(
            "fibre",
            solver.fibre_material_angle_deg,
            solver.fibre_material_angle_x_deg,
            solver.fibre_material_angle_y_deg,
            solver.fibre_material_angle_z_deg,
        )


def _validate_material_model(
    label: str,
    material_model: Literal["isotropic", "orthotropic"],
    dimension: int,
    e1: float | None,
    e2: float | None,
    e3: float | None,
    g12: float | None,
    g13: float | None,
    g23: float | None,
    nu12: float | None,
    nu13: float | None,
    nu23: float | None,
) -> None:
    if material_model == "isotropic":
        return
    values = {"e1": e1, "e2": e2, "g12": g12, "nu12": nu12}
    if dimension == 3:
        values.update(
            {
                "e3": e3,
                "g13": g13,
                "g23": g23,
                "nu13": nu13,
                "nu23": nu23,
            }
        )
    missing = [name for name, value in values.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Orthotropic {label} material requires: {joined}.")
    assert e1 is not None and e2 is not None and g12 is not None and nu12 is not None
    positive_values = [e1, e2, g12]
    if dimension == 3:
        assert e3 is not None and g13 is not None and g23 is not None
        positive_values.extend([e3, g13, g23])
    if any(value <= 0.0 for value in positive_values):
        raise ConfigError(f"Orthotropic {label} stiffness entries must be positive.")
    poisson_values = [nu12]
    if dimension == 3:
        assert nu13 is not None and nu23 is not None
        poisson_values.extend([nu13, nu23])
    if any(not -1.0 < value < 1.0 for value in poisson_values):
        raise ConfigError(f"Orthotropic {label} nu12 must lie between -1 and 1.")


def _validate_3d_rotation_inputs(
    label: str,
    angle_deg: float | None,
    angle_x_deg: float | None,
    angle_y_deg: float | None,
    angle_z_deg: float | None,
) -> None:
    if angle_deg is not None and (
        angle_x_deg is not None or angle_y_deg is not None or angle_z_deg is not None
    ):
        raise ConfigError(
            f"3D {label} material rotation cannot mix material_angle_deg with x/y/z rotations."
        )
