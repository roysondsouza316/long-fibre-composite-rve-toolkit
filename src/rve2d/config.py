from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast

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
    elements_per_circle: int = 0


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
class PhaseMaterialConfig:
    """Isotropic elastic phase with optional J2 plasticity (linear isotropic hardening)."""

    youngs_modulus: float
    poisson_ratio: float
    yield_stress: float | None = None
    hardening_modulus: float = 0.0


@dataclass(frozen=True)
class CohesiveInterfaceConfig:
    """Fibre/matrix cohesive interface (traction-separation law from diffcohesive)."""

    enabled: bool = True
    law: Literal[
        "bilinear_mixed_mode", "bilinear", "linear-parabolic", "exponential", "trapezoidal"
    ] = "bilinear_mixed_mode"
    penalty_stiffness: float | None = None
    normal_strength: float | None = None
    shear_strength: float | None = None
    mode_i_toughness: float | None = None
    mode_ii_toughness: float | None = None
    bk_exponent: float = 1.45
    mixed_mode_criterion: Literal["bk", "power"] = "bk"
    viscosity: float = 0.0
    shear_penalty_stiffness: float | None = None
    integration: Literal["nodal", "gauss"] = "nodal"


@dataclass(frozen=True)
class NonlinearLoadConfig:
    type: Literal["uniaxial_stress", "uniaxial_strain"] = "uniaxial_stress"
    component: Literal["xx", "yy", "zz", "yz", "xz", "xy"] = "xx"
    max_strain: float = 0.02
    steps: int = 40
    unload: bool = False


@dataclass(frozen=True)
class NonlinearSolveConfig:
    """Nonlinear RVE solve: J2 plasticity in both phases plus cohesive fibre/matrix interfaces."""

    enabled: bool = False
    backend: Literal["python", "julia"] = "python"
    kinematics: Literal["plane_strain", "generalized_plane_strain", "solid"] | None = None
    boundary_condition: Literal["periodic", "dirichlet"] = "periodic"
    matrix: PhaseMaterialConfig | None = None
    fibre: PhaseMaterialConfig | None = None
    interface: CohesiveInterfaceConfig | None = None
    load: NonlinearLoadConfig = field(default_factory=NonlinearLoadConfig)
    device: str = "cpu"
    linear_solver: Literal["auto", "scipy", "pardiso", "tensormesh"] = "auto"
    newton_max_iterations: int = 25
    newton_tolerance: float = 1.0e-8
    max_step_cuts: int = 10
    output_every: int = 0
    matrix_phase_id: int = 1
    fibre_phase_id: int = 2

    def resolved_kinematics(self, dimension: int) -> str:
        if self.kinematics is not None:
            return self.kinematics
        return "solid" if dimension == 3 else "generalized_plane_strain"


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
    nonlinear: NonlinearSolveConfig = field(default_factory=NonlinearSolveConfig)

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
            _section(SyntheticGenerationConfig, synthetic_payload, "synthetic")
            if synthetic_payload is not None
            else None
        ),
        image=(
            _section(ImageImportConfig, image_payload, "image")
            if image_payload is not None
            else None
        ),
        sem_to_synthetic=(
            _section(SemToSyntheticConfig, sem_to_synthetic_payload, "sem_to_synthetic")
            if sem_to_synthetic_payload is not None
            else None
        ),
        mesh=_section(MeshConfig, payload.get("mesh", {}), "mesh"),
        export=_section(ExportConfig, payload.get("export", {}), "export"),
        solver=_section(FerriteSolveConfig, payload.get("solver", {}), "solver"),
        nonlinear=_nonlinear_from_dict(payload.get("nonlinear")),
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
    if config.mesh.elements_per_circle < 0:
        raise ConfigError("mesh.elements_per_circle must be zero (off) or positive.")
    if not config.export.formats:
        raise ConfigError("At least one export format must be configured.")
    _validate_solver(config)
    _validate_nonlinear(config)


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


def _section(cls: Any, payload: Any, name: str) -> Any:
    """Build a dataclass section, reporting unknown or missing keys as ConfigError.

    Numeric strings in float fields are converted: YAML 1.1 (PyYAML) reads an exponent
    without a sign, e.g. ``1.0e8``, as text rather than as a number.
    """
    if not isinstance(payload, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping.")
    types = {f.name: str(f.type) for f in fields(cls)}
    unknown = sorted(set(payload) - set(types))
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {', '.join(unknown)}.")
    data = dict(payload)
    for key, value in payload.items():
        if isinstance(value, str) and "float" in types[key] and "list" not in types[key]:
            try:
                data[key] = float(value)
            except ValueError as exc:
                raise ConfigError(f"'{name}.{key}' must be a number, got {value!r}.") from exc
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{name}' section: {exc}") from exc


def _nonlinear_from_dict(payload: Any) -> NonlinearSolveConfig:
    if payload is None:
        return NonlinearSolveConfig()
    if not isinstance(payload, dict):
        raise ConfigError("Config section 'nonlinear' must be a mapping.")
    data = dict(payload)
    for key, cls in (("matrix", PhaseMaterialConfig), ("fibre", PhaseMaterialConfig)):
        if data.get(key) is not None:
            data[key] = _section(cls, data[key], f"nonlinear.{key}")
    if data.get("interface") is not None:
        data["interface"] = _section(
            CohesiveInterfaceConfig, data["interface"], "nonlinear.interface"
        )
    if data.get("load") is not None:
        data["load"] = _section(NonlinearLoadConfig, data["load"], "nonlinear.load")
    return cast(NonlinearSolveConfig, _section(NonlinearSolveConfig, data, "nonlinear"))


_MIN_TOUGHNESS_FACTOR = {
    "bilinear_mixed_mode": 0.5,
    "bilinear": 0.5,
    "linear-parabolic": 0.125,
    "exponential": math.e**2 - 2.0 * math.e,
    "trapezoidal": 1.0 - 1e-12,
}


def _default(value: float | None, default: float) -> float:
    return default if value is None else value


NONLINEAR_ACTIVE_COMPONENTS = {
    "plane_strain": ("xx", "yy", "xy"),
    "generalized_plane_strain": ("xx", "yy", "zz", "xy"),
    "solid": ("xx", "yy", "zz", "yz", "xz", "xy"),
}


def _validate_nonlinear(config: RVEConfig) -> None:
    nl = config.nonlinear
    if not nl.enabled:
        return
    kinematics = nl.resolved_kinematics(config.dimension)
    if config.dimension == 3 and kinematics != "solid":
        raise ConfigError("3D nonlinear solves require kinematics: solid.")
    if config.dimension == 2 and kinematics == "solid":
        raise ConfigError("2D nonlinear solves use plane_strain or generalized_plane_strain.")
    if nl.boundary_condition == "periodic":
        periodic = (
            (
                config.mode == "synthetic"
                and bool(config.synthetic and config.synthetic.periodic_compatible)
            )
            or (config.mode == "image" and bool(config.image and config.image.periodic_compatible))
            or (
                config.mode == "sem_to_synthetic"
                and bool(config.sem_to_synthetic and config.sem_to_synthetic.periodic_compatible)
            )
        )
        if not periodic:
            raise ConfigError("Periodic nonlinear solves require a periodic_compatible geometry.")
    if config.mesh.mesh_order != 1 or config.mesh.recombine:
        raise ConfigError(
            "The nonlinear solver needs linear triangles/tetrahedra (mesh_order: 1, no recombine)."
        )
    for label, phase in (("matrix", nl.matrix), ("fibre", nl.fibre)):
        if phase is None:
            raise ConfigError(
                f"nonlinear.{label} material (youngs_modulus, poisson_ratio, ...) is required."
            )
        if phase.youngs_modulus <= 0.0:
            raise ConfigError(f"nonlinear.{label}.youngs_modulus must be positive.")
        if not -1.0 < phase.poisson_ratio < 0.5:
            raise ConfigError(f"nonlinear.{label}.poisson_ratio must lie in (-1, 0.5).")
        if phase.yield_stress is not None and phase.yield_stress <= 0.0:
            raise ConfigError(
                f"nonlinear.{label}.yield_stress must be positive (or omitted for elastic)."
            )
        if phase.hardening_modulus < 0.0:
            raise ConfigError(f"nonlinear.{label}.hardening_modulus must be non-negative.")
    interface = nl.interface
    if interface is not None and interface.enabled:
        required = ("penalty_stiffness", "normal_strength", "mode_i_toughness")
        missing = [key for key in required if getattr(interface, key) is None]
        if missing:
            raise ConfigError("nonlinear.interface requires: " + ", ".join(missing) + ".")
        assert interface.penalty_stiffness is not None and interface.normal_strength is not None
        assert interface.mode_i_toughness is not None
        stiffness = interface.penalty_stiffness
        pairs = [
            (interface.normal_strength, interface.mode_i_toughness, stiffness, "normal / mode I")
        ]
        if interface.law == "bilinear_mixed_mode":
            pairs.append(
                (
                    _default(interface.shear_strength, interface.normal_strength),
                    _default(interface.mode_ii_toughness, interface.mode_i_toughness),
                    _default(interface.shear_penalty_stiffness, stiffness),
                    "shear / mode II",
                )
            )
        elif interface.viscosity > 0.0:
            raise ConfigError(
                "nonlinear.interface.viscosity is available for law: bilinear_mixed_mode only."
            )
        # Smallest toughness (in units of strength^2 / K) for which the envelope exists.
        factor = _MIN_TOUGHNESS_FACTOR[interface.law]
        for strength, toughness, k, label in pairs:
            if min(strength, toughness, k) <= 0.0:
                raise ConfigError(
                    f"nonlinear.interface {label}: strength, toughness and stiffness must be "
                    "positive."
                )
            if toughness <= factor * strength**2 / k:
                raise ConfigError(
                    f"nonlinear.interface {label}: toughness {toughness:g} must exceed "
                    f"{factor:.4g} strength^2 / K = {factor * strength**2 / k:g} for the "
                    f"{interface.law} law (otherwise the envelope ends before damage onset)."
                )
        if interface.integration not in ("nodal", "gauss"):
            raise ConfigError("nonlinear.interface.integration must be 'nodal' or 'gauss'.")
        if interface.bk_exponent <= 0.0 or interface.viscosity < 0.0:
            raise ConfigError(
                "nonlinear.interface: bk_exponent must be positive and viscosity non-negative."
            )
    load = nl.load
    if load.component not in NONLINEAR_ACTIVE_COMPONENTS[kinematics]:
        raise ConfigError(
            f"nonlinear.load.component {load.component!r} is not active for {kinematics} "
            f"(choose from {', '.join(NONLINEAR_ACTIVE_COMPONENTS[kinematics])})."
        )
    if load.max_strain == 0.0 or load.steps < 1:
        raise ConfigError("nonlinear.load needs a non-zero max_strain and at least one step.")
    if nl.newton_max_iterations < 1 or nl.newton_tolerance <= 0.0 or nl.max_step_cuts < 0:
        raise ConfigError("nonlinear Newton settings must be positive.")
    if not (nl.device == "cpu" or nl.device.startswith("cuda")):
        raise ConfigError("nonlinear.device must be 'cpu' or 'cuda[:index]'.")
    if nl.backend not in ("python", "julia"):
        raise ConfigError("nonlinear.backend must be 'python' or 'julia'.")
    if nl.backend == "julia" and nl.device != "cpu":
        raise ConfigError("The Julia backend runs on the CPU (nonlinear.device: cpu).")
