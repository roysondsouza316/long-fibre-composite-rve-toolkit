from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from rve2d.exceptions import ConfigError

ENGINE_NAMES = ("tensormesh", "julia")
# Older configs called the TensorMesh engine "python".
ENGINE_ALIASES = {"python": "tensormesh"}
LINEAR_SOLVERS = ("auto", "scipy", "tensormesh")


def canonical_engine(name: str) -> str:
    """The engine name for ``name`` (aliases resolved); ConfigError if unknown."""
    engine = ENGINE_ALIASES.get(name, name)
    if engine not in ENGINE_NAMES:
        raise ConfigError(f"Unknown engine {name!r}; use 'tensormesh' or 'julia'.")
    return engine


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    """Random circular (2D) or cylindrical (3D, fibres along z) fibre arrangement.

    Placement options:

    ``packing_algorithm``
        ``"random_sequential"`` (default) draws random centres one at a time and rejects
        overlapping ones (``max_attempts`` draws at most). It is the historical algorithm and
        reproduces earlier geometries for the same ``random_seed``, but it saturates around a
        fibre volume fraction of 0.45-0.5. ``"relaxation"`` places all fibres at random and
        then iteratively pushes apart every pair closer than ``2 * fibre_radius +
        min_spacing`` (periodic minimum-image distances when ``periodic_wrapping`` is on),
        followed by ``shake_sweeps`` Monte Carlo sweeps that randomise the contact network.
        It reaches volume fractions of 0.60-0.65 and above in about a second for a few
        hundred fibres. It requires ``min_spacing > 0`` because many fibre pairs end up
        exactly ``min_spacing`` apart.
    ``periodic_wrapping``
        Only with ``periodic_compatible: true``. Fibres may cross the domain boundary; each
        crossing fibre is stored together with its periodic images (same ``fibre_id``) and the
        mesher keeps the parts inside the domain. This removes the matrix-only band along the
        domain edges; ``edge_clearance`` is ignored.
    ``boundary_clearance``
        Minimum distance between a fibre surface and a domain face (or, with wrapping, a
        domain corner) that would otherwise produce sliver elements. With wrapping, a fibre
        either crosses a face by at least this much or stays at least this far away from it.
        Without wrapping it is a lower bound for the fibre-to-face gap, combined with
        ``edge_clearance`` by taking the larger value. ``None`` (default) selects
        ``max(min_spacing, 0.1 * fibre_radius)`` for wrapping or relaxation layouts and leaves
        the legacy random-sequential layout untouched. Choose about one mesh element size
        (``mesh.element_size_min``) or more.
    ``max_relaxation_iterations``
        Iteration cap of the relaxation algorithm before it reports failure.
    ``shake_sweeps``
        Monte Carlo sweeps after relaxation (0 disables them).
    """

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
    packing_algorithm: Literal["random_sequential", "relaxation"] = "random_sequential"
    periodic_wrapping: bool = False
    boundary_clearance: float | None = None
    max_relaxation_iterations: int = 20_000
    shake_sweeps: int = 20


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
class SolverConfig:
    """Linear homogenization (the ``solver`` section): effective stiffness of the RVE.

    ``engine`` selects the TensorMesh engine (Python: NumPy/SciPy for this solve, no extra
    dependencies) or the Julia engine (Ferrite.jl, needs Julia 1.11+); both give the same
    results. 2D kinematics:
    ``plane_stress`` (thin lamina), ``plane_strain`` (eps_zz = 0; gives plane-strain moduli)
    or ``generalized_plane_strain`` (cross-section of a unidirectional ply with fibres along
    z: all six strains, so the full 6x6 stiffness and true engineering constants); 3D:
    ``solid``.
    """

    enabled: bool = False
    engine: Literal["tensormesh", "julia"] = "tensormesh"
    boundary_condition: Literal["dirichlet", "periodic"] = "dirichlet"
    kinematics: Literal["plane_strain", "plane_stress", "generalized_plane_strain", "solid"] = (
        "plane_strain"
    )
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


FerriteSolveConfig = SolverConfig  # name used before the engines were split


@dataclass(frozen=True)
class PhaseMaterialConfig:
    """Isotropic elastic phase with optional plasticity and ductile damage.

    Plasticity (with ``yield_stress``): von Mises (J2) by default; with a larger
    ``compressive_yield_stress`` the pressure-dependent paraboloidal surface of polymer
    matrices (Tschoegl; Melro et al. 2013) with non-associative flow set by
    ``plastic_poisson_ratio`` (0.5: no plastic volume change). ``hardening_modulus`` is the
    hardening in tension (compression scales with it): linear, or with ``saturation_stress``
    the initial slope of Voce's saturating curve towards that stress. Ductile damage (with
    ``damage_onset_strain`` and ``fracture_energy``): beyond that equivalent plastic strain the
    stiffness degrades with the plastic displacement over one element (crack band), so that
    a crack dissipates about ``fracture_energy`` per unit area.
    """

    youngs_modulus: float
    poisson_ratio: float
    yield_stress: float | None = None
    hardening_modulus: float = 0.0
    compressive_yield_stress: float | None = None
    plastic_poisson_ratio: float = 0.5
    damage_onset_strain: float | None = None
    fracture_energy: float | None = None
    saturation_stress: float | None = None


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
    engine: Literal["tensormesh", "julia"] = "tensormesh"
    kinematics: Literal["plane_strain", "generalized_plane_strain", "solid"] | None = None
    boundary_condition: Literal["periodic", "dirichlet"] = "periodic"
    matrix: PhaseMaterialConfig | None = None
    fibre: PhaseMaterialConfig | None = None
    interface: CohesiveInterfaceConfig | None = None
    load: NonlinearLoadConfig = field(default_factory=NonlinearLoadConfig)
    device: str = "cpu"
    linear_solver: Literal["auto", "scipy", "tensormesh"] = "auto"
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
class LaminateTensileConfig:
    """Strain-controlled coupon test of each laminate (stress, strain, elongation, force)."""

    enabled: bool = True
    direction: Literal["x", "y", "xy"] = "x"
    max_strain: float = 0.02
    steps: int = 200
    gauge_length: float | None = None  # elongation = strain * gauge_length
    width: float | None = None  # force = stress * laminate thickness * width


@dataclass(frozen=True)
class LaminateConfig:
    """Laminates of plies made of the RVE microstructure (``rve2d laminate``).

    The ply stiffness comes from the linear homogenization (``solver`` section, kinematics
    ``generalized_plane_strain`` in 2D or ``solid`` in 3D); for the tensile test the ply's
    transverse and in-plane shear curves come from nonlinear RVE solves (``nonlinear``
    section; for a 2D RVE the shear curve is solved on one periodic layer of tetrahedra
    extruded from its mesh). Fibre failure is not part of the RVE model: the longitudinal
    strengths are inputs. Use the same units in every section (e.g. mm and MPa).
    """

    enabled: bool = False
    ply_thickness: float = 0.125
    stacking_sequences: list[Any] = field(default_factory=lambda: ["[0/90]s"])
    transversely_isotropic: bool = True
    longitudinal_tensile_strength: float | None = None
    longitudinal_compressive_strength: float | None = None
    ply_curves: bool = True
    transverse_compression_curve: bool = False
    transverse_max_strain: float = 0.03
    shear_max_strain: float = 0.06
    curve_steps: int = 60
    tensile_test: LaminateTensileConfig = field(default_factory=LaminateTensileConfig)


@dataclass(frozen=True)
class RVEConfig:
    mode: Literal["synthetic", "image", "sem_to_synthetic"]
    dimension: Literal[2, 3] = 2
    synthetic: SyntheticGenerationConfig | None = None
    image: ImageImportConfig | None = None
    sem_to_synthetic: SemToSyntheticConfig | None = None
    mesh: MeshConfig = field(default_factory=MeshConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    nonlinear: NonlinearSolveConfig = field(default_factory=NonlinearSolveConfig)
    laminate: LaminateConfig = field(default_factory=LaminateConfig)

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
        solver=_section(SolverConfig, _engine_alias(payload.get("solver", {})), "solver"),
        nonlinear=_nonlinear_from_dict(payload.get("nonlinear")),
        laminate=_laminate_from_dict(payload.get("laminate")),
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
    _validate_laminate(config)


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
    if config.packing_algorithm not in {"random_sequential", "relaxation"}:
        raise ConfigError(
            "synthetic packing_algorithm must be 'random_sequential' or 'relaxation'."
        )
    if not isinstance(config.periodic_wrapping, bool):
        raise ConfigError("synthetic periodic_wrapping must be true or false.")
    if config.max_relaxation_iterations <= 0:
        raise ConfigError("max_relaxation_iterations must be positive.")
    if config.shake_sweeps < 0:
        raise ConfigError("shake_sweeps must be non-negative.")
    if config.boundary_clearance is not None and config.boundary_clearance < 0.0:
        raise ConfigError("boundary_clearance must be non-negative.")
    if config.packing_algorithm == "relaxation" and config.min_spacing <= 0.0:
        raise ConfigError(
            "packing_algorithm 'relaxation' leaves many fibre pairs exactly min_spacing apart, "
            "so it requires min_spacing > 0 (about one mesh element, see "
            "mesh.element_size_min); touching fibres cannot be meshed."
        )
    if config.periodic_wrapping:
        if not config.periodic_compatible:
            raise ConfigError("periodic_wrapping requires periodic_compatible: true.")
        contact_distance = 2.0 * config.fibre_radius + config.min_spacing
        if min(config.domain_width, config.domain_height) < 2.0 * contact_distance:
            raise ConfigError(
                "periodic_wrapping requires domain_width and domain_height of at least "
                "2 * (2 * fibre_radius + min_spacing) so that periodic images are unambiguous."
            )
        if (
            config.boundary_clearance is not None
            and config.boundary_clearance >= config.fibre_radius
        ):
            raise ConfigError(
                "With periodic_wrapping, boundary_clearance must be smaller than fibre_radius."
            )


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


def geometry_is_periodic(config: RVEConfig) -> bool:
    """Whether the geometry section of the active mode is marked ``periodic_compatible``."""
    section = {
        "synthetic": config.synthetic,
        "image": config.image,
        "sem_to_synthetic": config.sem_to_synthetic,
    }[config.mode]
    return bool(section is not None and section.periodic_compatible)


def _validate_solver(config: RVEConfig) -> None:
    solver = config.solver
    if solver.engine not in ENGINE_NAMES:
        raise ConfigError("solver.engine must be 'tensormesh' or 'julia'.")
    if solver.matrix_youngs_modulus <= 0.0 or solver.fibre_youngs_modulus <= 0.0:
        raise ConfigError("Solver Young's moduli must be positive.")
    for poisson in (solver.matrix_poisson_ratio, solver.fibre_poisson_ratio):
        if not -1.0 < poisson < 0.5:
            raise ConfigError("Solver Poisson ratios must lie between -1 and 0.5.")
    if not solver.enabled:
        return
    if config.dimension == 3 and solver.kinematics != "solid":
        raise ConfigError("3D solves require kinematics='solid'.")
    if config.dimension == 2 and solver.kinematics == "solid":
        raise ConfigError(
            "2D solves use kinematics plane_stress, plane_strain or generalized_plane_strain."
        )
    if solver.boundary_condition == "periodic" and not geometry_is_periodic(config):
        raise ConfigError(
            f"Periodic solves require the {config.mode} section to set periodic_compatible: true."
        )
    if config.mesh.mesh_order != 1 or config.mesh.recombine:
        raise ConfigError(
            "The solvers need linear triangles/tetrahedra (mesh.mesh_order: 1, no recombine)."
        )
    planar = solver.kinematics == "plane_stress"
    for label in ("matrix", "fibre"):
        _validate_material_model(solver, label, planar)
        _validate_3d_rotation_inputs(
            label,
            getattr(solver, f"{label}_material_angle_deg"),
            getattr(solver, f"{label}_material_angle_x_deg"),
            getattr(solver, f"{label}_material_angle_y_deg"),
            getattr(solver, f"{label}_material_angle_z_deg"),
        )
        tilted = [
            axis
            for axis in ("x", "y")
            if getattr(solver, f"{label}_material_angle_{axis}_deg") not in (None, 0.0)
        ]
        if planar and tilted:
            raise ConfigError(
                f"plane_stress rotates the {label} material in plane only; "
                f"{label}_material_angle_{tilted[0]}_deg needs plane_strain, "
                "generalized_plane_strain or a 3D solve."
            )


def _validate_material_model(solver: SolverConfig, label: str, planar: bool) -> None:
    """Orthotropic phases need the constants of their kinematics and must be stable
    (positive-definite compliance); plane stress needs only the in-plane set."""
    if getattr(solver, f"{label}_material_model") == "isotropic":
        return
    names = ["e1", "e2", "g12", "nu12"]
    if not planar:
        names += ["e3", "g13", "g23", "nu13", "nu23"]
    values = {name: getattr(solver, f"{label}_{name}") for name in names}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        kind = "plane-stress" if planar else "plane-strain, generalized plane strain and 3D"
        raise ConfigError(
            f"Orthotropic {label} material needs {', '.join(missing)} for {kind} solves."
        )
    if any(values[name] <= 0.0 for name in names if not name.startswith("nu")):
        raise ConfigError(f"Orthotropic {label} moduli must be positive.")
    e1, e2, nu12 = values["e1"], values["e2"], values["nu12"]
    if planar:
        compliance = [[1.0 / e1, -nu12 / e1], [-nu12 / e1, 1.0 / e2]]
    else:
        e3, nu13, nu23 = values["e3"], values["nu13"], values["nu23"]
        compliance = [
            [1.0 / e1, -nu12 / e1, -nu13 / e1],
            [-nu12 / e1, 1.0 / e2, -nu23 / e2],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3],
        ]
    if not _positive_definite(compliance):
        raise ConfigError(
            f"Orthotropic {label} constants are not physically admissible (the compliance "
            "is not positive definite; check the Poisson ratios against the moduli, e.g. "
            "nu12^2 < e1/e2)."
        )


def _positive_definite(matrix: list[list[float]]) -> bool:
    """Leading principal minors of a small symmetric matrix are all positive (Sylvester)."""
    size = len(matrix)
    for order in range(1, size + 1):
        minor = [row[:order] for row in matrix[:order]]
        if _determinant(minor) <= 0.0:
            return False
    return True


def _determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return float(
        sum(
            (-1) ** col
            * matrix[0][col]
            * _determinant([row[:col] + row[col + 1 :] for row in matrix[1:]])
            for col in range(len(matrix))
        )
    )


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
            f"The {label} material rotation cannot mix material_angle_deg with x/y/z rotations."
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


def _engine_alias(payload: Any) -> Any:
    """A copy of a section payload with an engine alias ("python") replaced by its name."""
    if isinstance(payload, dict) and isinstance(payload.get("engine"), str):
        return {**payload, "engine": ENGINE_ALIASES.get(payload["engine"], payload["engine"])}
    return payload


def _laminate_from_dict(payload: Any) -> LaminateConfig:
    if payload is None:
        return LaminateConfig()
    if not isinstance(payload, dict):
        raise ConfigError("Config section 'laminate' must be a mapping.")
    data = dict(payload)
    if data.get("tensile_test") is not None:
        data["tensile_test"] = _section(
            LaminateTensileConfig, data["tensile_test"], "laminate.tensile_test"
        )
    if isinstance(data.get("stacking_sequences"), str):
        data["stacking_sequences"] = [data["stacking_sequences"]]
    return cast(LaminateConfig, _section(LaminateConfig, data, "laminate"))


def _validate_laminate(config: RVEConfig) -> None:
    lam = config.laminate
    if not lam.enabled:
        return
    from rve2d.laminate.stacking import parse_stacking_sequence  # (avoids an import cycle)

    if lam.ply_thickness <= 0.0:
        raise ConfigError("laminate.ply_thickness must be positive.")
    if not lam.stacking_sequences:
        raise ConfigError("laminate.stacking_sequences needs at least one sequence.")
    for sequence in lam.stacking_sequences:
        parse_stacking_sequence(sequence)
    for name in ("longitudinal_tensile_strength", "longitudinal_compressive_strength"):
        value = getattr(lam, name)
        if value is not None and value <= 0.0:
            raise ConfigError(f"laminate.{name} must be positive.")
    if min(lam.transverse_max_strain, lam.shear_max_strain) <= 0.0 or lam.curve_steps < 1:
        raise ConfigError("laminate curve strains must be positive and curve_steps at least 1.")
    test = lam.tensile_test
    if test.direction not in ("x", "y", "xy"):
        raise ConfigError("laminate.tensile_test.direction must be x, y or xy.")
    if test.max_strain == 0.0 or test.steps < 1:
        raise ConfigError("laminate.tensile_test needs a non-zero max_strain and steps >= 1.")
    for name in ("gauge_length", "width"):
        value = getattr(test, name)
        if value is not None and value <= 0.0:
            raise ConfigError(f"laminate.tensile_test.{name} must be positive.")
    if not config.solver.enabled:
        raise ConfigError(
            "The laminate pipeline needs the solver section (enabled: true) for the ply stiffness."
        )
    if config.solver.kinematics not in ("generalized_plane_strain", "solid"):
        raise ConfigError(
            "The laminate pipeline needs the full 6x6 ply stiffness: use solver.kinematics "
            "generalized_plane_strain (2D) or solid (3D)."
        )
    if test.enabled and lam.ply_curves and not config.nonlinear.enabled:
        raise ConfigError(
            "The laminate tensile test takes the ply curves from the nonlinear section: enable "
            "it, or set laminate.ply_curves: false for elastic plies (fibre failure only)."
        )


def _nonlinear_from_dict(payload: Any) -> NonlinearSolveConfig:
    if payload is None:
        return NonlinearSolveConfig()
    if not isinstance(payload, dict):
        raise ConfigError("Config section 'nonlinear' must be a mapping.")
    data = _engine_alias(payload)
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


def _validate_phase_plasticity(phase: PhaseMaterialConfig, label: str) -> None:
    extras = (
        phase.compressive_yield_stress is not None
        or phase.plastic_poisson_ratio != 0.5
        or phase.damage_onset_strain is not None
        or phase.fracture_energy is not None
        or phase.saturation_stress is not None
    )
    if phase.yield_stress is None:
        if extras:
            raise ConfigError(
                f"{label}: compressive_yield_stress, plastic_poisson_ratio, saturation_stress "
                "and damage need a yield_stress."
            )
        return
    if phase.saturation_stress is not None and (
        phase.saturation_stress <= phase.yield_stress or phase.hardening_modulus <= 0.0
    ):
        raise ConfigError(
            f"{label}.saturation_stress must exceed yield_stress, with a positive "
            "hardening_modulus (the initial slope of the Voce curve)."
        )
    compressive = phase.compressive_yield_stress
    if compressive is not None and compressive < phase.yield_stress:
        raise ConfigError(f"{label}.compressive_yield_stress must be at least yield_stress.")
    if not 0.0 <= phase.plastic_poisson_ratio <= 0.5:
        raise ConfigError(f"{label}.plastic_poisson_ratio must lie in [0, 0.5].")
    if compressive is not None and compressive > phase.yield_stress:
        if phase.plastic_poisson_ratio >= 0.5:
            raise ConfigError(
                f"{label}: a pressure-dependent yield surface (compressive_yield_stress > "
                "yield_stress) needs plastic_poisson_ratio < 0.5 (plastic volume change), "
                "e.g. 0.3."
            )
    onset, energy = phase.damage_onset_strain, phase.fracture_energy
    if (onset is None) != (energy is None):
        raise ConfigError(f"{label}: damage needs both damage_onset_strain and fracture_energy.")
    if onset is not None and onset < 0.0:
        raise ConfigError(f"{label}.damage_onset_strain must be non-negative.")
    if energy is not None and energy <= 0.0:
        raise ConfigError(f"{label}.fracture_energy must be positive.")


def _validate_nonlinear(config: RVEConfig) -> None:
    nl = config.nonlinear
    if nl.engine not in ENGINE_NAMES:
        raise ConfigError("nonlinear.engine must be 'tensormesh' or 'julia'.")
    if nl.linear_solver not in LINEAR_SOLVERS:
        removed = (
            " PARDISO (Intel MKL) is not supported: it only exists for x86-64 Linux and "
            "Windows."
            if str(nl.linear_solver) == "pardiso"
            else ""
        )
        raise ConfigError(
            f"nonlinear.linear_solver must be one of {', '.join(LINEAR_SOLVERS)}.{removed}"
        )
    if not nl.enabled:
        return
    kinematics = nl.resolved_kinematics(config.dimension)
    if config.dimension == 3 and kinematics != "solid":
        raise ConfigError("3D nonlinear solves require kinematics: solid.")
    if config.dimension == 2 and kinematics == "solid":
        raise ConfigError("2D nonlinear solves use plane_strain or generalized_plane_strain.")
    if nl.boundary_condition == "periodic" and not geometry_is_periodic(config):
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
        _validate_phase_plasticity(phase, f"nonlinear.{label}")
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
    if nl.engine == "julia" and nl.device != "cpu":
        raise ConfigError("The Julia engine runs on the CPU (nonlinear.device: cpu).")
