"""Build an RVE (geometry -> mesh -> metadata) and solve it with the chosen engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from rve2d.config import RVEConfig
from rve2d.engines import HomogenizationResult, NonlinearResult, homogenize
from rve2d.engines import solve_nonlinear as _solve_nonlinear
from rve2d.exceptions import ConfigError
from rve2d.export.writers import convert_mesh_formats, write_metadata_bundle
from rve2d.image_import.mask_to_geometry import import_mask_geometry
from rve2d.image_import.mask_to_geometry_3d import import_mask_geometry_3d
from rve2d.meshing.gmsh_builder import build_mesh_with_gmsh
from rve2d.models import GeometryModel
from rve2d.synthetic_generation.circular import generate_circular_fibre_rve
from rve2d.synthetic_generation.cylindrical import generate_cylindrical_fibre_rve
from rve2d.synthetic_generation.packing import mesh_compatibility_warnings
from rve2d.synthetic_generation.sem_image import generate_circular_fibre_rve_from_sem
from rve2d.validation.checks import QualityReport, validate_geometry


@dataclass(frozen=True)
class BuildResult:
    output_directory: Path
    mesh_files: list[Path]
    metadata_files: list[Path]
    quality_report: QualityReport
    geometry_metadata: dict[str, object]
    homogenization_result: HomogenizationResult | None = None
    nonlinear_result: NonlinearResult | None = None


def build_rve(
    config: RVEConfig,
    output_dir: str | None = None,
    basename: str | None = None,
) -> BuildResult:
    geometry, removed_artifacts = _build_geometry(config)
    synthetic = config.synthetic if config.mode == "synthetic" else None
    quality_report = validate_geometry(
        geometry,
        minimum_spacing_requirement=synthetic.min_spacing if synthetic is not None else 0.0,
        removed_artifacts=removed_artifacts,
        element_size=config.mesh.element_size_min,
    )
    if synthetic is not None:
        # Gaps the packing may leave that the mesh cannot resolve (sliver elements).
        notes = [*quality_report.notes, *mesh_compatibility_warnings(synthetic, config.mesh)]
        quality_report = replace(quality_report, notes=notes)

    export_dir = Path(output_dir or config.export.output_dir)
    mesh_basename = basename or config.export.basename
    msh_path = export_dir / f"{mesh_basename}.msh"
    mesh_result = build_mesh_with_gmsh(geometry, config.mesh, msh_path)
    mesh_files = convert_mesh_formats(mesh_result.mesh_path, config.export.formats, mesh_basename)
    metadata_files = write_metadata_bundle(
        export_dir,
        geometry,
        quality_report,
        phase_tags=mesh_result.phase_tags,
        boundary_tags=mesh_result.boundary_tags,
        write_quality_json=config.export.write_quality_json,
        write_phase_json=config.export.write_phase_json,
        write_periodic_json=config.export.write_periodic_json,
    )
    return BuildResult(
        output_directory=export_dir,
        mesh_files=mesh_files,
        metadata_files=metadata_files,
        quality_report=quality_report,
        geometry_metadata=dict(geometry.metadata),
    )


def solve_homogenization(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    geometry_metadata: dict[str, object] | None = None,
    engine: str | None = None,
) -> HomogenizationResult:
    """Linear homogenization of an existing mesh (``solver`` section).

    Without ``geometry_metadata``, the ``geometry_summary.json`` written by ``build`` next to
    the mesh is used, so material rotations that come from the geometry are kept.
    """
    if not config.solver.enabled:
        raise ConfigError("Homogenization requested but solver.enabled is false in the config.")
    mesh_path = _existing_mesh(mesh_path)
    if geometry_metadata is None:
        geometry_metadata = load_geometry_metadata(mesh_path)
    return homogenize(config, mesh_path, output_dir, geometry_metadata, engine)


def solve_with_ferrite(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    geometry_metadata: dict[str, object] | None = None,
) -> HomogenizationResult:
    """Linear homogenization with the Julia (Ferrite.jl) engine; see ``solve_homogenization``."""
    return solve_homogenization(config, mesh_path, output_dir, geometry_metadata, engine="julia")


def solve_nonlinear(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    engine: str | None = None,
) -> NonlinearResult:
    """Nonlinear solve (``nonlinear`` section): J2 plasticity + cohesive interfaces."""
    if not config.nonlinear.enabled:
        raise ConfigError("Nonlinear solve requested but nonlinear.enabled is false in the config.")
    return _solve_nonlinear(config, _existing_mesh(mesh_path), output_dir, engine)


def _existing_mesh(mesh_path: str | Path) -> Path:
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    return path


def build_and_solve_rve(
    config: RVEConfig,
    output_dir: str | None = None,
    basename: str | None = None,
    engine: str | None = None,
) -> BuildResult:
    """Build the RVE and run every enabled solve (``engine`` overrides the configured ones)."""
    if not config.solver.enabled and not config.nonlinear.enabled:
        raise ConfigError(
            "build-and-solve needs solver.enabled and/or nonlinear.enabled in the config."
        )
    build_result = build_rve(config, output_dir=output_dir, basename=basename)
    msh_files = [path for path in build_result.mesh_files if path.suffix == ".msh"]
    if not msh_files:
        raise ConfigError("Solving requires a generated .msh mesh file.")
    metadata_files = list(build_result.metadata_files)
    homogenization_result = None
    if config.solver.enabled:
        homogenization_result = solve_homogenization(
            config,
            msh_files[0],
            build_result.output_directory,
            geometry_metadata=build_result.geometry_metadata,
            engine=engine,
        )
        metadata_files.append(homogenization_result.summary_path)
    nonlinear_result = None
    if config.nonlinear.enabled:
        nonlinear_result = solve_nonlinear(
            config, msh_files[0], build_result.output_directory, engine=engine
        )
        metadata_files.append(nonlinear_result.summary_path)
    return BuildResult(
        output_directory=build_result.output_directory,
        mesh_files=build_result.mesh_files,
        metadata_files=metadata_files,
        quality_report=build_result.quality_report,
        geometry_metadata=build_result.geometry_metadata,
        homogenization_result=homogenization_result,
        nonlinear_result=nonlinear_result,
    )


def load_geometry_metadata(mesh_path: str | Path) -> dict[str, object] | None:
    """Geometry metadata from the ``geometry_summary.json`` next to a mesh, if present."""
    summary = Path(mesh_path).with_name("geometry_summary.json")
    if not summary.exists():
        return None
    try:
        metadata = json.loads(summary.read_text(encoding="utf-8")).get("metadata")
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _build_geometry(config: RVEConfig) -> tuple[GeometryModel, int]:
    if config.mode == "synthetic":
        if config.synthetic is None:
            raise ConfigError("Synthetic mode requires a synthetic configuration.")
        if config.dimension == 3:
            synthetic_3d_result = generate_cylindrical_fibre_rve(config.synthetic)
            return synthetic_3d_result.geometry, 0
        synthetic_2d_result = generate_circular_fibre_rve(config.synthetic)
        return synthetic_2d_result.geometry, 0

    if config.mode == "sem_to_synthetic":
        if config.sem_to_synthetic is None:
            raise ConfigError("sem_to_synthetic mode requires a sem_to_synthetic section.")
        sem_result = generate_circular_fibre_rve_from_sem(config.sem_to_synthetic)
        return sem_result.geometry, 0

    if config.image is None:
        raise ConfigError("Image mode requires an image configuration.")
    if config.dimension == 3:
        image_3d_result = import_mask_geometry_3d(config.image)
        return image_3d_result.geometry, image_3d_result.removed_artifacts
    image_2d_result = import_mask_geometry(config.image)
    return image_2d_result.geometry, image_2d_result.removed_artifacts
