from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rve2d.config import RVEConfig
from rve2d.exceptions import ConfigError
from rve2d.export.writers import convert_mesh_formats, write_metadata_bundle
from rve2d.ferrite_bridge import FerriteSolveResult, run_ferrite_homogenization
from rve2d.image_import.mask_to_geometry import import_mask_geometry
from rve2d.image_import.mask_to_geometry_3d import import_mask_geometry_3d
from rve2d.meshing.gmsh_builder import build_mesh_with_gmsh
from rve2d.models import GeometryModel
from rve2d.synthetic_generation.circular import generate_circular_fibre_rve
from rve2d.synthetic_generation.cylindrical import generate_cylindrical_fibre_rve
from rve2d.synthetic_generation.sem_image import generate_circular_fibre_rve_from_sem
from rve2d.validation.checks import QualityReport, validate_geometry

if TYPE_CHECKING:
    from rve2d.nonlinear.records import NonlinearResult


@dataclass(frozen=True)
class BuildResult:
    output_directory: Path
    mesh_files: list[Path]
    metadata_files: list[Path]
    quality_report: QualityReport
    geometry_metadata: dict[str, object]
    ferrite_result: FerriteSolveResult | None = None
    nonlinear_result: NonlinearResult | None = None


def build_rve(
    config: RVEConfig,
    output_dir: str | None = None,
    basename: str | None = None,
) -> BuildResult:
    geometry, disconnected_artifacts = _build_geometry(config)
    minimum_spacing_requirement = (
        config.synthetic.min_spacing if config.synthetic is not None else 0.0
    )
    quality_report = validate_geometry(
        geometry,
        minimum_spacing_requirement=minimum_spacing_requirement,
        disconnected_artifacts=disconnected_artifacts,
    )

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


def solve_with_ferrite(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    geometry_metadata: dict[str, object] | None = None,
) -> FerriteSolveResult:
    if not config.solver.enabled:
        raise ConfigError("Ferrite solve requested but solver.enabled is false in the config.")
    return run_ferrite_homogenization(
        mesh_path,
        output_dir,
        config.solver,
        geometry_metadata=geometry_metadata,
    )


def solve_nonlinear(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
) -> NonlinearResult:
    """Nonlinear RVE solve (J2 plasticity + cohesive interfaces).

    ``nonlinear.backend: python`` needs the ``nonlinear`` extra (PyTorch, diffcohesive);
    ``julia`` runs the Ferrite.jl solver in ``julia/NonlinearRVE`` (needs ``julia`` on PATH).
    """
    if config.nonlinear.backend == "julia":
        from rve2d.nonlinear.julia_bridge import run_julia_nonlinear

        return run_julia_nonlinear(config, mesh_path, output_dir)
    from rve2d.nonlinear.driver import run_nonlinear_homogenization

    return run_nonlinear_homogenization(config, mesh_path, output_dir)


def build_and_solve_rve(
    config: RVEConfig,
    output_dir: str | None = None,
    basename: str | None = None,
) -> BuildResult:
    if not config.solver.enabled and not config.nonlinear.enabled:
        raise ConfigError(
            "build-and-solve needs solver.enabled and/or nonlinear.enabled in the config."
        )
    build_result = build_rve(config, output_dir=output_dir, basename=basename)
    msh_files = [path for path in build_result.mesh_files if path.suffix == ".msh"]
    if not msh_files:
        raise ConfigError("Solving requires a generated .msh mesh file.")
    metadata_files = list(build_result.metadata_files)
    ferrite_result = None
    if config.solver.enabled:
        ferrite_result = solve_with_ferrite(
            config,
            msh_files[0],
            build_result.output_directory,
            geometry_metadata=build_result.geometry_metadata,
        )
        metadata_files.append(ferrite_result.summary_path)
    nonlinear_result = None
    if config.nonlinear.enabled:
        nonlinear_result = solve_nonlinear(config, msh_files[0], build_result.output_directory)
        metadata_files.append(nonlinear_result.summary_path)
    return BuildResult(
        output_directory=build_result.output_directory,
        mesh_files=build_result.mesh_files,
        metadata_files=metadata_files,
        quality_report=build_result.quality_report,
        geometry_metadata=build_result.geometry_metadata,
        ferrite_result=ferrite_result,
        nonlinear_result=nonlinear_result,
    )


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
