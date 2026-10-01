"""Linear homogenization: engine-independent inputs and outputs.

Both engines solve the same problem (see ``rve2d.engines.python.homogenization`` and the
Julia ``FerriteRVE`` package) and return an :class:`EngineSolution`; this module prepares the
phase stiffness matrices from the ``solver`` config section, derives engineering constants
and writes the summary JSON and CSV files, so the outputs do not depend on the engine.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rve2d.config import RVEConfig
from rve2d.engines.common.materials import (
    ACTIVE_VOIGT,
    VOIGT_NAMES,
    FloatArray,
    constitutive_matrix,
    phase_elasticity,
)
from rve2d.exceptions import SolverError

STRAIN_NAMES = {"xx": "exx", "yy": "eyy", "zz": "ezz", "yz": "gyz", "xz": "gxz", "xy": "gxy"}
STRESS_NAMES = {"xx": "sxx", "yy": "syy", "zz": "szz", "yz": "tyz", "xz": "txz", "xy": "txy"}
SUMMARY_NAME = "homogenization_summary.json"


@dataclass(frozen=True)
class HomogenizationProblem:
    """What an engine needs to solve: mesh, kinematics, boundary condition, phase stiffness."""

    mesh_path: Path
    output_dir: Path
    dimension: int
    kinematics: str
    boundary_condition: str
    matrix_stiffness: FloatArray
    fibre_stiffness: FloatArray
    matrix_phase_id: int
    fibre_phase_id: int
    write_vtk: bool

    @property
    def voigt(self) -> list[str]:
        return [VOIGT_NAMES[k] for k in ACTIVE_VOIGT[self.kinematics]]


@dataclass(frozen=True)
class EngineSolution:
    """Raw result of an engine: stiffness columns are the per-case average stresses."""

    stiffness: FloatArray
    tractions: list[dict[str, list[float]]]  # per load case: face -> facet-averaged traction
    fibre_volume_fraction: float
    unknowns: int
    vtk_path: Path | None
    log_path: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HomogenizationResult:
    engine: str
    kinematics: str
    summary_path: Path
    log_path: Path | None
    vtk_path: Path | None
    homogenized_stiffness: list[list[float]]
    engineering_constants: dict[str, float]
    response_files: list[Path]


Solver = Callable[[HomogenizationProblem], EngineSolution]


def run_homogenization(
    config: RVEConfig,
    mesh_path: str | Path,
    output_dir: str | Path,
    engine: str,
    solve: Solver,
    geometry_metadata: dict[str, object] | None = None,
) -> HomogenizationResult:
    """Solve with ``solve`` (an engine) and write the engine-independent result files."""
    started = time.time()
    solver = config.solver
    dimension = config.dimension
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rotations = {
        phase: resolve_material_rotation(config, phase, geometry_metadata)
        for phase in ("matrix", "fibre")
    }
    problem = HomogenizationProblem(
        mesh_path=Path(mesh_path).resolve(),
        output_dir=out_dir,
        dimension=dimension,
        kinematics=solver.kinematics,
        boundary_condition=solver.boundary_condition,
        matrix_stiffness=constitutive_matrix(
            phase_elasticity(solver, "matrix"), solver.kinematics, rotations["matrix"]
        ),
        fibre_stiffness=constitutive_matrix(
            phase_elasticity(solver, "fibre"), solver.kinematics, rotations["fibre"]
        ),
        matrix_phase_id=solver.matrix_phase_id,
        fibre_phase_id=solver.fibre_phase_id,
        write_vtk=solver.write_vtk,
    )
    solution = solve(problem)
    stiffness = np.asarray(solution.stiffness, dtype=np.float64)
    if not np.all(np.isfinite(stiffness)):
        raise SolverError(f"The {engine} engine returned a non-finite homogenized stiffness.")
    constants = engineering_constants(stiffness, solver.kinematics)
    average_stresses = [stiffness[:, j].tolist() for j in range(stiffness.shape[1])]
    summary: dict[str, Any] = {
        "engine": engine,
        "dimension": dimension,
        "kinematics": solver.kinematics,
        "boundary_condition": solver.boundary_condition,
        "voigt_components": problem.voigt,
        "homogenized_stiffness_voigt": stiffness.tolist(),
        "engineering_constants": constants,
        "average_stresses": average_stresses,
        "boundary_tractions": solution.tractions,
        "fibre_volume_fraction": solution.fibre_volume_fraction,
        "matrix_volume_fraction": 1.0 - solution.fibre_volume_fraction,
        "matrix_cellset": solver.matrix_cellset,
        "fibre_cellset": solver.fibre_cellset,
        "material_angles_deg": {phase: rotation[2] for phase, rotation in rotations.items()},
        "material_rotations_deg": {
            phase: dict(zip(("x", "y", "z"), rotation, strict=True))
            for phase, rotation in rotations.items()
        },
        "unknowns": solution.unknowns,
        "vtk_path": None if solution.vtk_path is None else str(solution.vtk_path),
        "log_path": None if solution.log_path is None else str(solution.log_path),
        **solution.details,
    }
    response_files = write_response_files(out_dir, problem, stiffness, constants, solution)
    summary["response_files"] = [str(path) for path in response_files]
    summary["runtime_seconds"] = round(time.time() - started, 3)
    summary_path = out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return HomogenizationResult(
        engine=engine,
        kinematics=solver.kinematics,
        summary_path=summary_path,
        log_path=solution.log_path,
        vtk_path=solution.vtk_path,
        homogenized_stiffness=stiffness.tolist(),
        engineering_constants=constants,
        response_files=response_files,
    )


def engineering_constants(stiffness: FloatArray, kinematics: str) -> dict[str, float]:
    """Engineering constants from the effective stiffness.

    * 6x6 (``solid``, ``generalized_plane_strain``): the full orthotropic-style set from the
      compliance, ``ex = 1/S11``, ``nuxy = -S12/S11``, ``gyz = 1/S44``, ...
    * ``plane_stress``: in-plane constants of a thin lamina.
    * ``plane_strain``: the inverse of the in-plane stiffness under ``eps_zz = 0`` gives
      plane-strain moduli, not Young's moduli (for an isotropic material
      ``ex_plane_strain = E / (1 - nu^2)``); they are reported with a ``_plane_strain``
      suffix. Use ``generalized_plane_strain`` for the true transverse constants.
    """
    compliance = np.linalg.inv(stiffness)
    if stiffness.shape == (6, 6):
        ex, ey, ez = (1.0 / compliance[i, i] for i in range(3))
        return {
            "ex": float(ex),
            "ey": float(ey),
            "ez": float(ez),
            "gyz": float(1.0 / compliance[3, 3]),
            "gxz": float(1.0 / compliance[4, 4]),
            "gxy": float(1.0 / compliance[5, 5]),
            "nuxy": float(-compliance[0, 1] * ex),
            "nuyx": float(-compliance[1, 0] * ey),
            "nuxz": float(-compliance[0, 2] * ex),
            "nuzx": float(-compliance[2, 0] * ez),
            "nuyz": float(-compliance[1, 2] * ey),
            "nuzy": float(-compliance[2, 1] * ez),
        }
    ex, ey = 1.0 / compliance[0, 0], 1.0 / compliance[1, 1]
    constants = {
        "ex": float(ex),
        "ey": float(ey),
        "gxy": float(1.0 / compliance[2, 2]),
        "nuxy": float(-compliance[0, 1] * ex),
        "nuyx": float(-compliance[1, 0] * ey),
    }
    if kinematics == "plane_strain":
        return {f"{name}_plane_strain": value for name, value in constants.items()}
    return constants


def write_response_files(
    out_dir: Path,
    problem: HomogenizationProblem,
    stiffness: FloatArray,
    constants: dict[str, float],
    solution: EngineSolution,
) -> list[Path]:
    names = problem.voigt
    stiffness_path = out_dir / "homogenized_stiffness.csv"
    np.savetxt(stiffness_path, stiffness, delimiter=",")

    engineering_path = out_dir / "engineering_constants.csv"
    lines = ["name,value"] + [f"{key},{value}" for key, value in constants.items()]
    engineering_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stress_strain_path = out_dir / "stress_strain_response.csv"
    header = ["case"] + [STRAIN_NAMES[n] for n in names] + [STRESS_NAMES[n] for n in names]
    rows = [",".join(header)]
    for case, column in enumerate(stiffness.T, start=1):
        strain = np.eye(len(names))[case - 1]
        rows.append(f"{case}," + ",".join(str(v) for v in [*strain, *column]))
    stress_strain_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    traction_path = out_dir / "traction_response.csv"
    components = len(next(iter(solution.tractions[0].values()))) if solution.tractions else 2
    rows = ["case,boundary," + ",".join(["tx", "ty", "tz"][:components]) + ",source"]
    for case, faces in enumerate(solution.tractions, start=1):
        for face, traction in faces.items():
            rows.append(f"{case},{face}," + ",".join(str(v) for v in traction) + ",facet_average")
    traction_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [stiffness_path, engineering_path, stress_strain_path, traction_path]


def resolve_material_rotation(
    config: RVEConfig,
    phase: str,
    geometry_metadata: dict[str, object] | None,
) -> tuple[float, float, float]:
    """Material rotation of ``phase`` in degrees about (x, y, z).

    Precedence: explicit config angles; then ``phase_orientation_rotations_deg`` from the
    geometry metadata; then (fibres only) the geometry's ``orientation_deg`` as a rotation
    about z. Plane stress only uses the rotation about z (config validation rejects x/y
    angles there).
    """
    solver = config.solver
    angle = getattr(solver, f"{phase}_material_angle_deg")
    angle_x = getattr(solver, f"{phase}_material_angle_x_deg")
    angle_y = getattr(solver, f"{phase}_material_angle_y_deg")
    angle_z = getattr(solver, f"{phase}_material_angle_z_deg")
    fallback = "orientation_deg" if phase == "fibre" else None
    if angle is not None:
        return (0.0, 0.0, float(angle))
    planar = solver.kinematics == "plane_stress"
    if not (angle_x is None and angle_y is None and angle_z is None):
        rotation = (
            0.0 if planar else float(angle_x or 0.0),
            0.0 if planar else float(angle_y or 0.0),
            float(angle_z or 0.0),
        )
        if rotation[2] == 0.0 and fallback is not None:
            rotation = (rotation[0], rotation[1], _metadata_angle(geometry_metadata, fallback))
        return rotation
    from_metadata = _metadata_rotation(geometry_metadata, phase)
    if from_metadata is not None:
        return (0.0, 0.0, from_metadata[2]) if planar else from_metadata
    if fallback is not None:
        return (0.0, 0.0, _metadata_angle(geometry_metadata, fallback))
    return (0.0, 0.0, 0.0)


def _metadata_rotation(
    geometry_metadata: dict[str, object] | None, phase: str
) -> tuple[float, float, float] | None:
    if geometry_metadata is None:
        return None
    rotations = geometry_metadata.get("phase_orientation_rotations_deg")
    if not isinstance(rotations, dict) or not isinstance(rotations.get(phase), dict):
        return None
    values = [rotations[phase].get(axis, 0.0) for axis in ("x", "y", "z")]
    if not all(isinstance(value, int | float) for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _metadata_angle(geometry_metadata: dict[str, object] | None, key: str) -> float:
    value = None if geometry_metadata is None else geometry_metadata.get(key)
    return float(value) if isinstance(value, int | float) else 0.0
