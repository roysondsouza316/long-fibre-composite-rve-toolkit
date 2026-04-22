from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import meshio
import numpy as np

from rve2d.config import FerriteSolveConfig
from rve2d.exceptions import SolverError


@dataclass(frozen=True)
class FerriteSolveResult:
    summary_path: Path
    stdout_path: Path
    vtk_path: Path | None
    homogenized_stiffness: list[list[float]]
    engineering_constants: dict[str, float]
    response_files: list[Path]


def run_ferrite_homogenization(
    mesh_path: str | Path,
    output_dir: str | Path,
    solver_config: FerriteSolveConfig,
    geometry_metadata: dict[str, object] | None = None,
) -> FerriteSolveResult:
    mesh_file = Path(mesh_path).resolve()
    result_dir = Path(output_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    exported = _export_ferrite_inputs(mesh_file, result_dir)
    dimension = cast(int, exported["dimension"])
    fibre_rotation_deg = _resolve_material_rotation_deg(
        dimension=dimension,
        phase_label="fibre",
        configured_angle_deg=solver_config.fibre_material_angle_deg,
        configured_angle_x_deg=solver_config.fibre_material_angle_x_deg,
        configured_angle_y_deg=solver_config.fibre_material_angle_y_deg,
        configured_angle_z_deg=solver_config.fibre_material_angle_z_deg,
        geometry_metadata=geometry_metadata,
        fallback_key="orientation_deg",
    )
    matrix_rotation_deg = _resolve_material_rotation_deg(
        dimension=dimension,
        phase_label="matrix",
        configured_angle_deg=solver_config.matrix_material_angle_deg,
        configured_angle_x_deg=solver_config.matrix_material_angle_x_deg,
        configured_angle_y_deg=solver_config.matrix_material_angle_y_deg,
        configured_angle_z_deg=solver_config.matrix_material_angle_z_deg,
        geometry_metadata=geometry_metadata,
        fallback_key=None,
    )
    matrix_constitutive = _constitutive_matrix(
        solver_config.matrix_material_model,
        solver_config.kinematics,
        youngs_modulus=solver_config.matrix_youngs_modulus,
        poisson_ratio=solver_config.matrix_poisson_ratio,
        e1=solver_config.matrix_e1,
        e2=solver_config.matrix_e2,
        e3=solver_config.matrix_e3,
        g12=solver_config.matrix_g12,
        g13=solver_config.matrix_g13,
        g23=solver_config.matrix_g23,
        nu12=solver_config.matrix_nu12,
        nu13=solver_config.matrix_nu13,
        nu23=solver_config.matrix_nu23,
        rotation_deg=matrix_rotation_deg,
    )
    fibre_constitutive = _constitutive_matrix(
        solver_config.fibre_material_model,
        solver_config.kinematics,
        youngs_modulus=solver_config.fibre_youngs_modulus,
        poisson_ratio=solver_config.fibre_poisson_ratio,
        e1=solver_config.fibre_e1,
        e2=solver_config.fibre_e2,
        e3=solver_config.fibre_e3,
        g12=solver_config.fibre_g12,
        g13=solver_config.fibre_g13,
        g23=solver_config.fibre_g23,
        nu12=solver_config.fibre_nu12,
        nu13=solver_config.fibre_nu13,
        nu23=solver_config.fibre_nu23,
        rotation_deg=fibre_rotation_deg,
    )
    matrix_matrix_path = result_dir / "ferrite_matrix_constitutive.csv"
    fibre_matrix_path = result_dir / "ferrite_fibre_constitutive.csv"
    np.savetxt(matrix_matrix_path, matrix_constitutive, delimiter=",")
    np.savetxt(fibre_matrix_path, fibre_constitutive, delimiter=",")

    solver_root = Path(__file__).resolve().parents[2] / "julia"
    solver_script = (
        solver_root / "ferrite_homogenization_3d.jl"
        if dimension == 3
        else solver_root / "ferrite_homogenization.jl"
    )
    summary_path = result_dir / "ferrite_homogenization_summary.json"
    stdout_path = result_dir / "ferrite_homogenization_stdout.txt"

    command = [
        "julia",
        str(solver_script),
        str(exported["nodes"]),
        str(exported["cells"]),
        str(exported["cell_tags"]),
        str(result_dir),
        solver_config.boundary_condition,
        solver_config.kinematics,
        str(matrix_matrix_path),
        str(fibre_matrix_path),
        solver_config.matrix_cellset,
        solver_config.fibre_cellset,
        str(solver_config.matrix_phase_id),
        str(solver_config.fibre_phase_id),
        str(exported["xmin"]),
        str(exported["xmax"]),
        str(exported["ymin"]),
        str(exported["ymax"]),
    ]
    if dimension == 3:
        command.extend([str(exported["zmin"]), str(exported["zmax"])])
    command.append("true" if solver_config.write_vtk else "false")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SolverError(
            "Ferrite.jl solve failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    summary_payload = _extract_summary_payload(completed.stdout)
    summary_payload["engineering_constants"] = _engineering_constants_from_stiffness(
        cast(list[list[float]], summary_payload["homogenized_stiffness_voigt"])
    )
    summary_payload["dimension"] = dimension
    summary_payload["material_angles_deg"] = {
        "matrix": matrix_rotation_deg[2],
        "fibre": fibre_rotation_deg[2],
    }
    summary_payload["material_rotations_deg"] = {
        "matrix": {
            "x": matrix_rotation_deg[0],
            "y": matrix_rotation_deg[1],
            "z": matrix_rotation_deg[2],
        },
        "fibre": {
            "x": fibre_rotation_deg[0],
            "y": fibre_rotation_deg[1],
            "z": fibre_rotation_deg[2],
        },
    }
    if solver_config.write_vtk:
        vtk_path = _write_vtk_visualization(mesh_file, result_dir, dimension)
        summary_payload["vtk_path"] = str(vtk_path)
    response_files = _write_response_exports(result_dir, summary_payload)
    summary_payload["response_files"] = [str(path) for path in response_files]
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    vtk_path = result_dir / "ferrite_homogenization.vtu"
    return FerriteSolveResult(
        summary_path=summary_path,
        stdout_path=stdout_path,
        vtk_path=vtk_path if vtk_path.exists() else None,
        homogenized_stiffness=cast(
            list[list[float]],
            summary_payload["homogenized_stiffness_voigt"],
        ),
        engineering_constants=cast(dict[str, float], summary_payload["engineering_constants"]),
        response_files=response_files,
    )


def _export_ferrite_inputs(
    mesh_path: Path,
    output_dir: Path,
) -> dict[str, Path | float | int | str]:
    mesh = meshio.read(mesh_path)
    cell_type: str
    cell_blocks: list[np.ndarray]
    dimension: int
    if any(block.type == "tetra" for block in mesh.cells):
        cell_type = "tetra"
        cell_blocks = [block.data for block in mesh.cells if block.type == "tetra"]
        dimension = 3
    elif any(block.type == "triangle" for block in mesh.cells):
        cell_type = "triangle"
        cell_blocks = [block.data for block in mesh.cells if block.type == "triangle"]
        dimension = 2
    else:
        raise SolverError("Ferrite bridge currently supports triangle and tetrahedral meshes only.")

    physical_data = mesh.cell_data_dict.get("gmsh:physical")
    if physical_data is None or cell_type not in physical_data:
        raise SolverError("Mesh is missing gmsh physical cell tags required for Ferrite materials.")

    nodes = np.asarray(mesh.points[:, :dimension], dtype=np.float64)
    cells = np.vstack([np.asarray(block, dtype=np.int64) for block in cell_blocks]) + 1
    cell_tags = np.asarray(physical_data[cell_type], dtype=np.int64).reshape(-1, 1)

    nodes_path = output_dir / "ferrite_nodes.csv"
    cells_path = output_dir / "ferrite_cells.csv"
    tags_path = output_dir / "ferrite_cell_tags.csv"
    np.savetxt(nodes_path, nodes, delimiter=",")
    np.savetxt(cells_path, cells, delimiter=",", fmt="%d")
    np.savetxt(tags_path, cell_tags, delimiter=",", fmt="%d")

    return {
        "nodes": nodes_path,
        "cells": cells_path,
        "cell_tags": tags_path,
        "cell_type": cell_type,
        "dimension": dimension,
        "xmin": float(nodes[:, 0].min()),
        "xmax": float(nodes[:, 0].max()),
        "ymin": float(nodes[:, 1].min()),
        "ymax": float(nodes[:, 1].max()),
        "zmin": float(nodes[:, 2].min()) if dimension == 3 else 0.0,
        "zmax": float(nodes[:, 2].max()) if dimension == 3 else 0.0,
    }


def _extract_summary_payload(stdout: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    stiffness_rows: dict[int, list[float]] = {}
    average_stresses: dict[int, list[float]] = {}
    boundary_tractions: dict[int, dict[str, list[float]]] = {}

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        if key.startswith("C_ROW_"):
            stiffness_rows[int(key.removeprefix("C_ROW_"))] = [
                float(item) for item in value.split(",")
            ]
            continue
        if key.startswith("SIGMA_BAR_"):
            average_stresses[int(key.removeprefix("SIGMA_BAR_"))] = [
                float(item) for item in value.split(",")
            ]
            continue
        if key.startswith("TRACTION_"):
            _, case_index, boundary = key.split("_", maxsplit=2)
            case_tractions = boundary_tractions.setdefault(int(case_index), {})
            case_tractions[boundary.lower()] = [float(item) for item in value.split(",")]
            continue
        if key == "MATRIX_VOLUME_FRACTION":
            summary["matrix_volume_fraction"] = float(value)
        elif key == "FIBRE_VOLUME_FRACTION":
            summary["fibre_volume_fraction"] = float(value)
        elif key == "BOUNDARY_CONDITION":
            summary["boundary_condition"] = value
        elif key == "KINEMATICS":
            summary["kinematics"] = value
        elif key == "MATRIX_CELLSET":
            summary["matrix_cellset"] = value
        elif key == "FIBRE_CELLSET":
            summary["fibre_cellset"] = value
        elif key == "VTK_PATH":
            summary["vtk_path"] = None if value == "none" else value

    if not stiffness_rows:
        raise SolverError("Ferrite.jl solve did not emit a complete homogenized stiffness matrix.")
    summary["homogenized_stiffness_voigt"] = [
        stiffness_rows[index] for index in sorted(stiffness_rows)
    ]
    summary["average_stresses"] = [average_stresses[index] for index in sorted(average_stresses)]
    if boundary_tractions:
        summary["boundary_tractions"] = [
            boundary_tractions[index] for index in sorted(boundary_tractions)
        ]
    return summary


def _write_vtk_visualization(mesh_path: Path, output_dir: Path, dimension: int) -> Path:
    mesh = meshio.read(mesh_path)
    point_data: dict[str, np.ndarray] = {}
    voigt_size = 6 if dimension == 3 else 3
    for case_index in range(1, voigt_size + 1):
        fluctuation_path = output_dir / f"ferrite_solution_case_{case_index}.csv"
        macro_path = output_dir / f"ferrite_macro_strain_case_{case_index}.csv"
        if not fluctuation_path.exists() or not macro_path.exists():
            raise SolverError(
                "Ferrite visualization export expected displacement CSV files "
                "but did not find them."
            )
        fluctuation = np.loadtxt(fluctuation_path, delimiter=",", dtype=np.float64).reshape(-1)
        macro_strain = np.loadtxt(macro_path, delimiter=",", dtype=np.float64).reshape(voigt_size)
        coordinates = np.asarray(mesh.points[:, :dimension], dtype=np.float64)
        if dimension == 3:
            macro_displacement = np.column_stack(
                [
                    macro_strain[0] * coordinates[:, 0]
                    + 0.5 * macro_strain[5] * coordinates[:, 1]
                    + 0.5 * macro_strain[4] * coordinates[:, 2],
                    0.5 * macro_strain[5] * coordinates[:, 0]
                    + macro_strain[1] * coordinates[:, 1]
                    + 0.5 * macro_strain[3] * coordinates[:, 2],
                    0.5 * macro_strain[4] * coordinates[:, 0]
                    + 0.5 * macro_strain[3] * coordinates[:, 1]
                    + macro_strain[2] * coordinates[:, 2],
                ]
            )
            total_displacement = macro_displacement + fluctuation.reshape((-1, 3))
        else:
            macro_displacement = np.column_stack(
                [
                    macro_strain[0] * coordinates[:, 0] + 0.5 * macro_strain[2] * coordinates[:, 1],
                    0.5 * macro_strain[2] * coordinates[:, 0] + macro_strain[1] * coordinates[:, 1],
                ]
            )
            total_displacement = macro_displacement + fluctuation.reshape((-1, 2))
        padded = np.zeros((total_displacement.shape[0], 3), dtype=np.float64)
        padded[:, :dimension] = total_displacement
        point_data[f"u_case_{case_index}"] = padded

    vtk_mesh = meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=mesh.cells,
        point_data=point_data,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
    )
    vtk_path = output_dir / "ferrite_homogenization.vtu"
    meshio.write(vtk_path, vtk_mesh)
    return vtk_path


def _constitutive_matrix(
    material_model: str,
    kinematics: str,
    *,
    youngs_modulus: float,
    poisson_ratio: float,
    e1: float | None,
    e2: float | None,
    e3: float | None,
    g12: float | None,
    g13: float | None,
    g23: float | None,
    nu12: float | None,
    nu13: float | None,
    nu23: float | None,
    rotation_deg: tuple[float, float, float],
) -> np.ndarray:
    if material_model == "isotropic":
        return _isotropic_constitutive_matrix(youngs_modulus, poisson_ratio, kinematics)
    assert e1 is not None and e2 is not None and g12 is not None and nu12 is not None
    return _orthotropic_constitutive_matrix(
        e1,
        e2,
        e3,
        g12,
        g13,
        g23,
        nu12,
        nu13,
        nu23,
        kinematics,
        rotation_deg,
    )


def _isotropic_constitutive_matrix(
    youngs_modulus: float,
    poisson_ratio: float,
    kinematics: str,
) -> np.ndarray:
    if kinematics == "solid":
        shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        lame_lambda = (
            youngs_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        return np.array(
            [
                [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
                [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0, 0.0, 0.0],
                [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
            ],
            dtype=np.float64,
        )
    if kinematics == "plane_stress":
        factor = youngs_modulus / (1.0 - poisson_ratio**2)
        return factor * np.array(
            [
                [1.0, poisson_ratio, 0.0],
                [poisson_ratio, 1.0, 0.0],
                [0.0, 0.0, (1.0 - poisson_ratio) / 2.0],
            ],
            dtype=np.float64,
        )

    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return np.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0],
            [0.0, 0.0, shear_modulus],
        ],
        dtype=np.float64,
    )


def _orthotropic_constitutive_matrix(
    e1: float,
    e2: float,
    e3: float | None,
    g12: float,
    g13: float | None,
    g23: float | None,
    nu12: float,
    nu13: float | None,
    nu23: float | None,
    kinematics: str,
    rotation_deg: tuple[float, float, float],
) -> np.ndarray:
    if kinematics == "solid":
        assert e3 is not None and g13 is not None and g23 is not None
        assert nu13 is not None and nu23 is not None
        nu21 = nu12 * e2 / e1
        nu31 = nu13 * e3 / e1
        nu32 = nu23 * e3 / e2
        compliance = np.array(
            [
                [1.0 / e1, -nu12 / e1, -nu13 / e1, 0.0, 0.0, 0.0],
                [-nu21 / e2, 1.0 / e2, -nu23 / e2, 0.0, 0.0, 0.0],
                [-nu31 / e3, -nu32 / e3, 1.0 / e3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / g23, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / g13, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / g12],
            ],
            dtype=np.float64,
        )
        stiffness = np.linalg.inv(compliance)
        return _rotate_solid_stiffness(stiffness, rotation_deg)
    nu21 = nu12 * e2 / e1
    compliance = np.array(
        [
            [1.0 / e1, -nu12 / e1, 0.0],
            [-nu21 / e2, 1.0 / e2, 0.0],
            [0.0, 0.0, 1.0 / g12],
        ],
        dtype=np.float64,
    )
    stiffness = np.linalg.inv(compliance)
    if kinematics == "plane_stress":
        return _rotate_plane_stress_stiffness(stiffness, rotation_deg[2])
    return stiffness


def _rotate_plane_stress_stiffness(stiffness: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-12:
        return stiffness
    m = float(np.cos(np.deg2rad(angle_deg)))
    n = float(np.sin(np.deg2rad(angle_deg)))
    q11 = stiffness[0, 0]
    q22 = stiffness[1, 1]
    q12 = stiffness[0, 1]
    q66 = stiffness[2, 2]
    q11b = q11 * m**4 + 2.0 * (q12 + 2.0 * q66) * m**2 * n**2 + q22 * n**4
    q22b = q11 * n**4 + 2.0 * (q12 + 2.0 * q66) * m**2 * n**2 + q22 * m**4
    q12b = (q11 + q22 - 4.0 * q66) * m**2 * n**2 + q12 * (m**4 + n**4)
    q16b = (q11 - q12 - 2.0 * q66) * m**3 * n - (q22 - q12 - 2.0 * q66) * m * n**3
    q26b = (q11 - q12 - 2.0 * q66) * m * n**3 - (q22 - q12 - 2.0 * q66) * m**3 * n
    q66b = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m**2 * n**2 + q66 * (m**4 + n**4)
    return np.array(
        [
            [q11b, q12b, q16b],
            [q12b, q22b, q26b],
            [q16b, q26b, q66b],
        ],
        dtype=np.float64,
    )


def _rotate_solid_stiffness(
    stiffness: np.ndarray,
    rotation_deg: tuple[float, float, float],
) -> np.ndarray:
    if max(abs(value) for value in rotation_deg) < 1e-12:
        return stiffness
    rotation = _rotation_matrix_xyz(rotation_deg)
    rotated_columns: list[np.ndarray] = []
    for basis_index in range(6):
        global_strain_voigt = np.zeros(6, dtype=np.float64)
        global_strain_voigt[basis_index] = 1.0
        global_strain_tensor = _strain_voigt_to_tensor_3d(global_strain_voigt)
        local_strain_tensor = rotation.T @ global_strain_tensor @ rotation
        local_strain_voigt = _strain_tensor_to_voigt_3d(local_strain_tensor)
        local_stress_voigt = stiffness @ local_strain_voigt
        local_stress_tensor = _stress_voigt_to_tensor_3d(local_stress_voigt)
        global_stress_tensor = rotation @ local_stress_tensor @ rotation.T
        rotated_columns.append(_stress_tensor_to_voigt_3d(global_stress_tensor))
    return np.column_stack(rotated_columns)


def _rotation_matrix_xyz(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    angle_x_deg, angle_y_deg, angle_z_deg = rotation_deg
    cx = float(np.cos(np.deg2rad(angle_x_deg)))
    sx = float(np.sin(np.deg2rad(angle_x_deg)))
    cy = float(np.cos(np.deg2rad(angle_y_deg)))
    sy = float(np.sin(np.deg2rad(angle_y_deg)))
    cz = float(np.cos(np.deg2rad(angle_z_deg)))
    sz = float(np.sin(np.deg2rad(angle_z_deg)))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _strain_voigt_to_tensor_3d(strain_voigt: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [strain_voigt[0], 0.5 * strain_voigt[5], 0.5 * strain_voigt[4]],
            [0.5 * strain_voigt[5], strain_voigt[1], 0.5 * strain_voigt[3]],
            [0.5 * strain_voigt[4], 0.5 * strain_voigt[3], strain_voigt[2]],
        ],
        dtype=np.float64,
    )


def _strain_tensor_to_voigt_3d(strain_tensor: np.ndarray) -> np.ndarray:
    return np.array(
        [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            strain_tensor[2, 2],
            2.0 * strain_tensor[1, 2],
            2.0 * strain_tensor[0, 2],
            2.0 * strain_tensor[0, 1],
        ],
        dtype=np.float64,
    )


def _stress_voigt_to_tensor_3d(stress_voigt: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
            [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
            [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
        ],
        dtype=np.float64,
    )


def _stress_tensor_to_voigt_3d(stress_tensor: np.ndarray) -> np.ndarray:
    return np.array(
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ],
        dtype=np.float64,
    )


def _engineering_constants_from_stiffness(
    homogenized_stiffness: list[list[float]],
) -> dict[str, float]:
    stiffness = np.asarray(homogenized_stiffness, dtype=np.float64)
    compliance = np.linalg.inv(stiffness)
    if stiffness.shape == (6, 6):
        ex = 1.0 / compliance[0, 0]
        ey = 1.0 / compliance[1, 1]
        ez = 1.0 / compliance[2, 2]
        gyz = 1.0 / compliance[3, 3]
        gxz = 1.0 / compliance[4, 4]
        gxy = 1.0 / compliance[5, 5]
        return {
            "ex": float(ex),
            "ey": float(ey),
            "ez": float(ez),
            "gyz": float(gyz),
            "gxz": float(gxz),
            "gxy": float(gxy),
            "nuxy": float(-compliance[0, 1] * ex),
            "nuyx": float(-compliance[1, 0] * ey),
            "nuxz": float(-compliance[0, 2] * ex),
            "nuzx": float(-compliance[2, 0] * ez),
            "nuyz": float(-compliance[1, 2] * ey),
            "nuzy": float(-compliance[2, 1] * ez),
        }
    ex = 1.0 / compliance[0, 0]
    ey = 1.0 / compliance[1, 1]
    gxy = 1.0 / compliance[2, 2]
    nuxy = -compliance[0, 1] * ex
    nuyx = -compliance[1, 0] * ey
    return {
        "ex": float(ex),
        "ey": float(ey),
        "gxy": float(gxy),
        "nuxy": float(nuxy),
        "nuyx": float(nuyx),
    }


def _resolve_material_rotation_deg(
    *,
    dimension: int,
    phase_label: str,
    configured_angle_deg: float | None,
    configured_angle_x_deg: float | None,
    configured_angle_y_deg: float | None,
    configured_angle_z_deg: float | None,
    geometry_metadata: dict[str, object] | None,
    fallback_key: str | None,
) -> tuple[float, float, float]:
    if dimension == 2:
        if configured_angle_deg is not None:
            return (0.0, 0.0, float(configured_angle_deg))
        metadata_rotation = _phase_rotation_from_metadata(geometry_metadata, phase_label)
        if metadata_rotation is not None:
            return metadata_rotation
        if geometry_metadata is not None and fallback_key is not None:
            value = geometry_metadata.get(fallback_key)
            if isinstance(value, (int, float)):
                return (0.0, 0.0, float(value))
        return (0.0, 0.0, 0.0)

    if configured_angle_deg is not None:
        return (0.0, 0.0, float(configured_angle_deg))
    angle_x_deg = 0.0 if configured_angle_x_deg is None else float(configured_angle_x_deg)
    angle_y_deg = 0.0 if configured_angle_y_deg is None else float(configured_angle_y_deg)
    angle_z_deg = 0.0 if configured_angle_z_deg is None else float(configured_angle_z_deg)
    metadata_rotation = _phase_rotation_from_metadata(geometry_metadata, phase_label)
    if metadata_rotation is not None and (
        configured_angle_x_deg is None
        and configured_angle_y_deg is None
        and configured_angle_z_deg is None
    ):
        return metadata_rotation
    if geometry_metadata is not None and fallback_key is not None and angle_z_deg == 0.0:
        value = geometry_metadata.get(fallback_key)
        if isinstance(value, (int, float)):
            angle_z_deg = float(value)
    return (angle_x_deg, angle_y_deg, angle_z_deg)


def _phase_rotation_from_metadata(
    geometry_metadata: dict[str, object] | None,
    phase_label: str,
) -> tuple[float, float, float] | None:
    if geometry_metadata is None:
        return None
    rotations = geometry_metadata.get("phase_orientation_rotations_deg")
    if not isinstance(rotations, dict):
        return None
    phase_rotation = rotations.get(phase_label)
    if not isinstance(phase_rotation, dict):
        return None
    x_value = phase_rotation.get("x", 0.0)
    y_value = phase_rotation.get("y", 0.0)
    z_value = phase_rotation.get("z", 0.0)
    if not all(isinstance(value, (int, float)) for value in (x_value, y_value, z_value)):
        return None
    return (float(x_value), float(y_value), float(z_value))


def _write_response_exports(output_dir: Path, summary_payload: dict[str, Any]) -> list[Path]:
    written: list[Path] = []
    stiffness = np.asarray(summary_payload["homogenized_stiffness_voigt"], dtype=np.float64)
    stiffness_path = output_dir / "homogenized_stiffness.csv"
    np.savetxt(stiffness_path, stiffness, delimiter=",")
    written.append(stiffness_path)

    engineering = cast(dict[str, float], summary_payload["engineering_constants"])
    engineering_path = output_dir / "engineering_constants.csv"
    engineering_lines = ["name,value"] + [f"{key},{value}" for key, value in engineering.items()]
    engineering_path.write_text("\n".join(engineering_lines) + "\n", encoding="utf-8")
    written.append(engineering_path)

    average_stresses = cast(list[list[float]], summary_payload["average_stresses"])
    dimension = cast(int, summary_payload.get("dimension", 2))
    macro_strains = (
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        if dimension == 2
        else [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    stress_strain_path = output_dir / "stress_strain_response.csv"
    stress_rows = (
        ["case,exx,eyy,gxy,sxx,syy,txy"]
        if dimension == 2
        else ["case,exx,eyy,ezz,gyz,gxz,gxy,sxx,syy,szz,tyz,txz,txy"]
    )
    for index, (strain, stress) in enumerate(
        zip(macro_strains, average_stresses, strict=True),
        start=1,
    ):
        stress_rows.append(f"{index}," + ",".join(str(value) for value in [*strain, *stress]))
    stress_strain_path.write_text("\n".join(stress_rows) + "\n", encoding="utf-8")
    written.append(stress_strain_path)

    traction_path = output_dir / "traction_response.csv"
    traction_rows = (
        ["case,boundary,tx,ty,source"]
        if dimension == 2
        else ["case,boundary,tx,ty,tz,source"]
    )
    boundary_tractions = cast(
        list[dict[str, list[float]]] | None,
        summary_payload.get("boundary_tractions"),
    )
    if boundary_tractions is not None:
        for index, case_tractions in enumerate(boundary_tractions, start=1):
            boundaries = (
                ["right", "left", "top", "bottom"]
                if dimension == 2
                else ["right", "left", "back", "front", "top", "bottom"]
            )
            for boundary in boundaries:
                traction = case_tractions[boundary]
                traction_rows.append(
                    f"{index},{boundary},{','.join(str(value) for value in traction)},facet_average"
                )
    else:
        boundary_normals = (
            {
                "right": (1.0, 0.0),
                "left": (-1.0, 0.0),
                "top": (0.0, 1.0),
                "bottom": (0.0, -1.0),
            }
            if dimension == 2
            else {
                "right": (1.0, 0.0, 0.0),
                "left": (-1.0, 0.0, 0.0),
                "back": (0.0, 1.0, 0.0),
                "front": (0.0, -1.0, 0.0),
                "top": (0.0, 0.0, 1.0),
                "bottom": (0.0, 0.0, -1.0),
            }
        )
        for index, stress in enumerate(average_stresses, start=1):
            if dimension == 2:
                sigma = np.array([[stress[0], stress[2]], [stress[2], stress[1]]], dtype=np.float64)
            else:
                sigma = np.array(
                    [
                        [stress[0], stress[5], stress[4]],
                        [stress[5], stress[1], stress[3]],
                        [stress[4], stress[3], stress[2]],
                    ],
                    dtype=np.float64,
                )
            for boundary, normal in boundary_normals.items():
                traction_vector = sigma @ np.asarray(normal, dtype=np.float64)
                traction_rows.append(
                    f"{index},{boundary},{','.join(str(value) for value in traction_vector)},"
                    "homogenized_stress"
                )
    traction_path.write_text("\n".join(traction_rows) + "\n", encoding="utf-8")
    written.append(traction_path)
    return written
