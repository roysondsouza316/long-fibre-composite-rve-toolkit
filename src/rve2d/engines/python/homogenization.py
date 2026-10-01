"""Linear homogenization in the Python engine (NumPy/SciPy, no PyTorch needed).

First-order, small-strain homogenization on linear triangles/tetrahedra: the nodal
fluctuations ``w`` solve ``K w = -F E`` with ``K = sum V B^T D B`` and ``F = sum V B^T D``,
factorised once and solved for every unit macro strain ``E``; column ``j`` of the effective
stiffness is the volume-averaged stress of load case ``j``. Periodic fluctuations are tied
node to node and one interior node is fixed (so the system is regular); ``dirichlet`` fixes
the fluctuation on the whole boundary. Identical formulation to the Julia engine's
``FerriteRVE.homogenize``.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from rve2d.engines.common.homogenization import EngineSolution, HomogenizationProblem
from rve2d.engines.common.materials import ACTIVE_VOIGT, FloatArray
from rve2d.engines.python.constraints import boundary_tolerance, build_dof_map
from rve2d.engines.python.mesh import RVEMesh, build_rve_mesh
from rve2d.exceptions import SolverError

COMPONENTS = {"plane_stress": 2, "plane_strain": 2, "generalized_plane_strain": 3, "solid": 3}
_FACETS = {
    2: np.array([[0, 1], [1, 2], [2, 0]]),
    3: np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]),
}
_FACES = {
    2: (("left", 0, -1), ("right", 0, 1), ("bottom", 1, -1), ("top", 1, 1)),
    3: (
        ("left", 0, -1),
        ("right", 0, 1),
        ("front", 1, -1),
        ("back", 1, 1),
        ("bottom", 2, -1),
        ("top", 2, 1),
    ),
}


def solve(problem: HomogenizationProblem) -> EngineSolution:
    mesh = build_rve_mesh(
        problem.mesh_path,
        problem.matrix_phase_id,
        problem.fibre_phase_id,
        cohesive_interfaces=False,
    )
    if mesh.dim != problem.dimension:
        raise SolverError(f"Mesh is {mesh.dim}D but the config has dimension {problem.dimension}.")
    ncomp = COMPONENTS[problem.kinematics]
    voigt = list(ACTIVE_VOIGT[problem.kinematics])
    ne = len(voigt)

    gradients, volumes = simplex_gradients(mesh.points, mesh.cells)
    strain = strain_matrices(gradients, ncomp)[:, voigt, :]  # (cells, ne, nb)
    is_fibre = mesh.phase == mesh.fibre_phase_id
    stiffness = np.where(
        is_fibre[:, None, None], problem.fibre_stiffness[None], problem.matrix_stiffness[None]
    )
    dofs = (mesh.cells[:, :, None] * ncomp + np.arange(ncomp)).reshape(mesh.n_cells, -1)
    dof_map = build_dof_map(mesh, problem.boundary_condition, components=ncomp)
    reduced = dof_map.full_to_reduced[dofs]  # (cells, nb), -1 where fixed
    nr = dof_map.n_reduced

    db = np.einsum("cij,cjk->cik", stiffness, strain)
    k_elem = np.einsum("cji,cjk->cik", strain, db) * volumes[:, None, None]
    f_elem = np.einsum("cji,cjk->cik", strain, stiffness) * volumes[:, None, None]
    rows = np.broadcast_to(reduced[:, :, None], k_elem.shape).reshape(-1)
    cols = np.broadcast_to(reduced[:, None, :], k_elem.shape).reshape(-1)
    keep = (rows >= 0) & (cols >= 0)
    matrix = scipy.sparse.coo_matrix(
        (k_elem.reshape(-1)[keep], (rows[keep], cols[keep])), shape=(nr, nr)
    ).tocsc()
    load = np.zeros((nr, ne))
    flat_rows = reduced.reshape(-1)
    has_row = flat_rows >= 0
    np.add.at(load, flat_rows[has_row], f_elem.reshape(-1, ne)[has_row])
    try:
        factor = scipy.sparse.linalg.splu(matrix, permc_spec="MMD_AT_PLUS_A")
    except RuntimeError as exc:
        raise SolverError(
            "The RVE stiffness matrix is singular: check the boundary condition and the mesh."
        ) from exc
    solution = factor.solve(-load)  # one factorisation, one solve per unit macro strain
    if not np.all(np.isfinite(solution)):
        raise SolverError("The linear solve produced a non-finite fluctuation field.")

    total_volume = float(volumes.sum())
    fluctuations = np.zeros((ne, mesh.n_nodes * ncomp))
    owned = dof_map.full_to_reduced >= 0
    fluctuations[:, owned] = solution[dof_map.full_to_reduced[owned]].T
    cell_strain = np.einsum("cij,ncj->nci", strain, fluctuations[:, dofs])  # (cases, cells, ne)
    cell_strain += np.eye(ne)[:, None, :]
    cell_stress = np.einsum("cij,ncj->nci", stiffness, cell_strain)
    effective = np.einsum("c,nci->in", volumes, cell_stress) / total_volume

    vtk_path = None
    if problem.write_vtk:
        vtk_path = write_vtk(
            problem.output_dir / "homogenization.vtu", mesh, fluctuations, cell_stress, problem
        )
    return EngineSolution(
        stiffness=effective,
        tractions=face_tractions(mesh, cell_stress, voigt),
        fibre_volume_fraction=float(volumes[is_fibre].sum() / total_volume),
        unknowns=nr,
        vtk_path=vtk_path,
        details={"anchor_node": dof_map.anchor_node},
    )


def simplex_gradients(points: FloatArray, cells: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Shape-function gradients ``(cells, nodes, dim)`` and volumes of linear simplices."""
    dim = points.shape[1]
    coords = points[cells]  # (cells, dim + 1, dim)
    ones = np.ones((cells.shape[0], dim + 1, 1))
    jacobian = np.concatenate([ones, coords], axis=2)  # rows [1, x_a]
    gradients = np.linalg.inv(jacobian)[:, 1:, :].transpose(0, 2, 1)
    volumes = np.abs(np.linalg.det(jacobian)) / (2.0 if dim == 2 else 6.0)
    return np.ascontiguousarray(gradients), volumes


def strain_matrices(gradients: FloatArray, ncomp: int) -> FloatArray:
    """All six engineering-strain rows ``(xx, yy, zz, yz, xz, xy)`` per cell."""
    n_cells, n_nodes, dim = gradients.shape
    gx, gy = gradients[:, :, 0], gradients[:, :, 1]
    gz = gradients[:, :, 2] if dim == 3 else np.zeros_like(gx)
    b = np.zeros((n_cells, 6, n_nodes, ncomp))
    b[:, 0, :, 0] = gx
    b[:, 5, :, 0] = gy
    b[:, 4, :, 0] = gz
    b[:, 1, :, 1] = gy
    b[:, 5, :, 1] = gx
    b[:, 3, :, 1] = gz
    if ncomp == 3:
        b[:, 2, :, 2] = gz
        b[:, 3, :, 2] = gy
        b[:, 4, :, 2] = gx
    return b.reshape(n_cells, 6, n_nodes * ncomp)


def face_tractions(
    mesh: RVEMesh, cell_stress: FloatArray, voigt: list[int]
) -> list[dict[str, list[float]]]:
    """Facet-averaged traction ``sigma n`` on every face of the RVE box, per load case."""
    tol = boundary_tolerance(mesh)
    facets = mesh.cells[:, _FACETS[mesh.dim]]  # (cells, facets, dim)
    coords = mesh.points[facets]  # (cells, facets, dim, dim)
    if mesh.dim == 2:
        edge = coords[:, :, 1] - coords[:, :, 0]
        measure = np.linalg.norm(edge, axis=2)
    else:
        a = coords[:, :, 1] - coords[:, :, 0]
        b = coords[:, :, 2] - coords[:, :, 0]
        measure = 0.5 * np.linalg.norm(np.cross(a, b), axis=2)
    full = np.zeros(cell_stress.shape[:2] + (6,))
    full[:, :, voigt] = cell_stress
    components = 3 if len(voigt) == 6 else 2
    tensor_index = {0: (0, 5, 4), 1: (5, 1, 3), 2: (4, 3, 2)}  # rows of sigma in Voigt slots
    result: list[dict[str, list[float]]] = [{} for _ in range(cell_stress.shape[0])]
    for name, axis, sign in _FACES[mesh.dim]:
        plane = mesh.lower[axis] if sign < 0 else mesh.upper[axis]
        on_face = np.all(np.abs(coords[..., axis] - plane) < tol, axis=2)  # (cells, facets)
        cell_index, _ = np.nonzero(on_face)
        weights = measure[on_face]
        total = float(weights.sum())
        for case in range(cell_stress.shape[0]):
            # traction_i = sigma_i,axis * sign
            column = [full[case, cell_index, tensor_index[i][axis]] for i in range(components)]
            traction = [sign * float((weights * c).sum()) / total for c in column]
            result[case][name] = traction if total > 0.0 else [float("nan")] * components
    return result


def write_vtk(
    path: Path,
    mesh: RVEMesh,
    fluctuations: FloatArray,
    cell_stress: FloatArray,
    problem: HomogenizationProblem,
) -> Path:
    """Per-case total displacement ``E x + w`` and cell stresses, plus the phases."""
    names = problem.voigt
    ncomp = fluctuations.shape[1] // mesh.n_nodes
    points = np.zeros((mesh.n_nodes, 3))
    points[:, : mesh.dim] = mesh.points
    point_data = {}
    cell_data: dict[str, list[np.ndarray]] = {"phase": [mesh.phase.astype(np.float64)]}
    for case, name in enumerate(names):
        strain = np.zeros(6)
        strain[ACTIVE_VOIGT[problem.kinematics][case]] = 1.0
        gradient = np.array(
            [
                [strain[0], strain[5] / 2, strain[4] / 2],
                [strain[5] / 2, strain[1], strain[3] / 2],
                [strain[4] / 2, strain[3] / 2, strain[2]],
            ]
        )
        u = points @ gradient.T
        u[:, ncomp:] = 0.0
        u[:, :ncomp] += fluctuations[case].reshape(mesh.n_nodes, ncomp)
        point_data[f"u_case_{case + 1}_{name}"] = u
        cell_data[f"stress_case_{case + 1}_{name}"] = [cell_stress[case]]
    cell_type = "triangle" if mesh.dim == 2 else "tetra"
    meshio.write(
        path,
        meshio.Mesh(points, [(cell_type, mesh.cells)], point_data=point_data, cell_data=cell_data),
    )
    return path
