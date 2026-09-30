"""Sparse linear solves for the Newton iterations.

``pardiso`` (CPU, used by ``auto`` when ``pypardiso`` is installed): Intel MKL PARDISO for
real non-symmetric matrices, several times faster than SuperLU on RVE tangents.

``scipy`` (CPU fallback): SuperLU with a minimum-degree ordering on ``A^T + A``; robust for
the non-symmetric, possibly indefinite tangents of softening cohesive interfaces.

``tensormesh``: builds a TensorMesh ``SparseMatrix`` (a ``torch_sla.SparseTensor``) on the
system's device and calls its ``solve``; torch-sla dispatches to SciPy on CPU and to its CUDA
backends (cuDSS for direct LU when installed) on GPU, so the whole Newton loop can stay on the
GPU with ``device: cuda``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from rve2d.exceptions import SolverError
from rve2d.nonlinear.assembly import SparseSystem


def resolve_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        if device.type != "cpu":
            return "tensormesh"
        return "pardiso" if _pardiso_available() else "scipy"
    if backend not in {"scipy", "pardiso", "tensormesh"}:
        raise SolverError(
            f"Unknown linear solver backend {backend!r}; use auto, scipy, pardiso or tensormesh."
        )
    if backend == "pardiso" and not _pardiso_available():
        raise SolverError("linear_solver: pardiso needs `pip install pypardiso`.")
    return backend


def _pardiso_available() -> bool:
    try:
        import pypardiso  # noqa: F401
    except ImportError:
        return False
    return True


def solve(
    system: SparseSystem, rhs: torch.Tensor, backend: str, method: str = "lu"
) -> torch.Tensor:
    if backend in {"scipy", "pardiso"}:
        matrix = scipy.sparse.coo_matrix(
            (
                system.value.cpu().numpy(),
                (system.row.cpu().numpy(), system.col.cpu().numpy()),
            ),
            shape=(system.size, system.size),
        )
        b = rhs.cpu().numpy()
        try:
            if backend == "pardiso":
                import pypardiso

                solution = pypardiso.spsolve(matrix.tocsr(), b)
            else:
                lu = scipy.sparse.linalg.splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
                solution = lu.solve(b)
        except (RuntimeError, ValueError) as exc:  # singular or failed factorisation
            raise SolverError(f"Sparse factorisation failed: {exc}") from exc
        if not np.all(np.isfinite(solution)):
            raise SolverError("Sparse direct solve produced a non-finite solution.")
        return torch.as_tensor(solution, dtype=rhs.dtype, device=rhs.device)

    try:
        from tensormesh.sparse import SparseMatrix
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SolverError("The tensormesh backend needs `pip install tensormesh-fem`.") from exc
    coo = torch.sparse_coo_tensor(
        torch.stack([system.row, system.col]), system.value, (system.size, system.size)
    ).coalesce()
    indices = coo.indices()
    matrix_tm = SparseMatrix(coo.values(), indices[0], indices[1], (system.size, system.size))
    solution_tm: torch.Tensor = matrix_tm.solve(rhs, method=method)
    if not bool(torch.isfinite(solution_tm).all()):
        raise SolverError("TensorMesh/torch-sla sparse solve produced a non-finite solution.")
    return solution_tm
