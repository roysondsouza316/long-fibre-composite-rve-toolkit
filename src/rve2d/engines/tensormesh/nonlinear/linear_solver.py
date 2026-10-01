"""Sparse linear solves for the Newton iterations.

``tensormesh``: builds a TensorMesh ``SparseMatrix`` (a ``torch_sla.SparseTensor``) on the
system's device and calls its ``solve``; torch-sla dispatches to its CUDA backends (cuDSS
for direct LU when installed) on GPU, so the whole Newton loop stays on the GPU with
``device: cuda``, and to SciPy's SuperLU on the CPU.

``scipy``: SciPy's SuperLU with a minimum-degree ordering on ``A^T + A``, the same CPU
library TensorMesh dispatches to, called directly (less conversion overhead and a better
ordering for these matrices); robust for the non-symmetric, possibly indefinite tangents of
softening cohesive interfaces.

``auto`` (default): ``scipy`` on the CPU, ``tensormesh`` on a GPU. Every option runs on
Linux, macOS and Windows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from rve2d.engines.tensormesh.nonlinear.assembly import SparseSystem
from rve2d.exceptions import SolverError


def resolve_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        return "scipy" if device.type == "cpu" else "tensormesh"
    if backend not in {"scipy", "tensormesh"}:
        raise SolverError(
            f"Unknown linear solver backend {backend!r}; use auto, scipy or tensormesh."
        )
    return backend


def solve(
    system: SparseSystem, rhs: torch.Tensor, backend: str, method: str = "lu"
) -> torch.Tensor:
    if backend == "scipy":
        matrix = scipy.sparse.coo_matrix(
            (
                system.value.cpu().numpy(),
                (system.row.cpu().numpy(), system.col.cpu().numpy()),
            ),
            shape=(system.size, system.size),
        )
        b = rhs.cpu().numpy()
        try:
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
