from __future__ import annotations

import numpy as np

from ..core.datatypes import SubspaceResult


def identify_dominant_subspace(projected_matrix: np.ndarray, *, rank_cap: int = 8, rank_floor: int = 1, subspace_overlap: float | None = None) -> SubspaceResult:
    projected_matrix = 0.5 * (np.asarray(projected_matrix, dtype=np.float64) + np.asarray(projected_matrix, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("projected sensitivity matrix has no mass")
    eigenvalues = eigenvalues / total
    cumulative = np.cumsum(eigenvalues)
    r90 = int(np.searchsorted(cumulative, 0.9) + 1)
    if rank_floor < 1 or rank_floor > rank_cap:
        raise ValueError("rank_floor must satisfy 1 <= rank_floor <= rank_cap")
    rank_used = min(max(r90, int(rank_floor)), rank_cap)
    basis = eigenvectors[:, :rank_used]
    projector = basis @ basis.T
    captured = float(eigenvalues[:rank_used].sum())
    status = "RESOLVED" if subspace_overlap is None or subspace_overlap >= 0.90 else "UNRESOLVED"
    return SubspaceResult(basis, projector, eigenvalues, rank_used, r90, captured, status, subspace_overlap)
