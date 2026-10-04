from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import LocalModeGeometry


@dataclass(frozen=True)
class ModeCapacityTarget:
    """Auditable NRGD response-allocation target for one population.

    Columns of ``weights`` are ``[smooth, mode_1, ..., mode_r, complement]``.
    The first column is the capacity not assigned to a nonlinear physical
    mode; all columns sum to one for every valid centre.
    """

    mode_strengths: np.ndarray
    complement_strength: np.ndarray
    total_strength: np.ndarray
    total_capacity: np.ndarray
    mode_fractions: np.ndarray
    weights: np.ndarray
    valid_centers: np.ndarray
    epsilon: float
    mean_strength: float
    basis: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.mode_strengths.shape[1])

    @property
    def num_centers(self) -> int:
        return int(self.weights.shape[0])


def build_mode_capacity_targets(
    geometry: LocalModeGeometry | np.ndarray,
    sensitivity_basis: np.ndarray,
    *,
    mean_strength: float | None = None,
    epsilon: float = 1e-12,
) -> ModeCapacityTarget:
    """Convert local physical geometry into conserved mode allocations.

    For each centre, ``b_ij = u_j.T B_i u_j`` and ``b_perp`` is the trace in
    the orthogonal complement.  The construction follows
    ``alpha=e/(e+mean(e))`` and ``w=[1-alpha, alpha*pi]`` exactly, so it
    separates *how much* nonlinear capacity is needed from *where* it goes.
    """
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    basis = np.asarray(sensitivity_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("sensitivity_basis must be a matrix")
    matrices = geometry.matrices if isinstance(geometry, LocalModeGeometry) else np.asarray(geometry, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] != basis.shape[0]:
        raise ValueError("geometry matrices and sensitivity_basis have incompatible shapes")
    # Re-orthonormalising is intentional: allocation must be invariant to a
    # harmless rescaling/roundoff of the frozen eigenvectors.
    basis, _ = np.linalg.qr(basis, mode="reduced")
    rank = basis.shape[1]
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    mode_strengths = np.einsum("ir,nij,jr->nr", basis, symmetric, basis)
    mode_strengths = np.maximum(mode_strengths, 0.0)
    total_strength = np.maximum(np.trace(symmetric, axis1=1, axis2=2), 0.0)
    captured = np.sum(mode_strengths, axis=1)
    complement_strength = np.maximum(total_strength - captured, 0.0)
    valid = np.isfinite(total_strength) & np.isfinite(mode_strengths).all(axis=1)
    if isinstance(geometry, LocalModeGeometry):
        valid &= geometry.valid_centers
    positive = total_strength[valid]
    reference_mean = float(np.mean(positive)) if mean_strength is None and positive.size else float(mean_strength or 0.0)
    if not np.isfinite(reference_mean) or reference_mean <= 0:
        reference_mean = 1.0
    alpha = np.zeros_like(total_strength)
    alpha[valid] = total_strength[valid] / (total_strength[valid] + reference_mean)
    raw = np.concatenate([mode_strengths, complement_strength[:, None]], axis=1) + epsilon
    fractions = raw / (total_strength[:, None] + (rank + 1) * epsilon)
    fractions[~valid] = 0.0
    weights = np.concatenate([1.0 - alpha[:, None], alpha[:, None] * fractions], axis=1)
    weights[~valid] = 0.0
    return ModeCapacityTarget(mode_strengths, complement_strength, total_strength, alpha, fractions, weights, valid, float(epsilon), reference_mean, basis)


def validate_capacity_target(target: ModeCapacityTarget, *, atol: float = 1e-8) -> dict[str, float | bool]:
    """Return machine-checkable conservation diagnostics."""
    valid = target.valid_centers
    sums = target.weights[valid].sum(axis=1) if np.any(valid) else np.empty(0)
    nonnegative = bool(np.all(target.weights[valid] >= -atol)) if np.any(valid) else False
    max_error = float(np.max(np.abs(sums - 1.0))) if sums.size else float("inf")
    return {"valid_centers": int(valid.sum()), "conservation_max_error": max_error, "conservation_pass": bool(max_error <= atol), "nonnegative_pass": nonnegative}
