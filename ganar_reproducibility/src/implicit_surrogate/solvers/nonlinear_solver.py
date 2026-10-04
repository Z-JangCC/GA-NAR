from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.exceptions import BranchContinuationError


@dataclass(frozen=True)
class NewtonResult:
    state: np.ndarray
    residual_norm: float
    iterations: int
    success: bool
    message: str


class NewtonSolver:
    """Fixed-equation Newton corrector; failure is explicit, never silently rescued."""

    def __init__(self, tolerance: float = 1e-12, max_iterations: int = 50, min_step: float = 1e-8) -> None:
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.min_step = float(min_step)

    def solve(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        jacobian: Callable[[np.ndarray], np.ndarray],
        initial_state: np.ndarray,
    ) -> NewtonResult:
        state = np.asarray(initial_state, dtype=np.float64).copy()
        for iteration in range(self.max_iterations + 1):
            values = np.asarray(residual(state), dtype=np.float64)
            residual_norm = float(np.max(np.abs(values)))
            if residual_norm <= self.tolerance:
                return NewtonResult(state, residual_norm, iteration, True, "converged")
            matrix = np.asarray(jacobian(state), dtype=np.float64)
            try:
                delta = np.linalg.solve(matrix, values)
            except np.linalg.LinAlgError as exc:
                raise BranchContinuationError("Newton coefficient matrix is singular") from exc
            # Backtracking only controls the Newton corrector step; it never changes the solver family.
            step = 1.0
            accepted = False
            while step >= self.min_step:
                candidate = state - step * delta
                candidate_norm = float(np.max(np.abs(residual(candidate))))
                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    state = candidate
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                raise BranchContinuationError("Newton corrector failed to decrease residual")
        raise BranchContinuationError("Newton corrector reached max_iterations")

