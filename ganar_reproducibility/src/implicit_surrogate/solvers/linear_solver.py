from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class LinearSystemSolver:
    """LU factorization wrapper with multiple-right-hand-side reuse."""

    coefficient_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.coefficient_matrix = np.asarray(self.coefficient_matrix, dtype=np.float64)
        self.factorization = lu_factor(self.coefficient_matrix, check_finite=True)

    @property
    def reciprocal_condition(self) -> float:
        return float(1.0 / np.linalg.cond(self.coefficient_matrix))

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        return np.asarray(lu_solve(self.factorization, right_hand_side, check_finite=True), dtype=np.float64)

