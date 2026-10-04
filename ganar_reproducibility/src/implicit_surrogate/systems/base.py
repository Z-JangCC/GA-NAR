from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ImplicitSystem(ABC):
    """Interface for a square regular implicit system ``F(z, q) = 0``."""

    state_dimension: int
    parameter_dimension: int

    @abstractmethod
    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        """Return the equation residual in physical coordinates."""

    @abstractmethod
    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        """Return the analytic ``F_z`` matrix."""

    @abstractmethod
    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        """Return the analytic ``F_q`` matrix."""

    @abstractmethod
    def solve_state(
        self,
        parameter: np.ndarray,
        initial_state: np.ndarray | None = None,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 80,
    ) -> np.ndarray:
        """Solve on the system's prescribed branch."""

    def validate_point(self, state: np.ndarray, parameter: np.ndarray) -> tuple[float, float]:
        residual_norm = float(np.max(np.abs(self.residual(state, parameter))))
        jacobian = self.state_jacobian(state, parameter)
        jacobian_rcond = float(1.0 / np.linalg.cond(jacobian))
        return residual_norm, jacobian_rcond

