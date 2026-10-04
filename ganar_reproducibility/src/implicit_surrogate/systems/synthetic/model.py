from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from ..base import ImplicitSystem
from ...core.random import RandomManager


class SyntheticImplicitSystem(ImplicitSystem):
    """The frozen ``z + kappa C.T tanh(C z) - q = 0`` benchmark."""

    def __init__(
        self,
        state_dimension: int = 50,
        parameter_dimension: int = 50,
        nonlinear_rank: int = 5,
        kappa: float = 0.8,
        projection_matrix: np.ndarray | None = None,
        seed: int = 2025,
    ) -> None:
        if state_dimension != parameter_dimension:
            raise ValueError("the frozen synthetic system is square with d=m")
        if nonlinear_rank > parameter_dimension:
            raise ValueError("nonlinear rank exceeds parameter dimension")
        self.state_dimension = int(state_dimension)
        self.parameter_dimension = int(parameter_dimension)
        self.nonlinear_rank = int(nonlinear_rank)
        self.kappa = float(kappa)
        if self.kappa >= 1.0:
            raise ValueError("the frozen contraction requires kappa < 1")
        if projection_matrix is None:
            raw = RandomManager(seed).normal((nonlinear_rank, parameter_dimension))
            q, _ = np.linalg.qr(raw.T, mode="reduced")
            projection_matrix = q.T
        projection_matrix = np.asarray(projection_matrix, dtype=np.float64)
        if projection_matrix.shape != (nonlinear_rank, parameter_dimension):
            raise ValueError("projection_matrix has the wrong shape")
        if not np.allclose(projection_matrix @ projection_matrix.T, np.eye(nonlinear_rank), atol=1e-12):
            raise ValueError("projection_matrix rows must be orthonormal")
        self.nonlinear_projection_matrix = projection_matrix

    @property
    def base_parameter(self) -> np.ndarray:
        return np.zeros(self.parameter_dimension, dtype=np.float64)

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        parameter = np.asarray(parameter, dtype=np.float64)
        c = self.nonlinear_projection_matrix
        return state + self.kappa * c.T @ np.tanh(c @ state) - parameter

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del parameter
        state = np.asarray(state, dtype=np.float64)
        c = self.nonlinear_projection_matrix
        projected = c @ state
        sech2 = 1.0 / np.cosh(projected) ** 2
        return np.eye(self.state_dimension) + self.kappa * c.T @ (sech2[:, None] * c)

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del state, parameter
        return -np.eye(self.parameter_dimension, dtype=np.float64)

    def solve_state(
        self,
        parameter: np.ndarray,
        initial_state: np.ndarray | None = None,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 80,
    ) -> np.ndarray:
        parameter = np.asarray(parameter, dtype=np.float64)
        if parameter.shape != (self.parameter_dimension,):
            raise ValueError("parameter has the wrong shape")
        state = parameter.copy() if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
        for _ in range(max_iterations):
            residual = self.residual(state, parameter)
            if np.max(np.abs(residual)) <= tolerance:
                return state
            state -= solve(self.state_jacobian(state, parameter), residual, assume_a="pos", check_finite=True)
        raise RuntimeError("synthetic Newton solve did not converge")

    def solve_states(self, parameters: np.ndarray, *, tolerance: float = 1e-12, max_iterations: int = 32) -> np.ndarray:
        """Solve many independent synthetic states in one vectorized pass."""
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.ndim != 2 or parameters.shape[1] != self.parameter_dimension:
            raise ValueError("parameters must have shape (batch, parameter_dimension)")
        projected_parameter = parameters @ self.nonlinear_projection_matrix.T
        projected_state = projected_parameter.copy()
        for _ in range(max_iterations):
            residual = projected_state + self.kappa * np.tanh(projected_state) - projected_parameter
            if np.max(np.abs(residual)) <= tolerance:
                break
            projected_state -= residual / (1.0 + self.kappa / np.cosh(projected_state) ** 2)
        else:
            raise RuntimeError("batched synthetic Newton solve did not converge")
        return parameters - self.kappa * (np.tanh(projected_state) @ self.nonlinear_projection_matrix)

    def sample_parameters(self, count: int, seed: int) -> np.ndarray:
        return RandomManager(seed).normal((count, self.parameter_dimension))

    def standardized_nonlinear_input_matrix(self, q_std: np.ndarray) -> np.ndarray:
        return self.nonlinear_projection_matrix @ np.diag(np.asarray(q_std, dtype=np.float64))

    def true_nonlinear_input_subspace(self, q_std: np.ndarray) -> np.ndarray:
        matrix = self.standardized_nonlinear_input_matrix(q_std)
        _, _, vh = np.linalg.svd(matrix, full_matrices=True)
        return vh[: self.nonlinear_rank].T

    def batch_directional_variation(
        self,
        centers_std: np.ndarray,
        center_parameters: np.ndarray,
        input_std: np.ndarray,
        state_std: np.ndarray,
        retained_q_indices: np.ndarray,
        unit_directions: np.ndarray,
        scales: tuple[float, ...],
        *,
        chunk_size: int = 65536,
    ) -> tuple[dict[float, np.ndarray], np.ndarray]:
        """Vectorized exact Frobenius energies for the V17.1 estimator.

        The synthetic equation separates into five scalar monotone equations
        in the nonlinear coordinates.  Solving those coordinates in batches
        and applying the rank-five Woodbury form of ``F_z^{-1}`` gives exactly
        the same standardized Jacobian as the generic LU path, without a
        Python loop over hundreds of thousands of geometry pairs.
        """
        if centers_std.shape[0] * unit_directions.shape[1] == 0:
            raise ValueError("empty synthetic geometry population")
        centers_std = np.asarray(centers_std, dtype=np.float64)
        center_parameters = np.asarray(center_parameters, dtype=np.float64)
        unit_directions = np.asarray(unit_directions, dtype=np.float64)
        retained_q_indices = np.asarray(retained_q_indices, dtype=np.int64)
        if unit_directions.shape[0] != centers_std.shape[0]:
            raise ValueError("center and direction populations have different sizes")
        directions_per_center = unit_directions.shape[1]
        flat_parameters = np.repeat(center_parameters, directions_per_center, axis=0)
        flat_directions = unit_directions.reshape(-1, unit_directions.shape[-1])
        input_scale = np.asarray(input_std, dtype=np.float64)[retained_q_indices]
        output_scale = np.asarray(state_std, dtype=np.float64)
        energies = {float(scale): np.empty(flat_directions.shape[0], dtype=np.float64) for scale in scales}
        c = self.nonlinear_projection_matrix

        def solve_batch(parameters: np.ndarray) -> np.ndarray:
            projected_parameter = parameters @ c.T
            projected_state = projected_parameter.copy()
            for _ in range(14):
                residual = projected_state + self.kappa * np.tanh(projected_state) - projected_parameter
                if np.max(np.abs(residual)) <= 1e-13:
                    break
                projected_state -= residual / (1.0 + self.kappa / np.cosh(projected_state) ** 2)
            return parameters - self.kappa * (np.tanh(projected_state) @ c)

        left_metric = (c / output_scale[None, :]) @ (c / output_scale[None, :]).T
        right_metric = (c * input_scale[None, :]) @ (c * input_scale[None, :]).T

        def nonlinear_coefficients(states: np.ndarray) -> np.ndarray:
            projected_state = states @ c.T
            return 1.0 / (1.0 + self.kappa / np.cosh(projected_state) ** 2) - 1.0

        for start in range(0, flat_directions.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_directions.shape[0])
            center_parameter_chunk = flat_parameters[start:stop]
            direction_chunk = flat_directions[start:stop]
            for scale in scales:
                delta = float(scale) * direction_chunk
                plus_parameters = center_parameter_chunk.copy()
                minus_parameters = center_parameter_chunk.copy()
                plus_parameters[:, retained_q_indices] += input_scale[None, :] * delta
                minus_parameters[:, retained_q_indices] -= input_scale[None, :] * delta
                plus_states = solve_batch(plus_parameters)
                minus_states = solve_batch(minus_parameters)
                coefficient_difference = nonlinear_coefficients(plus_states) - nonlinear_coefficients(minus_states)
                gram = left_metric[None, :, :] * right_metric[None, :, :]
                energies[float(scale)][start:stop] = np.einsum("na,nab,nb->n", coefficient_difference, gram, coefficient_difference) / (2.0 * float(scale)) ** 2
        return energies, np.ones(flat_directions.shape[0], dtype=bool)
