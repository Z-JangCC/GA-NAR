"""Additional regular equilibrium benchmarks for GA-NRL."""

from __future__ import annotations

import numpy as np

from ..core.random import RandomManager
from ..solvers.nonlinear_solver import NewtonSolver


def _grid_laplacian(side: int) -> np.ndarray:
    size = side * side
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in range(side):
        for col in range(side):
            index = row * side + col
            neighbours = []
            if row > 0: neighbours.append(index - side)
            if row + 1 < side: neighbours.append(index + side)
            if col > 0: neighbours.append(index - 1)
            if col + 1 < side: neighbours.append(index + 1)
            matrix[index, index] = len(neighbours)
            for neighbour in neighbours: matrix[index, neighbour] = -1.0
    return matrix


def _structured_grid_forcing(side: int, count: int, seed: int, amplitude: tuple[float, float]) -> np.ndarray:
    """Sample heterogeneous spatial regimes rather than IID pixel noise.

    The four regimes excite distinct low-frequency/localized spatial patterns;
    this makes sample-to-sample mode allocation a genuine benchmark target.
    """
    axis = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    patterns = np.stack([
        np.sin(np.pi * xx) * np.sin(np.pi * yy),
        np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy),
        np.sin(np.pi * xx) * np.sin(2.0 * np.pi * yy),
        np.exp(-((xx - 0.25) ** 2 + (yy - 0.75) ** 2) / 0.035) - np.exp(-((xx - 0.75) ** 2 + (yy - 0.25) ** 2) / 0.035),
    ], axis=0).reshape(4, -1)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 4, size=count)
    scale = rng.uniform(amplitude[0], amplitude[1], size=count)
    noise = rng.normal(0.0, 0.04, size=(count, side * side))
    return scale[:, None] * patterns[regime] + noise


class _EquilibriumSystem:
    state_dimension: int
    parameter_dimension: int

    def solve_state(self, parameter: np.ndarray, initial_state: np.ndarray | None = None, *, tolerance: float = 1e-10, max_iterations: int = 60) -> np.ndarray:
        initial = np.zeros(self.state_dimension, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64)
        result = NewtonSolver(tolerance=tolerance, max_iterations=max_iterations).solve(lambda state: self.residual(state, parameter), lambda state: self.state_jacobian(state, parameter), initial)
        return result.state

    def solve_states(self, parameters: np.ndarray) -> np.ndarray:
        states = np.empty((len(parameters), self.state_dimension), dtype=np.float64)
        for index, parameter in enumerate(np.asarray(parameters)):
            states[index] = self.solve_state(parameter)
        return states

    def sample_parameters(self, count: int, seed: int = 2025) -> np.ndarray:
        return RandomManager(seed).normal((count, self.parameter_dimension)) * self.parameter_scale


class CoupledDuffingEquilibriumSystem(_EquilibriumSystem):
    """Coupled Duffing equilibrium with spatially heterogeneous cubic stiffness."""

    benchmark_id = "coupled_duffing"

    def __init__(self, side: int = 4, coupling: float = 0.35, linear_stiffness: float = 1.0, cubic_stiffness: float = 0.85) -> None:
        self.side = int(side)
        self.state_dimension = self.parameter_dimension = self.side * self.side
        self.laplacian = _grid_laplacian(self.side)
        self.linear_operator = linear_stiffness * np.eye(self.state_dimension) + coupling * self.laplacian
        coordinates = np.arange(self.state_dimension) % self.side
        self.cubic_coefficient = cubic_stiffness * (1.0 + 0.35 * np.sin(2.0 * np.pi * coordinates / self.side))
        self.parameter_scale = 0.9

    @property
    def base_parameter(self) -> np.ndarray: return np.zeros(self.parameter_dimension)

    @property
    def base_state(self) -> np.ndarray: return np.zeros(self.state_dimension)

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        state, parameter = np.asarray(state), np.asarray(parameter)
        return self.linear_operator @ state + self.cubic_coefficient * state**3 - parameter

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del parameter
        state = np.asarray(state)
        return self.linear_operator + np.diag(3.0 * self.cubic_coefficient * state**2)

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del state, parameter
        return -np.eye(self.parameter_dimension)


class AllenCahn2DEquilibriumSystem(_EquilibriumSystem):
    """Steady 2-D Allen-Cahn reaction-diffusion equilibrium on a square grid."""

    benchmark_id = "allen_cahn_2d"

    def __init__(self, side: int = 8, diffusion: float = 0.16, reaction: float = 0.9, cubic: float = 0.55) -> None:
        self.side = int(side)
        self.state_dimension = self.parameter_dimension = self.side * self.side
        self.laplacian = _grid_laplacian(self.side)
        self.linear_operator = diffusion * self.laplacian + reaction * np.eye(self.state_dimension)
        self.cubic = float(cubic)
        self.parameter_scale = 0.65
        self.population_id = "structured_spatial_regimes_v1"

    def sample_parameters(self, count: int, seed: int = 2025) -> np.ndarray:
        return _structured_grid_forcing(self.side, count, seed, (0.35, 1.25))

    @property
    def base_parameter(self) -> np.ndarray: return np.zeros(self.parameter_dimension)

    @property
    def base_state(self) -> np.ndarray: return np.zeros(self.state_dimension)

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        state, parameter = np.asarray(state), np.asarray(parameter)
        return self.linear_operator @ state + self.cubic * state**3 - parameter

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del parameter
        state = np.asarray(state)
        return self.linear_operator + np.diag(3.0 * self.cubic * state**2)

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del state, parameter
        return -np.eye(self.parameter_dimension)


class ShallowWater2DFreeSurfaceSystem(_EquilibriumSystem):
    """2-D shallow-water free-surface equilibrium with nonlinear pressure."""

    benchmark_id = "shallow_water_2d"

    def __init__(self, side: int = 8, horizontal_diffusion: float = 0.10, restoring: float = 0.65, gravity_nonlinearity: float = 0.72, cubic_correction: float = 0.50, reference_depth: float = 1.0) -> None:
        self.side = int(side)
        self.state_dimension = self.parameter_dimension = self.side * self.side
        self.laplacian = _grid_laplacian(self.side)
        self.linear_operator = horizontal_diffusion * self.laplacian + restoring * np.eye(self.state_dimension)
        self.gravity_nonlinearity = float(gravity_nonlinearity) / float(reference_depth)
        self.cubic_correction = float(cubic_correction)
        self.reference_depth = float(reference_depth)
        self.parameter_scale = 0.7
        self.population_id = "structured_spatial_regimes_v1"

    def sample_parameters(self, count: int, seed: int = 2025) -> np.ndarray:
        return _structured_grid_forcing(self.side, count, seed, (0.30, 1.10))

    @property
    def base_parameter(self) -> np.ndarray: return np.zeros(self.parameter_dimension)

    @property
    def base_state(self) -> np.ndarray: return np.zeros(self.state_dimension)

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        state, parameter = np.asarray(state), np.asarray(parameter)
        # Integrated hydrostatic pressure contributes eta^2; the cubic term
        # regularizes steep free-surface excursions while preserving a unique
        # positive-depth branch in the sampled operating range.
        return self.linear_operator @ state + self.gravity_nonlinearity * state**2 + self.cubic_correction * state**3 - parameter

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del parameter
        state = np.asarray(state)
        return self.linear_operator + np.diag(2.0 * self.gravity_nonlinearity * state + 3.0 * self.cubic_correction * state**2)

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray | None = None) -> np.ndarray:
        del state, parameter
        return -np.eye(self.parameter_dimension)


__all__ = ["CoupledDuffingEquilibriumSystem", "AllenCahn2DEquilibriumSystem", "ShallowWater2DFreeSurfaceSystem"]
