from __future__ import annotations

import numpy as np

from ..solvers.linear_solver import LinearSystemSolver
from ..systems.base import ImplicitSystem


def implicit_solution_jvp(
    system: ImplicitSystem,
    state: np.ndarray,
    parameter: np.ndarray,
    direction_std: np.ndarray,
    input_std: np.ndarray,
    state_std: np.ndarray,
    retained_q_indices: np.ndarray | None = None,
    *,
    factorization: LinearSystemSolver | None = None,
) -> np.ndarray:
    """Evaluate ``J_G(x) direction_std`` without forming an inverse."""
    direction_std = np.asarray(direction_std, dtype=np.float64)
    input_std = np.asarray(input_std, dtype=np.float64)
    state_std = np.asarray(state_std, dtype=np.float64)
    if retained_q_indices is None:
        retained_q_indices = np.arange(system.parameter_dimension, dtype=int)
    retained_q_indices = np.asarray(retained_q_indices, dtype=int)
    if direction_std.shape != (retained_q_indices.size,):
        raise ValueError("direction_std shape does not match retained input coordinates")
    physical_direction = np.zeros(system.parameter_dimension, dtype=np.float64)
    physical_direction[retained_q_indices] = input_std[retained_q_indices] * direction_std
    rhs = -np.asarray(system.parameter_jacobian(state, parameter), dtype=np.float64) @ physical_direction
    solver = factorization or LinearSystemSolver(system.state_jacobian(state, parameter))
    state_direction = solver.solve(rhs)
    return state_direction / state_std


def implicit_solution_jacobian(
    system: ImplicitSystem,
    state: np.ndarray,
    parameter: np.ndarray,
    input_std: np.ndarray,
    state_std: np.ndarray,
    retained_q_indices: np.ndarray | None = None,
    *,
    factorization: LinearSystemSolver | None = None,
) -> np.ndarray:
    """Form the standardized implicit Jacobian with one reused factorization.

    This is the exact Frobenius estimator used by the V17.1 sensitivity gate;
    it does not form ``F_z^{-1}`` explicitly.  The right-hand side contains
    all retained input coordinates and is solved in one multi-RHS call.
    """
    input_std = np.asarray(input_std, dtype=np.float64)
    state_std = np.asarray(state_std, dtype=np.float64)
    if retained_q_indices is None:
        retained_q_indices = np.arange(system.parameter_dimension, dtype=int)
    retained_q_indices = np.asarray(retained_q_indices, dtype=int)
    parameter_jacobian = np.asarray(system.parameter_jacobian(state, parameter), dtype=np.float64)
    physical_scale = input_std[retained_q_indices]
    rhs = -parameter_jacobian[:, retained_q_indices] * physical_scale[None, :]
    solver = factorization or LinearSystemSolver(system.state_jacobian(state, parameter))
    state_jacobian = solver.solve(rhs)
    return state_jacobian / state_std[:, None]
