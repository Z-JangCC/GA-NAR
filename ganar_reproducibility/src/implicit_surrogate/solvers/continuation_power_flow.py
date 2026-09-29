from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.exceptions import BranchContinuationError


@dataclass(frozen=True)
class CPFFoldResult:
    loading_direction: np.ndarray
    critical_loading_parameter: float | None
    fold_state: np.ndarray | None
    success: bool
    steps: int
    message: str


class ContinuationPowerFlowSolver:
    """Pseudo-arclength CPF with explicit fold detection."""

    def __init__(self, system, *, step_size: float = 0.05, max_steps: int = 500, max_loading_parameter: float = 20.0, tolerance: float = 1e-10) -> None:
        self.system = system
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.max_loading_parameter = float(max_loading_parameter)
        self.tolerance = float(tolerance)

    def _tangent(self, state: np.ndarray, loading_parameter: float, direction: np.ndarray) -> np.ndarray:
        parameter = self.system.base_parameter + loading_parameter * direction
        fz = self.system.state_jacobian(state, parameter)
        flambda = self.system.parameter_jacobian(state, parameter) @ direction
        state_tangent = -np.linalg.solve(fz, flambda)
        tangent = np.concatenate([state_tangent, np.ones(1)])
        tangent /= np.linalg.norm(tangent)
        return tangent

    def _corrector(self, previous: np.ndarray, tangent: np.ndarray, direction: np.ndarray) -> np.ndarray:
        system = self.system
        state = previous[:-1].copy()
        loading_parameter = float(previous[-1])
        for _ in range(80):
            parameter = system.base_parameter + loading_parameter * direction
            values = system.residual(state, parameter)
            arc = float(np.dot(tangent, np.concatenate([state, [loading_parameter]]) - previous) - self.step_size)
            residual = np.concatenate([values, [arc]])
            if np.max(np.abs(residual)) <= self.tolerance:
                return np.concatenate([state, [loading_parameter]])
            fz = system.state_jacobian(state, parameter)
            flambda = system.parameter_jacobian(state, parameter) @ direction
            augmented = np.block([[fz, flambda[:, None]], [tangent[None, :]]])
            try:
                delta = np.linalg.solve(augmented, residual)
            except np.linalg.LinAlgError as exc:
                raise BranchContinuationError("CPF augmented Jacobian is singular") from exc
            step = 1.0
            norm_before = np.linalg.norm(residual, ord=np.inf)
            accepted = False
            while step >= 1e-8:
                candidate_state = state - step * delta[:-1]
                candidate_lambda = loading_parameter - step * delta[-1]
                candidate_parameter = system.base_parameter + candidate_lambda * direction
                candidate_arc = float(np.dot(tangent, np.concatenate([candidate_state, [candidate_lambda]]) - previous) - self.step_size)
                candidate_residual = np.concatenate([system.residual(candidate_state, candidate_parameter), [candidate_arc]])
                if np.all(np.isfinite(candidate_residual)) and np.linalg.norm(candidate_residual, ord=np.inf) < norm_before:
                    state, loading_parameter = candidate_state, candidate_lambda
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                raise BranchContinuationError("CPF corrector failed")
        raise BranchContinuationError("CPF corrector reached max_iterations")

    def identify_fold(self, loading_direction: np.ndarray) -> CPFFoldResult:
        direction = np.asarray(loading_direction, dtype=np.float64)
        direction = np.abs(direction)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("CPF loading direction must be nonzero")
        direction = direction / norm
        state = self.system.base_state
        previous = np.concatenate([state, [0.0]])
        tangent = self._tangent(state, 0.0, direction)
        if tangent[-1] < 0:
            tangent = -tangent
        previous_lambda_tangent = tangent[-1]
        for step_index in range(1, self.max_steps + 1):
            try:
                current = self._corrector(previous, tangent, direction)
            except BranchContinuationError as exc:
                return CPFFoldResult(direction, None, None, False, step_index, str(exc))
            current_state = current[:-1]
            current_lambda = float(current[-1])
            if not np.isfinite(current_lambda) or current_lambda < -1e-6:
                return CPFFoldResult(direction, None, None, False, step_index, "CPF left the admissible loading path")
            current_tangent = self._tangent(current_state, current_lambda, direction)
            if np.dot(current_tangent, tangent) < 0:
                current_tangent = -current_tangent
            if previous_lambda_tangent > 0 and current_tangent[-1] <= 0:
                # The turning point has been crossed; this is the fold certificate.
                return CPFFoldResult(direction, current_lambda, current_state, True, step_index, "fold identified by tangent sign change")
            if current_lambda >= self.max_loading_parameter:
                return CPFFoldResult(direction, None, None, False, step_index, "CPF reached loading cap before a fold")
            previous, tangent, previous_lambda_tangent = current, current_tangent, current_tangent[-1]
        return CPFFoldResult(direction, None, None, False, self.max_steps, "CPF reached max_steps before a fold")

