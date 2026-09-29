from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.datatypes import ContinuationSolution
from ..core.exceptions import BranchContinuationError
from .nonlinear_solver import NewtonSolver


class ParameterContinuationSolver:
    """Adaptive tangent-predictor/fixed-parameter-Newton continuation."""

    def __init__(
        self,
        tolerance: float = 1e-10,
        jacobian_rcond_threshold: float = 1e-10,
        initial_step: float = 0.125,
        min_step: float = 1e-4,
        max_step: float = 0.25,
        max_steps: int = 200,
    ) -> None:
        self.tolerance = tolerance
        self.jacobian_rcond_threshold = jacobian_rcond_threshold
        self.initial_step = initial_step
        self.min_step = min_step
        self.max_step = max_step
        self.max_steps = max_steps

    def solve(
        self,
        parameter_start: np.ndarray,
        parameter_target: np.ndarray,
        state_start: np.ndarray,
        residual: Callable[[np.ndarray, np.ndarray], np.ndarray],
        state_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
        parameter_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> ContinuationSolution:
        parameter_start = np.asarray(parameter_start, dtype=np.float64)
        parameter_target = np.asarray(parameter_target, dtype=np.float64)
        state = np.asarray(state_start, dtype=np.float64).copy()
        delta_parameter = parameter_target - parameter_start
        tau = 0.0
        step = min(self.initial_step, 1.0)
        accepted_steps = 0
        while tau < 1.0 - 1e-14:
            trial_step = min(step, 1.0 - tau)
            target_tau = tau + trial_step
            parameter = parameter_start + target_tau * delta_parameter
            current_parameter = parameter_start + tau * delta_parameter
            fz = np.asarray(state_jacobian(state, current_parameter), dtype=np.float64)
            fq = np.asarray(parameter_jacobian(state, current_parameter), dtype=np.float64)
            rcond = float(1.0 / np.linalg.cond(fz))
            if rcond < self.jacobian_rcond_threshold:
                raise BranchContinuationError("continuation reached an ill-conditioned Jacobian")
            tangent = -np.linalg.solve(fz, fq @ delta_parameter)
            predictor = state + trial_step * tangent
            newton = NewtonSolver(tolerance=self.tolerance, max_iterations=60)
            try:
                result = newton.solve(
                    lambda candidate: residual(candidate, parameter),
                    lambda candidate: state_jacobian(candidate, parameter),
                    predictor,
                )
            except BranchContinuationError:
                if trial_step <= self.min_step:
                    raise
                step = trial_step * 0.5
                continue
            new_rcond = float(1.0 / np.linalg.cond(state_jacobian(result.state, parameter)))
            if new_rcond < self.jacobian_rcond_threshold:
                if trial_step <= self.min_step:
                    raise BranchContinuationError("accepted continuation state fails Jacobian validity")
                step = trial_step * 0.5
                continue
            state = result.state
            tau = target_tau
            accepted_steps += 1
            if result.iterations <= 5:
                step = min(self.max_step, trial_step * 1.5)
            else:
                step = trial_step
        parameter = parameter_target.copy()
        residual_norm = float(np.max(np.abs(residual(state, parameter))))
        jacobian_rcond = float(1.0 / np.linalg.cond(state_jacobian(state, parameter)))
        return ContinuationSolution(
            parameter=parameter,
            state=state,
            residual_norm=residual_norm,
            jacobian_rcond=jacobian_rcond,
            success=residual_norm <= self.tolerance and jacobian_rcond >= self.jacobian_rcond_threshold,
            continuation_parameter=1.0,
            steps=accepted_steps,
            message="converged",
        )

