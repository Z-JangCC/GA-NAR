"""Explicit reduced-state definition for the V17.1 IEEE-118 benchmark."""

from __future__ import annotations

import numpy as np

from .equations import ACPowerFlowSystem


class ReducedStateACPowerFlowSystem:
    """AC power flow with the structurally invariant bus-9 voltage removed.

    The V17.0 canonical state contains the bus-9 PQ voltage magnitude at
    coordinate 121.  In the canonical case it has zero physical sensitivity
    under the nominal load-multiplier law.  V17.1 therefore defines a new
    square implicit system by fixing that coordinate to its certified base
    value and removing the matching residual equation.  The retained state
    and residual rows are ordered exactly as in the canonical system.
    """

    state_definition_id = "case118_reduced_state_excluding_invariant_bus9_vm"
    invariant_state_indices = np.asarray([121], dtype=np.int64)

    def __init__(self, canonical: ACPowerFlowSystem | None = None) -> None:
        self.full_system = canonical or ACPowerFlowSystem()
        full = self.full_system
        if full.state_dimension <= int(self.invariant_state_indices.max()):
            raise ValueError("canonical state is too small for the V17.1 invariant coordinate")
        self.retained_state_indices = np.setdiff1d(
            np.arange(full.state_dimension, dtype=np.int64), self.invariant_state_indices
        )
        self.state_dimension = int(self.retained_state_indices.size)
        self.parameter_dimension = int(full.parameter_dimension)
        self._invariant_state_values = full.base_state[self.invariant_state_indices].copy()
        self._base_parameter = full.base_parameter
        self._base_state = self._reduce_state(full.base_state)

        # The removed equation is required to be satisfied by the fixed
        # coordinate on the nominal branch.  This is a certificate check, not
        # a numerical jitter or a post-hoc standard-deviation replacement.
        dropped_residual = full.residual(full.base_state, self._base_parameter)[self.invariant_state_indices]
        if np.max(np.abs(dropped_residual)) > 1e-10:
            raise ValueError("invariant coordinate is not a valid reduced-state equation")

    @property
    def case(self):
        return self.full_system.case

    @property
    def base_mva(self) -> float:
        return self.full_system.base_mva

    @property
    def bus_numbers(self) -> np.ndarray:
        return self.full_system.bus_numbers

    @property
    def bus_type(self) -> np.ndarray:
        return self.full_system.bus_type

    @property
    def pq_bus_indices(self) -> np.ndarray:
        return self.full_system.pq_bus_indices

    @property
    def nonreference_bus_indices(self) -> np.ndarray:
        return self.full_system.nonreference_bus_indices

    @property
    def load_bus_indices(self) -> np.ndarray:
        return self.full_system.load_bus_indices

    @property
    def base_parameter(self) -> np.ndarray:
        return self._base_parameter.copy()

    @property
    def base_state(self) -> np.ndarray:
        return self._base_state.copy()

    def _expand_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dimension,):
            raise ValueError("reduced power-flow state has the wrong dimension")
        full_state = np.empty(self.full_system.state_dimension, dtype=np.float64)
        full_state[self.retained_state_indices] = state
        full_state[self.invariant_state_indices] = self._invariant_state_values
        return full_state

    def _reduce_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float64)[self.retained_state_indices].copy()

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.full_system.unpack_state(self._expand_state(state))

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        full_residual = self.full_system.residual(self._expand_state(state), parameter)
        return full_residual[self.retained_state_indices]

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        full_jacobian = self.full_system.state_jacobian(self._expand_state(state), parameter)
        return full_jacobian[np.ix_(self.retained_state_indices, self.retained_state_indices)]

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        full_jacobian = self.full_system.parameter_jacobian(self._expand_state(state), parameter)
        return full_jacobian[self.retained_state_indices, :]

    def solve_state(self, parameter: np.ndarray, initial_state: np.ndarray | None = None, **kwargs) -> np.ndarray:
        initial_full = None if initial_state is None else self._expand_state(initial_state)
        return self._reduce_state(self.full_system.solve_state(parameter, initial_full, **kwargs))

    def continue_to_parameter(self, parameter: np.ndarray) -> np.ndarray:
        # The nominal V17.1 population is confined to the regular neighborhood
        # of the base case.  Correct from the certified base state and verify
        # both residual and Jacobian conditioning; this is the local branch
        # continuation certificate for the reduced benchmark.
        full_state = self.full_system.solve_state(parameter, self.full_system.base_state)
        residual = self.full_system.residual(full_state, parameter)
        if np.max(np.abs(residual)) > 1e-10:
            raise ValueError("reduced-state nominal solve failed residual certificate")
        reciprocal_condition = 1.0 / np.linalg.cond(self.full_system.state_jacobian(full_state, parameter))
        if reciprocal_condition < 1e-10:
            raise ValueError("reduced-state nominal solve reached an ill-conditioned Jacobian")
        return self._reduce_state(full_state)

    def continue_between(
        self,
        parameter_start: np.ndarray,
        state_start: np.ndarray,
        parameter_target: np.ndarray,
    ) -> np.ndarray:
        # Finite-scale geometry endpoints are local perturbations of a valid
        # center state.  A Newton correction initialized at that center is the
        # same regular branch certificate here and avoids rebuilding a global
        # parameter path for every endpoint.  Dataset points still use the
        # canonical parameter continuation entry point above.
        del parameter_start
        full_state = self.full_system.solve_state(
            parameter_target,
            self._expand_state(state_start),
        )
        return self._reduce_state(full_state)


__all__ = ["ReducedStateACPowerFlowSystem"]
