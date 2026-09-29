"""Richer IEEE-118 operating-parameter benchmark for GA-NRL."""

from __future__ import annotations

import numpy as np

from ...core.random import RandomManager
from ...solvers.nonlinear_solver import NewtonSolver
from .equations import ACPowerFlowSystem
from .reduced_state import ReducedStateACPowerFlowSystem


class RichIEEE118PowerFlowSystem:
    """IEEE-118 with spatial load, generator redispatch, and voltage controls.

    The historical benchmark varied only 99 independent load multipliers. This
    benchmark exposes a richer operating parameter vector:

    ``q = [zone_load_multipliers(12), generator_redispatch(8),
    generator_voltage_shifts(8)]``.

    Bus types and the reduced 180-dimensional state remain fixed. The extra
    controls alter the physical operating point and therefore the Jacobian
    geometry; they are not artificial output noise or a neural feature.
    """

    state_definition_id = "case118_rich_operating_controls_reduced_state"

    def __init__(self, canonical: ACPowerFlowSystem | None = None, *, num_zones: int = 12, num_dispatch: int = 8, num_voltage_controls: int = 8, num_shunt_controls: int = 8) -> None:
        self.full_system = canonical or ACPowerFlowSystem()
        self.reduced = ReducedStateACPowerFlowSystem(self.full_system)
        self.case = self.full_system.case
        self.num_zones = int(num_zones)
        self.num_dispatch = int(num_dispatch)
        self.num_voltage_controls = int(num_voltage_controls)
        self.num_shunt_controls = int(num_shunt_controls)
        active_generators = np.flatnonzero(self.full_system.bus_type >= 2)
        if len(active_generators) < self.num_dispatch + self.num_voltage_controls:
            raise ValueError("case118 has too few controllable generators")
        self.dispatch_bus_indices = active_generators[:self.num_dispatch]
        self.voltage_bus_indices = active_generators[:self.num_voltage_controls]
        self.shunt_bus_indices = self.full_system.pq_bus_indices[:self.num_shunt_controls]
        load_buses = self.full_system.load_bus_indices
        self.load_zone = np.arange(len(load_buses), dtype=int) % self.num_zones
        self.bus_zone = np.full(self.full_system.num_buses, -1, dtype=int)
        self.bus_zone[load_buses] = self.load_zone
        self.load_bus_indices = load_buses
        self.zip_fraction_p = 0.15 + 0.45 * (np.arange(self.num_zones) % 5) / 4.0
        self.zip_fraction_q = 0.10 + 0.35 * ((2 * np.arange(self.num_zones) + 1) % 5) / 4.0
        self.state_dimension = self.reduced.state_dimension
        self.parameter_dimension = self.num_zones + self.num_dispatch + self.num_voltage_controls + self.num_shunt_controls
        self._base_parameter = np.concatenate([np.ones(self.num_zones), np.zeros(self.num_dispatch + self.num_voltage_controls + self.num_shunt_controls)])
        self._dispatch_scale = np.maximum(np.abs(self.full_system.base_p_generation[self.dispatch_bus_indices]), 0.2)
        self._base_state = self.solve_state(self._base_parameter, self.reduced.base_state)

    @property
    def base_parameter(self) -> np.ndarray:
        return self._base_parameter.copy()

    @property
    def base_state(self) -> np.ndarray:
        return self._base_state.copy()

    @property
    def base_mva(self) -> float:
        return self.full_system.base_mva

    @property
    def bus_numbers(self) -> np.ndarray:
        return self.full_system.bus_numbers

    @property
    def bus_type(self) -> np.ndarray:
        return self.full_system.bus_type

    def _unpack(self, state: np.ndarray, parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_state = self.reduced._expand_state(state)
        angles, magnitudes = self.full_system.unpack_state(full_state)
        voltage_start = self.num_zones + self.num_dispatch
        voltage_shift = np.asarray(parameter)[voltage_start:voltage_start + self.num_voltage_controls]
        shunt = np.asarray(parameter)[voltage_start + self.num_voltage_controls:]
        magnitudes = magnitudes.copy()
        magnitudes[self.voltage_bus_indices] += voltage_shift
        if np.any(magnitudes <= 0):
            raise ValueError("rich IEEE-118 voltage magnitude became non-positive")
        return angles, magnitudes, shunt

    def _specification(self, parameter: np.ndarray, magnitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parameter = np.asarray(parameter, dtype=np.float64)
        if parameter.shape != (self.parameter_dimension,):
            raise ValueError("rich IEEE-118 parameter has the wrong shape")
        zone_scale = parameter[:self.num_zones]
        dispatch = parameter[self.num_zones:self.num_zones + self.num_dispatch]
        p_spec = self.full_system.base_p_generation - self.full_system.base_p_demand
        q_spec = self.full_system.base_q_generation - self.full_system.base_q_demand
        p_spec = p_spec.copy(); q_spec = q_spec.copy()
        for bus, zone in zip(self.load_bus_indices, self.load_zone):
            vm_factor_p = 1.0 + self.zip_fraction_p[zone] * (magnitudes[bus] - 1.0)
            vm_factor_q = 1.0 + self.zip_fraction_q[zone] * (magnitudes[bus] - 1.0)
            p_spec[bus] = self.full_system.base_p_generation[bus] - self.full_system.base_p_demand[bus] * zone_scale[zone] * vm_factor_p
            q_spec[bus] = self.full_system.base_q_generation[bus] - self.full_system.base_q_demand[bus] * zone_scale[zone] * vm_factor_q
        for bus, delta, scale in zip(self.dispatch_bus_indices, dispatch, self._dispatch_scale):
            if self.bus_zone[bus] >= 0:
                dispatch_zone = self.bus_zone[bus]
                demand = self.full_system.base_p_demand[bus] * zone_scale[dispatch_zone] * (1.0 + self.zip_fraction_p[dispatch_zone] * (magnitudes[bus] - 1.0))
            else:
                demand = self.full_system.base_p_demand[bus]
            p_spec[bus] = self.full_system.base_p_generation[bus] + scale * delta - demand
        return p_spec, q_spec

    def _injection_derivatives(self, angles: np.ndarray, magnitudes: np.ndarray, shunt: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_bus = self.full_system.y_bus.copy()
        if shunt is not None:
            for bus, control in zip(self.shunt_bus_indices, shunt):
                y_bus[bus, bus] += 1j * control
        g_bus, b_bus = y_bus.real, y_bus.imag
        voltage = magnitudes * np.exp(1j * angles)
        injections = voltage * np.conjugate(y_bus @ voltage)
        p_values, q_values = injections.real, injections.imag
        difference = angles[:, None] - angles[None, :]
        cosine, sine = np.cos(difference), np.sin(difference)
        product = magnitudes[:, None] * magnitudes[None, :]
        all_angle = product * (g_bus * sine - b_bus * cosine)
        all_vm = magnitudes[:, None] * (g_bus * cosine + b_bus * sine)
        all_q_angle = -product * (g_bus * cosine + b_bus * sine)
        all_q_vm = magnitudes[:, None] * (g_bus * sine - b_bus * cosine)
        diagonal = np.diag_indices(self.full_system.num_buses)
        all_angle[diagonal] = -q_values - np.diag(b_bus) * magnitudes**2
        all_vm[diagonal] = p_values / magnitudes + np.diag(g_bus) * magnitudes
        all_q_angle[diagonal] = p_values - np.diag(g_bus) * magnitudes**2
        all_q_vm[diagonal] = q_values / magnitudes - np.diag(b_bus) * magnitudes
        return p_values, q_values, all_angle, all_vm, all_q_angle, all_q_vm

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        angles, magnitudes, shunt = self._unpack(state, parameter)
        p_spec, q_spec = self._specification(parameter, magnitudes)
        voltage = magnitudes * np.exp(1j * angles)
        y_voltage = self.full_system.y_bus @ voltage
        for bus, control in zip(self.shunt_bus_indices, shunt):
            y_voltage[bus] += 1j * control * voltage[bus]
        injections = voltage * np.conjugate(y_voltage)
        residual = np.concatenate([(injections.real - p_spec)[self.full_system.nonreference_bus_indices], (injections.imag - q_spec)[self.full_system.pq_bus_indices]])
        return residual[self.reduced.retained_state_indices]

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        angles, magnitudes, shunt = self._unpack(state, parameter)
        _, _, all_angle, all_vm, all_q_angle, all_q_vm = self._injection_derivatives(angles, magnitudes, shunt)
        p_rows = self.full_system.nonreference_bus_indices
        q_rows = self.full_system.pq_bus_indices
        theta_columns, vm_columns = self.full_system.nonreference_bus_indices, self.full_system.pq_bus_indices
        full = np.concatenate([
            np.concatenate([all_angle[np.ix_(p_rows, theta_columns)], all_vm[np.ix_(p_rows, vm_columns)]], axis=1),
            np.concatenate([all_q_angle[np.ix_(q_rows, theta_columns)], all_q_vm[np.ix_(q_rows, vm_columns)]], axis=1),
        ], axis=0)
        # ZIP load voltage dependence contributes to F_z on PQ voltage
        # coordinates. This is the key extra physical nonlinearity absent from
        # the historical constant-power-only benchmark.
        zone_scale = np.asarray(parameter)[:self.num_zones]
        for bus in self.load_bus_indices:
            zone = self.bus_zone[bus]
            vm_col = len(p_rows) + int(np.flatnonzero(self.full_system.pq_bus_indices == bus)[0]) if np.any(self.full_system.pq_bus_indices == bus) else None
            if vm_col is not None:
                p_row = np.flatnonzero(p_rows == bus)
                if p_row.size:
                    full[int(p_row[0]), vm_col] += self.full_system.base_p_demand[bus] * zone_scale[zone] * self.zip_fraction_p[zone]
                full[len(p_rows) + int(np.flatnonzero(self.full_system.pq_bus_indices == bus)[0]), vm_col] += self.full_system.base_q_demand[bus] * zone_scale[zone] * self.zip_fraction_q[zone]
        return full[np.ix_(self.reduced.retained_state_indices, self.reduced.retained_state_indices)]

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        angles, magnitudes, _ = self._unpack(state, parameter)
        _, _, _, all_vm, _, all_q_vm = self._injection_derivatives(angles, magnitudes, np.asarray(parameter)[self.num_zones + self.num_dispatch + self.num_voltage_controls:])
        p_rows, q_rows = self.full_system.nonreference_bus_indices, self.full_system.pq_bus_indices
        full = np.zeros((self.full_system.state_dimension, self.parameter_dimension), dtype=np.float64)
        angles, magnitudes, _ = self._unpack(state, parameter)
        for bus, zone in zip(self.load_bus_indices, self.load_zone):
            column = zone
            p_row = np.flatnonzero(p_rows == bus)
            q_row = np.flatnonzero(q_rows == bus)
            p_factor = 1.0 + self.zip_fraction_p[zone] * (magnitudes[bus] - 1.0)
            q_factor = 1.0 + self.zip_fraction_q[zone] * (magnitudes[bus] - 1.0)
            if p_row.size: full[p_row[0], column] = self.full_system.base_p_demand[bus] * p_factor
            if q_row.size: full[len(p_rows) + q_row[0], column] = self.full_system.base_q_demand[bus] * q_factor
        for index, (bus, scale) in enumerate(zip(self.dispatch_bus_indices, self._dispatch_scale)):
            p_row = np.flatnonzero(p_rows == bus)
            if p_row.size: full[p_row[0], self.num_zones + index] = -scale
        for index, bus in enumerate(self.voltage_bus_indices):
            column = self.num_zones + self.num_dispatch + index
            full[:len(p_rows), column] = all_vm[p_rows, bus]
            full[len(p_rows):, column] = all_q_vm[q_rows, bus]
            if self.bus_zone[bus] >= 0:
                zone = self.bus_zone[bus]
                p_row = np.flatnonzero(p_rows == bus)
                q_row = np.flatnonzero(q_rows == bus)
                zone_scale = np.asarray(parameter)[zone]
                if p_row.size:
                    full[p_row[0], column] += self.full_system.base_p_demand[bus] * zone_scale * self.zip_fraction_p[zone]
                if q_row.size:
                    full[len(p_rows) + q_row[0], column] += self.full_system.base_q_demand[bus] * zone_scale * self.zip_fraction_q[zone]
        for index, bus in enumerate(self.shunt_bus_indices):
            column = self.num_zones + self.num_dispatch + self.num_voltage_controls + index
            q_row = np.flatnonzero(q_rows == bus)
            if q_row.size:
                full[len(p_rows) + q_row[0], column] = -magnitudes[bus] ** 2
        return full[self.reduced.retained_state_indices]

    def solve_state(self, parameter: np.ndarray, initial_state: np.ndarray | None = None, *, tolerance: float = 1e-10, max_iterations: int = 50) -> np.ndarray:
        initial = self.base_state if initial_state is None and hasattr(self, "_base_state") else (self.reduced.base_state if initial_state is None else initial_state)
        result = NewtonSolver(tolerance=tolerance, max_iterations=max_iterations).solve(lambda state: self.residual(state, parameter), lambda state: self.state_jacobian(state, parameter), initial)
        return result.state

    def continue_to_parameter(self, parameter: np.ndarray) -> np.ndarray:
        return self.solve_state(parameter, self.base_state)

    def continue_between(self, parameter_start: np.ndarray, state_start: np.ndarray, parameter_target: np.ndarray) -> np.ndarray:
        del parameter_start
        return self.solve_state(parameter_target, state_start)

    def sample_parameters(self, count: int, seed: int = 4321) -> np.ndarray:
        rng = RandomManager(seed)
        values = np.repeat(self.base_parameter[None, :], count, axis=0)
        values[:, :self.num_zones] = rng.uniform(0.55, 1.45, (count, self.num_zones))
        values[:, self.num_zones:self.num_zones + self.num_dispatch] = rng.normal((count, self.num_dispatch)) * 0.45
        voltage_start = self.num_zones + self.num_dispatch
        values[:, voltage_start:voltage_start + self.num_voltage_controls] = rng.normal((count, self.num_voltage_controls)) * 0.10
        values[:, voltage_start + self.num_voltage_controls:] = rng.normal((count, self.num_shunt_controls)) * 0.45
        return values

    def solve_states(self, parameters: np.ndarray) -> np.ndarray:
        states = np.empty((len(parameters), self.state_dimension), dtype=np.float64)
        initial = self.base_state
        for index, parameter in enumerate(np.asarray(parameters)):
            states[index] = self.solve_state(parameter, initial)
            initial = states[index]
        return states


__all__ = ["RichIEEE118PowerFlowSystem"]
