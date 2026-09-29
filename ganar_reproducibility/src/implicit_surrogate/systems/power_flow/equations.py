from __future__ import annotations

import numpy as np

from ...core.exceptions import BranchContinuationError
from ...solvers.nonlinear_solver import NewtonSolver
from ...solvers.parameter_continuation import ParameterContinuationSolver
from .case_data import PowerFlowCase, load_case118


class ACPowerFlowSystem:
    """Canonical fixed-bus-type AC power-flow solution map."""

    def __init__(self, case: PowerFlowCase | None = None) -> None:
        self.case = case or load_case118()
        self.base_mva = self.case.base_mva
        bus = self.case.bus
        self.num_buses = bus.shape[0]
        self.bus_numbers = bus[:, 0].astype(int)
        self.bus_type = bus[:, 1].astype(int)
        self.reference_bus_indices = np.flatnonzero(self.bus_type == 3)
        self.pv_bus_indices = np.flatnonzero(self.bus_type == 2)
        self.pq_bus_indices = np.flatnonzero(self.bus_type == 1)
        self.nonreference_bus_indices = np.flatnonzero(np.arange(self.num_buses) != self.reference_bus_indices[0])
        self.load_bus_indices = np.flatnonzero((np.abs(bus[:, 2]) > 0) | (np.abs(bus[:, 3]) > 0))
        self.state_dimension = len(self.nonreference_bus_indices) + len(self.pq_bus_indices)
        self.parameter_dimension = len(self.load_bus_indices)
        self._build_network_admittance()
        self._build_generator_injections()
        self._base_parameter = np.ones(self.parameter_dimension, dtype=np.float64)
        self._base_state = self._initial_state_from_case()
        # Verify and refine the canonical base point once during construction.
        self._base_state = self.solve_state(self._base_parameter, self._base_state)

    def _build_network_admittance(self) -> None:
        bus = self.case.bus
        branch = self.case.branch
        y_bus = np.zeros((self.num_buses, self.num_buses), dtype=np.complex128)
        for row in branch:
            if int(row[10]) == 0:
                continue
            from_bus = int(row[0]) - 1
            to_bus = int(row[1]) - 1
            series = 1.0 / complex(row[2], row[3])
            charging = 1j * row[4] / 2.0
            tap_ratio = float(row[8]) if abs(row[8]) > 1e-15 else 1.0
            tap = tap_ratio * np.exp(1j * np.deg2rad(row[9]))
            y_ff = (series + charging) / (tap * np.conjugate(tap))
            y_ft = -series / np.conjugate(tap)
            y_tf = -series / tap
            y_tt = series + charging
            y_bus[from_bus, from_bus] += y_ff
            y_bus[from_bus, to_bus] += y_ft
            y_bus[to_bus, from_bus] += y_tf
            y_bus[to_bus, to_bus] += y_tt
        # GS/BS are specified in MW/MVAr at V=1 p.u.
        y_bus[np.diag_indices(self.num_buses)] += (bus[:, 4] + 1j * bus[:, 5]) / self.base_mva
        self.y_bus = y_bus
        self.g_bus = y_bus.real
        self.b_bus = y_bus.imag

    def _build_generator_injections(self) -> None:
        generator = self.case.generator
        pgen = np.zeros(self.num_buses, dtype=np.float64)
        qgen = np.zeros(self.num_buses, dtype=np.float64)
        for row in generator:
            if int(row[7]) == 0:
                continue
            index = int(row[0]) - 1
            pgen[index] += row[1] / self.base_mva
            qgen[index] += row[2] / self.base_mva
        self.base_p_generation = pgen
        self.base_q_generation = qgen
        self.base_p_demand = self.case.bus[:, 2] / self.base_mva
        self.base_q_demand = self.case.bus[:, 3] / self.base_mva
        # MATPOWER/PYPOWER defines PV and reference voltage setpoints in the
        # generator ``VG`` column.  Bus ``VM`` is only an initial value and can
        # differ for a subset of the canonical case118 generators.
        self.fixed_voltage_magnitude = self.case.bus[:, 7].copy()
        for row in self.case.generator:
            if int(row[7]) == 0:
                continue
            self.fixed_voltage_magnitude[int(row[0]) - 1] = float(row[5])

    def _initial_state_from_case(self) -> np.ndarray:
        angle = np.deg2rad(self.case.bus[:, 8])
        voltage = self.fixed_voltage_magnitude.copy()
        return np.concatenate([angle[self.nonreference_bus_indices], voltage[self.pq_bus_indices]])

    @property
    def base_parameter(self) -> np.ndarray:
        return self._base_parameter.copy()

    @property
    def base_state(self) -> np.ndarray:
        return self._base_state.copy()

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.state_dimension,):
            raise ValueError("power-flow state has the wrong dimension")
        angles = np.deg2rad(self.case.bus[:, 8]).copy()
        magnitudes = self.fixed_voltage_magnitude.copy()
        angles[self.nonreference_bus_indices] = state[: len(self.nonreference_bus_indices)]
        magnitudes[self.pq_bus_indices] = state[len(self.nonreference_bus_indices) :]
        return angles, magnitudes

    def residual(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        angles, magnitudes = self.unpack_state(state)
        parameter = np.asarray(parameter, dtype=np.float64)
        if parameter.shape != (self.parameter_dimension,):
            raise ValueError("load_multipliers has the wrong dimension")
        complex_voltage = magnitudes * np.exp(1j * angles)
        injections = complex_voltage * np.conjugate(self.y_bus @ complex_voltage)
        p_spec = self.base_p_generation - self.base_p_demand
        q_spec = self.base_q_generation - self.base_q_demand
        p_spec = p_spec.copy()
        q_spec = q_spec.copy()
        p_spec[self.load_bus_indices] = self.base_p_generation[self.load_bus_indices] - self.base_p_demand[self.load_bus_indices] * parameter
        q_spec[self.load_bus_indices] = self.base_q_generation[self.load_bus_indices] - self.base_q_demand[self.load_bus_indices] * parameter
        p_mismatch = injections.real - p_spec
        q_mismatch = injections.imag - q_spec
        return np.concatenate([p_mismatch[self.nonreference_bus_indices], q_mismatch[self.pq_bus_indices]])

    def state_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        angles, magnitudes = self.unpack_state(state)
        complex_voltage = magnitudes * np.exp(1j * angles)
        injections = complex_voltage * np.conjugate(self.y_bus @ complex_voltage)
        p_values = injections.real
        q_values = injections.imag
        angle_difference = angles[:, None] - angles[None, :]
        cosine = np.cos(angle_difference)
        sine = np.sin(angle_difference)
        voltage_product = magnitudes[:, None] * magnitudes[None, :]
        all_angle = voltage_product * (self.g_bus * sine - self.b_bus * cosine)
        all_vm = magnitudes[:, None] * (self.g_bus * cosine + self.b_bus * sine)
        all_q_angle = -voltage_product * (self.g_bus * cosine + self.b_bus * sine)
        all_q_vm = magnitudes[:, None] * (self.g_bus * sine - self.b_bus * cosine)
        diagonal = np.diag_indices(self.num_buses)
        all_angle[diagonal] = -q_values - np.diag(self.b_bus) * magnitudes**2
        all_vm[diagonal] = p_values / magnitudes + np.diag(self.g_bus) * magnitudes
        all_q_angle[diagonal] = p_values - np.diag(self.g_bus) * magnitudes**2
        all_q_vm[diagonal] = q_values / magnitudes - np.diag(self.b_bus) * magnitudes
        p_rows = self.nonreference_bus_indices
        q_rows = self.pq_bus_indices
        theta_columns = self.nonreference_bus_indices
        vm_columns = self.pq_bus_indices
        top = np.concatenate([all_angle[np.ix_(p_rows, theta_columns)], all_vm[np.ix_(p_rows, vm_columns)]], axis=1)
        bottom = np.concatenate([all_q_angle[np.ix_(q_rows, theta_columns)], all_q_vm[np.ix_(q_rows, vm_columns)]], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def parameter_jacobian(self, state: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        del state, parameter
        matrix = np.zeros((self.state_dimension, self.parameter_dimension), dtype=np.float64)
        p_row_lookup = {int(bus): row for row, bus in enumerate(self.nonreference_bus_indices)}
        q_row_lookup = {int(bus): len(self.nonreference_bus_indices) + row for row, bus in enumerate(self.pq_bus_indices)}
        for column, bus in enumerate(self.load_bus_indices):
            if int(bus) in p_row_lookup:
                matrix[p_row_lookup[int(bus)], column] = self.base_p_demand[bus]
            if int(bus) in q_row_lookup:
                matrix[q_row_lookup[int(bus)], column] = self.base_q_demand[bus]
        return matrix

    def solve_state(self, parameter: np.ndarray, initial_state: np.ndarray | None = None, *, tolerance: float = 1e-12, max_iterations: int = 80) -> np.ndarray:
        initial = self._initial_state_from_case() if initial_state is None else np.asarray(initial_state, dtype=np.float64)
        result = NewtonSolver(tolerance=tolerance, max_iterations=max_iterations).solve(
            lambda state: self.residual(state, parameter),
            lambda state: self.state_jacobian(state, parameter),
            initial,
        )
        return result.state

    def continue_to_parameter(self, parameter: np.ndarray) -> np.ndarray:
        result = ParameterContinuationSolver(tolerance=1e-12).solve(
            self.base_parameter,
            np.asarray(parameter, dtype=np.float64),
            self.base_state,
            self.residual,
            self.state_jacobian,
            self.parameter_jacobian,
        )
        if not result.success:
            raise BranchContinuationError(result.message)
        return result.state

    def continue_between(self, parameter_start: np.ndarray, state_start: np.ndarray, parameter_target: np.ndarray) -> np.ndarray:
        result = ParameterContinuationSolver(tolerance=1e-12).solve(
            np.asarray(parameter_start, dtype=np.float64),
            np.asarray(parameter_target, dtype=np.float64),
            np.asarray(state_start, dtype=np.float64),
            self.residual,
            self.state_jacobian,
            self.parameter_jacobian,
        )
        if not result.success:
            raise BranchContinuationError(result.message)
        return result.state
