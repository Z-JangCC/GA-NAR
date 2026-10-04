from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.datatypes import SensitivityEstimate
from ..core.exceptions import BranchContinuationError
from ..derivatives.implicit_jvp import implicit_solution_jacobian, implicit_solution_jvp
from ..solvers.linear_solver import LinearSystemSolver
from .trace_estimation import hutchinson_directional_energy


@dataclass(frozen=True)
class DirectionalVariationResult:
    scales: tuple[float, ...]
    energies: dict[float, np.ndarray]
    valid_pair_mask: np.ndarray
    invalid_rate: float
    sensitivity_estimates: dict[float, SensitivityEstimate]
    estimator: str = "hutchinson"
    split_matrices: dict[float, tuple[np.ndarray, np.ndarray]] | None = None


def estimate_directional_variation(
    system,
    centers_std: np.ndarray,
    center_states: np.ndarray,
    center_parameters: np.ndarray,
    input_std: np.ndarray,
    state_std: np.ndarray,
    retained_q_indices: np.ndarray,
    unit_directions: np.ndarray,
    rademacher_probes: np.ndarray,
    scales: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
    *,
    solve_target: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    estimator: str = "hutchinson",
    parallel_workers: int = 1,
) -> DirectionalVariationResult:
    """Evaluate all scales on one common retained pair population."""
    centers_std = np.asarray(centers_std, dtype=np.float64)
    center_states = np.asarray(center_states, dtype=np.float64)
    center_parameters = np.asarray(center_parameters, dtype=np.float64)
    unit_directions = np.asarray(unit_directions, dtype=np.float64)
    rademacher_probes = np.asarray(rademacher_probes, dtype=np.float64)
    if estimator not in {"hutchinson", "full_jacobian", "full_jacobian_matrix"}:
        raise ValueError(f"unknown directional variation estimator: {estimator}")
    num_centers, directions_per_center, _ = unit_directions.shape
    num_pairs = num_centers * directions_per_center
    if estimator == "full_jacobian" and hasattr(system, "batch_directional_variation"):
        batch_energies, batch_mask = system.batch_directional_variation(
            centers_std,
            center_parameters,
            input_std,
            state_std,
            retained_q_indices,
            unit_directions,
            tuple(float(scale) for scale in scales),
        )
        estimates: dict[float, SensitivityEstimate] = {}
        invalid_rate = float(1.0 - np.mean(batch_mask))
        for scale in scales:
            valid = batch_mask & np.isfinite(batch_energies[float(scale)])
            if not np.any(valid):
                raise RuntimeError("no common-valid geometry pairs")
            direction_flat = unit_directions.reshape(num_pairs, -1)[valid]
            energy_flat = batch_energies[float(scale)][valid]
            second = np.einsum("i,ij,ik->jk", energy_flat, direction_flat, direction_flat) / len(energy_flat)
            trace = float(np.trace(second))
            if trace <= 0 or not np.isfinite(trace):
                raise RuntimeError("sensitivity second moment has non-positive trace")
            normalized = second / trace
            centered = normalized - np.eye(normalized.shape[0]) / normalized.shape[0]
            estimates[float(scale)] = SensitivityEstimate(second, normalized, centered, int(valid.sum()), invalid_rate, tuple(float(s) for s in scales))
        return DirectionalVariationResult(tuple(float(s) for s in scales), batch_energies, batch_mask, invalid_rate, estimates, estimator)
    energies = {float(scale): np.full(num_pairs, np.nan, dtype=np.float64) for scale in scales}
    valid_pair_mask = np.ones(num_pairs, dtype=bool)
    matrix_full = {float(scale): np.zeros((unit_directions.shape[2], unit_directions.shape[2]), dtype=np.float64) for scale in scales} if estimator == "full_jacobian_matrix" else None
    matrix_first = {float(scale): np.zeros((unit_directions.shape[2], unit_directions.shape[2]), dtype=np.float64) for scale in scales} if estimator == "full_jacobian_matrix" else None
    matrix_second = {float(scale): np.zeros((unit_directions.shape[2], unit_directions.shape[2]), dtype=np.float64) for scale in scales} if estimator == "full_jacobian_matrix" else None
    matrix_counts = {float(scale): [0, 0, 0] for scale in scales} if estimator == "full_jacobian_matrix" else None

    def default_solve(parameter: np.ndarray, initial_state: np.ndarray, _center_parameter: np.ndarray) -> np.ndarray:
        return system.solve_state(parameter, initial_state=initial_state)

    solve_fn = solve_target or default_solve
    def evaluate_pair(pair_index: int) -> tuple[int, np.ndarray | None, list[np.ndarray] | None]:
        center_index, direction_index = divmod(pair_index, directions_per_center)
        center_std = centers_std[center_index]
        center_state = center_states[center_index]
        center_parameter = center_parameters[center_index]
        direction_std = unit_directions[center_index, direction_index]
        pair_energies = np.full(len(scales), np.nan, dtype=np.float64)
        pair_matrices: list[np.ndarray] | None = [None] * len(scales) if estimator == "full_jacobian_matrix" else None  # type: ignore[list-item]
        endpoints: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        try:
            for scale in scales:
                plus_std = center_std + float(scale) * direction_std
                minus_std = center_std - float(scale) * direction_std
                plus_parameter = np.asarray(center_parameter, dtype=np.float64).copy()
                minus_parameter = np.asarray(center_parameter, dtype=np.float64).copy()
                # Parameter coordinates are reconstructed by the caller's solve function.
                plus_parameter[retained_q_indices] = plus_parameter[retained_q_indices] + input_std[retained_q_indices] * (plus_std - center_std)
                minus_parameter[retained_q_indices] = minus_parameter[retained_q_indices] + input_std[retained_q_indices] * (minus_std - center_std)
                plus_state = solve_fn(plus_parameter, center_state, center_parameter)
                minus_state = solve_fn(minus_parameter, center_state, center_parameter)
                endpoints[float(scale)] = (plus_state, minus_state, plus_parameter, minus_parameter)
            # Factorizations and JVPs are part of pair validity too: a
            # numerically unusable endpoint cannot enter any scale or
            # split-half estimate.
            for scale_index, scale in enumerate(scales):
                plus_state, minus_state, plus_parameter, minus_parameter = endpoints[float(scale)]
                plus_factor = LinearSystemSolver(system.state_jacobian(plus_state, plus_parameter))
                minus_factor = LinearSystemSolver(system.state_jacobian(minus_state, minus_parameter))
                if estimator in {"full_jacobian", "full_jacobian_matrix"}:
                    plus_jacobian = implicit_solution_jacobian(
                        system, plus_state, plus_parameter, input_std, state_std,
                        retained_q_indices, factorization=plus_factor,
                    )
                    minus_jacobian = implicit_solution_jacobian(
                        system, minus_state, minus_parameter, input_std, state_std,
                        retained_q_indices, factorization=minus_factor,
                    )
                    difference = (plus_jacobian - minus_jacobian) / (2.0 * float(scale))
                    pair_energies[scale_index] = float(np.sum(difference * difference))
                    if pair_matrices is not None:
                        pair_matrices[scale_index] = difference.T @ difference
                else:
                    probes = rademacher_probes[center_index, direction_index]
                    pair_energies[scale_index] = hutchinson_directional_energy(
                        lambda probe: implicit_solution_jvp(system, plus_state, plus_parameter, probe, input_std, state_std, retained_q_indices, factorization=plus_factor),
                        lambda probe: implicit_solution_jvp(system, minus_state, minus_parameter, probe, input_std, state_std, retained_q_indices, factorization=minus_factor),
                        probes, float(scale),
                    )
            if np.any(~np.isfinite(pair_energies)):
                raise FloatingPointError("non-finite directional sensitivity energy")
            return pair_index, pair_energies, pair_matrices
        except (BranchContinuationError, RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            return pair_index, None, None

    pair_indices = range(num_pairs)
    if int(parallel_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            pair_results = executor.map(evaluate_pair, pair_indices)
            for pair_index, pair_energies, pair_matrices in pair_results:
                if pair_energies is None:
                    valid_pair_mask[pair_index] = False
                else:
                    for scale_index, scale in enumerate(scales):
                        energies[float(scale)][pair_index] = pair_energies[scale_index]
                        if pair_matrices is not None:
                            matrix_full[float(scale)] += pair_matrices[scale_index]
                            half = num_pairs // 2
                            target = matrix_first if pair_index < half else matrix_second
                            target[float(scale)] += pair_matrices[scale_index]
                            matrix_counts[float(scale)][0] += 1
                            matrix_counts[float(scale)][1 if pair_index < half else 2] += 1
    else:
        pair_results = map(evaluate_pair, pair_indices)
        for pair_index, pair_energies, pair_matrices in pair_results:
            if pair_energies is None:
                valid_pair_mask[pair_index] = False
            else:
                for scale_index, scale in enumerate(scales):
                    energies[float(scale)][pair_index] = pair_energies[scale_index]
                    if pair_matrices is not None:
                        matrix_full[float(scale)] += pair_matrices[scale_index]
                        half = num_pairs // 2
                        target = matrix_first if pair_index < half else matrix_second
                        target[float(scale)] += pair_matrices[scale_index]
                        matrix_counts[float(scale)][0] += 1
                        matrix_counts[float(scale)][1 if pair_index < half else 2] += 1
    invalid_rate = float(1.0 - np.mean(valid_pair_mask))
    estimates: dict[float, SensitivityEstimate] = {}
    split_matrices: dict[float, tuple[np.ndarray, np.ndarray]] | None = {} if estimator == "full_jacobian_matrix" else None
    for scale in scales:
        valid = valid_pair_mask & np.isfinite(energies[float(scale)])
        if not np.any(valid):
            raise RuntimeError("no common-valid geometry pairs")
        if estimator == "full_jacobian_matrix":
            second = matrix_full[float(scale)] / max(matrix_counts[float(scale)][0], 1)
        else:
            direction_flat = unit_directions.reshape(num_pairs, -1)[valid]
            energy_flat = energies[float(scale)][valid]
            second = np.einsum("i,ij,ik->jk", energy_flat, direction_flat, direction_flat) / len(energy_flat)
        trace = float(np.trace(second))
        if trace <= 0 or not np.isfinite(trace):
            raise RuntimeError("sensitivity second moment has non-positive trace")
        normalized = second / trace
        centered = normalized - np.eye(normalized.shape[0]) / normalized.shape[0]
        estimates[float(scale)] = SensitivityEstimate(second, normalized, centered, int(valid.sum()), invalid_rate, tuple(float(s) for s in scales))
        if split_matrices is not None:
            first_count = matrix_counts[float(scale)][1]
            second_count = matrix_counts[float(scale)][2]
            first_second = matrix_first[float(scale)] / max(first_count, 1)
            second_second = matrix_second[float(scale)] / max(second_count, 1)
            first_normalized = first_second / max(float(np.trace(first_second)), 1e-15)
            second_normalized = second_second / max(float(np.trace(second_second)), 1e-15)
            split_matrices[float(scale)] = (
                first_normalized - np.eye(first_normalized.shape[0]) / first_normalized.shape[0],
                second_normalized - np.eye(second_normalized.shape[0]) / second_normalized.shape[0],
            )
    return DirectionalVariationResult(tuple(float(s) for s in scales), energies, valid_pair_mask, invalid_rate, estimates, estimator, split_matrices)
