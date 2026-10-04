from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..sensitivity.sensitivity_matrix import build_sensitivity_matrices


def representation_jvp(model, standardized_input: np.ndarray, direction: np.ndarray, layer_name: str, whitening_matrix: np.ndarray, activation_mean: np.ndarray) -> np.ndarray:
    """Compute one representation JVP in a float64 analysis clone."""
    model.eval()
    model = model.double()
    input_tensor = torch.as_tensor(np.asarray(standardized_input, dtype=np.float64), dtype=torch.float64).reshape(1, -1).requires_grad_(True)
    direction_tensor = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=torch.float64).reshape(1, -1)
    whitening = torch.as_tensor(np.asarray(whitening_matrix, dtype=np.float64), dtype=torch.float64)
    mean = torch.as_tensor(np.asarray(activation_mean, dtype=np.float64), dtype=torch.float64)

    def representation_fn(value: torch.Tensor) -> torch.Tensor:
        _, activations = model(value, return_activations=True)
        activation = activations[layer_name]
        return (activation - mean) @ whitening.T

    _, derivative = torch.autograd.functional.jvp(representation_fn, (input_tensor,), (direction_tensor,), create_graph=False, strict=False)
    return derivative.detach().cpu().numpy().reshape(-1)


def representation_jacobian(model, standardized_input: np.ndarray, layer_name: str, whitening_matrix: np.ndarray, activation_mean: np.ndarray) -> np.ndarray:
    """Return one layer Jacobian so all frozen probes share one AD pass."""
    model.eval()
    model = model.double()
    input_tensor = torch.as_tensor(np.asarray(standardized_input, dtype=np.float64), dtype=torch.float64).reshape(1, -1).requires_grad_(True)
    whitening = torch.as_tensor(np.asarray(whitening_matrix, dtype=np.float64), dtype=torch.float64)
    mean = torch.as_tensor(np.asarray(activation_mean, dtype=np.float64), dtype=torch.float64)

    def representation_fn(value: torch.Tensor) -> torch.Tensor:
        _, activations = model(value, return_activations=True)
        return (activations[layer_name] - mean) @ whitening.T

    jacobian = torch.autograd.functional.jacobian(representation_fn, input_tensor, create_graph=False, strict=False, vectorize=True)
    return jacobian.detach().cpu().numpy().reshape(-1, input_tensor.shape[-1])


@dataclass(frozen=True)
class RepresentationSensitivityResult:
    second_moment: np.ndarray
    normalized_matrix: np.ndarray
    centered_matrix: np.ndarray
    invalid_rate: float
    trace: float
    energies: np.ndarray
    valid_pair_mask: np.ndarray


def estimate_representation_sensitivity(
    model,
    centers_std: np.ndarray,
    unit_directions: np.ndarray,
    rademacher_probes: np.ndarray,
    layer_name: str,
    whitening,
    scales: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
    selected_scale: float = 0.01,
    valid_pair_mask: np.ndarray | None = None,
    estimator: str = "hutchinson",
) -> RepresentationSensitivityResult:
    centers_std = np.asarray(centers_std, dtype=np.float64)
    directions = np.asarray(unit_directions, dtype=np.float64)
    probes = np.asarray(rademacher_probes, dtype=np.float64)
    if estimator not in {"hutchinson", "full_jacobian"}:
        raise ValueError(f"unknown representation estimator: {estimator}")
    pairs = directions.reshape(-1, directions.shape[-1])
    if valid_pair_mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    else:
        mask = np.asarray(valid_pair_mask, dtype=bool).reshape(-1).copy()
    energies = np.full(len(pairs), np.nan, dtype=np.float64)
    flat_index = 0
    for center_index, center in enumerate(centers_std):
        for direction_index, direction in enumerate(directions[center_index]):
            if not mask[flat_index]:
                flat_index += 1
                continue
            plus = center + selected_scale * direction
            minus = center - selected_scale * direction
            try:
                plus_jacobian = representation_jacobian(model, plus, layer_name, whitening.whitening_matrix, whitening.activation_mean)
                minus_jacobian = representation_jacobian(model, minus, layer_name, whitening.whitening_matrix, whitening.activation_mean)
                difference = (plus_jacobian - minus_jacobian) / (2.0 * selected_scale)
                if estimator == "full_jacobian":
                    energies[flat_index] = float(np.sum(difference * difference))
                else:
                    probe_values = np.asarray(probes[center_index, direction_index], dtype=np.float64) @ difference.T
                    energies[flat_index] = float(np.mean(np.sum(probe_values * probe_values, axis=1)))
                if not np.isfinite(energies[flat_index]):
                    raise FloatingPointError("non-finite representation sensitivity energy")
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
                mask[flat_index] = False
            flat_index += 1
    second, normalized, centered = build_sensitivity_matrices(energies, pairs, mask)
    trace = float(np.trace(second))
    if trace <= 1e-12 or not np.isfinite(trace):
        raise ValueError("representation sensitivity trace is below the frozen N/A threshold")
    return RepresentationSensitivityResult(second, normalized, centered, float(1.0 - np.mean(mask)), trace, energies, mask)
