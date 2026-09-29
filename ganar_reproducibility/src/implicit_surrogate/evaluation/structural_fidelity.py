"""Model-side structural fidelity metrics for GA-NRL V17.2."""

from __future__ import annotations

import numpy as np
import torch


def model_output_jacobian(model: torch.nn.Module, standardized_input: np.ndarray) -> np.ndarray:
    """Return ``d model(x) / dx`` in standardized coordinates."""
    model = model.double().eval()
    value = torch.as_tensor(np.asarray(standardized_input, dtype=np.float64), dtype=torch.float64).reshape(1, -1)
    value.requires_grad_(True)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(forward, value, create_graph=False, vectorize=True)
    return jacobian.detach().cpu().numpy().reshape(-1, value.shape[-1])


def model_output(model: torch.nn.Module, standardized_input: np.ndarray) -> np.ndarray:
    model.eval()
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    with torch.no_grad():
        value = torch.as_tensor(np.asarray(standardized_input, dtype=np.float64 if dtype == torch.float64 else np.float32), dtype=dtype).reshape(1, -1)
        return model(value).detach().cpu().numpy().reshape(-1).astype(np.float64)


def relative_frobenius_error(estimate: np.ndarray, reference: np.ndarray, epsilon: float = 1e-12) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(estimate - reference) / (np.linalg.norm(reference) + epsilon))


def centered_cosine(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-15) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= epsilon:
        return float("nan")
    return float(np.sum(left * right) / denominator)


def directional_curvature(values_plus: np.ndarray, values_center: np.ndarray, values_minus: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(values_plus) - 2.0 * np.asarray(values_center) + np.asarray(values_minus)) / float(scale) ** 2


def relative_vector_error(estimate: np.ndarray, reference: np.ndarray, epsilon: float = 1e-12) -> float:
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(reference)) / (np.linalg.norm(np.asarray(reference)) + epsilon))


__all__ = ["model_output_jacobian", "model_output", "relative_frobenius_error", "relative_vector_error", "directional_curvature", "centered_cosine"]
