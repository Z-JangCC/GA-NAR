from __future__ import annotations

import numpy as np


def project_to_probability_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, values.size + 1)
    active = sorted_values - cssv / indices > 0
    if not np.any(active):
        threshold = cssv[-1] / values.size
    else:
        rho = np.flatnonzero(active)[-1]
        threshold = cssv[rho] / float(rho + 1)
    return np.maximum(values - threshold, 0.0)


def project_sensitivity_spectrum(normalized_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_matrix = np.asarray(normalized_matrix, dtype=np.float64)
    dimension = normalized_matrix.shape[0]
    estimate = 0.5 * (((dimension + 2.0) * normalized_matrix - np.eye(dimension)) / 2.0 + ((dimension + 2.0) * normalized_matrix - np.eye(dimension)).T / 2.0)
    eigenvalues, eigenvectors = np.linalg.eigh(estimate)
    projected_values = project_to_probability_simplex(eigenvalues)
    projected = (eigenvectors * projected_values) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    return projected, projected_values[::-1], eigenvectors[:, ::-1]

