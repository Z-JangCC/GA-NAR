from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeWhitening:
    activation_mean: np.ndarray
    activation_covariance: np.ndarray
    whitening_regularization: float
    whitening_matrix: np.ndarray

    def transform(self, activation: np.ndarray) -> np.ndarray:
        return (np.asarray(activation, dtype=np.float64) - self.activation_mean) @ self.whitening_matrix.T

    def validation_error(self) -> float:
        dimension = self.activation_covariance.shape[0]
        regularized = self.activation_covariance + self.whitening_regularization * np.eye(dimension)
        identity = self.whitening_matrix @ regularized @ self.whitening_matrix.T
        return float(np.linalg.norm(identity - np.eye(dimension)) / np.linalg.norm(np.eye(dimension)))


def fit_ridge_whitening(activations: np.ndarray) -> RidgeWhitening:
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("activations must have shape (samples, features)")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, bias=True)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    regularization = max(1e-8, 1e-5 * float(np.trace(covariance)) / covariance.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    inverse_sqrt = np.maximum(eigenvalues + regularization, 1e-15) ** -0.5
    whitening_matrix = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    result = RidgeWhitening(mean, covariance, regularization, whitening_matrix)
    if result.validation_error() > 1e-8:
        raise ValueError("ridge whitening identity check failed")
    return result

