from __future__ import annotations

import numpy as np


def directional_derivative_error(function, point: np.ndarray, direction: np.ndarray, analytic: np.ndarray, step: float = 1e-6) -> float:
    finite_difference = (function(point + step * direction) - function(point - step * direction)) / (2.0 * step)
    return float(np.linalg.norm(np.asarray(finite_difference) - np.asarray(analytic)))

