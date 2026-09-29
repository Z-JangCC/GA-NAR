from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class JVPCheck:
    point_index: int
    absolute_error: float
    tolerance: float
    passed: bool


def richardson_jvp(
    solve_standardized: Callable[[np.ndarray], np.ndarray],
    center: np.ndarray,
    direction: np.ndarray,
    eta: float = 1e-3,
) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    def central(step: float) -> np.ndarray:
        return (solve_standardized(center + step * direction) - solve_standardized(center - step * direction)) / (2.0 * step)
    coarse = central(eta)
    fine = central(eta / 2.0)
    return (4.0 * fine - coarse) / 3.0


def validate_implicit_jvp(
    analytic_jvp: Callable[[np.ndarray, np.ndarray], np.ndarray],
    solve_standardized: Callable[[np.ndarray], np.ndarray],
    centers: np.ndarray,
    directions: np.ndarray,
    *,
    eta: float = 1e-3,
) -> list[JVPCheck]:
    checks: list[JVPCheck] = []
    for index, (center, direction) in enumerate(zip(centers, directions)):
        analytic = np.asarray(analytic_jvp(center, direction), dtype=np.float64)
        reference = richardson_jvp(solve_standardized, center, direction, eta)
        absolute_error = float(np.linalg.norm(analytic - reference))
        tolerance = float(1e-8 + 1e-4 * np.linalg.norm(analytic))
        checks.append(JVPCheck(index, absolute_error, tolerance, absolute_error <= tolerance))
    return checks

