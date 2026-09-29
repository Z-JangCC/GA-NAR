from __future__ import annotations

import numpy as np


def centered_sensitivity_cosine_distance(system_centered: np.ndarray, representation_centered: np.ndarray) -> float:
    system_centered = np.asarray(system_centered, dtype=np.float64)
    representation_centered = np.asarray(representation_centered, dtype=np.float64)
    system_norm = np.linalg.norm(system_centered)
    representation_norm = np.linalg.norm(representation_centered)
    if system_norm <= 0 or representation_norm <= 0:
        return float("nan")
    return float(1.0 - np.sum(system_centered * representation_centered) / (system_norm * representation_norm))

