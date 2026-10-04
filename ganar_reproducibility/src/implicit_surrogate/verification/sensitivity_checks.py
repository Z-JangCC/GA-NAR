from __future__ import annotations

import numpy as np


def check_normalized_sensitivity_trace(normalized_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(np.trace(normalized_matrix)) - 1.0) <= tolerance)


def check_projector(projector: np.ndarray, tolerance: float = 1e-10) -> dict[str, bool]:
    projector = np.asarray(projector)
    return {"symmetric": bool(np.allclose(projector, projector.T, atol=tolerance)), "idempotent": bool(np.allclose(projector @ projector, projector, atol=tolerance))}

