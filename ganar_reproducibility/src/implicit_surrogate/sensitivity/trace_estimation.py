from __future__ import annotations

import numpy as np


def hutchinson_directional_energy(
    plus_jvp,
    minus_jvp,
    rademacher_probes: np.ndarray,
    scale: float,
) -> float:
    """Estimate ``|| (J_plus-J_minus)/(2h) ||_F^2`` using shared probes."""
    values = []
    for probe in np.asarray(rademacher_probes, dtype=np.float64):
        difference = (np.asarray(plus_jvp(probe)) - np.asarray(minus_jvp(probe))) / (2.0 * scale)
        values.append(float(np.dot(difference, difference)))
    return float(np.mean(values))

