from __future__ import annotations

import numpy as np


def build_sensitivity_matrices(energies: np.ndarray, unit_directions: np.ndarray, valid_pair_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    directions = np.asarray(unit_directions, dtype=np.float64).reshape(energies.size, -1)
    mask = np.ones(energies.size, dtype=bool) if valid_pair_mask is None else np.asarray(valid_pair_mask, dtype=bool).reshape(-1)
    mask &= np.isfinite(energies)
    if not np.any(mask):
        raise ValueError("no valid pairs")
    second = np.einsum("i,ij,ik->jk", energies[mask], directions[mask], directions[mask]) / int(mask.sum())
    trace = float(np.trace(second))
    if trace <= 0:
        raise ValueError("second moment trace must be positive")
    normalized = second / trace
    centered = normalized - np.eye(second.shape[0]) / second.shape[0]
    return second, normalized, centered

