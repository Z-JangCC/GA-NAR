from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalModeGeometry:
    """Per-centre finite-scale response geometry used by NRGD."""

    matrices: np.ndarray
    energies: np.ndarray
    directions: np.ndarray
    valid_centers: np.ndarray
    scale: float

    @property
    def dimension(self) -> int:
        return int(self.matrices.shape[-1])


def build_local_mode_geometry(
    energies: np.ndarray,
    unit_directions: np.ndarray,
    valid_pair_mask: np.ndarray | None = None,
    *,
    scale: float,
) -> LocalModeGeometry:
    """Estimate each centre's local geometry matrix ``B_i``.

    ``energies[i,k]`` is the squared finite-scale Jacobian variation along
    direction ``unit_directions[i,k]``.  The directional second moment is the
    only information required by NRGD and makes this stage compatible with
    both full-Jacobian and Hutchinson estimators.
    """
    values = np.asarray(energies, dtype=np.float64)
    directions = np.asarray(unit_directions, dtype=np.float64)
    if values.ndim != 2 or directions.ndim != 3 or values.shape[:2] != directions.shape[:2]:
        raise ValueError("energies and unit_directions have incompatible shapes")
    if directions.shape[2] == 0:
        raise ValueError("directions must have a positive input dimension")
    mask = np.ones(values.shape, dtype=bool) if valid_pair_mask is None else np.asarray(valid_pair_mask, dtype=bool).reshape(values.shape)
    finite = mask & np.isfinite(values) & np.isfinite(directions).all(axis=2)
    matrices = np.zeros((values.shape[0], directions.shape[2], directions.shape[2]), dtype=np.float64)
    centre_energies = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid_centers = np.zeros(values.shape[0], dtype=bool)
    for index in range(values.shape[0]):
        if not np.any(finite[index]):
            continue
        local_values = np.maximum(values[index, finite[index]], 0.0)
        local_directions = directions[index, finite[index]]
        matrices[index] = np.einsum("i,ij,ik->jk", local_values, local_directions, local_directions) / len(local_values)
        matrices[index] = 0.5 * (matrices[index] + matrices[index].T)
        centre_energies[index] = float(np.trace(matrices[index]))
        valid_centers[index] = np.isfinite(centre_energies[index]) and centre_energies[index] >= 0.0
    return LocalModeGeometry(matrices, centre_energies, directions, valid_centers, float(scale))
