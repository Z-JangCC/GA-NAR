from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..core.datatypes import LocalNonlinearityResult


def compute_local_nonlinearity(
    energies: np.ndarray,
    valid_pair_mask: np.ndarray | None = None,
    *,
    reliability_threshold: float = 0.70,
) -> LocalNonlinearityResult:
    """Build a full-space activity target from an even direction count.

    V17.0 uses four directions; V17.1 re-freezes eight directions so the
    split-half rank ordering is not dominated by a two-direction average.
    """
    energies = np.asarray(energies, dtype=np.float64)
    if energies.ndim != 2 or energies.shape[1] < 4 or energies.shape[1] % 2:
        raise ValueError("local activity requires an even direction count of at least four")
    mask = np.ones(energies.shape, dtype=bool) if valid_pair_mask is None else np.asarray(valid_pair_mask, dtype=bool).reshape(energies.shape)
    active = np.all(mask & np.isfinite(energies), axis=1)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return LocalNonlinearityResult(np.empty(0), np.empty(0), indices, float("nan"), 0.0, "UNRESOLVED")
    valid_energies = energies[indices]
    scores = valid_energies.mean(axis=1)
    half_directions = energies.shape[1] // 2
    first_half = valid_energies[:, :half_directions].mean(axis=1)
    second_half = valid_energies[:, half_directions:].mean(axis=1)
    correlation = float(spearmanr(first_half, second_half).statistic)
    mean_score = float(scores.mean())
    if not np.isfinite(correlation) or correlation < reliability_threshold or mean_score <= 1e-12 or indices.size < 0.95 * energies.shape[0]:
        return LocalNonlinearityResult(scores, np.zeros_like(scores), indices, correlation, mean_score, "UNRESOLVED")
    target = scores / (scores + mean_score)
    return LocalNonlinearityResult(scores, target, indices, correlation, mean_score, "RESOLVED")
