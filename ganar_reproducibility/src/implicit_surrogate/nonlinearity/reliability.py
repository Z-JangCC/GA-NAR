"""Independent reliability primitives for local activity targets."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def split_half_spearman_reliability(energies: np.ndarray) -> float:
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or values.shape[1] % 2:
        raise ValueError("expected energies with an even direction count of at least 4")
    half = values.shape[1] // 2
    first = values[:, :half].mean(axis=1)
    second = values[:, half:].mean(axis=1)
    return float(spearmanr(first, second).statistic)


def activity_reliability_passes(energies: np.ndarray, *, threshold: float = 0.70) -> bool:
    correlation = split_half_spearman_reliability(energies)
    return bool(np.isfinite(correlation) and correlation >= threshold)


__all__ = ["split_half_spearman_reliability", "activity_reliability_passes"]
