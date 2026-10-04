from __future__ import annotations

import numpy as np

from ..core.datatypes import DatasetSplit, StandardizationStatistics
from ..core.exceptions import DegenerateStateCoordinateError


class Standardizer:
    """Train-only coordinate standardization with explicit degeneracy handling."""

    def __init__(self, statistics: StandardizationStatistics) -> None:
        self.statistics = statistics

    @classmethod
    def fit(cls, q_phys: np.ndarray, z_phys: np.ndarray) -> "Standardizer":
        q_phys = np.asarray(q_phys, dtype=np.float64)
        z_phys = np.asarray(z_phys, dtype=np.float64)
        q_mean = q_phys.mean(axis=0)
        q_std = q_phys.std(axis=0)
        z_mean = z_phys.mean(axis=0)
        z_std = z_phys.std(axis=0)
        retained = np.flatnonzero(q_std > 1e-10)
        removed = np.flatnonzero(q_std <= 1e-10)
        if np.any(z_std <= 1e-10):
            indices = np.flatnonzero(z_std <= 1e-10).tolist()
            raise DegenerateStateCoordinateError(f"state coordinate standard deviation is degenerate at indices {indices}; frozen protocol forbids silent deletion")
        stats = StandardizationStatistics(q_mean, q_std, z_mean, z_std, retained, removed)
        return cls(stats)

    def transform_input(self, q_phys: np.ndarray) -> np.ndarray:
        q_phys = np.asarray(q_phys, dtype=np.float64)
        return (q_phys[..., self.statistics.retained_q_indices] - self.statistics.q_mean[self.statistics.retained_q_indices]) / self.statistics.q_std[self.statistics.retained_q_indices]

    def transform_state(self, z_phys: np.ndarray) -> np.ndarray:
        return (np.asarray(z_phys, dtype=np.float64) - self.statistics.z_mean) / self.statistics.z_std

    def inverse_state(self, y_std: np.ndarray) -> np.ndarray:
        return np.asarray(y_std, dtype=np.float64) * self.statistics.z_std + self.statistics.z_mean

    def split(self, q_phys: np.ndarray, z_phys: np.ndarray, sample_ids: np.ndarray, metadata=None) -> DatasetSplit:
        return DatasetSplit(
            q_phys=np.asarray(q_phys, dtype=np.float64),
            z_phys=np.asarray(z_phys, dtype=np.float64),
            x_std=self.transform_input(q_phys),
            y_std=self.transform_state(z_phys),
            sample_ids=np.asarray(sample_ids, dtype=np.int64),
            metadata={} if metadata is None else metadata,
        )
