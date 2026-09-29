from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StandardizationStatistics:
    q_mean: np.ndarray
    q_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    retained_q_indices: np.ndarray
    removed_q_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


@dataclass(frozen=True)
class DatasetSplit:
    q_phys: np.ndarray
    z_phys: np.ndarray
    x_std: np.ndarray
    y_std: np.ndarray
    sample_ids: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.sample_ids.shape[0])


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetSplit
    prediction: DatasetSplit
    sensitivity: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    representation_probe: DatasetSplit
    standardization: StandardizationStatistics
    reference_activation_ids: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationSolution:
    parameter: np.ndarray
    state: np.ndarray
    residual_norm: float
    jacobian_rcond: float
    success: bool
    continuation_parameter: float = 1.0
    steps: int = 0
    message: str = ""


@dataclass(frozen=True)
class SensitivityEstimate:
    second_moment: np.ndarray
    normalized_matrix: np.ndarray
    centered_matrix: np.ndarray
    num_pairs: int
    invalid_rate: float
    scales: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubspaceResult:
    basis: np.ndarray
    projector: np.ndarray
    eigenvalues: np.ndarray
    rank_used: int
    r90: int
    captured_mass: float
    status: str
    q_u: float | None = None


@dataclass(frozen=True)
class LocalNonlinearityResult:
    scores: np.ndarray
    target: np.ndarray
    active_indices: np.ndarray
    q_act: float
    mean_score: float
    status: str


@dataclass(frozen=True)
class GANRLAllocationSummary:
    """Serializable summary of the standalone GA-NRL allocation stage."""

    rank: int
    num_centers: int
    valid_centers: int
    mean_capacity: float
    conservation_max_error: float
    status: str
