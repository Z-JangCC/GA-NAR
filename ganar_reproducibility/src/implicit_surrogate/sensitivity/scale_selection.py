from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleDiagnostics:
    scale: float
    split_half_relative_error: float
    split_half_snr: float
    reliable: bool
    adjacent_cosine_similarity: float | None = None
    adjacent_relative_error: float | None = None
    qualifies_as_parent: bool = False


@dataclass(frozen=True)
class ScaleSelectionResult:
    selected_scale: float | None
    status: str
    diagnostics: tuple[ScaleDiagnostics, ...]


class SensitivityScaleSelector:
    def __init__(self, relative_error_threshold: float = 0.25, snr_threshold: float = 2.0, cosine_threshold: float = 0.98, adjacent_error_threshold: float = 0.10) -> None:
        self.relative_error_threshold = relative_error_threshold
        self.snr_threshold = snr_threshold
        self.cosine_threshold = cosine_threshold
        self.adjacent_error_threshold = adjacent_error_threshold

    @staticmethod
    def _relative_error(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second) / (0.5 * (np.linalg.norm(first) + np.linalg.norm(second)) + 1e-12))

    def select(self, matrices: dict[float, np.ndarray], split_matrices: dict[float, tuple[np.ndarray, np.ndarray]]) -> ScaleSelectionResult:
        scales = sorted(matrices, reverse=True)
        diagnostics: list[ScaleDiagnostics] = []
        for scale in scales:
            full = matrices[scale]
            first, second = split_matrices[scale]
            split_error = self._relative_error(first, second)
            split_snr = float(np.linalg.norm(full) / (0.5 * np.linalg.norm(first - second) + 1e-12))
            reliable = split_error <= self.relative_error_threshold and split_snr >= self.snr_threshold
            diagnostics.append(ScaleDiagnostics(scale, split_error, split_snr, reliable))
        by_scale = {item.scale: item for item in diagnostics}
        for parent in scales:
            child = parent / 2.0
            if child not in by_scale or not by_scale[parent].reliable or not by_scale[child].reliable:
                continue
            cosine = float(np.sum(matrices[parent] * matrices[child]) / (np.linalg.norm(matrices[parent]) * np.linalg.norm(matrices[child]) + 1e-12))
            adjacent_error = self._relative_error(matrices[parent], matrices[child])
            item = by_scale[parent]
            by_scale[parent] = ScaleDiagnostics(item.scale, item.split_half_relative_error, item.split_half_snr, item.reliable, cosine, adjacent_error, cosine >= self.cosine_threshold and adjacent_error <= self.adjacent_error_threshold)
        diagnostics = [by_scale[scale] for scale in scales]
        qualifying = [item.scale for item in diagnostics if item.qualifies_as_parent]
        selected = max(qualifying) if qualifying else None
        return ScaleSelectionResult(selected, "RESOLVED" if selected is not None else "UNRESOLVED", tuple(diagnostics))

