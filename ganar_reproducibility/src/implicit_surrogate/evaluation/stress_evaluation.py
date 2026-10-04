from __future__ import annotations

import numpy as np

from .prediction_metrics import evaluate_predictions


def evaluate_stress_points(model, stress_splits, system, standardization) -> list[dict[str, float]]:
    results = []
    for loading_fraction, split in stress_splits:
        values = evaluate_predictions(model, split, system, standardization)
        values["loading_fraction"] = float(loading_fraction)
        results.append(values)
    return results

