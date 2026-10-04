from __future__ import annotations

import numpy as np


def mean_sample_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def paired_summary(full_values, control_values) -> dict[str, object]:
    full = np.asarray(full_values, dtype=np.float64)
    control = np.asarray(control_values, dtype=np.float64)
    if full.shape != control.shape:
        raise ValueError("paired arrays must have the same shape")
    differences = full - control
    full_mean, full_std = mean_sample_std(full)
    control_mean, control_std = mean_sample_std(control)
    delta_mean, delta_std = mean_sample_std(differences)
    return {
        "full_mean": full_mean,
        "full_sample_std": full_std,
        "control_mean": control_mean,
        "control_sample_std": control_std,
        "paired_delta_mean": delta_mean,
        "paired_delta_sample_std": delta_std,
        "all_three_seed_improved": bool(np.all(differences < 0)) if differences.size == 3 else False,
    }

