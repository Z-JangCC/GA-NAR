from __future__ import annotations

import numpy as np


def paired_bootstrap(differences, samples=10000, seed=20260924):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    return {"mean": float(differences.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "wins": int(np.sum(differences < 0)),
            "n": int(len(differences))}


def relative_improvement(ga, baseline):
    ga = np.asarray(ga, dtype=float); baseline = np.asarray(baseline, dtype=float)
    return float(np.mean((baseline - ga) / baseline))
