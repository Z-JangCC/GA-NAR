"""Low-dimensional descriptor-cache interface for oracle-level NRGD routing.

The cache is produced by a small, fixed number of local directional probes at
the operating point.  It is not a learned replacement for the physical
geometry estimator: it is the deployable interface that makes the local
curvature information identifiable at inference time.
"""
from __future__ import annotations

import numpy as np


def allocation_from_geometry_cache(
    mode_strengths: np.ndarray,
    complement_strength: np.ndarray,
    total_strength: np.ndarray,
    mean_strength: float,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert cached ``(b_ij, b_perp, e_i)`` into conserved allocations."""
    modes = np.asarray(mode_strengths, dtype=np.float64)
    complement = np.asarray(complement_strength, dtype=np.float64)
    total = np.asarray(total_strength, dtype=np.float64)
    if modes.ndim == 1:
        modes = modes[None, :]
    complement = np.broadcast_to(complement.reshape(-1, 1), (len(modes), 1))
    total = np.broadcast_to(total.reshape(-1, 1), (len(modes), 1))
    if modes.shape[0] != complement.shape[0] or not np.isfinite(modes).all() or not np.isfinite(complement).all() or not np.isfinite(total).all():
        raise ValueError("geometry cache arrays must be finite and row-aligned")
    if not np.isfinite(mean_strength) or mean_strength <= 0 or epsilon <= 0:
        raise ValueError("mean_strength and epsilon must be positive finite values")
    raw = np.concatenate([np.maximum(modes, 0.0), np.maximum(complement, 0.0)], axis=1)
    fraction = (raw + epsilon) / (total + (modes.shape[1] + 1) * epsilon)
    alpha = total / (total + float(mean_strength))
    return np.concatenate([1.0 - alpha, alpha * fraction], axis=1)


__all__ = ["allocation_from_geometry_cache"]
