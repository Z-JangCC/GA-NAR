"""Canonical NRGD API.

NRGD (Nonlinear Response Geometry Decomposition) is the system-side name for
the historical allocation-only geometry package.  The original
``implicit_surrogate.ga_nrl`` import path remains as a compatibility alias for
archived experiments; new code should import descriptors and routers here.
"""

from ..ga_nrl import *  # noqa: F401,F403
from ..ga_nrl import (
    GANRLAllocationRouter as NRGDAllocationRouter,
    ModeCapacityTarget as NRGDResponseAllocationTarget,
    LocalModeGeometry as LocalResponseGeometry,
)

__all__ = [
    "NRGDAllocationRouter",
    "NRGDResponseAllocationTarget",
    "LocalResponseGeometry",
    "build_mode_capacity_targets",
    "build_local_mode_geometry",
    "validate_capacity_target",
    "allocation_loss",
    "fit_allocation_router",
    "fit_best_allocation_router",
    "fit_best_fraction_router",
    "evaluate_allocation_router",
    "evaluate_allocation_predictions",
    "allocation_from_geometry_cache",
]
