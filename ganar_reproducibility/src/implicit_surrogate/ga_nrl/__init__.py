"""Historical compatibility namespace for the canonical NRGD API.

NRGD consumes frozen implicit-system response geometry and produces
subspace-response allocation descriptors.  This import path remains available
for archived experiments; new code should use :mod:`implicit_surrogate.nrgd`.
"""

from .capacity import ModeCapacityTarget, build_mode_capacity_targets, validate_capacity_target
from .geometry import LocalModeGeometry, build_local_mode_geometry
from .router import GANRLAllocationRouter, ModalProbeRouter, allocation_loss
from .training import AllocationTrainingResult, evaluate_allocation_predictions, evaluate_allocation_router, fit_allocation_router, fit_best_allocation_router, fit_best_fraction_router, fit_modal_probe_router, make_train_validation_indices
from .baselines import ScalarCapacityRouter, constant_target_metrics, fit_scalar_capacity_router, random_geometry_basis
from .geometry_cache import allocation_from_geometry_cache

__all__ = [
    "ModeCapacityTarget",
    "LocalModeGeometry",
    "build_mode_capacity_targets",
    "build_local_mode_geometry",
    "validate_capacity_target",
    "GANRLAllocationRouter",
    "ModalProbeRouter",
    "allocation_loss",
    "AllocationTrainingResult",
    "fit_allocation_router",
    "fit_best_fraction_router",
    "fit_best_allocation_router",
    "fit_modal_probe_router",
    "make_train_validation_indices",
    "evaluate_allocation_router",
    "evaluate_allocation_predictions",
    "ScalarCapacityRouter",
    "fit_scalar_capacity_router",
    "random_geometry_basis",
    "constant_target_metrics",
    "allocation_from_geometry_cache",
]
