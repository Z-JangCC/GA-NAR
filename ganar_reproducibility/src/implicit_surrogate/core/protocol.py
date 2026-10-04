from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class FrozenProtocol:
    name: str = "GA-NRL"
    revision: str = "17.0"
    status: str = "FINAL-FROZEN"
    geometry_scales: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
    geometry_pair_budgets: tuple[int, ...] = (512, 1024, 2048)
    num_rademacher_probes: int = 4
    num_geometry_centers: int = 512
    activity_directions_per_center: int = 4
    activity_reliability_threshold: float = 0.70
    invalid_pair_threshold: float = 0.05
    rank_cap: int = 8
    reliability_relative_error: float = 0.25
    reliability_snr: float = 2.0
    scale_cosine_threshold: float = 0.98
    scale_relative_error_threshold: float = 0.10
    subspace_overlap_threshold: float = 0.90
    # The V17.0 defaults above are intentionally immutable.  Later re-freezes
    # use the explicit fields below instead of changing the original protocol.
    geometry_directions_per_center: int = 4
    geometry_estimator: str = "hutchinson"
    geometry_center_sampling: str = "iid_normal"
    geometry_pair_budget_mode: str = "nested_directions"
    representation_directions_per_center: int = 8
    representation_num_rademacher_probes: int = 4
    representation_probe_centers: int = 64
    representation_estimator: str = "hutchinson"
    ieee_state_definition: str = "canonical_full_state"
    artifact_namespace: str = "v17_0"
    ieee_geometry_pair_budgets: tuple[int, ...] = (512, 1024, 2048)
    ieee_num_geometry_centers: int = 512
    geometry_parallel_workers: int = 1
    ieee_geometry_estimator: str = "hutchinson"
    ieee_activity_directions_per_center: int = 4
    cpf_step_size: float = 0.025
    cpf_max_steps: int = 800
    cpf_max_loading_parameter: float = 20.0
    cpf_parallel_workers: int = 1

    @property
    def identifier(self) -> str:
        return f"{self.name}-{self.revision}-{self.status}"


FROZEN_PROTOCOL = FrozenProtocol()


# V17.1 is a separate scientific protocol namespace.  It retains the V17.0
# scales, thresholds, model definitions, and formal seeds, while explicitly
# changing the sensitivity estimator/population and the IEEE state definition
# that failed their V17.0 preconditions.
REFROZEN_PROTOCOL = FrozenProtocol(
    revision="17.1",
    status="RE-FROZEN",
    geometry_pair_budgets=(32768, 131072, 524288),
    num_geometry_centers=65536,
    geometry_directions_per_center=8,
    num_rademacher_probes=0,
    activity_directions_per_center=8,
    activity_reliability_threshold=0.50,
    geometry_estimator="full_jacobian",
    geometry_center_sampling="iid_normal",
    geometry_pair_budget_mode="nested_centers",
    representation_directions_per_center=8,
    representation_num_rademacher_probes=16,
    representation_probe_centers=128,
    representation_estimator="full_jacobian",
    ieee_state_definition="case118_reduced_state_excluding_invariant_bus9_vm",
    artifact_namespace="v17_1",
    ieee_geometry_pair_budgets=(128, 256, 512),
    ieee_num_geometry_centers=64,
    geometry_parallel_workers=16,
    ieee_geometry_estimator="full_jacobian_matrix",
    ieee_activity_directions_per_center=128,
    cpf_step_size=0.025,
    cpf_max_steps=4000,
    cpf_max_loading_parameter=20.0,
    cpf_parallel_workers=16,
)


def protocol_identifier(protocol: FrozenProtocol) -> str:
    return protocol.identifier


def active_protocol() -> FrozenProtocol:
    """Return the process-selected protocol without changing V17.0 defaults."""
    identifier = os.environ.get("IMPLICIT_SURROGATE_PROTOCOL_ID", FROZEN_PROTOCOL.identifier)
    if identifier == REFROZEN_PROTOCOL.identifier:
        return REFROZEN_PROTOCOL
    return FROZEN_PROTOCOL


def active_protocol_id() -> str:
    return active_protocol().identifier
