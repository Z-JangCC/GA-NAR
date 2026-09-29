from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.exceptions import BranchContinuationError
from ..core.random import RandomManager
from ..systems.power_flow.equations import ACPowerFlowSystem
from .splits import build_bundle


def build_power_flow_dataset(
    system: ACPowerFlowSystem,
    *,
    counts: dict[str, int] | None = None,
    seed: int = 4321,
    max_proposals: int | None = None,
    parallel_workers: int = 1,
) -> object:
    counts = dict(counts or {})
    if not counts:
        counts = {"prediction": 8192, "sensitivity": 512, "validation": 1024, "test": 2048, "representation_probe": 64}
    total = sum(counts.values())
    rng = RandomManager(seed)
    parameters = []
    states = []
    proposals = 0
    max_proposals = max_proposals or total * 100
    def solve_one(parameter):
        try:
            return system.continue_to_parameter(parameter)
        except (BranchContinuationError, RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

    workers = max(1, int(parallel_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(parameters) < total and proposals < max_proposals:
            batch_size = min(max(workers * 4, 16), max_proposals - proposals)
            candidate_parameters = []
            for _ in range(batch_size):
                global_loading_factor = float(rng.uniform(0.85, 1.15, ()))
                local_load_perturbation = rng.uniform(-0.05, 0.05, (system.parameter_dimension,))
                candidate_parameters.append(global_loading_factor * (1.0 + local_load_perturbation))
            proposals += batch_size
            candidate_states = list(executor.map(solve_one, candidate_parameters))
            for parameter, state in zip(candidate_parameters, candidate_states):
                if state is not None:
                    parameters.append(parameter)
                    states.append(state)
                    if len(parameters) >= total:
                        break
    if len(parameters) < total:
        raise RuntimeError(f"could only accept {len(parameters)} of {total} nominal samples after {proposals} proposals")
    q_phys = np.asarray(parameters, dtype=np.float64)
    z_phys = np.asarray(states, dtype=np.float64)
    bundle = build_bundle(q_phys, z_phys, counts, metadata={"benchmark": "ieee118", "seed": seed, "proposal_count": proposals, "proposal_rejection_rate": 1.0 - total / proposals})
    return bundle
