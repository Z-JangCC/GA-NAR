from __future__ import annotations

import numpy as np

from ..systems.synthetic.model import SyntheticImplicitSystem
from .splits import build_bundle


def build_synthetic_dataset(
    system: SyntheticImplicitSystem,
    *,
    counts: dict[str, int] | None = None,
    seed: int = 1234,
) -> object:
    counts = dict(counts or {})
    if not counts:
        counts = {"prediction": 8192, "sensitivity": 512, "validation": 1024, "test": 2048, "representation_probe": 64}
    total = sum(counts.values())
    parameters = system.sample_parameters(total, seed)
    if hasattr(system, "solve_states"):
        states = system.solve_states(parameters)
    else:
        states = np.empty_like(parameters)
        for index, parameter in enumerate(parameters):
            states[index] = system.solve_state(parameter)
    return build_bundle(parameters, states, counts, metadata={"benchmark": "synthetic", "seed": seed})
