from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.random import RandomManager


@dataclass(frozen=True)
class GeometryDirections:
    unit_directions: np.ndarray
    rademacher_probes: np.ndarray
    metadata: dict


def generate_geometry_directions(
    num_centers: int,
    input_dimension: int,
    *,
    directions_per_center: int = 4,
    num_rademacher_probes: int = 4,
    seed: int = 2026,
) -> GeometryDirections:
    manager = RandomManager(seed)
    unit_directions = manager.unit_sphere(num_centers * directions_per_center, input_dimension).reshape(num_centers, directions_per_center, input_dimension)
    probes = manager.rademacher((num_centers * directions_per_center, num_rademacher_probes, input_dimension)).reshape(num_centers, directions_per_center, num_rademacher_probes, input_dimension)
    return GeometryDirections(
        unit_directions=unit_directions,
        rademacher_probes=probes,
        metadata={"seed": seed, "directions_per_center": directions_per_center, "num_rademacher_probes": num_rademacher_probes},
    )


def generate_probe_directions(num_centers: int, input_dimension: int, *, directions_per_center: int = 8, seed: int = 3030) -> np.ndarray:
    return RandomManager(seed).unit_sphere(num_centers * directions_per_center, input_dimension).reshape(num_centers, directions_per_center, input_dimension)

