"""Load standardized benchmark arrays and training-only NRGD descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SYSTEMS = ("ieee118_acpf", "kuramoto64", "nonlinear_heat_fem")


@dataclass(frozen=True)
class Benchmark:
    system: str
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    geometry_x: np.ndarray
    rho: np.ndarray
    pi: np.ndarray
    projector: np.ndarray
    q_mean: np.ndarray
    q_scale: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray
    source_hashes: dict[str, str]


def load_benchmark(root: str | Path, system: str) -> Benchmark:
    """Read fixed splits; geometry centers never enter validation or test loss."""
    if system not in SYSTEMS:
        raise ValueError(f"supported systems: {', '.join(SYSTEMS)}")
    root = Path(root)
    data_path = root / "data_store/processed" / system / "dataset.npz"
    geometry_path = root / "data_store/geometry" / system / "geometry.npz"
    descriptor_path = geometry_path.with_suffix(".json")
    with np.load(data_path, allow_pickle=False) as raw, np.load(
        geometry_path, allow_pickle=False
    ) as geometry:
        q_mean, q_scale = raw["q_mean"].copy(), raw["q_scale"].copy()
        z_mean, z_scale = raw["z_mean"].copy(), raw["z_scale"].copy()
        x = ((raw["q"] - q_mean) / q_scale).astype(np.float32)
        y = ((raw["z"] - z_mean) / z_scale).astype(np.float32)
        train = raw["train"].astype(np.int64)
        validation = raw["val"].astype(np.int64)
        test = raw["test"].astype(np.int64)
        geometry_x = ((geometry["centers"] - q_mean) / q_scale).astype(np.float32)
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor.get("status", "").lower() != "resolved":
        raise ValueError(f"geometry descriptor is unresolved for {system}")
    rho = np.asarray(descriptor["rho"], dtype=np.float32)
    pi = np.asarray(descriptor["allocation"], dtype=np.float32)
    projector = np.asarray(descriptor["projector"], dtype=np.float64)
    if rho.shape != (len(geometry_x),) or pi.shape != (len(geometry_x), 2):
        raise ValueError("geometry targets do not align with training centers")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(
        geometry_x
    ).all() or not np.isfinite(rho).all() or not np.isfinite(pi).all():
        raise ValueError("non-finite benchmark or geometry values")
    if (rho < 0).any() or (rho > 1).any() or (pi < 0).any() or not np.allclose(
        pi.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("invalid intensity or allocation targets")
    if any(len(set(a) & set(b)) for a, b in (
        (train, validation), (train, test), (validation, test)
    )):
        raise ValueError("benchmark splits overlap")
    paths = {"dataset": data_path, "geometry": geometry_path,
             "descriptor": descriptor_path}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    return Benchmark(system, x, y, train, validation, test, geometry_x, rho, pi,
                     projector, q_mean, q_scale, z_mean, z_scale, hashes)
