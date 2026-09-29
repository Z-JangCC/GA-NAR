from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IndependentDataset:
    system: str
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    validation_outer: np.ndarray
    test: np.ndarray
    geometry_x: np.ndarray
    rho: np.ndarray
    pi: np.ndarray
    projector: np.ndarray
    manifest: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_independent_dataset(system: str, root: str | Path = ".") -> IndependentDataset:
    """Read arrays directly; no dependency on the legacy Python package."""
    root = Path(root)
    data_path = root / "data_store" / "processed" / system / "dataset.npz"
    geometry_path = root / "data_store" / "geometry" / system / "geometry.npz"
    geometry_meta_path = root / "data_store" / "geometry" / system / "geometry.json"
    raw = np.load(data_path)
    x = (raw["q"] - raw["q_mean"]) / raw["q_scale"]
    y = (raw["z"] - raw["z_mean"]) / raw["z_scale"]
    geom = np.load(geometry_path)
    meta = json.loads(geometry_meta_path.read_text())
    centers = geom["centers"]
    geometry_x = (centers - raw["q_mean"]) / raw["q_scale"]
    rho = np.asarray(meta["rho"], dtype=np.float64)
    pi = np.asarray(meta["allocation"], dtype=np.float64)
    projector = np.asarray(meta["projector"], dtype=np.float64)
    manifest = {
        "system": system,
        "dataset_sha256": _sha256(data_path),
        "geometry_sha256": _sha256(geometry_path),
        "geometry_meta_sha256": _sha256(geometry_meta_path),
        "sizes": {"train": len(raw["train"]), "validation_outer": len(raw["val"]),
                  "test": len(raw["test"]), "geometry": len(geometry_x)},
    }
    return IndependentDataset(system, x, y, raw["train"].astype(int),
                              raw["val"].astype(int), raw["test"].astype(int),
                              geometry_x, rho, pi, projector, manifest)


def nested_train_validation(dataset: IndependentDataset, fold: int = 0,
                            fraction: float = 0.1):
    """Deterministic inner validation split drawn only from the training set."""
    keys = np.asarray([
        hashlib.sha256(f"ganar-lw-independent|{dataset.system}|{fold}|{int(i)}".encode()).hexdigest()
        for i in dataset.train
    ])
    order = np.argsort(keys)
    n_val = max(1, int(round(len(order) * fraction)))
    return dataset.train[order[n_val:]], dataset.train[order[:n_val]]
