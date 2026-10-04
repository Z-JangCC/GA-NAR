"""Dataset generation, finite-difference probes, and cache management."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .training import Standardizer, TrainingArrays


def _normal_directions(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    directions = rng.normal(size=shape)
    norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    return (directions / np.maximum(norm, 1e-12)).astype(np.float32)


def _diagnostic_mask(diagnostics: Any, n: int) -> np.ndarray:
    if diagnostics is None:
        return np.ones(n, dtype=bool)
    if isinstance(diagnostics, dict):
        value = diagnostics.get("converged", diagnostics.get("success", True))
    else:
        value = getattr(diagnostics, "converged", getattr(diagnostics, "success", True))
    mask = np.asarray(value, dtype=bool)
    if mask.ndim == 0:
        mask = np.full(n, bool(mask), dtype=bool)
    return mask.reshape(-1)


def solve_checked(system: Any, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    """Call a system solver while accepting its documented tuple or array form."""

    solve_arguments: dict[str, Any] = {}
    if "raise_on_failure" in inspect.signature(system.solve).parameters:
        solve_arguments["raise_on_failure"] = False
    result = system.solve(np.asarray(inputs, dtype=np.float64), **solve_arguments)
    if isinstance(result, tuple):
        outputs, diagnostics = result[0], result[1]
    else:
        outputs, diagnostics = result, None
    outputs = np.asarray(outputs)
    if outputs.ndim == 1:
        outputs = outputs[None, :]
    converged = _diagnostic_mask(diagnostics, len(outputs))
    converged &= np.isfinite(outputs).all(axis=1)
    return outputs.astype(np.float32), converged, diagnostics


def sample_inputs(system: Any, n: int, seed: int, margin: float) -> np.ndarray:
    """Sample normalized coordinates, preferring the system's sampler."""

    sampler = getattr(system, "sample_inputs", None)
    if sampler is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0 + margin, 1.0 - margin, (n, system.input_dim)).astype(
            np.float32
        )
    try:
        values = sampler(n=n, seed=seed, margin=margin)
    except TypeError:
        try:
            values = sampler(n, seed, margin)
        except TypeError:
            values = sampler(n, np.random.default_rng(seed), margin)
    return np.asarray(values, dtype=np.float32)


def _solve_or_raise(system: Any, inputs: np.ndarray, label: str) -> np.ndarray:
    output, converged, _ = solve_checked(system, inputs)
    if not np.all(converged):
        count = int((~converged).sum())
        raise RuntimeError(f"{system.name}: {count}/{len(inputs)} solves failed for {label}")
    return output


@dataclass
class DatasetBundle:
    system_name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    geom_x_plus: np.ndarray
    geom_x_minus: np.ndarray
    geom_y_plus: np.ndarray
    geom_y_minus: np.ndarray
    geometry_step: float
    probe_x: np.ndarray
    probe_y: np.ndarray
    probe_directions: np.ndarray
    probe_steps: np.ndarray
    probe_y_plus: np.ndarray
    probe_y_minus: np.ndarray
    probe_valid: np.ndarray
    metadata: dict[str, Any]

    def training_arrays(self, sample_weights: np.ndarray | None = None) -> TrainingArrays:
        return TrainingArrays(
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            geom_x_plus=self.geom_x_plus,
            geom_x_minus=self.geom_x_minus,
            geom_y_plus=self.geom_y_plus,
            geom_y_minus=self.geom_y_minus,
            sample_weights=sample_weights,
            geometry_step=self.geometry_step,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, np.ndarray)
        }
        arrays["system_name"] = np.asarray(self.system_name)
        arrays["geometry_step"] = np.asarray(self.geometry_step)
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetBundle":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                system_name=str(payload["system_name"]),
                x_train=payload["x_train"],
                y_train=payload["y_train"],
                x_val=payload["x_val"],
                y_val=payload["y_val"],
                x_test=payload["x_test"],
                y_test=payload["y_test"],
                geom_x_plus=payload["geom_x_plus"],
                geom_x_minus=payload["geom_x_minus"],
                geom_y_plus=payload["geom_y_plus"],
                geom_y_minus=payload["geom_y_minus"],
                geometry_step=float(payload["geometry_step"]),
                probe_x=payload["probe_x"],
                probe_y=payload["probe_y"],
                probe_directions=payload["probe_directions"],
                probe_steps=payload["probe_steps"],
                probe_y_plus=payload["probe_y_plus"],
                probe_y_minus=payload["probe_y_minus"],
                probe_valid=payload["probe_valid"],
                metadata=json.loads(str(payload["metadata_json"])),
            )


def _adaptive_probe_solve(
    system: Any,
    centers: np.ndarray,
    directions: np.ndarray,
    initial_step: float,
    max_halvings: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_centers, n_directions, input_dim = directions.shape
    flat_centers = np.repeat(centers[:, None, :], n_directions, axis=1).reshape(-1, input_dim)
    flat_directions = directions.reshape(-1, input_dim)
    steps = np.full(len(flat_centers), initial_step, dtype=np.float32)
    valid = np.zeros(len(flat_centers), dtype=bool)
    y_plus: np.ndarray | None = None
    y_minus: np.ndarray | None = None

    for attempt in range(max_halvings + 1):
        pending = np.flatnonzero(~valid)
        if len(pending) == 0:
            break
        delta = steps[pending, None] * flat_directions[pending]
        plus, plus_ok, _ = solve_checked(system, flat_centers[pending] + delta)
        minus, minus_ok, _ = solve_checked(system, flat_centers[pending] - delta)
        if y_plus is None:
            output_dim = plus.shape[1]
            y_plus = np.full((len(flat_centers), output_dim), np.nan, dtype=np.float32)
            y_minus = np.full_like(y_plus, np.nan)
        success = plus_ok & minus_ok
        accepted = pending[success]
        y_plus[accepted] = plus[success]
        y_minus[accepted] = minus[success]
        valid[accepted] = True
        failed = pending[~success]
        steps[failed] *= 0.5

    assert y_plus is not None and y_minus is not None
    shape = (n_centers, n_directions, y_plus.shape[1])
    return (
        y_plus.reshape(shape),
        y_minus.reshape(shape),
        steps.reshape(n_centers, n_directions),
        valid.reshape(n_centers, n_directions),
    )


def generate_dataset_bundle(
    system: Any,
    train_size: int,
    val_size: int,
    test_size: int,
    probe_size: int,
    probe_directions: int,
    geometry_step: float,
    seed: int = 20260919,
) -> DatasetBundle:
    """Generate one deterministic split plus train/test geometry queries."""

    if probe_size > test_size:
        raise ValueError("probe_size cannot exceed test_size")
    # Fit on a slightly wider domain than the one used for evaluation.  This
    # removes the ordinary support-boundary error that would otherwise be
    # confounded with (and often anticorrelated with) local curvature.
    train_margin = 0.01
    evaluation_margin = max(0.15, geometry_step * 1.25 + 0.02)
    x_train = sample_inputs(system, train_size, seed, train_margin)
    x_val = sample_inputs(system, val_size, seed + 1, evaluation_margin)
    x_test = sample_inputs(system, test_size, seed + 2, evaluation_margin)
    expected_shapes = (
        (x_train, (train_size, system.input_dim)),
        (x_val, (val_size, system.input_dim)),
        (x_test, (test_size, system.input_dim)),
    )
    for coordinates, expected in expected_shapes:
        if coordinates.shape != expected:
            raise ValueError(
                f"{system.name}.sample_inputs returned {coordinates.shape}, expected {expected}"
            )
    y_train = _solve_or_raise(system, x_train, "training split")
    y_val = _solve_or_raise(system, x_val, "validation split")
    y_test = _solve_or_raise(system, x_test, "test split")

    rng = np.random.default_rng(seed + 17)
    train_direction = _normal_directions(rng, (train_size, system.input_dim))
    geom_x_plus = x_train + geometry_step * train_direction
    geom_x_minus = x_train - geometry_step * train_direction
    geom_y_plus = _solve_or_raise(system, geom_x_plus, "training +geometry")
    geom_y_minus = _solve_or_raise(system, geom_x_minus, "training -geometry")

    probe_x = x_test[:probe_size]
    probe_y = y_test[:probe_size]
    directions = _normal_directions(
        rng, (probe_size, probe_directions, system.input_dim)
    )
    probe_y_plus, probe_y_minus, probe_steps_array, valid = _adaptive_probe_solve(
        system, probe_x, directions, geometry_step
    )
    failure_rate = float(1.0 - valid.mean())
    metadata = {
        "seed": seed,
        "system_name": system.name,
        "input_dim": int(system.input_dim),
        "output_dim": int(system.output_dim),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "probe_size": probe_size,
        "probe_directions": probe_directions,
        "geometry_step": geometry_step,
        "train_margin": train_margin,
        "evaluation_margin": evaluation_margin,
        "probe_failure_rate": failure_rate,
    }
    if failure_rate > 0.01:
        raise RuntimeError(
            f"{system.name}: finite-difference probe failure rate {failure_rate:.2%} exceeds 1%"
        )
    return DatasetBundle(
        system_name=system.name,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        geom_x_plus=geom_x_plus,
        geom_x_minus=geom_x_minus,
        geom_y_plus=geom_y_plus,
        geom_y_minus=geom_y_minus,
        geometry_step=geometry_step,
        probe_x=probe_x,
        probe_y=probe_y,
        probe_directions=directions,
        probe_steps=probe_steps_array,
        probe_y_plus=probe_y_plus,
        probe_y_minus=probe_y_minus,
        probe_valid=valid,
        metadata=metadata,
    )


def target_probe_responses(
    bundle: DatasetBundle,
    standardizer: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized directional responses and per-center kappa."""

    center = standardizer.transform(bundle.probe_y)[:, None, :]
    plus = standardizer.transform(bundle.probe_y_plus.reshape(-1, bundle.y_train.shape[1]))
    minus = standardizer.transform(bundle.probe_y_minus.reshape(-1, bundle.y_train.shape[1]))
    plus = plus.reshape(bundle.probe_y_plus.shape)
    minus = minus.reshape(bundle.probe_y_minus.shape)
    delta = plus - 2.0 * center + minus
    response = np.linalg.norm(delta, axis=-1)
    curvature = response / np.maximum(bundle.probe_steps, 1e-8) ** 2
    curvature[~bundle.probe_valid] = np.nan
    kappa = np.nanmean(curvature, axis=1)
    return response.astype(np.float32), kappa.astype(np.float32)


def allocation_weights(
    bundle: DatasetBundle,
    mode: str,
    seed: int = 2027,
) -> np.ndarray:
    """Construct matched-mass sampling weights for the allocation intervention."""

    output_scale = Standardizer.fit(bundle.y_train)
    center = output_scale.transform(bundle.y_train)
    plus = output_scale.transform(bundle.geom_y_plus)
    minus = output_scale.transform(bundle.geom_y_minus)
    curvature = np.linalg.norm(
        (plus - 2.0 * center + minus) / bundle.geometry_step**2, axis=1
    )
    logged = np.log1p(curvature)
    low, high = np.quantile(logged, (0.05, 0.95))
    score = np.clip((logged - low) / max(high - low, 1e-8), 0.0, 1.0)
    if mode == "uniform":
        weights = np.ones_like(score)
    elif mode == "aligned":
        weights = 0.25 + score
    elif mode == "reverse":
        weights = 0.25 + (1.0 - score)
    elif mode == "shuffled":
        weights = 0.25 + np.random.default_rng(seed).permutation(score)
    else:
        raise ValueError("mode must be uniform, aligned, reverse, or shuffled")
    return (weights / weights.mean()).astype(np.float32)


def generate_scaling_pool(
    system: Any,
    path: str | Path,
    size: int = 1_000_000,
    seed: int = 314159,
    chunk_size: int = 50_000,
) -> Path:
    """Generate a nested million-sample pool without retaining it in RAM."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    all_inputs = sample_inputs(system, size, seed, margin=0.02)
    chunk_size = min(chunk_size, size)
    with h5py.File(destination, "w") as handle:
        x_store = handle.create_dataset(
            "x", shape=(size, system.input_dim), dtype="f4", chunks=(chunk_size, system.input_dim)
        )
        y_store = handle.create_dataset(
            "y", shape=(size, system.output_dim), dtype="f4", chunks=(chunk_size, system.output_dim)
        )
        for start in range(0, size, chunk_size):
            stop = min(start + chunk_size, size)
            x_chunk = all_inputs[start:stop]
            y_chunk = _solve_or_raise(system, x_chunk, f"scaling pool {start}:{stop}")
            x_store[start:stop] = x_chunk
            y_store[start:stop] = y_chunk
        handle.attrs["system_name"] = system.name
        handle.attrs["seed"] = seed
        handle.attrs["size"] = size
    return destination


def load_scaling_subset(
    path: str | Path, size: int, val_size: int = 5_000
) -> TrainingArrays:
    with h5py.File(path, "r") as handle:
        if size + val_size > len(handle["x"]):
            raise ValueError("requested subset exceeds scaling pool")
        x_train = handle["x"][:size]
        y_train = handle["y"][:size]
        x_val = handle["x"][size : size + val_size]
        y_val = handle["y"][size : size + val_size]
    return TrainingArrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
