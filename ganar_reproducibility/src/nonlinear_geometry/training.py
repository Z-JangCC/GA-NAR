"""Deterministic, fixed-budget training used by every experiment family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        scale = np.asarray(values, dtype=np.float64).std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


@dataclass
class TrainingArrays:
    """Arrays required for supervised and local-geometry training.

    Inputs are normalized system coordinates. Targets remain in physical units;
    the trainer fits a target standardizer on the training split only.
    Geometry neighbours must be aligned row-for-row with ``x_train``.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    geom_x_plus: np.ndarray | None = None
    geom_x_minus: np.ndarray | None = None
    geom_y_plus: np.ndarray | None = None
    geom_y_minus: np.ndarray | None = None
    sample_weights: np.ndarray | None = None
    geometry_step: float = 0.08

    def validate(self) -> None:
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train must have equal length")
        if len(self.x_val) != len(self.y_val):
            raise ValueError("x_val and y_val must have equal length")
        geometry = (
            self.geom_x_plus,
            self.geom_x_minus,
            self.geom_y_plus,
            self.geom_y_minus,
        )
        if any(item is not None for item in geometry):
            if any(item is None for item in geometry):
                raise ValueError("all four geometry arrays must be supplied together")
            if any(len(item) != len(self.x_train) for item in geometry if item is not None):
                raise ValueError("geometry arrays must align with x_train")
        if self.geometry_step <= 0:
            raise ValueError("geometry_step must be positive")
        if self.sample_weights is not None:
            if len(self.sample_weights) != len(self.x_train):
                raise ValueError("sample_weights must align with x_train")
            if np.any(np.asarray(self.sample_weights) < 0) or not np.any(
                np.asarray(self.sample_weights) > 0
            ):
                raise ValueError("sample_weights must be non-negative with positive mass")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1500
    batch_size: int = 512
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    geometry_weight: float = 0.25
    geometry_warmup_fraction: float = 0.60
    router_weight: float = 0.05
    direction_weight: float = 0.0
    gradient_clip: float = 1.0
    eval_interval: int = 100
    use_amp: bool = True


@dataclass
class FitResult:
    model: nn.Module
    target_standardizer: Standardizer
    history: list[dict[str, float]]
    best_step: int
    best_val_loss: float
    elapsed_seconds: float
    device: str
    geometry_scale: float | None
    router_quantiles: tuple[float, float] | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _as_tensor(values: np.ndarray) -> Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)


def _router_targets(
    curvature: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    transformed = np.log1p(np.maximum(curvature, 0.0))
    low, high = np.quantile(transformed, (0.05, 0.95))
    if high - low < 1e-8:
        target = np.full_like(transformed, 0.5, dtype=np.float32)
    else:
        target = np.clip((transformed - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return target.reshape(-1, 1), (float(low), float(high))


def fit_model(
    model: nn.Module,
    arrays: TrainingArrays,
    config: TrainConfig,
    seed: int,
    device: str = "auto",
) -> FitResult:
    """Fit a model for a fixed number of optimizer updates.

    All architectures see the same sampled indices and update count for a given
    seed. The best validation checkpoint is returned, but early stopping is not
    used, keeping the training budget identical.
    """

    arrays.validate()
    if not 0.0 <= config.geometry_warmup_fraction < 1.0:
        raise ValueError("geometry_warmup_fraction must lie in [0, 1)")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    run_device = resolve_device(device)
    model = model.to(run_device)

    target_standardizer = Standardizer.fit(arrays.y_train)
    # Optional physics-informed residual parameterizations provide a strong,
    # fixed local linearization while the trainable network learns nonlinear
    # corrections.  The buffers are stored in physical units and converted
    # once to the train-only standardized target space.
    if hasattr(model, "set_physics_standardizer"):
        model.set_physics_standardizer(
            _as_tensor(target_standardizer.mean), _as_tensor(target_standardizer.scale)
        )
    y_train = target_standardizer.transform(arrays.y_train)
    y_val = target_standardizer.transform(arrays.y_val)
    x_train_t = _as_tensor(arrays.x_train)
    y_train_t = _as_tensor(y_train)
    x_val_t = _as_tensor(arrays.x_val)
    y_val_t = _as_tensor(y_val)

    has_geometry = arrays.geom_x_plus is not None
    geometry_scale: float | None = None
    router_quantiles: tuple[float, float] | None = None
    direction_target_t: Tensor | None = None
    if has_geometry:
        x_plus_t = _as_tensor(arrays.geom_x_plus)
        x_minus_t = _as_tensor(arrays.geom_x_minus)
        y_plus = target_standardizer.transform(arrays.geom_y_plus)
        y_minus = target_standardizer.transform(arrays.geom_y_minus)
        y_plus_t = _as_tensor(y_plus)
        y_minus_t = _as_tensor(y_minus)
        curvature = (y_plus - 2.0 * y_train + y_minus) / arrays.geometry_step**2
        geometry_scale = float(max(np.sqrt(np.mean(curvature**2)), 1e-6))
        curvature_norm = np.linalg.norm(curvature, axis=1)
        router_target, router_quantiles = _router_targets(curvature_norm)
        router_target_t = _as_tensor(router_target)
        direction_target = y_plus - y_minus
        direction_target /= np.maximum(
            np.linalg.norm(direction_target, axis=1, keepdims=True), 1e-6
        )
        direction_target_t = _as_tensor(direction_target)
        geometry_scale = float(max(geometry_scale, 1e-6))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.steps, 1), eta_min=config.learning_rate * 0.05
    )
    # The geometry term subtracts nearby predictions and divides by r^2.  FP16
    # can overflow/cancel here even when the ordinary prediction loss is safe.
    amp_enabled = bool(
        config.use_amp and run_device.type == "cuda" and config.geometry_weight == 0.0
    )
    # torch.cuda.amp is used for compatibility with the pinned PyTorch 2.0
    # environment as well as newer releases.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator(device="cpu").manual_seed(seed + 104729)
    sampling_weights = None
    if arrays.sample_weights is not None:
        sampling_weights = _as_tensor(np.asarray(arrays.sample_weights).reshape(-1))

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_step = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for step in range(1, config.steps + 1):
        batch_size = min(config.batch_size, len(x_train_t))
        if sampling_weights is None:
            index = torch.randint(
                0, len(x_train_t), (batch_size,), generator=generator
            )
        else:
            index = torch.multinomial(
                sampling_weights,
                batch_size,
                replacement=True,
                generator=generator,
            )
        xb = x_train_t[index].to(run_device, non_blocking=True)
        yb = y_train_t[index].to(run_device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        warmup_steps = int(config.steps * config.geometry_warmup_fraction)
        ramp_steps = max(1, int(config.steps * 0.10))
        if step <= warmup_steps:
            current_geometry_weight = 0.0
        else:
            current_geometry_weight = config.geometry_weight * min(
                1.0, (step - warmup_steps) / ramp_steps
            )

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction, aux = model(xb, return_aux=True)
            prediction_loss = torch.mean((prediction - yb) ** 2)
            geometry_loss = prediction_loss.new_zeros(())
            router_loss = prediction_loss.new_zeros(())
            direction_loss = prediction_loss.new_zeros(())
            if has_geometry and current_geometry_weight > 0:
                plus = x_plus_t[index].to(run_device, non_blocking=True)
                minus = x_minus_t[index].to(run_device, non_blocking=True)
                target_plus = y_plus_t[index].to(run_device, non_blocking=True)
                target_minus = y_minus_t[index].to(run_device, non_blocking=True)
                prediction_plus = model(plus)
                prediction_minus = model(minus)
                predicted_curvature = (
                    prediction_plus - 2.0 * prediction + prediction_minus
                ) / arrays.geometry_step**2
                target_curvature = (
                    target_plus - 2.0 * yb + target_minus
                ) / arrays.geometry_step**2
                geometry_loss = torch.mean(
                    ((predicted_curvature - target_curvature) / geometry_scale) ** 2
                )
                if config.direction_weight > 0 and direction_target_t is not None:
                    predicted_direction = prediction_plus - prediction_minus
                    predicted_direction = predicted_direction / torch.clamp(
                        torch.linalg.vector_norm(predicted_direction, dim=1, keepdim=True),
                        min=1e-6,
                    )
                    target_direction = direction_target_t[index].to(
                        run_device, non_blocking=True
                    )
                    direction_loss = torch.mean(
                        1.0 - torch.sum(predicted_direction * target_direction, dim=1)
                    )
            if has_geometry and "allocation" in aux and config.router_weight > 0:
                allocation_target = router_target_t[index].to(run_device, non_blocking=True)
                router_loss = torch.mean((aux["allocation"] - allocation_target) ** 2)
            loss = (
                prediction_loss
                + current_geometry_weight * geometry_loss
                + config.router_weight * router_loss
                + config.direction_weight * direction_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            model.eval()
            val_losses: list[Tensor] = []
            with torch.inference_mode():
                for start_idx in range(0, len(x_val_t), config.batch_size * 2):
                    stop_idx = start_idx + config.batch_size * 2
                    vx = x_val_t[start_idx:stop_idx].to(run_device)
                    vy = y_val_t[start_idx:stop_idx].to(run_device)
                    val_losses.append(torch.mean((model(vx) - vy) ** 2).detach().cpu())
            val_loss = float(torch.stack(val_losses).mean())
            model.train()
            record = {
                "step": float(step),
                "train_total": float(loss.detach().cpu()),
                "train_prediction": float(prediction_loss.detach().cpu()),
                "train_geometry": float(geometry_loss.detach().cpu()),
                "train_router": float(router_loss.detach().cpu()),
                "train_direction": float(direction_loss.detach().cpu()),
                "geometry_weight": float(current_geometry_weight),
                "val_mse": val_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            history.append(record)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(
        model=model,
        target_standardizer=target_standardizer,
        history=history,
        best_step=best_step,
        best_val_loss=best_val,
        elapsed_seconds=elapsed,
        device=str(run_device),
        geometry_scale=geometry_scale,
        router_quantiles=router_quantiles,
    )


def checkpoint_payload(
    result: FitResult,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a CPU-only, versionable checkpoint dictionary."""

    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in result.model.state_dict().items()
        },
        "target_standardizer": result.target_standardizer.to_dict(),
        "history": result.history,
        "best_step": result.best_step,
        "best_val_loss": result.best_val_loss,
        "elapsed_seconds": result.elapsed_seconds,
        "device": result.device,
        "geometry_scale": result.geometry_scale,
        "router_quantiles": result.router_quantiles,
        "metadata": metadata,
    }
