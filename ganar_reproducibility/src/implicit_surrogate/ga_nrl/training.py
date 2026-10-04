from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .router import GANRLAllocationRouter, ModalProbeRouter, allocation_loss


def _fraction_metrics(predicted_fraction: torch.Tensor, target_fraction: torch.Tensor) -> dict[str, float | list[float]]:
    """Return scale-aware diagnostics for simplex mode fractions.

    ``mode_fraction_mse`` is useful in absolute units, but it can make a
    low-variance target look deceptively difficult.  The normalized error and
    R2 use the evaluation-set target variance only as a denominator; they do
    not change training or leak target information into the predictor.
    """
    residual = predicted_fraction - target_fraction
    mse = torch.mean(residual.square())
    variance = torch.mean((target_fraction - torch.mean(target_fraction, dim=0, keepdim=True)).square())
    normalized = mse / variance.clamp_min(1e-12)
    total_ss = torch.sum((target_fraction - torch.mean(target_fraction, dim=0, keepdim=True)).square())
    r2 = 1.0 - torch.sum(residual.square()) / total_ss.clamp_min(1e-12)
    target_centered = target_fraction - torch.mean(target_fraction, dim=0, keepdim=True)
    pred_centered = predicted_fraction - torch.mean(predicted_fraction, dim=0, keepdim=True)
    covariance = torch.sum(target_centered * pred_centered, dim=0)
    denom = torch.sqrt(torch.sum(target_centered.square(), dim=0) * torch.sum(pred_centered.square(), dim=0)).clamp_min(1e-12)
    correlations = covariance / denom
    valid_corr = torch.isfinite(correlations)
    mean_corr = torch.mean(correlations[valid_corr]) if torch.any(valid_corr) else torch.tensor(0.0, device=target_fraction.device)
    return {
        "mode_fraction_mse": float(mse),
        "mode_fraction_normalized_mse": float(normalized),
        "mode_fraction_r2": float(r2),
        "mode_fraction_mean_correlation": float(mean_corr),
        # Keep the per-mode values in addition to the aggregate so a low
        # aggregate correlation cannot hide a mode-specific failure.  NaN is
        # retained for a constant target/prediction mode, where correlation
        # is mathematically undefined.
        "mode_fraction_correlations": [
            float(value) if torch.isfinite(value) else float("nan")
            for value in correlations.detach().cpu()
        ],
    }


@dataclass(frozen=True)
class AllocationTrainingResult:
    router: GANRLAllocationRouter
    train_loss: float
    validation_loss: float
    epochs: int
    status: str
    validation_indices: np.ndarray | None = None


def fit_modal_probe_router(
    probe_features: np.ndarray,
    target_weights: np.ndarray,
    *,
    seed: int = 0,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    hidden_width: int = 128,
    depth: int = 3,
    fraction_weight: float = 10.0,
) -> AllocationTrainingResult:
    """Fit a learned router from low-dimensional modal curvature probes."""
    x = np.asarray(probe_features, dtype=np.float32)
    y = np.asarray(target_weights, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 3:
        raise ValueError("probe_features and target_weights have incompatible shapes")
    torch.manual_seed(int(seed))
    if split_indices is None:
        train_idx, valid_idx = make_train_validation_indices(len(x), seed=seed)
    else:
        train_idx, valid_idx = np.asarray(split_indices[0]), np.asarray(split_indices[1])
    model = ModalProbeRouter(x.shape[1], y.shape[1] - 2, hidden_width=hidden_width, depth=depth, physics_base=True)
    # The probe-energy fractions are an identifiable physical sufficient
    # statistic. Keep that base path exact; learn the capacity calibration and
    # any shared representation only. A separate residual-calibration mode
    # can be enabled later, but must be regularized against this ceiling.
    for parameter in model.fraction_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=min(128, len(train_idx)), shuffle=True, generator=torch.Generator().manual_seed(seed + 11))
    vx, vy = torch.from_numpy(x[valid_idx]), torch.from_numpy(y[valid_idx])
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(400):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = allocation_loss(model(bx, return_components=True), by, fraction_weight=fraction_weight)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): value = float(allocation_loss(model(vx, return_components=True), vy, fraction_weight=fraction_weight))
        if value < best_loss:
            best_loss = value; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 40: break
    if best_state is None: return AllocationTrainingResult(model, float("inf"), float("inf"), epoch + 1, "UNRESOLVED", valid_idx)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): train_loss = float(allocation_loss(model(torch.from_numpy(x[train_idx]), return_components=True), torch.from_numpy(y[train_idx]), fraction_weight=fraction_weight))
    return AllocationTrainingResult(model, train_loss, best_loss, epoch + 1, "RESOLVED", valid_idx)


def fit_best_fraction_router(
    inputs: np.ndarray,
    target_weights: np.ndarray,
    *,
    seed: int = 0,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    sensitivity_basis: np.ndarray | None = None,
    mode_eigenvalues: np.ndarray | None = None,
) -> AllocationTrainingResult:
    """Select router capacity by held-out fraction error only.

    The candidate choice is made on the validation split, never on the test
    rows.  Fraction emphasis is intentional: the original combined loss was
    dominated by total allocation/capacity and left the indirect ``x -> B``
    signal under-trained.
    """
    candidates = ((128, 2), (256, 3), (256, 4), (512, 3))
    best_fit = None
    best_fraction = float("inf")
    for hidden_width, depth in candidates:
        fit = fit_allocation_router(
            inputs,
            target_weights,
            hidden_width=hidden_width,
            depth=depth,
            max_epochs=300,
            patience=30,
            seed=seed,
            sensitivity_basis=sensitivity_basis,
            mode_eigenvalues=mode_eigenvalues,
            split_indices=split_indices,
            fraction_weight=10.0,
            validation_metric="fraction",
        )
        valid_idx = fit.validation_indices if fit.validation_indices is not None else np.arange(len(inputs))
        metrics = evaluate_allocation_router(fit.router, np.asarray(inputs)[valid_idx], np.asarray(target_weights)[valid_idx])
        if metrics["mode_fraction_mse"] < best_fraction:
            best_fraction = float(metrics["mode_fraction_mse"])
            best_fit = fit
    if best_fit is None:
        raise RuntimeError("fraction-router candidate selection produced no fit")
    return best_fit


def fit_best_allocation_router(
    inputs: np.ndarray,
    target_weights: np.ndarray,
    *,
    seed: int = 0,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    sensitivity_basis: np.ndarray | None = None,
    mode_eigenvalues: np.ndarray | None = None,
) -> AllocationTrainingResult:
    """Select a router using held-out complete-allocation MSE only.

    This is a fair model-selection procedure for Table 4: all candidates use
    the same train/validation split, no held-out labels outside that split are
    consulted, and the final metrics are still computed on the fixed held-out
    rows.  The broader candidate set is useful for both x-only and finite
    geometry-sketch interfaces, whose optimal capacity/fraction trade-off is
    not necessarily the same.
    """
    candidates = ((128, 2, 1.0), (256, 3, 1.0), (512, 3, 1.0),
                  (128, 2, 10.0), (256, 3, 10.0), (256, 4, 20.0),
                  (512, 3, 10.0))
    best_fit = None
    best_mse = float("inf")
    for hidden_width, depth, fraction_weight in candidates:
        fit = fit_allocation_router(
            inputs,
            target_weights,
            hidden_width=hidden_width,
            depth=depth,
            max_epochs=300,
            patience=30,
            seed=seed,
            sensitivity_basis=sensitivity_basis,
            mode_eigenvalues=mode_eigenvalues,
            split_indices=split_indices,
            fraction_weight=fraction_weight,
            validation_metric="loss",
        )
        valid_idx = fit.validation_indices if fit.validation_indices is not None else np.arange(len(inputs))
        metrics = evaluate_allocation_router(fit.router, np.asarray(inputs)[valid_idx], np.asarray(target_weights)[valid_idx])
        if float(metrics["mse"]) < best_mse:
            best_mse = float(metrics["mse"])
            best_fit = fit
    if best_fit is None:
        raise RuntimeError("allocation-router candidate selection produced no fit")
    return best_fit


def make_train_validation_indices(
    count: int,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible disjoint train/validation partition.

    Keeping this split helper public lets target normalization, model fitting,
    and baseline estimation use exactly the same training rows.  In
    particular, physical target reference statistics must never be computed
    from the held-out rows.
    """
    if count < 4 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("count must be >= 4 and validation_fraction in (0, 1)")
    order = np.random.default_rng(int(seed)).permutation(int(count))
    split = min(max(2, int(round(count * (1.0 - validation_fraction)))), count - 1)
    return order[:split], order[split:]


def fit_allocation_router(
    inputs: np.ndarray,
    target_weights: np.ndarray,
    *,
    hidden_width: int = 64,
    depth: int = 2,
    validation_fraction: float = 0.2,
    max_epochs: int = 200,
    patience: int = 25,
    learning_rate: float = 1e-3,
    seed: int = 0,
    sensitivity_basis: np.ndarray | None = None,
    mode_eigenvalues: np.ndarray | None = None,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
    geometry_targets: np.ndarray | None = None,
    geometry_supervision: bool = False,
    capacity_weight: float = 1.0,
    fraction_weight: float = 1.0,
    geometry_weight: float = 1.0,
    validation_metric: str = "loss",
) -> AllocationTrainingResult:
    """Fit the NRGD allocation-fidelity predictor against frozen targets."""
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(target_weights, dtype=np.float32)
    g = None if geometry_targets is None else np.asarray(geometry_targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 4:
        raise ValueError("inputs and target_weights must be matching matrices with at least four rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("allocator training data must be finite")
    if validation_metric not in {"loss", "fraction"}:
        raise ValueError("validation_metric must be 'loss' or 'fraction'")
    if geometry_supervision and (g is None or g.ndim != 2 or len(g) != len(x) or g.shape[1] != y.shape[1]):
        raise ValueError("geometry_supervision requires geometry_targets with rank+2 columns")
    torch.manual_seed(int(seed))
    if split_indices is None:
        train_idx, valid_idx = make_train_validation_indices(len(x), validation_fraction=validation_fraction, seed=seed)
    else:
        train_idx = np.asarray(split_indices[0], dtype=np.int64)
        valid_idx = np.asarray(split_indices[1], dtype=np.int64)
        if len(train_idx) < 2 or len(valid_idx) < 2 or np.intersect1d(train_idx, valid_idx).size:
            raise ValueError("split_indices must be disjoint train/validation indices")
        if np.any(np.concatenate([train_idx, valid_idx]) < 0) or np.any(np.concatenate([train_idx, valid_idx]) >= len(x)):
            raise ValueError("split_indices contain out-of-range indices")
    if y.shape[1] < 3:
        raise ValueError("target_weights must contain smooth, at least one mode, and complement")
    model = GANRLAllocationRouter(x.shape[1], y.shape[1] - 2, sensitivity_basis=sensitivity_basis, mode_eigenvalues=mode_eigenvalues, hidden_width=hidden_width, depth=depth, geometry_supervision=geometry_supervision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    tensors = [torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])]
    if geometry_supervision:
        tensors.append(torch.from_numpy(g[train_idx]))
    train_loader = DataLoader(TensorDataset(*tensors), batch_size=min(128, len(train_idx)), shuffle=True, generator=torch.Generator().manual_seed(seed + 11))
    valid_x, valid_y = torch.from_numpy(x[valid_idx]), torch.from_numpy(y[valid_idx])
    valid_g = None if not geometry_supervision else torch.from_numpy(g[valid_idx])
    best_state, best_valid, stale, epochs_run = None, float("inf"), 0, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch_x, batch_y = batch[:2]
            batch_g = batch[2] if geometry_supervision else None
            optimizer.zero_grad(set_to_none=True)
            loss = allocation_loss(model(batch_x, return_components=True), batch_y, geometry_targets=batch_g, capacity_weight=capacity_weight, fraction_weight=fraction_weight, geometry_weight=geometry_weight)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_prediction = model(valid_x, return_components=True)
            if validation_metric == "fraction":
                target_capacity = 1.0 - valid_y[:, 0]
                target_fraction = valid_y[:, 1:] / target_capacity.clamp_min(1e-8).unsqueeze(-1)
                target_fraction = target_fraction / target_fraction.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                valid_loss = float(torch.mean((valid_prediction["fractions"] - target_fraction) ** 2))
            else:
                valid_loss = float(allocation_loss(valid_prediction, valid_y, geometry_targets=valid_g, capacity_weight=capacity_weight, fraction_weight=fraction_weight, geometry_weight=geometry_weight))
        epochs_run = epoch + 1
        if valid_loss < best_valid:
            best_valid, stale = valid_loss, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        return AllocationTrainingResult(model, float("inf"), float("inf"), epochs_run, "UNRESOLVED", valid_idx)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss = float(allocation_loss(model(torch.from_numpy(x[train_idx]), return_components=True), torch.from_numpy(y[train_idx]), geometry_targets=None if not geometry_supervision else torch.from_numpy(g[train_idx]), capacity_weight=capacity_weight, fraction_weight=fraction_weight, geometry_weight=geometry_weight))
    return AllocationTrainingResult(model, train_loss, best_valid, epochs_run, "RESOLVED", valid_idx)


def evaluate_allocation_router(router: GANRLAllocationRouter, inputs: np.ndarray, target_weights: np.ndarray) -> dict[str, float | bool | list[float]]:
    x = torch.as_tensor(inputs, dtype=torch.float32)
    target = torch.as_tensor(target_weights, dtype=torch.float32)
    router.eval()
    with torch.no_grad():
        prediction = router(x)
    error = torch.mean((prediction - target) ** 2).item()
    target_capacity = 1.0 - target[:, 0]
    predicted_capacity = 1.0 - prediction[:, 0]
    target_fraction = target[:, 1:] / target_capacity.clamp_min(1e-8).unsqueeze(-1)
    predicted_fraction = prediction[:, 1:] / predicted_capacity.clamp_min(1e-8).unsqueeze(-1)
    conservation = torch.max(torch.abs(prediction.sum(dim=-1) - 1.0)).item()
    metrics = _fraction_metrics(predicted_fraction, target_fraction)
    return {"mse": float(error), "capacity_mse": float(torch.mean((predicted_capacity - target_capacity) ** 2)), **metrics, "max_conservation_error": float(conservation), "conservation_pass": bool(conservation <= 1e-6), "mean_target_capacity": float(torch.mean(target_capacity))}


def evaluate_allocation_predictions(prediction: np.ndarray, target_weights: np.ndarray) -> dict[str, float | bool | list[float]]:
    target = torch.as_tensor(target_weights, dtype=torch.float32)
    value = torch.as_tensor(prediction, dtype=torch.float32)
    target_capacity = 1.0 - target[:, 0]
    predicted_capacity = 1.0 - value[:, 0]
    target_fraction = target[:, 1:] / target_capacity.clamp_min(1e-8).unsqueeze(-1)
    predicted_fraction = value[:, 1:] / predicted_capacity.clamp_min(1e-8).unsqueeze(-1)
    conservation = torch.max(torch.abs(value.sum(dim=-1) - 1.0)).item()
    metrics = _fraction_metrics(predicted_fraction, target_fraction)
    return {"mse": float(torch.mean((value - target) ** 2)), "capacity_mse": float(torch.mean((predicted_capacity - target_capacity) ** 2)), **metrics, "max_conservation_error": float(conservation), "conservation_pass": bool(conservation <= 1e-6), "mean_target_capacity": float(torch.mean(target_capacity))}
