"""Deterministic evaluation metrics for nonlinear-geometry experiments.

Prediction errors are reported in physical target units after undoing the
training standardization.  Dimensionless errors divide each output coordinate
by its *training-set* scale.  Geometry metrics use finite central differences,
which treats smooth and piecewise-linear models on the same footing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from torch import Tensor, nn


FloatArray = NDArray[np.float64]


class InverseTransformer(Protocol):
    """The part of :class:`training.Standardizer` needed by prediction."""

    def inverse_transform(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PointwiseErrorMetrics:
    normalized_mae: FloatArray
    normalized_rmse: FloatArray


@dataclass(frozen=True)
class OverallErrorMetrics:
    mae: float
    rmse: float
    normalized_mae: float
    nrmse: float


@dataclass(frozen=True)
class ResidualNormMetrics:
    pointwise: FloatArray
    mean: float
    rms: float
    median: float
    maximum: float


@dataclass(frozen=True)
class QuantileErrorMetrics:
    label: str
    count: int
    kappa_min: float
    kappa_median: float
    kappa_max: float
    errors: OverallErrorMetrics


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    pvalue: float
    n: int


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float
    confidence_level: float
    n_resamples: int


@dataclass(frozen=True)
class FiniteResponseMetrics:
    """Per-point, per-direction central-difference geometry quantities.

    Arrays ending in ``_response`` or ``_gap`` have shape ``(N, Q)``.  Response
    vectors have shape ``(N, Q, output_dim)``.  Kappa is the mean directional
    response divided by the squared finite-difference step and has shape ``N``.
    """

    true_vectors: FloatArray
    model_vectors: FloatArray
    true_response: FloatArray
    model_response: FloatArray
    absolute_gap: FloatArray
    relative_gap: FloatArray
    vector_gap: FloatArray
    relative_vector_gap: FloatArray
    true_kappa: FloatArray
    model_kappa: FloatArray
    epsilon: float


@dataclass(frozen=True)
class MagnitudeMatchMetrics:
    ratio: FloatArray
    epsilon: float
    median_ratio: float
    geometric_mean_ratio: float
    median_absolute_log_ratio: float
    underrepresented_fraction: float


@dataclass(frozen=True)
class SpatialAllocationMetrics:
    correlation: SpearmanResult
    q1_mean: float
    q5_mean: float
    q5_q1_concentration: float


@dataclass(frozen=True)
class DirectionalAlignmentMetrics:
    per_point: FloatArray
    principal_cosines: FloatArray
    mean: float
    median: float
    valid_count: int
    top_k: int


@dataclass(frozen=True)
class DeltaGapErrorMetrics:
    gap_reduction: FloatArray
    error_reduction: FloatArray
    correlation: SpearmanResult
    reference_index: int


@dataclass(frozen=True)
class CurvatureTailMetrics:
    """Joint high-curvature diagnostics for prediction and representation.

    The tail is selected from the *true* local curvature, so models cannot
    change the subgroup definition.  ``gap`` is the pointwise mean relative
    second-order response gap and ``error`` is pointwise normalized RMSE.
    """

    quantile: float
    threshold: float
    count: int
    gap_mean: float
    gap_median: float
    error_mean: float
    error_median: float
    gap_to_error_spearman: SpearmanResult
    tail_to_global_gap_ratio: float
    tail_to_global_error_ratio: float
    absolute_gap_mean: float
    log_magnitude_mismatch: float
    tail_absolute_gap_mean: float


def _to_numpy(values: ArrayLike | Tensor) -> np.ndarray:
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _observation_matrix(values: ArrayLike | Tensor, name: str) -> FloatArray:
    array = np.asarray(_to_numpy(values), dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have an observation axis")
    if array.ndim == 1:
        array = array[:, None]
    else:
        array = array.reshape(array.shape[0], -1)
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scale_vector(scale: ArrayLike, output_dim: int) -> FloatArray:
    result = np.asarray(scale, dtype=np.float64).reshape(-1)
    if result.size == 1:
        result = np.full(output_dim, float(result[0]), dtype=np.float64)
    if result.size != output_dim:
        raise ValueError(
            f"normalization_scale has {result.size} entries, expected {output_dim}"
        )
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("normalization_scale must be finite and strictly positive")
    return result


def _model_device(model: nn.Module) -> torch.device:
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    return torch.device("cpu")


def batched_predict(
    model: nn.Module,
    inputs: ArrayLike | Tensor,
    target_standardizer: InverseTransformer | None = None,
    *,
    batch_size: int = 4096,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Run deterministic batched inference and optionally undo target scaling.

    All leading input axes are preserved.  For example, ``(N, Q, input_dim)``
    inputs produce ``(N, Q, output_dim)`` predictions.  The model's training
    state is restored on return.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    run_device = torch.device(device) if device is not None else _model_device(model)
    if isinstance(inputs, Tensor):
        if inputs.ndim < 2:
            raise ValueError("inputs must have a feature axis")
        leading_shape = tuple(inputs.shape[:-1])
        flat = inputs.reshape(-1, inputs.shape[-1])
    else:
        array = np.asarray(inputs)
        if array.ndim < 2:
            raise ValueError("inputs must have a feature axis")
        leading_shape = array.shape[:-1]
        flat = np.ascontiguousarray(array.reshape(-1, array.shape[-1]))
    if len(flat) == 0:
        raise ValueError("inputs must not be empty")

    outputs: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(flat), batch_size):
                batch = flat[start : start + batch_size]
                tensor = torch.as_tensor(batch, dtype=torch.float32, device=run_device)
                prediction = model(tensor)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                if not isinstance(prediction, Tensor):
                    raise TypeError("model must return a Tensor or (Tensor, aux) tuple")
                outputs.append(prediction.detach().cpu().numpy())
    finally:
        model.train(was_training)

    result = np.concatenate(outputs, axis=0)
    if target_standardizer is not None:
        result = np.asarray(target_standardizer.inverse_transform(result))
    return result.reshape(*leading_shape, result.shape[-1])


def pointwise_normalized_errors(
    y_true: ArrayLike | Tensor,
    y_pred: ArrayLike | Tensor,
    normalization_scale: ArrayLike,
) -> PointwiseErrorMetrics:
    """Return dimension-normalized MAE and RMSE for every observation."""

    truth = _observation_matrix(y_true, "y_true")
    prediction = _observation_matrix(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    scale = _scale_vector(normalization_scale, truth.shape[1])
    normalized = (prediction - truth) / scale
    return PointwiseErrorMetrics(
        normalized_mae=np.mean(np.abs(normalized), axis=1),
        normalized_rmse=np.sqrt(np.mean(np.square(normalized), axis=1)),
    )


def overall_errors(
    y_true: ArrayLike | Tensor,
    y_pred: ArrayLike | Tensor,
    normalization_scale: ArrayLike,
) -> OverallErrorMetrics:
    """Compute physical MAE/RMSE and training-scale-normalized MAE/NRMSE."""

    truth = _observation_matrix(y_true, "y_true")
    prediction = _observation_matrix(y_pred, "y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    error = prediction - truth
    scale = _scale_vector(normalization_scale, truth.shape[1])
    normalized = error / scale
    return OverallErrorMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        normalized_mae=float(np.mean(np.abs(normalized))),
        nrmse=float(np.sqrt(np.mean(np.square(normalized)))),
    )


def residual_norms(
    system: Any,
    y: ArrayLike | Tensor,
    z: ArrayLike | Tensor,
    *,
    batch_size: int = 4096,
) -> FloatArray:
    """Evaluate ``system.residual(y, z)`` and return one L2 norm per point."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    y_array = np.asarray(_to_numpy(y))
    z_array = np.asarray(_to_numpy(z))
    if y_array.ndim == 0 or z_array.ndim == 0 or len(y_array) != len(z_array):
        raise ValueError("y and z must have equal non-empty observation axes")
    if len(y_array) == 0:
        raise ValueError("y and z must not be empty")
    result: list[np.ndarray] = []
    for start in range(0, len(y_array), batch_size):
        stop = start + batch_size
        residual = np.asarray(
            _to_numpy(system.residual(y_array[start:stop], z_array[start:stop])),
            dtype=np.float64,
        )
        expected = len(y_array[start:stop])
        if residual.ndim == 0 or residual.shape[0] != expected:
            raise ValueError("system.residual must preserve the observation axis")
        if residual.ndim == 1:
            norm = np.abs(residual)
        else:
            norm = np.linalg.norm(residual.reshape(expected, -1), axis=1)
        result.append(norm)
    return np.concatenate(result)


def summarize_residual_norms(norms: ArrayLike | Tensor) -> ResidualNormMetrics:
    values = np.asarray(_to_numpy(norms), dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("norms must be non-empty and finite")
    return ResidualNormMetrics(
        pointwise=values,
        mean=float(np.mean(values)),
        rms=float(np.sqrt(np.mean(np.square(values)))),
        median=float(np.median(values)),
        maximum=float(np.max(values)),
    )


def equal_frequency_bins(values: ArrayLike | Tensor, n_bins: int = 5) -> NDArray[np.int64]:
    """Assign stable, exactly rank-based bins numbered ``0 .. n_bins-1``."""

    array = np.asarray(_to_numpy(values), dtype=np.float64).reshape(-1)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if len(array) < n_bins:
        raise ValueError("at least one observation per bin is required")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    order = np.argsort(array, kind="stable")
    labels = np.empty(len(array), dtype=np.int64)
    labels[order] = np.floor(np.arange(len(array)) * n_bins / len(array)).astype(
        np.int64
    )
    return labels


def errors_by_kappa_quantile(
    kappa_true: ArrayLike | Tensor,
    y_true: ArrayLike | Tensor,
    y_pred: ArrayLike | Tensor,
    normalization_scale: ArrayLike,
    *,
    n_quantiles: int = 5,
) -> list[QuantileErrorMetrics]:
    """Report prediction error in equal-frequency true-kappa groups."""

    kappa = np.asarray(_to_numpy(kappa_true), dtype=np.float64).reshape(-1)
    truth = _observation_matrix(y_true, "y_true")
    prediction = _observation_matrix(y_pred, "y_pred")
    if len(kappa) != len(truth) or truth.shape != prediction.shape:
        raise ValueError("kappa_true, y_true and y_pred must align")
    if len(kappa) == 0:
        raise ValueError("kappa_true must not be empty")
    # Tiny diagnostic bundles can contain only one valid probe center.  In
    # that case a one-bin Q1 summary is the only defined quantile statistic;
    # do not fail the whole checkpoint re-evaluation because Q2--Q5 cannot be
    # formed.
    effective_bins = min(int(n_quantiles), len(kappa))
    if effective_bins < 2:
        return [
            QuantileErrorMetrics(
                label="Q1",
                count=int(len(kappa)),
                kappa_min=float(np.min(kappa)),
                kappa_median=float(np.median(kappa)),
                kappa_max=float(np.max(kappa)),
                errors=overall_errors(truth, prediction, normalization_scale),
            )
        ]
    labels = equal_frequency_bins(kappa, effective_bins)
    result: list[QuantileErrorMetrics] = []
    for index in range(effective_bins):
        selected = labels == index
        current = kappa[selected]
        result.append(
            QuantileErrorMetrics(
                label=f"Q{index + 1}",
                count=int(np.sum(selected)),
                kappa_min=float(np.min(current)),
                kappa_median=float(np.median(current)),
                kappa_max=float(np.max(current)),
                errors=overall_errors(
                    truth[selected], prediction[selected], normalization_scale
                ),
            )
        )
    return result


def _directional_outputs(
    center: ArrayLike | Tensor,
    plus: ArrayLike | Tensor,
    minus: ArrayLike | Tensor,
    prefix: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    center_array = np.asarray(_to_numpy(center), dtype=np.float64)
    plus_array = np.asarray(_to_numpy(plus), dtype=np.float64)
    minus_array = np.asarray(_to_numpy(minus), dtype=np.float64)
    if center_array.ndim == 1:
        center_array = center_array[:, None]
    if center_array.ndim != 2:
        raise ValueError(f"{prefix}_center must have shape (N, output_dim)")
    if plus_array.ndim == 2 and plus_array.shape == center_array.shape:
        plus_array = plus_array[:, None, :]
    elif plus_array.ndim == 2 and center_array.shape[1] == 1:
        plus_array = plus_array[:, :, None]
    if minus_array.ndim == 2 and minus_array.shape == center_array.shape:
        minus_array = minus_array[:, None, :]
    elif minus_array.ndim == 2 and center_array.shape[1] == 1:
        minus_array = minus_array[:, :, None]
    expected_tail = center_array.shape[1]
    if (
        plus_array.ndim != 3
        or minus_array.ndim != 3
        or plus_array.shape != minus_array.shape
        or plus_array.shape[0] != center_array.shape[0]
        or plus_array.shape[2] != expected_tail
    ):
        raise ValueError(
            f"{prefix}_plus/minus must have shape (N, Q, output_dim)"
        )
    if not (
        np.isfinite(center_array).all()
        and np.isfinite(plus_array).all()
        and np.isfinite(minus_array).all()
    ):
        raise ValueError(f"{prefix} response arrays must be finite")
    return center_array, plus_array, minus_array


def _step_matrix(step: float | ArrayLike, n: int, q: int) -> FloatArray:
    array = np.asarray(step, dtype=np.float64)
    if array.ndim == 0:
        result = np.full((n, q), float(array), dtype=np.float64)
    elif array.shape == (q,):
        result = np.broadcast_to(array[None, :], (n, q)).copy()
    elif array.shape == (n,) and q == 1:
        result = array[:, None]
    elif array.shape == (n, q):
        result = array
    else:
        raise ValueError("step must be scalar or have shape (Q,) or (N, Q)")
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("finite-difference steps must be finite and positive")
    return result


def _stable_epsilon(reference: ArrayLike, epsilon_ratio: float) -> float:
    if epsilon_ratio < 0.0:
        raise ValueError("epsilon_ratio must be non-negative")
    values = np.abs(np.asarray(reference, dtype=np.float64).reshape(-1))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("epsilon reference must contain finite values")
    median = float(np.median(values))
    numerical_floor = np.finfo(np.float64).eps * max(1.0, float(np.max(values)))
    return max(epsilon_ratio * median, numerical_floor)


def finite_response_metrics(
    true_center: ArrayLike | Tensor,
    true_plus: ArrayLike | Tensor,
    true_minus: ArrayLike | Tensor,
    model_center: ArrayLike | Tensor,
    model_plus: ArrayLike | Tensor,
    model_minus: ArrayLike | Tensor,
    *,
    step: float | ArrayLike = 1.0,
    epsilon_ratio: float = 0.01,
    epsilon_reference: ArrayLike | None = None,
) -> FiniteResponseMetrics:
    """Compare true and learned finite-scale nonlinear responses.

    ``epsilon`` equals ``epsilon_ratio`` times the median training/true response.
    Pass training responses through ``epsilon_reference`` when evaluating a test
    split; otherwise the supplied true responses provide the reference median.
    """

    tc, tp, tm = _directional_outputs(true_center, true_plus, true_minus, "true")
    mc, mp, mm = _directional_outputs(model_center, model_plus, model_minus, "model")
    if tc.shape != mc.shape or tp.shape != mp.shape:
        raise ValueError("true and model response arrays must have identical shapes")
    true_vectors = tp - 2.0 * tc[:, None, :] + tm
    model_vectors = mp - 2.0 * mc[:, None, :] + mm
    true_response = np.linalg.norm(true_vectors, axis=-1)
    model_response = np.linalg.norm(model_vectors, axis=-1)
    reference = true_response if epsilon_reference is None else epsilon_reference
    epsilon = _stable_epsilon(reference, epsilon_ratio)
    absolute_gap = np.abs(model_response - true_response)
    vector_gap = np.linalg.norm(model_vectors - true_vectors, axis=-1)
    denominator = true_response + epsilon
    steps = _step_matrix(step, true_response.shape[0], true_response.shape[1])
    return FiniteResponseMetrics(
        true_vectors=true_vectors,
        model_vectors=model_vectors,
        true_response=true_response,
        model_response=model_response,
        absolute_gap=absolute_gap,
        relative_gap=absolute_gap / denominator,
        vector_gap=vector_gap,
        relative_vector_gap=vector_gap / denominator,
        true_kappa=np.mean(true_response / np.square(steps), axis=1),
        model_kappa=np.mean(model_response / np.square(steps), axis=1),
        epsilon=epsilon,
    )


def model_finite_response_metrics(
    model: nn.Module,
    x_center: ArrayLike | Tensor,
    x_plus: ArrayLike | Tensor,
    x_minus: ArrayLike | Tensor,
    true_center: ArrayLike | Tensor,
    true_plus: ArrayLike | Tensor,
    true_minus: ArrayLike | Tensor,
    target_standardizer: InverseTransformer | None,
    *,
    step: float | ArrayLike = 1.0,
    epsilon_ratio: float = 0.01,
    epsilon_reference: ArrayLike | None = None,
    batch_size: int = 4096,
    device: str | torch.device | None = None,
) -> FiniteResponseMetrics:
    """Predict center/neighbour outputs in batches, then compare responses."""

    model_center = batched_predict(
        model, x_center, target_standardizer, batch_size=batch_size, device=device
    )
    model_plus = batched_predict(
        model, x_plus, target_standardizer, batch_size=batch_size, device=device
    )
    model_minus = batched_predict(
        model, x_minus, target_standardizer, batch_size=batch_size, device=device
    )
    return finite_response_metrics(
        true_center,
        true_plus,
        true_minus,
        model_center,
        model_plus,
        model_minus,
        step=step,
        epsilon_ratio=epsilon_ratio,
        epsilon_reference=epsilon_reference,
    )


def spearman_correlation(
    x: ArrayLike | Tensor, y: ArrayLike | Tensor
) -> SpearmanResult:
    """Spearman correlation after pairwise removal of non-finite observations."""

    x_array = np.asarray(_to_numpy(x), dtype=np.float64).reshape(-1)
    y_array = np.asarray(_to_numpy(y), dtype=np.float64).reshape(-1)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have identical lengths")
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[keep]
    y_array = y_array[keep]
    if len(x_array) < 2:
        return SpearmanResult(rho=float("nan"), pvalue=float("nan"), n=len(x_array))
    result = stats.spearmanr(x_array, y_array)
    return SpearmanResult(
        rho=float(result.statistic), pvalue=float(result.pvalue), n=len(x_array)
    )


def bootstrap_confidence_interval(
    *samples: ArrayLike | Tensor,
    statistic: Callable[..., float] | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Deterministic paired percentile-bootstrap confidence interval.

    Every sample is resampled with the same indices.  With one sample, the
    default statistic is its mean; multi-sample calls must provide a statistic.
    """

    if not samples:
        raise ValueError("at least one sample is required")
    arrays = [np.asarray(_to_numpy(sample)) for sample in samples]
    n = len(arrays[0]) if arrays[0].ndim > 0 else 0
    if n == 0 or any(array.ndim == 0 or len(array) != n for array in arrays):
        raise ValueError("all samples must share a non-empty first axis")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if statistic is None:
        if len(arrays) != 1:
            raise ValueError("a statistic is required for multiple samples")
        statistic = lambda values: float(np.mean(values))
    estimate = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        selected = rng.integers(0, n, size=n)
        estimates[index] = float(statistic(*(array[selected] for array in arrays)))
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(estimates, (alpha, 1.0 - alpha))
    return BootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )


def magnitude_matching(
    kappa_true: ArrayLike | Tensor,
    kappa_model: ArrayLike | Tensor,
    *,
    epsilon_ratio: float = 1e-8,
) -> MagnitudeMatchMetrics:
    """Summarize the stable pointwise magnitude ratio ``kappa_f/kappa_s``."""

    truth = np.asarray(_to_numpy(kappa_true), dtype=np.float64).reshape(-1)
    model = np.asarray(_to_numpy(kappa_model), dtype=np.float64).reshape(-1)
    if truth.shape != model.shape or len(truth) == 0:
        raise ValueError("kappa_true and kappa_model must have equal non-empty lengths")
    if not np.isfinite(truth).all() or not np.isfinite(model).all():
        raise ValueError("kappa values must be finite")
    if np.any(truth < 0.0) or np.any(model < 0.0):
        raise ValueError("kappa values must be non-negative")
    epsilon = _stable_epsilon(truth, epsilon_ratio)
    ratio = (model + epsilon) / (truth + epsilon)
    log_ratio = np.log(ratio)
    return MagnitudeMatchMetrics(
        ratio=ratio,
        epsilon=epsilon,
        median_ratio=float(np.median(ratio)),
        geometric_mean_ratio=float(np.exp(np.mean(log_ratio))),
        median_absolute_log_ratio=float(np.median(np.abs(log_ratio))),
        underrepresented_fraction=float(np.mean(ratio < 1.0)),
    )


def spatial_allocation_metrics(
    kappa_true: ArrayLike | Tensor,
    allocation: ArrayLike | Tensor,
) -> SpatialAllocationMetrics:
    """Measure whether nonlinear capacity is concentrated where kappa is high."""

    kappa = np.asarray(_to_numpy(kappa_true), dtype=np.float64).reshape(-1)
    capacity = np.asarray(_to_numpy(allocation), dtype=np.float64).reshape(-1)
    if kappa.shape != capacity.shape or len(kappa) < 5:
        raise ValueError("kappa_true and allocation must align and contain >=5 points")
    if not np.isfinite(kappa).all() or not np.isfinite(capacity).all():
        raise ValueError("kappa_true and allocation must be finite")
    bins = equal_frequency_bins(kappa, 5)
    q1 = float(np.mean(capacity[bins == 0]))
    q5 = float(np.mean(capacity[bins == 4]))
    floor = np.finfo(np.float64).eps * max(1.0, abs(q1), abs(q5))
    return SpatialAllocationMetrics(
        correlation=spearman_correlation(kappa, capacity),
        q1_mean=q1,
        q5_mean=q5,
        q5_q1_concentration=float((q5 + floor) / (q1 + floor)),
    )


def _response_matrix(
    directions: FloatArray, responses: FloatArray, weight_power: float
) -> FloatArray:
    direction_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(direction_norm <= 0.0):
        raise ValueError("directions must be non-zero")
    unit_directions = directions / direction_norm
    weights = np.abs(responses) ** weight_power
    return unit_directions * np.sqrt(weights)[..., None]


def _top_right_subspace(matrix: FloatArray, top_k: int) -> FloatArray | None:
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if len(singular_values) < top_k or singular_values[0] <= 0.0:
        return None
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    if singular_values[top_k - 1] <= tolerance:
        return None
    return right[:top_k].T


def topk_directional_alignment(
    directions: ArrayLike | Tensor,
    true_responses: ArrayLike | Tensor,
    model_responses: ArrayLike | Tensor,
    *,
    top_k: int = 3,
    weight_power: float = 2.0,
) -> DirectionalAlignmentMetrics:
    """Compare top-k nonlinear direction subspaces via principal angles.

    For each point, rows of the response-weighted matrix are
    ``|R(v)|**(weight_power/2) * v/||v||``.  Its top right-singular vectors span
    the dominant input directions.  The returned score is the mean squared
    principal cosine, ``||U_s.T @ U_f||_F^2 / k``, in ``[0, 1]``.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if weight_power <= 0.0:
        raise ValueError("weight_power must be positive")
    true = np.asarray(_to_numpy(true_responses), dtype=np.float64)
    model = np.asarray(_to_numpy(model_responses), dtype=np.float64)
    if true.ndim == 1:
        true = true[None, :]
    if model.ndim == 1:
        model = model[None, :]
    if true.ndim != 2 or true.shape != model.shape:
        raise ValueError("response arrays must have shape (N, Q) or (Q,)")
    direction_array = np.asarray(_to_numpy(directions), dtype=np.float64)
    if direction_array.ndim == 2:
        direction_array = np.broadcast_to(
            direction_array[None, :, :], (len(true), *direction_array.shape)
        )
    if (
        direction_array.ndim != 3
        or direction_array.shape[:2] != true.shape
        or not np.isfinite(direction_array).all()
        or not np.isfinite(true).all()
        or not np.isfinite(model).all()
    ):
        raise ValueError("directions must align with finite response arrays")
    if top_k > min(direction_array.shape[1], direction_array.shape[2]):
        raise ValueError("top_k exceeds the available sampled/input dimensions")

    true_matrix = _response_matrix(direction_array, true, weight_power)
    model_matrix = _response_matrix(direction_array, model, weight_power)
    scores = np.full(len(true), np.nan, dtype=np.float64)
    cosines = np.full((len(true), top_k), np.nan, dtype=np.float64)
    for index in range(len(true)):
        true_basis = _top_right_subspace(true_matrix[index], top_k)
        model_basis = _top_right_subspace(model_matrix[index], top_k)
        if true_basis is None or model_basis is None:
            continue
        current = np.linalg.svd(true_basis.T @ model_basis, compute_uv=False)
        current = np.clip(current, 0.0, 1.0)
        cosines[index] = current
        scores[index] = float(np.mean(np.square(current)))
    valid = np.isfinite(scores)
    return DirectionalAlignmentMetrics(
        per_point=scores,
        principal_cosines=cosines,
        mean=float(np.mean(scores[valid])) if np.any(valid) else float("nan"),
        median=float(np.median(scores[valid])) if np.any(valid) else float("nan"),
        valid_count=int(np.sum(valid)),
        top_k=top_k,
    )


def delta_gap_error_correlation(
    gaps: ArrayLike | Tensor,
    errors: ArrayLike | Tensor,
    *,
    reference_index: int = 0,
    exclude_reference: bool = True,
) -> DeltaGapErrorMetrics:
    """Correlate cross-model representation-gap and error reductions.

    Positive deltas mean an improvement relative to ``reference_index``.
    """

    gap_array = np.asarray(_to_numpy(gaps), dtype=np.float64).reshape(-1)
    error_array = np.asarray(_to_numpy(errors), dtype=np.float64).reshape(-1)
    if gap_array.shape != error_array.shape or len(gap_array) < 2:
        raise ValueError("gaps and errors must have equal lengths of at least two")
    if not -len(gap_array) <= reference_index < len(gap_array):
        raise IndexError("reference_index is out of range")
    reference_index %= len(gap_array)
    gap_reduction = gap_array[reference_index] - gap_array
    error_reduction = error_array[reference_index] - error_array
    if exclude_reference:
        keep = np.arange(len(gap_array)) != reference_index
        gap_reduction = gap_reduction[keep]
        error_reduction = error_reduction[keep]
    return DeltaGapErrorMetrics(
        gap_reduction=gap_reduction,
        error_reduction=error_reduction,
        correlation=spearman_correlation(gap_reduction, error_reduction),
        reference_index=reference_index,
    )


def curvature_tail_metrics(
    true_kappa: ArrayLike | Tensor,
    response_gap: ArrayLike | Tensor,
    prediction_error: ArrayLike | Tensor,
    *,
    quantile: float = 0.80,
    absolute_gap: ArrayLike | Tensor | None = None,
    log_magnitude_mismatch: ArrayLike | Tensor | None = None,
) -> CurvatureTailMetrics:
    """Summarize the coupled high-curvature failure mode.

    ``response_gap`` and ``prediction_error`` must be pointwise values aligned
    with ``true_kappa``.  The function is intentionally descriptive: it does
    not claim that gap causes error, but makes the proposed failure boundary
    directly auditable in every run.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    kappa = np.asarray(_to_numpy(true_kappa), dtype=np.float64).reshape(-1)
    gap = np.asarray(_to_numpy(response_gap), dtype=np.float64).reshape(-1)
    error = np.asarray(_to_numpy(prediction_error), dtype=np.float64).reshape(-1)
    if not (len(kappa) and kappa.shape == gap.shape == error.shape):
        raise ValueError("true_kappa, response_gap and prediction_error must align")
    if not (np.isfinite(kappa).all() and np.isfinite(gap).all() and np.isfinite(error).all()):
        raise ValueError("curvature-tail inputs must be finite")
    threshold = float(np.quantile(kappa, quantile))
    selected = kappa >= threshold
    if not np.any(selected):
        raise ValueError("curvature tail is empty")
    global_gap = max(float(np.mean(gap)), np.finfo(np.float64).eps)
    global_error = max(float(np.mean(error)), np.finfo(np.float64).eps)
    abs_gap = gap if absolute_gap is None else np.asarray(_to_numpy(absolute_gap), dtype=np.float64).reshape(-1)
    log_gap = np.log1p(gap) if log_magnitude_mismatch is None else np.asarray(_to_numpy(log_magnitude_mismatch), dtype=np.float64).reshape(-1)
    if abs_gap.shape != gap.shape or log_gap.shape != gap.shape:
        raise ValueError("absolute_gap and log_magnitude_mismatch must align")
    return CurvatureTailMetrics(
        quantile=float(quantile),
        threshold=threshold,
        count=int(np.sum(selected)),
        gap_mean=float(np.mean(gap[selected])),
        gap_median=float(np.median(gap[selected])),
        error_mean=float(np.mean(error[selected])),
        error_median=float(np.median(error[selected])),
        gap_to_error_spearman=spearman_correlation(gap[selected], error[selected]),
        tail_to_global_gap_ratio=float(np.mean(gap[selected]) / global_gap),
        tail_to_global_error_ratio=float(np.mean(error[selected]) / global_error),
        absolute_gap_mean=float(np.mean(abs_gap)),
        log_magnitude_mismatch=float(np.mean(np.abs(log_gap))),
        tail_absolute_gap_mean=float(np.mean(abs_gap[selected])),
    )


__all__: Sequence[str] = (
    "BootstrapCI",
    "DeltaGapErrorMetrics",
    "CurvatureTailMetrics",
    "DirectionalAlignmentMetrics",
    "FiniteResponseMetrics",
    "MagnitudeMatchMetrics",
    "OverallErrorMetrics",
    "PointwiseErrorMetrics",
    "QuantileErrorMetrics",
    "ResidualNormMetrics",
    "SpatialAllocationMetrics",
    "SpearmanResult",
    "batched_predict",
    "bootstrap_confidence_interval",
    "delta_gap_error_correlation",
    "curvature_tail_metrics",
    "equal_frequency_bins",
    "errors_by_kappa_quantile",
    "finite_response_metrics",
    "magnitude_matching",
    "model_finite_response_metrics",
    "overall_errors",
    "pointwise_normalized_errors",
    "residual_norms",
    "spatial_allocation_metrics",
    "spearman_correlation",
    "summarize_residual_norms",
    "topk_directional_alignment",
)
