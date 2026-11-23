"""NRGD from finite-scale physical Jacobian-vector actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


JacobianAction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ResponseGeometry:
    projector: np.ndarray
    intensity: np.ndarray
    allocation: np.ndarray
    rank: int
    intensity_scale: float
    allocation_scale: float
    eigenvalues: np.ndarray


def _unit_vectors(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    vectors = rng.normal(size=(count, dimension))
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _simplex_projection(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    candidate = (np.cumsum(sorted_values) - 1.0) / np.arange(1, len(values) + 1)
    eligible = np.flatnonzero(sorted_values > candidate)
    threshold = candidate[eligible[-1]]
    return np.maximum(values - threshold, 0.0)


def build_response_geometry(
    centers: np.ndarray, jacobian_action: JacobianAction,
    *, scales: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
    hutchinson_probes: int = 8, directions_per_center: int = 8,
    subspace_probes: int = 4, response_mass: float = 0.9,
    seed: int = 0, intensity_scale: float | None = None,
    allocation_scale: float | None = None,
) -> ResponseGeometry:
    """Estimate NRGD targets on training-only, standardized operating points.

    ``jacobian_action(x, v)`` returns the physical solution Jacobian applied to
    ``v`` at standardized input ``x``. The caller is responsible for keeping
    all perturbed points on the selected regular solution branch.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or len(centers) < 4 or centers.shape[1] < 2:
        raise ValueError("centers must have shape (N>=4, d>=2)")
    if not np.isfinite(centers).all() or len(scales) < 2 or any(h <= 0 for h in scales):
        raise ValueError("invalid centers or scales")
    if any(n < 1 for n in (hutchinson_probes, directions_per_center, subspace_probes)):
        raise ValueError("probe counts must be positive")
    if not 0 < response_mass <= 1:
        raise ValueError("response mass must lie in (0, 1]")
    n_centers, dimension = centers.shape
    rng = np.random.default_rng(seed)
    directions = np.stack([
        _unit_vectors(rng, directions_per_center, dimension)
        for _ in centers
    ])
    probes = rng.choice((-1.0, 1.0), size=(n_centers, hutchinson_probes, dimension))

    def score(x: np.ndarray, direction: np.ndarray, h: float,
              test_vectors: np.ndarray) -> float:
        plus, minus = x + h * direction, x - h * direction
        responses = np.asarray([
            (np.asarray(jacobian_action(plus, v), dtype=np.float64) -
             np.asarray(jacobian_action(minus, v), dtype=np.float64)) / (2 * h)
            for v in test_vectors
        ])
        if not np.isfinite(responses).all():
            raise ValueError("physical Jacobian action is non-finite")
        return float(np.mean(np.sum(responses.reshape(len(test_vectors), -1) ** 2,
                                    axis=1)))

    scores = np.asarray([
        [[score(x, direction, h, probes[i]) for direction in directions[i]]
         for h in scales]
        for i, x in enumerate(centers)
    ])
    half = n_centers // 2
    selected = None
    for index in range(len(scales) - 1):
        large = scores[:, index].mean(axis=1)
        small = scores[:, index + 1].mean(axis=1)
        halves = (slice(0, half), slice(half, None))
        if all(large[part].mean() >= 1e-8 and small[part].mean() >= 1e-8 and
               2 * abs(large[part].mean() - small[part].mean()) /
               (large[part].mean() + small[part].mean()) <= 0.25
               for part in halves):
            selected = scales[index]
            break
    if selected is None:
        raise ValueError("finite-scale response is unresolved")
    h_rho = selected if intensity_scale is None else intensity_scale
    h_pi = selected if allocation_scale is None else allocation_scale
    if h_rho <= 0 or h_pi <= 0:
        raise ValueError("descriptor scales must be positive")

    def directional_scores(h: float) -> np.ndarray:
        if h in scales:
            return scores[:, scales.index(h)]
        return np.asarray([
            [score(x, direction, h, probes[i]) for direction in directions[i]]
            for i, x in enumerate(centers)
        ])

    response = directional_scores(h_pi)
    moment = np.einsum("nk,nki,nkj->ij", response, directions, directions) / (
        n_centers * directions_per_center
    )
    trace = float(np.trace(moment))
    if trace <= 0:
        raise ValueError("response moment has zero mass")
    normalized = moment / trace
    backprojected = ((dimension + 2) * normalized - np.eye(dimension)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh((backprojected + backprojected.T) / 2)
    eigenvalues = _simplex_projection(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    rank = int(np.searchsorted(np.cumsum(eigenvalues), response_mass) + 1)
    if rank == dimension:
        raise ValueError("DRS occupies the full input space; complement is empty")
    basis = eigenvectors[:, :rank]
    projector = basis @ basis.T

    local_total = dimension * directional_scores(h_rho).mean(axis=1)
    normalizer = float(local_total.mean())
    intensity = local_total / (local_total + normalizer)
    complement_projector = np.eye(dimension) - projector
    allocation = []
    for index, x in enumerate(centers):
        totals = []
        for projection, subspace_dim in ((projector, rank),
                                         (complement_projector, dimension - rank)):
            latent = _unit_vectors(rng, subspace_probes, dimension)
            projected = latent @ projection.T
            projected /= np.linalg.norm(projected, axis=1, keepdims=True)
            totals.append(subspace_dim * np.mean([
                score(x, direction, h_pi, probes[index]) for direction in projected
            ]))
        total = sum(totals)
        if total <= 0:
            raise ValueError("local subspace response has zero mass")
        allocation.append(np.asarray(totals) / total)
    return ResponseGeometry(projector, intensity, np.asarray(allocation), rank,
                            h_rho, h_pi, eigenvalues)
