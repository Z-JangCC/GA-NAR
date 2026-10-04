from __future__ import annotations

import torch
from torch import nn


class GANRLAllocationRouter(nn.Module):
    """Input-conditioned predictor of NRGD response allocations.

    This is an allocator, not a state surrogate and not a nonlinear expert.
    Its output ordering is ``[smooth, mode_1..mode_r, complement]`` and is
    guaranteed to lie on the probability simplex without a solver call.
    """

    def __init__(self, input_dimension: int, rank: int, *, sensitivity_basis: torch.Tensor | None = None, mode_eigenvalues: torch.Tensor | None = None, hidden_width: int = 64, depth: int = 2, geometry_supervision: bool = False) -> None:
        super().__init__()
        if input_dimension <= 0 or rank <= 0:
            raise ValueError("input_dimension and rank must be positive")
        self.input_dimension = int(input_dimension)
        self.rank = int(rank)
        self.geometry_supervision = bool(geometry_supervision)
        if sensitivity_basis is not None:
            basis = torch.as_tensor(sensitivity_basis, dtype=torch.float32)
            if basis.ndim != 2 or basis.shape[0] != input_dimension or basis.shape[1] > rank:
                raise ValueError("sensitivity_basis must have shape (input_dimension, geometry_rank<=rank)")
            basis = torch.linalg.qr(basis, mode="reduced").Q
            geometry_rank = basis.shape[1]
            eigenvalues = torch.ones(geometry_rank, dtype=torch.float32) if mode_eigenvalues is None else torch.as_tensor(mode_eigenvalues, dtype=torch.float32).reshape(-1)
            if eigenvalues.numel() != geometry_rank:
                raise ValueError("mode_eigenvalues must match sensitivity_basis columns")
            self.register_buffer("sensitivity_basis", basis)
            self.register_buffer("mode_eigenvalues", eigenvalues.clamp_min(0.0))
            # Curvature-derived allocation targets are often even in a mode
            # coordinate. Expose signed, absolute, and quadratic physical
            # coordinates explicitly instead of asking a shallow router to
            # rediscover all of them from raw x.
            feature_dimension = input_dimension + 5 * geometry_rank + 2
        else:
            feature_dimension = input_dimension
        self.uses_geometry = sensitivity_basis is not None
        layers: list[nn.Module] = [nn.Linear(feature_dimension, hidden_width), nn.SiLU()]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.SiLU()])
        self.encoder = nn.Sequential(*layers)
        self.capacity_head = nn.Linear(hidden_width, 1)
        self.allocation_head = nn.Linear(hidden_width, rank + 1)
        if self.geometry_supervision:
            # Predict the local geometry components explicitly.  The first
            # rank entries are retained mode strengths and the last is the
            # complement strength; fractions are derived from this positive
            # vector instead of being an unconstrained categorical head.
            self.geometry_component_head = nn.Linear(hidden_width, rank + 1)
            self.geometry_total_head = nn.Linear(hidden_width, 1)

    def forward(self, standardized_input: torch.Tensor, *, return_components: bool = False):
        features = standardized_input
        if self.uses_geometry:
            coordinates = standardized_input @ self.sensitivity_basis
            weighted_coordinates = coordinates * self.mode_eigenvalues.clamp_min(0.0).sqrt()
            complement = standardized_input - coordinates @ self.sensitivity_basis.T
            complement_norm = torch.linalg.vector_norm(complement, dim=-1, keepdim=True)
            features = torch.cat([
                standardized_input,
                coordinates,
                coordinates.abs(),
                coordinates.square(),
                weighted_coordinates,
                weighted_coordinates.square(),
                complement_norm,
                complement_norm.square(),
            ], dim=-1)
        hidden = self.encoder(features)
        total_capacity = torch.sigmoid(self.capacity_head(hidden)).squeeze(-1)
        if self.geometry_supervision:
            components = torch.nn.functional.softplus(self.geometry_component_head(hidden)) + 1e-8
            fractions = components / components.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            predicted_geometry_total = torch.nn.functional.softplus(self.geometry_total_head(hidden)).squeeze(-1)
        else:
            fractions = torch.softmax(self.allocation_head(hidden), dim=-1)
            components = None
            predicted_geometry_total = None
        weights = torch.cat([(1.0 - total_capacity).unsqueeze(-1), total_capacity.unsqueeze(-1) * fractions], dim=-1)
        if return_components:
            return {"total_capacity": total_capacity, "fractions": fractions, "weights": weights, "geometry_components": components, "geometry_total": predicted_geometry_total}
        return weights


class ModalProbeRouter(nn.Module):
    """Learned allocation map from a low-dimensional modal curvature sketch.

    The probe vector contains directional curvature energies measured along
    the frozen physical modes (and one complement/total-energy token).  This
    is a learned model, unlike the analytic geometry-cache ceiling; the probes
    are the information interface that makes local geometry identifiable.
    """

    def __init__(self, probe_dimension: int, rank: int, *, hidden_width: int = 128, depth: int = 3, physics_base: bool = True) -> None:
        super().__init__()
        self.probe_dimension = int(probe_dimension)
        self.rank = int(rank)
        self.physics_base = bool(physics_base)
        layers: list[nn.Module] = [nn.Linear(self.probe_dimension, hidden_width), nn.SiLU()]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.SiLU()])
        self.encoder = nn.Sequential(*layers)
        self.capacity_head = nn.Linear(hidden_width, 1)
        self.fraction_head = nn.Linear(hidden_width, rank + 1)
        if self.physics_base:
            # Start from the exact normalized probe-energy map.  Learning then
            # estimates only a residual calibration instead of destroying a
            # physically identifiable fraction signal at initialization.
            nn.init.zeros_(self.fraction_head.weight)
            nn.init.zeros_(self.fraction_head.bias)

    def forward(self, probe_features: torch.Tensor, *, return_components: bool = False):
        hidden = self.encoder(probe_features)
        capacity = torch.sigmoid(self.capacity_head(hidden)).squeeze(-1)
        logits = self.fraction_head(hidden)
        if self.physics_base:
            base = probe_features[..., :-1].clamp_min(1e-30)
            base = base / base.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            fractions = torch.softmax(torch.log(base) + logits, dim=-1)
        else:
            fractions = torch.softmax(logits, dim=-1)
        weights = torch.cat([(1.0 - capacity).unsqueeze(-1), capacity.unsqueeze(-1) * fractions], dim=-1)
        if return_components:
            return {"total_capacity": capacity, "fractions": fractions, "weights": weights}
        return weights


def allocation_loss(prediction: dict[str, torch.Tensor] | torch.Tensor, target_weights: torch.Tensor, *, geometry_targets: torch.Tensor | None = None, capacity_weight: float = 1.0, fraction_weight: float = 1.0, geometry_weight: float = 1.0) -> torch.Tensor:
    """Supervise conserved weights, with optional separate alpha/fraction terms."""
    if isinstance(prediction, torch.Tensor):
        return torch.mean((prediction - target_weights) ** 2)
    weights = prediction["weights"]
    target = target_weights.detach()
    loss = torch.mean((weights - target) ** 2)
    if target.shape[-1] > 1:
        target_alpha = 1.0 - target[..., 0]
        target_fraction = target[..., 1:] / target_alpha.clamp_min(1e-8).unsqueeze(-1)
        target_fraction = target_fraction / target_fraction.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = loss + float(capacity_weight) * torch.mean((prediction["total_capacity"] - target_alpha) ** 2)
        loss = loss + float(fraction_weight) * torch.mean((prediction["fractions"] - target_fraction) ** 2)
    if geometry_targets is not None:
        if prediction.get("geometry_components") is None or prediction.get("geometry_total") is None:
            raise ValueError("geometry_targets require geometry_supervision=True")
        geometry_target = geometry_targets.detach()
        loss = loss + float(geometry_weight) * (
            torch.mean((prediction["geometry_components"] - geometry_target[..., :-1]) ** 2)
            + torch.mean((prediction["geometry_total"] - geometry_target[..., -1]) ** 2)
        )
    return loss
