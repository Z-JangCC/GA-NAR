"""Control predictors for the NRGD response-allocation study."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .router import allocation_loss


class ScalarCapacityRouter(nn.Module):
    """Predict total capacity but distribute it uniformly across modes."""

    def __init__(self, input_dimension: int, rank: int, *, hidden_width: int = 64) -> None:
        super().__init__()
        self.rank = int(rank)
        self.encoder = nn.Sequential(nn.Linear(input_dimension, hidden_width), nn.SiLU(), nn.Linear(hidden_width, hidden_width), nn.SiLU())
        self.capacity_head = nn.Linear(hidden_width, 1)

    def forward(self, standardized_input: torch.Tensor) -> torch.Tensor:
        capacity = torch.sigmoid(self.capacity_head(self.encoder(standardized_input))).squeeze(-1)
        fractions = torch.full((standardized_input.shape[0], self.rank + 1), 1.0 / (self.rank + 1), device=standardized_input.device)
        return torch.cat([(1.0 - capacity).unsqueeze(-1), capacity.unsqueeze(-1) * fractions], dim=-1)


def fit_scalar_capacity_router(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 0,
    max_epochs: int = 300,
    patience: int = 30,
    hidden_width: int = 128,
    split_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> ScalarCapacityRouter:
    x = np.asarray(inputs, dtype=np.float32); y = np.asarray(targets, dtype=np.float32)
    torch.manual_seed(int(seed))
    if split_indices is None:
        order = np.random.default_rng(seed).permutation(len(x))
        split = min(max(2, int(round(len(x) * 0.8))), len(x) - 1)
        train_idx, valid_idx = order[:split], order[split:]
    else:
        train_idx = np.asarray(split_indices[0], dtype=np.int64)
        valid_idx = np.asarray(split_indices[1], dtype=np.int64)
        if len(train_idx) < 2 or len(valid_idx) < 2 or np.intersect1d(train_idx, valid_idx).size:
            raise ValueError("split_indices must be disjoint train/validation indices")
    model = ScalarCapacityRouter(x.shape[1], y.shape[1] - 2, hidden_width=hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])), batch_size=min(128, len(train_idx)), shuffle=True, generator=torch.Generator().manual_seed(seed + 11))
    valid_x, valid_y = torch.from_numpy(x[valid_idx]), torch.from_numpy(y[valid_idx])
    best_state = None; best_loss = float("inf"); stale = 0
    for _ in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = allocation_loss(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): valid_loss = float(allocation_loss(model(valid_x), valid_y))
        if valid_loss < best_loss:
            best_loss = valid_loss; stale = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    model.validation_indices = valid_idx
    return model


def random_geometry_basis(input_dimension: int, rank: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(input_dimension, rank))
    return np.linalg.qr(matrix, mode="reduced")[0]


def constant_target_metrics(targets: np.ndarray) -> np.ndarray:
    mean = np.mean(np.asarray(targets, dtype=np.float64), axis=0)
    mean = mean / np.sum(mean)
    return np.repeat(mean[None, :], len(targets), axis=0).astype(np.float32)


__all__ = ["ScalarCapacityRouter", "fit_scalar_capacity_router", "random_geometry_basis", "constant_target_metrics"]
