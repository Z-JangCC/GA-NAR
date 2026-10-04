from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.random import seed_torch
from .losses import local_nonlinearity_supervision_loss, state_prediction_mse, training_objective


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_mse: float
    train_history: tuple[dict, ...]
    checkpoint_path: str | None
    lambda_value: float


class Trainer:
    def __init__(self, *, learning_rate: float = 1e-3, weight_decay: float = 1e-4, prediction_batch_size: int = 256, local_target_batch_size: int = 64, max_epochs: int = 300, patience: int = 30, device: str = "cpu") -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prediction_batch_size = prediction_batch_size
        self.local_target_batch_size = local_target_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = torch.device(device)

    def fit(
        self,
        model: nn.Module,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        *,
        activity_x: np.ndarray | None = None,
        activity_target: np.ndarray | None = None,
        local_nonlinearity_loss_weight: float = 0.0,
        seed: int = 0,
        checkpoint_path: str | Path | None = None,
        save_epoch_checkpoints: bool = False,
    ) -> TrainingResult:
        seed_torch(seed)
        model = model.to(self.device).float()
        prediction_loader = DataLoader(TensorDataset(torch.as_tensor(train_x, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32)), batch_size=self.prediction_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 123))
        activity_loader = None
        if activity_x is not None and activity_target is not None and len(activity_x) > 0 and local_nonlinearity_loss_weight != 0:
            activity_loader = DataLoader(TensorDataset(torch.as_tensor(activity_x, dtype=torch.float32), torch.as_tensor(activity_target, dtype=torch.float32)), batch_size=self.local_target_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 321))
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32, device=self.device)
        validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.float32, device=self.device)
        history = []
        best_validation = float("inf")
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        activity_iterator = None
        if save_epoch_checkpoints and checkpoint_path is not None:
            checkpoint_dir = Path(checkpoint_path).with_suffix("")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"epoch": 0, "model_state": initial_state, "checkpoint": "initial"}, checkpoint_dir / "initial.pt")
        for epoch in range(self.max_epochs):
            model.train()
            activity_iterator = iter(activity_loader) if activity_loader is not None else None
            train_prediction_sum = 0.0
            train_activity_sum = 0.0
            num_batches = 0
            for batch_x, batch_y in prediction_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x)
                prediction_loss = state_prediction_mse(prediction, batch_y)
                activity_loss = None
                if activity_iterator is not None:
                    try:
                        activity_batch_x, activity_batch_target = next(activity_iterator)
                    except StopIteration:
                        activity_iterator = iter(activity_loader)
                        activity_batch_x, activity_batch_target = next(activity_iterator)
                    activity_batch_x = activity_batch_x.to(self.device)
                    activity_batch_target = activity_batch_target.to(self.device)
                    if hasattr(model, "gate"):
                        stem_hidden = model.stem(activity_batch_x)
                        coefficient = model.gate(stem_hidden)
                        activity_loss = local_nonlinearity_supervision_loss(coefficient, activity_batch_target)
                    else:
                        activity_loss = None
                objective = training_objective(prediction_loss, activity_loss, local_nonlinearity_loss_weight)
                objective.backward()
                optimizer.step()
                train_prediction_sum += float(prediction_loss.detach().cpu())
                train_activity_sum += float(activity_loss.detach().cpu()) if activity_loss is not None else 0.0
                num_batches += 1
            model.eval()
            with torch.no_grad():
                validation_mse = float(state_prediction_mse(model(validation_x_tensor), validation_y_tensor).cpu())
            record = {"epoch": epoch, "train_prediction_mse": train_prediction_sum / max(num_batches, 1), "train_activity_loss": train_activity_sum / max(num_batches, 1), "validation_mse": validation_mse}
            history.append(record)
            if save_epoch_checkpoints and checkpoint_path is not None:
                checkpoint_dir = Path(checkpoint_path).with_suffix("")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "checkpoint": "post_update"}, checkpoint_dir / f"epoch_{epoch:04d}.pt")
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(best_state)
        saved_path = None
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": best_epoch, "model_state": best_state, "validation_mse": best_validation}, path)
            saved_path = str(path)
        return TrainingResult(best_epoch, best_validation, tuple(history), saved_path, float(local_nonlinearity_loss_weight))


def initialize_paired_models(models: list[nn.Module], seed: int) -> None:
    """Copy the identical trainable initialization into paired GA variants."""
    if not models:
        return
    seed_torch(seed)
    reference = models[0]
    with torch.no_grad():
        reference_state = {name: value.detach().clone() for name, value in reference.named_parameters()}
        for model in models[1:]:
            names = dict(model.named_parameters())
            missing = set(reference_state) - set(names)
            if missing:
                raise ValueError(f"paired model is missing trainable tensors: {sorted(missing)}")
            for name, value in reference_state.items():
                if names[name].shape != value.shape:
                    raise ValueError(f"paired tensor shape differs for {name}")
                names[name].copy_(value)
