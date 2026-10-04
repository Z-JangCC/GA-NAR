from __future__ import annotations

import torch


def state_prediction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def local_nonlinearity_supervision_loss(modulation_coefficient: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((modulation_coefficient.reshape(-1) - target.detach().reshape(-1)) ** 2)


def training_objective(prediction_loss: torch.Tensor, activity_loss: torch.Tensor | None, local_nonlinearity_loss_weight: float) -> torch.Tensor:
    if activity_loss is None or local_nonlinearity_loss_weight == 0:
        return prediction_loss
    return prediction_loss + float(local_nonlinearity_loss_weight) * activity_loss

