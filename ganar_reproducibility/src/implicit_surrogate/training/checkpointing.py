"""Checkpoint serialization helpers shared by formal training and recovery."""

from __future__ import annotations

from pathlib import Path

import torch


def save_state_dict(path: str | Path, model, *, epoch: int, checkpoint: str, validation_mse: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": int(epoch), "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "checkpoint": checkpoint}
    if validation_mse is not None:
        payload["validation_mse"] = float(validation_mse)
    torch.save(payload, path)
    return path


def load_state_dict(path: str | Path, model):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload


__all__ = ["save_state_dict", "load_state_dict"]
