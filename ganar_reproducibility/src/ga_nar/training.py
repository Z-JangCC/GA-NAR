"""Prediction and NRGD-target training for the layer-wise GA-NAR model."""

from __future__ import annotations

import copy

import numpy as np
import torch

from .data import Benchmark
from .model import GeometryAdaptiveSurrogate, ga_nar_loss


def predict_batches(model: GeometryAdaptiveSurrogate, x: np.ndarray,
                    batch_size: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], device=device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fit(model: GeometryAdaptiveSurrogate, data: Benchmark, *, seed: int = 0,
        epochs: int = 400, patience: int = 50, batch_size: int = 256,
        learning_rate: float = 1e-3, lambda_rho: float = 0.5,
        lambda_pi: float = 0.5, device: str = "cpu") -> dict:
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, patience, batch size and learning rate must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    x = torch.as_tensor(data.x, device=device)
    y = torch.as_tensor(data.y, device=device)
    geometry_x = torch.as_tensor(data.geometry_x, device=device)
    rho = torch.as_tensor(data.rho, device=device)
    pi = torch.as_tensor(data.pi, device=device)
    best_loss, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(data.train)
        geometry_order = rng.permutation(len(geometry_x))
        losses = []
        for offset in range(0, len(order), batch_size):
            ids = order[offset:offset + batch_size]
            geo_ids = geometry_order[(np.arange(len(ids)) + offset) % len(geometry_order)]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ids])
            _, geo_aux = model(geometry_x[geo_ids], return_aux=True)
            loss = ga_nar_loss(prediction, y[ids], geo_aux,
                               rho_target=rho[geo_ids], pi_target=pi[geo_ids],
                               lambda_rho=lambda_rho, lambda_pi=lambda_pi)
            if not torch.isfinite(loss):
                raise FloatingPointError("GA-NAR training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_prediction = predict_batches(model, data.x[data.validation], batch_size)
        val_mse = float(np.mean((validation_prediction - data.y[data.validation]) ** 2))
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)),
                        "validation_mse": val_mse})
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_mse": best_loss,
            "epochs_completed": len(history), "history": history}
