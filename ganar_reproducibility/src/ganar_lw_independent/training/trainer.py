from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def order_for(seed: int, epoch: int, n: int, namespace: str):
    raw = f"ganar-lw-independent|{namespace}|{seed}|{epoch}".encode()
    local = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return np.random.default_rng(local).permutation(n)


@dataclass
class TrainResult:
    best_val: float
    best_epoch: int
    epochs_completed: int
    wall_time: float
    history: list
    parameter_count: int
    device: str
    order_namespace: str


def _geometry_loss(output, rho_target, pi_target, lambda_rho, lambda_pi):
    _, rho, pi, _, _ = output
    loss = torch.zeros((), dtype=rho.dtype, device=rho.device)
    if rho_target is not None:
        loss = loss + lambda_rho * torch.mean((rho - rho_target) ** 2)
    if pi_target is not None:
        # Hellinger squared, with explicit simplex clipping.
        loss = loss + lambda_pi * torch.mean(
            1.0 - torch.sum(torch.sqrt(torch.clamp(pi, 0, 1) *
                                        torch.clamp(pi_target, 0, 1)), dim=-1))
    return loss


def train_model(model: nn.Module, x_train, y_train, x_val, y_val,
                x_geometry=None, rho_target=None, pi_target=None,
                seed: int = 0, epochs: int = 400, patience: int = 50,
                batch_size: int = 256, lr_max: float = 1e-3,
                lr_min: float = 1e-5, weight_decay: float = 1e-5,
                lambda_rho: float = 0.0, lambda_pi: float = 0.0,
                device: str | None = None,
                order_namespace: str = "formal_common",
                gradient_clip: float = 1.0,
                q_train_physical=None, physics_residual_fn=None,
                physics_weight: float = 0.0, validation_residual_fn=None,
                validation_q_physical=None, validation_residual_weight: float = 0.0) -> TrainResult:
    seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_obj).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max,
                            weight_decay=weight_decay, foreach=False, fused=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr_min)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device_obj)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device_obj)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device_obj)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device_obj)
    qv = torch.as_tensor(validation_q_physical, dtype=torch.float32, device=device_obj) if validation_q_physical is not None else None
    qt = torch.as_tensor(q_train_physical, dtype=torch.float32, device=device_obj) if q_train_physical is not None else None
    xg = torch.as_tensor(x_geometry if x_geometry is not None else x_train,
                         dtype=torch.float32, device=device_obj)
    rt = torch.as_tensor(rho_target, dtype=torch.float32, device=device_obj) if rho_target is not None else None
    pt = torch.as_tensor(pi_target, dtype=torch.float32, device=device_obj) if pi_target is not None else None
    n = len(xt)
    best = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    start = time.time()
    # For paired baseline+adapter runs, the initialized frozen anchor is a
    # valid epoch-0 checkpoint.  This prevents an exploratory correction from
    # degrading a strong baseline when it has not yet learned a useful
    # geometry-conditioned residual.
    if getattr(model, "anchor_only", False):
        model.eval()
        with torch.no_grad():
            initial_vals = []
            for start_idx in range(0, len(xv), batch_size):
                pred0=model(xv[start_idx:start_idx + batch_size]); val0=torch.mean((pred0-yv[start_idx:start_idx + batch_size]) ** 2)
                if validation_residual_fn is not None and validation_residual_weight:
                    val0=val0+validation_residual_weight*torch.mean(validation_residual_fn(pred0,qv[start_idx:start_idx+batch_size]).square())
                initial_vals.append(float(val0))
        best = float(np.mean(initial_vals))
        best_epoch = -1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        order = order_for(seed, epoch, n, order_namespace)
        geo_order = order_for(seed, epoch, len(xg), order_namespace + "|geometry")
        steps = math.ceil(n / batch_size)
        running = 0.0
        for step in range(steps):
            ids = order[step * batch_size:(step + 1) * batch_size]
            if len(ids) == 0:
                continue
            positions = (np.arange(batch_size) + step * batch_size) % len(xg)
            gids = geo_order[positions]
            xb, yb = xt[torch.as_tensor(ids, device=device_obj)], yt[torch.as_tensor(ids, device=device_obj)]
            xgb = xg[torch.as_tensor(gids, device=device_obj)]
            opt.zero_grad(set_to_none=True)
            output = model(xb, return_aux=True) if hasattr(model, "rho_head") else model(xb)
            pred = output[0] if isinstance(output, tuple) else output
            loss = torch.mean((pred - yb) ** 2)
            if physics_residual_fn is not None and physics_weight:
                qbatch = qt[torch.as_tensor(ids, device=device_obj)]
                residual = physics_residual_fn(pred, qbatch)
                loss = loss + physics_weight * torch.mean(residual.square())
            if hasattr(model, "rho_head") and (lambda_rho or lambda_pi):
                geo_output = model(xgb, return_aux=True)
                rg = rt[torch.as_tensor(gids, device=device_obj)] if rt is not None else None
                pg = pt[torch.as_tensor(gids, device=device_obj)] if pt is not None else None
                loss = loss + _geometry_loss(geo_output, rg, pg, lambda_rho, lambda_pi)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite independent GA-NAR loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            opt.step()
            running += float(loss.detach())
        sched.step()
        model.eval()
        with torch.no_grad():
            vals = []
            for start_idx in range(0, len(xv), batch_size):
                predv=model(xv[start_idx:start_idx + batch_size]); valv=torch.mean((predv-yv[start_idx:start_idx + batch_size]) ** 2)
                if validation_residual_fn is not None and validation_residual_weight:
                    valv=valv+validation_residual_weight*torch.mean(validation_residual_fn(predv,qv[start_idx:start_idx+batch_size]).square())
                vals.append(float(valv))
        val_loss = float(np.mean(vals))
        history.append({"epoch": epoch, "train_loss": running / max(1, steps),
                        "val_pred": val_loss, "lr": opt.param_groups[0]["lr"]})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best, best_epoch, len(history), time.time() - start,
                       history, sum(p.numel() for p in model.parameters() if p.requires_grad),
                       str(device_obj), order_namespace)
