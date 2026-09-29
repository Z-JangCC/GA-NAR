from __future__ import annotations

import numpy as np
import torch


def predict_mse(model, x, y, device=None):
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=dtype, device=device))
        target = torch.as_tensor(y, dtype=dtype, device=device)
        mse = torch.mean((pred - target) ** 2).item()
    return float(mse)


def layerwise_diagnostics(model, x, device=None):
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        model(torch.as_tensor(x, dtype=dtype, device=device),
              return_aux=True, return_diagnostics=True)
    last = model._last
    out = {"rho_mean": float(last["rho"].mean()),
           "pi_mean": last["pi"].mean(0).cpu().numpy().tolist(),
           "beta_mean": last["beta"].mean(0).cpu().numpy().tolist(),
           "gamma_mean": float(last["gamma"].mean()),
           "alpha": last["alpha"].detach().cpu().numpy().tolist()}
    if last["layers"]:
        out["modulation_rms"] = [float(d["modulation"].square().mean().sqrt())
                                  for d in last["layers"]]
    return out


def paired_differences(ga_values, baseline_values):
    ga = np.asarray(ga_values, dtype=float)
    base = np.asarray(baseline_values, dtype=float)
    if ga.shape != base.shape:
        raise ValueError("paired arrays must have identical shape")
    return ga - base
