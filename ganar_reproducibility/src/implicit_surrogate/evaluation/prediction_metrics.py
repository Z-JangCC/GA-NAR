from __future__ import annotations

import numpy as np
import torch


def evaluate_predictions(model, dataset_split, system, standardization) -> dict[str, float]:
    model.eval()
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    with torch.no_grad():
        prediction_std = model(torch.as_tensor(dataset_split.x_std, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
    target_std = np.asarray(dataset_split.y_std, dtype=np.float64)
    mse = float(np.mean((prediction_std - target_std) ** 2))
    predicted_state = prediction_std * standardization.z_std + standardization.z_mean
    residuals = np.asarray([system.residual(state, parameter) for state, parameter in zip(predicted_state, dataset_split.q_phys)], dtype=np.float64)
    physical_residual = float(np.mean(np.linalg.norm(residuals, axis=1) / np.sqrt(residuals.shape[1])))
    return {"mse": mse, "physical_residual": physical_residual}


def evaluate_batch(model, x_std: np.ndarray) -> np.ndarray:
    model.eval()
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    with torch.no_grad():
        return model(torch.as_tensor(x_std, dtype=torch.float32, device=device)).cpu().numpy()
