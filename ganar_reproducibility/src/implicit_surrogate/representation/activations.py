from __future__ import annotations

import torch


def extract_activations(model, standardized_input: torch.Tensor) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        _, activations = model(standardized_input, return_activations=True)
    return activations

