from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def multiply_accumulate_count(model: nn.Module, input_dimension: int) -> int:
    """Count linear MACs, including the frozen basis projection in the GA graph."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += int(module.in_features * module.out_features)
    if hasattr(model, "sensitivity_basis"):
        rank = int(model.sensitivity_basis.shape[1])
        total += 2 * int(input_dimension * rank)
    return total


def batch_one_latency(model: nn.Module, input_dimension: int, *, warmup: int = 100, measurements: int = 1000) -> float:
    model.eval()
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    value = torch.zeros((1, input_dimension), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(value)
        times = []
        for _ in range(measurements):
            start = time.perf_counter()
            model(value)
            if value.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)
    return float(np.median(times))
