from __future__ import annotations

from .swiglu_mlp import SwiGLUMultilayerPerceptron


def count_trainable_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def find_parameter_matched_width(input_dimension: int, output_dimension: int, target_parameter_count: int, *, lower: int = 1, upper: int = 512) -> int:
    candidates = []
    for width in range(lower, upper + 1):
        candidate = SwiGLUMultilayerPerceptron(input_dimension, output_dimension, width=width)
        count = count_trainable_parameters(candidate)
        mismatch = abs(count - target_parameter_count)
        within = mismatch / max(target_parameter_count, 1) <= 0.05
        candidates.append((0 if within else 1, mismatch, width))
    return min(candidates)[2]

