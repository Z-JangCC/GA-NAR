from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.random import RandomManager
from .parameter_matching import count_trainable_parameters, find_parameter_matched_width
from .relu_mlp import ReLUMultilayerPerceptron
from .sensitivity_informed_surrogate import SensitivityInformedSurrogate
from .swiglu_mlp import SwiGLUMultilayerPerceptron


@dataclass(frozen=True)
class ModelSpec:
    registry_name: str
    display_name: str


MODEL_SPECS = (
    ModelSpec("relu_mlp", "ReLU-MLP"),
    ModelSpec("swiglu_mlp", "SwiGLU-MLP"),
    ModelSpec("parameter_matched_swiglu", "Parameter-Matched SwiGLU"),
    ModelSpec("random_subspace_sensitivity_model", "Random-Subspace GA-NRL"),
    ModelSpec("subspace_only_sensitivity_model", "Subspace-Only GA-NRL"),
    ModelSpec("full_sensitivity_informed_model", "Full GA-NRL"),
)


def random_subspace_basis(input_dimension: int, rank: int, seed: int) -> np.ndarray:
    matrix = RandomManager(10000 + int(seed)).normal((input_dimension, rank))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def build_model(registry_name: str, input_dimension: int, output_dimension: int, sensitivity_basis: np.ndarray | None = None, *, seed: int = 0, matched_width: int | None = None):
    if registry_name == "relu_mlp":
        return ReLUMultilayerPerceptron(input_dimension, output_dimension)
    if registry_name == "swiglu_mlp":
        return SwiGLUMultilayerPerceptron(input_dimension, output_dimension)
    if registry_name == "parameter_matched_swiglu":
        if matched_width is None:
            raise ValueError("matched_width is required")
        return SwiGLUMultilayerPerceptron(input_dimension, output_dimension, width=matched_width)
    if registry_name in {"random_subspace_sensitivity_model", "subspace_only_sensitivity_model", "full_sensitivity_informed_model"}:
        if sensitivity_basis is None:
            raise ValueError("sensitivity_basis is required")
        basis = sensitivity_basis if registry_name != "random_subspace_sensitivity_model" else random_subspace_basis(input_dimension, sensitivity_basis.shape[1], seed)
        return SensitivityInformedSurrogate(input_dimension, output_dimension, basis, activity_supervision=registry_name != "subspace_only_sensitivity_model")
    raise KeyError(registry_name)
