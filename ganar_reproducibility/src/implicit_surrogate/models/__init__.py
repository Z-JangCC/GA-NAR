"""Surrogate model implementations and registry."""

from .relu_mlp import ReLUMultilayerPerceptron
from .swiglu_mlp import SwiGLUMultilayerPerceptron
from .sensitivity_informed_surrogate import SensitivityInformedSurrogate
from .parameter_matching import find_parameter_matched_width, count_trainable_parameters

__all__ = [
    "ReLUMultilayerPerceptron",
    "SwiGLUMultilayerPerceptron",
    "SensitivityInformedSurrogate",
    "find_parameter_matched_width",
    "count_trainable_parameters",
]

