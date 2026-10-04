"""Public neural-representation JVP interface.

The implementation remains in the representation package so the analysis
formula has one source of truth; this module is the replaceable public entry
point planned for future neural architectures.
"""

from __future__ import annotations

import numpy as np


def neural_representation_jvp(model, standardized_input: np.ndarray, direction: np.ndarray, layer_name: str, whitening_matrix: np.ndarray, activation_mean: np.ndarray) -> np.ndarray:
    from ..representation.representation_sensitivity import representation_jvp

    return representation_jvp(model, standardized_input, direction, layer_name, whitening_matrix, activation_mean)


__all__ = ["neural_representation_jvp"]
