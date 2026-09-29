"""Standalone GA-NAR Layer-wise implementation.

This package intentionally does not import the legacy :mod:`ganar` package.
"""

from .models.layerwise_ganar import LayerwiseGANAR
from .models.baselines import MatchedSwiGLU, MatchedReLU, MatchedActivation

__all__ = ["LayerwiseGANAR", "MatchedSwiGLU", "MatchedReLU", "MatchedActivation"]
