"""Finite-scale directional Jacobian sensitivity analysis."""

from .directions import GeometryDirections, generate_geometry_directions
from .directional_jacobian_variation import estimate_directional_variation
from .sensitivity_matrix import build_sensitivity_matrices
from .scale_selection import SensitivityScaleSelector
from .spectral_projection import project_sensitivity_spectrum
from .dominant_subspace import identify_dominant_subspace

__all__ = [
    "GeometryDirections",
    "generate_geometry_directions",
    "estimate_directional_variation",
    "build_sensitivity_matrices",
    "SensitivityScaleSelector",
    "project_sensitivity_spectrum",
    "identify_dominant_subspace",
]

