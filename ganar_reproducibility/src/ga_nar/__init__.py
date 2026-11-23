"""Geometry-Adaptive Nonlinearity Allocation and Representation."""

from .model import GeometryAdaptiveSurrogate, ga_nar_loss
from .geometry import ResponseGeometry, build_response_geometry

__all__ = ["GeometryAdaptiveSurrogate", "ResponseGeometry",
           "build_response_geometry", "ga_nar_loss"]
