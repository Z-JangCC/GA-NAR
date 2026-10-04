"""Implicit-system definitions."""

from .base import ImplicitSystem
from .synthetic.model import SyntheticImplicitSystem
from .field_equilibria import CoupledDuffingEquilibriumSystem, AllenCahn2DEquilibriumSystem, ShallowWater2DFreeSurfaceSystem

__all__ = ["ImplicitSystem", "SyntheticImplicitSystem", "CoupledDuffingEquilibriumSystem", "AllenCahn2DEquilibriumSystem", "ShallowWater2DFreeSurfaceSystem"]
