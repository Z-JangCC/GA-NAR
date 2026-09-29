"""Protocol-frozen surrogate training utilities."""

from .losses import training_objective
from .trainer import Trainer, TrainingResult, initialize_paired_models

__all__ = ["training_objective", "Trainer", "TrainingResult", "initialize_paired_models"]

