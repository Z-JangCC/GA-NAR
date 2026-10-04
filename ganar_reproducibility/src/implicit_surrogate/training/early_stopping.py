"""Small, deterministic validation-only early-stopping state machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    patience: int = 30
    best_value: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, epoch: int, validation_mse: float) -> bool:
        if validation_mse < self.best_value:
            self.best_value = float(validation_mse)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


__all__ = ["EarlyStoppingState"]
