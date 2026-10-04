from __future__ import annotations

from dataclasses import dataclass

from .trainer import Trainer


@dataclass(frozen=True)
class LambdaSelection:
    lambda_value: float
    validation_mse: float
    candidates: tuple[tuple[float, float], ...]


def select_local_nonlinearity_weight(build_model, trainer: Trainer, train_kwargs: dict, candidates=(0.1, 0.3, 1.0), seed: int = 999) -> LambdaSelection:
    scores = []
    for value in candidates:
        model = build_model()
        result = trainer.fit(model, **train_kwargs, local_nonlinearity_loss_weight=value, seed=seed)
        scores.append((float(value), float(result.best_validation_mse)))
    selected = min(scores, key=lambda item: item[1])
    return LambdaSelection(selected[0], selected[1], tuple(scores))

