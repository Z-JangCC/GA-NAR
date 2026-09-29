from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RandomManager:
    """Single auditable source of NumPy and PyTorch randomness."""

    seed: int

    def __post_init__(self) -> None:
        self.generator = np.random.default_rng(self.seed)

    def normal(self, shape: tuple[int, ...], *, dtype=np.float64) -> np.ndarray:
        return self.generator.normal(size=shape).astype(dtype, copy=False)

    def uniform(self, low: float, high: float, shape: tuple[int, ...], *, dtype=np.float64) -> np.ndarray:
        return self.generator.uniform(low, high, size=shape).astype(dtype, copy=False)

    def rademacher(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.generator.choice(np.array([-1.0, 1.0]), size=shape).astype(np.float64)

    def unit_sphere(self, count: int, dimension: int) -> np.ndarray:
        values = self.normal((count, dimension))
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("zero norm in unit-sphere direction generation")
        return values / norms

    def permutation(self, n: int) -> np.ndarray:
        return self.generator.permutation(n)

    def record(self, purpose: str, shape: tuple[int, ...], array: Any) -> dict[str, Any]:
        import hashlib

        values = np.asarray(array)
        return {
            "purpose": purpose,
            "seed": int(self.seed),
            "shape": list(shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }


def seed_torch(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
