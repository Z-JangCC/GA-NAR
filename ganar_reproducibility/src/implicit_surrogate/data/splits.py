from __future__ import annotations

import numpy as np

from ..core.datatypes import DatasetBundle
from .standardization import Standardizer


def build_bundle(
    q_phys: np.ndarray,
    z_phys: np.ndarray,
    counts: dict[str, int],
    *,
    metadata: dict | None = None,
) -> DatasetBundle:
    expected = {"prediction": 8192, "sensitivity": 512, "validation": 1024, "test": 2048, "representation_probe": 64}
    for key, default in expected.items():
        counts.setdefault(key, default)
    total = sum(counts.values())
    if len(q_phys) != total or len(z_phys) != total:
        raise ValueError(f"expected {total} samples, got {len(q_phys)}")
    standardizer = Standardizer.fit(q_phys[: counts["prediction"] + counts["sensitivity"]], z_phys[: counts["prediction"] + counts["sensitivity"]])
    position = 0
    splits = {}
    for name in ("prediction", "sensitivity", "validation", "test", "representation_probe"):
        size = counts[name]
        q_chunk = q_phys[position : position + size]
        z_chunk = z_phys[position : position + size]
        ids = np.arange(position, position + size, dtype=np.int64)
        splits[name] = standardizer.split(q_chunk, z_chunk, ids, {"split": name})
        position += size
    train_q = np.concatenate([splits["prediction"].q_phys, splits["sensitivity"].q_phys], axis=0)
    train_z = np.concatenate([splits["prediction"].z_phys, splits["sensitivity"].z_phys], axis=0)
    train_ids = np.concatenate([splits["prediction"].sample_ids, splits["sensitivity"].sample_ids])
    train = standardizer.split(train_q, train_z, train_ids, {"split": "train"})
    ref_count = min(1024, counts["prediction"])
    reference_ids = splits["prediction"].sample_ids[:ref_count].copy()
    return DatasetBundle(
        train=train,
        prediction=splits["prediction"],
        sensitivity=splits["sensitivity"],
        validation=splits["validation"],
        test=splits["test"],
        representation_probe=splits["representation_probe"],
        standardization=standardizer.statistics,
        reference_activation_ids=reference_ids,
        metadata={} if metadata is None else metadata,
    )

