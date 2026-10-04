"""Central experiment profiles.

The published ``full`` profile enumerates every experiment in the study.  A
small ``smoke`` profile uses identical code paths and is used for tests and
installation checks; it is never mixed into reported full results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    train_size: int
    val_size: int
    test_size: int
    geometry_train_size: int
    probe_size: int
    probe_directions: int
    geometry_step: float


@dataclass(frozen=True)
class BudgetConfig:
    main_steps: int
    mechanism_steps: int
    scale_steps: int
    batch_size: int
    eval_interval: int


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    seeds: tuple[int, ...]
    ablation_seeds: tuple[int, ...]
    systems: tuple[str, ...]
    baselines: tuple[str, ...]
    reference_width: int
    reference_depth: int
    data: DataConfig
    budget: BudgetConfig
    width_levels: tuple[int, ...] = (128, 256, 512, 1024)
    depth_levels: tuple[int, ...] = (2, 4, 8)
    sample_levels: tuple[int, ...] = (10_000, 100_000, 1_000_000)
    alpha_levels: tuple[float, ...] = (0.10, 0.25, 0.40, 0.49)
    coupling_levels: tuple[float, ...] = (0.0, 0.33, 0.66, 1.0)
    rank_levels: tuple[int, ...] = (1, 2, 4, 8)
    condition_levels: tuple[float, ...] = (1.0, 5.0, 15.0, 40.0)
    localization_levels: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    interaction_levels: tuple[float, ...] = (0.0, 0.5, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_PROFILE = ExperimentProfile(
    name="full",
    seeds=(11, 29, 47),
    ablation_seeds=(11,),
    systems=("controlled", "ac_power", "duffing", "ieee118", "allen_cahn", "shallow_water"),
    baselines=("relu", "gelu", "silu", "swiglu", "resmlp", "fourier"),
    reference_width=256,
    reference_depth=4,
    data=DataConfig(
        train_size=50_000,
        val_size=5_000,
        test_size=10_000,
        geometry_train_size=50_000,
        probe_size=1_024,
        probe_directions=16,
        geometry_step=0.06,
    ),
    budget=BudgetConfig(
        main_steps=2_500,
        mechanism_steps=1_500,
        scale_steps=2_500,
        batch_size=512,
        eval_interval=125,
    ),
)


SMOKE_PROFILE = ExperimentProfile(
    name="smoke",
    seeds=(11,),
    ablation_seeds=(11,),
    systems=("controlled", "ac_power", "duffing", "ieee118", "allen_cahn", "shallow_water"),
    baselines=("relu", "gelu", "swiglu"),
    reference_width=32,
    reference_depth=2,
    data=DataConfig(
        train_size=192,
        val_size=64,
        test_size=96,
        geometry_train_size=192,
        probe_size=32,
        probe_directions=3,
        geometry_step=0.06,
    ),
    budget=BudgetConfig(
        main_steps=15,
        mechanism_steps=10,
        scale_steps=10,
        batch_size=64,
        eval_interval=5,
    ),
    width_levels=(16, 32),
    depth_levels=(1, 2),
    sample_levels=(128, 256),
    alpha_levels=(0.15, 0.40),
    coupling_levels=(0.2, 0.8),
    rank_levels=(1, 4),
    condition_levels=(1.0, 10.0),
    localization_levels=(0.75, 2.0),
    interaction_levels=(0.0, 1.0),
)


MODULAR_PROFILE = ExperimentProfile(
    name="modular",
    seeds=(11, 29, 47),
    ablation_seeds=(11,),
    systems=("controlled", "ac_power", "duffing", "ieee118", "allen_cahn", "shallow_water"),
    baselines=("relu", "gelu", "silu", "swiglu"),
    reference_width=256,
    reference_depth=4,
    data=DataConfig(
        train_size=30_000,
        val_size=3_000,
        test_size=6_000,
        geometry_train_size=30_000,
        probe_size=768,
        probe_directions=16,
        geometry_step=0.06,
    ),
    budget=BudgetConfig(
        main_steps=2_000,
        mechanism_steps=1_000,
        scale_steps=2_000,
        batch_size=512,
        eval_interval=100,
    ),
)


def get_profile(name: str) -> ExperimentProfile:
    if name == "full":
        return FULL_PROFILE
    if name == "smoke":
        return SMOKE_PROFILE
    if name == "modular":
        return MODULAR_PROFILE
    raise ValueError(f"Unknown profile {name!r}; choose 'full' or 'smoke'")
