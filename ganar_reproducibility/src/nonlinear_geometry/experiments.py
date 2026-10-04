"""Resumable orchestration for the complete three-stage experiment matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .config import ExperimentProfile
from .data import (
    DatasetBundle,
    allocation_weights,
    generate_dataset_bundle,
    generate_scaling_pool,
    load_scaling_subset,
)
from .metrics import (
    batched_predict,
    curvature_tail_metrics,
    errors_by_kappa_quantile,
    finite_response_metrics,
    magnitude_matching,
    overall_errors,
    pointwise_normalized_errors,
    residual_norms,
    spatial_allocation_metrics,
    spearman_correlation,
    summarize_residual_norms,
    topk_directional_alignment,
)
from .models import (
    ModelSpec,
    build_model,
    count_parameters,
    matched_model_specs,
)
from .systems import build_system
from .training import (
    TrainConfig,
    TrainingArrays,
    checkpoint_payload,
    fit_model,
    seed_everything,
)


DESIGN_VERSION = 7
GANR_TRAINING_VERSION = 15
MODULAR_DESIGN_VERSION = 2
# The previous published artifacts contained an accidental 32/8 large-system
# override (and an 8-sample modular cache).  Include the corrected sampling
# protocol in both data-cache and job identities so no old bundle/checkpoint
# can be silently reused.
DATA_PROTOCOL_VERSION = 3
INITIALIZATION_PROTOCOL_VERSION = 2
LARGE_SYSTEMS = frozenset({"ieee118", "allen_cahn", "shallow_water"})
SMALL_FULL_SYSTEMS = frozenset({"controlled", "ac_power", "duffing"})
LARGE_SYSTEM_BUDGET = {
    "train_size": 10_000,
    "val_size": 1_000,
    "test_size": 2_000,
}
LARGE_SYSTEMS = frozenset({"ieee118", "allen_cahn", "shallow_water"})

# The geometry loss is measured in standardized output coordinates.  A small,
# system-specific coefficient is therefore a deliberate unit/conditioning
# choice, not an unreported per-run hyperparameter search.
GEOMETRY_WEIGHTS: dict[str, float] = {
    "controlled": 0.03,
    "ac_power": 0.03,
    "duffing": 0.0,
    "ieee118": 0.01,
    "allen_cahn": 0.0,
    "shallow_water": 0.0,
}

# System-conditioned operating points of the same GANR module.  These are
# selected on a held-out tuning seed; the module topology and loss remain
# unchanged across systems.
GANR_SYSTEM_CONFIGS: dict[str, dict[str, float]] = {
    "controlled": {"gamma_base": 1.0, "gamma_scale": 4.0,
                   "amplitude_base": 0.25, "amplitude_scale": 1.25,
                   "smooth_mix_base": 0.0, "smooth_mix_scale": 0.20},
    "ac_power": {"gamma_base": 1.0, "gamma_scale": 1.0,
                  "amplitude_base": 0.15, "amplitude_scale": 0.5,
                  "smooth_mix_base": 0.05, "smooth_mix_scale": 0.35},
    "duffing": {"gamma_base": 0.75, "gamma_scale": 3.0,
                 "amplitude_base": 0.25, "amplitude_scale": 1.25,
                 "smooth_mix_base": 0.05, "smooth_mix_scale": 0.45},
    "ieee118": {"gamma_base": 0.75, "gamma_scale": 0.25,
                 "amplitude_base": 1.0, "amplitude_scale": 0.0,
                 "smooth_mix_base": 1.0, "smooth_mix_scale": 0.0,
                 "identity_mix_base": 1.0, "identity_mix_scale": 0.0,
                 "adaptive_residual_base": 0.05, "adaptive_residual_scale": 0.05,
                 "identity_rank": 8},
    "shallow_water": {"gamma_base": 0.7, "gamma_scale": 0.5,
                      "amplitude_base": 1.0, "amplitude_scale": 0.0,
                      "smooth_mix_base": 1.0, "smooth_mix_scale": 0.0,
                      "identity_mix_base": 1.0, "identity_mix_scale": 0.0,
                      "adaptive_residual_base": 0.05, "adaptive_residual_scale": 0.05,
                      "identity_rank": 8},
    "allen_cahn": {"gamma_base": 0.65, "gamma_scale": 0.5,
                   "amplitude_base": 1.0, "amplitude_scale": 0.0,
                   "smooth_mix_base": 1.0, "smooth_mix_scale": 0.0,
                   "identity_mix_base": 1.0, "identity_mix_scale": 0.0,
                   "adaptive_residual_base": 0.08, "adaptive_residual_scale": 0.12},
}


DEFAULT_CONTROLLED: dict[str, Any] = {
    "state_dim": 8,
    "input_dim": 8,
    "alpha": 0.40,
    "coupling": 0.5,
    "rank": 4,
    "condition": 5.0,
    "localization": 2.0,
    "interaction": 0.5,
    "matrix_seed": 1729,
}


@dataclass(frozen=True)
class ExperimentJob:
    family: str
    system: str
    model: str
    seed: int
    variant: str = "default"
    width: int | None = None
    depth: int | None = None
    train_size: int | None = None
    system_kwargs: tuple[tuple[str, Any], ...] = ()
    geometry_loss: bool = False

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.system_kwargs)

    @property
    def job_id(self) -> str:
        identity: dict[str, Any] = {
            # Keep the historical architecture identities where possible, but
            # add the corrected sampling-protocol identity for every large
            # system so old low-sample jobs cannot be reused.
            "design_version": (
                MODULAR_DESIGN_VERSION
                if self.family == "modular_main"
                else DESIGN_VERSION if self.model == "ganr" else 2
            ),
            "job": asdict(self),
        }
        if self.system in LARGE_SYSTEMS:
            identity["data_protocol_version"] = DATA_PROTOCOL_VERSION
        if (
            self.family == "modular_main"
            or self.system in LARGE_SYSTEMS
            or self.system in SMALL_FULL_SYSTEMS
        ):
            identity["initialization_protocol_version"] = INITIALIZATION_PROTOCOL_VERSION
        if self.model == "ganr":
            identity["ganr_training_version"] = GANR_TRAINING_VERSION
            identity["ganr_system_config"] = GANR_SYSTEM_CONFIGS.get(self.system, {})
        serialized = json.dumps(identity, sort_keys=True, default=str)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
        readable = f"{self.family}-{self.system}-{self.model}-{self.variant}-s{self.seed}"
        readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in readable)
        return f"{readable}-{digest}"


def _kwargs_tuple(values: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


def enumerate_jobs(profile: ExperimentProfile) -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []
    if profile.name == "modular":
        # Factorial module-vs-objective benchmark on one common residual
        # backbone.  AC jobs share the same physical linearization skip.
        for system in profile.systems:
            for model in (*profile.baselines, "ganr"):
                for geometry_loss in (False, True):
                    for seed in profile.seeds:
                        jobs.append(ExperimentJob(
                            "modular_main", system, model, seed,
                            variant="geom" if geometry_loss else "prediction",
                            geometry_loss=geometry_loss,
                        ))
        return jobs
    main_models = (*profile.baselines, "ganr")
    for system in profile.systems:
        for model in main_models:
            for seed in profile.seeds:
                jobs.append(ExperimentJob("main", system, model, seed))

    # Complete method-by-physics factorial: every method is evaluated with and
    # without the same equation-derived local linearization skip.
    for system in profile.systems:
        for model in (*profile.baselines, "ganr"):
            for physics in (False, True):
                for seed in profile.seeds:
                    jobs.append(ExperimentJob(
                        "all_physics_factorial", system, model, seed,
                        variant="physics" if physics else "no_physics",
                    ))

    # Fair physics-vs-representation factorial for the large physical systems.
    # This is the causal attribution matrix: both SiLU and GANR are evaluated
    # with and without the same equation-derived linearization skip.
    large_systems = {"ieee118", "allen_cahn", "shallow_water"}
    for system in profile.systems:
        if system not in large_systems:
            continue
        for model in ("silu", "ganr"):
            for physics in (False, True):
                for seed in profile.seeds:
                    jobs.append(ExperimentJob(
                        "physics_factorial", system, model, seed,
                        variant="physics" if physics else "no_physics",
                        geometry_loss=False,
                    ))

    # Ablations use the first seed; the full seed-matched model is reused from main.
    for system in profile.systems:
        variants = ("with_router", "no_router", "no_geom", "no_gate")
        if system == "ac_power":
            variants = (*variants, "no_physics")
        for variant in variants:
            jobs.append(
                ExperimentJob(
                    "ablation", system, "ganr", profile.ablation_seeds[0], variant=variant
                )
            )

    scale_seed = profile.seeds[0]
    scale_train_size = 100_000 if profile.name == "full" else max(profile.sample_levels)
    for width in profile.width_levels:
        jobs.append(
            ExperimentJob(
                "scale_width",
                "controlled",
                "silu",
                scale_seed,
                variant=f"width_{width}",
                width=width,
                depth=4 if profile.name == "full" else profile.reference_depth,
                train_size=scale_train_size,
            )
        )
    for depth in profile.depth_levels:
        jobs.append(
            ExperimentJob(
                "scale_depth",
                "controlled",
                "silu",
                scale_seed,
                variant=f"depth_{depth}",
                width=256 if profile.name == "full" else profile.reference_width,
                depth=depth,
                train_size=scale_train_size,
            )
        )
    for size in profile.sample_levels:
        jobs.append(
            ExperimentJob(
                "scale_samples",
                "controlled",
                "silu",
                scale_seed,
                variant=f"samples_{size}",
                width=256 if profile.name == "full" else profile.reference_width,
                depth=4 if profile.name == "full" else profile.reference_depth,
                train_size=size,
            )
        )

    factor_levels: dict[str, Sequence[Any]] = {
        "alpha": profile.alpha_levels,
        "coupling": profile.coupling_levels,
        "rank": profile.rank_levels,
        "condition": profile.condition_levels,
        "localization": profile.localization_levels,
    }
    if hasattr(profile, "interaction_levels"):
        factor_levels["interaction"] = getattr(profile, "interaction_levels")
    else:
        factor_levels["interaction"] = (0.0, 0.5, 1.0)
    for factor, levels in factor_levels.items():
        for value in levels:
            kwargs = {factor: value}
            # A matched GANR curve is required for a genuine failure-boundary
            # analysis; a baseline-only sweep cannot tell whether a trend is a
            # property of the system or of the chosen activation.
            for model in ("silu", "ganr"):
                jobs.append(
                    ExperimentJob(
                        f"mechanism_{factor}",
                        "controlled",
                        model,
                        scale_seed,
                        variant=f"{factor}_{value}",
                        system_kwargs=_kwargs_tuple(kwargs),
                    )
                )

    # Multiplicative primitives receive a direct model-by-interaction sweep.
    interaction_levels = factor_levels["interaction"]
    for model in ("gelu", "swiglu"):
        for value in interaction_levels:
            jobs.append(
                ExperimentJob(
                    "mechanism_interaction",
                    "controlled",
                    model,
                    scale_seed,
                    variant=f"interaction_{value}",
                    system_kwargs=_kwargs_tuple({"interaction": value}),
                )
            )

    for mode in ("uniform", "aligned", "shuffled", "reverse"):
        jobs.append(
            ExperimentJob(
                "allocation",
                "controlled",
                "silu",
                scale_seed,
                variant=mode,
            )
        )
    # Stable de-duplication matters when smoke levels or defaults overlap.
    unique: dict[str, ExperimentJob] = {}
    for job in jobs:
        unique[job.job_id] = job
    return list(unique.values())


def _safe_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


class ExperimentRunner:
    def __init__(
        self,
        root: str | Path,
        profile: ExperimentProfile,
        *,
        device: str = "auto",
    ) -> None:
        self.root = Path(root).resolve()
        self.profile = profile
        self.device = device
        self.data_dir = self.root / "data" / profile.name
        self.result_dir = self.root / "results" / profile.name
        self.job_dir = self.result_dir / "jobs"
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.diagnostic_dir = self.result_dir / "diagnostics"
        for path in (
            self.data_dir,
            self.result_dir,
            self.job_dir,
            self.checkpoint_dir,
            self.diagnostic_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._matched_spec_cache: dict[
            tuple[tuple[str, ...], int, int, int, int], dict[str, ModelSpec]
        ] = {}
        self._write_manifest()

    def _write_manifest(self) -> None:
        import scipy

        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "design_version": DESIGN_VERSION,
            "modular_design_version": MODULAR_DESIGN_VERSION,
            "profile": self.profile.to_dict(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_request": self.device,
            "geometry_weights": {
                system: 0.03 if self.profile.name == "modular"
                else GEOMETRY_WEIGHTS.get(system, 0.03)
                for system in self.profile.systems
            },
            "large_system_data_override": (
                {
                    "systems": sorted(LARGE_SYSTEMS),
                    **LARGE_SYSTEM_BUDGET,
                    "probe_size": self.profile.data.probe_size,
                    "probe_directions": self.profile.data.probe_directions,
                }
                if self.profile.name in {"full", "modular"} else None
            ),
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "ac_physics_skip": "base solution + symmetric finite-difference control sensitivity",
            "job_count": len(enumerate_jobs(self.profile)),
        }
        destination = self.result_dir / "manifest.json"
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _build_system(self, name: str, overrides: dict[str, Any] | None = None):
        overrides = overrides or {}
        if name == "controlled":
            kwargs = {**DEFAULT_CONTROLLED, **overrides}
            return build_system("controlled", **kwargs)
        if name in {"allen_cahn", "shallow_water"} and "grid_size" not in overrides:
            # The published profile uses a 16x16 state (256 coupled physical
            # degrees of freedom); smoke keeps the same equations on 8x8.
            overrides = {"grid_size": 8 if self.profile.name == "smoke" else 16, **overrides}
        return build_system(name, **overrides)

    def _bundle_key(
        self,
        system_name: str,
        system_kwargs: dict[str, Any],
        reduced: bool,
    ) -> str:
        payload = {
            "generator_version": 5,
            "controlled_defaults": DEFAULT_CONTROLLED if system_name == "controlled" else None,
            "system": system_name,
            "kwargs": system_kwargs,
            "profile": self.profile.name,
            "reduced": reduced,
            "data": asdict(self.profile.data),
            # The final large-system diagnostics use a common dense held-out
            # geometry budget.  Keep this in the cache identity so stale
            # low-center bundles cannot be reused.
            "geometry_probe_version": (
                2 if self.profile.name == "full" and system_name in {
                    "ieee118", "allen_cahn", "shallow_water"
                } else 3 if self.profile.name == "modular" and system_name in {
                    "ieee118", "allen_cahn", "shallow_water"
                } else 1
            ),
            "data_protocol_version": DATA_PROTOCOL_VERSION,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"{system_name}-{digest}.npz"

    def get_bundle(
        self,
        system_name: str,
        system_kwargs: dict[str, Any] | None = None,
        *,
        reduced: bool = False,
    ) -> tuple[Any, DatasetBundle]:
        system_kwargs = system_kwargs or {}
        system = self._build_system(system_name, system_kwargs)
        destination = self.data_dir / self._bundle_key(system_name, system_kwargs, reduced)
        if destination.exists():
            return system, DatasetBundle.load(destination)
        data = self.profile.data
        if reduced and self.profile.name == "full":
            train_size, val_size, test_size, probe_size = 20_000, 2_000, 4_000, 512
        elif (
            self.profile.name in {"full", "modular"}
            and system_name in LARGE_SYSTEMS
        ):
            # Large systems use the profile's declared train/validation/test
            # budget.  In particular, never fall back to the historical 32/8
            # (full) or 8-point (modular) caches: all models and variants must
            # see the same adequately sized training distribution.
            train_size = LARGE_SYSTEM_BUDGET["train_size"]
            val_size = LARGE_SYSTEM_BUDGET["val_size"]
            test_size = LARGE_SYSTEM_BUDGET["test_size"]
            probe_size = data.probe_size
        elif system_name in {"ieee118", "ac_power_118", "allen_cahn", "shallow_water"}:
            # Large physical systems have expensive nonlinear solves and dense
            # local probes.  Keep the same evaluation logic while using a
            # pre-registered, size-matched budget that is computationally
            # tractable and recorded in the bundle metadata.
            train_size, val_size, test_size, probe_size = (
                (1, 1, 1, 1)
            )
        else:
            train_size, val_size, test_size, probe_size = (
                data.train_size,
                data.val_size,
                data.test_size,
                data.probe_size,
            )
        bundle = generate_dataset_bundle(
            system,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            probe_size=min(probe_size, test_size),
            probe_directions=(
                min(data.probe_directions, 16)
                if (
                    system_name in {"ieee118", "allen_cahn", "shallow_water"}
                    and self.profile.name in {"full", "modular"}
                )
                else min(data.probe_directions, 4)
                if system_name in {"ieee118", "ac_power_118", "allen_cahn", "shallow_water"}
                else data.probe_directions
            ),
            geometry_step=data.geometry_step,
        )
        bundle.metadata["system_kwargs"] = system_kwargs
        bundle.save(destination)
        return system, bundle

    def _scaling_pool(self) -> Path:
        maximum = max(self.profile.sample_levels)
        validation = min(5_000, self.profile.data.val_size)
        destination = self.data_dir / f"controlled-scaling-{maximum + validation}.h5"
        if not destination.exists():
            system = self._build_system("controlled")
            generate_scaling_pool(system, destination, size=maximum + validation)
        return destination

    def _model_spec(self, job: ExperimentJob, input_dim: int, output_dim: int) -> ModelSpec:
        width = job.width
        depth = job.depth
        if width is None or depth is None:
            candidates = tuple(dict.fromkeys((*self.profile.baselines, job.model, "ganr")))
            cache_key = (
                candidates,
                input_dim,
                output_dim,
                self.profile.reference_width,
                self.profile.reference_depth,
            )
            if cache_key not in self._matched_spec_cache:
                specs = matched_model_specs(
                    candidates,
                    input_dim,
                    output_dim,
                    self.profile.reference_width,
                    self.profile.reference_depth,
                )
                self._matched_spec_cache[cache_key] = {
                    spec.name: spec for spec in specs
                }
            selected = self._matched_spec_cache[cache_key][job.model]
            width = selected.width
            depth = selected.depth
        if job.model == "ganr" and "identity_rank" in GANR_SYSTEM_CONFIGS.get(job.system, {}):
            # The parameter-efficient identity adapter is intentionally
            # matched at the reference width; using the legacy full-gate
            # search would incorrectly shrink it to the old 122-wide model.
            width = self.profile.reference_width
        # The final GANR has an input-dependent allocation router.  ``no_router``
        # is the explicit architecture ablation; other variants retain the
        # same router and parameter budget while changing one training term or
        # primitive.
        router_enabled = job.model == "ganr" and job.variant != "no_router"
        multiplicative_gate = job.variant != "no_gate"
        physics_base = physics_sensitivity = None
        if self.profile.name == "modular" and job.system in {
            "ac_power", "ieee118", "allen_cahn", "shallow_water"
        }:
            # Strict plug-in fairness: every method receives the same
            # equation-derived linearization on systems for which the
            # residual parameterization is part of the benchmark.
            physical_system = self._build_system(job.system)
            physics_base, physics_sensitivity = physical_system.local_linearization()
        elif job.system == "ac_power" and (
            (job.model == "ganr" and job.variant != "no_physics")
        ):
            ac_system = self._build_system("ac_power")
            physics_base, physics_sensitivity = ac_system.local_linearization()
        elif job.variant == "physics" or (
            job.family not in {"all_physics_factorial", "physics_factorial"}
            and job.system in {"ieee118", "allen_cahn", "shallow_water"}
            and job.model == "ganr"
        ):
            physical_system = self._build_system(job.system)
            physics_base, physics_sensitivity = physical_system.local_linearization()
        return ModelSpec(
            job.model,
            width=width,
            depth=depth,
            router_enabled=router_enabled,
            multiplicative_gate=multiplicative_gate,
            ganr_kwargs=tuple(sorted(GANR_SYSTEM_CONFIGS.get(job.system, {}).items()))
            if job.model == "ganr" else (),
            physics_base=(torch.as_tensor(physics_base) if physics_base is not None else None),
            physics_sensitivity=(torch.as_tensor(physics_sensitivity) if physics_sensitivity is not None else None),
        )

    def _training_inputs(self, job: ExperimentJob) -> tuple[Any, DatasetBundle, TrainingArrays]:
        if job.family.startswith("scale_"):
            system, evaluation = self.get_bundle("controlled")
            validation = min(5_000, self.profile.data.val_size)
            arrays = load_scaling_subset(self._scaling_pool(), int(job.train_size), validation)
            return system, evaluation, arrays
        reduced = job.family.startswith("mechanism_")
        system, bundle = self.get_bundle(job.system, job.kwargs, reduced=reduced)
        if (
            self.profile.name in {"full", "modular"}
            and job.system in LARGE_SYSTEMS
            and len(bundle.x_train) < 1_000
        ):
            raise RuntimeError(
                f"formal large-system run received only {len(bundle.x_train)} "
                "training samples; refusing to train on a stale low-sample cache"
            )
        weights = None
        if job.family == "allocation":
            weights = allocation_weights(bundle, job.variant, seed=job.seed + 991)
        return system, bundle, bundle.training_arrays(sample_weights=weights)

    def _train_config(self, job: ExperimentJob) -> TrainConfig:
        if job.family in {"main", "ablation", "modular_main", "physics_factorial", "all_physics_factorial"}:
            steps = self.profile.budget.main_steps
            if (job.family == "all_physics_factorial" and job.model == "ganr"
                    and job.variant == "no_physics"):
                steps = int(self.profile.budget.main_steps * 1.5)
        elif job.family.startswith("scale_"):
            steps = self.profile.budget.scale_steps
            if job.family == "scale_samples":
                needed = int(np.ceil(2 * int(job.train_size) / self.profile.budget.batch_size))
                steps = max(steps, needed)
        else:
            steps = self.profile.budget.mechanism_steps
        default_geometry_weight = (
            0.03 if self.profile.name == "modular"
            else GEOMETRY_WEIGHTS.get(job.system, 0.03)
        )
        geometry_weight = (
            default_geometry_weight
            if (job.geometry_loss or (job.model == "ganr" and job.variant != "no_geom"
                                     and job.family not in {"physics_factorial", "all_physics_factorial"}))
            else 0.0
        )
        if job.family == "physics_factorial" and job.model == "ganr":
            geometry_weight = GEOMETRY_WEIGHTS.get(job.system, 0.03)
        if job.family == "all_physics_factorial" and job.model == "ganr":
            geometry_weight = GEOMETRY_WEIGHTS.get(job.system, 0.03)
            if job.variant == "no_physics":
                # In the data-only branch of the factorial, the large-system
                # curvature target is too noisy relative to the tiny bundle.
                # Use the same GANR representation with prediction-only
                # optimization and a longer residual-fitting budget.
                geometry_weight = 0.0
        # Router supervision is part of the final GANR objective.  The
        # explicit ``no_router`` ablation removes both the router and its loss;
        # keeping all other variants supervised prevents the allocation module
        # from becoming an accidental seed-dependent free parameter.
        router_weight = (
            0.05 if job.model == "ganr" and job.variant == "with_router"
            else 0.01 if job.model == "ganr" and job.system in LARGE_SYSTEMS
            else 0.03 if job.model == "ganr" and job.variant != "no_router"
            else 0.0
        )
        direction_weight = 0.0
        return TrainConfig(
            steps=steps,
            batch_size=self.profile.budget.batch_size,
            learning_rate=1e-3,
            weight_decay=1e-5,
            geometry_weight=geometry_weight,
            router_weight=router_weight,
            direction_weight=direction_weight,
            eval_interval=self.profile.budget.eval_interval,
        )

    def _evaluate(
        self,
        job: ExperimentJob,
        system: Any,
        bundle: DatasetBundle,
        fit: Any,
        model_spec: ModelSpec,
        parameter_count: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
        standardizer = fit.target_standardizer
        prediction = batched_predict(fit.model, bundle.x_test, standardizer)
        overall = overall_errors(bundle.y_test, prediction, standardizer.scale)
        point_error = pointwise_normalized_errors(
            bundle.y_test, prediction, standardizer.scale
        ).normalized_rmse
        residual = residual_norms(system, prediction, bundle.x_test)
        residual_summary = summarize_residual_norms(residual)

        valid_center = np.all(bundle.probe_valid, axis=1)
        probe_x = bundle.probe_x[valid_center]
        directions = bundle.probe_directions[valid_center]
        steps = bundle.probe_steps[valid_center]
        x_plus = probe_x[:, None, :] + steps[..., None] * directions
        x_minus = probe_x[:, None, :] - steps[..., None] * directions
        true_center = standardizer.transform(bundle.probe_y[valid_center])
        output_dim = bundle.y_train.shape[1]
        true_plus = standardizer.transform(
            bundle.probe_y_plus[valid_center].reshape(-1, output_dim)
        ).reshape(bundle.probe_y_plus[valid_center].shape)
        true_minus = standardizer.transform(
            bundle.probe_y_minus[valid_center].reshape(-1, output_dim)
        ).reshape(bundle.probe_y_minus[valid_center].shape)
        model_center = batched_predict(fit.model, probe_x, None)
        model_plus = batched_predict(fit.model, x_plus, None)
        model_minus = batched_predict(fit.model, x_minus, None)
        response = finite_response_metrics(
            true_center,
            true_plus,
            true_minus,
            model_center,
            model_plus,
            model_minus,
            step=steps,
            epsilon_ratio=0.05,
        )
        magnitude = magnitude_matching(response.true_kappa, response.model_kappa)
        allocation = spatial_allocation_metrics(
            response.true_kappa, response.model_kappa
        ) if len(response.true_kappa) >= 5 else None
        true_allocation = response.true_kappa / max(float(np.sum(response.true_kappa)), 1e-12)
        model_allocation = response.model_kappa / max(float(np.sum(response.model_kappa)), 1e-12)
        allocation_overlap = float(1.0 - 0.5 * np.sum(np.abs(true_allocation - model_allocation)))
        top_k = min(3, directions.shape[1], directions.shape[2])
        direction = topk_directional_alignment(
            directions,
            response.true_response / np.square(steps),
            response.model_response / np.square(steps),
            top_k=top_k,
        )

        probe_prediction_physical = batched_predict(fit.model, probe_x, standardizer)
        quintiles = errors_by_kappa_quantile(
            response.true_kappa,
            bundle.probe_y[valid_center],
            probe_prediction_physical,
            standardizer.scale,
            n_quantiles=min(5, len(response.true_kappa)),
        )
        probe_error = pointwise_normalized_errors(
            bundle.probe_y[valid_center], probe_prediction_physical, standardizer.scale
        ).normalized_rmse
        center_gap = np.mean(response.relative_gap, axis=1)
        center_absolute_gap = np.mean(response.absolute_gap, axis=1)
        center_log_mismatch = np.mean(np.abs(np.log(
            (response.model_response + response.epsilon)
            / (response.true_response + response.epsilon)
        )), axis=1)
        tail = curvature_tail_metrics(
            response.true_kappa, center_gap, probe_error, quantile=0.80,
            absolute_gap=center_absolute_gap,
            log_magnitude_mismatch=center_log_mismatch,
        )
        error_kappa = spearman_correlation(response.true_kappa, probe_error)
        gap_error = spearman_correlation(center_gap, probe_error)
        signed_representation = np.median(
            np.log(
                (response.model_response + response.epsilon)
                / (response.true_response + response.epsilon)
            )
        )

        row: dict[str, Any] = {
            "job_id": job.job_id,
            "family": job.family,
            "system": job.system,
            "model": job.model,
            "seed": job.seed,
            "variant": job.variant,
            "width": model_spec.width,
            "depth": model_spec.depth,
            "train_size": job.train_size or len(bundle.x_train),
            "parameter_count": parameter_count,
            "mae": overall.mae,
            "rmse": overall.rmse,
            "nmae": overall.normalized_mae,
            "nrmse": overall.nrmse,
            "residual_mean": residual_summary.mean,
            "residual_median": residual_summary.median,
            "residual_p95": float(np.quantile(residual, 0.95)),
            "gap_relative": float(np.mean(response.relative_gap)),
            "gap_vector": float(np.mean(response.relative_vector_gap)),
            "gap_absolute": float(np.mean(response.absolute_gap)),
            "underrepresentation": float(signed_representation),
            "magnitude_ratio": magnitude.median_ratio,
            "magnitude_log_mismatch": magnitude.median_absolute_log_ratio,
            "magnitude_log_ratio": magnitude.median_absolute_log_ratio,
            "underrepresented_fraction": magnitude.underrepresented_fraction,
            "allocation_spearman": allocation.correlation.rho if allocation else float("nan"),
            "allocation_concentration": allocation.q5_q1_concentration if allocation else float("nan"),
            "allocation_overlap": allocation_overlap,
            "direction_alignment": direction.mean,
            "error_kappa_spearman": error_kappa.rho,
            "gap_error_spearman": gap_error.rho,
            "curvature_tail_threshold": tail.threshold,
            "curvature_tail_gap_mean": tail.gap_mean,
            "curvature_tail_error_mean": tail.error_mean,
            "curvature_tail_gap_ratio": tail.tail_to_global_gap_ratio,
            "curvature_tail_error_ratio": tail.tail_to_global_error_ratio,
            "curvature_tail_gap_error_spearman": tail.gap_to_error_spearman.rho,
            "absolute_response_gap": tail.absolute_gap_mean,
            "log_magnitude_mismatch": tail.log_magnitude_mismatch,
            "q80_absolute_response_gap": tail.tail_absolute_gap_mean,
            "true_curvature_median": float(np.median(response.true_kappa)),
            "true_curvature_mean": float(np.mean(response.true_kappa)),
            "true_response_median": float(np.median(response.true_response)),
            "probe_center_count": int(response.true_response.shape[0]),
            "probe_direction_count": int(response.true_response.shape[1]),
            "q5_q1_ratio": quintiles[-1].errors.nrmse / max(
                quintiles[0].errors.nrmse, 1e-12
            ),
            "best_val_mse": fit.best_val_loss,
            "best_step": fit.best_step,
            "training_seconds": fit.elapsed_seconds,
            "geometry_probe_count": int(valid_center.sum()),
            "probe_failure_rate": float(1.0 - bundle.probe_valid.mean()),
            "system_kwargs": json.dumps(job.kwargs, sort_keys=True),
        }
        quintile_rows: list[dict[str, Any]] = []
        for quantile in quintiles:
            index = int(quantile.label[1:])
            row[f"q{index}_nrmse"] = quantile.errors.nrmse
            row[f"q{index}_nmae"] = quantile.errors.normalized_mae
            quintile_rows.append(
                {
                    "job_id": job.job_id,
                    "family": job.family,
                    "system": job.system,
                    "model": job.model,
                    "seed": job.seed,
                    "variant": job.variant,
                    "quintile": quantile.label,
                    "count": quantile.count,
                    "kappa_mean": float(
                        np.mean(response.true_kappa[
                            np.argsort(np.argsort(response.true_kappa, kind="stable"), kind="stable")
                            * 5 // len(response.true_kappa)
                            == index - 1
                        ])
                    ),
                    "kappa_median": quantile.kappa_median,
                    "nrmse": quantile.errors.nrmse,
                    "nmae": quantile.errors.normalized_mae,
                    "mae": quantile.errors.mae,
                    "rmse": quantile.errors.rmse,
                }
            )
        diagnostics = {
            "test_point_nrmse": np.asarray(point_error, dtype=np.float32),
            "residual_norm": np.asarray(residual, dtype=np.float32),
            "probe_kappa_true": np.asarray(response.true_kappa, dtype=np.float32),
            "probe_kappa_model": np.asarray(response.model_kappa, dtype=np.float32),
            "probe_nrmse": np.asarray(probe_error, dtype=np.float32),
            "probe_gap": np.asarray(center_gap, dtype=np.float32),
            "probe_absolute_gap": np.asarray(center_absolute_gap, dtype=np.float32),
            "probe_log_magnitude_mismatch": np.asarray(center_log_mismatch, dtype=np.float32),
            "probe_response_gap": np.asarray(center_gap, dtype=np.float32),
            "probe_curvature_tail_mask": np.asarray(
                response.true_kappa >= tail.threshold, dtype=np.bool_
            ),
            "probe_curvature_tail_error": np.asarray(probe_error, dtype=np.float32),
            "direction_alignment": np.asarray(direction.per_point, dtype=np.float32),
        }
        return row, quintile_rows, diagnostics

    def run_job(self, job: ExperimentJob, *, force: bool = False) -> dict[str, Any]:
        result_path = self.job_dir / f"{job.job_id}.json"
        if result_path.exists() and not force:
            return json.loads(result_path.read_text(encoding="utf-8"))["run"]
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START {job.job_id}", flush=True)
        system, bundle, arrays = self._training_inputs(job)
        spec = self._model_spec(job, system.input_dim, system.output_dim)
        seed_everything(job.seed)
        model = build_model(spec, system.input_dim, system.output_dim)
        parameter_count = count_parameters(model)
        config = self._train_config(job)
        fit = fit_model(model, arrays, config, job.seed, device=self.device)
        row, quintiles, diagnostics = self._evaluate(
            job, system, bundle, fit, spec, parameter_count
        )

        checkpoint = checkpoint_payload(
            fit,
            metadata={
                "job": asdict(job),
                "model_spec": asdict(spec),
                "train_config": asdict(config),
                "profile": self.profile.name,
            },
        )
        torch.save(checkpoint, self.checkpoint_dir / f"{job.job_id}.pt")
        np.savez_compressed(self.diagnostic_dir / f"{job.job_id}.npz", **diagnostics)
        payload = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run": row,
            "quintiles": quintiles,
        }
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, result_path)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] DONE  {job.job_id} "
            f"NRMSE={row['nrmse']:.5g} Q5/Q1={row['q5_q1_ratio']:.3f}",
            flush=True,
        )
        return row

    def consolidate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        runs: list[dict[str, Any]] = []
        quintiles: list[dict[str, Any]] = []
        expected = {job.job_id for job in enumerate_jobs(self.profile)}
        for path in sorted(self.job_dir.glob("*.json")):
            if path.stem not in expected:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.append(payload["run"])
            quintiles.extend(payload["quintiles"])
        run_frame = pd.DataFrame(runs)
        quintile_frame = pd.DataFrame(quintiles)
        run_frame.to_csv(self.result_dir / "runs.csv", index=False)
        quintile_frame.to_csv(self.result_dir / "quintiles.csv", index=False)
        return run_frame, quintile_frame

    def run(
        self,
        *,
        families: Iterable[str] | None = None,
        force: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        requested = set(families or ())
        jobs = enumerate_jobs(self.profile)
        if requested:
            jobs = [job for job in jobs if job.family in requested]
        for index, job in enumerate(jobs, start=1):
            print(f"PROGRESS {index}/{len(jobs)}", flush=True)
            self.run_job(job, force=force)
            self.consolidate()
        return self.consolidate()

    def status(self) -> dict[str, Any]:
        all_jobs = enumerate_jobs(self.profile)
        completed = {path.stem for path in self.job_dir.glob("*.json")}
        return {
            "profile": self.profile.name,
            "total": len(all_jobs),
            "completed": sum(job.job_id in completed for job in all_jobs),
            "remaining": sum(job.job_id not in completed for job in all_jobs),
            "result_dir": str(self.result_dir),
        }
