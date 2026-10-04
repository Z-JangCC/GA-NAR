"""Run the complete NRGD response-geometry study on five benchmarks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from implicit_surrogate.core.protocol import REFROZEN_PROTOCOL
from implicit_surrogate.data.standardization import Standardizer
from implicit_surrogate.ga_nrl import (
    build_local_mode_geometry,
    build_mode_capacity_targets,
    evaluate_allocation_router,
    fit_allocation_router,
    fit_best_fraction_router,
    make_train_validation_indices,
    validate_capacity_target,
    constant_target_metrics,
    evaluate_allocation_predictions,
    fit_scalar_capacity_router,
    random_geometry_basis,
)
from implicit_surrogate.sensitivity.directional_jacobian_variation import estimate_directional_variation
from implicit_surrogate.sensitivity.directions import generate_geometry_directions
from implicit_surrogate.sensitivity.dominant_subspace import identify_dominant_subspace
from implicit_surrogate.sensitivity.spectral_projection import project_sensitivity_spectrum
from implicit_surrogate.systems import (
    AllenCahn2DEquilibriumSystem,
    CoupledDuffingEquilibriumSystem,
    ShallowWater2DFreeSurfaceSystem,
)
from implicit_surrogate.systems.power_flow import ACPowerFlowSystem, RichIEEE118PowerFlowSystem
from implicit_surrogate.systems.synthetic import SyntheticImplicitSystem


SPECS = {
    "synthetic": (SyntheticImplicitSystem, 2025),
    "ieee118_rich": (RichIEEE118PowerFlowSystem, 4321),
    "coupled_duffing": (CoupledDuffingEquilibriumSystem, 6101),
    "allen_cahn_2d": (AllenCahn2DEquilibriumSystem, 6102),
    "shallow_water_2d": (ShallowWater2DFreeSurfaceSystem, 6103),
}

# Distinct, preregistered budgets reflect the intended complexity/continuation
# cost of each system. These are not post-hoc choices made from the results.
BENCHMARK_BUDGETS = {
    "synthetic": {"centers": 1024, "directions": 12, "rank_cap": 5, "rank_floor": 5},
    "ieee118_rich": {"centers": 1024, "directions": 12, "rank_cap": 6, "rank_floor": 6},
    "coupled_duffing": {"centers": 1024, "directions": 16, "rank_cap": 12, "rank_floor": 10},
    "allen_cahn_2d": {"centers": 1536, "directions": 12, "rank_cap": 14, "rank_floor": 10},
    "shallow_water_2d": {"centers": 1280, "directions": 16, "rank_cap": 16, "rank_floor": 12},
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _metric_columns(metrics: dict[str, float | bool]) -> dict[str, float | bool]:
    """Select the comparable allocation diagnostics for result tables."""
    return {
        "allocation_mse": metrics["mse"],
        "capacity_mse": metrics["capacity_mse"],
        "mode_fraction_mse": metrics["mode_fraction_mse"],
        "mode_fraction_normalized_mse": metrics["mode_fraction_normalized_mse"],
        "mode_fraction_r2": metrics["mode_fraction_r2"],
        "mode_fraction_mean_correlation": metrics["mode_fraction_mean_correlation"],
        "mode_fraction_correlations": json.dumps(metrics.get("mode_fraction_correlations", []), allow_nan=True),
        "predicted_simplex_error": metrics["max_conservation_error"],
        "mean_capacity": metrics["mean_target_capacity"],
    }


def _artifact(root: Path, benchmark: str, target, geometry, protocol_id: str, population: str, inputs: np.ndarray | None = None) -> None:
    path = root / "artifacts" / f"{benchmark}_allocation_target.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mode_strengths": target.mode_strengths,
        "complement_strength": target.complement_strength,
        "total_strength": target.total_strength,
        "total_capacity": target.total_capacity,
        "mode_fractions": target.mode_fractions,
        "weights": target.weights,
        "valid_centers": target.valid_centers,
        "basis": target.basis,
    }
    if inputs is not None:
        arrays["standardized_inputs"] = np.asarray(inputs, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    metadata = {
        "formal_flag": True,
        "benchmark": benchmark,
        "protocol": protocol_id,
        "allocation_protocol": "NRGD-FIVE-SYSTEMS",
        "population": population,
        "scale": geometry.scale,
        "epsilon": target.epsilon,
        "mean_strength": target.mean_strength,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name, value in arrays.items()},
    }
    path.with_suffix(".npz.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _dataset(system, seed: int, *, centers: int, seed_offset: int = 0, heterogeneous: bool = False, mode_heterogeneous: bool = False):
    # This allocation-only study consumes one center population.  Do not
    # solve unrelated prediction/validation/test populations here: doing so
    # multiplied the expensive implicit solves without contributing rows to
    # the GA-NRL target or its held-out evaluation.
    total = centers
    if hasattr(system, "sample_parameters"):
        parameters = system.sample_parameters(total, seed + seed_offset)
    else:
        parameters = np.random.default_rng(seed + seed_offset).normal(size=(total, system.parameter_dimension))
    base = np.asarray(getattr(system, "base_parameter", np.zeros(system.parameter_dimension)), dtype=float)
    if heterogeneous:
        rng = np.random.default_rng(seed + seed_offset + 91000)
        regimes = rng.choice(np.asarray([0.75, 1.0, 1.25, 1.5]), size=total, p=[0.15, 0.35, 0.35, 0.15])
        if hasattr(system, "num_zones"):
            # For AC power flow, vary only regional load multipliers. Scaling
            # redispatch, voltage setpoints, and shunts together can leave the
            # prescribed continuation branch and is not a valid regime.
            parameters[:, :system.num_zones] = base[None, :system.num_zones] + regimes[:, None] * (parameters[:, :system.num_zones] - base[None, :system.num_zones])
            parameters[:, :system.num_zones] = np.clip(parameters[:, :system.num_zones], 0.55, 1.45)
        else:
            parameters = base[None, :] + regimes[:, None] * (parameters - base[None, :])
    if mode_heterogeneous:
        rng = np.random.default_rng(seed + seed_offset + 92000)
        regime = rng.integers(0, 4, size=total)
        if hasattr(system, "nonlinear_projection_matrix"):
            # Synthetic: alter nonlinear coordinates anisotropically, so the
            # relative mode fractions—not only total magnitude—change.
            projection = np.asarray(system.nonlinear_projection_matrix)
            projected = parameters @ projection.T
            patterns = np.asarray([[1.45, .75, .75, .75, .75], [.75, 1.45, .75, .75, .75], [.75, .75, 1.45, .75, .75], [.75, .75, .75, 1.45, .75]])
            transformed = projected * patterns[regime]
            parameters += (transformed - projected) @ projection
        elif hasattr(system, "num_zones"):
            # Rich AC power flow: apply mild, complementary regional-load
            # regimes while leaving redispatch/voltage/shunt controls intact.
            zone_patterns = np.asarray([[1.15] * 6 + [.85] * 6, [.85] * 6 + [1.15] * 6, [1.15, .85] * 6, [.85, 1.15] * 6])
            parameters[:, :system.num_zones] = base[None, :system.num_zones] + zone_patterns[regime] * (parameters[:, :system.num_zones] - base[None, :system.num_zones])
            parameters[:, :system.num_zones] = np.clip(parameters[:, :system.num_zones], 0.55, 1.45)
    states = system.solve_states(parameters)
    q = parameters
    z = states
    # Fit preprocessing statistics on the exact protocol-training rows used
    # by the allocator.  This keeps the validation path frozen even for
    # unsupervised centering/scaling statistics.
    preprocessing_train, _ = make_train_validation_indices(centers, seed=17000 + seed)
    standardizer = Standardizer.fit(q[preprocessing_train], z[preprocessing_train])
    return standardizer.statistics, standardizer.transform_input(q), q, z


def _scale_reliability(variation, scales: tuple[float, ...]) -> list[dict]:
    rows = []
    for scale in scales:
        estimate = variation.sensitivity_estimates[scale]
        if variation.split_matrices is not None and scale in variation.split_matrices:
            first, second = variation.split_matrices[scale]
        else:
            first, second = estimate.centered_matrix, estimate.centered_matrix
        denom = 0.5 * (np.linalg.norm(first) + np.linalg.norm(second)) + 1e-12
        relative = float(np.linalg.norm(first - second) / denom)
        cosine = float(np.sum(first * second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12))
        rows.append({"scale": scale, "invalid_pair_rate": variation.invalid_rate, "split_half_relative_error": relative, "split_half_cosine": cosine, "reliable": bool(variation.invalid_rate <= 0.05 and relative <= 0.35 and cosine >= 0.90)})
    return rows


def run_one(name: str, *, mode: str, root: Path, centers: int, directions_per_center: int, rank_cap: int, rank_floor: int, seed: int, heterogeneous: bool = False, mode_heterogeneous: bool = False) -> tuple[list[dict], dict, list[dict], list[dict], list[dict]]:
    system = SPECS[name][0]()
    stats, centers_std, center_parameters, center_states = _dataset(system, seed, centers=centers, heterogeneous=heterogeneous, mode_heterogeneous=mode_heterogeneous)
    directions = generate_geometry_directions(centers, system.parameter_dimension, directions_per_center=directions_per_center, num_rademacher_probes=0, seed=7000 + seed)
    scales = (0.02, 0.01, 0.005, 0.0025)
    solve_target = (lambda parameter, initial_state, center_parameter: system.continue_between(center_parameter, initial_state, parameter)) if name == "ieee118_rich" else (lambda parameter, initial_state, center_parameter: system.solve_state(parameter, initial_state=initial_state))
    variation = estimate_directional_variation(system, centers_std, center_states, center_parameters, stats.q_std, stats.z_std, stats.retained_q_indices, directions.unit_directions, directions.rademacher_probes, scales=scales, solve_target=solve_target, estimator="full_jacobian_matrix", parallel_workers=4 if name == "ieee118_rich" else 1)
    reliability = _scale_reliability(variation, scales)
    selected_scale = next((row["scale"] for row in reliability if row["reliable"]), 0.02)
    estimate = variation.sensitivity_estimates[selected_scale]
    # All five allocation runs use exact full-Jacobian second moments. The
    # historical simplex spectral projection was designed for noisy
    # Hutchinson estimates and can collapse a high-dimensional physical
    # spectrum to rank one. For allocation-only full-Jacobian evidence we
    # retain the PSD normalized second moment directly.
    physical_spectrum = 0.5 * (estimate.normalized_matrix + estimate.normalized_matrix.T)
    # Allow field benchmarks to expose their genuinely high-dimensional
    # geometry while keeping a common finite expert budget for allocation.
    subspace = identify_dominant_subspace(physical_spectrum, rank_cap=min(rank_cap, system.parameter_dimension), rank_floor=min(rank_floor, system.parameter_dimension))
    energy = variation.energies[selected_scale].reshape(centers, directions_per_center)
    mask = variation.valid_pair_mask.reshape(centers, directions_per_center)
    geometry = build_local_mode_geometry(energy, directions.unit_directions, mask, scale=selected_scale)
    valid = geometry.valid_centers
    geometry_total_strength = np.trace(geometry.matrices, axis1=1, axis2=2)
    protocol_train_idx, protocol_valid_idx = make_train_validation_indices(int(valid.sum()), seed=17000 + seed)
    mean_strength = float(np.mean(geometry_total_strength[valid][protocol_train_idx]))
    target = build_mode_capacity_targets(geometry, subspace.basis, mean_strength=mean_strength)
    report = validate_capacity_target(target)
    valid = target.valid_centers & np.isfinite(target.weights).all(axis=1)
    x, y = centers_std[valid], target.weights[valid]
    rows = []
    comparison_rows = []
    oracle_rows = []
    for router_seed in (0, 1, 2):
        train_idx, eval_idx = make_train_validation_indices(len(x), seed=17000 + seed)
        fit = fit_best_fraction_router(x, y, seed=router_seed, sensitivity_basis=target.basis, mode_eigenvalues=subspace.eigenvalues[:target.rank], split_indices=(train_idx, eval_idx))
        eval_idx = fit.validation_indices if fit.validation_indices is not None else np.arange(len(x))
        metrics = evaluate_allocation_router(fit.router, x[eval_idx], y[eval_idx])
        population_label = "mode_heterogeneous_regimes" if mode_heterogeneous else ("heterogeneous_regimes" if heterogeneous else "nominal")
        common = {"benchmark": name, "population": population_label, "seed": router_seed, "rank": target.rank, "rank_cap": rank_cap, "rank_floor": rank_floor, "centers": int(valid.sum()), "directions_per_center": directions_per_center, "selected_scale": selected_scale, "target_conservation_error": report["conservation_max_error"], "r90": subspace.r90, "captured_mass": subspace.captured_mass}
        full_row = {**common, "method": "ga_nrl_geometry_router", "status": fit.status, **_metric_columns(metrics), "validation_loss": fit.validation_loss, "train_loss": fit.train_loss}
        rows.append(full_row); comparison_rows.append(full_row)

        input_fit = fit_allocation_router(x, y, hidden_width=128, max_epochs=300, patience=30, seed=router_seed, split_indices=(train_idx, eval_idx))
        input_idx = input_fit.validation_indices if input_fit.validation_indices is not None else np.arange(len(x))
        input_metrics = evaluate_allocation_router(input_fit.router, x[input_idx], y[input_idx])
        comparison_rows.append({**common, "method": "input_only_simplex_router", "status": input_fit.status, **_metric_columns(input_metrics), "validation_loss": input_fit.validation_loss, "train_loss": input_fit.train_loss})

        random_basis = random_geometry_basis(x.shape[1], target.rank, seed=10000 + router_seed)
        random_fit = fit_allocation_router(x, y, hidden_width=128, max_epochs=300, patience=30, seed=router_seed, sensitivity_basis=random_basis, mode_eigenvalues=subspace.eigenvalues[:target.rank], split_indices=(train_idx, eval_idx))
        random_idx = random_fit.validation_indices if random_fit.validation_indices is not None else np.arange(len(x))
        random_metrics = evaluate_allocation_router(random_fit.router, x[random_idx], y[random_idx])
        comparison_rows.append({**common, "method": "random_geometry_router", "status": random_fit.status, **_metric_columns(random_metrics), "validation_loss": random_fit.validation_loss, "train_loss": random_fit.train_loss})

        scalar_model = fit_scalar_capacity_router(x, y, seed=router_seed, max_epochs=300, patience=30, hidden_width=128)
        scalar_idx = getattr(scalar_model, "validation_indices", np.arange(len(x)))
        scalar_prediction = scalar_model(torch.as_tensor(x[scalar_idx], dtype=torch.float32)).detach().cpu().numpy()
        scalar_metrics = evaluate_allocation_predictions(scalar_prediction, y[scalar_idx])
        comparison_rows.append({**common, "method": "scalar_capacity_uniform_router", "status": "RESOLVED", **_metric_columns(scalar_metrics), "validation_loss": np.nan, "train_loss": np.nan})

        # Formal constant baseline: estimate the simplex mean on the training
        # partition only, then evaluate it on the held-out GA-NRL validation
        # partition.  The old validation-mean construction is retained below
        # as an explicit oracle audit row, never as a formal comparator.
        constant_idx = eval_idx
        constant_prediction = constant_target_metrics(y[train_idx])
        constant_metrics = evaluate_allocation_predictions(np.repeat(constant_prediction[:1], len(constant_idx), axis=0), y[constant_idx])
        comparison_rows.append({**common, "method": "constant_mean_target", "status": "RESOLVED", "partition": "train_mean_evaluated_on_validation", **_metric_columns(constant_metrics), "validation_loss": np.nan, "train_loss": np.nan})

        oracle_prediction = constant_target_metrics(y[constant_idx])
        oracle_metrics = evaluate_allocation_predictions(np.repeat(oracle_prediction[:1], len(constant_idx), axis=0), y[constant_idx])
        oracle_rows.append({**common, "method": "constant_validation_mean_oracle", "status": "AUDIT_ONLY", "partition": "validation_mean_evaluated_on_validation", **_metric_columns(oracle_metrics)})
    uniform_fraction = np.full(target.weights.shape[1] - 1, 1.0 / (target.weights.shape[1] - 1))
    uniform = np.concatenate([1.0 - target.total_capacity[:, None], target.total_capacity[:, None] * uniform_fraction[None, :]], axis=1)
    control = {"uniform_mode_fraction_mse": float(np.mean((uniform[valid] - target.weights[valid]) ** 2)), "uniform_mode_fraction": "unstructured_equal_mode_capacity"}
    diagnostics = {"benchmark": name, "population": population_label, "mean_strength_train": mean_strength, "status": "RESOLVED" if report["conservation_pass"] and np.all([row["status"] == "RESOLVED" for row in rows]) else "UNRESOLVED", "rank": target.rank, "r90": subspace.r90, "captured_mass": subspace.captured_mass, "selected_scale": selected_scale, "valid_centers": int(valid.sum()), "geometry": report, "control": control, "system_state_dimension": system.state_dimension, "system_parameter_dimension": system.parameter_dimension}
    _artifact(root, name, target, geometry, "NRGD-FIVE-SYSTEMS-LEAKAGE-FIXED", f"{centers}_centers_x_{directions_per_center}_directions", inputs=centers_std)
    return rows, diagnostics, [{"benchmark": name, **item} for item in reliability], comparison_rows, oracle_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output-root", default="results/ga_nrl_five")
    parser.add_argument("--benchmark", choices=("all", *SPECS), default="all")
    parser.add_argument("--centers", type=int, default=None, help="override preregistered center budget for every benchmark")
    parser.add_argument("--directions-per-center", type=int, default=None, help="override preregistered direction budget for every benchmark")
    parser.add_argument("--population", choices=("nominal", "heterogeneous", "mode_heterogeneous"), default="nominal")
    args = parser.parse_args()
    root = Path(args.output_root)
    names = list(SPECS) if args.benchmark == "all" else [args.benchmark]
    rows, diagnostics, reliability, comparisons, oracle_comparisons = [], {}, [], [], []
    for index, name in enumerate(names):
        budget = BENCHMARK_BUDGETS[name]
        if args.centers is not None:
            budget = {**budget, "centers": args.centers}
        if args.directions_per_center is not None:
            budget = {**budget, "directions": args.directions_per_center}
        centers = budget["centers"]
        row, diag, rel, comp, oracle = run_one(name, mode=args.mode, root=root, centers=centers, directions_per_center=budget["directions"], rank_cap=budget["rank_cap"], rank_floor=budget["rank_floor"], seed=SPECS[name][1] + index, heterogeneous=args.population == "heterogeneous", mode_heterogeneous=args.population == "mode_heterogeneous")
        rows.extend(row); diagnostics[name] = diag; reliability.extend(rel); comparisons.extend(comp); oracle_comparisons.extend(oracle)
    _write_csv(root / "allocation_router.csv", rows)
    _write_csv(root / "geometry_reliability.csv", reliability)
    _write_csv(root / "baseline_comparison.csv", comparisons)
    _write_csv(root / "baseline_comparison_oracle.csv", oracle_comparisons)
    (root / "allocation_router.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))
    print(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
