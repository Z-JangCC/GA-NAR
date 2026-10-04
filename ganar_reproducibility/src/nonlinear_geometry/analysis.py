"""Statistical summaries and a Chinese report for experiment results.

The analysis intentionally separates the replicated ``main`` experiment from
the mostly single-seed mechanism, scaling, allocation, and ablation studies.
All calculations are deterministic, including the paired bootstrap intervals.
Missing optional columns or experiment families are recorded in the report and
JSON summary rather than causing the complete analysis to fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MAIN_METRICS = (
    "nrmse",
    "q5_nrmse",
    "q5_q1_ratio",
    "gap_relative",
    "residual_mean",
    "error_kappa_spearman",
    "gap_error_spearman",
    "direction_alignment",
    "curvature_tail_gap_ratio",
    "curvature_tail_error_ratio",
    "curvature_tail_gap_error_spearman",
)
PAIRED_METRICS = ("nrmse", "q5_nrmse", "gap_relative", "residual_mean")
MECHANISM_METRICS = ("nrmse", "q5_q1_ratio", "gap_relative")
ALLOCATION_MODES = ("aligned", "uniform", "shuffled", "reverse")
ABLATION_VARIANTS = ("with_router", "no_router", "no_geom", "no_gate", "no_physics")


def _finite(values: Iterable[Any]) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    array = numeric.to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _mean_std(values: Iterable[Any]) -> tuple[float, float, int]:
    array = _finite(values)
    if not len(array):
        return math.nan, math.nan, 0
    std = float(np.std(array, ddof=1)) if len(array) > 1 else math.nan
    return float(np.mean(array)), std, int(len(array))


def _display(mean: float, std: float, n: int, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "NA"
    if n > 1 and np.isfinite(std):
        return f"{mean:.{digits}g} ± {std:.{digits}g}"
    return f"{mean:.{digits}g} (n={n})"


def _safe_spearman(x: Iterable[Any], y: Iterable[Any]) -> tuple[float, float, int, str]:
    left = pd.to_numeric(pd.Series(list(x), dtype="object"), errors="coerce").to_numpy(float)
    right = pd.to_numeric(pd.Series(list(y), dtype="object"), errors="coerce").to_numpy(float)
    count = min(len(left), len(right))
    left, right = left[:count], right[:count]
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2:
        return math.nan, math.nan, int(len(left)), "insufficient_data"
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return math.nan, math.nan, int(len(left)), "constant_input"
    result = spearmanr(left, right)
    return float(result.statistic), float(result.pvalue), int(len(left)), "ok"


def _relative_percent(reference: float, candidate: float) -> float:
    """Return positive percent when a lower-is-better candidate improves."""

    if not (np.isfinite(reference) and np.isfinite(candidate)) or reference == 0:
        return math.nan
    return float(100.0 * (reference - candidate) / abs(reference))


def _degradation_percent(reference: float, candidate: float) -> float:
    if not (np.isfinite(reference) and np.isfinite(candidate)) or reference == 0:
        return math.nan
    return float(100.0 * (candidate - reference) / abs(reference))


def _stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little") ^ int(seed)
    return np.random.default_rng(derived)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    key: Sequence[str],
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    if len(array) == 1 or samples <= 1:
        value = float(np.mean(array))
        return value, value
    rng = _stable_rng(seed, *key)
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _add_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str], warnings: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    for column in missing:
        frame[column] = np.nan
    if missing:
        _add_warning(warnings, f"runs.csv 缺少列：{', '.join(missing)}；相关统计标记为不可用。")


def _enrich_quintiles(
    runs: pd.DataFrame, quintiles: pd.DataFrame, warnings: list[str]
) -> pd.DataFrame:
    result = runs.copy()
    if quintiles.empty:
        _add_warning(warnings, "quintiles.csv 缺失或为空；优先使用 runs.csv 中的 Q1/Q5 列。")
    required = {"job_id", "quintile", "nrmse"}
    if not quintiles.empty and required.issubset(quintiles.columns) and "job_id" in result.columns:
        q = quintiles.copy()
        q["quintile"] = q["quintile"].astype(str).str.upper()
        q["nrmse"] = pd.to_numeric(q["nrmse"], errors="coerce")
        q = q[q["quintile"].isin(("Q1", "Q5"))]
        if not q.empty:
            pivot = q.pivot_table(index="job_id", columns="quintile", values="nrmse", aggfunc="mean")
            pivot = pivot.rename(columns={"Q1": "_q1_from_file", "Q5": "_q5_from_file"})
            result = result.merge(pivot, how="left", left_on="job_id", right_index=True)
            for target, source in (("q1_nrmse", "_q1_from_file"), ("q5_nrmse", "_q5_from_file")):
                if target not in result:
                    result[target] = result[source]
                else:
                    result[target] = pd.to_numeric(result[target], errors="coerce").fillna(result[source])
            result = result.drop(columns=[column for column in pivot.columns if column in result])
    elif not quintiles.empty:
        _add_warning(warnings, "quintiles.csv 缺少 job_id/quintile/nrmse，无法回填 Q1/Q5。")
    for column in ("q1_nrmse", "q5_nrmse"):
        if column not in result:
            result[column] = np.nan
    q1 = pd.to_numeric(result["q1_nrmse"], errors="coerce")
    q5 = pd.to_numeric(result["q5_nrmse"], errors="coerce")
    computed = q5 / q1.where(q1.abs() > 0)
    if "q5_q1_ratio" not in result:
        result["q5_q1_ratio"] = computed
    else:
        result["q5_q1_ratio"] = pd.to_numeric(result["q5_q1_ratio"], errors="coerce").fillna(computed)
    return result


def _main_aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["system", "model", "seed_count", "seeds", "three_seed_complete", "evidence_tier"]
    for metric in MAIN_METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_n", f"{metric}_mean_std"))
    main = runs[runs["family"].astype(str).str.lower().isin(("main", "modular_main"))].copy()
    rows: list[dict[str, Any]] = []
    for (system, model), group in main.groupby(["system", "model"], dropna=False, sort=True):
        seeds = sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().unique().tolist())
        row: dict[str, Any] = {
            "system": system,
            "model": model,
            "seed_count": len(seeds),
            "seeds": ",".join(f"{seed:g}" for seed in seeds),
            "three_seed_complete": len(seeds) >= 3,
            "evidence_tier": "主结论（3 seeds）" if len(seeds) >= 3 else "不完整重复/探索性",
        }
        for metric in MAIN_METRICS:
            mean, std, n = _mean_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_n"] = n
            row[f"{metric}_mean_std"] = _display(mean, std, n)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _modular_factorial(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize the plug-in benchmark without conflating loss conditions."""
    columns = [
        "system", "model", "variant", "geometry_loss", "seed_count",
        "parameter_count_mean", "nrmse_mean", "nrmse_std", "q5_nrmse_mean",
        "q5_q1_ratio_mean", "gap_relative_mean", "direction_alignment_mean",
    ]
    subset = runs[runs["family"].astype(str).str.lower() == "modular_main"].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    subset["geometry_loss"] = subset["variant"].astype(str).str.lower().eq("geom")
    rows = []
    for (system, model, variant), group in subset.groupby(["system", "model", "variant"], sort=True):
        rows.append({
            "system": system, "model": model, "variant": variant,
            "geometry_loss": bool(group["geometry_loss"].iloc[0]),
            "seed_count": int(pd.to_numeric(group["seed"], errors="coerce").nunique()),
            "parameter_count_mean": float(pd.to_numeric(group["parameter_count"], errors="coerce").mean()),
            "nrmse_mean": float(pd.to_numeric(group["nrmse"], errors="coerce").mean()),
            "nrmse_std": float(pd.to_numeric(group["nrmse"], errors="coerce").std(ddof=1)),
            "q5_nrmse_mean": float(pd.to_numeric(group["q5_nrmse"], errors="coerce").mean()),
            "q5_q1_ratio_mean": float(pd.to_numeric(group["q5_q1_ratio"], errors="coerce").mean()),
            "gap_relative_mean": float(pd.to_numeric(group["gap_relative"], errors="coerce").mean()),
            "direction_alignment_mean": float(pd.to_numeric(group["direction_alignment"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows, columns=columns)


def _paired_main(
    runs: pd.DataFrame,
    warnings: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "system", "baseline_model", "metric", "n_pairs", "paired_seeds",
        "mean_relative_improvement_pct", "std_relative_improvement_pct",
        "ci95_low_pct", "ci95_high_pct", "wins", "ties", "losses",
        "win_count", "three_seed_complete", "evidence_tier", "status",
    ]
    detail_columns = [
        "system", "baseline_model", "metric", "seed", "baseline_value",
        "ganr_value", "relative_improvement_pct", "ganr_wins",
    ]
    main = runs[runs["family"].astype(str).str.lower() == "main"].copy()
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for system in sorted(main["system"].dropna().astype(str).unique()):
        system_rows = main[main["system"].astype(str) == system]
        baseline_rows = system_rows[system_rows["model"].astype(str).str.lower() != "ganr"]
        model_means = (
            baseline_rows.assign(_nrmse=pd.to_numeric(baseline_rows["nrmse"], errors="coerce"))
            .groupby("model", dropna=True)["_nrmse"].mean().dropna().sort_values(kind="stable")
        )
        baseline_model = str(model_means.index[0]) if len(model_means) else ""
        ganr_rows = system_rows[system_rows["model"].astype(str).str.lower() == "ganr"]
        if not baseline_model or ganr_rows.empty:
            reason = "missing_baseline" if not baseline_model else "missing_ganr"
            _add_warning(warnings, f"主实验 {system} 无法配对：{reason}。")
            for metric in PAIRED_METRICS:
                rows.append({"system": system, "baseline_model": baseline_model, "metric": metric,
                             "n_pairs": 0, "paired_seeds": "", "win_count": "0/0",
                             "three_seed_complete": False, "evidence_tier": "不可用", "status": reason})
            continue
        base = baseline_rows[baseline_rows["model"].astype(str) == baseline_model]
        for metric in PAIRED_METRICS:
            left = base.groupby("seed", dropna=False)[metric].mean().rename("baseline_value")
            right = ganr_rows.groupby("seed", dropna=False)[metric].mean().rename("ganr_value")
            paired = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            paired = paired[paired["baseline_value"].abs() > 0]
            improvements: list[float] = []
            wins = ties = losses = 0
            for seed, pair in paired.sort_index().iterrows():
                baseline_value, ganr_value = float(pair["baseline_value"]), float(pair["ganr_value"])
                improvement = _relative_percent(baseline_value, ganr_value)
                if np.isfinite(improvement):
                    improvements.append(improvement)
                tolerance = 1e-12 * max(1.0, abs(baseline_value), abs(ganr_value))
                if ganr_value < baseline_value - tolerance:
                    wins += 1
                    outcome: Any = True
                elif ganr_value > baseline_value + tolerance:
                    losses += 1
                    outcome = False
                else:
                    ties += 1
                    outcome = "tie"
                details.append({
                    "system": system, "baseline_model": baseline_model, "metric": metric,
                    "seed": seed, "baseline_value": baseline_value, "ganr_value": ganr_value,
                    "relative_improvement_pct": improvement, "ganr_wins": outcome,
                })
            mean, std, n = _mean_std(improvements)
            low, high = _bootstrap_mean_ci(
                improvements, samples=bootstrap_samples, seed=bootstrap_seed,
                key=(system, baseline_model, metric),
            )
            status = "ok" if n else "no_matched_finite_pairs"
            rows.append({
                "system": system, "baseline_model": baseline_model, "metric": metric,
                "n_pairs": n, "paired_seeds": ",".join(str(value) for value in paired.index.tolist()),
                "mean_relative_improvement_pct": mean,
                "std_relative_improvement_pct": std,
                "ci95_low_pct": low, "ci95_high_pct": high,
                "wins": wins, "ties": ties, "losses": losses,
                "win_count": f"{wins}/{n}", "three_seed_complete": n >= 3,
                "evidence_tier": "主结论（3 seeds）" if n >= 3 else "不完整重复/探索性",
                "status": status,
            })
    return pd.DataFrame(rows, columns=summary_columns), pd.DataFrame(details, columns=detail_columns)


def _geometry_summary(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["system", "model", "seed_count", "three_seed_complete"]
    metrics = (
        "error_kappa_spearman", "gap_error_spearman", "q5_q1_ratio",
        "curvature_tail_gap_ratio", "curvature_tail_error_ratio",
        "curvature_tail_gap_error_spearman",
    )
    for metric in metrics:
        columns.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_n", f"{metric}_mean_std"))
    main = runs[runs["family"].astype(str).str.lower() == "main"]
    rows: list[dict[str, Any]] = []
    for (system, model), group in main.groupby(["system", "model"], dropna=False, sort=True):
        seed_count = int(pd.to_numeric(group["seed"], errors="coerce").nunique())
        row: dict[str, Any] = {"system": system, "model": model, "seed_count": seed_count,
                               "three_seed_complete": seed_count >= 3}
        for metric in metrics:
            mean, std, n = _mean_std(group[metric])
            row.update({f"{metric}_mean": mean, f"{metric}_std": std, f"{metric}_n": n,
                        f"{metric}_mean_std": _display(mean, std, n)})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _failure_boundary_summary(runs: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Identify operating regions where GANR stops being competitive.

    The boundary is deliberately empirical and auditable: for each system and
    mechanism factor, compare GANR with the best non-GANR run at the same
    level.  ``winner`` is never imputed when a matched comparison is absent.
    """

    columns = [
        "system", "factor", "level", "ganr_nrmse", "baseline_model",
        "baseline_nrmse", "relative_improvement_pct", "ganr_wins", "status",
    ]
    subset = runs[runs["family"].astype(str).str.lower().str.startswith("mechanism_")].copy()
    if subset.empty:
        _add_warning(warnings, "未找到 mechanism 扫描，无法生成 GANR 失效边界表。")
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    subset["_factor"] = subset["family"].astype(str).str.replace("mechanism_", "", regex=False)
    for (system, factor, level), group in subset.groupby(["system", "_factor", "variant"], dropna=False):
        group = group.copy()
        group["_nrmse"] = pd.to_numeric(group["nrmse"], errors="coerce")
        ganr = group[group["model"].astype(str).str.lower() == "ganr"]
        baselines = group[group["model"].astype(str).str.lower() != "ganr"]
        if ganr.empty or baselines.empty:
            rows.append({"system": system, "factor": factor, "level": level, "status": "unmatched"})
            continue
        ganr_mean = float(ganr["_nrmse"].mean())
        baseline_means = baselines.groupby("model")["_nrmse"].mean().dropna().sort_values()
        if baseline_means.empty:
            rows.append({"system": system, "factor": factor, "level": level, "status": "invalid"})
            continue
        baseline_model = str(baseline_means.index[0])
        baseline = float(baseline_means.iloc[0])
        rows.append({
            "system": system, "factor": factor, "level": level,
            "ganr_nrmse": ganr_mean, "baseline_model": baseline_model,
            "baseline_nrmse": baseline,
            "relative_improvement_pct": _relative_percent(baseline, ganr_mean),
            "ganr_wins": bool(ganr_mean < baseline), "status": "ok",
        })
    return pd.DataFrame(rows, columns=columns)


def _scaling_extremes(runs: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    columns = [
        "factor", "family", "system", "model", "low_level", "high_level",
        "low_n", "high_n", "nrmse_low", "nrmse_high", "nrmse_change_pct",
        "mean_error_decreased", "q5_q1_low", "q5_q1_high",
        "high_nonlinearity_penalty_persists", "evidence_tier", "status",
    ]
    specifications = (
        ("width", "scale_width", "width"),
        ("depth", "scale_depth", "depth"),
        ("samples", "scale_samples", "train_size"),
    )
    rows: list[dict[str, Any]] = []
    families = runs["family"].astype(str).str.lower()
    for factor, family, level_column in specifications:
        subset = runs[families == family].copy()
        if subset.empty:
            rows.append({"factor": factor, "family": family, "evidence_tier": "不可用",
                         "status": "missing_family"})
            _add_warning(warnings, f"未找到 {family}，无法分析 {factor} 极端设置。")
            continue
        subset["_level"] = pd.to_numeric(subset[level_column], errors="coerce")
        for (system, model), group in subset.groupby(["system", "model"], dropna=False, sort=True):
            levels = np.sort(group["_level"].dropna().unique())
            if len(levels) < 2:
                rows.append({"factor": factor, "family": family, "system": system, "model": model,
                             "low_level": levels[0] if len(levels) else math.nan,
                             "high_level": levels[-1] if len(levels) else math.nan,
                             "evidence_tier": "探索性", "status": "insufficient_levels"})
                continue
            low_level, high_level = float(levels[0]), float(levels[-1])
            low_group, high_group = group[group["_level"] == low_level], group[group["_level"] == high_level]
            low_error, _, low_n = _mean_std(low_group["nrmse"])
            high_error, _, high_n = _mean_std(high_group["nrmse"])
            low_ratio, _, _ = _mean_std(low_group["q5_q1_ratio"])
            high_ratio, _, _ = _mean_std(high_group["q5_q1_ratio"])
            seed_count = int(pd.to_numeric(pd.concat([low_group["seed"], high_group["seed"]]),
                                           errors="coerce").nunique())
            rows.append({
                "factor": factor, "family": family, "system": system, "model": model,
                "low_level": low_level, "high_level": high_level,
                "low_n": low_n, "high_n": high_n, "nrmse_low": low_error,
                "nrmse_high": high_error,
                "nrmse_change_pct": _degradation_percent(low_error, high_error),
                "mean_error_decreased": bool(high_error < low_error) if np.isfinite(low_error + high_error) else None,
                "q5_q1_low": low_ratio, "q5_q1_high": high_ratio,
                "high_nonlinearity_penalty_persists": bool(high_ratio > 1.0) if np.isfinite(high_ratio) else None,
                "evidence_tier": "多 seed 探索性" if seed_count >= 3 else "单 seed 探索性",
                "status": "ok",
            })
    return pd.DataFrame(rows, columns=columns)


def _factor_level(row: pd.Series, factor: str) -> float:
    raw = row.get("system_kwargs")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
            if factor in payload:
                return float(payload[factor])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    variant = str(row.get("variant", ""))
    prefix = f"{factor}_"
    if variant.startswith(prefix):
        try:
            return float(variant[len(prefix):])
        except ValueError:
            return math.nan
    return math.nan


def _cohen_d(low: np.ndarray, high: np.ndarray) -> float:
    low, high = low[np.isfinite(low)], high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2:
        return math.nan
    variance = ((len(low) - 1) * np.var(low, ddof=1) + (len(high) - 1) * np.var(high, ddof=1))
    variance /= len(low) + len(high) - 2
    if variance <= 0:
        return math.nan
    return float((np.mean(high) - np.mean(low)) / np.sqrt(variance))


def _mechanism_analysis(
    runs: pd.DataFrame, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_columns = ["factor", "family", "system", "model", "level", "seed_count"]
    for metric in MECHANISM_METRICS:
        level_columns.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_n"))
    trend_columns = [
        "factor", "family", "system", "model", "metric", "n_levels", "n_observations",
        "spearman_rho", "spearman_pvalue", "linear_slope_per_unit",
        "standardized_slope", "low_level", "high_level", "low_mean", "high_mean",
        "extreme_relative_change_pct", "extreme_log_ratio", "cohen_d_high_vs_low", "status",
    ]
    mechanism = runs[runs["family"].astype(str).str.startswith("mechanism_")].copy()
    mechanism = mechanism[mechanism["family"].astype(str).str.lower() != "mechanism_interaction"]
    if mechanism.empty:
        _add_warning(warnings, "未找到非 interaction 的 mechanism_* 实验。")
        return pd.DataFrame(columns=level_columns), pd.DataFrame(columns=trend_columns)
    mechanism["factor"] = mechanism["family"].astype(str).str.replace("mechanism_", "", n=1)
    mechanism["level"] = [
        _factor_level(row, str(row["factor"])) for _, row in mechanism.iterrows()
    ]
    invalid_factors = mechanism.loc[~np.isfinite(mechanism["level"]), "factor"].dropna().unique()
    for factor in invalid_factors:
        _add_warning(warnings, f"机制因子 {factor} 的部分 level 无法解析，已跳过。")
    valid = mechanism[np.isfinite(mechanism["level"])].copy()
    level_rows: list[dict[str, Any]] = []
    for (factor, family, system, model, level), group in valid.groupby(
        ["factor", "family", "system", "model", "level"], dropna=False, sort=True
    ):
        row: dict[str, Any] = {
            "factor": factor, "family": family, "system": system, "model": model,
            "level": level, "seed_count": int(pd.to_numeric(group["seed"], errors="coerce").nunique()),
        }
        for metric in MECHANISM_METRICS:
            mean, std, n = _mean_std(group[metric])
            row.update({f"{metric}_mean": mean, f"{metric}_std": std, f"{metric}_n": n})
        level_rows.append(row)
    trend_rows: list[dict[str, Any]] = []
    for (factor, family, system, model), group in valid.groupby(
        ["factor", "family", "system", "model"], dropna=False, sort=True
    ):
        for metric in MECHANISM_METRICS:
            data = group[["level", metric]].copy()
            data[metric] = pd.to_numeric(data[metric], errors="coerce")
            data = data.dropna()
            means = data.groupby("level", sort=True)[metric].mean()
            levels = means.index.to_numpy(dtype=float)
            values = means.to_numpy(dtype=float)
            rho, pvalue, _, rho_status = _safe_spearman(levels, values)
            slope = math.nan
            standardized = math.nan
            if len(levels) >= 2 and np.ptp(levels) > 0:
                slope = float(np.polyfit(levels, values, 1)[0])
                if np.std(values) > 0:
                    standardized = float(np.corrcoef(levels, values)[0, 1])
            low_level = float(levels[0]) if len(levels) else math.nan
            high_level = float(levels[-1]) if len(levels) else math.nan
            low_mean = float(values[0]) if len(values) else math.nan
            high_mean = float(values[-1]) if len(values) else math.nan
            relative = _degradation_percent(low_mean, high_mean)
            log_ratio = (
                float(np.log(high_mean / low_mean))
                if np.isfinite(low_mean) and np.isfinite(high_mean) and low_mean > 0 and high_mean > 0
                else math.nan
            )
            low_raw = pd.to_numeric(group.loc[group["level"] == low_level, metric], errors="coerce").to_numpy(float)
            high_raw = pd.to_numeric(group.loc[group["level"] == high_level, metric], errors="coerce").to_numpy(float)
            trend_rows.append({
                "factor": factor, "family": family, "system": system, "model": model,
                "metric": metric, "n_levels": len(levels), "n_observations": len(data),
                "spearman_rho": rho, "spearman_pvalue": pvalue,
                "linear_slope_per_unit": slope, "standardized_slope": standardized,
                "low_level": low_level, "high_level": high_level,
                "low_mean": low_mean, "high_mean": high_mean,
                "extreme_relative_change_pct": relative, "extreme_log_ratio": log_ratio,
                "cohen_d_high_vs_low": _cohen_d(low_raw, high_raw),
                "status": "ok" if len(levels) >= 2 else rho_status,
            })
    return pd.DataFrame(level_rows, columns=level_columns), pd.DataFrame(trend_rows, columns=trend_columns)


def _interaction_analysis(
    runs: pd.DataFrame, warnings: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_columns = [
        "system", "model", "interaction", "n", "seed_count", "nrmse_mean", "nrmse_std",
        "q5_q1_ratio_mean", "gap_relative_mean",
    ]
    advantage_columns = [
        "system", "comparator", "n_pairs", "n_levels", "low_interaction", "high_interaction",
        "advantage_low_pct", "advantage_high_pct", "advantage_slope_pct_per_unit",
        "advantage_spearman_rho", "advantage_spearman_pvalue", "status",
    ]
    subset = runs[runs["family"].astype(str).str.lower() == "mechanism_interaction"].copy()
    if subset.empty:
        _add_warning(warnings, "未找到 mechanism_interaction 实验。")
        return pd.DataFrame(columns=level_columns), pd.DataFrame(columns=advantage_columns)
    subset["interaction"] = [_factor_level(row, "interaction") for _, row in subset.iterrows()]
    subset = subset[np.isfinite(subset["interaction"])].copy()
    subset["_model"] = subset["model"].astype(str).str.lower()
    level_rows: list[dict[str, Any]] = []
    for (system, model, level), group in subset.groupby(["system", "model", "interaction"], sort=True):
        nrmse_mean, nrmse_std, n = _mean_std(group["nrmse"])
        ratio, _, _ = _mean_std(group["q5_q1_ratio"])
        gap, _, _ = _mean_std(group["gap_relative"])
        level_rows.append({
            "system": system, "model": model, "interaction": level, "n": n,
            "seed_count": int(pd.to_numeric(group["seed"], errors="coerce").nunique()),
            "nrmse_mean": nrmse_mean, "nrmse_std": nrmse_std,
            "q5_q1_ratio_mean": ratio, "gap_relative_mean": gap,
        })
    advantage_rows: list[dict[str, Any]] = []
    systems = sorted(subset["system"].dropna().astype(str).unique()) or [""]
    for system in systems:
        system_rows = subset[subset["system"].astype(str) == system]
        swiglu = system_rows[system_rows["_model"] == "swiglu"]
        for comparator in ("gelu", "silu"):
            baseline = system_rows[system_rows["_model"] == comparator]
            left = baseline.groupby(["seed", "interaction"], dropna=False)["nrmse"].mean().rename("baseline")
            right = swiglu.groupby(["seed", "interaction"], dropna=False)["nrmse"].mean().rename("swiglu")
            paired = pd.concat([left, right], axis=1).dropna().reset_index()
            if paired.empty:
                advantage_rows.append({"system": system, "comparator": comparator,
                                       "n_pairs": 0, "n_levels": 0, "status": "missing_matched_pairs"})
                _add_warning(warnings, f"interaction 中缺少 SwiGLU 与 {comparator} 的同 seed/level 配对。")
                continue
            paired["advantage"] = [
                _relative_percent(float(base), float(swiglu_value))
                for base, swiglu_value in zip(paired["baseline"], paired["swiglu"])
            ]
            by_level = paired.groupby("interaction", sort=True)["advantage"].mean().dropna()
            x, y = by_level.index.to_numpy(float), by_level.to_numpy(float)
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 and np.ptp(x) > 0 else math.nan
            rho, pvalue, _, status = _safe_spearman(x, y)
            advantage_rows.append({
                "system": system, "comparator": comparator, "n_pairs": len(paired),
                "n_levels": len(x), "low_interaction": float(x[0]) if len(x) else math.nan,
                "high_interaction": float(x[-1]) if len(x) else math.nan,
                "advantage_low_pct": float(y[0]) if len(y) else math.nan,
                "advantage_high_pct": float(y[-1]) if len(y) else math.nan,
                "advantage_slope_pct_per_unit": slope, "advantage_spearman_rho": rho,
                "advantage_spearman_pvalue": pvalue,
                "status": "ok" if len(x) >= 2 else status,
            })
    return pd.DataFrame(level_rows, columns=level_columns), pd.DataFrame(advantage_rows, columns=advantage_columns)


def _allocation_modes(runs: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    columns = [
        "system", "model", "mode", "n", "seed_count", "nrmse_mean", "nrmse_std",
        "q5_q1_ratio_mean", "gap_relative_mean", "allocation_spearman_mean",
        "aligned_advantage_nrmse_pct", "status",
    ]
    subset = runs[runs["family"].astype(str).str.lower() == "allocation"].copy()
    if subset.empty:
        _add_warning(warnings, "未找到 allocation 实验；四种分配模式均不可用。")
        return pd.DataFrame([{"mode": mode, "status": "missing_family"} for mode in ALLOCATION_MODES],
                            columns=columns)
    subset["_mode"] = subset["variant"].astype(str).str.lower()
    rows: list[dict[str, Any]] = []
    for (system, model), group in subset.groupby(["system", "model"], dropna=False, sort=True):
        aligned_mean, _, _ = _mean_std(group[group["_mode"] == "aligned"]["nrmse"])
        for mode in ALLOCATION_MODES:
            mode_group = group[group["_mode"] == mode]
            if mode_group.empty:
                rows.append({"system": system, "model": model, "mode": mode, "n": 0,
                             "seed_count": 0, "status": "missing_mode"})
                _add_warning(warnings, f"allocation 缺少 {system}/{model}/{mode} 模式。")
                continue
            nrmse, nrmse_std, n = _mean_std(mode_group["nrmse"])
            ratio, _, _ = _mean_std(mode_group["q5_q1_ratio"])
            gap, _, _ = _mean_std(mode_group["gap_relative"])
            allocation, _, _ = _mean_std(mode_group["allocation_spearman"])
            rows.append({
                "system": system, "model": model, "mode": mode, "n": n,
                "seed_count": int(pd.to_numeric(mode_group["seed"], errors="coerce").nunique()),
                "nrmse_mean": nrmse, "nrmse_std": nrmse_std,
                "q5_q1_ratio_mean": ratio, "gap_relative_mean": gap,
                "allocation_spearman_mean": allocation,
                "aligned_advantage_nrmse_pct": _relative_percent(nrmse, aligned_mean),
                "status": "ok",
            })
    return pd.DataFrame(rows, columns=columns)


def _direction_associations(
    runs: pd.DataFrame, diagnostics_dir: Path, warnings: list[str]
) -> pd.DataFrame:
    columns = [
        "association_level", "scope", "family", "system", "model", "job_id", "outcome",
        "rho", "p_value", "n", "rho_std", "n_runs", "status",
    ]
    rows: list[dict[str, Any]] = []
    main = runs[runs["family"].astype(str).str.lower() == "main"]
    scopes: list[tuple[str, pd.DataFrame, Any]] = [("main_all", main, "")]
    scopes.extend(("main_system", group, system) for system, group in main.groupby("system", sort=True))
    for scope, group, system in scopes:
        for outcome in ("nrmse", "gap_relative"):
            rho, pvalue, n, status = _safe_spearman(group["direction_alignment"], group[outcome])
            rows.append({
                "association_level": "run_level", "scope": scope, "family": "main",
                "system": system, "model": "", "job_id": "", "outcome": outcome,
                "rho": rho, "p_value": pvalue, "n": n, "n_runs": n, "status": status,
            })
    diagnostic_rows: list[dict[str, Any]] = []
    if not diagnostics_dir.exists():
        _add_warning(warnings, f"诊断目录不存在：{diagnostics_dir}")
    else:
        metadata = runs.drop_duplicates("job_id").set_index("job_id") if "job_id" in runs else pd.DataFrame()
        for path in sorted(diagnostics_dir.glob("*.npz")):
            job_id = path.stem
            meta = metadata.loc[job_id] if not metadata.empty and job_id in metadata.index else None
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if "direction_alignment" not in payload.files:
                        _add_warning(warnings, f"{path.name} 缺少 direction_alignment。")
                        continue
                    alignment = np.asarray(payload["direction_alignment"]).ravel()
                    for key, outcome in (("probe_nrmse", "probe_nrmse"), ("probe_gap", "probe_gap")):
                        if key not in payload.files:
                            _add_warning(warnings, f"{path.name} 缺少 {key}。")
                            continue
                        rho, pvalue, n, status = _safe_spearman(alignment, np.asarray(payload[key]).ravel())
                        diagnostic_rows.append({
                            "association_level": "diagnostic_run", "scope": "within_job",
                            "family": meta.get("family", "") if meta is not None else "",
                            "system": meta.get("system", "") if meta is not None else "",
                            "model": meta.get("model", "") if meta is not None else "",
                            "job_id": job_id, "outcome": outcome, "rho": rho,
                            "p_value": pvalue, "n": n, "n_runs": 1, "status": status,
                        })
            except (OSError, ValueError, KeyError) as exc:
                _add_warning(warnings, f"无法读取诊断文件 {path.name}: {exc}")
    rows.extend(diagnostic_rows)
    if diagnostic_rows:
        diagnostic_frame = pd.DataFrame(diagnostic_rows)
        for (family, system, model, outcome), group in diagnostic_frame.groupby(
            ["family", "system", "model", "outcome"], dropna=False, sort=True
        ):
            mean, std, n_runs = _mean_std(group["rho"])
            rows.append({
                "association_level": "diagnostic_group", "scope": "mean_within_job",
                "family": family, "system": system, "model": model, "job_id": "",
                "outcome": outcome, "rho": mean, "rho_std": std,
                "n": int(pd.to_numeric(group["n"], errors="coerce").sum()),
                "n_runs": n_runs, "status": "ok" if n_runs else "no_finite_correlations",
            })
    return pd.DataFrame(rows, columns=columns)


def _ablation_degradation(runs: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    columns = [
        "system", "variant", "metric", "seed", "full_value", "ablation_value",
        "absolute_change", "degradation_pct", "evidence_tier", "status",
    ]
    seeds = pd.to_numeric(runs["seed"], errors="coerce")
    main = runs[(runs["family"].astype(str).str.lower() == "main") &
                (runs["model"].astype(str).str.lower() == "ganr")].copy()
    ablation = runs[(runs["family"].astype(str).str.lower() == "ablation") &
                    (runs["model"].astype(str).str.lower() == "ganr")].copy()
    all_ganr = runs[(runs["family"].astype(str).str.lower().isin(("main", "ablation"))) &
                    (runs["model"].astype(str).str.lower() == "ganr")]
    systems = sorted(all_ganr["system"].dropna().astype(str).unique())
    if not systems:
        _add_warning(warnings, "未找到 GANR 主实验/消融，单 seed 消融不可用。")
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for system in systems:
        full_group = main[main["system"].astype(str) == system]
        if "variant" in full_group and (full_group["variant"].astype(str).str.lower() == "default").any():
            full_group = full_group[full_group["variant"].astype(str).str.lower() == "default"]
        for variant in ABLATION_VARIANTS:
            ablation_group = ablation[(ablation["system"].astype(str) == system) &
                                      (ablation["variant"].astype(str).str.lower() == variant)]
            full_seeds = set(pd.to_numeric(full_group["seed"], errors="coerce").dropna())
            ablation_seeds = set(pd.to_numeric(ablation_group["seed"], errors="coerce").dropna())
            common_seeds = sorted(full_seeds & ablation_seeds)
            selected_seed = common_seeds[0] if common_seeds else math.nan
            current_full = full_group[
                pd.to_numeric(full_group["seed"], errors="coerce") == selected_seed
            ]
            current_ablation = ablation_group[
                pd.to_numeric(ablation_group["seed"], errors="coerce") == selected_seed
            ]
            for metric in PAIRED_METRICS:
                full_value, _, full_n = _mean_std(current_full[metric])
                ablation_value, _, ablation_n = _mean_std(current_ablation[metric])
                if not full_n:
                    status = "missing_seed_matched_full"
                elif not ablation_n:
                    status = "missing_seed_matched_ablation"
                else:
                    status = "ok"
                rows.append({
                    "system": system, "variant": variant, "metric": metric, "seed": selected_seed,
                    "full_value": full_value, "ablation_value": ablation_value,
                    "absolute_change": ablation_value - full_value
                    if np.isfinite(full_value) and np.isfinite(ablation_value) else math.nan,
                    "degradation_pct": _degradation_percent(full_value, ablation_value),
                    "evidence_tier": "单 seed 探索性", "status": status,
                })
    if any(row["status"] != "ok" for row in rows):
        _add_warning(warnings, "Full 与部分消融组缺少共同 seed；对应退化量标记为不可用。")
    return pd.DataFrame(rows, columns=columns)


def _format_number(value: Any, digits: int = 3, suffix: str = "") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}g}{suffix}" if np.isfinite(numeric) else "NA"


def _report_text(
    runs: pd.DataFrame,
    main: pd.DataFrame,
    paired: pd.DataFrame,
    geometry: pd.DataFrame,
    scaling: pd.DataFrame,
    mechanism_trends: pd.DataFrame,
    interaction: pd.DataFrame,
    allocation: pd.DataFrame,
    direction: pd.DataFrame,
    ablation: pd.DataFrame,
    failure_boundary: pd.DataFrame,
    warnings: list[str],
) -> str:
    lines = [
        "# 非线性几何实验分析报告",
        "",
        "## 证据口径",
        "",
        f"本报告读取 {len(runs)} 条运行记录。`main` 实验只有在同一 system/model 具备至少 3 个独立 seed 时才进入“三 seed 主结论”；尺度、机制、分配和单 seed 消融均明确视为探索性证据。Bootstrap 区间基于同 seed 配对差值，样本仅 3 时区间只能描述这 3 次运行的不确定性，不能替代更大样本推断。",
        "",
        "所有趋势均表述为关联或受控设置下的响应，不据此宣称普适因果关系。",
        "",
        "## 1. 三 seed 主实验",
        "",
    ]
    if main.empty:
        lines.append("未发现 `family=main` 的记录，主结论不可用。")
    else:
        for _, row in main.sort_values(["system", "nrmse_mean"], na_position="last").iterrows():
            completeness = "3-seed" if bool(row["three_seed_complete"]) else f"仅 {int(row['seed_count'])} seed"
            lines.append(
                f"- {row['system']} / {row['model']}（{completeness}）：NRMSE "
                f"{row['nrmse_mean_std']}，Q5/Q1 {row['q5_q1_ratio_mean_std']}，"
                f"relative gap {row['gap_relative_mean_std']}，residual {row['residual_mean_mean_std']}。"
            )
    lines.extend(["", "### GANR 与自动选择的最强非 GANR 基线（同 seed 配对）", ""])
    successful = paired[paired["status"] == "ok"] if not paired.empty else paired
    if successful.empty:
        lines.append("缺少可配对的 GANR/基线记录。")
    else:
        for system in successful["system"].drop_duplicates():
            block = successful[successful["system"] == system]
            baseline = block["baseline_model"].iloc[0]
            lines.append(f"- **{system}**：按主实验平均 NRMSE 自动选择 `{baseline}`。")
            for _, row in block.iterrows():
                lines.append(
                    f"  - {row['metric']}：GANR 相对改善 {_format_number(row['mean_relative_improvement_pct'], suffix='%')}，"
                    f"确定性 paired-bootstrap 95% CI "
                    f"[{_format_number(row['ci95_low_pct'], suffix='%')}, {_format_number(row['ci95_high_pct'], suffix='%')}]，"
                    f"胜出 {row['win_count']}（配对 seed={row['paired_seeds']}）。"
                )
    lines.extend(["", "## 2. 非线性—误差—表征缺口", ""])
    if geometry.empty:
        lines.append("几何汇总不可用。")
    else:
        for _, row in geometry.sort_values(["system", "model"]).iterrows():
            lines.append(
                f"- {row['system']} / {row['model']}：error~κ Spearman "
                f"{row['error_kappa_spearman_mean_std']}，gap~error Spearman "
                f"{row['gap_error_spearman_mean_std']}，Q5/Q1 {row['q5_q1_ratio_mean_std']}。"
            )
        lines.append("这些量共同描述高非线性区域的误差集中与 representation gap 关联；相关系数本身不建立因果方向。")
    lines.extend(["", "## 3. Width / depth / sample 极端设置（探索性）", ""])
    for _, row in scaling.iterrows():
        if row.get("status") != "ok":
            lines.append(f"- {row['factor']}：不可用（{row.get('status')}）。")
            continue
        persistence = "仍存在" if row["high_nonlinearity_penalty_persists"] else "未观察到"
        lines.append(
            f"- {row['factor']} {row['low_level']:g}→{row['high_level']:g}：平均 NRMSE "
            f"{_format_number(row['nrmse_low'])}→{_format_number(row['nrmse_high'])} "
            f"（变化 {_format_number(row['nrmse_change_pct'], suffix='%')}）；高端设置 Q5/Q1="
            f"{_format_number(row['q5_q1_high'])}，高非线性惩罚{persistence}。证据级别：{row['evidence_tier']}。"
        )
    lines.extend(["", "## 4. 机制因子趋势与效应量（探索性）", ""])
    nrmse_trends = (
        mechanism_trends[
            (mechanism_trends["metric"] == "nrmse")
            & (mechanism_trends["status"] == "ok")
        ]
        if not mechanism_trends.empty
        else mechanism_trends
    )
    if nrmse_trends.empty:
        lines.append("机制因子趋势不可用。")
    else:
        for _, row in nrmse_trends.sort_values(["factor", "model"]).iterrows():
            lines.append(
                f"- {row['factor']} / {row['model']}：level {row['low_level']:g}→{row['high_level']:g}，"
                f"NRMSE 极端变化 {_format_number(row['extreme_relative_change_pct'], suffix='%')}，"
                f"Spearman ρ={_format_number(row['spearman_rho'])}，标准化线性斜率="
                f"{_format_number(row['standardized_slope'])}，log-ratio effect="
                f"{_format_number(row['extreme_log_ratio'])}。"
            )
    lines.extend(["", "### Multiplicative interaction", ""])
    valid_interaction = interaction[interaction["status"] == "ok"] if not interaction.empty else interaction
    if valid_interaction.empty:
        lines.append("SwiGLU 与 GELU/SiLU 的 interaction 配对不足。")
    else:
        for _, row in valid_interaction.iterrows():
            direction_word = "扩大" if row["advantage_slope_pct_per_unit"] > 0 else "缩小"
            lines.append(
                f"- SwiGLU 相对 {row['comparator']}：interaction 每增加 1，NRMSE 优势"
                f"{direction_word} {_format_number(abs(row['advantage_slope_pct_per_unit']), suffix=' 个百分点')}；"
                f"低/高端优势 {_format_number(row['advantage_low_pct'], suffix='%')} / "
                f"{_format_number(row['advantage_high_pct'], suffix='%')}。"
            )
    lines.extend(["", "## 5. 四种 capacity allocation（探索性）", ""])
    valid_allocation = allocation[allocation["status"] == "ok"] if not allocation.empty else allocation
    if valid_allocation.empty:
        lines.append("allocation 四模式数据不可用。")
    else:
        for (system, model), group in valid_allocation.groupby(["system", "model"], dropna=False):
            ordered = group.sort_values("nrmse_mean")
            ranking = " < ".join(
                f"{row['mode']}({_format_number(row['nrmse_mean'])})" for _, row in ordered.iterrows()
            )
            lines.append(f"- {system} / {model} 的 NRMSE 排序：{ranking}（小者更好）。")
    lines.extend(["", "## 6. 方向对齐关联", ""])
    main_direction = direction[(direction["association_level"] == "run_level") &
                               (direction["scope"] == "main_all")] if not direction.empty else direction
    if main_direction.empty:
        lines.append("direction alignment 关联不可用。")
    else:
        for _, row in main_direction.iterrows():
            lines.append(
                f"- 主实验 run-level：direction_alignment 与 {row['outcome']} 的 Spearman "
                f"ρ={_format_number(row['rho'])}（n={int(row['n'])}，status={row['status']}）。"
            )
        lines.append("另见 `tables/direction_alignment_associations.csv` 中每个诊断文件的点级关联；run-level 与点级结果不可混为同一估计量。")
    lines.extend(["", "## 7. GANR 配对单 seed 消融（探索性）", ""])
    valid_ablation = ablation[(ablation["status"] == "ok") & (ablation["metric"] == "nrmse")] if not ablation.empty else ablation
    if valid_ablation.empty:
        lines.append("Full 与消融缺少共同 seed，无法形成退化量。")
    else:
        for _, row in valid_ablation.sort_values(["system", "variant"]).iterrows():
            lines.append(
                f"- {row['system']} / {row['variant']}：相对同 seed Full 的 NRMSE 退化 "
                f"{_format_number(row['degradation_pct'], suffix='%')}（"
                f"{_format_number(row['full_value'])}→{_format_number(row['ablation_value'])}）。"
            )
    lines.extend(["", "## 8. 限制与数据完整性", ""])
    lines.append("尺度/机制/allocation/消融若只有单 seed，只能作为生成假设和机制一致性证据；即使在受控系统中改变单一配置，也仍可能受训练随机性与有限预算影响。")
    if warnings:
        lines.extend(["", "自动检查记录："])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("未发现输入结构缺失。")
    if not failure_boundary.empty:
        lines.extend(["", "## 9. GANR 适用域与失效边界", ""])
        valid_boundary = failure_boundary[failure_boundary["status"] == "ok"]
        if valid_boundary.empty:
            lines.append("缺少同水平 GANR/基线配对，边界不可用。")
        else:
            for (system, factor), group in valid_boundary.groupby(["system", "factor"], sort=True):
                wins = int(group["ganr_wins"].sum())
                lines.append(
                    f"- {system} / {factor}：GANR 在 {wins}/{len(group)} 个水平优于同水平最强基线；"
                    "该表用于标出适用域转折，不把单因素扫描解释为普适因果。"
                )
    lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if not isinstance(value, (str, bytes, Path)):
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, Path):
        return str(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in frame.to_dict(orient="records")]


def generate_analysis(
    runs_csv: str | Path,
    quintiles_csv: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    """Generate all analysis tables, a Chinese report, and a JSON summary.

    Parameters are paths so this function works for both ``smoke`` and ``full``
    profiles as well as synthetic test fixtures.  ``output_dir`` defaults to the
    directory containing ``runs.csv``.
    """

    runs_path = Path(runs_csv)
    if runs_path.is_dir():
        runs_path = runs_path / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_path}")
    profile_dir = runs_path.parent
    quintiles_path = Path(quintiles_csv) if quintiles_csv is not None else profile_dir / "quintiles.csv"
    diagnostic_path = Path(diagnostics_dir) if diagnostics_dir is not None else profile_dir / "diagnostics"
    destination = Path(output_dir) if output_dir is not None else profile_dir
    table_dir = destination / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    try:
        runs = pd.read_csv(runs_path)
    except pd.errors.EmptyDataError:
        runs = pd.DataFrame()
        _add_warning(warnings, "runs.csv 为空；所有分析均标记为不可用。")
    if quintiles_path.exists():
        try:
            quintiles = pd.read_csv(quintiles_path)
        except pd.errors.EmptyDataError:
            quintiles = pd.DataFrame()
            _add_warning(warnings, "quintiles.csv 为空。")
    else:
        quintiles = pd.DataFrame()
    if not quintiles_path.exists():
        _add_warning(warnings, f"quintiles.csv 不存在：{quintiles_path}")
    runs = _enrich_quintiles(runs, quintiles, warnings)
    _ensure_columns(
        runs,
        (
            "job_id", "family", "system", "model", "seed", "variant", "system_kwargs",
            "width", "depth", "train_size", "nrmse", "q1_nrmse", "q5_nrmse",
            "q5_q1_ratio", "gap_relative", "residual_mean", "error_kappa_spearman",
            "gap_error_spearman", "direction_alignment", "allocation_spearman",
            "curvature_tail_gap_ratio", "curvature_tail_error_ratio",
            "curvature_tail_gap_error_spearman",
            "absolute_response_gap", "log_magnitude_mismatch",
            "q80_absolute_response_gap",
        ),
        warnings,
    )
    numeric_columns = set(MAIN_METRICS) | {
        "seed", "width", "depth", "train_size", "q1_nrmse", "allocation_spearman",
    }
    for column in numeric_columns:
        runs[column] = pd.to_numeric(runs[column], errors="coerce")

    main = _main_aggregate(runs)
    modular = _modular_factorial(runs)
    paired, paired_details = _paired_main(
        runs, warnings, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
    )
    geometry = _geometry_summary(runs)
    scaling = _scaling_extremes(runs, warnings)
    mechanism_levels, mechanism_trends = _mechanism_analysis(runs, warnings)
    interaction_levels, interaction = _interaction_analysis(runs, warnings)
    allocation = _allocation_modes(runs, warnings)
    direction = _direction_associations(runs, diagnostic_path, warnings)
    ablation = _ablation_degradation(runs, warnings)
    failure_boundary = _failure_boundary_summary(runs, warnings)

    tables = {
        "main_aggregate.csv": main,
        "modular_factorial.csv": modular,
        "main_paired_improvements.csv": paired,
        "main_paired_seed_details.csv": paired_details,
        "geometry_summary.csv": geometry,
        "scaling_extremes.csv": scaling,
        "mechanism_levels.csv": mechanism_levels,
        "mechanism_trends.csv": mechanism_trends,
        "interaction_levels.csv": interaction_levels,
        "interaction_advantage.csv": interaction,
        "allocation_modes.csv": allocation,
        "direction_alignment_associations.csv": direction,
        "ablation_single_seed_degradation.csv": ablation,
        "failure_boundary.csv": failure_boundary,
    }
    for name, frame in tables.items():
        frame.to_csv(table_dir / name, index=False)

    report = _report_text(
        runs, main, paired, geometry, scaling, mechanism_trends, interaction,
        allocation, direction, ablation, failure_boundary, warnings,
    )
    if not modular.empty:
        report += "\n## 9. 可插拔模块 factorial benchmark\n\n"
        for system, group in modular.groupby("system", sort=True):
            report += f"- **{system}**："
            entries = []
            for _, row in group.sort_values(["model", "variant"]).iterrows():
                entries.append(
                    f"{row['model']}+{row['variant']} NRMSE={row['nrmse_mean']:.5g} "
                    f"(n={int(row['seed_count'])})"
                )
            report += "; ".join(entries) + "。\n"
    report_path = destination / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "runs_csv": str(runs_path.resolve()),
            "quintiles_csv": str(quintiles_path.resolve()),
            "diagnostics_dir": str(diagnostic_path.resolve()),
        },
        "outputs": {
            "report": str(report_path.resolve()),
            "tables": [str((table_dir / name).resolve()) for name in tables],
        },
        "data_overview": {
            "run_count": len(runs),
            "quintile_row_count": len(quintiles),
            "main_run_count": int((runs["family"].astype(str).str.lower() == "main").sum()),
            "systems": sorted(runs["system"].dropna().astype(str).unique().tolist()),
            "families": sorted(runs["family"].dropna().astype(str).unique().tolist()),
        },
        "evidence_policy": {
            "main": "同组至少3个独立seed为主结论",
            "other_families": "默认探索性；单seed不作稳健性或普适因果声明",
            "bootstrap": {
                "method": "paired percentile bootstrap of mean relative improvement",
                "samples": int(bootstrap_samples),
                "seed": int(bootstrap_seed),
            },
        },
        "main_aggregate": _records(main),
        "modular_factorial": _records(modular),
        "paired_improvements": _records(paired),
        "geometry_summary": _records(geometry),
        "scaling_extremes": _records(scaling),
        "mechanism_trends": _records(mechanism_trends),
        "interaction_advantage": _records(interaction),
        "allocation_modes": _records(allocation),
        "direction_alignment_associations": _records(direction),
        "ablation_single_seed_degradation": _records(ablation),
        "failure_boundary": _records(failure_boundary),
        "warnings": warnings,
    }
    summary = _json_safe(summary)
    summary_path = destination / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return summary


def analyze_profile(
    profile_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    """Convenience wrapper for a ``results/<profile>`` directory."""

    directory = Path(profile_dir)
    return generate_analysis(
        directory / "runs.csv",
        directory / "quintiles.csv",
        directory / "diagnostics",
        output_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="Path such as results/full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    summary = analyze_profile(
        args.profile_dir,
        output_dir=args.output_dir,
        bootstrap_samples=max(1, args.bootstrap_samples),
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
