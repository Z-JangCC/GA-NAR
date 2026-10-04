"""Publication-ready figures for the nonlinear-geometry study.

The module is intentionally usable in headless batch jobs.  It consumes the
two tidy CSV tables written by the experiment runner and emits paired 300-dpi
PNG/PDF figures.  Missing columns or empty experiment families skip only the
affected figure instead of aborting a long result-generation run.

Uncertainty is always computed across independent seeds.  Rows are first
averaged within a seed, so systems, duplicate records, or quintile samples are
never presented as independent uncertainty replicates.  A single seed produces
no error bar or uncertainty band.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "ink": "#272727",
    "mid_gray": "#767676",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]

_SUPPORTED_FORMATS = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
_MODEL_PRIORITY = ("relu", "gelu", "silu", "swiglu", "resmlp", "fourier", "ganr")
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
_HATCHES = ("//", "\\\\", "..", "xx", "--", "++", "oo", "**")


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = (
        "DejaVu Sans",
        "Helvetica",
        "Arial",
        "sans-serif",
    )


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Apply the project-wide publication rcParams and return the style used."""

    style = style or FigureStyle()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.titleweight": "semibold",
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": max(8, style.font_size - 2),
            "ytick.labelsize": max(8, style.font_size - 2),
            "xtick.major.width": max(1.0, style.axes_linewidth * 0.65),
            "ytick.major.width": max(1.0, style.axes_linewidth * 0.65),
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.frameon": False,
            "legend.fontsize": max(8, style.font_size - 2),
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
            "patch.edgecolor": "black",
            "patch.linewidth": 1.1,
            "hatch.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "text.usetex": style.use_tex,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return style


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Create subplots and always return axes as a flattened array."""

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    return fig, np.asarray(axes, dtype=object).reshape(-1)


def finalize_figure(
    fig: Figure,
    out_path: str | Path,
    formats: Sequence[str] | str | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
    *,
    layout_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    **savefig_kwargs,
) -> list[Path]:
    """Finalize layout and save a figure, by default as PNG and PDF."""

    destination = Path(out_path)
    if destination.suffix.lower().lstrip(".") in _SUPPORTED_FORMATS:
        destination = destination.with_suffix("")
    if formats is None:
        requested = ("png", "pdf")
    elif isinstance(formats, str):
        requested = (formats,)
    else:
        requested = tuple(formats)
    normalized = tuple(str(fmt).lower().lstrip(".") for fmt in requested)
    unsupported = sorted(set(normalized) - _SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"unsupported figure format(s): {', '.join(unsupported)}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(pad=1.2, rect=layout_rect)
    except ValueError:
        # Some externally supplied axes layouts are not tight-layout compatible.
        pass
    saved: list[Path] = []
    for fmt in normalized:
        path = destination.with_suffix(f".{fmt}")
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad,
            facecolor="white",
            **savefig_kwargs,
        )
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def _warn_skip(figure_name: str, reason: str) -> None:
    warnings.warn(f"Skipping {figure_name}: {reason}", RuntimeWarning, stacklevel=3)


def _required(frame: pd.DataFrame, columns: Iterable[str], figure_name: str) -> bool:
    if frame.empty:
        _warn_skip(figure_name, "input table or selected subset is empty")
        return False
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        _warn_skip(figure_name, f"missing column(s): {', '.join(missing)}")
        return False
    return True


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        _warn_skip(label, f"file not found: {source}")
        return pd.DataFrame()
    try:
        return pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _warn_skip(label, f"could not read {source}: {exc}")
        return pd.DataFrame()


def _family_text(frame: pd.DataFrame) -> pd.Series:
    return frame["family"].fillna("").astype(str).str.lower()


def _main_subset(frame: pd.DataFrame) -> pd.DataFrame:
    if "family" not in frame:
        return frame.iloc[0:0]
    frame = frame.loc[frame["system"].astype(str).str.lower() != "ac_power"].copy() if "system" in frame else frame
    family = _family_text(frame)
    preferred = family.str.contains(r"main|overall|performance|baseline", regex=True)
    if preferred.any():
        return frame.loc[preferred].copy()
    excluded = family.str.contains(r"scal|ablat|mechan", regex=True)
    fallback = frame.loc[~excluded].copy()
    return fallback if not fallback.empty else frame.copy()


def _mechanism_subset(frame: pd.DataFrame) -> pd.DataFrame:
    main = _main_subset(frame)
    metric_columns = {
        "underrepresentation",
        "magnitude_log_ratio",
        "allocation_spearman",
        "allocation_overlap",
        "direction_alignment",
    }
    if not main.empty and any(
        column in main and pd.to_numeric(main[column], errors="coerce").notna().any()
        for column in metric_columns
    ):
        return main
    family = _family_text(frame)
    return frame.loc[family.str.contains("mechan", regex=False)].copy()


def _ordered_models(values: Iterable[object]) -> list[str]:
    models = {str(value) for value in values if pd.notna(value)}
    normalized = {model.lower(): model for model in models}
    result = [normalized[name] for name in _MODEL_PRIORITY if name in normalized]
    result.extend(sorted(models - set(result), key=str.lower))
    return result


def _model_display(model: object) -> str:
    key = str(model).lower()
    names = {
        "relu": "ReLU",
        "gelu": "GELU",
        "silu": "SiLU",
        "swiglu": "SwiGLU",
        "resmlp": "ResMLP",
        "fourier": "Fourier",
        "fourier_resmlp": "Fourier",
        "ganr": "GANR",
        "ours": "GANR",
    }
    return names.get(key, str(model).replace("_", " "))


def _system_display(system: object) -> str:
    key = str(system).lower()
    names = {
        "controlled": "Controlled",
        "controlled_implicit": "Controlled",
        "ac_power": "AC power flow",
        "ac_power_flow": "AC power flow",
        "powerflow": "AC power flow",
        "case9": "AC power flow",
        "duffing": "Coupled Duffing",
        "ieee118": "IEEE-118 AC",
        "allen_cahn": "Allen–Cahn",
        "shallow_water": "Shallow water",
    }
    return names.get(key, str(system).replace("_", " ").title())


def _model_color(model: object, index: int = 0) -> str:
    key = str(model).lower()
    colors = {
        "relu": PALETTE["red_strong"],
        "gelu": PALETTE["red_2"],
        "silu": PALETTE["green_3"],
        "swiglu": PALETTE["teal"],
        "resmlp": PALETTE["violet"],
        "fourier": PALETTE["green_2"],
        "fourier_resmlp": PALETTE["green_2"],
        "ganr": PALETTE["blue_main"],
        "ours": PALETTE["blue_main"],
    }
    return colors.get(key, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def _seed_summary(
    frame: pd.DataFrame,
    groups: Sequence[str],
    value: str,
) -> pd.DataFrame:
    """Aggregate duplicates within seed, then summarize independent seeds."""

    needed = [*groups, "seed", value]
    if any(column not in frame for column in needed):
        return pd.DataFrame(columns=[*groups, "mean", "sd", "n_seed"])
    working = frame[needed].copy()
    working[value] = pd.to_numeric(working[value], errors="coerce")
    working = working.dropna(subset=[*groups, "seed", value])
    if working.empty:
        return pd.DataFrame(columns=[*groups, "mean", "sd", "n_seed"])
    per_seed = (
        working.groupby([*groups, "seed"], sort=False, observed=True, dropna=False)[value]
        .mean()
        .reset_index()
    )
    summary = (
        per_seed.groupby(list(groups), sort=False, observed=True, dropna=False)[value]
        .agg(mean="mean", sd="std", n_seed="count")
        .reset_index()
    )
    summary.loc[summary["n_seed"] < 2, "sd"] = np.nan
    return summary


def _panel_label(ax: Axes, index: int) -> None:
    ax.text(
        -0.13,
        1.06,
        f"({chr(ord('a') + index)})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def _polish_axis(ax: Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color=PALETTE["neutral"], linewidth=0.7, alpha=0.42)
        ax.set_axisbelow(True)
    ax.margins(y=0.12)


def _figure_legend(fig: Figure, axes: Sequence[Axes], ncol: int | None = None) -> bool:
    handles: list[object] = []
    labels: list[str] = []
    for ax in axes:
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label and not label.startswith("_") and label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return False
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol or min(len(labels), 7),
        columnspacing=1.3,
        handletextpad=0.5,
    )
    return True


def _plot_seed_trends(
    ax: Axes,
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
) -> bool:
    summary = _seed_summary(frame, ["model", x_column], value_column)
    if summary.empty:
        return False
    plotted = False
    for model_index, model in enumerate(_ordered_models(summary["model"])):
        current = summary.loc[summary["model"].astype(str) == str(model)].copy()
        current[x_column] = pd.to_numeric(current[x_column], errors="coerce")
        current = current.dropna(subset=[x_column, "mean"]).sort_values(x_column)
        if current.empty:
            continue
        x = current[x_column].to_numpy(dtype=float)
        mean = current["mean"].to_numpy(dtype=float)
        color = _model_color(model, model_index)
        linewidth = 3.0 if str(model).lower() in {"ganr", "ours"} else 2.2
        ax.plot(
            x,
            mean,
            marker=_MARKERS[model_index % len(_MARKERS)],
            color=color,
            linewidth=linewidth,
            label=_model_display(model),
            zorder=3,
        )
        sd = current["sd"].to_numpy(dtype=float)
        valid_uncertainty = np.isfinite(sd) & (current["n_seed"].to_numpy() >= 2)
        if np.any(valid_uncertainty):
            lower = np.where(valid_uncertainty, mean - sd, np.nan)
            upper = np.where(valid_uncertainty, mean + sd, np.nan)
            ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
        plotted = True
    return plotted


def _grouped_seed_bars(
    ax: Axes,
    frame: pd.DataFrame,
    category: str,
    value: str,
) -> bool:
    summary = _seed_summary(frame, [category, "model"], value)
    if summary.empty:
        return False
    categories = [str(value) for value in pd.unique(summary[category])]
    models = _ordered_models(summary["model"])
    if not categories or not models:
        return False
    x = np.arange(len(categories), dtype=float)
    width = min(0.8 / len(models), 0.18)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for model_index, model in enumerate(models):
        current = summary.loc[summary["model"].astype(str) == str(model)].set_index(
            category
        )
        means = np.array(
            [float(current.loc[item, "mean"]) if item in current.index else np.nan for item in categories]
        )
        positions = x + offsets[model_index]
        bars = ax.bar(
            positions,
            means,
            width=width * 0.92,
            color=_model_color(model, model_index),
            edgecolor="black",
            linewidth=1.0,
            hatch=_HATCHES[model_index % len(_HATCHES)],
            label=_model_display(model),
            zorder=2,
        )
        for patch in bars.patches:
            patch.set_gid(f"model-{model}")
        uncertainty_positions: list[float] = []
        uncertainty_means: list[float] = []
        uncertainty_sd: list[float] = []
        for position, item, mean in zip(positions, categories, means):
            if item not in current.index or not np.isfinite(mean):
                continue
            row = current.loc[item]
            if int(row["n_seed"]) >= 2 and np.isfinite(float(row["sd"])):
                uncertainty_positions.append(float(position))
                uncertainty_means.append(float(mean))
                uncertainty_sd.append(float(row["sd"]))
        if uncertainty_positions:
            ax.errorbar(
                uncertainty_positions,
                uncertainty_means,
                yerr=uncertainty_sd,
                fmt="none",
                ecolor=PALETTE["ink"],
                elinewidth=1.2,
                capsize=2.5,
                capthick=1.2,
                zorder=4,
            )
    ax.set_xticks(x, [_system_display(item) for item in categories])
    return True


def _model_seed_bars(ax: Axes, frame: pd.DataFrame, value: str) -> bool:
    summary = _seed_summary(frame, ["model"], value)
    if summary.empty:
        return False
    models = _ordered_models(summary["model"])
    current = summary.set_index("model")
    x = np.arange(len(models), dtype=float)
    means = np.array([float(current.loc[model, "mean"]) for model in models])
    colors = [_model_color(model, index) for index, model in enumerate(models)]
    bars = ax.bar(
        x,
        means,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=1.1,
        zorder=2,
    )
    for index, bar in enumerate(bars):
        bar.set_hatch(_HATCHES[index % len(_HATCHES)])
    positions, centers, errors = [], [], []
    for index, model in enumerate(models):
        row = current.loc[model]
        if int(row["n_seed"]) >= 2 and np.isfinite(float(row["sd"])):
            positions.append(float(x[index]))
            centers.append(float(row["mean"]))
            errors.append(float(row["sd"]))
    if positions:
        ax.errorbar(
            positions,
            centers,
            yerr=errors,
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=1.2,
            capsize=2.5,
            zorder=4,
        )
    ax.set_xticks(x, [_model_display(model) for model in models], rotation=32, ha="right")
    return True


def plot_error_vs_nonlinearity(
    quintiles: pd.DataFrame,
    output_dir: str | Path,
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Plot NRMSE against true system nonlinearity for each system/model."""

    name = "error-vs-nonlinearity"
    needed = {"system", "model", "seed", "quintile", "nrmse", "kappa_mean"}
    if not _required(quintiles, needed, name):
        return []
    working = quintiles[list(needed)].copy()
    working["nrmse"] = pd.to_numeric(working["nrmse"], errors="coerce")
    working["kappa_mean"] = pd.to_numeric(working["kappa_mean"], errors="coerce")
    working["quintile_order"] = pd.to_numeric(
        working["quintile"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    working = working.dropna(
        subset=["system", "model", "seed", "quintile_order", "nrmse", "kappa_mean"]
    )
    if working.empty:
        _warn_skip(name, "no finite quintile records")
        return []
    per_seed = (
        working.groupby(
            ["system", "model", "quintile_order", "seed"],
            sort=False,
            observed=True,
        )[["nrmse", "kappa_mean"]]
        .mean()
        .reset_index()
    )
    summary = (
        per_seed.groupby(
            ["system", "model", "quintile_order"], sort=False, observed=True
        )
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_sd=("nrmse", "std"),
            kappa_mean=("kappa_mean", "mean"),
            n_seed=("seed", "count"),
        )
        .reset_index()
    )
    summary.loc[summary["n_seed"] < 2, "nrmse_sd"] = np.nan
    systems = [str(value) for value in pd.unique(summary["system"])]
    apply_publication_style(style)
    fig, axes = create_subplots(1, len(systems), figsize=(5.3 * len(systems), 4.7), squeeze=False)
    for panel, (ax, system) in enumerate(zip(axes, systems)):
        selected = summary.loc[summary["system"].astype(str) == system]
        for model_index, model in enumerate(_ordered_models(selected["model"])):
            current = selected.loc[selected["model"].astype(str) == str(model)].sort_values(
                "quintile_order"
            )
            x = current["kappa_mean"].to_numpy(dtype=float)
            y = current["nrmse_mean"].to_numpy(dtype=float)
            color = _model_color(model, model_index)
            ax.plot(
                x,
                y,
                color=color,
                marker=_MARKERS[model_index % len(_MARKERS)],
                linewidth=3.0 if str(model).lower() in {"ganr", "ours"} else 2.2,
                label=_model_display(model),
            )
            sd = current["nrmse_sd"].to_numpy(dtype=float)
            valid = np.isfinite(sd) & (current["n_seed"].to_numpy() >= 2)
            if np.any(valid):
                ax.fill_between(
                    x,
                    np.where(valid, y - sd, np.nan),
                    np.where(valid, y + sd, np.nan),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        ax.set_title(_system_display(system))
        ax.set_xlabel(r"System nonlinearity $\kappa_S$")
        if panel == 0:
            ax.set_ylabel("Normalized RMSE")
        _panel_label(ax, panel)
        _polish_axis(ax)
    has_legend = _figure_legend(fig, axes)
    return finalize_figure(
        fig,
        Path(output_dir) / "error_vs_nonlinearity",
        layout_rect=(0.0, 0.0, 1.0, 0.90 if has_legend else 1.0),
    )


def plot_scaling(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Plot width, depth, and training-sample scaling in three panels."""

    name = "scaling"
    if not _required(runs, {"family", "model", "seed", "nrmse"}, name):
        return []
    family = _family_text(runs)
    panels = (
        ("width", "Width", family.str.contains("width", regex=False)),
        ("depth", "Depth", family.str.contains("depth", regex=False)),
        (
            "train_size",
            "Training samples",
            family.str.contains(r"sample|data|train", regex=True),
        ),
    )
    apply_publication_style(style)
    fig, axes = create_subplots(1, 3, figsize=(15.8, 4.6), squeeze=False)
    any_panel = False
    for index, (ax, (x_column, xlabel, mask)) in enumerate(zip(axes, panels)):
        if x_column not in runs:
            ax.set_axis_off()
            continue
        selected = runs.loc[mask].copy()
        if selected.empty or not _plot_seed_trends(ax, selected, x_column, "nrmse"):
            ax.set_axis_off()
            continue
        any_panel = True
        ax.set_xlabel(xlabel)
        if index == 0:
            ax.set_ylabel("Normalized RMSE")
        if x_column == "width":
            positive = pd.to_numeric(selected[x_column], errors="coerce").dropna()
            if len(positive) and (positive > 0).all():
                ax.set_xscale("log", base=2)
        elif x_column == "train_size":
            positive = pd.to_numeric(selected[x_column], errors="coerce").dropna()
            if len(positive) and (positive > 0).all():
                ax.set_xscale("log")
        ax.set_title(f"{xlabel} scaling")
        _panel_label(ax, index)
        _polish_axis(ax)
    if not any_panel:
        plt.close(fig)
        _warn_skip(name, "no non-empty width/depth/data-scaling family")
        return []
    has_legend = _figure_legend(fig, [ax for ax in axes if ax.axison])
    return finalize_figure(
        fig,
        Path(output_dir) / "scaling",
        layout_rect=(0.0, 0.0, 1.0, 0.90 if has_legend else 1.0),
    )


def plot_performance_comparison(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Compare overall error, Q5 error, and representation gap."""

    name = "gap-overall-Q5 comparison"
    if not _required(runs, {"family", "system", "model", "seed"}, name):
        return []
    selected = _main_subset(runs)
    metrics = (
        ("nrmse", "Overall NRMSE", "Overall prediction"),
        ("q5_nrmse", "Q5 NRMSE", "High nonlinearity"),
        ("gap_relative", "Relative gap", "Representation gap"),
    )
    apply_publication_style(style)
    fig, axes = create_subplots(1, 3, figsize=(17.0, 4.8), squeeze=False)
    any_panel = False
    for index, (ax, (metric, ylabel, title)) in enumerate(zip(axes, metrics)):
        if metric not in selected or not _grouped_seed_bars(ax, selected, "system", metric):
            ax.set_axis_off()
            continue
        any_panel = True
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _panel_label(ax, index)
        _polish_axis(ax)
    if not any_panel:
        plt.close(fig)
        _warn_skip(name, "no finite main-performance metrics")
        return []
    has_legend = _figure_legend(fig, [ax for ax in axes if ax.axison])
    return finalize_figure(
        fig,
        Path(output_dir) / "gap_overall_q5_comparison",
        layout_rect=(0.0, 0.0, 1.0, 0.88 if has_legend else 1.0),
    )


def _allocation_panel(ax: Axes, frame: pd.DataFrame) -> bool:
    metrics = [
        ("allocation_spearman", "Spearman", PALETTE["green_3"], "//"),
        ("allocation_overlap", "Q5 overlap", PALETTE["teal"], ".."),
    ]
    available = [item for item in metrics if item[0] in frame]
    if not available:
        return False
    models = _ordered_models(frame["model"])
    if not models:
        return False
    x = np.arange(len(models), dtype=float)
    width = 0.72 / len(available)
    offsets = (np.arange(len(available)) - (len(available) - 1) / 2.0) * width
    plotted = False
    for metric_index, (metric, label, color, hatch) in enumerate(available):
        summary = _seed_summary(frame, ["model"], metric)
        if summary.empty:
            continue
        summary = summary.set_index("model")
        means = np.array(
            [float(summary.loc[model, "mean"]) if model in summary.index else np.nan for model in models]
        )
        positions = x + offsets[metric_index]
        ax.bar(
            positions,
            means,
            width=width * 0.9,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            hatch=hatch,
            label=label,
            zorder=2,
        )
        error_x, error_y, error_sd = [], [], []
        for position, model, mean in zip(positions, models, means):
            if model not in summary.index or not np.isfinite(mean):
                continue
            row = summary.loc[model]
            if int(row["n_seed"]) >= 2 and np.isfinite(float(row["sd"])):
                error_x.append(float(position))
                error_y.append(float(mean))
                error_sd.append(float(row["sd"]))
        if error_x:
            ax.errorbar(
                error_x,
                error_y,
                yerr=error_sd,
                fmt="none",
                ecolor=PALETTE["ink"],
                elinewidth=1.1,
                capsize=2.3,
                zorder=4,
            )
        plotted = True
    ax.set_xticks(x, [_model_display(model) for model in models], rotation=32, ha="right")
    return plotted


def plot_mechanisms(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Plot four mechanism diagnostics: strength, magnitude, allocation, direction."""

    name = "four-mechanism diagnostics"
    if not _required(runs, {"family", "model", "seed"}, name):
        return []
    selected = _mechanism_subset(runs)
    apply_publication_style(style)
    fig, axes = create_subplots(1, 4, figsize=(20.0, 4.8), squeeze=False)
    strength_metric = (
        "underrepresented_fraction"
        if "underrepresented_fraction" in selected
        else "underrepresentation"
    )
    specifications = (
        (strength_metric, "Under-represented fraction", "Response strength"),
        ("magnitude_log_ratio", r"$|\log(\kappa_f/\kappa_S)|$", "Magnitude matching"),
        (None, "Allocation score", "Spatial allocation"),
        ("direction_alignment", "Alignment", "Directional matching"),
    )
    any_panel = False
    for index, (ax, (metric, ylabel, title)) in enumerate(zip(axes, specifications)):
        if metric is None:
            plotted = _allocation_panel(ax, selected)
        else:
            plotted = metric in selected and _model_seed_bars(ax, selected, metric)
        if not plotted:
            ax.set_axis_off()
            continue
        any_panel = True
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _panel_label(ax, index)
        _polish_axis(ax)
    if not any_panel:
        plt.close(fig)
        _warn_skip(name, "no finite mechanism metrics")
        return []
    has_legend = _figure_legend(fig, [ax for ax in axes if ax.axison], ncol=2)
    return finalize_figure(
        fig,
        Path(output_dir) / "mechanisms",
        layout_rect=(0.0, 0.0, 1.0, 0.90 if has_legend else 1.0),
    )


def _variant_order(values: Iterable[object]) -> list[str]:
    variants = {str(value) for value in values if pd.notna(value) and str(value).strip()}
    priority = ("full", "+router", "with_router", "-geometry", "no_geometry", "no_geom", "-gate", "no_gate", "no_physics")
    lower = {variant.lower(): variant for variant in variants}
    result = [lower[item] for item in priority if item in lower]
    result.extend(sorted(variants - set(result), key=str.lower))
    return result


def _variant_display(variant: object) -> str:
    key = str(variant).lower().strip()
    names = {
        "full": "Full",
        "-router": "− Router",
        "no_router": "− Router",
        "+router": "+ Router",
        "with_router": "+ Router",
        "-geometry": "− Geometry loss",
        "no_geometry": "− Geometry loss",
        "no_geom": "− Geometry loss",
        "-gate": "− Multiplicative gate",
        "no_gate": "- Multiplicative gate",
        "no_physics": "- AC physics skip",
    }
    return names.get(key, str(variant).replace("_", " "))


def _variant_seed_bars(ax: Axes, frame: pd.DataFrame, value: str) -> bool:
    summary = _seed_summary(frame, ["variant"], value)
    if summary.empty:
        return False
    variants = _variant_order(summary["variant"])
    if not variants:
        return False
    current = summary.set_index("variant")
    x = np.arange(len(variants), dtype=float)
    means = np.array([float(current.loc[item, "mean"]) for item in variants])
    alphas = np.linspace(1.0, 0.42, len(variants))
    bars = ax.bar(
        x,
        means,
        width=0.70,
        color=PALETTE["blue_secondary"],
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )
    for index, (bar, alpha) in enumerate(zip(bars, alphas)):
        bar.set_alpha(float(alpha))
        bar.set_hatch(_HATCHES[index % len(_HATCHES)])
    error_x, error_y, error_sd = [], [], []
    for index, item in enumerate(variants):
        row = current.loc[item]
        if int(row["n_seed"]) >= 2 and np.isfinite(float(row["sd"])):
            error_x.append(float(x[index]))
            error_y.append(float(row["mean"]))
            error_sd.append(float(row["sd"]))
    if error_x:
        ax.errorbar(
            error_x,
            error_y,
            yerr=error_sd,
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=1.2,
            capsize=2.5,
            zorder=4,
        )
    ax.set_xticks(x, [_variant_display(item) for item in variants], rotation=25, ha="right")
    return True


def plot_ablation(
    runs: pd.DataFrame,
    output_dir: str | Path,
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Plot the three core GANR ablations without pseudo-replicate errors."""

    name = "ablation"
    if not _required(runs, {"family", "variant", "seed"}, name):
        return []
    family = _family_text(runs)
    selected = runs.loc[family.str.contains("ablat", regex=False)].copy()
    main_full = runs.loc[
        family.str.contains("main", regex=False)
        & runs.get("model", pd.Series("", index=runs.index)).fillna("").astype(str).str.lower().isin({"ganr", "ours"})
    ].copy()
    if not main_full.empty:
        main_full["variant"] = "full"
        if not selected.empty and "system" in selected and "system" in main_full:
            systems = set(selected["system"].dropna().astype(str))
            main_full = main_full.loc[main_full["system"].astype(str).isin(systems)]
        selected = pd.concat([main_full, selected], ignore_index=True)
    if selected.empty:
        variant = runs["variant"].fillna("").astype(str).str.lower()
        selected = runs.loc[
            variant.str.contains(r"router|geometry|gate|full", regex=True)
        ].copy()
    if selected.empty:
        _warn_skip(name, "no ablation family or recognized variants")
        return []
    metrics = (
        ("nrmse", "Overall NRMSE", "Overall prediction"),
        ("q5_nrmse", "Q5 NRMSE", "High nonlinearity"),
        ("gap_relative", "Relative gap", "Representation gap"),
    )
    apply_publication_style(style)
    fig, axes = create_subplots(1, 3, figsize=(16.0, 4.8), squeeze=False)
    any_panel = False
    for index, (ax, (metric, ylabel, title)) in enumerate(zip(axes, metrics)):
        if metric not in selected or not _variant_seed_bars(ax, selected, metric):
            ax.set_axis_off()
            continue
        any_panel = True
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _panel_label(ax, index)
        _polish_axis(ax)
    if not any_panel:
        plt.close(fig)
        _warn_skip(name, "no finite ablation metrics")
        return []
    return finalize_figure(fig, Path(output_dir) / "ablation")


def generate_all_figures(
    runs_csv: str | Path = "results/runs.csv",
    quintiles_csv: str | Path = "results/quintiles.csv",
    output_dir: str | Path = "figures",
    *,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Generate every available paper figure and return all written files."""

    runs = _read_csv(runs_csv, "runs input")
    quintiles = _read_csv(quintiles_csv, "quintiles input")
    # The manuscript-level artifact intentionally omits the AC-9 benchmark;
    # preserve the raw CSVs while filtering it from every generated figure.
    if "system" in runs:
        runs = runs.loc[runs["system"].astype(str).str.lower() != "ac_power"].copy()
    if "system" in quintiles:
        quintiles = quintiles.loc[
            quintiles["system"].astype(str).str.lower() != "ac_power"
        ].copy()
    generators = (
        lambda: plot_error_vs_nonlinearity(quintiles, output_dir, style=style),
        lambda: plot_scaling(runs, output_dir, style=style),
        lambda: plot_performance_comparison(runs, output_dir, style=style),
        lambda: plot_mechanisms(runs, output_dir, style=style),
        lambda: plot_ablation(runs, output_dir, style=style),
    )
    generated: list[Path] = []
    for generator in generators:
        generated.extend(generator())
    return generated


__all__ = [
    "DEFAULT_COLORS",
    "FigureStyle",
    "PALETTE",
    "apply_publication_style",
    "create_subplots",
    "finalize_figure",
    "generate_all_figures",
    "plot_ablation",
    "plot_error_vs_nonlinearity",
    "plot_mechanisms",
    "plot_performance_comparison",
    "plot_scaling",
]
