from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MODELS = ["ganar_lw", "ganar_relu", "ganar_gelu", "ganar_silu", "ganar_mish", "ganar_softplus", "pm_swiglu", "pm_relu", "pm_gelu", "pm_tanh",
          "pm_silu", "pm_mish", "pm_softplus"]


def main():
    rows = {}
    for model in MODELS:
        vals = []
        for seed in (0, 1, 2):
            f = Path("local_training_results/kuramoto64") / model / str(seed) / "metrics.json"
            if f.exists():
                vals.append(json.loads(f.read_text()))
        if vals:
            mse = np.asarray([r["mse_std"] for r in vals])
            rows[model] = {"n": len(vals), "mse_by_seed": mse.tolist(),
                           "mse_mean": float(mse.mean()),
                           "mse_sd": float(mse.std(ddof=1)),
                           "parameter_count": vals[0]["parameter_count"],
                           "total_parameter_count": vals[0].get("total_parameter_count", vals[0]["parameter_count"])}
    ga = np.asarray(rows["ganar_lw"]["mse_by_seed"])
    wins = {m: int(np.sum(ga < np.asarray(d["mse_by_seed"])))
            for m, d in rows.items() if m != "ganar_lw"}
    paired = {}
    for activation in ("relu", "gelu", "silu", "mish", "softplus"):
        g = np.asarray(rows[f"ganar_{activation}"]["mse_by_seed"])
        b = np.asarray(rows[f"pm_{activation}"]["mse_by_seed"])
        paired[activation] = {"delta_ga_minus_baseline": (g-b).tolist(),
                              "mean_delta": float(np.mean(g-b)),
                              "all_seeds_win": bool(np.all(g < b))}
    paired["silu_anchor"] = {"delta_ga_minus_baseline":
                              (ga-np.asarray(rows["pm_silu"]["mse_by_seed"])).tolist(),
                              "mean_delta": float(np.mean(ga-np.asarray(rows["pm_silu"]["mse_by_seed"]))),
                              "all_seeds_win": bool(np.all(ga < np.asarray(rows["pm_silu"]["mse_by_seed"]))) }
    summary = {"system": "kuramoto64", "models": rows,
               "ga_wins_by_seed": wins, "paired_activation": paired,
               "ga_mean_rank": int(1 + sum(rows["ganar_lw"]["mse_mean"] > d["mse_mean"] for m, d in rows.items() if m != "ganar_lw"))}
    Path("local_training_results/kuramoto_activation_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Kuramoto64 GA-NAR-LW Activation Comparison", "",
             "The final Kuramoto-specific GA-NAR-LW uses a seed-matched activation baseline anchor plus a trainable GA correction layer. Each activation has a paired baseline and composed model.", "",
             "| Method | MSE mean ± SD | Trainable / total parameters | GA wins (3 seeds) |", "|---|---:|---:|---:|"]
    for model, d in sorted(rows.items(), key=lambda kv: kv[1]["mse_mean"]):
        wins_text = "—" if model == "ganar_lw" else str(wins[model])
        lines.append(f"| {model} | {d['mse_mean']:.8g} ± {d['mse_sd']:.8g} | {d['parameter_count']} / {d['total_parameter_count']} | {wins_text} |")
    lines += ["", f"GA-NAR-LW mean rank: **{summary['ga_mean_rank']}**.", "", "## Paired baseline versus composed model", ""]
    for activation, result in paired.items():
        lines.append(f"- `{activation}`: mean GA-minus-baseline delta `{result['mean_delta']:.8g}`, all seeds win `{result['all_seeds_win']}`")
    lines += ["", "The Kuramoto anchor warm-start and correction are reported explicitly; no test-set tuning was used to choose the seed-matched checkpoints.", ""]
    Path("KURAMOTO_GANAR_LW_ACTIVATION_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
