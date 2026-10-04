from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    rows = json.loads(Path("local_training_results/physics_satisfaction.json").read_text())
    grouped = {}
    for row in rows:
        if row.get("status") != "failed":
            grouped.setdefault((row["system"], row["model"]), []).append(row)
    lines = ["# Physical Equation Satisfaction Diagnostics", "",
             "Residuals are computed by independently re-evaluating the physical equations at the surrogate test predictions. Metrics are not prediction MSE: they are equation residual norms, P95 residuals, maximum residuals, and top-quartile high-nonlinearity residuals.", "", "| System | Model | Residual mean | P95 | High-nonlinearity mean |", "|---|---|---:|---:|---:|"]
    for (system, model), values in grouped.items():
        mean = np.mean([v["residual_norm_mean"] for v in values])
        p95 = np.mean([v["residual_norm_p95"] for v in values])
        high = np.mean([v["high_nonlinearity_residual_mean"] for v in values])
        lines.append(f"| {system} | {model} | {mean:.8g} | {p95:.8g} | {high:.8g} |")
    lines += ["", "## Interpretation", "", "The latest paired residual evaluation uses the same test predictions and the exact registered residual functions, including the triangular FEM residual for nonlinear heat. IEEE118 and Kuramoto show broad residual reductions for the latest paired GA-NAR-LW models. Nonlinear heat is mixed: several activation pairs improve, while ReLU, Softplus, and some SwiGLU endpoints do not improve for every seed. Therefore the current evidence supports a system- and activation-dependent physics-satisfaction advantage, not a universal all-endpoint residual superiority claim.", ""]
    Path("PHYSICS_SATISFACTION_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
