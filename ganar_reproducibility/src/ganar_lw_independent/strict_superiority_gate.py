"""Strict all-combination superiority audit.

This gate is intentionally fail-closed: every registered system/activation
pair must win on prediction and all physical residual endpoints. Missing data,
failed runs, or a single losing seed invalidate the gate.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SYSTEMS = ("ieee118_acpf", "kuramoto64", "nonlinear_heat_fem")
ACTIVATIONS = ("relu", "gelu", "silu", "mish", "softplus", "swiglu")


def audit():
    paired = json.loads(Path("local_training_results/paired_activation_summary.json").read_text())
    physics = json.loads(Path("local_training_results/paired_physics_summary.json").read_text())
    paired_idx = {(r["system"], r["activation"]): r for r in paired}
    physics_idx = {(r["system"], r["activation"]): r for r in physics}
    failures = []
    for system in SYSTEMS:
        for activation in ACTIVATIONS:
            key = (system, activation)
            if key not in paired_idx:
                failures.append({"system": system, "activation": activation,
                                 "endpoint": "prediction", "reason": "missing paired row"})
                continue
            p = paired_idx[key]
            if not bool(p.get("all_seeds_win", False)):
                failures.append({"system": system, "activation": activation,
                                 "endpoint": "prediction", "reason": "not all seeds win",
                                 "delta_mean": p.get("delta_mean")})
            if key not in physics_idx:
                failures.append({"system": system, "activation": activation,
                                 "endpoint": "physics", "reason": "missing physics row"})
                continue
            q = physics_idx[key]
            for metric in ("residual_norm_mean", "residual_norm_p95",
                           "high_nonlinearity_residual_mean"):
                if not bool(q.get(metric + "_all_seeds_lower", False)):
                    failures.append({"system": system, "activation": activation,
                                     "endpoint": metric, "reason": "not all seeds lower",
                                     "delta": q.get(metric + "_delta")})
    return {"strict_all_combination_superiority": not failures,
            "total_combinations": len(SYSTEMS) * len(ACTIVATIONS),
            "failures": failures}


def main():
    result = audit()
    Path("local_training_results/strict_superiority_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["strict_all_combination_superiority"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
