#!/usr/bin/env python3
"""End-to-end audit for the normalized numerical reproduction package.

This verifies all four canonical result namespaces without using parent-project
paths or a manuscript archive. Training entry points are smoke-checked
separately because full retraining requires the optional weights/data compute
budget.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def check_table2() -> dict:
    source = pd.read_csv(ROOT / "results/ganr_shared_physics_factorial/runs.csv")
    expected_models = {"ganr", "relu", "gelu", "silu", "swiglu", "resmlp", "fourier"}
    expected_systems = {"controlled", "duffing", "ieee118", "allen_cahn", "shallow_water"}
    expected_variants = {"no_physics", "physics"}
    source = source[(source.family == "all_physics_factorial") & source.system.isin(expected_systems)]
    assert len(source) == 210
    assert set(source.model) == expected_models
    assert set(source.system) == expected_systems
    assert set(source.variant) == expected_variants
    assert source.groupby(["system", "model", "variant"]).size().eq(3).all()
    return {"rows": 70, "seed_rows": len(source), "passed": True}


def check_layerwise() -> dict:
    rows = json.loads((ROOT / "results/ga_nar_lw_activation_matrix/main_activation_matrix.json").read_text())
    assert len(rows) == 18
    assert all(r.get("all_seeds_win") for r in rows)
    return {"rows": len(rows), "all_seed_wins": True}


def check_ablation() -> dict:
    rows = json.loads((ROOT / "results/ga_nar_lw_controlled_ablation/final_ga_ablation_summary.json").read_text())
    expected = {"Original Baseline", "Capacity Control", "Generic Geometry Control", "Full GA-NAR"}
    assert len(rows) == 7
    assert all(expected.issubset(set(r["methods"])) for r in rows)
    physics = json.loads((ROOT / "results/ga_nar_lw_controlled_ablation/physics_satisfaction.json").read_text())
    assert len(physics) == 84 and not any(x.get("status") == "failed" for x in physics)
    return {"settings": len(rows), "physics_rows": len(physics), "methods": sorted(expected)}


def check_nrgd() -> dict:
    summary = json.loads((ROOT / "results/nrgd_five_benchmark_descriptor/five_benchmark_summary.json").read_text())
    assert len(summary) == 5
    assert all(x["status"] == "RESOLVED" for x in summary)
    return {"benchmarks": [x["benchmark"] for x in summary], "rows": len(summary)}


def check_required_experiments() -> dict:
    required = [
        ROOT / "experiments/run_nrgd_five_benchmark.py",
        ROOT / "experiments/run_ganr_shared_physics_factorial.py",
        ROOT / "experiments/run_ga_nar_lw_matrix.py",
        ROOT / "experiments/build_ga_nar_lw_controlled_report.py",
        ROOT / "configs/nrgd_five_benchmark.yaml",
        ROOT / "configs/ganr_shared_physics_factorial.yaml",
        ROOT / "configs/ga_nar_lw_layerwise.yaml",
        ROOT / "configs/ga_nar_lw_controlled_ablation.yaml",
    ]
    assert all(path.exists() for path in required), [str(path) for path in required if not path.exists()]
    return {"entry_points": len(required), "all_present": True}


def main() -> None:
    result = {"table2": check_table2(), "layerwise": check_layerwise(), "controlled_ablation": check_ablation(), "nrgd": check_nrgd(), "required_experiments": check_required_experiments()}
    result["pass"] = True
    (ROOT / "manifests/final_reproduction_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
