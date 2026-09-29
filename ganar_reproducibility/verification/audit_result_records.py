#!/usr/bin/env python3
"""Audit the four canonical result namespaces shipped in this package.

The audit intentionally depends only on the frozen result records.  It does
not require a manuscript archive, a parent project, or a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def audit_nrgd() -> dict:
    rows = json.loads(
        (ROOT / "results/nrgd_five_benchmark_descriptor/five_benchmark_summary.json").read_text()
    )
    assert len(rows) == 5
    assert all(row.get("status") == "RESOLVED" for row in rows)
    return {"rows": len(rows), "benchmarks": [row["benchmark"] for row in rows]}


def audit_ganr() -> dict:
    frame = pd.read_csv(ROOT / "results/ganr_shared_physics_factorial/runs.csv")
    systems = {"controlled", "duffing", "ieee118", "allen_cahn", "shallow_water"}
    frame = frame[frame.family.eq("all_physics_factorial") & frame.system.isin(systems)]
    expected = 5 * 7 * 2 * 3
    assert len(frame) == expected
    assert frame[["system", "model", "variant", "seed"]].drop_duplicates().shape[0] == expected
    return {"rows": len(frame), "systems": int(frame.system.nunique()), "models": int(frame.model.nunique())}


def audit_layerwise() -> dict:
    rows = json.loads(
        (ROOT / "results/ga_nar_lw_activation_matrix/main_activation_matrix.json").read_text()
    )
    assert len(rows) == 18
    assert all(row.get("all_seeds_win") for row in rows)
    return {"rows": len(rows), "all_seed_wins": True}


def audit_ablation() -> dict:
    rows = json.loads(
        (ROOT / "results/ga_nar_lw_controlled_ablation/final_ga_ablation_summary.json").read_text()
    )
    methods = {"Original Baseline", "Capacity Control", "Generic Geometry Control", "Full GA-NAR"}
    assert len(rows) == 7
    assert all(methods.issubset(set(row["methods"])) for row in rows)
    physics = json.loads(
        (ROOT / "results/ga_nar_lw_controlled_ablation/physics_satisfaction.json").read_text()
    )
    assert len(physics) == 84
    assert not any(item.get("status") == "failed" for item in physics)
    return {"settings": len(rows), "physics_rows": len(physics), "methods": sorted(methods)}


def main() -> None:
    result = {
        "nrgd": audit_nrgd(),
        "ganr": audit_ganr(),
        "ga_nar_lw": audit_layerwise(),
        "controlled_ablation": audit_ablation(),
        "pass": True,
    }
    (ROOT / "manifests/result_record_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
