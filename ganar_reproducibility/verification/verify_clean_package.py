#!/usr/bin/env python3
"""Fail-closed verification for an assembled clean GitHub package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / "manifests/package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    listed = {row["path"]: row for row in manifest["files"]}
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".pyc"}
    forbidden_parts = {"__pycache__", ".pytest_cache", "tmp", "checkpoints", "paper"}
    forbidden = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if path.suffix.lower() in forbidden_suffixes or any(part in forbidden_parts for part in path.relative_to(ROOT).parts):
            forbidden.append(rel)
        generated_audit = rel in {
            "manifests/result_record_audit.json",
            "manifests/final_reproduction_audit.json",
        }
        if rel != "manifests/package_manifest.json" and not generated_audit and rel not in listed:
            forbidden.append(f"unmanifested:{rel}")
    if forbidden:
        raise SystemExit(f"forbidden/unmanifested files: {forbidden[:20]}")
    for rel, row in listed.items():
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing manifest file: {rel}")
        if rel in {
            "manifests/result_record_audit.json",
            "manifests/final_reproduction_audit.json",
        }:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"hash mismatch: {rel}")
    required = [
        "configs/nrgd_five_benchmark.yaml",
        "configs/ganr_shared_physics_factorial.yaml",
        "configs/ga_nar_lw_layerwise.yaml",
        "configs/ga_nar_lw_controlled_ablation.yaml",
        "results/nrgd_five_benchmark_descriptor/five_benchmark_summary.json",
        "results/ganr_shared_physics_factorial/runs.csv",
        "results/ga_nar_lw_activation_matrix/main_activation_matrix.json",
        "results/ga_nar_lw_controlled_ablation/final_ga_ablation_summary.json",
        "manifests/result_record_audit.json",
        "experiments/reproduce_all.py",
        "reports/ganr_shared_physics_factorial_summary.json",
        "reports/ga_nar_lw_controlled_ablation.md",
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            raise SystemExit(f"missing required artifact: {rel}")
    audit = json.loads((ROOT / "manifests/result_record_audit.json").read_text())
    if not audit.get("pass"):
        raise SystemExit("result-record audit did not pass")
    print(json.dumps({"pass": True, "manifest_files": len(listed), "forbidden_files": 0}, indent=2))


if __name__ == "__main__":
    main()
