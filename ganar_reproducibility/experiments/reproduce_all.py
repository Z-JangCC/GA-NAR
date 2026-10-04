#!/usr/bin/env python3
"""Public staged reproduction orchestrator.

The stages are intentionally sequential where a paired model consumes a
seed-matched baseline checkpoint. Results are written to
`local_reproduction_results/` and never replace the distributed numerical
result records.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("all", "nrgd", "ganr", "activation", "ablation"), default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--gpu-ids", default="0")
    args = p.parse_args()
    py = sys.executable

    if args.stage in {"all", "nrgd"}:
        run([py, "experiments/run_nrgd_five_benchmark.py", "--mode", "formal",
             "--benchmark", "all", "--output-root",
             "local_reproduction_results/nrgd_five_benchmark_descriptor"])

    if args.stage in {"all", "ganr"}:
        run([py, "experiments/run_ganr_shared_physics_factorial.py",
             "--output-root", "local_reproduction_results/ganr_shared_physics_factorial"])

    systems = {
        "ieee118_acpf": "pm_relu,pm_gelu,pm_silu,pm_mish,pm_softplus,pm_swiglu",
        "kuramoto64": "pm_relu,pm_gelu,pm_silu,pm_mish,pm_softplus,pm_swiglu",
        "nonlinear_heat_fem": "pm_relu,pm_gelu,pm_silu,pm_mish,pm_softplus,pm_swiglu",
    }
    adapters = {
        "ieee118_acpf": "ganar_relu,ganar_gelu,ganar_silu,ganar_mish,ganar_softplus,ganar_swiglu",
        "kuramoto64": "ganar_relu,ganar_gelu,ganar_silu,ganar_mish,ganar_softplus,ganar_swiglu",
        "nonlinear_heat_fem": "ganar_relu,ganar_gelu,ganar_silu,ganar_mish,ganar_softplus,ganar_swiglu",
    }
    if args.stage in {"all", "activation"}:
        for system, models in systems.items():
            run([py, "experiments/run_ga_nar_lw_matrix.py", "--system", system,
                 "--models", models, "--seeds", args.seeds, "--epochs", str(args.epochs),
                 "--workers", str(args.workers), "--device-prefix", args.gpu_ids,
                 "--device", args.device, "--lambda-rho", "0", "--lambda-pi", "0"])
        for system, models in adapters.items():
            run([py, "experiments/run_ga_nar_lw_matrix.py", "--system", system,
                 "--models", models, "--seeds", args.seeds, "--epochs", str(args.epochs),
                 "--workers", str(args.workers), "--device-prefix", args.gpu_ids,
                 "--device", args.device, "--lambda-rho", "0.5", "--lambda-pi", "0.5"])

    if args.stage in {"all", "ablation"}:
        ablation = {
            "ieee118_acpf": "capacity_relu,capacity_gelu,capacity_swiglu,generic_geometry_relu,generic_geometry_gelu,generic_geometry_swiglu",
            "kuramoto64": "capacity_relu,capacity_gelu,generic_geometry_relu,generic_geometry_gelu",
            "nonlinear_heat_fem": "capacity_relu,capacity_softplus,generic_geometry_relu,generic_geometry_softplus",
        }
        for system, models in ablation.items():
            run([py, "experiments/run_ga_nar_lw_matrix.py", "--system", system,
                 "--models", models, "--seeds", args.seeds, "--epochs", str(args.epochs),
                 "--workers", str(args.workers), "--device-prefix", args.gpu_ids,
                 "--device", args.device, "--lambda-rho", "0", "--lambda-pi", "0"])
        run([py, "experiments/build_ga_nar_lw_controlled_report.py",
             "--output", "local_reproduction_results/ga_nar_lw_controlled_ablation.md"])

    run([py, "verification/verify_final_reproduction.py"])


if __name__ == "__main__":
    main()
