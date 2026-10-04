from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MATRIX = {
    "ieee118_acpf": ["ganar_lw", "ganar_relu", "ganar_gelu", "ganar_silu", "ganar_mish", "ganar_softplus", "ganar_uniform_beta", "ganar_no_geometry",
                      "ganar_prebackbone", "ganar_shuffled_geometry",
                      "pm_swiglu", "pm_relu", "pm_gelu", "pm_tanh",
                      "pm_silu", "pm_mish", "pm_softplus"],
    "kuramoto64": ["ganar_lw", "ganar_relu", "ganar_gelu", "ganar_silu", "ganar_mish", "ganar_softplus", "pm_swiglu", "pm_relu", "pm_gelu",
                    "pm_tanh", "pm_silu", "pm_mish", "pm_softplus"],
    "nonlinear_heat_fem": ["ganar_lw", "ganar_relu", "ganar_gelu", "ganar_silu", "ganar_mish", "ganar_softplus", "pm_swiglu", "pm_relu", "pm_gelu",
                            "pm_tanh", "pm_silu", "pm_mish", "pm_softplus"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="ieee118_acpf")
    p.add_argument("--models", default=None)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device-prefix", default=None,
                   help="comma-separated CUDA device ids; workers cycle through them")
    p.add_argument("--device", default="cuda",
                   help="torch device passed to each run (cuda or cpu)")
    p.add_argument("--lambda-rho", type=float, default=.5)
    p.add_argument("--lambda-pi", type=float, default=.5)
    p.add_argument("--physics-weight", type=float, default=0.0)
    a = p.parse_args()
    systems = [a.system]
    models = a.models.split(",") if a.models else MATRIX[a.system]
    seeds = [int(x) for x in a.seeds.split(",")]
    devices = a.device_prefix.split(",") if a.device_prefix else [None]
    specs = [(a.system, m, seed, devices[i % len(devices)])
             for i, (m, seed) in enumerate((m, s) for m in models for s in seeds)]
    def launch(spec):
        system, model, seed, physical_gpu = spec
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1"})
        if physical_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        cmd = ["python", "-m", "ganar_lw_independent.run_one",
               "--system", system, "--model", model, "--seed", str(seed),
               "--epochs", str(a.epochs), "--lambda-rho", str(a.lambda_rho),
               "--lambda-pi", str(a.lambda_pi), "--device", a.device,
               "--output-root", "local_training_results"]
        cmd += ["--physics-weight", str(a.physics_weight)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{system}/{model}/{seed} failed:\n{proc.stdout}\n{proc.stderr}")
        metrics_path = Path("local_training_results") / "results" / system / model / str(seed) / "metrics.json"
        return json.loads(metrics_path.read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(launch, spec) for spec in specs]
        for f in as_completed(futures):
            row = f.result(); rows.append(row)
            print(row["system"], row["model"], row["seed"], row["mse_std"], flush=True)
    out = Path("local_training_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "all_runs.json").write_text(json.dumps(sorted(rows, key=lambda r: (r["system"], r["model"], r["seed"])), indent=2, default=str))


if __name__ == "__main__":
    main()
