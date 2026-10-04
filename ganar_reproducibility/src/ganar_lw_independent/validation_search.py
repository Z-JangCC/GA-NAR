from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .data import load_independent_dataset, nested_train_validation
from .run_one import build_model
from .training.trainer import train_model


DEFAULT_CANDIDATES = ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.25),
                      (0.5, 0.25), (0.25, 0.5), (0.1, 0.1))


def evaluate_candidate(system, candidate, seed, epochs, root=".", device=None):
    data = load_independent_dataset(system, root)
    inner_train, inner_val = nested_train_validation(data, fold=seed)
    model = build_model("ganar_lw", data)
    result = train_model(model, data.x[inner_train], data.y[inner_train],
                         data.x[inner_val], data.y[inner_val],
                         x_geometry=data.geometry_x, rho_target=data.rho,
                         pi_target=data.pi, seed=seed, epochs=epochs,
                         lambda_rho=candidate[0], lambda_pi=candidate[1],
                         device=device, order_namespace="validation_common")
    return {"system": system, "lambda_rho": candidate[0],
            "lambda_pi": candidate[1], "seed": seed,
            "best_val": result.best_val, "best_epoch": result.best_epoch,
            "epochs": epochs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="ieee118_acpf")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--devices", default=None)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    devices = [f"cuda:{x}" for x in args.devices.split(",")] if args.devices else [None]
    jobs = [(candidate, seed, devices[i % len(devices)])
            for i, (candidate, seed) in enumerate((c, s)
            for c in DEFAULT_CANDIDATES for s in seeds)]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_candidate, args.system, candidate,
                               seed, args.epochs, ".", device)
                   for candidate, seed, device in jobs]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); print(row, flush=True)
    grouped = {}
    for row in rows:
        key = (row["lambda_rho"], row["lambda_pi"])
        grouped.setdefault(key, []).append(row["best_val"])
    ranking = sorted(({"lambda_rho": k[0], "lambda_pi": k[1],
                        "mean_inner_val": float(np.mean(v)),
                        "seed_values": v} for k, v in grouped.items()),
                     key=lambda x: x["mean_inner_val"])
    out = Path("local_training_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_search.json").write_text(json.dumps({"rows": rows,
        "ranking": ranking}, indent=2, default=str))
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
