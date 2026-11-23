#!/usr/bin/env python3
"""Train, evaluate, or run inference with the layer-wise GA-NAR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ga_nar.data import SYSTEMS, load_benchmark
from ga_nar.model import ACTIVATIONS, GeometryAdaptiveSurrogate
from ga_nar.training import fit, predict_batches


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GeometryAdaptiveSurrogate(
        projector=checkpoint["projector"], **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fit on the fixed training split")
    train.add_argument("--root", type=Path, default=ROOT)
    train.add_argument("--system", required=True, choices=SYSTEMS)
    train.add_argument("--activation", required=True, choices=sorted(ACTIVATIONS))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=400)
    train.add_argument("--patience", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--lambda-rho", type=float, default=0.5)
    train.add_argument("--lambda-pi", type=float, default=0.5)
    train.add_argument("--adapter-scale", type=float, default=0.1)
    train.add_argument("--beta-temperature", type=float, default=1.0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--threads", type=int, default=1,
                       help="CPU threads used for training")
    train.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="measure held-out standardized MSE")
    evaluate.add_argument("--root", type=Path, default=ROOT)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--threads", type=int, default=1)
    predict = sub.add_parser("predict", help="predict physical states from raw input array")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True,
                         help="NumPy .npy array of physical inputs, shape (N, d)")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    if args.command == "train":
        data = load_benchmark(args.root, args.system)
        torch.manual_seed(args.seed)
        config = {"input_dim": data.x.shape[1], "output_dim": data.y.shape[1],
                  "activation": args.activation, "adapter_scale": args.adapter_scale,
                  "beta_temperature": args.beta_temperature}
        model = GeometryAdaptiveSurrogate(projector=data.projector, **config)
        training = fit(model, data, seed=args.seed, epochs=args.epochs,
                       patience=args.patience, batch_size=args.batch_size,
                       learning_rate=args.learning_rate,
                       lambda_rho=args.lambda_rho, lambda_pi=args.lambda_pi,
                       device=args.device)
        test_prediction = predict_batches(model, data.x[data.test], args.batch_size)
        test_mse = float(np.mean((test_prediction - data.y[data.test]) ** 2))
        args.output.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output / "model.pt"
        torch.save({"state_dict": model.cpu().state_dict(), "model_config": config,
                    "projector": torch.as_tensor(data.projector), "system": args.system,
                    "seed": args.seed, "source_hashes": data.source_hashes,
                    "q_mean": torch.as_tensor(data.q_mean),
                    "q_scale": torch.as_tensor(data.q_scale),
                    "z_mean": torch.as_tensor(data.z_mean),
                    "z_scale": torch.as_tensor(data.z_scale)}, checkpoint)
        metrics = {"system": args.system, "activation": args.activation,
                   "seed": args.seed, "test_mse": test_mse,
                   "training": training, "source_hashes": data.source_hashes,
                   "checkpoint": str(checkpoint)}
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "training"} | {
            "best_epoch": training["best_epoch"],
            "validation_mse": training["validation_mse"]}, indent=2))
    elif args.command == "evaluate":
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        data = load_benchmark(args.root, checkpoint["system"])
        if data.source_hashes != checkpoint["source_hashes"]:
            raise ValueError("dataset or geometry differs from the training checkpoint")
        prediction = predict_batches(model, data.x[data.test])
        mse = float(np.mean((prediction - data.y[data.test]) ** 2))
        print(json.dumps({"system": data.system, "activation": model.activation,
                          "test_mse": mse, "test_samples": len(data.test)}, indent=2))
    else:
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        raw = np.load(args.input, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != model.input_dim:
            raise ValueError("input must have shape (N, input_dim)")
        standardized = (raw - checkpoint["q_mean"].numpy()) / checkpoint["q_scale"].numpy()
        output = predict_batches(model, standardized.astype(np.float32))
        physical = output * checkpoint["z_scale"].numpy() + checkpoint["z_mean"].numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, physical)
        print(json.dumps({"output": str(args.output), "samples": len(physical)}))


if __name__ == "__main__":
    main()
