from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_independent_dataset
from .evaluation import layerwise_diagnostics, predict_mse
from .models.baselines import MatchedActivation, MatchedReLU, MatchedSwiGLU, count_parameters, matched_relu_width, matched_swiglu_width
from .models.controls import create_control
from .models.layerwise_ganar import LayerwiseGANAR, trainable_parameter_count
from .models.ablation import GenericAblation
from .protocol.protocol_hash import protocol_hash
from .training.trainer import train_model
from .physics_torch import HeatFEMTorchResidual
from .physics_torch_kuramoto import KuramotoTorchResidual
from .physics_torch_powerflow import PowerFlowTorchResidual


def _ridge_adapter_init(model, x_train, y_train, x_val, y_val,
                        batch_size: int = 2048):
    """Initialize the zero highway with a validation-selected residual fit.

    The anchor is fixed at this stage, so this is a closed-form warm start for
    the existing ``input_skip`` correction.  Only train/validation arrays are
    used; the independent test split is untouched.
    """
    model.anchor.eval()
    device = next(model.anchor.parameters()).device
    dtype = next(model.anchor.parameters()).dtype

    def predict(x):
        out = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(x[start:start + batch_size], dtype=dtype, device=device)
                out.append(model.anchor(xb).detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    p_train = predict(x_train)
    p_val = predict(x_val)
    residual_train = np.asarray(y_train) - p_train
    residual_val = np.asarray(y_val) - p_val
    X_train = np.c_[np.asarray(x_train), np.ones(len(x_train))]
    X_val = np.c_[np.asarray(x_val), np.ones(len(x_val))]
    n_features = X_train.shape[1]
    best = None
    for ridge_lambda in np.logspace(-10, 4, 29):
        gram = X_train.T @ X_train + ridge_lambda * np.eye(n_features)
        # Do not regularize the intercept.
        gram[-1, -1] = X_train.shape[0]
        coeff = np.linalg.solve(gram, X_train.T @ residual_train)
        val_mse = float(np.mean((p_val + X_val @ coeff - np.asarray(y_val)) ** 2))
        if best is None or val_mse < best[0]:
            best = (val_mse, float(ridge_lambda), coeff)
    _, ridge_lambda, coeff = best
    with torch.no_grad():
        model.input_skip.weight.copy_(torch.as_tensor(coeff[:-1].T, dtype=model.input_skip.weight.dtype))
        model.input_skip.bias.copy_(torch.as_tensor(coeff[-1], dtype=model.input_skip.bias.dtype))
    return ridge_lambda, best[0]


def build_model(name, data, gamma_init: float = -8.0, anchor_features: bool = False):
    d, m = data.x.shape[1], data.y.shape[1]
    common = dict(input_dim=d, output_dim=m, projector=data.projector)
    if name.startswith('capacity_') or name.startswith('generic_geometry_'):
        activation=name.split('_')[-1]; mode='capacity' if name.startswith('capacity_') else 'generic'
        reference=LayerwiseGANAR(d,m,data.projector,activation='silu' if activation=='swiglu' else activation,
                                 block_kind='swiglu' if activation=='swiglu' else 'activation')
        target=sum(p.numel() for p in reference.delta_output.parameters())+sum(p.numel() for p in reference.input_skip.parameters())
        return GenericAblation(d,m,data.projector,activation,mode,geometry_x=data.geometry_x,
                               rho_target=data.rho,pi_target=data.pi,target_params=target)
    if name in {"ganar_lw", "ganar_uniform_beta", "ganar_no_geometry", "ganar_prebackbone"}:
        if name == "ganar_lw":
            return LayerwiseGANAR(d, m, data.projector, activation="silu", gamma_init=gamma_init,
                                  anchor_features=anchor_features)
        return create_control(name, gamma_init=gamma_init, anchor_features=anchor_features, **common)
    if name.startswith("ganar_") and name[6:] in {"relu", "gelu", "silu", "mish", "softplus"}:
        return LayerwiseGANAR(d, m, data.projector, activation=name[6:], gamma_init=gamma_init,
                              anchor_features=anchor_features)
    if name == "ganar_swiglu":
        return LayerwiseGANAR(d, m, data.projector, activation="silu", block_kind="swiglu", gamma_init=gamma_init,
                              anchor_features=anchor_features)
    if name == "ganar_shuffled_geometry":
        rng = np.random.default_rng(20260924)
        p = rng.normal(size=data.projector.shape)
        q, _ = np.linalg.qr(p)
        rank = int(round(np.trace(data.projector)))
        random_projector = q[:, :rank] @ q[:, :rank].T
        return LayerwiseGANAR(d, m, random_projector, gamma_init=gamma_init,
                              anchor_features=anchor_features)
    if name == "pm_swiglu":
        # The independent protocol intentionally keeps the conventional
        # comparator at its standard 256-wide backbone.  GA-NAR-LW's extra
        # geometry/residual parameters are reported explicitly rather than
        # silently forcing an artificial parameter match.
        return MatchedSwiGLU(d, m, ff_width=256)
    if name == "pm_relu":
        return MatchedReLU(d, m, ff_width=256)
    if name.startswith("pm_") and name[3:] in MatchedActivation.ACTIVATIONS:
        return MatchedActivation(d, m, activation=name[3:], ff_width=256)
    raise KeyError(name)


def run(system: str, model_name: str, seed: int, epochs: int = 400,
        lambda_rho: float = 0.5, lambda_pi: float = 0.5,
        root: str = ".", device: str | None = None,
        physics_weight: float = 0.0,
        output_root: str = "ganar_lw_independent",
        gamma_init: float = -8.0,
        anchor_features: bool = False,
        tanh_gate: bool = False,
        ridge_adapter: bool = True):
    data = load_independent_dataset(system, root)
    model = build_model(model_name, data, gamma_init=gamma_init, anchor_features=anchor_features)
    # Kuramoto's response map is especially well served by a wide smooth
    # anchor. Keep the complete Layer-wise geometry modules in the model and
    # checkpoint, but avoid letting their much larger parameter set dominate
    # the joint optimizer during this system-specific final run.
    activation_adapter = (model_name.startswith("ganar_") and model_name[6:] in {"relu","gelu","silu","mish","softplus","swiglu"})
    ablation_control = model_name.startswith('capacity_') or model_name.startswith('generic_geometry_')
    paired_anchor_mode = (model_name == "ganar_lw" or activation_adapter or ablation_control)
    if paired_anchor_mode:
        model.anchor_only = True
        if tanh_gate:
            model.anchor_gate = True
            # Exact epoch-0 identity with a non-saturated, trainable gate.  The
            # anchor remains a safe fallback while the correction learns.
            with torch.no_grad():
                model.gamma_head.bias.zero_()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Seed-matched PM-Mish warm start; the final GA run learns only its
        # Layer-wise correction on top of that fixed anchor.
        if model_name == 'ganar_lw': warm_model='pm_silu'
        elif ablation_control: warm_model='pm_'+model_name.split('_')[-1]
        else: warm_model="pm_"+model_name[6:]
        warm = Path(root) / "ganar_lw_independent" / "results" / system / warm_model / str(seed) / "checkpoint.pt"
        if warm.exists():
            state = torch.load(warm, map_location="cpu", weights_only=False).get("state_dict", {})
            model.anchor.load_state_dict(state, strict=False)
        if ablation_control:
            for parameter in model.correction.parameters(): parameter.requires_grad_(True)
        else:
            for module in (model.input_skip, model.delta_output, model.gamma_head,
                           model.geometry, model.rho_head, model.pi_head,
                           model.beta_head, model.stem, model.anchor_projection):
                for parameter in module.parameters(): parameter.requires_grad_(True)
    train_idx, val_idx = data.train, data.validation_outer
    ridge_lambda = None
    ridge_val = None
    if paired_anchor_mode and ridge_adapter and hasattr(model, 'input_skip'):
        ridge_lambda, ridge_val = _ridge_adapter_init(
            model, data.x[train_idx], data.y[train_idx],
            data.x[val_idx], data.y[val_idx])
    geometry_model = hasattr(model, "rho_head")
    raw = np.load(Path(root) / "data_store" / "processed" / system / "dataset.npz")
    physics_fn = None
    if physics_weight and system == "nonlinear_heat_fem" and geometry_model:
        physics_fn = HeatFEMTorchResidual(raw["z_mean"], raw["z_scale"])
    if physics_weight and system == "kuramoto64" and geometry_model:
        physics_fn = KuramotoTorchResidual(raw["z_mean"], raw["z_scale"])
    if physics_weight and system == "ieee118_acpf" and geometry_model:
        physics_fn = PowerFlowTorchResidual(raw["z_mean"], raw["z_scale"])
    result = train_model(
        model, data.x[train_idx], data.y[train_idx], data.x[val_idx], data.y[val_idx],
        x_geometry=data.geometry_x if geometry_model else None,
        rho_target=data.rho if geometry_model else None,
        pi_target=data.pi if geometry_model else None,
        seed=seed, epochs=epochs, lambda_rho=lambda_rho if geometry_model else 0.0,
        lambda_pi=lambda_pi if geometry_model else 0.0, device=device,
        order_namespace="formal_common",
        gradient_clip=1.0e9 if geometry_model else 1.0,
        lr_max=1.0e-3,
        q_train_physical=raw["q"][train_idx] if physics_fn is not None else None,
        physics_residual_fn=physics_fn, physics_weight=physics_weight,
        validation_residual_fn=physics_fn, validation_q_physical=raw["q"][val_idx] if physics_fn is not None else None,
        validation_residual_weight=physics_weight)
    model = model.double().eval()
    mse = predict_mse(model, data.x[data.test], data.y[data.test])
    row = {
        "protocol": "ganar-lw-independent-v1",
        "protocol_hash": protocol_hash(Path(__file__).parent / "protocol" / "final_protocol.yaml"),
        "system": system, "model": model_name, "seed": seed,
        "mse_std": mse, "best_val": result.best_val,
        "best_epoch": result.best_epoch, "epochs_completed": result.epochs_completed,
        "wall_time": result.wall_time, "parameter_count": result.parameter_count,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "status": "completed", "data_manifest": data.manifest,
        "train_order_namespace": result.order_namespace,
    }
    if ridge_lambda is not None:
        row["ridge_adapter"] = True
        row["ridge_lambda"] = ridge_lambda
        row["ridge_val_mse"] = ridge_val
    else:
        row["ridge_adapter"] = False
    if geometry_model:
        row["diagnostics"] = layerwise_diagnostics(model, data.geometry_x)
    out = Path(root) / output_root / "results" / system / model_name / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model": model_name,
                "system": system, "seed": seed}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lambda-rho", type=float, default=.5)
    parser.add_argument("--lambda-pi", type=float, default=.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--output-root", default="ganar_lw_independent")
    parser.add_argument("--gamma-init", type=float, default=-8.0)
    parser.add_argument("--anchor-features", action="store_true")
    parser.add_argument("--tanh-gate", action="store_true")
    parser.add_argument("--no-ridge-adapter", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.system, args.model, args.seed, args.epochs,
                         args.lambda_rho, args.lambda_pi, ".", args.device,
                         args.physics_weight, args.output_root, args.gamma_init,
                         args.anchor_features, args.tanh_gate, not args.no_ridge_adapter),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
