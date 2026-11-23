"""Layer-wise GA-NAR model for standardized physical-system inputs and states."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


ACTIVATIONS = {"relu", "gelu", "silu", "mish", "softplus", "swiglu"}


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "silu" or name == "swiglu":
        return F.silu(x)
    if name == "mish":
        return F.mish(x)
    if name == "softplus":
        return F.softplus(x)
    raise ValueError(f"unsupported activation: {name}")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_width: int, activation: str):
        super().__init__()
        self.activation = activation
        self.norm = RMSNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, ff_width)
        self.value = nn.Linear(hidden_dim, ff_width) if activation == "swiglu" else None
        self.output = nn.Linear(ff_width, hidden_dim)

    def base_terms(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.norm(hidden)
        return self.gate(normalized), self.value(normalized) if self.value is not None else None

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor,
                value: torch.Tensor | None) -> torch.Tensor:
        activated = _activation(self.activation, gate)
        if value is not None:
            activated = activated * value
        return hidden + self.output(activated)


class BranchProjection(nn.Module):
    def __init__(self, code_dim: int, gate_dim: int, rank: int,
                 generator: torch.Generator):
        super().__init__()
        highway = torch.randn(gate_dim, code_dim, generator=generator) / math.sqrt(code_dim)
        self.register_buffer("highway", highway)
        self.down = nn.Linear(code_dim, rank, bias=False)
        self.up = nn.Linear(rank, gate_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return F.linear(code, self.highway) + self.up(self.down(code))


class GeometryAdaptiveSurrogate(nn.Module):
    """NRGD-conditioned, additive preactivation modulation at every block.

    The projector uses standardized input coordinates. Physical Jacobians and
    geometry targets are needed during geometry construction/training only.
    """

    def __init__(self, input_dim: int, output_dim: int, projector,
                 activation: str = "relu", *, hidden_dim: int = 128,
                 ff_width: int = 256, layers: int = 4, code_dim: int = 64,
                 controller_dim: int = 32, projection_rank: int = 8,
                 adapter_scale: float = 0.1, beta_temperature: float = 1.0,
                 highway_seed: int = 0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation: {activation}")
        if layers < 1 or projection_rank < 1 or beta_temperature <= 0 or adapter_scale < 0:
            raise ValueError("invalid layer count, projection rank, temperature or scale")
        matrix = torch.as_tensor(projector, dtype=torch.float64)
        if matrix.shape != (input_dim, input_dim):
            raise ValueError("projector must be a square input-dimensional matrix")
        if not torch.allclose(matrix, matrix.T, atol=1e-7, rtol=0) or not torch.allclose(
            matrix @ matrix, matrix, atol=1e-7, rtol=0
        ):
            raise ValueError("projector must be symmetric and idempotent")
        rank = round(float(torch.trace(matrix)))
        if not 1 <= rank < input_dim:
            raise ValueError("projector must have nontrivial rank")
        self.register_buffer("projector", matrix.float())
        self.input_dim = input_dim
        self.activation = activation
        self.adapter_scale = float(adapter_scale)
        self.beta_temperature = float(beta_temperature)
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, ff_width, activation) for _ in range(layers)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dominant_encoder = nn.Sequential(nn.Linear(input_dim, code_dim), nn.SiLU())
        self.complement_encoder = nn.Sequential(nn.Linear(input_dim, code_dim), nn.SiLU())
        self.controller = nn.Sequential(
            nn.Linear(input_dim + 2 * code_dim, controller_dim), nn.SiLU()
        )
        self.rho_head = nn.Linear(controller_dim, 1)
        self.pi_head = nn.Linear(controller_dim, 2)
        self.beta_head = nn.Linear(controller_dim, layers)
        generator = torch.Generator().manual_seed(highway_seed)
        self.layer_geometry = nn.ModuleList([
            nn.ModuleDict({
                "dominant": BranchProjection(code_dim, ff_width, projection_rank, generator),
                "complement": BranchProjection(code_dim, ff_width, projection_rank, generator),
            }) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, *, return_aux: bool = False,
                return_diagnostics: bool = False):
        projector = self.projector.to(dtype=x.dtype, device=x.device)
        dominant_input = x @ projector.T
        complement_input = x - dominant_input
        dominant = self.dominant_encoder(dominant_input)
        complement = self.complement_encoder(complement_input)
        shared = self.controller(torch.cat((x, dominant, complement), dim=-1))
        rho = torch.sigmoid(self.rho_head(shared)).squeeze(-1)
        pi = torch.softmax(self.pi_head(shared), dim=-1)
        beta = torch.softmax(self.beta_head(shared) / self.beta_temperature, dim=-1)

        hidden = self.stem(x)
        diagnostics = []
        for layer, block in enumerate(self.blocks):
            gate, value = block.base_terms(hidden)
            branches = self.layer_geometry[layer]
            direction = (
                torch.sqrt(pi[:, :1]) * branches["dominant"](dominant)
                + torch.sqrt(pi[:, 1:]) * branches["complement"](complement)
            )
            direction_rms = torch.sqrt(direction.square().mean(dim=-1, keepdim=True))
            gate_rms = torch.sqrt(gate.square().mean(dim=-1, keepdim=True))
            shift = (self.adapter_scale * rho[:, None] *
                     torch.sqrt(beta[:, layer:layer + 1]) *
                     direction / (direction_rms + 1e-12) * gate_rms)
            hidden = block(hidden, gate + shift, value)
            if return_diagnostics:
                diagnostics.append({"base_gate": gate, "shift": shift,
                                    "value": value, "modulated_gate": gate + shift})
        prediction = self.output(hidden)
        if return_aux:
            return prediction, {"rho": rho, "pi": pi, "beta": beta,
                                "layers": diagnostics}
        return prediction


def ga_nar_loss(prediction: torch.Tensor, target: torch.Tensor, aux: dict,
                *, rho_target: torch.Tensor | None = None,
                pi_target: torch.Tensor | None = None,
                lambda_rho: float = 0.0, lambda_pi: float = 0.0) -> torch.Tensor:
    """Prediction MSE plus intensity MSE and squared Hellinger allocation loss."""
    if lambda_rho < 0 or lambda_pi < 0:
        raise ValueError("loss weights must be nonnegative")
    loss = (prediction - target).square().mean()
    if lambda_rho:
        if rho_target is None:
            raise ValueError("intensity target is required")
        loss = loss + lambda_rho * (aux["rho"] - rho_target).square().mean()
    if lambda_pi:
        if pi_target is None:
            raise ValueError("allocation target is required")
        distance = 0.5 * (torch.sqrt(aux["pi"].clamp_min(0)) -
                          torch.sqrt(pi_target.clamp_min(0))).square().sum(dim=-1)
        loss = loss + lambda_pi * distance.mean()
    return loss
