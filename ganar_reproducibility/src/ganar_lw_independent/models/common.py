from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


def rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class PreNormSwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_width: int, activation: str = "silu"):
        super().__init__()
        if activation not in {"silu", "relu", "gelu", "mish"}:
            raise ValueError(activation)
        self.activation_name = activation
        self.norm = RMSNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, ff_width)
        self.value = nn.Linear(hidden_dim, ff_width)
        self.output = nn.Linear(ff_width, hidden_dim)

    def base_terms(self, h: torch.Tensor):
        hn = self.norm(h)
        return self.gate(hn), self.value(hn)

    def update(self, h: torch.Tensor, gate: torch.Tensor, value: torch.Tensor):
        if self.activation_name == "relu":
            activated = torch.relu(gate)
        elif self.activation_name == "gelu":
            activated = torch.nn.functional.gelu(gate)
        elif self.activation_name == "mish":
            activated = torch.nn.functional.mish(gate)
        else:
            activated = torch.nn.functional.silu(gate)
        return h + self.output(activated * value)

    def forward(self, h: torch.Tensor):
        gate, value = self.base_terms(h)
        return self.update(h, gate, value)


class PreNormActivationBlock(nn.Module):
    """Residual MLP block used by the final Layer-wise activation backbone."""

    def __init__(self, hidden_dim: int, ff_width: int, activation: str = "mish"):
        super().__init__()
        if activation not in {"relu", "gelu", "silu", "mish", "tanh", "softplus"}:
            raise ValueError(activation)
        self.activation_name = activation
        self.norm = RMSNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, ff_width)
        self.output = nn.Linear(ff_width, hidden_dim)

    def base_terms(self, h: torch.Tensor):
        return self.gate(self.norm(h)), None

    def update(self, h: torch.Tensor, gate: torch.Tensor, value=None):
        if self.activation_name == "relu":
            activated = torch.relu(gate)
        elif self.activation_name == "gelu":
            activated = torch.nn.functional.gelu(gate)
        elif self.activation_name == "silu":
            activated = torch.nn.functional.silu(gate)
        elif self.activation_name == "softplus":
            activated = torch.nn.functional.softplus(gate)
        elif self.activation_name == "tanh":
            activated = torch.tanh(gate)
        else:
            activated = torch.nn.functional.mish(gate)
        return h + self.output(activated)

    def forward(self, h: torch.Tensor):
        gate, value = self.base_terms(h)
        return self.update(h, gate, value)
