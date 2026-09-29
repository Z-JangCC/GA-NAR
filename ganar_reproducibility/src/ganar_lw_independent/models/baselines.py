from __future__ import annotations

import torch
from torch import nn

from .common import PreNormSwiGLUBlock, RMSNorm


class MatchedSwiGLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 ff_width: int = 256, blocks: int = 4):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([PreNormSwiGLUBlock(hidden_dim, ff_width)
                                     for _ in range(blocks)])
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_protocol_parameters()

    def reset_protocol_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class MatchedReLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 ff_width: int = 256, blocks: int = 4):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(blocks)])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, ff_width), nn.ReLU(),
                          nn.Linear(ff_width, hidden_dim)) for _ in range(blocks)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.stem(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.output(h)


class MatchedActivation(nn.Module):
    """Standard residual MLP baseline with a named pointwise activation."""

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "elu": nn.ELU,
    }

    def __init__(self, input_dim: int, output_dim: int, activation: str,
                 hidden_dim: int = 128, ff_width: int = 256, blocks: int = 4):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise KeyError(activation)
        self.activation_name = activation
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(blocks)])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, ff_width),
                          self.ACTIVATIONS[activation](),
                          nn.Linear(ff_width, hidden_dim))
            for _ in range(blocks)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.stem(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.output(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_swiglu_width(input_dim: int, output_dim: int, target: int,
                         hidden_dim: int = 128, blocks: int = 4) -> int:
    candidates = range(1, 2049)
    def count(width):
        return count_parameters(MatchedSwiGLU(input_dim, output_dim, hidden_dim,
                                              width, blocks))
    return min(candidates, key=lambda width: (abs(count(width) - target), width))


def matched_relu_width(input_dim: int, output_dim: int, target: int,
                       hidden_dim: int = 128, blocks: int = 4) -> int:
    candidates = range(1, 3073)
    def count(width):
        return count_parameters(MatchedReLU(input_dim, output_dim, hidden_dim,
                                            width, blocks))
    return min(candidates, key=lambda width: (abs(count(width) - target), width))
