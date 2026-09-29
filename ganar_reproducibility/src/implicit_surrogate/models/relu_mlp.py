from __future__ import annotations

import torch
from torch import nn


class ReLUMultilayerPerceptron(nn.Module):
    """Four-block width-128 ReLU baseline."""

    def __init__(self, input_dimension: int, output_dimension: int, width: int = 128, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.width = width
        self.num_blocks = num_blocks
        self.stem = nn.Linear(input_dimension, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(num_blocks)])
        self.output = nn.Linear(width, output_dimension)

    def forward(self, standardized_input: torch.Tensor, *, return_activations: bool = False):
        hidden = self.stem(standardized_input)
        activations = {}
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            activations[f"block{index}_output"] = hidden
        prediction = self.output(hidden)
        return (prediction, activations) if return_activations else prediction

