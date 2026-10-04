from __future__ import annotations

import torch
from torch import nn

from .swiglu_mlp import _SwiGLUBlock


class SensitivitySubspaceAdapter(nn.Module):
    def __init__(self, input_dimension: int, width: int, adapter_rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(input_dimension, adapter_rank)
        self.up = nn.Linear(adapter_rank, width)

    def forward(self, projected_input: torch.Tensor) -> torch.Tensor:
        return self.up(torch.nn.functional.silu(self.down(projected_input)))


class ScalarModulationGate(nn.Module):
    def __init__(self, width: int, hidden_width: int = 8) -> None:
        super().__init__()
        self.first = nn.Linear(width, hidden_width)
        self.second = nn.Linear(hidden_width, 1)

    def forward(self, stem_hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.second(torch.nn.functional.silu(self.first(stem_hidden))))


class SensitivityInformedSurrogate(nn.Module):
    """SwiGLU backbone preceded by one frozen subspace projection, adapter, and scalar gate."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        sensitivity_basis: torch.Tensor,
        *,
        width: int = 128,
        adapter_rank: int = 8,
        gate_hidden_width: int = 8,
        num_blocks: int = 4,
        activity_supervision: bool = True,
    ) -> None:
        super().__init__()
        basis = torch.as_tensor(sensitivity_basis, dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] != input_dimension:
            raise ValueError("sensitivity_basis must have shape (input_dimension, rank)")
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.width = width
        self.num_blocks = num_blocks
        self.activity_supervision = activity_supervision
        self.register_buffer("sensitivity_basis", basis)
        self.stem = nn.Linear(input_dimension, width)
        self.adapter = SensitivitySubspaceAdapter(input_dimension, width, adapter_rank)
        self.gate = ScalarModulationGate(width, gate_hidden_width)
        self.blocks = nn.ModuleList([_SwiGLUBlock(width) for _ in range(num_blocks)])
        self.output = nn.Linear(width, output_dimension)

    def forward(self, standardized_input: torch.Tensor, *, return_activations: bool = False):
        stem_hidden = self.stem(standardized_input)
        projected_coordinates = standardized_input @ self.sensitivity_basis
        projected_input = projected_coordinates @ self.sensitivity_basis.T
        adapter_hidden = self.adapter(projected_input)
        modulation_coefficient = self.gate(stem_hidden)
        hidden = stem_hidden + modulation_coefficient * adapter_hidden
        activations = {}
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden)
            activations[f"block{index}_output"] = hidden
        prediction = self.output(hidden)
        return (prediction, activations) if return_activations else prediction
