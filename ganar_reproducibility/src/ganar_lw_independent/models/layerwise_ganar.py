from __future__ import annotations

import math
import torch
from torch import nn

from .common import RMSNorm, PreNormActivationBlock, PreNormSwiGLUBlock, rms
from .baselines import MatchedActivation, MatchedSwiGLU


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, geometry_rank: int):
        super().__init__()
        self.dominant = nn.Sequential(nn.Linear(input_dim, geometry_rank), nn.SiLU(),
                                      RMSNorm(geometry_rank))
        self.complement = nn.Sequential(nn.Linear(input_dim, geometry_rank), nn.SiLU(),
                                        RMSNorm(geometry_rank))
        self.shared = nn.Sequential(nn.Linear(2 * geometry_rank, 2 * geometry_rank),
                                    nn.SiLU(), RMSNorm(2 * geometry_rank))

    def forward(self, xr, xc):
        er = self.dominant(xr)
        ec = self.complement(xc)
        return er, ec, self.shared(torch.cat((er, ec), dim=-1))


class LayerwiseGANAR(nn.Module):
    MODES = {"layerwise", "uniform_beta", "no_geometry", "prebackbone"}

    def __init__(self, input_dim: int, output_dim: int, projector,
                 hidden_dim: int = 192, ff_width: int = 384,
                 blocks: int = 4, geometry_rank: int = 96,
                 delta_width: int = 384,
                 activation: str = "mish",
                 block_kind: str = "activation",
                 mode: str = "layerwise",
                 gamma_init: float = -8.0,
                 anchor_features: bool = True):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.ff_width = ff_width
        self.n_blocks = blocks
        self.geometry_rank = geometry_rank
        self.delta_width = delta_width
        self.activation = activation
        self.block_kind = block_kind
        self.mode = mode
        self.gamma_init = float(gamma_init)
        self.anchor_features = bool(anchor_features)
        self.register_buffer("projector", torch.as_tensor(projector, dtype=torch.float32))
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.geometry = GeometryEncoder(input_dim, geometry_rank)
        shared_dim = 2 * geometry_rank
        self.rho_head = nn.Linear(shared_dim, 1)
        self.pi_head = nn.Linear(shared_dim, 2)
        self.beta_head = nn.Linear(shared_dim, blocks)
        self.gamma_head = nn.Linear(shared_dim, 1)
        block_cls = PreNormActivationBlock if block_kind == "activation" else PreNormSwiGLUBlock
        self.blocks = nn.ModuleList([block_cls(hidden_dim, ff_width, activation)
                                     for _ in range(blocks)])
        self.layer_geometry = nn.ModuleList([
            nn.Linear(shared_dim, ff_width, bias=False) for _ in range(blocks)
        ])
        self.alpha = nn.Parameter(torch.zeros(blocks))
        # A small, learnable geometry skip preserves the useful global
        # geometry path while the layer-wise gates learn depth allocation.
        self.pre_alpha = nn.Parameter(torch.tensor(0.1))
        self.prebackbone = nn.Linear(shared_dim, hidden_dim, bias=False)
        self.base_output = nn.Linear(hidden_dim, output_dim)
        # Strong conventional predictor highway.  The Layer-wise branch is a
        # geometry-conditioned residual correction on top of this anchor.
        if block_kind == "swiglu":
            self.anchor = MatchedSwiGLU(input_dim, output_dim,
                                        hidden_dim=128, ff_width=256, blocks=4)
        else:
            self.anchor = MatchedActivation(input_dim, output_dim, activation,
                                            hidden_dim=128, ff_width=256, blocks=4)
        # Avoid perturbing the legacy random stream when the adapter uses its
        # original independent stem.  The projection is only needed for the
        # anchor-feature variant.
        self.anchor_projection = (nn.Linear(128, hidden_dim)
                                  if self.anchor_features else nn.Identity())
        self.input_skip = nn.Linear(input_dim, output_dim)
        self.delta_output = nn.Sequential(
            nn.Linear(hidden_dim + shared_dim, delta_width), nn.SiLU(),
            nn.Linear(delta_width, output_dim)
        )
        self._last = None
        self.reset_protocol_parameters()

    def reset_protocol_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.rho_head.weight); nn.init.zeros_(self.rho_head.bias)
        nn.init.zeros_(self.pi_head.weight); nn.init.zeros_(self.pi_head.bias)
        nn.init.zeros_(self.beta_head.weight); nn.init.zeros_(self.beta_head.bias)
        # Near-zero highway, but not so tiny that the geometry residual is
        # effectively frozen for the entire optimization horizon.
        nn.init.zeros_(self.gamma_head.weight); nn.init.constant_(self.gamma_head.bias, self.gamma_init)
        nn.init.zeros_(self.alpha)
        with torch.no_grad():
            self.pre_alpha.fill_(0.0)
        # The highway gate gamma is near zero, but the delta branch itself
        # must remain Xavier-initialized so that gamma receives a learning
        # signal and the geometry residual can grow during training.
        for module in self.delta_output.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.base_output.weight)
        nn.init.zeros_(self.base_output.bias)
        nn.init.zeros_(self.input_skip.weight)
        nn.init.zeros_(self.input_skip.bias)

    def _geometry(self, x):
        p = self.projector.to(x)
        xr = x @ p.T
        xc = x @ (torch.eye(self.input_dim, dtype=x.dtype, device=x.device) - p).T
        er, ec, eg = self.geometry(xr, xc)
        if self.mode == "no_geometry":
            er = torch.zeros_like(er); ec = torch.zeros_like(ec); eg = torch.zeros_like(eg)
        rho = torch.sigmoid(self.rho_head(eg)).squeeze(-1)
        pi = torch.softmax(self.pi_head(eg), dim=-1)
        beta = torch.softmax(self.beta_head(eg), dim=-1)
        if self.mode == "uniform_beta":
            beta = torch.full_like(beta, 1.0 / self.n_blocks)
        gamma_raw = self.gamma_head(eg)
        gamma = (torch.tanh(gamma_raw) if getattr(self, "anchor_gate", False)
                 else torch.sigmoid(gamma_raw)).squeeze(-1)
        return er, ec, eg, rho, pi, beta, gamma

    def forward(self, x, return_aux: bool = False, return_diagnostics: bool = False):
        er, ec, eg, rho, pi, beta, gamma = self._geometry(x)
        if getattr(self, "anchor_only", False):
            # The correction should be conditioned on the representation that
            # produced the warm-start prediction.  Using the separate random
            # layer-wise stem here wastes the anchor and makes the residual
            # branch unnecessarily hard to optimize.
            h_anchor = (self.anchor_projection(self.anchor.stem(x))
                        if self.anchor_features else self.stem(x))
            delta = self.delta_output(torch.cat((h_anchor, eg), dim=-1))
            prediction = self.anchor(x) + self.input_skip(x) + gamma[:, None] * delta
            self._last = {"rho": rho, "pi": pi, "beta": beta, "gamma": gamma,
                          "alpha": torch.tanh(self.alpha), "layers": [],
                          "base": self.anchor(x) + self.input_skip(x), "delta": delta}
            return (prediction, rho, pi, beta, gamma) if return_aux else prediction
        h = self.stem(x)
        if self.mode == "prebackbone":
            h = h + rho[:, None] * self.prebackbone(eg) / (rms(self.prebackbone(eg)) + 1e-6)
        elif self.mode == "layerwise":
            global_geometry = self.prebackbone(eg)
            h = h + self.pre_alpha * rho[:, None] * global_geometry / (rms(global_geometry) + 1e-6)
        diagnostics = []
        for layer, (block, projection) in enumerate(zip(self.blocks, self.layer_geometry)):
            gate, value = block.base_terms(h)
            if self.mode in {"layerwise", "uniform_beta"}:
                # pi remains physically interpretable by weighting disjoint
                # halves before the layer-specific projection.
                weighted = torch.cat((torch.sqrt(pi[:, :1]) * er,
                                      torch.sqrt(pi[:, 1:]) * ec), dim=-1)
                raw = projection(weighted)
                normalized = raw / (rms(raw) + 1e-6) * rms(gate)
                modulation = (torch.tanh(self.alpha[layer]) * rho *
                              torch.sqrt(beta[:, layer]))[:, None] * normalized
                modulated_gate = gate + modulation
            else:
                modulation = torch.zeros_like(gate)
                modulated_gate = gate
            h = block.update(h, modulated_gate, value)
            if return_diagnostics:
                diagnostics.append({"base_gate": gate, "modulation": modulation,
                                    "modulated_gate": modulated_gate})
        base = self.base_output(h) + self.anchor(x)
        delta = self.delta_output(torch.cat((h, eg), dim=-1))
        prediction = base + gamma[:, None] * delta
        self._last = {"rho": rho, "pi": pi, "beta": beta, "gamma": gamma,
                      "alpha": torch.tanh(self.alpha), "layers": diagnostics,
                      "base": base, "delta": delta}
        if return_aux:
            return prediction, rho, pi, beta, gamma
        return prediction


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
