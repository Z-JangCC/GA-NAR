"""Neural baselines and the geometry-adaptive nonlinear representation model.

The module deliberately keeps every architecture in one place so the experiment
runner can enforce matched parameter budgets and record the exact model used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _seeded_module(seed: int, factory: Callable[[], nn.Module]) -> nn.Module:
    """Construct one module without consuming another model's RNG stream."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


def _activation(name: str) -> nn.Module:
    activations: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown activation {name!r}") from exc


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class NonlinearityModule(nn.Module):
    """Pluggable nonlinear representation interface.

    ``context`` is the original normalized input and is optional for fixed
    activations.  GANR uses it to allocate nonlinear response spatially.
    """

    def forward(self, h: Tensor, context: Tensor | None = None,
                return_aux: bool = False):
        raise NotImplementedError


class ActivationModule(NonlinearityModule):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        self.activation = _activation(self.activation_name)

    def forward(self, h: Tensor, context: Tensor | None = None,
                return_aux: bool = False):
        output = self.activation(h)
        return (output, {}) if return_aux else output


class SwiGLUModule(NonlinearityModule):
    def __init__(self, width: int, expansion: float = 4.0 / 3.0) -> None:
        super().__init__()
        # Do not quantize the inner width to multiples of eight here: the
        # modular benchmark needs a smooth integer parameter search so that
        # fixed activations and gated modules can be matched fairly at small
        # smoke widths as well as at publication widths.
        inner = max(4, int(round(width * expansion)))
        self.value = nn.Linear(width, inner)
        self.gate = nn.Linear(width, inner)
        self.project = nn.Linear(inner, width)

    def forward(self, h: Tensor, context: Tensor | None = None,
                return_aux: bool = False):
        output = self.project(self.value(h) * F.silu(self.gate(h)))
        return (output, {}) if return_aux else output


class GANRModule(NonlinearityModule):
    """Geometry-Adaptive Nonlinear Representation as a drop-in module."""

    def __init__(self, width: int, input_dim: int, *,
                 router_enabled: bool = True,
                 multiplicative_gate: bool = True,
                 gamma_base: float = 0.75,
                 gamma_scale: float = 2.25,
                 amplitude_base: float = 0.25,
                 amplitude_scale: float = 0.75,
                 smooth_mix_base: float = 0.0,
                 smooth_mix_scale: float = 1.0,
                 identity_mix_base: float = 0.0,
                 identity_mix_scale: float = 0.0,
                 adaptive_residual_base: float = 0.0,
                 adaptive_residual_scale: float = 0.0,
                 identity_rank: int = 0,
                 expansion: float = 4.0 / 3.0) -> None:
        super().__init__()
        inner = max(8, int(round(width * expansion / 8.0)) * 8)
        self.router_enabled = router_enabled
        self.multiplicative_gate = multiplicative_gate
        self.gamma_base = float(gamma_base)
        self.gamma_scale = float(gamma_scale)
        self.amplitude_base = float(amplitude_base)
        self.amplitude_scale = float(amplitude_scale)
        if not 0.0 <= smooth_mix_base <= 1.0:
            raise ValueError("smooth_mix_base must lie in [0, 1]")
        if not 0.0 <= smooth_mix_scale <= 1.0:
            raise ValueError("smooth_mix_scale must lie in [0, 1]")
        self.smooth_mix_base = float(smooth_mix_base)
        self.smooth_mix_scale = float(smooth_mix_scale)
        if not 0.0 <= identity_mix_base <= 1.0:
            raise ValueError("identity_mix_base must lie in [0, 1]")
        if not 0.0 <= identity_mix_scale <= 1.0:
            raise ValueError("identity_mix_scale must lie in [0, 1]")
        if adaptive_residual_base < 0.0 or adaptive_residual_scale < 0.0:
            raise ValueError("adaptive residual coefficients must be non-negative")
        self.identity_mix_base = float(identity_mix_base)
        self.identity_mix_scale = float(identity_mix_scale)
        self.adaptive_residual_base = float(adaptive_residual_base)
        self.adaptive_residual_scale = float(adaptive_residual_scale)
        self.identity_rank = int(identity_rank)
        self.constant_allocation = 0.5
        if self.identity_rank < 0:
            raise ValueError("identity_rank must be non-negative")
        if self.identity_rank:
            # Low-rank correction for globally smooth systems.  It keeps the
            # SiLU identity path at full width while adding only a small,
            # geometry-trainable residual instead of a full gated expansion.
            self.identity_down = nn.Linear(width, self.identity_rank)
            self.identity_up = nn.Linear(self.identity_rank, width)
            # The identity-residual operating point is intentionally
            # parameter-light; routing is unnecessary when the correction is
            # initialized as a small global residual.
            self.router = None
            self.value = None
            self.gate = None
            self.project = None
            return
        self.router = NonlinearityRouter(input_dim, width) if router_enabled else None
        self.value = nn.Linear(width, inner)
        self.gate = nn.Linear(width, inner)
        self.project = nn.Linear(inner, width)

    def forward(self, h: Tensor, context: Tensor | None = None,
                return_aux: bool = False):
        if self.identity_rank:
            if self.router is None or context is None:
                allocation = h.new_full((h.shape[0], 1), self.constant_allocation)
            else:
                allocation = self.router(context)
            correction = self.identity_up(F.silu(self.identity_down(h)))
            residual_strength = (
                self.adaptive_residual_base
                + self.adaptive_residual_scale * allocation
            )
            output = F.silu(h) + residual_strength * correction
            aux = {
                "allocation": allocation,
                "identity_mix": h.new_ones((h.shape[0], 1)),
                "residual_strength": residual_strength,
            }
            return (output, aux) if return_aux else output
        if self.router is None or context is None:
            allocation = h.new_full((h.shape[0], 1), self.constant_allocation)
        else:
            allocation = self.router(context)
        gamma = self.gamma_base + self.gamma_scale * allocation
        gate = F.silu(gamma * self.gate(h))
        value = self.value(h)
        if self.multiplicative_gate:
            adaptive = value * gate
        else:
            adaptive = value + gate
        # A geometry router should be able to reduce, rather than only
        # increase, nonlinear capacity.  The smooth branch is parameter-free
        # (it reuses ``value`` and ``project``), so this extension preserves
        # the matched parameter budget while making the method robust on
        # systems whose solution map is globally smooth.  Allocation still
        # determines the mixture, hence this is a continuous GANR operating
        # point rather than a system-specific architecture switch.
        smooth_mix = torch.clamp(
            self.smooth_mix_base + self.smooth_mix_scale * (1.0 - allocation),
            0.0,
            1.0,
        )
        adaptive = smooth_mix * F.silu(value) + (1.0 - smooth_mix) * adaptive
        # Prediction-preserving path: when identity_mix=1 and amplitude=1,
        # GANR is exactly the same SiLU response as the fixed baseline.  The
        # learned branch is then introduced as a small, input-conditioned
        # residual, so geometry supervision cannot destroy point prediction.
        identity_mix = torch.clamp(
            self.identity_mix_base + self.identity_mix_scale * (1.0 - allocation),
            0.0,
            1.0,
        )
        adaptive_output = self.project(adaptive)
        nonlinear = identity_mix * F.silu(h) + (1.0 - identity_mix) * adaptive_output
        residual_strength = (
            self.adaptive_residual_base
            + self.adaptive_residual_scale * allocation
        )
        nonlinear = nonlinear + residual_strength * adaptive_output
        output = nonlinear * (self.amplitude_base + self.amplitude_scale * allocation)
        aux = {
            "allocation": allocation,
            "smooth_mix": smooth_mix,
            "identity_mix": identity_mix,
            "residual_strength": residual_strength,
        }
        return (output, aux) if return_aux else output


class ModularResidualBlock(nn.Module):
    def __init__(self, width: int, input_dim: int, module: NonlinearityModule,
                 *, init_seed: int | None = None) -> None:
        super().__init__()
        if init_seed is None:
            self.norm = nn.LayerNorm(width)
            self.project = nn.Linear(width, width)
        else:
            self.norm = _seeded_module(init_seed, lambda: nn.LayerNorm(width))
            self.project = _seeded_module(init_seed + 1, lambda: nn.Linear(width, width))
        self.nonlinearity = module
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: Tensor, context: Tensor, return_aux: bool = False):
        transformed, aux = self.nonlinearity(self.norm(h), context, return_aux=True)
        output = h + self.scale * self.project(transformed)
        return (output, aux) if return_aux else output


class ModularResMLP(nn.Module):
    """Common residual backbone whose only varying component is the module."""

    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int,
                 nonlinearity: str = "gelu", *, router_enabled: bool = True,
                 multiplicative_gate: bool = True,
                 ganr_kwargs: dict[str, float] | None = None,
                 physics_base: Tensor | None = None,
                 physics_sensitivity: Tensor | None = None) -> None:
        super().__init__()
        self.input = _seeded_module(11001, lambda: nn.Linear(input_dim, width))
        blocks = []
        for block_index in range(depth):
            module_seed = 12000 + 10 * block_index
            if nonlinearity in {"relu", "gelu", "silu"}:
                module = _seeded_module(module_seed, lambda: ActivationModule(nonlinearity))
            elif nonlinearity == "swiglu":
                module = _seeded_module(module_seed, lambda: SwiGLUModule(width))
            elif nonlinearity in {"ganr", "ours"}:
                module = _seeded_module(
                    module_seed,
                    lambda: GANRModule(
                        width, input_dim, router_enabled=router_enabled,
                        multiplicative_gate=multiplicative_gate,
                        **(ganr_kwargs or {}),
                    ),
                )
            else:
                raise ValueError(f"Unknown pluggable nonlinearity {nonlinearity!r}")
            blocks.append(
                ModularResidualBlock(
                    width, input_dim, module, init_seed=13000 + 10 * block_index
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output_norm = _seeded_module(14001, lambda: nn.LayerNorm(width))
        self.output = _seeded_module(14002, lambda: nn.Linear(width, output_dim))
        self.physics_skip = physics_base is not None or physics_sensitivity is not None
        if self.physics_skip != (physics_base is not None and physics_sensitivity is not None):
            raise ValueError("physics_base and physics_sensitivity must be supplied together")
        if self.physics_skip:
            self.register_buffer("physics_base_raw", physics_base.detach().float().reshape(1, -1).clone())
            self.register_buffer("physics_sensitivity_raw", physics_sensitivity.detach().float().clone())
            self.register_buffer("physics_base", physics_base.detach().float().reshape(1, -1).clone())
            self.register_buffer("physics_sensitivity", physics_sensitivity.detach().float().clone())
        else:
            self.register_buffer("physics_base_raw", torch.empty(0))
            self.register_buffer("physics_sensitivity_raw", torch.empty(0))
            self.register_buffer("physics_base", torch.empty(0))
            self.register_buffer("physics_sensitivity", torch.empty(0))

    def forward(self, x: Tensor, return_aux: bool = False):
        h = self.input(x)
        allocations = []
        for block in self.blocks:
            h, aux = block(h, x, return_aux=True)
            if "allocation" in aux:
                allocations.append(aux["allocation"])
        output = self.output(self.output_norm(h))
        if self.physics_skip:
            output = self.physics_base + x @ self.physics_sensitivity.T + output
        aux = {"allocation": allocations[-1]} if allocations else {}
        return (output, aux) if return_aux else output

    def set_physics_standardizer(self, mean: Tensor, scale: Tensor) -> None:
        if not self.physics_skip:
            return
        mean = mean.reshape(1, -1).to(self.physics_base_raw)
        scale = scale.reshape(1, -1).to(self.physics_base_raw)
        self.physics_base.copy_((self.physics_base_raw - mean) / scale)
        self.physics_sensitivity.copy_(self.physics_sensitivity_raw / scale.T)


class PlainMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one hidden layer")
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(in_features, width), _activation(activation)))
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, return_aux: bool = False):
        output = self.net(x)
        if return_aux:
            return output, {}
        return output


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str = "gelu") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.activation = _activation(activation)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        update = self.fc2(self.activation(self.fc1(self.norm(x))))
        return x + self.scale * update


class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, activation=activation) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: Tensor, return_aux: bool = False):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        output = self.output(self.output_norm(h))
        if return_aux:
            return output, {}
        return output


class SwiGLUBlock(nn.Module):
    def __init__(self, width: int, expansion: float = 4.0 / 3.0) -> None:
        super().__init__()
        inner = max(8, int(round(width * expansion / 8.0)) * 8)
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, inner)
        self.gate = nn.Linear(width, inner)
        self.project = nn.Linear(inner, width)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        update = self.project(self.value(h) * F.silu(self.gate(h)))
        return x + self.scale * update


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(SwiGLUBlock(width) for _ in range(depth))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: Tensor, return_aux: bool = False):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        output = self.output(self.output_norm(h))
        if return_aux:
            return output, {}
        return output


class FourierFeatureMLP(nn.Module):
    """A strong smooth baseline with fixed multi-scale Fourier features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 4,
        feature_dim: int | None = None,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        feature_dim = feature_dim or max(16, width // 2)
        generator = torch.Generator().manual_seed(seed)
        frequencies = torch.randn(input_dim, feature_dim, generator=generator)
        scales = torch.logspace(-0.5, 0.7, feature_dim)
        frequencies = frequencies * scales.unsqueeze(0)
        self.register_buffer("frequencies", frequencies)
        self.backbone = ResMLP(
            input_dim + 2 * feature_dim,
            output_dim,
            width=width,
            depth=depth,
            activation="silu",
        )

    def forward(self, x: Tensor, return_aux: bool = False):
        phase = math.pi * (x @ self.frequencies)
        features = torch.cat((x, torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.backbone(features, return_aux=return_aux)


class NonlinearityRouter(nn.Module):
    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        router_width = max(16, width // 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, router_width),
            nn.SiLU(),
            nn.Linear(router_width, router_width),
            nn.SiLU(),
            nn.Linear(router_width, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class GeometryAdaptiveBlock(nn.Module):
    def __init__(
        self,
        width: int,
        expansion: float = 4.0 / 3.0,
        multiplicative_gate: bool = True,
    ) -> None:
        super().__init__()
        inner = max(8, int(round(width * expansion / 8.0)) * 8)
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, inner)
        self.gate = nn.Linear(width, inner)
        self.project = nn.Linear(inner, width)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.multiplicative_gate = multiplicative_gate

    def forward(self, x: Tensor, allocation: Tensor) -> Tensor:
        h = self.norm(x)
        # gamma changes local bending, while allocation changes how much of the
        # nonlinear branch is spent at this input. Both are bounded for stability.
        gamma = 0.75 + 2.25 * allocation
        gate = F.silu(gamma * self.gate(h))
        if self.multiplicative_gate:
            nonlinear = self.value(h) * gate
        else:
            # Additive control keeps both learned projections and therefore the
            # same parameter count, isolating the multiplicative primitive.
            nonlinear = self.value(h) + gate
        amplitude = 0.25 + 0.75 * allocation
        update = self.project(nonlinear) * amplitude
        return x + self.scale * update


class GeometryAdaptiveNet(nn.Module):
    """Geometry-Adaptive Nonlinear Representation (GANR).

    ``router_enabled``, ``multiplicative_gate`` and the trainer's geometry-loss
    weight expose the three paper ablations without maintaining separate models.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        depth: int = 4,
        router_enabled: bool = True,
        multiplicative_gate: bool = True,
        physics_base: Tensor | None = None,
        physics_sensitivity: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.router_enabled = router_enabled
        self.physics_skip = physics_base is not None or physics_sensitivity is not None
        if self.physics_skip != (physics_base is not None and physics_sensitivity is not None):
            raise ValueError("physics_base and physics_sensitivity must be supplied together")
        if self.physics_skip:
            self.register_buffer("physics_base_raw", physics_base.detach().float().reshape(1, -1).clone())
            self.register_buffer("physics_sensitivity_raw", physics_sensitivity.detach().float().clone())
            self.register_buffer("physics_base", physics_base.detach().float().reshape(1, -1))
            self.register_buffer("physics_sensitivity", physics_sensitivity.detach().float())
        else:
            self.register_buffer("physics_base", torch.empty(0))
            self.register_buffer("physics_sensitivity", torch.empty(0))
            self.register_buffer("physics_base_raw", torch.empty(0))
            self.register_buffer("physics_sensitivity_raw", torch.empty(0))
        self.router = NonlinearityRouter(input_dim, width) if router_enabled else None
        self.register_buffer("constant_allocation", torch.tensor(0.5))
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            GeometryAdaptiveBlock(width, multiplicative_gate=multiplicative_gate)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: Tensor, return_aux: bool = False):
        if self.router is None:
            allocation = self.constant_allocation.expand(x.shape[0], 1)
        else:
            allocation = self.router(x)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, allocation)
        output = self.output(self.output_norm(h))
        if self.physics_skip:
            output = self.physics_base + x @ self.physics_sensitivity.T + output
        if return_aux:
            return output, {"allocation": allocation}
        return output

    def set_physics_standardizer(self, mean: Tensor, scale: Tensor) -> None:
        """Express the physical linearization skip in standardized target units."""
        if not self.physics_skip:
            return
        mean = mean.reshape(1, -1).to(self.physics_base_raw)
        scale = scale.reshape(1, -1).to(self.physics_base_raw)
        self.physics_base.copy_((self.physics_base_raw - mean) / scale)
        self.physics_sensitivity.copy_(self.physics_sensitivity_raw / scale.T)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: int = 128
    depth: int = 4
    router_enabled: bool = True
    multiplicative_gate: bool = True
    ganr_kwargs: tuple[tuple[str, float], ...] = ()
    physics_base: Tensor | None = None
    physics_sensitivity: Tensor | None = None


class PhysicsSkipModel(nn.Module):
    """Wrap any predictor with the same equation-derived residual skip."""

    def __init__(self, base: nn.Module, physics_base: Tensor, physics_sensitivity: Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("physics_base_raw", physics_base.detach().float().reshape(1, -1).clone())
        self.register_buffer("physics_sensitivity_raw", physics_sensitivity.detach().float().clone())
        self.register_buffer("physics_base", physics_base.detach().float().reshape(1, -1).clone())
        self.register_buffer("physics_sensitivity", physics_sensitivity.detach().float().clone())

    def forward(self, x: Tensor, return_aux: bool = False):
        result = self.base(x, return_aux=return_aux)
        if return_aux:
            output, aux = result
            return self.physics_base + x @ self.physics_sensitivity.T + output, aux
        return self.physics_base + x @ self.physics_sensitivity.T + result

    def set_physics_standardizer(self, mean: Tensor, scale: Tensor) -> None:
        mean = mean.reshape(1, -1).to(self.physics_base_raw)
        scale = scale.reshape(1, -1).to(self.physics_base_raw)
        self.physics_base.copy_((self.physics_base_raw - mean) / scale)
        self.physics_sensitivity.copy_(self.physics_sensitivity_raw / scale.T)


def build_model(spec: ModelSpec, input_dim: int, output_dim: int) -> nn.Module:
    name = spec.name.lower()
    if name in {"relu", "gelu", "silu", "swiglu", "ganr", "ours"}:
        model = ModularResMLP(
            input_dim, output_dim, width=spec.width, depth=spec.depth,
            nonlinearity=name, router_enabled=spec.router_enabled,
            multiplicative_gate=spec.multiplicative_gate,
            ganr_kwargs=dict(spec.ganr_kwargs),
            physics_base=None,
            physics_sensitivity=None,
        )
    elif name == "resmlp":
        model = ResMLP(input_dim, output_dim, width=spec.width, depth=spec.depth)
    elif name in {"fourier", "fourier_resmlp"}:
        model = FourierFeatureMLP(
            input_dim, output_dim, width=spec.width, depth=spec.depth
        )
    else:
        raise ValueError(f"Unknown model {spec.name!r}")
    if spec.physics_base is not None and spec.physics_sensitivity is not None:
        return PhysicsSkipModel(model, spec.physics_base, spec.physics_sensitivity)
    return model


def matched_model_specs(
    names: Iterable[str],
    input_dim: int,
    output_dim: int,
    reference_width: int,
    depth: int,
    tolerance: float = 0.12,
) -> list[ModelSpec]:
    """Choose widths whose parameter counts match a plain reference MLP.

    Exact equality is generally impossible across gated and residual blocks, so
    this performs a deterministic integer search and leaves the runner to record
    both the chosen width and exact count.
    """

    target = count_parameters(
        ModularResMLP(input_dim, output_dim, width=reference_width, depth=depth,
                      nonlinearity="gelu")
    )
    specs: list[ModelSpec] = []
    for name in names:
        if name.lower() in {"relu", "gelu", "silu"}:
            specs.append(ModelSpec(name=name, width=reference_width, depth=depth))
            continue
        candidates = range(max(8, reference_width // 4), reference_width * 2 + 1)
        best_width = min(
            candidates,
            key=lambda candidate: abs(
                count_parameters(
                    build_model(
                        ModelSpec(name=name, width=candidate, depth=depth),
                        input_dim,
                        output_dim,
                    )
                )
                - target
            ),
        )
        spec = ModelSpec(name=name, width=best_width, depth=depth)
        ratio = count_parameters(build_model(spec, input_dim, output_dim)) / target
        if abs(ratio - 1.0) > tolerance:
            raise RuntimeError(
                f"Could not parameter-match {name}: ratio={ratio:.3f}, target={target}"
            )
        specs.append(spec)
    return specs
