from __future__ import annotations

import numpy as np
import pytest
import torch

from ga_nar.data import Benchmark, SYSTEMS, load_benchmark
from ga_nar.geometry import build_response_geometry
from ga_nar.model import GeometryAdaptiveSurrogate, ga_nar_loss
from ga_nar.training import fit, predict_batches


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "mish", "softplus", "swiglu"])
def test_modulation_is_layerwise_and_zero_scale_recovers_backbone(activation):
    torch.manual_seed(4)
    model = GeometryAdaptiveSurrogate(
        4, 3, np.diag([1.0, 1.0, 0.0, 0.0]), activation,
        hidden_dim=12, ff_width=16, layers=3, code_dim=6,
        controller_dim=8, projection_rank=2,
    )
    x = torch.randn(5, 4)
    with torch.no_grad():
        modulated, aux = model(x, return_aux=True, return_diagnostics=True)
        model.adapter_scale = 0.0
        baseline, base_aux = model(x, return_aux=True, return_diagnostics=True)
    assert len(aux["layers"]) == 3
    assert torch.allclose(aux["beta"].sum(dim=-1), torch.ones(5))
    assert torch.allclose(aux["pi"].sum(dim=-1), torch.ones(5))
    assert all(torch.count_nonzero(layer["shift"]) > 0 for layer in aux["layers"])
    assert all(torch.count_nonzero(layer["shift"]) == 0 for layer in base_aux["layers"])
    assert not torch.allclose(modulated, baseline)
    if activation == "swiglu":
        assert torch.allclose(aux["layers"][0]["value"],
                              base_aux["layers"][0]["value"])


def test_loss_uses_squared_hellinger_allocation():
    prediction = torch.zeros(2, 1)
    target = torch.zeros_like(prediction)
    aux = {"rho": torch.tensor([0.5, 0.5]),
           "pi": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    loss = ga_nar_loss(prediction, target, aux,
                       rho_target=torch.tensor([0.5, 0.5]),
                       pi_target=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                       lambda_rho=1.0, lambda_pi=2.0)
    assert loss.item() == pytest.approx(2.0)


def test_nrgd_recovers_quadratic_response_direction():
    centers = np.linspace(-1, 1, 12)[:, None] * np.ones((1, 3))

    def jacobian_action(x, vector):
        return np.asarray([4.0 * x[0] * vector[0]])

    geometry = build_response_geometry(
        centers, jacobian_action, directions_per_center=48,
        hutchinson_probes=8, subspace_probes=8, seed=2,
    )
    assert geometry.rank == 1
    assert geometry.projector[0, 0] > 0.8
    assert np.all(np.abs(geometry.intensity - 0.5) < 0.06)
    assert np.allclose(geometry.allocation.sum(axis=-1), 1.0)
    assert np.all(geometry.allocation[:, 0] > 0.8)


def test_training_and_prediction_use_geometry_centers():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = (x[:, :2] ** 2).astype(np.float32)
    geometry_x = x[:12].copy()
    data = Benchmark("synthetic", x, y, np.arange(20), np.arange(20, 26),
                     np.arange(26, 32), geometry_x,
                     np.full(12, 0.5, np.float32),
                     np.tile([0.6, 0.4], (12, 1)).astype(np.float32),
                     np.diag([1.0, 1.0, 0.0, 0.0]),
                     np.zeros(4), np.ones(4), np.zeros(2), np.ones(2), {})
    model = GeometryAdaptiveSurrogate(4, 2, data.projector, "relu",
                                      hidden_dim=12, ff_width=16, layers=2,
                                      code_dim=6, controller_dim=8,
                                      projection_rank=2)
    result = fit(model, data, epochs=2, patience=2, batch_size=8)
    assert result["epochs_completed"] == 2
    assert np.isfinite(result["validation_mse"])
    assert predict_batches(model, data.x[data.test]).shape == (6, 2)


@pytest.mark.parametrize("system", SYSTEMS)
def test_packaged_geometry_matches_benchmark(system):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = load_benchmark(root, system)
    assert data.projector.shape == (data.x.shape[1], data.x.shape[1])
    assert data.geometry_x.shape[0] == len(data.rho) == len(data.pi)
    assert not set(data.train) & set(data.test)
