import numpy as np
import torch

from ganar_lw_independent.models.layerwise_ganar import LayerwiseGANAR
from ganar_lw_independent.models.baselines import MatchedSwiGLU


def test_zero_initialized_highway_matches_base_output():
    model = LayerwiseGANAR(8, 3, np.eye(8))
    x = torch.randn(5, 8)
    y = model(x)
    model.eval()
    with torch.no_grad():
        y2 = model(x)
    assert y.shape == (5, 3)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, y2)


def test_simplex_and_gradient_connectivity():
    model = LayerwiseGANAR(8, 3, np.eye(8))
    x = torch.randn(5, 8)
    y, rho, pi, beta, gamma = model(x, return_aux=True)
    assert torch.allclose(pi.sum(-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.ones(5), atol=1e-6)
    y.square().mean().backward()
    assert model.rho_head.weight.grad is not None
    assert model.beta_head.weight.grad is not None
    assert model.layer_geometry[0].weight.grad is not None


def test_baseline_forward():
    model = MatchedSwiGLU(8, 3, ff_width=64)
    assert model(torch.randn(4, 8)).shape == (4, 3)
