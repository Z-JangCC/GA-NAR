import torch

from ganar_lw_independent.models.baselines import MatchedActivation


def test_all_registered_activation_baselines_forward():
    for name in ("gelu", "tanh", "silu", "mish", "softplus", "elu"):
        model = MatchedActivation(8, 3, name, ff_width=32)
        assert model(torch.randn(4, 8)).shape == (4, 3)
