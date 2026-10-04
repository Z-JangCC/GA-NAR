import numpy as np
import json
from pathlib import Path
import torch
import pytest

from ganar_lw_independent.data import load_independent_dataset
from ganar_lw_independent.physics_satisfaction import (
    _heat_residual,
    _kuramoto_residual,
    _model,
    _powerflow_residual,
)


def test_exact_heat_fem_residual_matches_registered_states():
    data = load_independent_dataset("nonlinear_heat_fem")
    raw = np.load("data_store/processed/nonlinear_heat_fem/dataset.npz")
    values = []
    for index in data.test[:5]:
        values.append(np.linalg.norm(_heat_residual(raw["z"][index], raw["q"][index])))
    assert max(values) < 1e-8


def test_exact_kuramoto_and_powerflow_residuals_match_registered_states():
    for system, residual in (("kuramoto64", _kuramoto_residual),
                             ("ieee118_acpf", _powerflow_residual)):
        data = load_independent_dataset(system)
        raw = np.load(f"data_store/processed/{system}/dataset.npz")
        values = [np.linalg.norm(residual(raw["z"][i], raw["q"][i]))
                  for i in data.test[:5]]
        assert max(values) < 1e-8


def test_full_ganar_checkpoint_replay_uses_training_forward_graph():
    """Guard against a load-success/forward-mismatch in anchor_features."""
    system, name, seed = "nonlinear_heat_fem", "ganar_softplus", 0
    checkpoint = Path("local_training_results") / system / name / str(seed) / "checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("checkpoint replay requires the optional weights archive")
    data, model = _model(system, name, seed)
    x = torch.as_tensor(data.x[data.test], dtype=torch.float64)
    y = torch.as_tensor(data.y[data.test], dtype=torch.float64)
    with torch.no_grad():
        replayed = float(torch.mean((model(x) - y) ** 2))
    path = Path("local_training_results") / system / name / str(seed) / "metrics.json"
    expected = json.loads(path.read_text())["mse_std"]
    assert abs(replayed - expected) / expected < 1e-10
