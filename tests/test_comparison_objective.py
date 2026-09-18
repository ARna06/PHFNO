import math

import pytest
import torch
from torch import nn

from experiments.comparison import ComparisonConfig, predict_next, train_model
from phfno import FNOBaseline, PHFNO
from phfno.fourier import RealFourierCoordinates


class ModeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.coordinates = RealFourierCoordinates((1, 1, 1), 1)

    def forward(self, field, control):
        return self.weight * field

    def step(self, field, control, dt, method, solver_options=None):
        # Accept the explicit solver settings now shared by training and evaluation.
        assert method == "avf"
        return field + dt.reshape(-1, 1, 1, 1, 1) * self.forward(field, control)


def test_training_uses_h1_gradients_and_keeps_l2_error_history(monkeypatch):
    axis = torch.arange(8, dtype=torch.float64) / 8
    field = torch.sin(2 * math.pi * axis).reshape(1, 1, 8, 1, 1).expand(1, 1, 8, 8, 8)
    transitions = {
        "field": field,
        "noisy_target": 0.7 * field + 0.2,
        "clean_target": 1.5 * field + 0.1,
        "control": torch.empty(1, 0, dtype=torch.float64),
        "dt": torch.tensor([0.5], dtype=torch.float64),
    }
    gradients = []
    clip_grad = torch.nn.utils.clip_grad_norm_

    def capture_gradient(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients.append(parameters[0].grad.item())
        return clip_grad(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_gradient)
    config = ComparisonConfig(seeds=(7,), steps=1, batch_size=1, eval_every=1,
                              learning_rate=0.01, gradient_clip=1e6,
                              device="cpu", progress=False)
    model = ModeModel()
    result = train_model(model, transitions, transitions, config, 7, 2.5, 0.0)
    weight = 1 + 4 * math.pi**2
    expected_gradient = 0.5 * 0.5 * weight / 2.5**2
    assert gradients == pytest.approx([expected_gradient])
    initial, trained = result["history"]
    assert initial["train_h1_loss"] is None
    assert initial["train_nrmse"] is None
    assert trained["train_h1_loss"] == pytest.approx((0.2**2 + 0.5**2 * weight / 2) / 2.5**2)
    assert trained["train_nrmse"] == pytest.approx(math.sqrt(0.2**2 + 0.5**2 / 2) / 2.5)
    assert initial["validation_nrmse"] == pytest.approx(math.sqrt(0.1**2 + 0.3**2 / 2) / 2.5)
    updated_weight = 0.4 - 0.01 * expected_gradient / (expected_gradient + 1e-8)
    validation_difference = 1 + 0.5 * updated_weight - 1.5
    expected_validation = math.sqrt(0.1**2 + validation_difference**2 / 2) / 2.5
    assert trained["validation_nrmse"] == pytest.approx(expected_validation)
    assert result["best_step"] == 0
    assert model.weight.item() == pytest.approx(0.4)


def test_prediction_passes_selected_batch_intervals_to_phfno_step(monkeypatch):
    model = PHFNO((0, 0, 0), 1, control_channels=1, hidden_channels=2,
                  n_layers=1, mlp_width=2, parameter_grid=(4, 4, 4))
    transitions = {
        "field": torch.randn(3, 1, 4, 4, 4),
        "control": torch.tensor([[1.0], [2.0], [3.0]]),
        "dt": torch.tensor([0.1, 0.2, 0.3]),
    }
    calls = []

    def step(field, control, dt, method):
        assert method == "avf"
        calls.append((field, control, dt))
        return field + (dt[:, None] * control).reshape(-1, 1, 1, 1, 1)

    def forward(*args):
        raise AssertionError("PHFNO predictions must use its discrete step")

    monkeypatch.setattr(model, "step", step)
    monkeypatch.setattr(model, "forward", forward)
    indices = torch.tensor([2, 0])
    result = predict_next(model, transitions, indices)
    assert len(calls) == 1
    for received, key in zip(calls[0], ("field", "control", "dt")):
        torch.testing.assert_close(received, transitions[key][indices])
    expected = transitions["field"][indices] + torch.tensor([0.9, 0.1]).reshape(-1, 1, 1, 1, 1)
    torch.testing.assert_close(result, expected)


def test_baseline_prediction_uses_the_comparison_avf_update(monkeypatch):
    model = FNOBaseline((0, 0, 0), 1, control_channels=1, hidden_channels=2, n_layers=1)
    field = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1).expand(2, 1, 4, 4, 4)
    transitions = {"field": field, "control": torch.zeros(2, 1), "dt": torch.tensor([0.1, 0.3])}

    def step(field, control, dt, method):
        assert method == "avf"
        return field + dt.reshape(-1, 1, 1, 1, 1) * (-2 * field)

    monkeypatch.setattr(model, "step", step)
    monkeypatch.setattr(model, "forward", lambda state, control: -2 * state)
    prediction = predict_next(model, transitions, slice(None))
    assert prediction.shape == field.shape
