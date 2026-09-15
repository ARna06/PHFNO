import math

import pytest
import torch
from torch import nn

from experiments.metrics import evaluate_model


class DecayModel(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, field, control):
        return -self.rate * field

    def rollout(self, initial, controls, times):
        states = [initial]
        for control, dt in zip(controls.unbind(dim=1), times[1:] - times[:-1]):
            states.append(states[-1] + dt * self(states[-1], control))
        return torch.stack(states, dim=1)


class DivergentModel(DecayModel):
    def forward(self, field, control):
        axis = torch.arange(field.shape[-1], dtype=field.dtype, device=field.device)
        mode = torch.sin(2 * math.pi * axis / field.shape[-1])
        result = torch.zeros_like(field)
        result[:, 0] = mode[:, None, None]
        return result


class NonfiniteModel(DecayModel):
    def forward(self, field, control):
        return torch.full_like(field, math.nan)


class ForcedDecayModel(DecayModel):
    def __init__(self, rate, basis):
        super().__init__(rate)
        self.register_buffer("basis", basis)

    def forward(self, field, control):
        return -self.rate * field + control[:, :, None, None, None] * self.basis


def wave_data():
    grid_size = 8
    viscosity = 0.01
    rate = 4 * math.pi**2 * viscosity
    axis = torch.arange(grid_size, dtype=torch.float64) / grid_size
    initial = torch.zeros(2, 3, grid_size, grid_size, grid_size, dtype=torch.float64)
    initial[:, 1] = torch.sin(2 * math.pi * axis)[:, None, None]
    initial[1] *= 2
    times = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    clean = initial[:, None] * torch.exp(-rate * times)[None, :, None, None, None, None]
    return {
        "clean": clean,
        "noisy": clean.clone(),
        "times": times,
        "controls": torch.empty(2, 2, 0, dtype=torch.float64),
        "forcing": torch.zeros_like(clean[:, :-1]),
        "metadata": {"viscosity": viscosity},
    }, rate


def test_exact_rhs_and_physical_power_with_euler_rollout_error():
    data, rate = wave_data()
    model = DecayModel(rate)
    result = evaluate_model(model, data, [1], device="cpu")
    assert model.training
    assert all(value.device.type == "cpu" for value in result.values())
    assert result["prediction"].shape == (1, 3, 3, 8, 8, 8)
    assert result["rhs_relative_error_time"].max() < 1e-13
    assert result["power_residual_time"].abs().max() < 1e-13
    assert result["divergence_rms"].max() < 1e-13
    torch.testing.assert_close(result["true_kinetic_energy"][:, 0], torch.tensor([1.0], dtype=torch.float64))
    expected = torch.abs((1 - rate * data["times"][1]) ** 2 - torch.exp(-rate * data["times"][-1]))
    torch.testing.assert_close(result["nrmse_time"][0, -1], expected)
    assert result["ns_euler_nrmse_time"].min() > 0
    assert result["secant_rhs_relative_error_time"].min() > 0
    assert result["force_nrmse_time"].max() == 0


def test_persistence_metrics_and_initial_normalization():
    data, rate = wave_data()
    model = DecayModel(0).eval()
    result = evaluate_model(model, data, [0, 1], device="cpu")
    assert not model.training
    expected = 1 - torch.exp(-rate * data["times"])
    torch.testing.assert_close(result["nrmse_time"], expected.expand(2, -1))
    torch.testing.assert_close(result["rhs_relative_error_time"], torch.ones(2, 2, dtype=torch.float64))
    torch.testing.assert_close(result["power_residual_time"], result["dissipation_rate"])
    torch.testing.assert_close(result["nrmse_time"][:, 1], result["persistence_nrmse_time"][:, 0])
    assert result["relative_l2_time"][0, -1] > result["nrmse_time"][0, -1]


def test_divergence_is_measured_without_projecting_predictions():
    data, _ = wave_data()
    result = evaluate_model(DivergentModel(0), data, [0], device="cpu")
    expected = data["times"] * (2 * math.pi / math.sqrt(2))
    torch.testing.assert_close(result["divergence_rms"][0], expected, atol=1e-13, rtol=1e-13)
    assert result["true_divergence_rms"].max() == 0


def test_forcing_and_dissipation_follow_the_left_endpoint_controls():
    data, rate = wave_data()
    basis = data["clean"][:1, 0].clone()
    data["controls"] = torch.tensor([[[0.2], [-0.1]]], dtype=torch.float64).expand(2, -1, -1)
    data["forcing"] = data["controls"][..., None, None, None] * basis[:, None]
    states = [data["clean"][:, 0]]
    for forcing, dt in zip(data["forcing"].unbind(dim=1), data["times"].diff()):
        decay = torch.exp(-rate * dt)
        states.append(decay * states[-1] + (1 - decay) * forcing / rate)
    data["clean"] = torch.stack(states, dim=1)
    result = evaluate_model(ForcedDecayModel(rate, basis), data, [0, 1], device="cpu")
    assert result["rhs_relative_error_time"].max() < 1e-13
    assert result["force_nrmse_time"].max() < 1e-13
    assert result["power_residual_time"].abs().max() < 1e-13
    assert (result["input_power"][:, 0] > 0).all()
    assert (result["input_power"][:, 1] < 0).all()
    assert (result["dissipation_rate"] > 0).all()


def test_nonfinite_predictions_are_reported_and_mode_restored():
    data, _ = wave_data()
    model = NonfiniteModel(0)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        evaluate_model(model, data, [0], device="cpu")
    assert model.training


def test_empty_split_is_rejected():
    data, rate = wave_data()
    with pytest.raises(ValueError, match="at least one"):
        evaluate_model(DecayModel(rate), data, [], device="cpu")
