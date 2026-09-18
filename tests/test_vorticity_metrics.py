import math

import pytest
import torch
from torch import nn

from experiments.vorticity_metrics import VelocityFromVorticity, evaluate_vorticity_model
from experiments.vorticity_operators import PeriodicVorticity


class VorticityDecay(nn.Module):
    def __init__(self, rate, basis):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=basis.dtype))
        self.register_buffer("basis", basis)

    def forward(self, vorticity, control=None):
        rate = -self.rate * vorticity
        if control is not None:
            rate = rate + torch.einsum("bm,mcxyz->bcxyz", control, self.basis)
        return rate

    def rollout(self, initial, controls, times):
        states = [initial]
        for step, dt in enumerate(times.diff()):
            control = None if controls is None else controls[:, step]
            states.append(states[-1] + dt * self(states[-1], control))
        return torch.stack(states, dim=1)


class VorticityWithDefects(VorticityDecay):
    def forward(self, vorticity, control=None):
        rate = super().forward(vorticity, control)
        axis = torch.arange(vorticity.shape[-1], dtype=vorticity.dtype, device=vorticity.device)
        defect = torch.zeros_like(vorticity)
        defect[:, 0] = torch.sin(2 * math.pi * axis / axis.numel())[:, None, None]
        defect[:, 2] = 0.25
        return rate + defect


def wave_data(forced=False):
    grid_size = 8
    viscosity = 0.01
    rate = 4 * math.pi**2 * viscosity
    operators = PeriodicVorticity(grid_size)
    axis = torch.arange(grid_size, dtype=torch.float64) / grid_size
    initial = torch.zeros(2, 3, grid_size, grid_size, grid_size, dtype=torch.float64)
    initial[:, 1] = torch.sin(2 * math.pi * axis)[:, None, None]
    initial[1] *= 2
    basis = initial[:1].clone()
    controls = torch.zeros(2, 2, 1, dtype=torch.float64)
    if forced:
        basis[:, 1] += 0.3
        controls = torch.tensor([[[0.2], [-0.1]]], dtype=torch.float64).expand(2, -1, -1)
    times = torch.tensor([0.0, 0.1, 0.25], dtype=torch.float64)
    mean = torch.tensor([0.0, 0.4, 0.0], dtype=torch.float64).expand(2, -1).clone()
    initial = initial + mean[..., None, None, None]
    forcing = torch.einsum("btm,mcxyz->btcxyz", controls, basis)
    states = [initial]
    for step, dt in enumerate(times.diff()):
        decay = torch.exp(-rate * dt)
        mean_force = forcing[:, step].mean(dim=(-3, -2, -1))
        zero_mean = states[-1] - mean[..., None, None, None]
        mean = mean + dt * mean_force
        forced_mode = forcing[:, step] - mean_force[..., None, None, None]
        states.append(
            decay * zero_mean + (1 - decay) * forced_mode / rate + mean[..., None, None, None]
        )
    clean = torch.stack(states, dim=1)
    data = {
        "clean": clean,
        "noisy": clean.clone(),
        "times": times,
        "controls": controls,
        "forcing_basis": basis,
        "forcing": forcing,
        "metadata": {"viscosity": viscosity},
    }
    return data, operators, rate


@pytest.mark.parametrize("forced", [False, True])
def test_velocity_adapter_recovers_rates_and_preserves_mean(forced):
    data, operators, rate = wave_data(forced)
    model = VorticityDecay(rate, operators.curl(data["forcing_basis"]))
    adapter = VelocityFromVorticity(model, operators, data["forcing_basis"])
    initial = data["clean"][:, 0]
    prediction = adapter.rollout(initial, data["controls"], data["times"])
    torch.testing.assert_close(prediction[:, 0], initial)
    torch.testing.assert_close(
        prediction.mean(dim=(-3, -2, -1)), data["clean"].mean(dim=(-3, -2, -1))
    )
    mean = initial.mean(dim=(-3, -2, -1))[..., None, None, None]
    expected_rate = -rate * (initial - mean) + data["forcing"][:, 0]
    torch.testing.assert_close(adapter(initial, data["controls"][:, 0]), expected_rate)
    assert adapter.last_vorticity.shape == data["clean"].shape


def test_velocity_adapter_keeps_autograd_and_handles_missing_controls():
    data, operators, rate = wave_data()
    model = VorticityDecay(rate, operators.curl(data["forcing_basis"]))
    adapter = VelocityFromVorticity(model, operators, data["forcing_basis"])
    prediction = adapter.rollout(data["clean"][:, 0], None, data["times"])
    prediction.square().mean().backward()
    assert torch.isfinite(model.rate.grad)
    assert model.rate.grad.abs() > 0
    torch.testing.assert_close(
        prediction.mean(dim=(-3, -2, -1)), data["clean"].mean(dim=(-3, -2, -1))
    )


@pytest.mark.parametrize("training", [False, True])
def test_vorticity_metrics_recover_true_equations_and_restore_mode(training):
    data, operators, rate = wave_data(forced=True)
    model = VorticityDecay(rate, operators.curl(data["forcing_basis"])).train(training)
    result = evaluate_vorticity_model(model, data, [0, 1], device="cpu")
    assert model.training == training
    assert all(value.device.type == "cpu" for value in result.values())
    assert result["rhs_relative_error_time"].max() < 1e-12
    assert result["vorticity_rhs_relative_error_time"].max() < 1e-12
    assert result["force_nrmse_time"].max() < 1e-12
    assert result["power_residual_time"].abs().max() < 1e-12
    assert result["vorticity_consistency_nrmse_time"].max() < 1e-12
    assert result["vorticity_mean_norm"].max() < 1e-12
    assert result["vorticity_divergence_rms"].max() < 1e-12
    torch.testing.assert_close(
        result["true_enstrophy"][:, 0], torch.tensor([math.pi**2, 4 * math.pi**2], dtype=torch.float64)
    )
    assert result["vorticity_nrmse_time"][:, -1].min() > 0


def test_raw_vorticity_defects_remain_visible_after_velocity_reconstruction():
    data, operators, rate = wave_data()
    basis = operators.curl(data["forcing_basis"])
    reference = evaluate_vorticity_model(VorticityDecay(rate, basis), data, [0], device="cpu")
    result = evaluate_vorticity_model(VorticityWithDefects(rate, basis), data, [0], device="cpu")
    torch.testing.assert_close(result["prediction"], reference["prediction"])
    assert result["divergence_rms"].max() < 1e-12
    assert result["vorticity_divergence_rms"][0, -1] > 0.5
    assert result["vorticity_mean_norm"][0, -1] > 0.05
    assert result["vorticity_consistency_nrmse_time"][0, -1] > 0.01
    assert result["vorticity_rhs_relative_error_time"].min() > 0.1


def test_vorticity_metrics_reject_an_empty_split():
    data, operators, rate = wave_data()
    model = VorticityDecay(rate, operators.curl(data["forcing_basis"]))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_vorticity_model(model, data, [], device="cpu")


def test_vorticity_evaluation_forwards_method_and_solver_options_through_adapter(monkeypatch):
    # Both representation conversion layers must retain the requested integrator.
    data, operators, rate = wave_data()
    model = VorticityDecay(rate, operators.curl(data["forcing_basis"]))
    original = model.rollout
    received = []
    options = {"max_iterations": 29, "atol": 1e-9}

    def rollout(initial, controls, times, method, solver_options):
        received.append((method, solver_options))
        return original(initial, controls, times)

    monkeypatch.setattr(model, "rollout", rollout)
    evaluate_vorticity_model(model, data, [0], device="cpu", method="avf", solver_options=options)
    assert received == [("avf", options)]
