import math
import json
from dataclasses import asdict

import pytest
import torch
from torch import nn

from experiments.comparison import ComparisonConfig, build_model, copy_state
from experiments.long_rollout import extend_reference, guarded_rollout, evaluate_long_model, load_selected_checkpoint
from nsdata import generate_dataset


class IdentityCoordinates:
    def project(self, field):
        return field


class MeanCoordinates:
    def project(self, field):
        return field.mean(dim=(-3, -2, -1), keepdim=True).expand_as(field).clone()


class ForcedRecurrence(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.coordinates = MeanCoordinates()
        self.control_channels = 1
        self.seen_dt = []

    def step(self, field, control, dt, method="native"):
        assert method == "native"
        assert not torch.is_grad_enabled()
        self.seen_dt.append(float(dt))
        return field + dt * (self.rate * field + control[:, :, None, None, None])


class SelectiveFailure(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.coordinates = IdentityCoordinates()
        self.control_channels = 1
        self.kind = kind

    def step(self, field, control, dt):
        fail = (field[:, 0, 0, 0, 0] > 0.5) & (field[:, 0, 0, 0, 0] < 1.5)
        if fail.any() and self.kind == "gonzalez":
            raise RuntimeError("Gonzalez step did not converge after 50 iterations (residual nan)")
        if fail.any() and self.kind == "unexpected":
            raise RuntimeError("unrelated implementation error")
        result = field + 1
        if self.kind == "nonfinite":
            result[fail] = torch.inf
        return result


@pytest.mark.parametrize("kind", ["wave", "taylor_green", "random"])
def test_reference_extension_keeps_heldout_initial_states_and_left_endpoint_forcing(kind):
    source = generate_dataset(
        kind, grid_size=12, n_trajectories=3, n_snapshots=3,
        final_time=0.05, viscosity=0.01, forcing_amplitude=0.13,
        forcing_frequency=0.7, max_dt=0.005, seed=17, device="cpu",
    )
    original = source["clean"].clone()
    original_metadata = dict(source["metadata"])
    source["noisy"].fill_(1234)
    reference = extend_reference(source, [2, 0], final_time=0.1, dt=0.025, device="cpu")
    assert reference["clean"].shape == (2, 5, 3, 12, 12, 12)
    assert reference["clean"].device.type == "cpu"
    assert reference["clean"].dtype == torch.float32
    assert list(reference["source_indices"]) == [2, 0]
    torch.testing.assert_close(reference["times"], torch.arange(5, dtype=torch.float64) * 0.025)
    torch.testing.assert_close(reference["clean"][:, :3], original[[2, 0]], atol=2e-7, rtol=2e-5)
    expected_controls = 0.13 * torch.cos(2 * math.pi * 0.7 * reference["times"][:-1])
    torch.testing.assert_close(
        reference["controls"].double(), expected_controls[None, :, None].expand(2, -1, -1),
        atol=1e-8, rtol=1e-6,
    )
    torch.testing.assert_close(reference["forcing_basis"], source["forcing_basis"])
    assert not torch.allclose(reference["clean"][:, -1], original[[2, 0], -1])
    assert reference["metadata"]["n_snapshots"] == 5
    assert reference["metadata"]["final_time"] == 0.1
    for key in ("domain", "grid_size", "viscosity", "forcing_amplitude", "forcing_frequency"):
        assert reference["metadata"][key] == original_metadata[key]
    assert source["metadata"] == original_metadata
    torch.testing.assert_close(source["clean"], original, rtol=0, atol=0)


def test_default_reference_horizon_has_801_genuinely_new_snapshots():
    source = generate_dataset(
        "wave", grid_size=10, n_trajectories=1, n_snapshots=3,
        final_time=0.05, seed=7, device="cpu",
    )
    reference = extend_reference(source, [0], device="cpu")
    assert reference["clean"].shape[1] == 801
    assert reference["times"][0] == 0
    assert reference["times"][-1] == 20
    assert reference["controls"].shape == (1, 800, 1)
    torch.testing.assert_close(reference["times"].diff(), torch.full((800,), 0.025, dtype=torch.float64))
    torch.testing.assert_close(reference["clean"][:, :3], source["clean"], atol=2e-7, rtol=2e-5)
    assert not torch.allclose(reference["clean"][:, -1], source["clean"][:, -1])


def test_guarded_rollout_is_autonomous_and_uses_native_step_with_forcing():
    model = ForcedRecurrence()
    initial = torch.arange(3 * 4**3, dtype=torch.float64).reshape(1, 3, 4, 4, 4) / 100
    times = torch.arange(5, dtype=torch.float64) * 0.025
    controls = torch.tensor([[[0.2], [-0.4], [0.1], [0.8]]], dtype=torch.float64)
    prediction, failures = guarded_rollout(model, initial, controls, times)
    expected = [model.coordinates.project(initial)]
    for index in range(4):
        expected.append(expected[-1] + 0.025 * (0.5 * expected[-1] + controls[:, index, :, None, None, None]))
    torch.testing.assert_close(prediction, torch.stack(expected, dim=1))
    assert failures == []
    assert len(model.seen_dt) == 4
    assert model.seen_dt == pytest.approx([0.025] * 4)
    assert prediction.device.type == "cpu"
    assert not prediction.requires_grad


@pytest.mark.parametrize("kind", ["nonfinite", "gonzalez"])
def test_guarded_rollout_localizes_failure_and_continues_survivors(kind):
    initial = torch.zeros(2, 3, 4, 4, 4, dtype=torch.float64)
    initial[1] = 10
    times = torch.arange(5, dtype=torch.float64) * 0.025
    controls = torch.zeros(2, 4, 1, dtype=torch.float64)
    prediction, failures = guarded_rollout(SelectiveFailure(kind), initial, controls, times)
    assert torch.isfinite(prediction[0, :2]).all()
    assert not torch.isfinite(prediction[0, 2]).any()
    assert torch.isnan(prediction[0, 3:]).all()
    expected_survivor = (10 + torch.arange(5, dtype=torch.float64))[:, None, None, None, None]
    torch.testing.assert_close(prediction[1], expected_survivor.expand_as(prediction[1]))
    assert len(failures) == 1
    assert failures[0]["trajectory_index"] == 0
    assert failures[0]["step"] == 2
    assert failures[0]["time"] == pytest.approx(0.05)
    assert failures[0]["reason"]


def test_guarded_rollout_does_not_misreport_programming_errors_as_divergence():
    initial = torch.ones(1, 3, 4, 4, 4, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="unrelated implementation error"):
        guarded_rollout(
            SelectiveFailure("unexpected"), initial,
            torch.zeros(1, 1, 1, dtype=torch.float64), torch.tensor([0.0, 0.025], dtype=torch.float64),
        )


@pytest.mark.parametrize("name", ["PHFNO", "FNO"])
def test_guarded_rollout_matches_existing_native_model_rollout(name):
    torch.manual_seed(7)
    config = ComparisonConfig(
        cutoff=(1, 1, 1), hidden_channels=2, n_layers=1, mlp_width=4, device="cpu", progress=False,
    )
    model = build_model(name, config, (6, 6, 6)).eval()
    initial = 0.01 * torch.randn(2, 3, 6, 6, 6)
    controls = 0.02 * torch.randn(2, 3, 1)
    times = torch.arange(4, dtype=torch.float64) * 0.025
    with torch.no_grad():
        expected = model.rollout(initial, controls, times)
    prediction, failures = guarded_rollout(model, initial, controls, times)
    torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-6)
    assert failures == []


def test_checkpoint_loader_requires_the_minimum_validation_error_step(tmp_path):
    config = ComparisonConfig(
        cutoff=(1, 1, 1), hidden_channels=2, n_layers=1, mlp_width=4, device="cpu", progress=False,
    )
    model = build_model("FNO", config, (6, 6, 6))
    checkpoint = {"state_dict": copy_state(model), "config": asdict(config),
                  "model": "FNO", "seed": 7, "best_step": 1}
    path = tmp_path / "FNO_7.pt"
    torch.save(checkpoint, path)
    run = {"model": "FNO", "seed": 7, "best_step": 1,
           "history": [{"step": 0, "validation_nrmse": 0.4},
                       {"step": 1, "validation_nrmse": 0.1},
                       {"step": 2, "validation_nrmse": 0.2}]}
    summary = {"config": json.loads(json.dumps(asdict(config))), "runs": [run], "evaluation_grid_size": 6}
    restored, record = load_selected_checkpoint(path, "velocity", summary, device="cpu")
    assert not restored.training
    assert record["best_step"] == 1
    assert record["best_validation_nrmse"] == 0.1
    for name, value in copy_state(restored).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, checkpoint["state_dict"][name], atol=0, rtol=0)
    checkpoint["best_step"] = 2
    run["best_step"] = 2
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="validation-selected step"):
        load_selected_checkpoint(path, "velocity", summary, device="cpu")


def test_velocity_metrics_normalize_by_initial_rms_and_map_failure_to_source_index():
    initial = torch.full((2, 3, 4, 4, 4), 0.25, dtype=torch.float64)
    initial[1] = 10
    times = torch.arange(5, dtype=torch.float64) * 0.025
    clean = initial[:, None] + 2 * torch.arange(5, dtype=torch.float64)[None, :, None, None, None, None]
    reference = {
        "clean": clean, "times": times, "controls": torch.zeros(2, 4, 1, dtype=torch.float64),
        "forcing_basis": torch.zeros(1, 3, 4, 4, 4, dtype=torch.float64),
        "metadata": {"viscosity": 0.01}, "source_indices": [28, 31],
    }
    metrics = evaluate_long_model(SelectiveFailure("nonfinite"), reference, "velocity", device="cpu")
    torch.testing.assert_close(metrics["initial_rms"], torch.tensor([0.25, 10.0], dtype=torch.float64))
    torch.testing.assert_close(metrics["nrmse_time"][1], torch.arange(5, dtype=torch.float64) / 10)
    expected_energy = 1.5 * (10 + torch.arange(5, dtype=torch.float64)).square()
    torch.testing.assert_close(metrics["kinetic_energy"][1], expected_energy)
    assert metrics["failures"][0]["trajectory_index"] == 28
    assert not torch.isfinite(metrics["nrmse_time"][0, 2])
    assert torch.isnan(metrics["nrmse_time"][0, 3:]).all()
    assert torch.isfinite(metrics["nrmse_time"][1]).all()


class FiniteGrowth(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.coordinates = IdentityCoordinates()
        self.control_channels = 1

    def step(self, field, control, dt):
        return 2 * field


def test_diagnostic_reductions_do_not_overflow_before_finite_float32_states():
    initial = torch.full((1, 3, 4, 4, 4), 1e20, dtype=torch.float32)
    reference = {
        "clean": initial[:, None].expand(-1, 2, -1, -1, -1, -1).clone(),
        "times": torch.tensor([0.0, 0.025], dtype=torch.float64),
        "controls": torch.zeros(1, 1, 1), "forcing_basis": torch.zeros(1, 3, 4, 4, 4),
        "metadata": {"viscosity": 0.01}, "source_indices": [28],
    }
    metrics = evaluate_long_model(FiniteGrowth(), reference, "velocity", device="cpu")
    assert torch.isfinite(metrics["kinetic_energy"]).all()
    assert torch.isfinite(metrics["nrmse_time"]).all()
    assert metrics["kinetic_energy"][0, -1] > 5e40
    torch.testing.assert_close(metrics["nrmse_time"], torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    assert metrics["failures"] == []


class RawVorticityDefects(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.coordinates = IdentityCoordinates()
        self.control_channels = 1

    def step(self, field, control, dt):
        axis = torch.arange(field.shape[-1], dtype=field.dtype, device=field.device) / field.shape[-1]
        defect = torch.zeros_like(field)
        defect[:, 0] = torch.sin(2 * math.pi * axis)[:, None, None]
        defect[:, 2] = 0.25
        return field + dt * defect


def test_vorticity_diagnostics_keep_raw_defects_and_reconstruct_forced_velocity_mean():
    initial = torch.zeros(1, 3, 8, 8, 8, dtype=torch.float64)
    initial[:, 1] = torch.sin(2 * math.pi * torch.arange(8, dtype=torch.float64) / 8)[:, None, None]
    initial += torch.tensor([0.1, 0.4, -0.2], dtype=torch.float64)[None, :, None, None, None]
    basis = torch.zeros_like(initial)
    basis[:, 0] = 0.3
    times = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    controls = torch.tensor([[[0.2], [-0.1]]], dtype=torch.float64)
    clean = [initial]
    for index, dt in enumerate(times.diff()):
        clean.append(clean[-1] + dt * controls[:, index, :, None, None, None] * basis)
    reference = {
        "clean": torch.stack(clean, dim=1), "times": times, "controls": controls,
        "forcing_basis": basis, "metadata": {"viscosity": 0.01}, "source_indices": [28],
    }
    metrics = evaluate_long_model(RawVorticityDefects(), reference, "vorticity", device="cpu")
    torch.testing.assert_close(metrics["prediction"], reference["clean"], atol=1e-12, rtol=1e-12)
    assert metrics["nrmse_time"].max() < 1e-12
    assert metrics["nrmse_time"].dtype == torch.float64
    assert metrics["vorticity_divergence_rms"][0, -1] > 0.5
    assert metrics["vorticity_mean_norm"][0, -1] > 0.04
    assert metrics["vorticity_consistency_nrmse_time"][0, -1] > 0.02
    assert metrics["vorticity_prediction"].shape == reference["clean"].shape
    assert metrics["failures"] == []
    initial_rms = initial.square().mean().sqrt()
    torch.testing.assert_close(metrics["initial_rms"], initial_rms[None])
    energy = 0.5 * reference["clean"].square().sum(dim=2).mean(dim=(-3, -2, -1))
    torch.testing.assert_close(metrics["kinetic_energy"], energy, atol=1e-12, rtol=1e-12)
