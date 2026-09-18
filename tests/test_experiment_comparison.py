import json
from dataclasses import asdict

import pytest
import torch
from torch import nn

from experiments.comparison import (
    ComparisonConfig,
    build_model,
    copy_state,
    parameter_count,
    predict_next,
    prepare_transitions,
    run_comparison,
    train_model,
    validation_error,
)
from phfno.fourier import RealFourierCoordinates


class ScalarModel(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))
        self.coordinates = RealFourierCoordinates((0, 0, 0), 3)
        self.training_inputs = []

    def forward(self, field, control):
        if self.training:
            self.training_inputs.append(field[:, 0, 0, 0, 0].detach().clone())
        return self.weight * field

    def step(self, field, control, dt, method="avf", solver_options=None):
        # Training fixtures implement the discrete-step interface used by both models.
        assert method == "avf"
        return field + dt.reshape(-1, 1, 1, 1, 1) * self(field, control)


def small_config(**kwargs):
    options = dict(
        seeds=(7,), cutoff=(1, 1, 1), hidden_channels=2, n_layers=1,
        mlp_width=4, steps=3, batch_size=2, eval_every=2, device="cpu", progress=False,
    )
    return ComparisonConfig(**{**options, **kwargs})


def sample_data(offset=0):
    values = offset + torch.arange(12, dtype=torch.float32).reshape(4, 3)
    clean = values[:, :, None, None, None, None].expand(-1, -1, 3, 8, 8, 8).clone()
    controls = offset + torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    return {
        "clean": clean,
        "noisy": clean + 0.25,
        "times": torch.tensor([0.0, 0.1, 0.3]),
        "controls": controls,
        "forcing": torch.zeros_like(clean[:, :-1]),
        "metadata": {"viscosity": 0.01},
    }


@pytest.mark.parametrize("name", ["PHFNO", "FNO"])
def test_predict_next_matches_native_steps_for_varying_intervals(name):
    torch.manual_seed(1)
    model = build_model(name, small_config(), (8, 8, 8)).eval()
    transitions = {
        "field": torch.randn(3, 3, 8, 8, 8),
        "control": torch.randn(3, 1),
        "dt": torch.tensor([0.01, 0.02, 0.04]),
    }
    with torch.no_grad():
        prediction = predict_next(model, transitions, torch.arange(3))
        expected = torch.cat([
            model.step(
                transitions["field"][i:i + 1], transitions["control"][i:i + 1], dt,
                method="avf",
            )
            for i, dt in enumerate(transitions["dt"])
        ])
    torch.testing.assert_close(prediction, expected, atol=1e-6, rtol=1e-5)


def test_prepare_transitions_preserves_family_trajectory_and_time_alignment():
    families = {"first": sample_data(), "second": sample_data(100)}
    result = prepare_transitions(families, [2, 0], "cpu")
    torch.testing.assert_close(
        result["field"][:, 0, 0, 0, 0], torch.tensor([6.25, 7.25, 0.25, 1.25, 106.25, 107.25, 100.25, 101.25])
    )
    torch.testing.assert_close(
        result["clean_target"][:, 0, 0, 0, 0], torch.tensor([7.0, 8.0, 1.0, 2.0, 107.0, 108.0, 101.0, 102.0])
    )
    torch.testing.assert_close(result["noisy_target"], result["clean_target"] + 0.25)
    torch.testing.assert_close(result["control"][:, 0], torch.tensor([4.0, 5.0, 0.0, 1.0, 104.0, 105.0, 100.0, 101.0]))
    torch.testing.assert_close(result["dt"], torch.tensor([0.1, 0.2] * 4))
    for indices in ([-1], [4]):
        with pytest.raises(ValueError, match="outside"):
            prepare_transitions(families, indices, "cpu")


def test_overlapping_trajectory_splits_are_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        small_config(validation_indices=(1,))
    with pytest.raises(ValueError, match="disjoint"):
        small_config(test_indices=(2,))


def test_saved_config_requires_explicit_protocol_and_solver_metadata():
    # Old dictionaries must not acquire current integration and loss settings by default.
    recorded = asdict(small_config())
    assert ComparisonConfig.from_record(recorded) == small_config()
    assert ComparisonConfig.from_record(json.loads(json.dumps(recorded))) == small_config()
    for key in ("protocol_version", "loss_definition", "integration_method",
                "theta_update_start", "theta_update_interval", "avf_max_iterations",
                "avf_rtol", "avf_atol", "avf_quadrature_points"):
        legacy = {name: value for name, value in recorded.items() if name != key}
        with pytest.raises(ValueError, match=f"missing {key}"):
            ComparisonConfig.from_record(legacy)


@pytest.mark.parametrize("steps,start,interval,expected", [
    (23, 20, 5, [*range(1, 21), 23]),
    (14, 3, 5, [1, 2, 3, 8, 13, 14]),
    (7, 0, 3, [3, 6, 7]),
])
def test_accumulation_updates_at_relative_group_ends(monkeypatch, steps, start, interval, expected):
    # Offset starts and partial final groups previously caused early/discarded updates.
    training = prepare_transitions({"sample": sample_data()}, [0, 1], "cpu")
    validation = prepare_transitions({"sample": sample_data()}, [2], "cpu")
    model = ScalarModel()
    updates = []
    original_step = torch.optim.Adam.step

    def capture_step(optimizer, *args, **kwargs):
        updates.append(len(model.training_inputs) - 1)  # Exclude the initialization pass.
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_step)
    config = small_config(steps=steps, theta_update_start=start,
                          theta_update_interval=interval, eval_every=1)
    result = train_model(model, training, validation, config, 7, 1.0, 0.0)
    assert updates == expected
    assert result["optimizer_updates"] == len(expected)
    for entry in result["history"]:
        assert entry["optimizer_updates"] == sum(step <= entry["step"] for step in expected)
    assert result["best_optimizer_updates"] == sum(step <= result["best_step"] for step in expected)


def test_training_uses_seeded_batches_and_restores_best_validation_weights():
    training = prepare_transitions({"sample": sample_data()}, [0, 1], "cpu")
    training["dt"].fill_(1.0)
    training["noisy_target"] = 2 * training["field"]
    validation = {key: value.clone() for key, value in training.items()}
    validation["clean_target"] = validation["field"].clone()
    config = small_config(learning_rate=0.1)
    recordings = []
    for global_seed in (1, 999):
        torch.manual_seed(global_seed)
        model = ScalarModel()
        result = train_model(model, training, validation, config, 17, 1.0, 0.0)
        recordings.append(model.training_inputs[1:])
        assert len(model.training_inputs) == config.steps + 1
        assert result["best_step"] == 0
        assert result["best_validation_nrmse"] == 0
        assert result["threshold_step"] == 0
        assert result["threshold_seconds"] == 0
        assert model.weight.item() == 0
        assert validation_error(model, validation, 1.0, config.batch_size) == 0
        assert [entry["step"] for entry in result["history"]] == [0, 2, 3]
        seconds = [entry["optimization_seconds"] for entry in result["history"]]
        assert seconds[0] == 0
        assert seconds[0] < seconds[1] < seconds[2]
        assert result["history"][-1]["validation_nrmse"] > 0
        assert not model.training
    for first, second in zip(*recordings):
        torch.testing.assert_close(first, second)


def test_parameter_count_counts_trainable_real_scalar_values():
    model = nn.ParameterList([
        nn.Parameter(torch.zeros(3)),
        nn.Parameter(torch.zeros(2, dtype=torch.complex64)),
        nn.Parameter(torch.zeros(4), requires_grad=False),
    ])
    assert parameter_count(model) == 7


@pytest.mark.parametrize("enabled", [True, False])
def test_training_progress_reports_the_model_seed_and_validation(enabled, capsys):
    training = prepare_transitions({"sample": sample_data()}, [0, 1], "cpu")
    validation = prepare_transitions({"sample": sample_data()}, [2], "cpu")
    config = small_config(progress=enabled)
    train_model(ScalarModel(), training, validation, config, 7, 1.0, 0.1)
    output = capsys.readouterr().err
    if enabled:
        assert "ScalarModel seed 7" in output
        assert "3/3" in output
        assert "validation=" in output
    else:
        assert output == ""


def test_training_updates_one_notebook_progress_widget(monkeypatch, capsys):
    widgets = pytest.importorskip("ipywidgets")
    from tqdm.notebook import tqdm as notebook_tqdm

    created = []

    def create_progress(*args, **kwargs):
        progress = notebook_tqdm(*args, **kwargs)
        created.append(progress)
        return progress

    monkeypatch.setattr("experiments.comparison.tqdm", create_progress)
    training = prepare_transitions({"sample": sample_data()}, [0, 1], "cpu")
    validation = prepare_transitions({"sample": sample_data()}, [2], "cpu")
    config = small_config(progress=True)
    train_model(ScalarModel(), training, validation, config, 7, 1.0, 0.1)
    assert len(created) == 1
    progress = created[0]
    assert progress.n == progress.total == config.steps
    bars = [child for child in progress.container.children if isinstance(child, widgets.FloatProgress)]
    assert len(bars) == 1
    assert bars[0].value == config.steps
    assert "validation" in progress.postfix
    assert "3/3" not in capsys.readouterr().err


@pytest.mark.parametrize("name", ["PHFNO", "FNO"])
def test_model_snapshots_are_independent_and_load_strictly(name):
    model = build_model(name, small_config(), (8, 8, 8))
    snapshot = copy_state(model)
    assert "_metadata" not in snapshot
    assert all(value.device.type == "cpu" for value in snapshot.values())
    key, parameter = next(model.named_parameters())
    original = snapshot[key].clone()
    with torch.no_grad():
        parameter.add_(1)
    torch.testing.assert_close(snapshot[key], original)
    assert not torch.equal(parameter, original)
    loaded = model.load_state_dict(snapshot, strict=True)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys
    torch.testing.assert_close(parameter, original)


def test_small_cpu_comparison_saves_selected_models_and_test_results(tmp_path):
    data = sample_data()
    data["clean"] = (data["clean"][:, :1] + 1).expand(-1, 3, -1, -1, -1, -1).clone() * 0.01
    data["noisy"] = data["clean"].clone()
    data["controls"].zero_()
    config = small_config(steps=2)
    result = run_comparison({"constant": data}, config, tmp_path)
    assert [run["model"] for run in result["runs"]] == ["PHFNO", "FNO"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"]["test_indices"] == [3]
    assert summary["hardware"] == "CPU"
    # Saved predictions must reproduce the explicitly recorded training integrator.
    assert result["evaluation_method"] == "avf"
    for run in result["runs"]:
        checkpoint = torch.load(tmp_path / f"{run['model']}_7.pt", weights_only=True)
        assert checkpoint["best_step"] == run["best_step"]
        assert checkpoint["best_optimizer_updates"] == run["best_optimizer_updates"]
        metrics = run["test"]["constant"]
        torch.testing.assert_close(metrics["truth"], data["clean"][[3]])
        assert torch.isfinite(metrics["nrmse_time"]).all()
        model = build_model(run["model"], config, (8, 8, 8))
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            expected = model.rollout(data["clean"][[3], 0], data["controls"][[3]], data["times"],
                                     method=config.integration_method, solver_options=config.solver_options)
        torch.testing.assert_close(metrics["prediction"], expected)
        assert all(value.device.type == "cpu" for value in checkpoint["state_dict"].values())
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "results.pt").is_file()
