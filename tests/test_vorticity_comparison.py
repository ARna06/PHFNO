from dataclasses import asdict
import json

import pytest
import torch

from experiments.comparison import ComparisonConfig, run_comparison
from experiments.vorticity_comparison import run_vorticity_comparison, validate_velocity_reference, vorticity_dataset
from experiments.vorticity_operators import PeriodicVorticity
from nsdata import generate_dataset


@pytest.fixture
def data():
    result = generate_dataset(
        "random", grid_size=8, n_trajectories=4, n_snapshots=3,
        final_time=0.01, forcing_amplitude=0, cutoff=1, seed=42, device="cpu",
    )
    result["sha256"] = "example-data-hash"
    return result


def small_config():
    return ComparisonConfig(seeds=(7,), cutoff=(1, 1, 1), hidden_channels=2,
                            n_layers=1, mlp_width=4, steps=2, batch_size=2,
                            eval_every=1, device="cpu", progress=False)


def test_dataset_curls_observations_and_forcing_without_changing_source(data):
    original = data["clean"].clone()
    transformed = vorticity_dataset(data, device="cpu", batch_size=3)
    operators = PeriodicVorticity(8, dtype=torch.float32)
    for key in ("clean", "noisy", "forcing", "forcing_basis"):
        torch.testing.assert_close(transformed[key], operators.curl(data[key]))
    torch.testing.assert_close(transformed["noisy"] - transformed["clean"],
                               operators.curl(data["noisy"] - data["clean"]), atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(data["clean"], original)
    assert "state_representation" not in data["metadata"]
    assert transformed["metadata"]["state_representation"] == "vorticity"
    assert transformed["sha256"] == data["sha256"]
    torch.testing.assert_close(transformed["controls"], data["controls"])


def test_velocity_reference_checks_data_budget_and_seed_coverage(data):
    config = small_config()
    reference = {"config": asdict(config), "families": ["random"],
                 "dataset_sha256": {"random": data["sha256"]},
                 "runs": [{"model": name, "seed": 7} for name in ("PHFNO", "FNO")]}
    validate_velocity_reference(reference, {"random": data}, config)
    reference["config"]["steps"] += 1
    with pytest.raises(ValueError, match="different steps"):
        validate_velocity_reference(reference, {"random": data}, config)
    reference["config"] = asdict(config)
    reference["dataset_sha256"]["random"] = "another-hash"
    with pytest.raises(ValueError, match="different data"):
        validate_velocity_reference(reference, {"random": data}, config)
    reference["dataset_sha256"]["random"] = data["sha256"]
    reference["runs"].pop()
    with pytest.raises(ValueError, match="one run"):
        validate_velocity_reference(reference, {"random": data}, config)


@pytest.mark.parametrize("key,replacement", [
    ("integration_method", "euler"), ("theta_update_start", 0), ("theta_update_interval", 1),
    ("avf_max_iterations", 80), ("avf_rtol", 1e-5), ("avf_atol", 1e-7),
    ("avf_quadrature_points", 3), ("loss_definition", "mse"), ("protocol_version", 1),
])
def test_velocity_reference_rejects_mismatched_and_missing_protocol(data, key, replacement):
    # Representation comparisons require the same solver, update schedule and objective.
    config = small_config()
    reference = {"config": asdict(config), "families": ["random"],
                 "dataset_sha256": {"random": data["sha256"]},
                 "runs": [{"model": name, "seed": 7} for name in ("PHFNO", "FNO")]}
    reference["config"][key] = replacement
    with pytest.raises(ValueError, match=key):
        validate_velocity_reference(reference, {"random": data}, config)
    del reference["config"][key]
    with pytest.raises(ValueError, match=f"missing {key}"):
        validate_velocity_reference(reference, {"random": data}, config)


def test_small_vorticity_comparison_saves_raw_and_velocity_diagnostics(data, tmp_path):
    config = small_config()
    reference = run_comparison({"random": data}, config, tmp_path / "velocity")
    result = run_vorticity_comparison({"random": data}, config, tmp_path, velocity_reference=reference)
    assert len(result["runs"]) == 2
    assert result["validation_representation"] == "vorticity"
    operators = PeriodicVorticity(8, dtype=torch.float32)
    for run in result["runs"]:
        metrics = run["test"]["random"]
        torch.testing.assert_close(metrics["truth"], data["clean"][[3]])
        torch.testing.assert_close(metrics["vorticity_truth"], operators.curl(data["clean"][[3]]))
        assert torch.isfinite(metrics["vorticity_nrmse_time"]).all()
        assert torch.isfinite(metrics["nrmse_time"]).all()
        checkpoint = torch.load(tmp_path / f"{run['model']}_7.pt", weights_only=True)
        assert checkpoint["state_representation"] == "vorticity"
        assert checkpoint["best_step"] == run["best_step"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "vorticity_consistency_nrmse" in summary["runs"][0]["test"]["random"]
    assert (tmp_path / "results.pt").exists()
    saved_reference = json.loads((tmp_path / "velocity_reference.json").read_text())
    assert saved_reference["runs"][0]["best_validation_nrmse"] == reference["runs"][0]["best_validation_nrmse"]
