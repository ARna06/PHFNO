import json

import pytest
import torch

from nsdata import add_observation_noise, generate_dataset, resolve_device, save_dataset
from nsdata.solver import PeriodicNavierStokes
from phfno import FNOBaseline, PHFNO


def small_dataset(kind="wave", **kwargs):
    return generate_dataset(
        kind, grid_size=12, n_trajectories=2, n_snapshots=3,
        final_time=0.02, initial_rms=0.2, device="cpu", **kwargs,
    )


@pytest.mark.parametrize("kind", ["wave", "taylor_green", "random"])
def test_generated_trajectories_have_expected_physics_and_layout(kind):
    dataset = small_dataset(kind)
    clean = dataset["clean"]
    assert clean.shape == dataset["noisy"].shape == (2, 3, 3, 12, 12, 12)
    assert clean.dtype == torch.float32 and clean.device.type == "cpu"
    assert dataset["controls"].shape == (2, 2, 1)
    assert dataset["forcing"].shape == (2, 2, 3, 12, 12, 12)
    assert dataset["viscous_term"].shape == clean.shape
    torch.testing.assert_close(dataset["grid"], torch.arange(12) / 12)
    torch.testing.assert_close(dataset["times"], torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64))
    torch.testing.assert_close(clean[:, 0].square().mean(dim=(1, 2, 3, 4)).sqrt(), torch.full((2,), 0.2))
    assert torch.isfinite(clean).all() and torch.isfinite(dataset["noisy"]).all()
    assert dataset["diagnostics"]["divergence_rms"].max() < 2e-6
    assert dataset["diagnostics"]["dissipation_rate"].min() >= -1e-7
    assert not torch.equal(clean[0, 0], clean[1, 0])


def test_gaussian_noise_uses_fixed_initial_rms_and_preserves_clean_data():
    clean = torch.ones(4, 5, 3, 8, 8, 8, dtype=torch.float64)
    clean[:, 1:] *= 0.1
    original = clean.clone()
    rng_state = torch.random.get_rng_state()
    noisy, sigma = add_observation_noise(clean, noise_level=0.05, seed=123)
    torch.testing.assert_close(sigma, torch.full((4,), 0.05, dtype=torch.float64))
    torch.testing.assert_close(clean, original)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    repeated, repeated_sigma = add_observation_noise(clean, 0.05, seed=123)
    torch.testing.assert_close(noisy, repeated, rtol=0, atol=0)
    torch.testing.assert_close(sigma, repeated_sigma)
    residual = noisy - clean
    assert abs(residual.mean().item()) < 0.002
    torch.testing.assert_close(
        residual.square().mean(dim=(0, 2, 3, 4, 5)).sqrt(),
        torch.full((5,), 0.05, dtype=torch.float64), rtol=0.04, atol=0,
    )
    noiseless, zero_sigma = add_observation_noise(clean, 0.0)
    torch.testing.assert_close(noiseless, clean, rtol=0, atol=0)
    assert torch.count_nonzero(zero_sigma) == 0


def test_projected_observation_noise_stays_divergence_free():
    dataset = small_dataset(project_noise=True, noise_level=0.1)
    solver = PeriodicNavierStokes(12, 0.01, dtype=torch.float32)
    divergence = solver.divergence(dataset["noisy"].flatten(0, 1))
    assert divergence.square().mean().sqrt() < 2e-6
    assert dataset["metadata"]["noise_type"] == "projected Gaussian"
    independent = small_dataset(project_noise=False, noise_level=0.1)
    assert solver.divergence(independent["noisy"].flatten(0, 1)).square().mean().sqrt() > 0.01


def test_saved_dataset_loads_without_custom_python_objects(tmp_path):
    dataset = small_dataset()
    path = save_dataset(dataset, tmp_path / "wave.pt")
    restored = torch.load(path, weights_only=True)
    torch.testing.assert_close(restored["clean"], dataset["clean"])
    assert restored["metadata"] == dataset["metadata"]
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["shape"] == list(dataset["clean"].shape)
    assert manifest["seed"] == dataset["metadata"]["seed"]
    with pytest.raises(FileExistsError):
        save_dataset(dataset, path)
    save_dataset(dataset, path, overwrite=True)


def test_observation_noise_does_not_change_the_generated_dynamics():
    clean_run = small_dataset("random", noise_level=0)
    noisy_run = small_dataset("random", noise_level=0.1)
    torch.testing.assert_close(clean_run["clean"], noisy_run["clean"], rtol=0, atol=0)
    assert not torch.equal(clean_run["noisy"], noisy_run["noisy"])


def test_forcing_and_dissipation_are_saved_as_physical_terms():
    dataset = small_dataset(noise_level=0)
    controls = dataset["controls"][..., 0]
    expected_forcing = controls[:, :, None, None, None, None] * dataset["forcing_basis"][None]
    torch.testing.assert_close(dataset["forcing"], expected_forcing)
    decay_rate = 56 * torch.pi**2 * dataset["metadata"]["viscosity"]
    torch.testing.assert_close(dataset["viscous_term"], -decay_rate * dataset["clean"], atol=3e-6, rtol=2e-5)
    derivative = dataset["viscous_term"][:, :-1] + dataset["forcing"]
    power = (dataset["clean"][:, :-1] * derivative).sum(dim=2).mean(dim=(2, 3, 4))
    torch.testing.assert_close(dataset["diagnostics"]["energy_rate"], power)
    unforced = small_dataset(noise_level=0, forcing_amplitude=0)
    torch.testing.assert_close(dataset["clean"][:, 0], unforced["clean"][:, 0])
    assert not torch.equal(dataset["clean"][:, -1], unforced["clean"][:, -1])
    assert torch.count_nonzero(unforced["forcing"]) == 0


def test_zero_viscosity_removes_dissipation_without_removing_forcing():
    dataset = small_dataset(viscosity=0, noise_level=0)
    assert torch.count_nonzero(dataset["viscous_term"]) == 0
    assert torch.count_nonzero(dataset["diagnostics"]["dissipation_rate"]) == 0
    assert dataset["forcing"].abs().sum() > 0
    increments = dataset["clean"][:, 1:] - dataset["clean"][:, :-1]
    dt = dataset["times"].diff().float()[None, :, None, None, None, None]
    torch.testing.assert_close(increments, dt * dataset["forcing"], atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_generated_fields_fit_both_model_interfaces(device):
    data = generate_dataset(
        "taylor_green", grid_size=8, n_trajectories=1, n_snapshots=2,
        final_time=0.01, forcing_amplitude=0, device=device,
    )
    initial, target = data["clean"][:, 0].to(device), data["clean"][:, 1].to(device)
    models = (
        PHFNO((1, 1, 1), 3, control_channels=1, hidden_channels=4, n_layers=1, mlp_width=8, parameter_grid=(8, 8, 8)),
        FNOBaseline((1, 1, 1), 3, control_channels=1, hidden_channels=4, n_layers=1),
    )
    for model in models:
        model = model.to(device)
        prediction = model.step(initial, data["controls"][:, 0].to(device), 0.01)
        assert prediction.shape == target.shape
        torch.nn.functional.mse_loss(prediction, target).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_generation_cli_writes_a_complete_dataset(tmp_path, monkeypatch, capsys):
    from nsdata.__main__ import main

    monkeypatch.setattr("sys.argv", [
        "nsdata", "--kind", "wave", "--grid-size", "8", "--trajectories", "1",
        "--snapshots", "2", "--final-time", "0.01", "--device", "cpu",
        "--output-dir", str(tmp_path),
    ])
    main()
    assert (tmp_path / "wave.pt").exists()
    assert (tmp_path / "wave.json").exists()
    assert "Saved" in capsys.readouterr().out


def test_explicit_cuda_request_is_not_silently_downgraded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
