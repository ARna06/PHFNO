import json
import math
from numbers import Integral
from pathlib import Path

import torch

from .initial_conditions import exact_wave_trajectory, plane_wave, random_velocity, taylor_green, wave_forcing
from .solver import PeriodicNavierStokes


def resolve_device(device="cuda"):
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def add_observation_noise(clean, noise_level=0.01, seed=0, projector=None):
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative")
    if clean.ndim != 6 or clean.shape[2] != 3:
        raise ValueError("clean must have shape [batch, time, 3, x, y, z]")
    reference_rms = clean[:, 0].square().mean(dim=(1, 2, 3, 4)).sqrt()
    sigma = noise_level * reference_rms
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(clean.device)
    if projector is not None:
        noise = projector(noise.flatten(0, 1)).reshape_as(clean)
    noisy = clean + sigma[:, None, None, None, None, None] * noise
    return noisy, sigma


@torch.no_grad()
def generate_dataset(
    kind,
    grid_size=16,
    n_trajectories=4,
    n_snapshots=21,
    final_time=1.0,
    viscosity=0.01,
    initial_rms=0.2,
    forcing_amplitude=0.1,
    forcing_frequency=1.0,
    cutoff=2,
    max_dt=0.005,
    noise_level=0.01,
    project_noise=False,
    seed=0,
    device="cuda",
    dtype=torch.float64,
):
    if kind not in ("wave", "taylor_green", "random"):
        raise ValueError("kind must be wave, taylor_green, or random")
    for name, value, minimum in (
        ("grid_size", grid_size, 4),
        ("n_trajectories", n_trajectories, 1),
        ("n_snapshots", n_snapshots, 2),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    for name, value in (("final_time", final_time), ("initial_rms", initial_rms), ("max_dt", max_dt)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("viscosity", viscosity), ("noise_level", noise_level),
        ("forcing_amplitude", forcing_amplitude), ("forcing_frequency", forcing_frequency),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer between 0 and 2**63 - 2")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    device = resolve_device(device)
    times = torch.linspace(0, final_time, n_snapshots, device=device, dtype=dtype)
    if forcing_amplitude and kind != "wave" and grid_size <= 9:
        raise ValueError("nonzero forcing requires grid_size >= 10 with two-thirds dealiasing")
    initial_options = dict(batch_size=n_trajectories, seed=seed, device=device, dtype=dtype)
    if kind == "wave":
        initial = plane_wave(grid_size, **initial_options)
    elif kind == "taylor_green":
        initial = taylor_green(grid_size, **initial_options)
    else:
        initial = random_velocity(grid_size, cutoff=cutoff, rms=initial_rms, **initial_options)
    initial = initial * (
        initial_rms / initial.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
    )
    controls = forcing_amplitude * torch.cos(2 * math.pi * forcing_frequency * times[:-1])
    controls = controls[None, :, None].expand(n_trajectories, -1, -1).clone()
    forcing_basis = (
        wave_forcing(grid_size, device=device, dtype=dtype) if forcing_amplitude
        else initial.new_zeros(1, 3, grid_size, grid_size, grid_size)
    )
    forcing = controls[..., 0, None, None, None, None] * forcing_basis[None]
    solver = PeriodicNavierStokes(grid_size, viscosity, device=device, dtype=dtype)
    if kind == "wave":
        clean = exact_wave_trajectory(initial, times, viscosity, forcing=forcing)
    else:
        clean = solver.solve(initial, times, max_dt=max_dt, forcing=forcing)
    noisy, sigma = add_observation_noise(
        clean, noise_level, seed=seed + 1,
        projector=solver.project if project_noise else None,
    )
    if not torch.isfinite(clean).all() or not torch.isfinite(noisy).all():
        raise RuntimeError("Generated data contains nonfinite values")
    clean, noisy, forcing = clean.cpu().float(), noisy.cpu().float(), forcing.cpu().float()
    if not all(torch.isfinite(field).all() for field in (clean, noisy, forcing)):
        raise RuntimeError("Generated fields overflow float32 storage")
    checker = PeriodicNavierStokes(grid_size, viscosity, dtype=torch.float32)
    divergence = checker.divergence(clean.flatten(0, 1)).reshape(
        n_trajectories, n_snapshots, -1
    )
    viscous_term = checker.viscous_term(clean.flatten(0, 1)).reshape_as(clean)
    dissipation_rate = -(clean * viscous_term).sum(dim=2).mean(dim=(2, 3, 4))
    input_power = (clean[:, :-1] * forcing).sum(dim=2).mean(dim=(2, 3, 4))
    metadata = {
        "format_version": 1,
        "kind": kind,
        "equation": "incompressible Navier-Stokes with prescribed forcing",
        "domain": [[0.0, 1.0]] * 3,
        "periodic": True,
        "endpoint_included": False,
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_snapshots": n_snapshots,
        "final_time": final_time,
        "viscosity": viscosity,
        "dissipation_operator": "viscosity * Laplacian(velocity)",
        "initial_rms": initial_rms,
        "forcing_amplitude": forcing_amplitude,
        "forcing_frequency": forcing_frequency,
        "forcing_wavevector": [2, 3, 1],
        "forcing_polarization": [1 / math.sqrt(5), 0.0, -2 / math.sqrt(5)],
        "forcing_basis_rms": 1.0 if forcing_amplitude else 0.0,
        "forcing_time_profile": "amplitude * cos(2*pi*frequency*t), held at each interval's left endpoint",
        "initial_random_cutoff": cutoff if kind == "random" else None,
        "wavevector": [2, 3, 1] if kind == "wave" else None,
        "method": "exact forced mode evolution" if kind == "wave" else "Fourier projection with RK4",
        "dealiasing": "none needed" if kind == "wave" else "strict two-thirds truncation",
        "max_dt": None if kind == "wave" else max_dt,
        "cfl": None if kind == "wave" else 0.4,
        "noise_level": noise_level,
        "noise_reference": "initial per-trajectory RMS over components and space",
        "noise_type": "projected Gaussian" if project_noise else "independent Gaussian",
        "noise_projection": "Leray projection and two-thirds truncation" if project_noise else None,
        "noise_sigma_convention": "standard deviation before optional projection",
        "seed": seed,
        "noise_seed": seed + 1,
        "generation_device": str(device),
        "generation_dtype": str(dtype),
        "stored_dtype": "torch.float32",
        "torch_version": str(torch.__version__),
    }
    return {
        "clean": clean,
        "noisy": noisy,
        "times": times.cpu(),
        "grid": torch.arange(grid_size, dtype=torch.float32) / grid_size,
        "controls": controls.cpu().float(),
        "forcing_basis": forcing_basis.cpu().float(),
        "forcing": forcing,
        "viscous_term": viscous_term,
        "noise_sigma": sigma.cpu().float(),
        "diagnostics": {
            "kinetic_energy": 0.5 * clean.square().sum(dim=2).mean(dim=(2, 3, 4)),
            "divergence_rms": divergence.square().mean(dim=-1).sqrt(),
            "mean_velocity": clean.mean(dim=(3, 4, 5)),
            "dissipation_rate": dissipation_rate,
            "input_power": input_power,
            "energy_rate": input_power - dissipation_rate[:, :-1],
        },
        "metadata": metadata,
    }


def save_dataset(dataset, path, overwrite=False):
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("dataset path must end in .pt")
    manifest_path = path.with_suffix(".json")
    if not overwrite and (path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Dataset already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = dataset["diagnostics"]
    manifest = {
        **dataset["metadata"],
        "shape": list(dataset["clean"].shape),
        "noise_sigma": dataset["noise_sigma"].tolist(),
        "max_divergence_rms": diagnostics["divergence_rms"].max().item(),
        "initial_kinetic_energy": diagnostics["kinetic_energy"][:, 0].tolist(),
        "final_kinetic_energy": diagnostics["kinetic_energy"][:, -1].tolist(),
        "initial_dissipation_rate": diagnostics["dissipation_rate"][:, 0].tolist(),
        "mean_input_power": diagnostics["input_power"].mean(dim=1).tolist(),
    }
    torch.save(dataset, path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path
