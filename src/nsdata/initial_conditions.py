import math

import torch


def _validate_grid(grid_size, batch_size, dtype, minimum):
    if type(grid_size) is not int or grid_size < minimum:
        raise ValueError(f"grid_size must be an integer of at least {minimum}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")


def _validate_scale(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _coordinates(grid_size, device, dtype):
    axis = torch.arange(grid_size, device=device, dtype=dtype) / grid_size
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))


def plane_wave(
    grid_size,
    batch_size=1,
    amplitude=1.0,
    seed=0,
    device="cpu",
    dtype=torch.float64,
):
    _validate_grid(grid_size, batch_size, dtype, minimum=7)
    _validate_scale(amplitude, "amplitude")
    x, y, z = _coordinates(grid_size, device, dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phases = 2 * math.pi * torch.rand(batch_size, generator=generator, dtype=dtype)
    phases = phases.to(device).reshape(batch_size, 1, 1, 1)
    wave = torch.sin(2 * math.pi * (2 * x + 3 * y + z) + phases)
    direction = torch.tensor([1, 0, -2], device=device, dtype=dtype) / math.sqrt(5)
    return amplitude * direction[None, :, None, None, None] * wave[:, None]


def taylor_green(
    grid_size,
    batch_size=1,
    amplitude=1.0,
    seed=0,
    device="cpu",
    dtype=torch.float64,
):
    _validate_grid(grid_size, batch_size, dtype, minimum=3)
    _validate_scale(amplitude, "amplitude")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shifts = torch.rand(batch_size, 3, generator=generator, dtype=dtype).to(device)
    coordinates = _coordinates(grid_size, device, dtype)[None]
    angles = 2 * math.pi * (coordinates + shifts[:, :, None, None, None])
    x, y, z = angles.unbind(dim=1)
    return amplitude * torch.stack(
        (
            torch.sin(x) * torch.cos(y) * torch.cos(z),
            -torch.cos(x) * torch.sin(y) * torch.cos(z),
            torch.zeros_like(x),
        ),
        dim=1,
    )


def random_velocity(
    grid_size,
    batch_size=1,
    rms=0.2,
    cutoff=2,
    seed=0,
    device="cpu",
    dtype=torch.float64,
):
    _validate_grid(grid_size, batch_size, dtype, minimum=4)
    _validate_scale(rms, "rms")
    if type(cutoff) is not int or cutoff < 1 or 3 * cutoff >= grid_size:
        raise ValueError("cutoff must be a positive integer strictly below grid_size / 3")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    field = torch.randn(
        batch_size, 3, grid_size, grid_size, grid_size, generator=generator, dtype=dtype
    ).to(device)
    frequencies = torch.fft.fftfreq(grid_size, d=1 / grid_size, device=device, dtype=dtype)
    modes = torch.stack(torch.meshgrid(frequencies, frequencies, frequencies, indexing="ij"))
    squared_norm = modes.square().sum(dim=0)
    mask = (modes.abs() <= cutoff).all(dim=0) & (squared_norm > 0)
    spectrum = torch.fft.fftn(field, dim=(-3, -2, -1))
    longitudinal = (spectrum * modes).sum(dim=1, keepdim=True)
    spectrum = (spectrum - modes * longitudinal / squared_norm.clamp_min(1)) * mask
    field = torch.fft.ifftn(spectrum, dim=(-3, -2, -1)).real
    current_rms = field.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
    return field * (rms / current_rms)


def wave_forcing(grid_size, device="cpu", dtype=torch.float64):
    _validate_grid(grid_size, 1, dtype, minimum=7)
    x, y, z = _coordinates(grid_size, device, dtype)
    wave = torch.sin(2 * math.pi * (2 * x + 3 * y + z))
    direction = torch.tensor([1, 0, -2], device=device, dtype=dtype) * math.sqrt(6 / 5)
    return direction[None, :, None, None, None] * wave[None, None]


def exact_wave_trajectory(initial, times, viscosity, forcing=None):
    _validate_scale(viscosity, "viscosity")
    if initial.ndim != 5 or initial.shape[1] != 3:
        raise ValueError("initial must have shape [batch, 3, x, y, z]")
    if initial.dtype not in (torch.float32, torch.float64):
        raise TypeError("initial must use torch.float32 or torch.float64")
    times = torch.as_tensor(times, dtype=initial.dtype, device=initial.device)
    if times.ndim != 1 or times.numel() == 0 or not torch.isfinite(times).all():
        raise ValueError("times must be a nonempty finite one-dimensional sequence")
    if (times[1:] <= times[:-1]).any():
        raise ValueError("times must be strictly increasing")
    rate = 56 * math.pi**2 * viscosity
    if forcing is None:
        decay = torch.exp(-rate * (times - times[0]))
        return initial[:, None] * decay[None, :, None, None, None, None]
    forcing = torch.as_tensor(forcing, dtype=initial.dtype, device=initial.device)
    expected_shape = (initial.shape[0], times.numel() - 1, *initial.shape[1:])
    if forcing.shape != expected_shape:
        raise ValueError("forcing must have shape [batch, time_intervals, 3, x, y, z]")
    if not torch.isfinite(forcing).all():
        raise ValueError("forcing must contain finite values")
    states = [initial]
    for index, dt in enumerate(times[1:] - times[:-1]):
        decay = torch.exp(-rate * dt)
        weight = -torch.expm1(-rate * dt) / rate if rate > 0 else dt
        states.append(decay * states[-1] + weight * forcing[:, index])
    return torch.stack(states, dim=1)
