import math

import pytest
import torch

from nsdata.initial_conditions import (
    exact_wave_trajectory,
    plane_wave,
    random_velocity,
    taylor_green,
    wave_forcing,
)


def spectral_divergence(velocity):
    grid_size = velocity.shape[-1]
    frequencies = torch.fft.fftfreq(grid_size, d=1 / grid_size, dtype=velocity.dtype)
    modes = torch.stack(torch.meshgrid(frequencies, frequencies, frequencies, indexing="ij"))
    spectrum = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    divergence = 2j * math.pi * (spectrum * modes).sum(dim=1)
    return torch.fft.ifftn(divergence, dim=(-3, -2, -1))


@pytest.mark.parametrize("initializer", [plane_wave, taylor_green, random_velocity])
def test_initial_velocity_is_periodic_and_divergence_free(initializer):
    velocity = initializer(12, batch_size=2, seed=4)
    assert velocity.shape == (2, 3, 12, 12, 12)
    assert velocity.dtype == torch.float64
    assert torch.isfinite(velocity).all()
    torch.testing.assert_close(
        velocity.mean(dim=(-3, -2, -1)), torch.zeros(2, 3, dtype=velocity.dtype),
        atol=1e-14, rtol=0,
    )
    assert spectral_divergence(velocity).abs().max() < 1e-12


@pytest.mark.parametrize("initializer", [plane_wave, taylor_green, random_velocity])
def test_seeds_are_local_and_reproducible(initializer):
    rng_state = torch.random.get_rng_state().clone()
    first = initializer(10, batch_size=2, seed=18)
    second = initializer(10, batch_size=2, seed=18)
    other = initializer(10, batch_size=2, seed=19)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert not torch.equal(first[0], first[1])
    assert torch.equal(rng_state, torch.random.get_rng_state())


def test_plane_wave_has_expected_direction_and_energy():
    velocity = plane_wave(10, batch_size=2, amplitude=1.3)
    torch.testing.assert_close(velocity[:, 1], torch.zeros_like(velocity[:, 1]))
    torch.testing.assert_close(velocity[:, 2], -2 * velocity[:, 0])
    expected = torch.full((2,), 1.3**2 / 6, dtype=velocity.dtype)
    torch.testing.assert_close(velocity.square().mean(dim=(1, 2, 3, 4)), expected)


def test_plane_wave_has_zero_advection_and_expected_laplacian():
    velocity = plane_wave(10, amplitude=0.7)
    frequencies = torch.fft.fftfreq(10, d=1 / 10, dtype=velocity.dtype)
    modes = torch.stack(torch.meshgrid(frequencies, frequencies, frequencies, indexing="ij"))
    spectrum = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    gradient = torch.fft.ifftn(
        2j * math.pi * spectrum[:, :, None] * modes, dim=(-3, -2, -1)
    ).real
    advection = (velocity[:, None] * gradient).sum(dim=2)
    laplacian = torch.fft.ifftn(
        -4 * math.pi**2 * modes.square().sum(dim=0) * spectrum, dim=(-3, -2, -1)
    ).real
    torch.testing.assert_close(advection, torch.zeros_like(advection), atol=1e-13, rtol=0)
    torch.testing.assert_close(laplacian, -56 * math.pi**2 * velocity, atol=1e-11, rtol=1e-12)


def test_taylor_green_has_expected_energy_and_zero_vertical_velocity():
    velocity = taylor_green(8, batch_size=2, amplitude=0.7)
    torch.testing.assert_close(velocity[:, 2], torch.zeros_like(velocity[:, 2]))
    expected = torch.full((2,), 0.7**2 / 12, dtype=velocity.dtype)
    torch.testing.assert_close(velocity.square().mean(dim=(1, 2, 3, 4)), expected)


def test_random_velocity_has_requested_rms_and_only_low_frequencies():
    velocity = random_velocity(12, batch_size=3, rms=0.4, cutoff=2, seed=2)
    torch.testing.assert_close(
        velocity.square().mean(dim=(1, 2, 3, 4)).sqrt(),
        torch.full((3,), 0.4, dtype=velocity.dtype),
    )
    frequencies = torch.fft.fftfreq(12, d=1 / 12, dtype=velocity.dtype)
    modes = torch.stack(torch.meshgrid(frequencies, frequencies, frequencies, indexing="ij"))
    high_frequencies = (modes.abs() > 2).any(dim=0)
    spectrum = torch.fft.fftn(velocity, dim=(-3, -2, -1), norm="forward")
    assert spectrum[:, :, high_frequencies].abs().max() < 1e-14


def test_exact_wave_trajectory_matches_viscous_decay():
    velocity = plane_wave(10, batch_size=2)
    times = torch.tensor([0.0, 0.125, 1.0], dtype=velocity.dtype)
    viscosity = 0.01
    trajectory = exact_wave_trajectory(velocity, times, viscosity)
    assert trajectory.shape == (2, 3, 3, 10, 10, 10)
    torch.testing.assert_close(trajectory[:, 0], velocity)
    torch.testing.assert_close(trajectory[:, -1], velocity * math.exp(-56 * math.pi**2 * viscosity))
    energy = trajectory.square().mean(dim=(2, 3, 4, 5))
    expected = torch.exp(-112 * math.pi**2 * viscosity * times)
    torch.testing.assert_close(energy / energy[:, :1], expected.expand_as(energy))
    stationary = exact_wave_trajectory(velocity, times, viscosity=0)
    torch.testing.assert_close(stationary, velocity[:, None].expand_as(stationary))


def test_wave_forcing_is_divergence_free_and_has_unit_rms():
    forcing = wave_forcing(10)
    assert forcing.shape == (1, 3, 10, 10, 10)
    torch.testing.assert_close(forcing.square().mean(), torch.tensor(1.0, dtype=forcing.dtype))
    assert spectral_divergence(forcing).abs().max() < 1e-12
    torch.testing.assert_close(forcing[:, 2], -2 * forcing[:, 0])


def test_exact_wave_trajectory_with_different_interval_forcing():
    initial = plane_wave(10, batch_size=2)
    basis = wave_forcing(10).expand_as(initial)
    forcing = torch.stack((2 * basis, -basis), dim=1)
    trajectory = exact_wave_trajectory(initial, [0, 0.25, 0.75], 1 / (56 * math.pi**2), forcing)
    first = math.exp(-0.25) * initial + 2 * (1 - math.exp(-0.25)) * basis
    last = math.exp(-0.75) * initial + (
        2 * math.exp(-0.5) * (1 - math.exp(-0.25)) - (1 - math.exp(-0.5))
    ) * basis
    torch.testing.assert_close(trajectory[:, 0], initial)
    torch.testing.assert_close(trajectory[:, 1], first)
    torch.testing.assert_close(trajectory[:, 2], last)


@pytest.mark.parametrize("viscosity", [0, 1e-20])
def test_exact_wave_forcing_with_zero_or_tiny_viscosity(viscosity):
    initial = plane_wave(10)
    basis = wave_forcing(10)
    forcing = torch.stack((2 * basis, -basis), dim=1)
    trajectory = exact_wave_trajectory(initial, [0, 0.25, 1], viscosity, forcing)
    torch.testing.assert_close(trajectory[:, 1], initial + 0.5 * basis)
    torch.testing.assert_close(trajectory[:, 2], initial - 0.25 * basis)


def test_invalid_exact_wave_forcing():
    initial = plane_wave(10)
    with pytest.raises(ValueError, match="forcing must have shape"):
        exact_wave_trajectory(initial, [0, 1], 0.1, torch.zeros_like(initial))
    with pytest.raises(ValueError, match="finite values"):
        exact_wave_trajectory(initial, [0, 1], 0.1, torch.full_like(initial[:, None], float("nan")))


@pytest.mark.parametrize("initializer", [plane_wave, taylor_green, random_velocity])
def test_float32_initial_velocity(initializer):
    velocity = initializer(10, dtype=torch.float32)
    assert velocity.dtype == torch.float32
    assert spectral_divergence(velocity).abs().max() < 1e-4


@pytest.mark.parametrize("initializer, keyword", [(plane_wave, "amplitude"), (taylor_green, "amplitude"), (random_velocity, "rms")])
def test_zero_scale_gives_zero_velocity(initializer, keyword):
    velocity = initializer(10, **{keyword: 0})
    assert torch.count_nonzero(velocity) == 0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        initializer(10, **{keyword: -1})


def test_invalid_initial_condition_parameters():
    with pytest.raises(ValueError, match="at least 7"):
        plane_wave(6)
    with pytest.raises(ValueError, match="at least 3"):
        taylor_green(2)
    with pytest.raises(ValueError, match="strictly below"):
        random_velocity(6, cutoff=2)
    with pytest.raises(ValueError, match="strictly below"):
        random_velocity(10, cutoff=0)
    with pytest.raises(ValueError, match="positive integer"):
        plane_wave(10, batch_size=0)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        random_velocity(10, dtype=torch.float16)


def test_invalid_exact_trajectory_parameters():
    initial = plane_wave(10)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        exact_wave_trajectory(initial, [0, 1], viscosity=-0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        exact_wave_trajectory(initial, [0, 0], viscosity=0.1)
    with pytest.raises(ValueError, match="nonempty finite"):
        exact_wave_trajectory(initial, [], viscosity=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("initializer", [plane_wave, taylor_green, random_velocity])
def test_cuda_matches_cpu_initial_velocity(initializer):
    cpu = initializer(10, batch_size=2, seed=8)
    gpu = initializer(10, batch_size=2, seed=8, device="cuda")
    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-12, rtol=1e-12)
