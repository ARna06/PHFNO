import math

import pytest
import torch

from nsdata.solver import PeriodicNavierStokes


def plane_wave(grid_size=8, amplitude=0.2):
    coordinates = torch.arange(grid_size, dtype=torch.float64) / grid_size
    x, y, _ = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    wave = amplitude * torch.sin(2 * math.pi * (x + y))
    return torch.stack((wave, -wave, torch.zeros_like(wave)))[None]


def test_projection_is_divergence_free_idempotent_and_preserves_mean():
    solver = PeriodicNavierStokes(8, viscosity=0.01)
    generator = torch.Generator().manual_seed(3)
    field = torch.randn(2, 3, 8, 8, 8, generator=generator, dtype=torch.float64)
    projected = solver.project(field)
    torch.testing.assert_close(solver.divergence(projected), torch.zeros(2, 8, 8, 8, dtype=torch.float64))
    torch.testing.assert_close(solver.project(projected), projected)
    torch.testing.assert_close(projected.mean(dim=(-3, -2, -1)), field.mean(dim=(-3, -2, -1)))


def test_dealiasing_excludes_two_thirds_boundary_and_nyquist():
    solver = PeriodicNavierStokes(12, viscosity=0)
    x = torch.arange(12, dtype=torch.float64) / 12
    field = torch.zeros(1, 3, 12, 12, 12, dtype=torch.float64)
    field[:, 1] = torch.cos(8 * math.pi * x)[:, None, None]
    torch.testing.assert_close(solver.project(field), torch.zeros_like(field), atol=1e-14, rtol=0)
    torch.testing.assert_close(solver.project(field, dealias=False), field)
    field[:, 1] = torch.cos(12 * math.pi * x)[:, None, None]
    torch.testing.assert_close(solver.project(field, dealias=False), torch.zeros_like(field), atol=1e-14, rtol=0)


def test_exact_plane_wave_decay_and_irregular_snapshot_times():
    viscosity = 0.03
    solver = PeriodicNavierStokes(8, viscosity=viscosity)
    initial = plane_wave()
    times = torch.tensor([0.0, 0.017, 0.043, 0.1], dtype=torch.float64)
    trajectory = solver.solve(initial, times, max_dt=0.003)
    decay = torch.exp(-8 * math.pi**2 * viscosity * times)
    expected = initial[:, None] * decay[None, :, None, None, None, None]
    assert trajectory.shape == (1, 4, 3, 8, 8, 8)
    torch.testing.assert_close(trajectory, expected, rtol=1e-10, atol=1e-12)


def test_constant_velocity_and_single_snapshot():
    solver = PeriodicNavierStokes(8, viscosity=0.1)
    initial = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)[None, :, None, None, None]
    initial = initial.expand(2, 3, 8, 8, 8).contiguous()
    torch.testing.assert_close(solver.solve(initial, [0]), initial[:, None])
    trajectory = solver.solve(initial, [0, 0.01, 0.03])
    torch.testing.assert_close(trajectory, initial[:, None].expand_as(trajectory))


@pytest.mark.parametrize("viscosity", [0.0, 0.015])
def test_nonlinear_rhs_obeys_energy_balance(viscosity):
    solver = PeriodicNavierStokes(8, viscosity=viscosity)
    generator = torch.Generator().manual_seed(8)
    field = solver.project(torch.randn(1, 3, 8, 8, 8, generator=generator, dtype=torch.float64))
    spectrum = torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")
    derivative = solver.rhs(spectrum)
    nonlinear = derivative + viscosity * solver.k2 * spectrum
    assert nonlinear.abs().amax() > 1e-3
    energy_derivative = (spectrum.conj() * derivative).real.sum()
    expected = -viscosity * (solver.k2 * spectrum.abs().square()).sum()
    torch.testing.assert_close(energy_derivative, expected, atol=1e-12, rtol=1e-12)


def test_rk4_error_decreases_with_time_step():
    solver = PeriodicNavierStokes(8, viscosity=0.03)
    initial = plane_wave()
    exact = initial * math.exp(-8 * math.pi**2 * 0.03 * 0.1)
    coarse = solver.solve(initial, [0, 0.1], max_dt=0.025)[:, -1]
    fine = solver.solve(initial, [0, 0.1], max_dt=0.0125)[:, -1]
    coarse_error = torch.linalg.vector_norm(coarse - exact)
    fine_error = torch.linalg.vector_norm(fine - exact)
    assert coarse_error > 12 * fine_error


def test_taylor_green_initial_acceleration():
    solver = PeriodicNavierStokes(8, viscosity=0)
    angles = 2 * math.pi * torch.arange(8, dtype=torch.float64) / 8
    x, y, z = torch.meshgrid(angles, angles, angles, indexing="ij")
    initial = torch.stack((x.sin() * y.cos() * z.cos(), -x.cos() * y.sin() * z.cos(), torch.zeros_like(x)))[None]
    spectrum = torch.fft.fftn(initial, dim=(-3, -2, -1), norm="forward")
    acceleration = torch.fft.ifftn(solver.rhs(spectrum), dim=(-3, -2, -1), norm="forward").real
    expected = math.pi / 4 * torch.stack((
        -(2 * x).sin() * (2 * z).cos(),
        -(2 * y).sin() * (2 * z).cos(),
        ((2 * x).cos() + (2 * y).cos()) * (2 * z).sin(),
    ))[None]
    torch.testing.assert_close(acceleration, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu():
    initial = plane_wave()
    forcing = 0.3 * initial[:, None]
    cpu = PeriodicNavierStokes(8, viscosity=0.03).solve(initial, [0, 0.01], forcing=forcing)
    cuda = PeriodicNavierStokes(8, viscosity=0.03, device="cuda").solve(
        initial.cuda(), [0, 0.01], forcing=forcing.cuda()
    )
    torch.testing.assert_close(cuda.cpu(), cpu, atol=1e-12, rtol=1e-12)


def test_float32_nonlinear_evolution_preserves_divergence_and_mean():
    solver = PeriodicNavierStokes(8, viscosity=0.02, dtype=torch.float32)
    generator = torch.Generator().manual_seed(12)
    initial = solver.project(0.1 * torch.randn(1, 3, 8, 8, 8, generator=generator))
    trajectory = solver.solve(initial, [0, 0.02])
    final = trajectory[:, -1]
    assert torch.linalg.vector_norm(final - initial) > 1e-3
    torch.testing.assert_close(solver.divergence(final), torch.zeros(1, 8, 8, 8), atol=2e-6, rtol=0)
    torch.testing.assert_close(final.mean(dim=(-3, -2, -1)), initial.mean(dim=(-3, -2, -1)), atol=1e-8, rtol=1e-5)
    assert final.square().sum() < initial.square().sum()


@pytest.mark.parametrize("times", [[0, 0], [0.1, 0.2], [0, -0.1], [0, math.nan], [], [[0, 1]]])
def test_invalid_times(times):
    solver = PeriodicNavierStokes(8, viscosity=0.01)
    with pytest.raises(ValueError, match="times"):
        solver.solve(plane_wave(), times)


def test_invalid_parameters_and_fields():
    with pytest.raises(ValueError, match="grid_size"):
        PeriodicNavierStokes(3, viscosity=0.01)
    with pytest.raises(ValueError, match="viscosity"):
        PeriodicNavierStokes(8, viscosity=-0.01)
    with pytest.raises(TypeError, match="dtype"):
        PeriodicNavierStokes(8, viscosity=0.01, dtype=torch.float16)
    solver = PeriodicNavierStokes(8, viscosity=0.01)
    with pytest.raises(ValueError, match="shape"):
        solver.project(torch.zeros(3, 8, 8, 8, dtype=torch.float64))
    with pytest.raises(TypeError, match="dtype"):
        solver.project(plane_wave().float())
    with pytest.raises(ValueError, match="finite"):
        solver.solve(plane_wave() * math.nan, [0, 1])
    with pytest.raises(ValueError, match="max_dt"):
        solver.solve(plane_wave(), [0, 1], max_dt=0)
    with pytest.raises(ValueError, match="cfl"):
        solver.solve(plane_wave(), [0, 1], cfl=1.1)


def test_piecewise_uniform_forcing_drives_mean_velocity():
    solver = PeriodicNavierStokes(8, viscosity=0.03)
    initial = torch.zeros(2, 3, 8, 8, 8, dtype=torch.float64)
    controls = torch.tensor(
        [[[0.3, -0.2, 0.1], [-0.4, 0.2, 0.3]], [[0.1, 0.2, 0.3], [0.4, 0.1, -0.2]]],
        dtype=torch.float64,
    )
    forcing = controls[..., None, None, None].expand(2, 2, 3, 8, 8, 8).contiguous()
    times = torch.tensor([0.0, 0.02, 0.05], dtype=torch.float64)
    trajectory = solver.solve(initial, times, forcing=forcing)
    expected_mean = torch.zeros(2, 3, 3, dtype=torch.float64)
    expected_mean[:, 1:] = (controls * times.diff()[None, :, None]).cumsum(dim=1)
    torch.testing.assert_close(
        trajectory, expected_mean[..., None, None, None].expand_as(trajectory), atol=1e-14, rtol=1e-13
    )


@pytest.mark.parametrize("viscosity", [0.0, 0.03])
def test_piecewise_forced_plane_wave_matches_exact_solution(viscosity):
    solver = PeriodicNavierStokes(8, viscosity=viscosity)
    initial = plane_wave()
    controls = [0.4, -0.25]
    times = [0.0, 0.04, 0.1]
    forcing = torch.stack([control * initial for control in controls], dim=1)
    trajectory = solver.solve(initial, times, max_dt=0.003, forcing=forcing)
    amplitudes = [1.0]
    rate = 8 * math.pi**2 * viscosity
    for start, end, control in zip(times, times[1:], controls):
        dt = end - start
        response = -math.expm1(-rate * dt) / rate if rate else dt
        amplitudes.append(amplitudes[-1] * math.exp(-rate * dt) + control * response)
    expected = initial[:, None] * torch.tensor(amplitudes, dtype=torch.float64)[None, :, None, None, None, None]
    torch.testing.assert_close(trajectory, expected, rtol=1e-10, atol=1e-12)


def test_forced_rhs_obeys_energy_balance_and_viscous_term():
    solver = PeriodicNavierStokes(8, viscosity=0.02)
    generator = torch.Generator().manual_seed(31)
    field = solver.project(torch.randn(1, 3, 8, 8, 8, generator=generator, dtype=torch.float64))
    forcing = torch.randn(1, 3, 8, 8, 8, generator=generator, dtype=torch.float64)
    spectrum = torch.fft.fftn(field, dim=(-3, -2, -1), norm="forward")
    forcing_spectrum = torch.fft.fftn(forcing, dim=(-3, -2, -1), norm="forward")
    derivative = solver.rhs(spectrum, forcing_spectrum)
    energy_derivative = (spectrum.conj() * derivative).real.sum()
    viscous_power = (field * solver.viscous_term(field)).sum(dim=1).mean()
    forcing_power = (field * forcing).sum(dim=1).mean()
    expected_dissipation = solver.viscosity * (solver.k2 * spectrum.abs().square()).sum()
    torch.testing.assert_close(viscous_power, -expected_dissipation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(energy_derivative, viscous_power + forcing_power, atol=1e-12, rtol=1e-12)


def test_viscous_term_plane_wave():
    solver = PeriodicNavierStokes(8, viscosity=0.03)
    field = plane_wave()
    torch.testing.assert_close(solver.viscous_term(field), -8 * math.pi**2 * 0.03 * field)


def test_zero_forcing_matches_unforced_solution():
    solver = PeriodicNavierStokes(8, viscosity=0.03)
    initial = plane_wave()
    unforced = solver.solve(initial, [0, 0.01])
    forced = solver.solve(initial, [0, 0.01], forcing=torch.zeros_like(initial[:, None]))
    torch.testing.assert_close(forced, unforced, atol=0, rtol=0)


def test_invalid_forcing():
    solver = PeriodicNavierStokes(8, viscosity=0.03)
    initial = plane_wave()
    with pytest.raises(ValueError, match="forcing must have shape"):
        solver.solve(initial, [0, 0.1], forcing=initial)
    with pytest.raises(TypeError, match="forcing dtype"):
        solver.solve(initial, [0, 0.1], forcing=initial[:, None].float())
    with pytest.raises(ValueError, match="forcing must be finite"):
        solver.solve(initial, [0, 0.1], forcing=initial[:, None] * math.nan)
    spectrum = torch.fft.fftn(initial, dim=(-3, -2, -1), norm="forward")
    with pytest.raises(ValueError, match="same shape"):
        solver.rhs(spectrum, spectrum.expand(2, 3, 8, 8, 8))
