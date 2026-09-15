import math

import pytest
import torch

from experiments.vorticity_operators import PeriodicVorticity
from nsdata.solver import PeriodicNavierStokes


def coordinates(grid_size):
    angles = 2 * math.pi * torch.arange(grid_size, dtype=torch.float64) / grid_size
    return torch.meshgrid(angles, angles, angles, indexing="ij")


def smooth_velocity(grid_size=12):
    x, y, z = coordinates(grid_size)
    return torch.stack((x.sin() * y.cos() * z.cos(), -x.cos() * y.sin() * z.cos(), torch.zeros_like(x)))


def inner(a, b):
    return (a * b).sum(dim=-4).mean(dim=(-3, -2, -1))


def test_analytic_curl_and_velocity_roundtrip_with_mean():
    operators = PeriodicVorticity(12)
    x, y, z = coordinates(12)
    phase = x + 2 * y + z
    velocity = torch.stack((phase.sin(), torch.zeros_like(phase), -phase.sin()))
    expected = 2 * math.pi * torch.stack((-2 * phase.cos(), 2 * phase.cos(), -2 * phase.cos()))
    velocity = velocity.expand(2, 4, -1, -1, -1, -1).clone()
    mean = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 100
    velocity += mean[..., None, None, None]
    omega = operators.curl(velocity)
    torch.testing.assert_close(omega, expected.expand_as(omega), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(operators.velocity(omega, mean), velocity, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(operators.velocity(omega).mean(dim=(-3, -2, -1)), torch.zeros_like(mean), atol=1e-14, rtol=0)


def test_gradient_has_zero_curl_and_curl_has_zero_divergence():
    operators = PeriodicVorticity(12)
    x, y, z = coordinates(12)
    gradient = 2 * math.pi * torch.stack((x.cos() * y.sin(), x.sin() * y.cos(), torch.zeros_like(z)))
    torch.testing.assert_close(operators.curl(gradient), torch.zeros_like(gradient), atol=1e-12, rtol=0)
    omega = operators.curl(smooth_velocity())
    torch.testing.assert_close(operators.divergence(omega), torch.zeros_like(x), atol=1e-12, rtol=0)


def test_nyquist_planes_are_removed_without_removing_resolved_modes():
    operators = PeriodicVorticity(8)
    x, _, _ = coordinates(8)
    field = torch.stack((torch.zeros_like(x), (4 * x).cos(), torch.zeros_like(x)))
    for transform in (operators.curl, operators.velocity, operators.inverse_laplacian, operators.project):
        torch.testing.assert_close(transform(field), torch.zeros_like(field), atol=1e-13, rtol=0)
    field[1] = (3 * x).sin()
    expected = torch.zeros_like(field)
    expected[2] = 6 * math.pi * (3 * x).cos()
    torch.testing.assert_close(operators.curl(field), expected, atol=1e-12, rtol=1e-12)


def test_odd_grid_keeps_highest_frequency():
    operators = PeriodicVorticity(9)
    x, _, _ = coordinates(9)
    field = torch.stack((torch.zeros_like(x), (4 * x).sin(), torch.zeros_like(x)))
    expected = torch.zeros_like(field)
    expected[2] = 8 * math.pi * (4 * x).cos()
    torch.testing.assert_close(operators.curl(field), expected, atol=1e-12, rtol=1e-12)


def test_energy_matches_kinetic_energy_and_autodiff_effort():
    operators = PeriodicVorticity(12)
    velocity = smooth_velocity().expand(2, -1, -1, -1, -1)
    omega = operators.curl(velocity).requires_grad_()
    energy = operators.kinetic_energy(omega)
    torch.testing.assert_close(energy, 0.5 * inner(velocity, velocity), atol=1e-13, rtol=1e-13)
    effort = torch.autograd.grad(energy.sum(), omega)[0] * 12**3
    torch.testing.assert_close(effort, operators.inverse_laplacian(omega), atol=1e-13, rtol=1e-13)


def test_projection_exposes_unobservable_vorticity_components():
    operators = PeriodicVorticity(12)
    x, _, _ = coordinates(12)
    omega = operators.curl(smooth_velocity())
    invalid = omega + torch.stack((x.cos(), torch.zeros_like(x), torch.ones_like(x)))
    torch.testing.assert_close(operators.velocity(invalid), operators.velocity(omega), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(operators.project(invalid), omega, atol=1e-12, rtol=1e-12)
    assert (invalid - operators.curl(operators.velocity(invalid))).square().mean() > 0.1


def test_j_is_skew_on_resolved_fields_and_r_is_positive():
    operators = PeriodicVorticity(12)
    generator = torch.Generator().manual_seed(14)
    fields = torch.randn(3, 3, 12, 12, 12, generator=generator, dtype=torch.float64)
    spectrum = torch.fft.fftn(fields, dim=(-3, -2, -1), norm="forward")
    low_modes = (operators.k.abs() <= 4 * math.pi).all(dim=0)
    a, b, omega = torch.fft.ifftn(spectrum * low_modes, dim=(-3, -2, -1), norm="forward").real
    skew = inner(a, operators.apply_j(omega, b)) + inner(operators.apply_j(omega, a), b)
    torch.testing.assert_close(skew, torch.zeros_like(skew), atol=1e-12, rtol=0)
    assert inner(a, operators.apply_r(a, 0.02)) > 0
    torch.testing.assert_close(operators.apply_r(a, 0), torch.zeros_like(a), atol=0, rtol=0)


def test_j_is_skew_for_arbitrary_grid_fields():
    operators = PeriodicVorticity(12)
    generator = torch.Generator().manual_seed(19)
    a, b, omega = torch.randn(3, 2, 3, 12, 12, 12, generator=generator, dtype=torch.float64)
    skew = inner(a, operators.apply_j(omega, b)) + inner(operators.apply_j(omega, a), b)
    torch.testing.assert_close(skew, torch.zeros_like(skew), atol=1e-12, rtol=0)


@pytest.mark.parametrize("with_mean", [False, True])
def test_reference_matches_curl_of_forced_navier_stokes_rhs(with_mean):
    operators = PeriodicVorticity(12)
    solver = PeriodicNavierStokes(12, viscosity=0.02)
    velocity = smooth_velocity()[None]
    mean = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    if with_mean:
        velocity = velocity + mean[..., None, None, None]
    x, y, z = coordinates(12)
    forcing = torch.stack(((2 * y).sin(), z.cos(), x.sin()))[None]
    spectrum = torch.fft.fftn(velocity, dim=(-3, -2, -1), norm="forward")
    forcing_spectrum = torch.fft.fftn(forcing, dim=(-3, -2, -1), norm="forward")
    velocity_rhs = torch.fft.ifftn(solver.rhs(spectrum, forcing_spectrum), dim=(-3, -2, -1), norm="forward").real
    omega = operators.curl(velocity)
    reference = operators.reference_rhs(omega, forcing, 0.02, mean if with_mean else None)
    torch.testing.assert_close(reference, operators.curl(velocity_rhs), atol=1e-11, rtol=1e-11)
    if not with_mean:
        phi = operators.inverse_laplacian(omega)
        assembled = operators.apply_j(omega, phi) - operators.apply_r(phi, 0.02) + operators.curl(forcing)
        torch.testing.assert_close(assembled, reference, atol=1e-11, rtol=1e-11)
    derivatives = torch.stack([
        torch.fft.ifftn(1j * operators.k[axis] * spectrum, dim=(-3, -2, -1), norm="forward").real
        for axis in range(3)
    ], dim=1)
    stretching = (omega[:, :, None] * derivatives).sum(dim=1)
    assert stretching.square().mean().sqrt() > 0.1


def test_reference_obeys_forced_kinetic_energy_balance():
    operators = PeriodicVorticity(12)
    velocity = smooth_velocity()
    omega = operators.curl(velocity)
    phi = operators.inverse_laplacian(omega)
    forcing = 0.3 * velocity
    rate = operators.reference_rhs(omega, forcing, viscosity=0.02)
    expected = -inner(phi, operators.apply_r(phi, 0.02)) + inner(velocity, forcing)
    torch.testing.assert_close(inner(phi, rate), expected, atol=1e-12, rtol=1e-12)


def test_module_to_dtype_and_inverse_curl_backward():
    operators = PeriodicVorticity(8).float()
    field = smooth_velocity(8).float().requires_grad_()
    reconstructed = operators.velocity(operators.curl(field))
    reconstructed.square().mean().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert field.grad.abs().max() > 0
    torch.testing.assert_close(reconstructed, field, atol=3e-7, rtol=3e-6)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="grid_size"):
        PeriodicVorticity(3)
    with pytest.raises(TypeError, match="dtype"):
        PeriodicVorticity(8, dtype=torch.float16)
    operators = PeriodicVorticity(8)
    field = smooth_velocity(8)
    with pytest.raises(ValueError, match="shape"):
        operators.curl(field[0])
    with pytest.raises(TypeError, match="dtype"):
        operators.curl(field.float())
    with pytest.raises(ValueError, match="mean"):
        operators.velocity(field, torch.zeros(2))
    with pytest.raises(ValueError, match="viscosity"):
        operators.apply_r(field, -0.1)
    with pytest.raises(ValueError, match="same shape"):
        operators.apply_j(field, field[None])
    with pytest.raises(ValueError, match="same shape"):
        operators.reference_rhs(field, forcing=field[None])
