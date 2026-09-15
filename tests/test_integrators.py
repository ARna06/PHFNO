import pytest
import torch

from phfno.integrators import energy_gradient, euler_step


def test_energy_gradient_and_training_derivatives():
    z = torch.tensor([[0.3, -0.7], [0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    energy = lambda state: scale * state.pow(4).sum(dim=-1) / 4
    effort = energy_gradient(energy, z)
    torch.testing.assert_close(effort, scale * z.pow(3))
    dz, ds = torch.autograd.grad(effort.square().sum(), (z, scale))
    torch.testing.assert_close(dz, 6 * scale.square() * z.pow(5))
    torch.testing.assert_close(ds, 2 * scale * z.pow(6).sum())


def test_energy_gradient_in_evaluation_and_constant_energy():
    z = torch.randn(2, 4)
    with torch.no_grad():
        effort = energy_gradient(lambda state: 0.5 * state.square().sum(dim=-1), z)
    torch.testing.assert_close(effort, z)
    assert not effort.requires_grad
    assert not z.requires_grad
    torch.testing.assert_close(
        energy_gradient(lambda state: state.new_ones(state.shape[0]), z), torch.zeros_like(z)
    )
    with torch.inference_mode(), pytest.raises(RuntimeError, match="autograd"):
        energy_gradient(lambda state: state.sum(dim=-1), z)


def test_euler_matches_analytic_update_and_validates_dt():
    z = torch.tensor([[1., -2.]])
    rhs = lambda state, control: -state + control
    control = torch.tensor([[0.5, 0.1]])
    torch.testing.assert_close(euler_step(rhs, z, control, 0.1), z + 0.1 * (-z + control))
    for bad_dt in [0, -1, float("nan"), [0.1, 0.2]]:
        with pytest.raises(ValueError, match="dt"):
            euler_step(rhs, z, control, bad_dt)
