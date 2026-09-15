import pytest
import torch
from torch import nn

from phfno import PHFNO
from phfno.integrators import energy_gradient, gonzalez_gradient, gonzalez_step
from phfno.model import PHStructure


def factors_for(z, damping=0.0):
    return PHStructure(
        a=z.new_tensor([2.0, 0.0]).expand_as(z),
        b=z.new_tensor([0.0, 1.0]).expand_as(z),
        d=z.new_full((len(z),), damping**0.5),
        B=torch.eye(2, dtype=z.dtype, device=z.device).expand(len(z), -1, -1),
    )


def quartic_energy(z):
    return (z.square() / 2 + z.pow(4) / 4).sum(dim=-1)


def test_discrete_gradient_obeys_chain_rule_and_endpoint_symmetry():
    z = torch.tensor([[0.2, -0.7], [0.9, 0.1]], dtype=torch.float64)
    next_z = torch.tensor([[0.6, -0.4], [-0.2, 0.3]], dtype=torch.float64)
    effort = gonzalez_gradient(quartic_energy, z, next_z)
    torch.testing.assert_close((effort * (next_z - z)).sum(-1),
                               quartic_energy(next_z) - quartic_energy(z))
    torch.testing.assert_close(effort, gonzalez_gradient(quartic_energy, next_z, z))
    assert not torch.allclose(effort, energy_gradient(quartic_energy, (z + next_z) / 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coincident_and_tiny_increments_have_finite_training_gradients(dtype):
    z = torch.tensor([[0.2, -0.7]], dtype=dtype, requires_grad=True)
    next_z = (z.detach() + torch.finfo(dtype).eps).requires_grad_()
    effort = gonzalez_gradient(quartic_energy, z, next_z)
    midpoint = (z + next_z) / 2
    torch.testing.assert_close(effort, midpoint + midpoint.pow(3))
    gradients = torch.autograd.grad(effort.square().sum(), (z, next_z))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    same = gonzalez_gradient(quartic_energy, z, z)
    torch.testing.assert_close(same, z + z.pow(3))
    derivative = torch.autograd.grad(same.sum(), z)[0]
    torch.testing.assert_close(derivative, 1 + 3 * z.square())


@pytest.mark.parametrize("damping, forced", [(0.0, False), (0.3, False), (0.3, True)])
def test_step_satisfies_discrete_energy_balance(damping, forced):
    z = torch.tensor([[0.4, -0.7], [-0.2, 0.6]], dtype=torch.float64)
    factors = factors_for(z, damping)
    control = z.new_tensor([[0.2, -0.1], [0.4, 0.3]]) if forced else torch.zeros_like(z)
    dt = z.new_tensor([0.1, 0.04])
    with torch.no_grad():
        next_z = gonzalez_step(quartic_energy, factors, z, control, dt)
        effort = gonzalez_gradient(quartic_energy, z, next_z)
    power = -(effort * factors.apply_r(effort)).sum(-1) + (effort * control).sum(-1)
    torch.testing.assert_close(quartic_energy(next_z) - quartic_energy(z), dt * power,
                               atol=1e-13, rtol=1e-11)
    residual = next_z - z - dt[:, None] * (
        factors.apply_j(effort) - factors.apply_r(effort) + control
    )
    torch.testing.assert_close(residual, torch.zeros_like(z), atol=1e-13, rtol=0)
    assert not next_z.requires_grad
    if damping and not forced:
        assert (quartic_energy(next_z) < quartic_energy(z)).all()


def test_quadratic_step_matches_implicit_midpoint_with_batched_timesteps():
    z = torch.tensor([[0.4, -0.7], [-0.2, 0.6]], dtype=torch.float64)
    control = z.new_tensor([[0.2, -0.1], [0.4, 0.3]])
    factors = factors_for(z, damping=0.3)
    dt = z.new_tensor([0.1, 0.04])
    energy = lambda state: state.square().sum(-1) / 2
    next_z = gonzalez_step(energy, factors, z, control, dt)
    operator = z.new_tensor([[-0.3, 1.0], [-1.0, -0.3]])
    identity = torch.eye(2, dtype=z.dtype)
    lhs = identity - dt[:, None, None] * operator / 2
    rhs = z + dt[:, None] * (z @ operator.T / 2 + control)
    expected = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(next_z, expected, atol=1e-13, rtol=1e-11)
    separate = torch.cat([
        gonzalez_step(energy, factors_for(z[i:i + 1], 0.3), z[i:i + 1], control[i:i + 1], interval)
        for i, interval in enumerate(dt)
    ])
    torch.testing.assert_close(next_z, separate, atol=1e-13, rtol=1e-11)


def test_step_backprop_matches_finite_differences():
    z = torch.tensor([[0.4, -0.7]], dtype=torch.float64, requires_grad=True)
    stiffness = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    control = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)

    def step(state, weight, forcing):
        energy = lambda value: weight * quartic_energy(value)
        return gonzalez_step(energy, factors_for(state, 0.3), state, forcing, 0.1)

    assert torch.autograd.gradcheck(step, (z, stiffness, control))


def test_step_reports_failure_to_converge():
    z = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="did not converge.*3 iterations"):
        gonzalez_step(quartic_energy, factors_for(z, 50), z, torch.zeros_like(z),
                      1.0, max_iterations=3)


def test_small_iterate_difference_cannot_hide_a_large_equation_residual():
    z = torch.full((1, 2), 1e-16)
    factors = factors_for(z, damping=1e8)
    factors.a.zero_()
    factors.b.zero_()
    energy = lambda state: state.square().sum(-1) / 2
    with pytest.raises(RuntimeError, match="did not converge"):
        gonzalez_step(energy, factors, z, torch.zeros_like(z), 1.0, max_iterations=3)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), [0.1, 0.2, 0.3], [[0.1, 0.2]]])
def test_step_rejects_invalid_timesteps(dt):
    z = torch.ones(2, 2)
    with pytest.raises(ValueError, match="dt"):
        gonzalez_step(quartic_energy, factors_for(z), z, torch.zeros_like(z), dt)


def test_model_default_step_and_rollout_use_gonzalez(monkeypatch):
    class Energy(nn.Module):
        def forward(self, z):
            return quartic_energy(z).unsqueeze(-1)

    model = PHFNO((0,), 2, 2, hidden_channels=2, n_layers=1, mlp_width=4, parameter_grid=(4,))
    model.energy_net = Energy()
    z = torch.tensor([[0.4, -0.7]])
    field = model.coordinates.decode(z, (4,))
    control = torch.tensor([[0.2, -0.1]])
    sampled = []

    def structure(state):
        sampled.append(state.detach().clone())
        return factors_for(state, 0.3)

    monkeypatch.setattr(model, "structure", structure)
    with torch.no_grad():
        prediction = model.step(field, control, 0.1)
        assert len(sampled) == 1
        torch.testing.assert_close(sampled[0], z)
        expected = gonzalez_step(quartic_energy, factors_for(z, 0.3), z, control, 0.1)
        torch.testing.assert_close(model.coordinates.encode(prediction), expected)
        euler = model.step(field, control, 0.1, method="euler")
        assert not torch.allclose(prediction, euler)
        rollout = model.rollout(field, control[:, None], torch.tensor([0.0, 0.1]))
        torch.testing.assert_close(rollout[:, -1], prediction)


def test_default_model_step_trains_all_networks():
    torch.manual_seed(31)
    model = PHFNO((1,), 1, 1, hidden_channels=4, n_layers=1, mlp_width=8, parameter_grid=(8,))
    field = torch.randn(2, 1, 8, requires_grad=True)
    control = torch.tensor([[0.4], [-0.3]], requires_grad=True)
    prediction = model.step(field, control, 0.01)
    prediction.square().mean().backward()
    for module in (model.factor_net, model.energy_net, model.damping_net):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert control.grad is not None and torch.isfinite(control.grad).all()
