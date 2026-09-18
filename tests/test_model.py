import pytest
import torch

from phfno import FNOBaseline, PHFNO


def small_model(cutoff=(1,), state_channels=1, control_channels=1, parameter_grid=(8,)):
    torch.manual_seed(31)
    return PHFNO(
        cutoff, state_channels, control_channels, hidden_channels=4,
        n_layers=1, mlp_width=8, parameter_grid=parameter_grid,
    )


def assert_trainable_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_structure_skew_dissipation_and_port_power():
    model = small_model()
    z = 0.2 * torch.randn(2, model.coordinates.coordinate_dim)
    p, q = torch.randn_like(z), torch.randn_like(z)
    u = torch.tensor([[0.3], [-0.2]])
    structure = model.structure(z)
    skew_pairing = (p * structure.apply_j(q)).sum(-1) + (q * structure.apply_j(p)).sum(-1)
    torch.testing.assert_close(skew_pairing, torch.zeros_like(skew_pairing), atol=1e-7, rtol=0)
    assert torch.all((q * structure.apply_r(q)).sum(-1) >= 0)
    effort = model.effort(z)
    power = (effort * model.rhs_coordinates(z, u)).sum(-1)
    expected = -(effort * structure.apply_r(effort)).sum(-1) + (structure.output(effort) * u).sum(-1)
    torch.testing.assert_close(power, expected, atol=1e-7, rtol=1e-5)
    control_effect = model.rhs_coordinates(z, u) - model.rhs_coordinates(z)
    torch.testing.assert_close(control_effect, structure.apply_b(u), atol=1e-7, rtol=1e-5)
    assert control_effect.abs().sum() > 0


def test_backward_reaches_every_network_and_control():
    model = small_model()
    field = torch.randn(2, 1, 8, requires_grad=True)
    control = torch.tensor([[0.4], [-0.3]], requires_grad=True)
    prediction = model(field, control)
    prediction.square().mean().backward()
    for module in (model.factor_net, model.energy_net, model.damping_net):
        assert_trainable_gradients(module)
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert control.grad is not None and torch.isfinite(control.grad).all()
    assert control.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "cutoff, channels, controls, grid", [((1,), 1, 0, (8,)), ((1, 2), 2, 2, (5, 7))]
)
def test_field_shapes_no_grad_and_rectangular_two_dimensions(cutoff, channels, controls, grid):
    model = small_model(cutoff, channels, controls, grid)
    field = torch.randn(2, channels, *grid)
    control = torch.randn(2, controls)
    prediction = model(field, control)
    assert prediction.shape == field.shape
    with torch.no_grad():
        inferred = model(field, control)
        stepped = model.step(field, control, 0.01)
    torch.testing.assert_close(inferred, prediction)
    assert not inferred.requires_grad and not stepped.requires_grad
    assert torch.isfinite(stepped).all()
    prediction.square().mean().backward()
    for module in (model.factor_net, model.energy_net, model.damping_net):
        assert_trainable_gradients(module)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        model(field, control)


def test_fixed_parameter_grid_preserves_coordinate_dynamics_across_resolutions():
    model = small_model()
    z = torch.randn(2, model.coordinates.coordinate_dim)
    u = torch.tensor([[0.2], [-0.4]])
    expected = model.rhs_coordinates(z, u)
    for grid in ((5,), (13,)):
        field = model.coordinates.decode(z, grid)
        actual = model.coordinates.encode(model(field, u))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_euler_rollout_carries_gradients_through_all_steps():
    model = small_model()
    initial = torch.randn(2, 1, 8, requires_grad=True)
    controls = torch.randn(2, 3, 1, requires_grad=True)
    times = torch.tensor([0.0, 0.01, 0.03, 0.05])
    states = model.rollout(initial, controls, times)
    assert states.shape == (2, 4, 1, 8)
    torch.testing.assert_close(states[:, 0], model.coordinates.project(initial))
    states[:, -1].square().mean().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert controls.grad is not None and torch.isfinite(controls.grad).all()
    assert torch.all(controls.grad.abs().sum(dim=(0, 2)) > 0)
    assert_trainable_gradients(model.energy_net)


def test_baseline_matches_derivative_step_and_rollout_interfaces():
    torch.manual_seed(32)
    model = FNOBaseline((1,), 1, 1, hidden_channels=4, n_layers=1)
    field = torch.randn(2, 1, 8)
    u = torch.tensor([[0.2], [-0.5]])
    derivative = model(field, u)
    assert derivative.shape == field.shape
    projected = model.coordinates.project(field)
    torch.testing.assert_close(model.step(field, u, 0.01), projected + 0.01 * derivative)
    controls = u[:, None].expand(-1, 2, -1)
    states = model.rollout(field, controls, torch.tensor([0.0, 0.01, 0.02]))
    assert states.shape == (2, 3, 1, 8)
    states[:, -1].square().mean().backward()
    assert_trainable_gradients(model.operator)
    with torch.no_grad():
        assert not model(field, u).requires_grad
    # The baseline now supports AVF as well as Euler; reject only unknown methods.
    with pytest.raises(ValueError, match="'avf' or 'euler'"):
        model.step(field, u, 0.01, method="unsupported")


@pytest.mark.parametrize("kind", [PHFNO, FNOBaseline])
def test_avf_rollout_preserves_gradients_and_solver_options(kind):
    # Exercise actual networks and multiple steps, including PhFNO's energy Hessian.
    torch.manual_seed(31)
    options = dict(cutoff=(1,), state_channels=1, control_channels=1,
                   hidden_channels=4, n_layers=1)
    if kind is PHFNO:
        options.update(mlp_width=8, parameter_grid=(8,))
    model = kind(**options)
    initial = torch.randn(2, 1, 8, requires_grad=True)
    controls = torch.randn(2, 2, 1, requires_grad=True)
    times = torch.tensor([0.0, 0.01, 0.03])
    solver_options = {"max_iterations": 64, "quadrature_points": 2}
    trajectory = model.rollout(initial, controls, times, method="avf",
                               solver_options=solver_options)
    with torch.no_grad():
        expected = model.rollout(initial, controls, times, method="avf",
                                 solver_options=solver_options)
    torch.testing.assert_close(trajectory, expected)
    trajectory[:, -1].square().mean().backward()
    assert_trainable_gradients(model)
    assert torch.isfinite(initial.grad).all()
    assert torch.isfinite(controls.grad).all()
    assert (controls.grad.abs().sum(dim=(0, 2)) > 0).all()
    # An invalid solver setting must reach the integrator rather than be ignored.
    with pytest.raises(ValueError, match="max_iterations"):
        model.step(initial, controls[:, 0], 0.01, method="avf",
                   solver_options={"max_iterations": 0})


def test_baseline_avf_has_the_same_coordinate_norm_across_grids(monkeypatch):
    # For a resolution-independent derivative, the same Fourier state must take
    # the same numerical step and satisfy the same absolute tolerance on each grid.
    model = FNOBaseline((0,), 2, hidden_channels=2, n_layers=1)

    def rhs(field, control):
        result = torch.zeros_like(field)
        result[:, 1] = field[:, 0] - 4 * field[:, 1]
        return result

    monkeypatch.setattr(model, "forward", rhs)
    z = torch.tensor([[1.0, 0.0]])
    results = []
    with torch.no_grad():
        for grid in ((4,), (32,)):
            field = model.coordinates.decode(z, grid)
            results.append(model.coordinates.encode(model.step(field, None, 0.1, method="avf")))
    torch.testing.assert_close(results[0], results[1], atol=1e-7, rtol=0)
