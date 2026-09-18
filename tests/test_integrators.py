import pytest
import torch

from phfno.integrators import avf_step, energy_gradient, euler_step


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


def test_avf_converges_for_linear_decay_and_validates_quadrature():
    z = torch.tensor([[1.0]])
    result = avf_step(lambda state, control: -state, z, None, 0.1)
    torch.testing.assert_close(result, torch.tensor([[0.9047619]]), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="quadrature"):
        avf_step(lambda state, control: -state, z, None, 0.1, quadrature_points=3)


def test_avf_broadcasts_per_batch_steps_over_field_dimensions():
    field = torch.ones(3, 2, 4, 4, 4)
    dt = torch.tensor([0.1, 0.2, 0.3])
    result = avf_step(lambda state, control: -state, field, None, dt)
    assert result.shape == field.shape


def test_avf_rejects_overflow():
    # The audit's inf<=inf example must report solver failure.
    z = torch.ones(1, 1)
    with pytest.raises(RuntimeError, match="AVF step did not converge"):
        avf_step(lambda state, control: -1e20 * state, z, None, 1.0)


def test_avf_tiny_state_returns_only_an_endpoint_within_tolerance():
    # A tiny endpoint may be within absolute tolerance, but returning the next
    # unchecked candidate used to amplify its equation residual by a factor 5000.
    z = torch.tensor([[1e-16]], dtype=torch.float64)
    result = avf_step(lambda state, control: -1e4 * state, z, None, 1.0)
    residual = (result - z + 1e4 * (result + z) / 2).abs()
    assert (residual <= 1e-8 + 1e-6 * torch.maximum(z.abs(), result.abs())).all()


def test_avf_checks_the_equation_at_the_returned_endpoint():
    # A slowly contractive linear step has a known implicit-midpoint solution.
    z = torch.ones(1, 2, dtype=torch.float64)
    options = {"rtol": 1e-10, "atol": 1e-12}
    result = avf_step(lambda state, control: -10 * state, z, None, 0.1, **options)
    residual = (result - z + (result + z) / 2).norm(dim=1)
    tolerance = options["atol"] + options["rtol"] * torch.maximum(z.norm(dim=1), result.norm(dim=1))
    assert (residual <= tolerance).all()
    torch.testing.assert_close(result, z / 3, atol=1e-10, rtol=1e-10)


def test_avf_convergence_does_not_depend_on_tensor_layout():
    # Reshaping a state must not turn the full-state stopping norm into per-line tests.
    field = torch.zeros(1, 2, 4, 4, 4, dtype=torch.float64)
    field[:, 0] = 1

    def rhs(state, control):
        value = state.reshape_as(field)
        result = torch.zeros_like(value)
        result[:, 1] = value[:, 0] - 4 * value[:, 1]
        return result.reshape_as(state)

    flat = avf_step(rhs, field.flatten(1), None, 0.1)
    spatial = avf_step(rhs, field, None, 0.1)
    torch.testing.assert_close(flat.reshape_as(spatial), spatial, atol=0, rtol=0)


def test_avf_equilibrium_has_the_correct_implicit_sensitivity():
    # State iterations stop immediately here; sensitivities still need their own solve.
    z = torch.zeros(1, 1, dtype=torch.float64)
    parameter = torch.zeros_like(z, requires_grad=True)

    def step(theta):
        return avf_step(lambda state, control: -10 * state + theta, z, None, 0.1,
                        max_iterations=64, rtol=1e-12, atol=1e-14)

    derivative = torch.autograd.grad(step(parameter).sum(), parameter)[0]
    torch.testing.assert_close(derivative, torch.full_like(z, 0.1 / 1.5))
    assert torch.autograd.gradcheck(step, (parameter,))


@pytest.mark.parametrize("loss_scale", [1.0, 1e-8, 1e-20])
def test_avf_sensitivity_is_independent_of_loss_scale(loss_scale):
    # Adjoint tolerances must not accept an inaccurate gradient just because it is small.
    theta = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    result = avf_step(lambda state, control: -10 * state + theta,
                      torch.zeros_like(theta), None, 0.1)
    derivative = torch.autograd.grad((loss_scale * result).sum(), theta)[0] / loss_scale
    torch.testing.assert_close(derivative, torch.full_like(theta, 0.1 / 1.5), atol=1e-7, rtol=1e-6)


def test_avf_output_keeps_local_energy_gradients_unchanged():
    # A local derivative dH/dy must not pass through the step's implicit adjoint.
    z = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    result = avf_step(lambda state, control: -10 * state, z, None, 0.1)
    identity = torch.autograd.grad(result.sum(), result, retain_graph=True)[0]
    torch.testing.assert_close(identity, torch.ones_like(result))
    effort = energy_gradient(lambda state: state.square().sum(-1) / 2,
                             result, create_graph=False)
    torch.testing.assert_close(effort, result)


def test_avf_nonlinear_gradients_include_state_energy_control_and_dt():
    # Verify the whole implicit dependency chain, including the energy Hessian.
    z = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    control = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    dt = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    def step(state, stiffness, forcing, interval):
        def rhs(value, supplied):
            energy = lambda point: stiffness * (point.square() / 2 + point.pow(4) / 4).sum(-1)
            return -energy_gradient(energy, value) + supplied
        return avf_step(rhs, state, forcing, interval, rtol=1e-12, atol=1e-14)

    assert torch.autograd.gradcheck(step, (z, weight, control, dt))


def test_avf_reports_nonconvergent_sensitivities_at_equilibrium():
    # A converged zero state must not hide a divergent adjoint fixed-point solve.
    forcing = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    result = avf_step(lambda state, control: -100 * state + control,
                      torch.zeros_like(forcing), forcing, 0.1, max_iterations=4)
    with pytest.raises(RuntimeError, match="AVF backward solve did not converge"):
        result.sum().backward()


@pytest.mark.parametrize("options", [{"rtol": float("inf")}, {"atol": float("nan")},
                                     {"max_iterations": 1.5}])
def test_avf_rejects_invalid_solver_settings(options):
    # Nonfinite tolerances would otherwise make a convergence assertion meaningless.
    with pytest.raises(ValueError):
        avf_step(lambda state, control: -state, torch.ones(1, 1), None, 0.1, **options)
