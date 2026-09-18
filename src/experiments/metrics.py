import torch

from nsdata.solver import PeriodicNavierStokes


def _rms(field):
    return field.square().flatten(start_dim=2).mean(dim=-1).sqrt()


def _power(velocity, rate):
    return (velocity * rate).sum(dim=2).mean(dim=(-3, -2, -1))


@torch.no_grad()
def evaluate_model(model, data, indices, device="cuda", method=None, solver_options=None):
    indices = list(indices)
    if not indices:
        raise ValueError("indices must contain at least one trajectory")
    parameters = (p for p in model.parameters() if not p.is_complex())
    parameter = next(parameters, None)
    dtype = parameter.dtype if parameter is not None else data["clean"].dtype
    truth = data["clean"][indices].to(device=device, dtype=dtype)
    times = data["times"].to(device=device, dtype=dtype)
    controls = data["controls"][indices].to(device=device, dtype=dtype)
    forcing = data["forcing"][indices].to(device=device, dtype=dtype)
    if truth.ndim != 6 or truth.shape[2] != 3:
        raise ValueError("clean must have shape [batch, time, 3, x, y, z]")
    if times.numel() < 2:
        raise ValueError("evaluation requires at least two snapshots")
    solver = PeriodicNavierStokes(
        truth.shape[-1], data["metadata"]["viscosity"], device=device, dtype=dtype
    )
    was_training = model.training
    model.eval()
    try:
        # Experiment runners pass the training method/settings; None preserves generic adapters.
        options = {} if method is None else {"method": method}
        if solver_options is not None:
            options["solver_options"] = solver_options
        prediction = model.rollout(truth[:, 0], controls, times, **options)
        # Compare vector fields at clean reference states, independently of rollout drift.
        left = truth[:, :-1]
        flat_left = left.flatten(0, 1)
        flat_control = controls.flatten(0, 1)
        predicted_rhs = model(flat_left, flat_control).reshape_as(left)
        unforced_rhs = model(flat_left, torch.zeros_like(flat_control)).reshape_as(left)
    finally:
        model.train(was_training)
    if prediction.shape != truth.shape:
        raise ValueError("rollout must have the same shape as the reference trajectory")
    if not all(torch.isfinite(value).all() for value in (prediction, predicted_rhs, unforced_rhs)):
        raise FloatingPointError("Model evaluation produced nonfinite predictions")
    spectrum = torch.fft.fftn(flat_left, dim=(-3, -2, -1), norm="forward")
    forcing_spectrum = torch.fft.fftn(
        forcing.flatten(0, 1), dim=(-3, -2, -1), norm="forward"
    )
    true_rhs = torch.fft.ifftn(
        solver.rhs(spectrum, forcing_spectrum), dim=(-3, -2, -1), norm="forward"
    ).real.reshape_as(left)
    viscous_term = solver.viscous_term(flat_left).reshape_as(left)
    initial_rms = _rms(truth[:, :1]).squeeze(1).clamp_min(1e-12)
    rhs_rms = _rms(true_rhs).clamp_min(1e-12)
    dt = (times[1:] - times[:-1]).reshape(1, -1, 1, 1, 1, 1)
    input_power = _power(left, forcing)
    dissipation_rate = -_power(left, viscous_term)
    true_power = input_power - dissipation_rate
    predicted_power = _power(left, predicted_rhs)
    divergence = solver.divergence(prediction.flatten(0, 1)).reshape(
        *truth.shape[:2], -1
    )
    true_divergence = solver.divergence(truth.flatten(0, 1)).reshape(
        *truth.shape[:2], -1
    )
    force_rms = forcing.square().flatten(start_dim=1).mean(dim=-1).sqrt()
    result = {
        "prediction": prediction,
        "truth": truth,
        "times": times,
        "initial_rms": initial_rms,
        "nrmse_time": _rms(prediction - truth) / initial_rms[:, None],
        "relative_l2_time": _rms(prediction - truth) / _rms(truth).clamp_min(1e-12),
        "kinetic_energy": 0.5 * _power(prediction, prediction),
        "true_kinetic_energy": 0.5 * _power(truth, truth),
        "divergence_rms": divergence.square().mean(dim=-1).sqrt(),
        "true_divergence_rms": true_divergence.square().mean(dim=-1).sqrt(),
        "predicted_rhs": predicted_rhs,
        "true_rhs": true_rhs,
        "rhs_relative_error_time": _rms(predicted_rhs - true_rhs) / rhs_rms,
        "secant_rhs_relative_error_time": _rms(
            (truth[:, 1:] - left) / dt - true_rhs
        ) / rhs_rms,
        "ns_euler_nrmse_time": _rms(left + dt * true_rhs - truth[:, 1:]) / initial_rms[:, None],
        "persistence_nrmse_time": _rms(left - truth[:, 1:]) / initial_rms[:, None],
        "power_residual_time": predicted_power - true_power,
        "predicted_power": predicted_power,
        "true_power": true_power,
        "input_power": input_power,
        "dissipation_rate": dissipation_rate,
        "reference_power_scale": (input_power.abs() + dissipation_rate).mean(dim=1).clamp_min(1e-12),
        "force_nrmse_time": _rms(predicted_rhs - unforced_rhs - forcing) / force_rms[:, None].clamp_min(1e-12),
        "force_rms": force_rms,
    }
    return {key: value.detach().cpu() for key, value in result.items()}
