import torch
from torch import nn

from .metrics import evaluate_model
from .vorticity_operators import PeriodicVorticity


def _rms(field):
    return field.square().flatten(start_dim=2).mean(dim=-1).sqrt()


class VelocityFromVorticity(nn.Module):
    def __init__(self, model, operators, forcing_basis):
        super().__init__()
        if forcing_basis.ndim != 5 or forcing_basis.shape[1] != 3:
            raise ValueError("forcing_basis must have shape [inputs, 3, x, y, z]")
        self.model = model
        self.operators = operators
        self.register_buffer("forcing_basis", forcing_basis)
        self.last_vorticity = None
        self.train(model.training)

    def _mean_force(self, control):
        return control @ self.forcing_basis.mean(dim=(-3, -2, -1))

    def forward(self, velocity, control=None):
        vorticity_rate = self.model(self.operators.curl(velocity), control)
        mean_rate = None if control is None else self._mean_force(control)
        return self.operators.velocity(vorticity_rate, mean=mean_rate)

    def reconstruct(self, vorticity, initial_velocity, controls, times):
        if vorticity.shape[:2] != (initial_velocity.shape[0], times.numel()):
            raise ValueError("vorticity must match the batch size and time grid")
        # Curl loses mean velocity; evolve that component from the mean force separately.
        mean = initial_velocity.mean(dim=(-3, -2, -1)).unsqueeze(1)
        if controls is None:
            mean = mean.expand(-1, times.numel(), -1)
        else:
            if controls.shape[:2] != (initial_velocity.shape[0], times.numel() - 1):
                raise ValueError("controls must have one input per time interval")
            increments = self._mean_force(controls) * times.diff()[None, :, None]
            offsets = torch.cat((torch.zeros_like(mean), increments.cumsum(dim=1)), dim=1)
            mean = mean + offsets
        return self.operators.velocity(vorticity, mean=mean)

    def rollout(self, initial, controls, times, method=None, solver_options=None):
        # Forward the selected integrator through the representation adapter unchanged.
        options = {} if method is None else {"method": method}
        if solver_options is not None:
            options["solver_options"] = solver_options
        vorticity = self.model.rollout(self.operators.curl(initial), controls, times, **options)
        self.last_vorticity = vorticity.detach()
        return self.reconstruct(vorticity, initial, controls, times)


@torch.no_grad()
def evaluate_vorticity_model(model, data, indices, device="cuda", method=None, solver_options=None):
    parameters = (p for p in model.parameters() if not p.is_complex())
    parameter = next(parameters, None)
    dtype = parameter.dtype if parameter is not None else data["clean"].dtype
    operators = PeriodicVorticity(data["clean"].shape[-1], device=device, dtype=dtype)
    forcing_basis = data["forcing_basis"].to(device=device, dtype=dtype)
    adapter = VelocityFromVorticity(model, operators, forcing_basis)
    indices = list(indices)
    # Velocity and raw-vorticity metrics must describe the same selected discrete rollout.
    result = evaluate_model(adapter, data, indices, device=device,
                            method=method, solver_options=solver_options)
    truth = result["truth"].to(device=device, dtype=dtype)
    prediction = result["prediction"].to(device=device, dtype=dtype)
    vorticity_truth = operators.curl(truth)
    vorticity_prediction = adapter.last_vorticity
    initial_rms = _rms(vorticity_truth[:, :1]).clamp_min(1e-12)
    controls = data["controls"][indices].to(device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        predicted_rhs = model(
            vorticity_truth[:, :-1].flatten(0, 1), controls.flatten(0, 1)
        ).reshape_as(vorticity_truth[:, :-1])
    finally:
        model.train(was_training)
    if not torch.isfinite(vorticity_prediction).all() or not torch.isfinite(predicted_rhs).all():
        raise FloatingPointError("Model evaluation produced nonfinite vorticity")
    true_rhs = operators.curl(result["true_rhs"].to(device=device, dtype=dtype))
    divergence = operators.divergence(vorticity_prediction)
    extra = {
        "vorticity_prediction": vorticity_prediction,
        "vorticity_truth": vorticity_truth,
        "vorticity_nrmse_time": _rms(vorticity_prediction - vorticity_truth) / initial_rms,
        "vorticity_divergence_rms": divergence.square().flatten(start_dim=2).mean(dim=-1).sqrt(),
        "vorticity_mean_norm": vorticity_prediction.mean(dim=(-3, -2, -1)).norm(dim=-1),
        "vorticity_consistency_nrmse_time": _rms(
            vorticity_prediction - operators.curl(prediction)
        ) / initial_rms,
        "vorticity_rhs_relative_error_time": _rms(predicted_rhs - true_rhs) / _rms(true_rhs).clamp_min(1e-12),
        "enstrophy": 0.5 * vorticity_prediction.square().sum(dim=2).mean(dim=(-3, -2, -1)),
        "true_enstrophy": 0.5 * vorticity_truth.square().sum(dim=2).mean(dim=(-3, -2, -1)),
    }
    result.update({key: value.detach().cpu() for key, value in extra.items()})
    return result
