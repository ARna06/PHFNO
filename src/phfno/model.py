from dataclasses import dataclass
from numbers import Integral

import torch
from neuralop.models import FNO
from torch import Tensor, nn

from .fourier import RealFourierCoordinates
from .integrators import energy_gradient, euler_step


def make_fno(cutoff, in_channels, out_channels, hidden_channels, n_layers):
    return FNO(
        n_modes=tuple(2 * n + 1 for n in cutoff),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=None,
        factorization=None,
        fno_block_precision="full",
    )


@dataclass
class PHStructure:
    a: Tensor
    b: Tensor
    d: Tensor
    B: Tensor

    def apply_j(self, effort: Tensor) -> Tensor:
        be = (self.b * effort).sum(dim=-1, keepdim=True)
        ae = (self.a * effort).sum(dim=-1, keepdim=True)
        return 0.5 * (self.a * be - self.b * ae)

    def apply_r(self, effort: Tensor) -> Tensor:
        return self.d.square().unsqueeze(-1) * effort

    def apply_b(self, control: Tensor) -> Tensor:
        return torch.bmm(self.B, control.unsqueeze(-1)).squeeze(-1)

    def output(self, effort: Tensor) -> Tensor:
        return torch.bmm(self.B.transpose(1, 2), effort.unsqueeze(-1)).squeeze(-1)


def _scalar_mlp(input_dim: int, width: int, output_bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1, bias=output_bias),
    )


class PHFNO(nn.Module):
    def __init__(
        self,
        cutoff,
        state_channels: int,
        control_channels: int = 0,
        hidden_channels: int = 32,
        n_layers: int = 2,
        mlp_width: int = 64,
        parameter_grid=None,
    ):
        super().__init__()
        self.coordinates = RealFourierCoordinates(cutoff, state_channels)
        if not isinstance(control_channels, Integral) or control_channels < 0:
            raise ValueError("control_channels must be a nonnegative integer")
        self.state_channels = state_channels
        self.control_channels = int(control_channels)
        self.parameter_grid = self.coordinates._grid_shape(
            parameter_grid if parameter_grid is not None
            else tuple(2 * n + 1 for n in self.coordinates.cutoff)
        )
        self.factor_net = make_fno(
            self.coordinates.cutoff, state_channels,
            (2 + self.control_channels) * state_channels,
            hidden_channels, n_layers,
        )
        dim = self.coordinates.coordinate_dim
        self.energy_net = _scalar_mlp(dim, mlp_width, output_bias=False)
        self.damping_net = _scalar_mlp(dim, mlp_width)

    def energy(self, z: Tensor) -> Tensor:
        return self.energy_net(z).squeeze(-1)

    def effort(self, z: Tensor, create_graph: bool | None = None) -> Tensor:
        return energy_gradient(self.energy, z, create_graph=create_graph)

    def structure(self, z: Tensor) -> PHStructure:
        field = self.coordinates.decode(z, self.parameter_grid)
        raw = self.factor_net(field)
        groups = 2 + self.control_channels
        factors = self.coordinates.encode(
            raw.reshape(z.shape[0] * groups, self.state_channels, *self.parameter_grid)
        ).reshape(z.shape[0], groups, self.coordinates.coordinate_dim)
        return PHStructure(
            a=factors[:, 0], b=factors[:, 1],
            d=self.damping_net(z).squeeze(-1),
            B=factors[:, 2:].transpose(1, 2),
        )

    def _control(self, z: Tensor, control: Tensor | None) -> Tensor:
        if control is None:
            return z.new_zeros(z.shape[0], self.control_channels)
        if control.shape != (z.shape[0], self.control_channels):
            raise ValueError(f"control must have shape [batch, {self.control_channels}]")
        if control.dtype != z.dtype or control.device != z.device:
            raise ValueError("control must have the same dtype and device as the state")
        return control

    def rhs_coordinates(self, z: Tensor, control: Tensor | None = None) -> Tensor:
        control = self._control(z, control)
        e = self.effort(z)
        factors = self.structure(z)
        return factors.apply_j(e) - factors.apply_r(e) + factors.apply_b(control)

    def forward(self, field: Tensor, control: Tensor | None = None) -> Tensor:
        z = self.coordinates.encode(field)
        return self.coordinates.decode(self.rhs_coordinates(z, control), field.shape[2:])

    def step(self, field: Tensor, control: Tensor | None, dt, method="euler") -> Tensor:
        if method != "euler":
            raise ValueError("Only Euler time stepping is implemented")
        z = self.coordinates.encode(field)
        control = self._control(z, control)
        next_z = euler_step(self.rhs_coordinates, z, control, dt)
        return self.coordinates.decode(next_z, field.shape[2:])

    def rollout(self, initial: Tensor, controls: Tensor | None, times: Tensor,
                method="euler") -> Tensor:
        return _rollout(
            self, self.coordinates.project(initial), controls, times, method
        )


def _rollout(model, initial, controls, times, method):
    times = torch.as_tensor(times, dtype=initial.dtype, device=initial.device)
    if times.ndim != 1 or times.numel() < 1 or not torch.isfinite(times).all():
        raise ValueError("times must be a nonempty finite one-dimensional array")
    if not ((times[1:] - times[:-1]) > 0).all():
        raise ValueError("times must be strictly increasing")
    count = times.numel() - 1
    if controls is None:
        controls = initial.new_zeros(initial.shape[0], count, model.control_channels)
    if controls.shape != (initial.shape[0], count, model.control_channels):
        raise ValueError("controls must have shape [batch, len(times)-1, control_channels]")
    states = [initial]
    for index, dt in enumerate(times[1:] - times[:-1]):
        states.append(model.step(states[-1], controls[:, index], dt, method=method))
    return torch.stack(states, dim=1)
