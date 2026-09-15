import torch
from torch import Tensor, nn

from .fourier import RealFourierCoordinates
from .model import _rollout, make_fno
from .integrators import euler_step


class FNOBaseline(nn.Module):

    def __init__(self, cutoff, state_channels: int, control_channels: int = 0,
                 hidden_channels: int = 32, n_layers: int = 2):
        super().__init__()
        self.coordinates = RealFourierCoordinates(cutoff, state_channels)
        if not isinstance(control_channels, int) or control_channels < 0:
            raise ValueError("control_channels must be a nonnegative integer")
        self.control_channels = control_channels
        self.operator = make_fno(
            self.coordinates.cutoff, state_channels + control_channels,
            state_channels, hidden_channels, n_layers,
        )

    def forward(self, field: Tensor, control: Tensor | None = None) -> Tensor:
        field = self.coordinates.project(field)
        if control is None:
            control = field.new_zeros(field.shape[0], self.control_channels)
        if control.shape != (field.shape[0], self.control_channels):
            raise ValueError(f"control must have shape [batch, {self.control_channels}]")
        if control.dtype != field.dtype or control.device != field.device:
            raise ValueError("control must have the same dtype and device as the state")
        broadcast = control.reshape(
            field.shape[0], self.control_channels, *((1,) * (field.ndim - 2))
        ).expand(-1, -1, *field.shape[2:])
        return self.coordinates.project(self.operator(torch.cat((field, broadcast), dim=1)))

    def step(self, field: Tensor, control: Tensor | None, dt, method="euler") -> Tensor:
        if method != "euler":
            raise ValueError("FNOBaseline supports only Euler time stepping")
        field = self.coordinates.project(field)
        return euler_step(self, field, control, dt)

    def rollout(self, initial: Tensor, controls: Tensor | None, times: Tensor,
                method="euler") -> Tensor:
        return _rollout(self, self.coordinates.project(initial), controls, times, method)
